"""Train DirectS2ST.

    torchrun --nproc_per_node=1 training/train.py --config training/configs/es.yaml \
        --campplus-model-root /path/to/seed-vc --campplus-checkpoint-path /path/to/campplus_cn_en_common.pt

The config has three sections:

  model     architecture and loss weights (the keys of a checkpoint's model_config.json)
  data      training/dev TSV manifests and column names (see training/data.py)
  training  optimisation and checkpointing (Hugging Face TrainingArguments)

The w2v-BERT 2.0 encoder, Mimi and CAMPPlus stay frozen. The text-to-text and
text-to-C0 objectives (``text_codec_t2t_loss_weight`` /
``text_codec_t2c_loss_weight``) run the temporal decoder over the gold
source-text -> target-text -> C0 stream without the speech memory; they add
no parameters and are used only in training.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from transformers import AutoFeatureExtractor, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "inference"))

from data import WaveformCodeStackCollator, load_manifest_dataset  # noqa: E402
from model_core import WhisperTemporalDepthTransformer  # noqa: E402

SOURCE_SAMPLE_RATE = 16000


def build_text_tokenizer(model_cfg: Dict[str, Any]) -> Tuple[Any, Dict[str, int]]:
    """Load the SentencePiece tokenizer and register the prefix control tokens.

    The ids of these tokens are baked into the text embedding table by
    position, so registration always happens in this order: BOS/SEP (only if
    their ids are not fixed by the config), then the content START/END tokens.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["text_tokenizer_name"], use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    def fixed(key: str) -> Optional[int]:
        value = model_cfg.get(key)
        return None if value is None or str(value).strip() == "" else int(value)

    bos_token = str(model_cfg.get("text_prefix_bos_token", "<|text_bos|>"))
    sep_token = str(model_cfg.get("text_prefix_sep_token", "<|text_sep|>"))
    start_token = str(model_cfg.get("uniss_start_content_token", "<|start_content|>"))
    end_token = str(model_cfg.get("uniss_end_content_token", "<|end_content|>"))
    use_content_controls = bool(model_cfg.get("enable_uniss_content_controls", False))
    added: List[str] = []
    if fixed("text_bos_token_id") is None:
        added.append(bos_token)
    if fixed("text_sep_token_id") is None:
        added.append(sep_token)
    if use_content_controls:
        added.extend([start_token, end_token])
    if added:
        existing = list(getattr(tokenizer, "additional_special_tokens", []) or [])
        tokenizer.add_special_tokens(
            {"additional_special_tokens": existing + [t for t in added if t not in existing]}
        )

    ids: Dict[str, int] = {
        "text_vocab_size": len(tokenizer),
        "text_pad_token_id": fixed("text_pad_token_id")
        if fixed("text_pad_token_id") is not None
        else int(tokenizer.pad_token_id),
        "text_bos_token_id": fixed("text_bos_token_id")
        if fixed("text_bos_token_id") is not None
        else int(tokenizer.convert_tokens_to_ids(bos_token)),
        "text_sep_token_id": fixed("text_sep_token_id")
        if fixed("text_sep_token_id") is not None
        else int(tokenizer.convert_tokens_to_ids(sep_token)),
    }
    if use_content_controls:
        ids["uniss_start_content_token_id"] = int(tokenizer.convert_tokens_to_ids(start_token))
        ids["uniss_end_content_token_id"] = int(tokenizer.convert_tokens_to_ids(end_token))
        if tokenizer.unk_token_id in (ids["uniss_start_content_token_id"], ids["uniss_end_content_token_id"]):
            raise ValueError("Failed to register the content START/END tokens.")
    return tokenizer, ids


def build_model(model_cfg: Dict[str, Any]) -> WhisperTemporalDepthTransformer:
    """Construct the model from ``model`` config keys, matched to the
    constructor's parameter names (``codebook_size`` -> ``codebook_size_``)."""
    params = set(inspect.signature(WhisperTemporalDepthTransformer.__init__).parameters)
    kwargs: Dict[str, Any] = {}
    for key, value in model_cfg.items():
        if f"{key}_" in params:
            kwargs[f"{key}_"] = value
        elif key in params:
            kwargs[key] = value
    kwargs.setdefault("vocab_size_", int(model_cfg["codebook_size"]) + 2)  # + BOS/EOS
    if "qwen_name" in params:
        kwargs.setdefault("qwen_name", "")
    return WhisperTemporalDepthTransformer(**kwargs)


def initialize_random_components(model: nn.Module, scheme: str) -> None:
    """Independently initialise the trainable DirectS2ST modules without
    touching the pretrained encoders.

    ``nn.TransformerDecoder`` deep-copies one prototype layer, so its default
    construction gives every layer identical initial tensors. The Xavier
    scheme redraws each parameter independently.
    """
    scheme = str(scheme or "constructor_default").strip().lower()
    if scheme == "constructor_default":
        return
    if scheme != "xavier_independent":
        raise ValueError(f"Unsupported random initialization scheme: {scheme}")
    component_names = (
        "encoder_to_temporal", "source_adapter", "source_adapter_norm", "source_unit_head",
        "source_unit_transformer_emb", "source_unit_transformer_pos_emb", "source_unit_transformer_decoder",
        "source_unit_transformer_norm", "source_unit_transformer_head", "temporal_codebook_emb",
        "temporal_pos_emb", "temporal_transformer", "temporal_norm", "text_token_emb",
        "text_lm_head", "text_codec_lm_head", "depth_decoder",
        "mimi_source_speaker_prompt_proj", "mimi_source_speaker_prompt_norm", "source_speaker_prompt_frame_proj",
        "source_speaker_prompt_input_norm", "source_speaker_prompt_encoder", "source_speaker_prompt_output_norm",
    )
    roots = [(name, getattr(model, name)) for name in component_names if isinstance(getattr(model, name, None), nn.Module)]
    initialized: Set[int] = set()
    with torch.no_grad():
        # Embeddings and norms keep their standard specialised initialisation.
        for _, root in roots:
            for module in root.modules():
                if isinstance(module, nn.Embedding):
                    if id(module.weight) not in initialized:
                        nn.init.normal_(module.weight, mean=0.0, std=0.02)
                        if module.padding_idx is not None:
                            module.weight[int(module.padding_idx)].zero_()
                        initialized.add(id(module.weight))
                elif isinstance(module, nn.LayerNorm):
                    if module.weight is not None and id(module.weight) not in initialized:
                        nn.init.ones_(module.weight)
                        initialized.add(id(module.weight))
                    if module.bias is not None and id(module.bias) not in initialized:
                        nn.init.zeros_(module.bias)
                        initialized.add(id(module.bias))
        # Every Transformer layer receives a fresh draw instead of a cloned one.
        for root_name, root in roots:
            for param_name, param in root.named_parameters(recurse=True):
                if id(param) in initialized:
                    continue
                full_name = f"{root_name}.{param_name}"
                if "position" in full_name or full_name.endswith("_sep"):
                    nn.init.normal_(param, mean=0.0, std=0.02)
                elif param.ndim >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.zeros_(param)
                initialized.add(id(param))
    print(f"[init] {scheme}: {len(initialized)} tensors in {[name for name, _ in roots]}")


def load_init_weights(model: nn.Module, checkpoint_dir: str, require_exact: bool) -> None:
    """Load model weights (no optimizer state) from an existing checkpoint."""
    for name in ("pytorch_model.bin", "pytorch_model_fsdp.bin"):
        path = os.path.join(checkpoint_dir, name)
        if os.path.isfile(path):
            break
    else:
        raise FileNotFoundError(f"No pytorch_model.bin in {checkpoint_dir}")
    state_dict = torch.load(path, map_location="cpu")
    if isinstance(state_dict, dict) and isinstance(state_dict.get("model"), dict):
        state_dict = state_dict["model"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[init] loaded {path}: missing={len(missing)} unexpected={len(unexpected)}")
    if require_exact and (missing or unexpected):
        raise RuntimeError(f"Inexact initialisation: missing={missing[:20]} unexpected={unexpected[:20]}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DirectS2STTrainer(Trainer):
    """Logs each loss component the model returns next to the total loss."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._component_sums: Dict[str, float] = {}
        self._component_count = 0

    def _set_global_step_on_model(self, model: nn.Module) -> None:
        # Step-scheduled options (scheduled sampling, auxiliary RNG seeding)
        # read the step from the unwrapped model.
        step = int(getattr(self.state, "global_step", 0) or 0)
        seen: List[Any] = []
        queue: List[Any] = [model, self.accelerator.unwrap_model(model)]
        while queue:
            obj = queue.pop()
            if obj is None or any(obj is s for s in seen):
                continue
            seen.append(obj)
            if hasattr(obj, "_current_global_step"):
                obj._current_global_step = step
            queue.extend(getattr(obj, name, None) for name in ("module", "_fsdp_wrapped_module"))

    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch=None):
        self._set_global_step_on_model(model)
        outputs = model(**inputs)
        loss = outputs["loss"]
        if model.training:
            for key, value in outputs.items():
                if key != "loss" and key.endswith("loss") and torch.is_tensor(value) and value.numel() == 1:
                    self._component_sums[key] = self._component_sums.get(key, 0.0) + float(value.detach().float())
            self._component_count += 1
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        if self._component_count and "loss" in logs:
            for key, total in self._component_sums.items():
                logs[f"train/{key}"] = total / self._component_count
            self._component_sums, self._component_count = {}, 0
        super().log(logs, *args, **kwargs)


class SaveModelConfigCallback(TrainerCallback):
    """Write model_config.json next to every checkpoint's weights, so
    scripts/clean_checkpoint.py and inference/infer.py can load it."""

    def __init__(self, model_cfg: Dict[str, Any]):
        self.payload = json.dumps(
            {k: v for k, v in model_cfg.items() if k not in ("campplus_model_root", "campplus_checkpoint_path")},
            indent=2, sort_keys=True,
        ) + "\n"

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if ckpt.is_dir():
                (ckpt / "model_config.json").write_text(self.payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--campplus-model-root", required=True,
                        help="Local clone of https://github.com/Plachtaa/seed-vc (see docs/model_dependencies.md).")
    parser.add_argument("--campplus-checkpoint-path", required=True,
                        help="campplus_cn_en_common.pt from iic/speech_campplus_sv_zh_en_16k-common_advanced.")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    model_cfg: Dict[str, Any] = dict(cfg["model"])
    data_cfg: Dict[str, Any] = dict(cfg["data"])
    train_cfg: Dict[str, Any] = dict(cfg["training"])
    model_cfg["campplus_model_root"] = args.campplus_model_root
    model_cfg["campplus_checkpoint_path"] = args.campplus_checkpoint_path
    config_dir = Path(args.config).resolve().parent
    tok_path = Path(model_cfg["text_tokenizer_name"])
    if not tok_path.is_absolute() and (config_dir / tok_path).exists():
        model_cfg["text_tokenizer_name"] = str((config_dir / tok_path).resolve())

    seed = int(train_cfg.get("seed", 42))
    model_init_seed = int(train_cfg.get("model_init_seed", seed))

    tokenizer, text_ids = build_text_tokenizer(model_cfg)
    model_cfg.update(text_ids)

    seed_everything(model_init_seed)
    model = build_model(model_cfg)
    init_from = train_cfg.get("init_from_checkpoint")
    if init_from:
        load_init_weights(model, init_from, require_exact=bool(model_cfg.get("train_only_depth_decoder", False)))
    else:
        # Decouple the Xavier draw from RNG consumed while building the
        # pretrained encoders above.
        seed_everything(model_init_seed)
        initialize_random_components(model, train_cfg.get("random_init_scheme", "xavier_independent"))
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {total / 1e9:.3f}B parameters, {trainable / 1e6:.1f}M trainable")

    num_quantizers = int(data_cfg.get("num_quantizers", model_cfg["input_num_quantizers"]))
    passthrough = [data_cfg["target_text_column"], data_cfg["source_text_column"]] + [
        f"{prefix}_{suffix}"
        for prefix in ("mimi_source_acoustic_embedding", "campplus_source_embedding", "campplus_target_embedding")
        for suffix in ("path", "index", "dim")
    ]
    load_kwargs = dict(
        audio_column="src",
        codes_column="tgt",
        num_quantizers=num_quantizers,
        num_modeled_levels=int(model_cfg["num_codebook_levels"]),
        codebook_model_order=model_cfg.get("codebook_model_order"),
        passthrough_columns=passthrough,
        num_proc=int(data_cfg.get("num_proc", 1)),
    )
    train_dataset = load_manifest_dataset(data_cfg["train_tsv"], **load_kwargs)
    eval_dataset = load_manifest_dataset(data_cfg["dev_tsv"], **load_kwargs)
    eval_max_samples = int(data_cfg.get("eval_max_samples", 0))
    if 0 < eval_max_samples < len(eval_dataset):
        eval_dataset = eval_dataset.shuffle(seed=int(data_cfg.get("eval_subset_seed", 42))).select(range(eval_max_samples))

    codebook_size = int(model_cfg["codebook_size"])
    collator = WaveformCodeStackCollator(
        feature_extractor=AutoFeatureExtractor.from_pretrained(model_cfg["speech_encoder_name"]),
        audio_key="src",
        codes_key="tgt",
        sample_rate=SOURCE_SAMPLE_RATE,
        num_q=num_quantizers,
        bos_id=codebook_size,
        eos_id=codebook_size + 1,
        speech_max_audio_samples=int(SOURCE_SAMPLE_RATE * float(data_cfg.get("speech_max_audio_seconds", 30.0))),
        text_tokenizer=tokenizer,
        text_key=data_cfg["target_text_column"],
        text_max_tokens=int(data_cfg.get("text_max_tokens", 128)),
        text_bos_token_id=text_ids["text_bos_token_id"],
        text_eos_token_id=text_ids["text_sep_token_id"],
        text_pad_token_id=text_ids["text_pad_token_id"],
        text_extra_token_ids_after_bos=model_cfg.get("text_prefix_extra_token_ids_after_bos") or [],
        source_text_key=data_cfg["source_text_column"],
        source_text_max_tokens=int(data_cfg.get("source_text_max_tokens", 128)),
        source_text_extra_token_ids_after_bos=model_cfg.get("quality_cot_source_text_prefix_extra_token_ids_after_bos") or [],
        uniss_start_content_token_id=text_ids.get("uniss_start_content_token_id"),
        uniss_end_content_token_id=text_ids.get("uniss_end_content_token_id"),
        return_source_acoustic_wav=True,
        use_precomputed_campplus_embeddings=True,
        use_precomputed_campplus_target_embeddings=float(model_cfg.get("campplus_depth_identity_loss_weight", 0.0)) > 0.0,
        allow_online_campplus_fallback=bool(data_cfg.get("allow_online_campplus_fallback", False)),
    )

    arg_names = set(inspect.signature(TrainingArguments.__init__).parameters)
    training_kwargs = {k: v for k, v in train_cfg.items() if k in arg_names}
    training_kwargs.setdefault("remove_unused_columns", False)  # the collator needs every column
    training_kwargs.setdefault("prediction_loss_only", True)  # eval logits are [B, T, 16, 2050]
    training_kwargs.setdefault("save_safetensors", False)  # text head is tied to the text embedding
    training_kwargs.setdefault("seed", seed)
    training_kwargs.setdefault("data_seed", seed)
    trainer = DirectS2STTrainer(
        model=model,
        args=TrainingArguments(**training_kwargs),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[SaveModelConfigCallback(model_cfg)],
    )
    trainer.train(resume_from_checkpoint=train_cfg.get("resume_from_checkpoint") or None)
    trainer.save_model()
    if trainer.is_world_process_zero():
        (Path(trainer.args.output_dir) / "model_config.json").write_text(
            SaveModelConfigCallback(model_cfg).payload
        )


if __name__ == "__main__":
    main()
