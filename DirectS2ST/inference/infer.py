"""DirectS2ST inference.

    python inference/infer.py --checkpoint-dir checkpoints/en-es \
        --campplus-model-root /path/to/seed-vc --campplus-checkpoint-path /path/to/campplus_cn_en_common.pt \
        --source-wav input_en.wav --output-wav output_es.wav

Pipeline: w2v-BERT 2.0 source encoding; Mimi source prompt (pooled features for
the <SEP> embedding, source codes for the depth-decoder prompt) and CAMPPlus
speaker embedding; beam search over source text, target text and C0 with the
shared text+codec head; one depth-decoder pass for C1-C15; Mimi decoding.
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from transformers import AutoFeatureExtractor, MimiModel

from model_core import WhisperTemporalDepthTransformer


# Decoding settings of the released checkpoints.
FREE_RUNNING_BEAM_SIZE = 5
FREE_RUNNING_BEAM_TOPK = 6
FREE_RUNNING_MIN_EOS_TARGET_RATIO = 0.8
FREE_RUNNING_EOS_BONUS = 0.2
FREE_RUNNING_MAX_TARGET_RATIO = 4.0
FREE_RUNNING_MAX_ROWS = 0  # 0: only the ratio limit applies
FREE_RUNNING_LENGTH_PENALTY = 1.0
FREE_RUNNING_CAP_PENALTY = 5.0
FREE_RUNNING_REPEAT_PENALTY = 0.0
FREE_RUNNING_REPEAT_WINDOW = 0
FREE_RUNNING_NO_REPEAT_NGRAM_SIZE = 2
FREE_RUNNING_STOP_ON_LEVEL0_EOS = True
FREE_RUNNING_LEVEL0_EOS_EXTRA_ROWS = 15
EVAL_AUDIO_TARGET_ROWS_PER_SECOND = 16.0

SOURCE_SAMPLE_RATE = 16000


def _invert_codebook_model_order(
    model_to_mimi_order: Optional[List[int]], num_levels: int
) -> Optional[List[int]]:
    if model_to_mimi_order is None:
        return None
    mimi_to_model = [0] * num_levels
    for model_idx, mimi_idx in enumerate(model_to_mimi_order):
        mimi_to_model[int(mimi_idx)] = int(model_idx)
    return mimi_to_model


def _model_order_codes_to_mimi_order(
    codes_q_f: torch.Tensor, codebook_mimi_to_model_order: Optional[List[int]]
) -> torch.Tensor:
    if codebook_mimi_to_model_order is None:
        return codes_q_f
    index = torch.as_tensor(
        codebook_mimi_to_model_order, dtype=torch.long, device=codes_q_f.device
    )
    return codes_q_f.index_select(0, index)


def _delayed_codes_to_mimi_codes(
    delayed_t_q: torch.Tensor,
    codebook_size: int,
    codebook_mimi_to_model_order: Optional[List[int]],
) -> torch.Tensor:
    """Undo the RVQ delay pattern and strip BOS/EOS. Input [T, Q] -> output [Q, F]."""
    target_len, num_q = delayed_t_q.shape
    full_frames = target_len - num_q + 1
    if full_frames < 3:
        raise ValueError(
            f"Delayed sequence too short to recover BOS/code/EOS rows: {tuple(delayed_t_q.shape)}"
        )
    codes_f_q = torch.empty((full_frames, num_q), dtype=torch.long, device=delayed_t_q.device)
    for q in range(num_q):
        codes_f_q[:, q] = delayed_t_q[q : q + full_frames, q]
    acoustic_codes = codes_f_q[1:-1, :].transpose(0, 1).contiguous()
    acoustic_codes = acoustic_codes.clamp(min=0, max=codebook_size - 1)
    return _model_order_codes_to_mimi_order(acoustic_codes, codebook_mimi_to_model_order)


def resolve_text_tokenizer_ids(model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Register the text-prefix control tokens and return their ids.

    Token ids index the checkpoint's text embedding table, so the tokens are
    registered in the same order as in training: BOS/SEP (only when their ids
    are not fixed by the config), then the content START/END tokens.
    """
    if not model_config.get("enable_text_prefix_ar", False):
        return {}
    from transformers import AutoTokenizer

    text_tokenizer_name = model_config.get(
        "text_tokenizer_name", model_config.get("tokenizer_name")
    )
    tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    text_bos_token_id_config = model_config.get("text_bos_token_id")
    text_sep_token_id_config = model_config.get("text_sep_token_id")
    text_prefix_bos_token = str(model_config.get("text_prefix_bos_token", "<|text_bos|>"))
    text_prefix_sep_token = str(model_config.get("text_prefix_sep_token", "<|text_sep|>"))
    enable_uniss_content_controls = bool(model_config.get("enable_uniss_content_controls", False))
    uniss_start_content_token = str(model_config.get("uniss_start_content_token", "<|start_content|>"))
    uniss_end_content_token = str(model_config.get("uniss_end_content_token", "<|end_content|>"))

    added_specials: List[str] = []
    if text_bos_token_id_config is None or str(text_bos_token_id_config).strip() == "":
        added_specials.append(text_prefix_bos_token)
    if text_sep_token_id_config is None or str(text_sep_token_id_config).strip() == "":
        added_specials.append(text_prefix_sep_token)
    if enable_uniss_content_controls:
        added_specials.extend([uniss_start_content_token, uniss_end_content_token])
    if added_specials:
        existing = list(getattr(tokenizer, "additional_special_tokens", []) or [])
        merged = existing + [tok for tok in added_specials if tok not in existing]
        tokenizer.add_special_tokens({"additional_special_tokens": merged})

    resolved: Dict[str, Any] = {"text_vocab_size": len(tokenizer)}
    resolved["text_pad_token_id"] = (
        int(text_pad_token_id)
        if (text_pad_token_id := model_config.get("text_pad_token_id")) is not None
        and str(text_pad_token_id).strip() != ""
        else int(tokenizer.pad_token_id)
    )
    resolved["text_bos_token_id"] = (
        int(text_bos_token_id_config)
        if text_bos_token_id_config is not None and str(text_bos_token_id_config).strip() != ""
        else int(tokenizer.convert_tokens_to_ids(text_prefix_bos_token))
    )
    resolved["text_sep_token_id"] = (
        int(text_sep_token_id_config)
        if text_sep_token_id_config is not None and str(text_sep_token_id_config).strip() != ""
        else int(tokenizer.convert_tokens_to_ids(text_prefix_sep_token))
    )
    if enable_uniss_content_controls:
        start_id = int(tokenizer.convert_tokens_to_ids(uniss_start_content_token))
        end_id = int(tokenizer.convert_tokens_to_ids(uniss_end_content_token))
        if start_id == tokenizer.unk_token_id or end_id == tokenizer.unk_token_id:
            raise ValueError("Failed to register UniSS START/END content tokens.")
        resolved["uniss_start_content_token_id"] = start_id
        resolved["uniss_end_content_token_id"] = end_id
    return resolved


def matched_ctor_config_keys(model_config: Dict[str, Any]) -> Dict[str, str]:
    """Map config keys to constructor parameter names (``codebook_size`` -> ``codebook_size_``)."""
    ctor_params = set(inspect.signature(WhisperTemporalDepthTransformer.__init__).parameters)
    matched: Dict[str, str] = {}
    for key in model_config:
        if f"{key}_" in ctor_params:
            matched[key] = f"{key}_"
        elif key in ctor_params:
            matched[key] = key
    return matched


def build_model_from_config(
    model_config: Dict[str, Any], state_dict: Dict[str, torch.Tensor]
) -> WhisperTemporalDepthTransformer:
    """Construct the model from a released ``model_config.json``."""
    ctor_params = set(inspect.signature(WhisperTemporalDepthTransformer.__init__).parameters)
    kwargs: Dict[str, Any] = {
        param_name: model_config[key]
        for key, param_name in matched_ctor_config_keys(model_config).items()
    }
    # Required by the constructor; unused when temporal_hidden_size is set.
    if "qwen_name" in ctor_params and "qwen_name" not in kwargs:
        kwargs["qwen_name"] = model_config.get("qwen_model_name", "")
    model = WhisperTemporalDepthTransformer(**kwargs)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[infer] load_state_dict: {len(missing)} missing keys (showing up to 5): {missing[:5]}")
    if unexpected:
        print(f"[infer] load_state_dict: {len(unexpected)} unexpected keys (showing up to 5): {unexpected[:5]}")
    return model


def load_merged_model_config(checkpoint_dir: Path) -> Dict[str, Any]:
    """Load model_config.json (and, for raw training checkpoints, experiment_config.yaml on top)."""
    model_config: Dict[str, Any] = {}
    json_path = checkpoint_dir / "model_config.json"
    if json_path.exists():
        model_config.update(json.loads(json_path.read_text()))
    yaml_path = checkpoint_dir / "experiment_config.yaml"
    if yaml_path.exists():
        import yaml

        model_config.update(yaml.safe_load(yaml_path.read_text()) or {})
    return model_config


class DirectS2STInference:
    def __init__(
        self,
        checkpoint_dir: str,
        device: str = "cuda",
        campplus_model_root: Optional[str] = None,
        campplus_checkpoint_path: Optional[str] = None,
        codec_decode_device: str = "cpu",
        buffer_dtype: Optional[torch.dtype] = torch.bfloat16,
    ):
        checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device(device)

        model_config = load_merged_model_config(checkpoint_dir)

        # CAMPPlus is loaded from local paths (see docs/model_dependencies.md).
        if campplus_model_root:
            model_config["campplus_model_root"] = campplus_model_root
        if campplus_checkpoint_path:
            model_config["campplus_checkpoint_path"] = campplus_checkpoint_path
        if model_config.get("enable_campplus_depth_conditioning", False):
            if not model_config.get("campplus_model_root"):
                raise ValueError(
                    "This checkpoint requires a CAMPPlus model definition. Pass "
                    "--campplus-model-root pointing at a local clone of "
                    "https://github.com/Plachtaa/seed-vc (see docs/model_dependencies.md)."
                )
            if not model_config.get("campplus_checkpoint_path"):
                raise ValueError(
                    "This checkpoint requires CAMPPlus weights. Pass "
                    "--campplus-checkpoint-path pointing at a local copy of "
                    "iic/speech_campplus_sv_zh_en_16k-common_advanced "
                    "(see docs/model_dependencies.md)."
                )

        # The tokenizer path in a released config is relative to the checkpoint.
        text_tokenizer_name = model_config.get("text_tokenizer_name")
        if text_tokenizer_name and not os.path.isabs(text_tokenizer_name):
            model_config["text_tokenizer_name"] = str(
                (checkpoint_dir / text_tokenizer_name).resolve()
            )

        model_config.update(resolve_text_tokenizer_ids(model_config))
        state_dict = torch.load(checkpoint_dir / "pytorch_model.bin", map_location="cpu")
        self.model = build_model_from_config(model_config, state_dict)
        self.model.eval().to(self.device)
        # Training ran with bf16 buffers (FSDP mixed precision) and fp32
        # parameters; the only floating-point buffers are in the source Mimi.
        if buffer_dtype is not None:
            for buffer in self.model.buffers():
                if torch.is_floating_point(buffer):
                    buffer.data = buffer.data.to(buffer_dtype)

        self.codebook_size = int(model_config["codebook_size"])
        self.num_codebook_levels = int(model_config["num_codebook_levels"])
        self.codebook_mimi_to_model_order = _invert_codebook_model_order(
            model_config.get("codebook_model_order"), self.num_codebook_levels
        )

        self.speech_feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_config["speech_encoder_name"]
        )

        # CPU decoding by default; GPU decoding is faster but not bit-identical.
        mimi_name = model_config.get("mimi_source_acoustic_model_name", "kyutai/mimi")
        self.codec_decode_device = torch.device(codec_decode_device)
        self.mimi_decode_model = MimiModel.from_pretrained(mimi_name)
        self.mimi_decode_model.eval().to(self.codec_decode_device)
        self.mimi_output_sample_rate = int(self.mimi_decode_model.config.sampling_rate)
        self.max_source_samples = int(
            SOURCE_SAMPLE_RATE * float(model_config.get("speech_max_audio_seconds", 30.0))
        )
        self._resamplers: Dict[int, torchaudio.transforms.Resample] = {}
        self.last_batch_debug: List[Dict[str, Any]] = []
        self._beam_result: Dict[str, Any] = {}

    def _load_source_wav(self, path: str) -> torch.Tensor:
        """Mono, 16 kHz, cropped to the encoder's maximum input length."""
        wav, sr = torchaudio.load(path)
        wav = wav.mean(dim=0) if wav.ndim == 2 else wav.reshape(-1)
        if sr != SOURCE_SAMPLE_RATE:
            if sr not in self._resamplers:
                self._resamplers[sr] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SOURCE_SAMPLE_RATE)
            wav = self._resamplers[sr](wav)
        return wav[: self.max_source_samples]

    @torch.no_grad()
    def _encode_batch_conditions(self, wavs: List[torch.Tensor]) -> Dict[str, Any]:
        """Source conditions for a batch.

        The Mimi prompt and CAMPPlus are computed on the zero-padded batch, so
        they depend on the batch's padded length.
        """
        model = self.model
        device = self.device

        features = self.speech_feature_extractor(
            [w.cpu().numpy() for w in wavs],
            sampling_rate=SOURCE_SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
            padding=True,
            truncation=False,
        )
        input_features = features["input_features"].to(device=device, dtype=torch.float32)
        audio_attention_mask = features.get("attention_mask")
        if audio_attention_mask is not None:
            audio_attention_mask = audio_attention_mask.to(device=device)

        encoder_hidden, encoder_mask, unity_spk_embed = model._encode_source_inputs(
            input_features=input_features,
            audio_attention_mask=audio_attention_mask,
            return_spk_embed=True,
        )

        max_len = max(int(w.numel()) for w in wavs)
        source_wavs = torch.zeros((len(wavs), max_len), dtype=torch.float32)
        source_wav_lens = torch.zeros((len(wavs),), dtype=torch.long)
        for i, w in enumerate(wavs):
            w = w.detach().cpu().to(torch.float32).flatten()
            source_wavs[i, : w.numel()] = w
            source_wav_lens[i] = int(w.numel())
        source_wavs = source_wavs.to(device)
        source_wav_lens = source_wav_lens.to(device)

        # Also sets the depth decoder's source codec prompt.
        source_acoustic_embedding = model._encode_mimi_source_acoustic_embedding(
            source_wavs, source_wav_lens
        )
        mimi_spk_embed = model._build_mimi_source_speaker_prompt_embedding(
            source_acoustic_embedding, device=encoder_hidden.device, dtype=encoder_hidden.dtype
        )
        if mimi_spk_embed is not None:
            unity_spk_embed = mimi_spk_embed
        encoder_hidden = model._apply_mimi_source_acoustic_temporal_conditioning(
            encoder_hidden, source_acoustic_embedding
        )
        campplus = model._encode_campplus_source_embedding(source_wavs, source_wav_lens)
        model.depth_decoder.set_runtime_campplus_source_embedding(campplus)

        return {
            "encoder_hidden": encoder_hidden,
            "encoder_mask": encoder_mask,
            "unity_spk_embed": unity_spk_embed,
            "source_acoustic_embedding": source_acoustic_embedding,
            "campplus": campplus,
            "codec_tokens": getattr(model.depth_decoder, "_runtime_source_codec_tokens", None),
            "codec_mask": getattr(model.depth_decoder, "_runtime_source_codec_mask", None),
            "target_rows": [
                max(1, int(float(w.numel()) / SOURCE_SAMPLE_RATE * EVAL_AUDIO_TARGET_ROWS_PER_SECOND + 0.999))
                for w in wavs
            ],
        }

    # C0 next-token logits from the shared text+codec head.
    def _decode_next_level0_logits(
        self,
        seq: torch.Tensor,
        encoder_hidden: torch.Tensor,
        encoder_mask: Optional[torch.Tensor],
        text_prefix_ids: torch.Tensor,
        unity_spk_embed: Optional[torch.Tensor],
    ) -> torch.Tensor:
        model = self.model
        input_labels = seq[:, :, : self.num_codebook_levels]
        target_len = input_labels.size(1)
        temporal_inputs, temporal_mask, code_start_positions = model._build_textprefix_generation_batch(
            text_prefix_ids=text_prefix_ids.to(device=input_labels.device),
            code_prefix_labels=input_labels,
            sep_prompt_embedding=unity_spk_embed,
        )
        full_target_len = temporal_inputs.size(1)
        temporal_hidden = model.temporal_transformer(
            tgt=temporal_inputs,
            memory=encoder_hidden,
            tgt_mask=model._causal_mask(full_target_len, temporal_inputs.device),
            tgt_key_padding_mask=(temporal_mask == 0),
            memory_key_padding_mask=(encoder_mask == 0) if encoder_mask is not None else None,
        )
        temporal_hidden = model.temporal_norm(temporal_hidden)
        offset = int(model.text_vocab_size)
        unified_code_logits = model.text_codec_lm_head(temporal_hidden)[:, :, offset : offset + model.vocab_size]
        out = []
        for i, code_start in enumerate(code_start_positions):
            out.append(unified_code_logits[i : i + 1, code_start + target_len - 1, :])
        return torch.cat(out, dim=0)

    def _make_two_stage_next_row(
        self, batch_size: int, forced_level0: torch.Tensor, row_idx: int, bos_id: int, ignore_index: int
    ) -> torch.Tensor:
        next_row = torch.full(
            (batch_size, self.num_codebook_levels),
            fill_value=ignore_index,
            dtype=torch.long,
            device=forced_level0.device,
        )
        for q in range(self.num_codebook_levels):
            if row_idx == q:
                next_row[:, q] = bos_id
        if row_idx >= 1:
            next_row[:, 0] = forced_level0
        return next_row

    def _level0_banned_ngram_tokens(self, seq: torch.Tensor) -> List[int]:
        ngram_size = FREE_RUNNING_NO_REPEAT_NGRAM_SIZE
        if ngram_size <= 0:
            return []
        level0 = seq[:, 0]
        valid = (level0 >= 0) & (level0 <= self.codebook_size + 1)
        tokens = level0[valid].detach().cpu().long().tolist()
        if len(tokens) < ngram_size - 1:
            return []
        prefix = tokens[-(ngram_size - 1) :] if ngram_size > 1 else []
        banned = set()
        if ngram_size == 1:
            banned.update(tokens)
        else:
            for i in range(0, len(tokens) - ngram_size + 1):
                if tokens[i : i + ngram_size - 1] == prefix:
                    banned.add(tokens[i + ngram_size - 1])
        return [tok for tok in banned if 0 <= tok <= self.codebook_size + 1]

    def _level0_repeat_penalty(self, seq: torch.Tensor, token_id: int) -> float:
        if FREE_RUNNING_REPEAT_PENALTY <= 0.0 or FREE_RUNNING_REPEAT_WINDOW <= 0:
            return 0.0
        level0 = seq[:, 0]
        valid = (level0 >= 0) & (level0 < self.codebook_size)
        recent = level0[valid][-FREE_RUNNING_REPEAT_WINDOW:]
        if recent.numel() == 0:
            return 0.0
        matches = (recent == int(token_id)).sum().item()
        return float(matches) * FREE_RUNNING_REPEAT_PENALTY

    def _apply_level0_no_repeat_ngram_mask(self, scores: torch.Tensor, seqs: torch.Tensor) -> torch.Tensor:
        if FREE_RUNNING_NO_REPEAT_NGRAM_SIZE <= 0:
            return scores
        adjusted = scores.clone()
        for row_idx in range(seqs.size(0)):
            banned = self._level0_banned_ngram_tokens(seqs[row_idx])
            if banned:
                adjusted[row_idx, torch.tensor(banned, device=adjusted.device, dtype=torch.long)] = -torch.inf
        return adjusted

    def _beam_rank_score(self, score: float, used_rows: int, status: str) -> float:
        denom = float(max(1, used_rows))
        if FREE_RUNNING_LENGTH_PENALTY > 0.0:
            score = score / (denom ** FREE_RUNNING_LENGTH_PENALTY)
        if status == "duration_capped":
            score -= FREE_RUNNING_CAP_PENALTY
        return score

    def _free_running_max_rows_for_sample(self, target_rows: int) -> int:
        ratio_limit = int(max(3, round(target_rows * FREE_RUNNING_MAX_TARGET_RATIO)))
        if FREE_RUNNING_MAX_ROWS > 0:
            return min(FREE_RUNNING_MAX_ROWS, ratio_limit)
        return ratio_limit

    # C1..C15 in one depth-decoder pass once C0 is final.
    def _fill_depth_levels(
        self,
        seq: torch.Tensor,
        encoder_hidden: torch.Tensor,
        encoder_mask: Optional[torch.Tensor],
        text_prefix_ids: torch.Tensor,
        unity_spk_embed: Optional[torch.Tensor],
        source_acoustic_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        model = self.model
        seq_device = seq.to(device=self.device, dtype=torch.long).unsqueeze(0)
        input_labels = seq_device[:, :, : self.num_codebook_levels]
        target_len = input_labels.size(1)

        temporal_inputs, temporal_mask, code_start_positions = model._build_textprefix_generation_batch(
            text_prefix_ids=text_prefix_ids.to(device=self.device),
            code_prefix_labels=input_labels,
            sep_prompt_embedding=unity_spk_embed,
        )
        full_target_len = temporal_inputs.size(1)
        temporal_hidden = model.temporal_transformer(
            tgt=temporal_inputs,
            memory=encoder_hidden,
            tgt_mask=model._causal_mask(full_target_len, temporal_inputs.device),
            tgt_key_padding_mask=(temporal_mask == 0),
            memory_key_padding_mask=(encoder_mask == 0) if encoder_mask is not None else None,
        )
        temporal_hidden = model.temporal_norm(temporal_hidden)

        codec_temporal_hidden = temporal_hidden.new_zeros(
            (input_labels.size(0), target_len, temporal_hidden.size(-1))
        )
        for i, code_start in enumerate(code_start_positions):
            codec_temporal_hidden[i, :target_len] = temporal_hidden[i, code_start : code_start + target_len]

        logits_per_level, _, _ = model.depth_decoder(
            temporal_hidden=codec_temporal_hidden,
            source_memory=None,
            source_memory_mask=None,
            source_acoustic_embedding=source_acoustic_embedding,
            labels=None,
            teacher_force=False,
            ignore_index=model.ignore_index,
            forced_first_ids=input_labels[:, :, 0],
        )
        stacked = torch.stack(logits_per_level, dim=2)[0]  # [T, Q, vocab]
        pred_ids = stacked.argmax(dim=-1)

        filled = seq_device[0].clone()
        bos_id = self.codebook_size
        for row_idx in range(filled.size(0)):
            for q in range(1, self.num_codebook_levels):
                if row_idx < q:
                    filled[row_idx, q] = model.ignore_index
                elif row_idx == q:
                    filled[row_idx, q] = bos_id
                else:
                    filled[row_idx, q] = pred_ids[row_idx, q]
        return filled.detach().cpu()

    # Beam search over C0 rows.
    @torch.no_grad()
    def _generate_beam(
        self,
        encoder_hidden: torch.Tensor,
        encoder_mask: Optional[torch.Tensor],
        text_prefix_ids: torch.Tensor,
        unity_spk_embed: Optional[torch.Tensor],
        target_rows: int,
        source_acoustic_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        model = self.model
        device = self.device
        max_rows = self._free_running_max_rows_for_sample(target_rows)
        min_eos_row = max(1, int(round(float(target_rows) * FREE_RUNNING_MIN_EOS_TARGET_RATIO)))
        eos_id = self.codebook_size + 1
        bos_id = self.codebook_size

        initial_seq = torch.full(
            (1, self.num_codebook_levels), fill_value=model.ignore_index, dtype=torch.long, device=device
        )
        initial_seq[0, 0] = bos_id
        beams: List[Dict[str, Any]] = [
            {
                "seq": initial_seq,
                "score": 0.0,
                "finished": torch.zeros(self.num_codebook_levels, dtype=torch.bool, device=device),
                "eos_pos": torch.full((self.num_codebook_levels,), fill_value=-1, dtype=torch.long, device=device),
            }
        ]
        finalized: List[Tuple[float, torch.Tensor, str, int]] = []

        for row_idx in range(1, max_rows):
            if not beams:
                break
            seq_batch = torch.stack([beam["seq"] for beam in beams], dim=0)
            enc = encoder_hidden.expand(seq_batch.size(0), -1, -1)
            enc_mask = encoder_mask.expand(seq_batch.size(0), -1) if encoder_mask is not None else None
            prefix_batch = text_prefix_ids.expand(seq_batch.size(0), -1)
            spk_batch = unity_spk_embed.expand(seq_batch.size(0), -1) if unity_spk_embed is not None else None

            level0_logits = self._decode_next_level0_logits(
                seq_batch, enc, enc_mask, prefix_batch, spk_batch
            )
            level0_log_probs = F.log_softmax(level0_logits, dim=-1)
            level0_log_probs[:, bos_id] = -torch.inf
            if row_idx < min_eos_row:
                level0_log_probs[:, eos_id] = -torch.inf
            elif FREE_RUNNING_EOS_BONUS != 0.0:
                level0_log_probs[:, eos_id] = level0_log_probs[:, eos_id] + FREE_RUNNING_EOS_BONUS
            level0_log_probs = self._apply_level0_no_repeat_ngram_mask(level0_log_probs, seq_batch)

            candidates: List[Tuple[float, int, int]] = []
            top_scores, top_ids = torch.topk(
                level0_log_probs, k=min(FREE_RUNNING_BEAM_TOPK, level0_log_probs.size(-1)), dim=-1
            )
            for beam_idx, beam in enumerate(beams):
                seen_ids = set(top_ids[beam_idx].detach().cpu().tolist())
                if row_idx >= min_eos_row:
                    seen_ids.add(eos_id)
                for token_id in seen_ids:
                    token_score = float(level0_log_probs[beam_idx, token_id].item())
                    if not math.isfinite(token_score):
                        continue
                    if token_id in self._level0_banned_ngram_tokens(beam["seq"]):
                        continue
                    repeat_penalty = self._level0_repeat_penalty(beam["seq"], int(token_id))
                    score = float(beam["score"]) + token_score - repeat_penalty
                    candidates.append((score, beam_idx, int(token_id)))

            candidates.sort(key=lambda item: item[0], reverse=True)
            selected = candidates[:FREE_RUNNING_BEAM_SIZE]
            if not selected:
                break

            parent_indices = [beam_idx for _, beam_idx, _ in selected]
            forced_level0 = torch.tensor([tok for _, _, tok in selected], device=device, dtype=torch.long)
            parent_seq = torch.stack([beams[idx]["seq"] for idx in parent_indices], dim=0)
            parent_finished = torch.stack([beams[idx]["finished"] for idx in parent_indices], dim=0)
            parent_eos_pos = torch.stack([beams[idx]["eos_pos"] for idx in parent_indices], dim=0)

            next_row = self._make_two_stage_next_row(
                batch_size=parent_seq.size(0),
                forced_level0=forced_level0,
                row_idx=row_idx,
                bos_id=bos_id,
                ignore_index=model.ignore_index,
            )

            new_beams: List[Dict[str, Any]] = []
            for new_idx, (score, _, _) in enumerate(selected):
                finished = parent_finished[new_idx].clone()
                eos_pos = parent_eos_pos[new_idx].clone()
                for q in range(self.num_codebook_levels):
                    token = int(next_row[new_idx, q].item())
                    if not bool(finished[q].item()) and token == eos_id:
                        finished[q] = True
                        eos_pos[q] = row_idx

                seq = torch.cat([parent_seq[new_idx], next_row[new_idx : new_idx + 1]], dim=0)
                if FREE_RUNNING_STOP_ON_LEVEL0_EOS:
                    level0_eos_pos = int(eos_pos[0].item())
                    completed = bool(finished[0].item()) and (
                        seq.size(0) >= level0_eos_pos + 1 + FREE_RUNNING_LEVEL0_EOS_EXTRA_ROWS
                    )
                else:
                    completed = bool(finished.all().item())
                capped = seq.size(0) >= max_rows

                if completed or capped:
                    if completed:
                        used_rows = int(eos_pos[0].item()) + 1 + FREE_RUNNING_LEVEL0_EOS_EXTRA_ROWS
                        status = "completed_level0_eos"
                    else:
                        used_rows = max_rows
                        status = "duration_capped"
                    used_rows = min(used_rows, seq.size(0))
                    rank_score = self._beam_rank_score(score, used_rows, status)
                    finalized.append((rank_score, seq[:used_rows].detach().cpu(), status, used_rows))
                else:
                    new_beams.append({"seq": seq, "score": score, "finished": finished, "eos_pos": eos_pos})

            beams = sorted(new_beams, key=lambda item: float(item["score"]), reverse=True)[:FREE_RUNNING_BEAM_SIZE]

        for beam in beams:
            used_rows = min(max_rows, beam["seq"].size(0))
            status = "max_loop_reached"
            rank_score = self._beam_rank_score(float(beam["score"]), used_rows, status)
            finalized.append((rank_score, beam["seq"][:used_rows].detach().cpu(), status, used_rows))

        if not finalized:
            raise RuntimeError("Beam free-running decode failed to produce a sequence.")
        _, seq, status, used_rows = max(finalized, key=lambda item: item[0])
        print(f"[infer] level-0 decode finished: status={status} rows={used_rows}")
        self._beam_result = {"status": status, "rows": int(used_rows)}
        return self._fill_depth_levels(
            seq, encoder_hidden, encoder_mask, text_prefix_ids, unity_spk_embed, source_acoustic_embedding
        )

    @torch.no_grad()
    def translate_batch(self, source_wav_paths: List[str]) -> List[torch.Tensor]:
        """Translate utterances encoded together as one batch; returns mono
        waveforms at ``self.mimi_output_sample_rate``."""
        model = self.model
        wavs = [self._load_source_wav(p) for p in source_wav_paths]
        cond = self._encode_batch_conditions(wavs)

        def row(t, i):
            return t[i : i + 1] if t is not None else None

        spk = cond["unity_spk_embed"]
        prefix_rows = model.generate_text_prefix_ids(
            cond["encoder_hidden"], cond["encoder_mask"], spk_embed=spk, sep_prompt_embedding=spk
        )

        outputs: List[torch.Tensor] = []
        self.last_batch_debug = []
        for i in range(len(wavs)):
            # Per-sample depth-decoder prompts.
            if cond["codec_tokens"] is not None and cond["codec_mask"] is not None:
                model.depth_decoder.set_runtime_source_codec_prompt(
                    row(cond["codec_tokens"], i), row(cond["codec_mask"], i), zero_prompt=False
                )
            if cond["campplus"] is not None:
                model.depth_decoder.set_runtime_campplus_source_embedding(row(cond["campplus"], i))
            delayed_seq = self._generate_beam(
                cond["encoder_hidden"][i : i + 1],
                row(cond["encoder_mask"], i),
                prefix_rows[i : i + 1],
                row(cond["unity_spk_embed"], i),
                cond["target_rows"][i],
                source_acoustic_embedding=row(cond["source_acoustic_embedding"], i),
            )
            mimi_codes = _delayed_codes_to_mimi_codes(
                delayed_seq, self.codebook_size, self.codebook_mimi_to_model_order
            )
            audio = self.mimi_decode_model.decode(
                audio_codes=mimi_codes.unsqueeze(0).to(device=self.codec_decode_device), return_dict=True
            ).audio_values
            outputs.append(audio.squeeze(0).squeeze(0).float().detach().cpu())
            self.last_batch_debug.append(
                {
                    "text_prefix_ids": prefix_rows[i].detach().cpu().tolist(),
                    "target_rows": cond["target_rows"][i],
                    **self._beam_result,
                }
            )
        return outputs

    def translate(self, source_wav_path: str) -> torch.Tensor:
        """Translate one utterance on its own."""
        return self.translate_batch([source_wav_path])[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, help="Directory with model_config.json + pytorch_model.bin")
    parser.add_argument("--source-wav", help="Single utterance to translate (a batch of one)")
    parser.add_argument("--output-wav", help="Output path for --source-wav")
    parser.add_argument("--manifest", help="TSV of utterances to translate in manifest order")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--audio-col", default="audio_path")
    parser.add_argument(
        "--audio-root",
        default=None,
        help="Directory that relative paths in --audio-col are resolved against.",
    )
    parser.add_argument("--output-dir", help="Output directory for --manifest (<id>.wav + generation.tsv)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=30,
        help="Consecutive manifest rows encoded together.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--codec-decode-device",
        default="cpu",
        help="Device for the final Mimi decode. cpu matches training-time evaluation.",
    )
    parser.add_argument(
        "--buffer-dtype",
        choices=["bfloat16", "float32"],
        default="bfloat16",
        help="dtype of the model's floating-point buffers. bfloat16 matches training "
        "(see DirectS2STInference.__init__).",
    )
    parser.add_argument(
        "--campplus-model-root",
        default=None,
        help="Local clone of https://github.com/Plachtaa/seed-vc (GPL-3.0, not vendored here; "
        "see docs/model_dependencies.md). Only needed if the checkpoint's config doesn't set it.",
    )
    parser.add_argument(
        "--campplus-checkpoint-path",
        default=None,
        help="Local copy of iic/speech_campplus_sv_zh_en_16k-common_advanced's campplus_cn_en_common.pt "
        "(Apache-2.0, ModelScope). Only needed if the checkpoint's config doesn't set it.",
    )
    args = parser.parse_args()
    if bool(args.source_wav) == bool(args.manifest):
        parser.error("pass exactly one of --source-wav or --manifest")
    if args.source_wav and not args.output_wav:
        parser.error("--source-wav requires --output-wav")
    if args.manifest and not args.output_dir:
        parser.error("--manifest requires --output-dir")

    engine = DirectS2STInference(
        args.checkpoint_dir,
        device=args.device,
        campplus_model_root=args.campplus_model_root,
        campplus_checkpoint_path=args.campplus_checkpoint_path,
        codec_decode_device=args.codec_decode_device,
        buffer_dtype=getattr(torch, args.buffer_dtype),
    )
    if args.source_wav:
        audio = engine.translate(args.source_wav)
        torchaudio.save(args.output_wav, audio.unsqueeze(0), engine.mimi_output_sample_rate)
        print(f"[infer] wrote {args.output_wav}")
        return

    rows = list(csv.DictReader(open(args.manifest, newline="", encoding="utf-8"), delimiter="\t"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batches = [rows[s : s + args.batch_size] for s in range(0, len(rows), args.batch_size)]

    def audio_path(r):
        path = r[args.audio_col]
        return os.path.join(args.audio_root, path) if args.audio_root and not os.path.isabs(path) else path

    records = []
    done = 0
    for chunk in batches:
        audios = engine.translate_batch([audio_path(r) for r in chunk])
        done += len(chunk)
        for r, audio, dbg in zip(chunk, audios, engine.last_batch_debug):
            wav_path = out_dir / f"{r[args.id_col]}.wav"
            torchaudio.save(str(wav_path), audio.unsqueeze(0), engine.mimi_output_sample_rate)
            records.append(
                {
                    "id": r[args.id_col],
                    "source_audio": audio_path(r),
                    "hypo_audio": str(wav_path),
                    "status": dbg["status"],
                    "rows": dbg["rows"],
                    "target_rows": dbg["target_rows"],
                }
            )
        print(f"[infer] {done}/{len(rows)}", flush=True)
    with open(out_dir / "generation.tsv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"[infer] wrote {len(records)} utterances to {out_dir}")


if __name__ == "__main__":
    main()
