"""Training data: TSV manifests -> HF datasets -> padded DirectS2ST batches.

Each manifest row is one source/target pair:

  src_audio                     source (English) waveform path
  mimi_codes                    target Mimi codes, space-separated, flattened [1, Q, F]
  mimi_codes_shape              "1x<Q>x<F>"
  <target_text_column>          target transcript
  <source_text_column>          source transcript
  mimi_source_acoustic_embedding_{path,index,dim}   precomputed pooled Mimi source features
  campplus_source_embedding_{path,index,dim}        precomputed CAMPPlus source speaker vector
  campplus_target_embedding_{path,index,dim}        precomputed CAMPPlus target speaker vector

``*_path`` points to an ``.npy`` array ``[N, D]`` and ``*_index`` selects the row.
The scripts in ``scripts/data/`` produce these columns (see README.md).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio

_resample_cache: Dict[Tuple[int, int], torchaudio.transforms.Resample] = {}


def _load_waveform(audio_item: Any, out_sr: int) -> torch.Tensor:
    """
    Supports:
    - HF Audio dict: {"array": np.ndarray/list/tensor, "sampling_rate": int}
    - path string
    - torch tensor (assumed already at out_sr)
    """
    if isinstance(audio_item, dict) and "array" in audio_item and "sampling_rate" in audio_item:
        wav = torch.as_tensor(audio_item["array"], dtype=torch.float32)
        sr = int(audio_item["sampling_rate"])
    elif isinstance(audio_item, str):
        wav, sr = torchaudio.load(audio_item)
    elif torch.is_tensor(audio_item):
        wav = audio_item.float()
        sr = out_sr
    else:
        raise ValueError(
            "Unsupported audio item. Expected HF Audio dict, file path, or tensor."
        )

    if wav.ndim == 2:
        # [channels, time] -> mono
        wav = wav.mean(dim=0)
    elif wav.ndim > 2:
        wav = wav.reshape(-1)

    if sr != out_sr:
        cache_key = (sr, out_sr)
        if cache_key not in _resample_cache:
            _resample_cache[cache_key] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=out_sr)
        wav = _resample_cache[cache_key](wav)

    return wav


def _normalize_code_stack(codes: Any, expected_quantizers: int) -> torch.Tensor:
    """
    Convert code input to [F, N].
    Accepts:
    - [N, F] nested list / tensor (preferred request format)
    - [F, N] (accepted)
    - [F] if N=1
    """
    x = torch.as_tensor(codes, dtype=torch.long)

    if x.ndim == 1:
        if expected_quantizers != 1:
            raise ValueError(
                f"Received 1D codes but num_quantizers={expected_quantizers}. "
                "Provide stacked codes with shape [N, F]."
            )
        return x.unsqueeze(-1)

    if x.ndim != 2:
        raise ValueError(f"Expected codes to be 1D or 2D, got shape={tuple(x.shape)}.")

    # Convert request format [N, F] -> [F, N].
    if x.shape[0] == expected_quantizers and x.shape[1] != expected_quantizers:
        return x.transpose(0, 1)
    if x.shape[1] == expected_quantizers:
        return x
    raise ValueError(
        f"Code stack shape mismatch for tgt. Expected [N,F] with N={expected_quantizers} "
        f"or [F,N], got {tuple(x.shape)}."
    )


def _add_bos_eos(codes_f_n: torch.Tensor, bos_id: int, eos_id: int) -> torch.Tensor:
    """
    Prepend BOS and append EOS to each codebook sequence.
    Input: [F, N] where F = frames, N = num_quantizers
    Output: [F + 2, N] with BOS at position 0 and EOS at position F+1
    """
    if codes_f_n.ndim != 2:
        raise ValueError(f"_add_bos_eos expects [F, N], got {tuple(codes_f_n.shape)}.")

    frames, n_codebooks = codes_f_n.shape
    result = torch.empty(
        (frames + 2, n_codebooks),
        dtype=codes_f_n.dtype,
        device=codes_f_n.device,
    )
    result[0, :] = bos_id
    result[1 : frames + 1, :] = codes_f_n
    result[frames + 1, :] = eos_id
    return result


def _load_indexed_vector(
    cache: Dict[str, np.ndarray],
    feature: Dict[str, Any],
    path_key: str,
    index_key: str,
    dim_key: str,
    label: str,
    mmap: bool,
) -> Optional[torch.Tensor]:
    """Row ``feature[index_key]`` of the ``[N, D]`` array at ``feature[path_key]``."""
    path_value = feature.get(path_key)
    if path_value is None or str(path_value).strip() == "":
        return None
    index_value = feature.get(index_key)
    if index_value is None or str(index_value).strip() == "":
        raise KeyError(f"{label} column '{path_key}' is present, but '{index_key}' is missing/empty.")
    path = str(path_value).strip()
    index = int(index_value)
    array = cache.get(path)
    if array is None:
        array = np.load(path, mmap_mode="r") if mmap else np.load(path)
        if array.ndim != 2:
            raise ValueError(f"Expected {label} array [N, D], got {array.shape}: {path}")
        cache[path] = array
    if index < 0 or index >= int(array.shape[0]):
        raise IndexError(f"{label} index {index} out of range for {path} with rows={array.shape[0]}")
    vector = np.asarray(array[index], dtype=np.float32)
    dim_value = feature.get(dim_key)
    if dim_value is not None and str(dim_value).strip() != "":
        expected_dim = int(dim_value)
        if expected_dim != int(vector.shape[0]):
            raise ValueError(
                f"Expected {label} dim {expected_dim}, got {vector.shape[0]} from {path}[{index}]"
            )
    return torch.from_numpy(vector.copy()).float()


class WaveformCodeStackCollator:
    """Pads a list of manifest rows into one DirectS2ST training batch.

    Target codes are frame-aligned [F + 2, Q] with BOS/EOS rows; the target text
    prefix is ``BOS <start_content> text <end_content> SEP`` and the source
    text prefix ``BOS <start_content> text <end_content>`` (UniSS-style content
    controls). The zero-padded source waveforms are returned for the Mimi
    source prompt and CAMPPlus, alongside the precomputed conditioning vectors.
    """

    def __init__(
        self,
        feature_extractor,
        audio_key: str,
        codes_key: str,
        sample_rate: int,
        num_q: int,
        bos_id: int,
        eos_id: int,
        speech_max_audio_samples: Optional[int] = None,
        ignore_index: int = -100,
        text_tokenizer: Optional[Any] = None,
        text_key: str = "sentence",
        text_max_tokens: int = 128,
        text_bos_token_id: Optional[int] = None,
        text_eos_token_id: Optional[int] = None,
        text_pad_token_id: Optional[int] = None,
        text_extra_token_ids_after_bos: Optional[List[int]] = None,
        source_text_key: Optional[str] = None,
        source_text_max_tokens: Optional[int] = None,
        source_text_extra_token_ids_after_bos: Optional[List[int]] = None,
        uniss_start_content_token_id: Optional[int] = None,
        uniss_end_content_token_id: Optional[int] = None,
        return_source_acoustic_wav: bool = True,
        source_acoustic_embedding_path_key: str = "mimi_source_acoustic_embedding_path",
        source_acoustic_embedding_index_key: str = "mimi_source_acoustic_embedding_index",
        source_acoustic_embedding_dim_key: str = "mimi_source_acoustic_embedding_dim",
        use_precomputed_campplus_embeddings: bool = True,
        campplus_embedding_path_key: str = "campplus_source_embedding_path",
        campplus_embedding_index_key: str = "campplus_source_embedding_index",
        campplus_embedding_dim_key: str = "campplus_source_embedding_dim",
        use_precomputed_campplus_target_embeddings: bool = True,
        campplus_target_embedding_path_key: str = "campplus_target_embedding_path",
        campplus_target_embedding_index_key: str = "campplus_target_embedding_index",
        campplus_target_embedding_dim_key: str = "campplus_target_embedding_dim",
        allow_online_campplus_fallback: bool = False,
    ):
        self.feature_extractor = feature_extractor
        self.audio_key = audio_key
        self.codes_key = codes_key
        self.sample_rate = sample_rate
        self.speech_max_audio_samples = speech_max_audio_samples
        self.num_q = num_q
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.ignore_index = ignore_index
        self.text_tokenizer = text_tokenizer
        self.text_key = text_key
        self.text_max_tokens = text_max_tokens
        self.text_bos_token_id = text_bos_token_id
        self.text_eos_token_id = text_eos_token_id
        self.text_pad_token_id = text_pad_token_id
        self.text_extra_token_ids_after_bos = [int(t) for t in (text_extra_token_ids_after_bos or [])]
        self.source_text_key = str(source_text_key).strip() if source_text_key else None
        self.source_text_max_tokens = int(
            self.text_max_tokens if source_text_max_tokens is None else source_text_max_tokens
        )
        self.source_text_extra_token_ids_after_bos = [
            int(t) for t in (source_text_extra_token_ids_after_bos or [])
        ]
        self.uniss_start_content_token_id = uniss_start_content_token_id
        self.uniss_end_content_token_id = uniss_end_content_token_id
        self.return_source_acoustic_wav = bool(return_source_acoustic_wav)
        self.source_acoustic_keys = (
            source_acoustic_embedding_path_key,
            source_acoustic_embedding_index_key,
            source_acoustic_embedding_dim_key,
        )
        self.use_precomputed_campplus_embeddings = bool(use_precomputed_campplus_embeddings)
        self.campplus_keys = (campplus_embedding_path_key, campplus_embedding_index_key, campplus_embedding_dim_key)
        self.use_precomputed_campplus_target_embeddings = bool(use_precomputed_campplus_target_embeddings)
        self.campplus_target_keys = (
            campplus_target_embedding_path_key,
            campplus_target_embedding_index_key,
            campplus_target_embedding_dim_key,
        )
        self.allow_online_campplus_fallback = bool(allow_online_campplus_fallback)
        self._source_acoustic_embedding_cache: Dict[str, np.ndarray] = {}
        self._campplus_embedding_cache: Dict[str, np.ndarray] = {}

    def _encode_plain_text(
        self,
        text: Any,
        *,
        max_tokens: int,
        extra_token_ids_after_bos: List[int],
        label: str,
        append_terminal_sep: bool = False,
    ) -> torch.Tensor:
        bos_id = self.text_bos_token_id
        eos_id = self.text_eos_token_id
        if bos_id is None:
            bos_id = eos_id
        if eos_id is None:
            eos_id = bos_id
        if bos_id is None or eos_id is None:
            raise ValueError(f"{label} prefix must provide BOS and EOS/SEP IDs.")
        if self.text_tokenizer is None:
            raise RuntimeError(f"text_tokenizer is required for {label} encoding.")
        if text is None:
            text = ""
        use_content_controls = (
            self.uniss_start_content_token_id is not None
            and self.uniss_end_content_token_id is not None
        )
        structural_tokens = 3 if use_content_controls else 2
        if append_terminal_sep:
            structural_tokens += 1
        token_ids = self.text_tokenizer.encode(
            str(text),
            add_special_tokens=False,
            truncation=True,
            max_length=max(0, int(max_tokens) - structural_tokens - len(extra_token_ids_after_bos)),
        )
        if use_content_controls:
            sequence = [
                int(bos_id),
                *extra_token_ids_after_bos,
                int(self.uniss_start_content_token_id),
                *token_ids,
                int(self.uniss_end_content_token_id),
            ]
            if append_terminal_sep:
                sequence.append(int(eos_id))
        else:
            sequence = [int(bos_id), *extra_token_ids_after_bos, *token_ids, int(eos_id)]
        return torch.tensor(sequence, dtype=torch.long)

    def _encode_target_text(self, text: Any) -> torch.Tensor:
        return self._encode_plain_text(
            text,
            max_tokens=self.text_max_tokens,
            extra_token_ids_after_bos=self.text_extra_token_ids_after_bos,
            label="target-text",
            append_terminal_sep=(self.uniss_start_content_token_id is not None),
        )

    def _encode_source_text(self, text: Any) -> torch.Tensor:
        return self._encode_plain_text(
            text,
            max_tokens=self.source_text_max_tokens,
            extra_token_ids_after_bos=self.source_text_extra_token_ids_after_bos,
            label="source-text",
        )

    def _load_precomputed_source_acoustic_embedding(self, feature: Dict[str, Any]) -> Optional[torch.Tensor]:
        # Absent -> the model encodes the source with its internal Mimi online.
        return _load_indexed_vector(
            self._source_acoustic_embedding_cache, feature, *self.source_acoustic_keys,
            label="source acoustic embedding", mmap=True,
        )

    def _load_precomputed_campplus_embedding(self, feature: Dict[str, Any]) -> Optional[torch.Tensor]:
        if not self.use_precomputed_campplus_embeddings:
            return None
        path_value = feature.get(self.campplus_keys[0])
        if path_value is None or str(path_value).strip() == "":
            if self.allow_online_campplus_fallback:
                return None
            raise KeyError(f"Required precomputed CAMPPlus column '{self.campplus_keys[0]}' is missing/empty.")
        # CAMPPlus tables are compact (192 floats per utterance); load whole.
        return _load_indexed_vector(
            self._campplus_embedding_cache, feature, *self.campplus_keys,
            label="CAMPPlus embedding", mmap=False,
        )

    def _load_precomputed_campplus_target_embedding(self, feature: Dict[str, Any]) -> Optional[torch.Tensor]:
        if not self.use_precomputed_campplus_target_embeddings:
            return None
        return _load_indexed_vector(
            self._campplus_embedding_cache, feature, *self.campplus_target_keys,
            label="target CAMPPlus embedding", mmap=False,
        )

    @staticmethod
    def _stack_all_or_none(values: List[Optional[torch.Tensor]], label: str) -> Optional[torch.Tensor]:
        if not any(v is not None for v in values):
            return None
        if not all(v is not None for v in values):
            raise ValueError(f"Mixed precomputed/non-precomputed {label} in one batch.")
        return torch.stack(values, dim=0)

    @staticmethod
    def _pad_1d(sequences: List[torch.Tensor], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        max_len = max(t.numel() for t in sequences)
        ids = torch.full((len(sequences), max_len), fill_value=pad_id, dtype=torch.long)
        mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
        for i, seq in enumerate(sequences):
            ids[i, : seq.numel()] = seq
            mask[i, : seq.numel()] = 1
        return ids, mask

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        waveforms: List[torch.Tensor] = []
        code_stacks: List[torch.Tensor] = []
        text_sequences: List[torch.Tensor] = []
        source_text_sequences: List[torch.Tensor] = []
        source_acoustic_embeddings: List[Optional[torch.Tensor]] = []
        campplus_source_embeddings: List[Optional[torch.Tensor]] = []
        campplus_target_embeddings: List[Optional[torch.Tensor]] = []

        for feature in features:
            for key in (self.audio_key, self.codes_key):
                if key not in feature:
                    raise KeyError(f"Missing column '{key}'. Available columns: {list(feature.keys())}")
            if self.text_tokenizer is not None and self.text_key not in feature:
                raise KeyError(f"Missing target text column '{self.text_key}'.")
            if self.source_text_key is not None and self.source_text_key not in feature:
                raise KeyError(f"Missing source text column '{self.source_text_key}'.")

            source_acoustic_embeddings.append(self._load_precomputed_source_acoustic_embedding(feature))
            campplus_source_embeddings.append(self._load_precomputed_campplus_embedding(feature))
            campplus_target_embeddings.append(self._load_precomputed_campplus_target_embedding(feature))

            wav = _load_waveform(feature[self.audio_key], self.sample_rate)
            if self.speech_max_audio_samples is not None:
                wav = wav[: self.speech_max_audio_samples]
            waveforms.append(wav)
            codes_f_n = _normalize_code_stack(feature[self.codes_key], self.num_q)
            codes_f_n = _add_bos_eos(codes_f_n, bos_id=self.bos_id, eos_id=self.eos_id)
            code_stacks.append(codes_f_n)
            if self.text_tokenizer is not None:
                text_sequences.append(self._encode_target_text(feature[self.text_key]))
            if self.source_text_key is not None:
                source_text_sequences.append(self._encode_source_text(feature[self.source_text_key]))

        speech_inputs = self.feature_extractor(
            [w.cpu().numpy() for w in waveforms],
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
            padding=True,
            truncation=False,
        )

        max_t = max(c.shape[0] for c in code_stacks)
        code_labels = torch.full(
            (len(code_stacks), max_t, self.num_q), fill_value=self.ignore_index, dtype=torch.long
        )
        code_attention_mask = torch.zeros((len(code_stacks), max_t), dtype=torch.long)
        for i, c in enumerate(code_stacks):
            code_labels[i, : c.shape[0], :] = c
            code_attention_mask[i, : c.shape[0]] = 1

        batch = {
            "input_features": speech_inputs["input_features"],
            "labels": code_labels,
            "code_attention_mask": code_attention_mask,
        }
        if "attention_mask" in speech_inputs:
            batch["audio_attention_mask"] = speech_inputs["attention_mask"]
        if self.return_source_acoustic_wav:
            max_wav_len = max(int(wav.numel()) for wav in waveforms)
            source_wavs = torch.zeros((len(waveforms), max_wav_len), dtype=torch.float32)
            source_wav_lens = torch.zeros((len(waveforms),), dtype=torch.long)
            for i, wav in enumerate(waveforms):
                wav_1d = wav.detach().cpu().to(torch.float32).flatten()
                source_wavs[i, : wav_1d.numel()] = wav_1d
                source_wav_lens[i] = int(wav_1d.numel())
            batch["source_acoustic_wavs"] = source_wavs
            batch["source_acoustic_wav_lens"] = source_wav_lens
        for key, values, label in (
            ("source_acoustic_embeddings", source_acoustic_embeddings, "source acoustic embeddings"),
            ("campplus_source_embeddings", campplus_source_embeddings, "CAMPPlus embeddings"),
            ("campplus_target_embeddings", campplus_target_embeddings, "target CAMPPlus embeddings"),
        ):
            stacked = self._stack_all_or_none(values, label)
            if stacked is not None:
                batch[key] = stacked
        if self.text_tokenizer is not None:
            batch["text_input_ids"], batch["text_attention_mask"] = self._pad_1d(
                text_sequences, int(self.text_pad_token_id)
            )
        if self.source_text_key is not None:
            batch["source_text_input_ids"], batch["source_text_attention_mask"] = self._pad_1d(
                source_text_sequences, int(self.text_pad_token_id)
            )
        return batch


def load_manifest_dataset(
    sources: Any,
    *,
    audio_column: str,
    codes_column: str,
    num_quantizers: int,
    num_modeled_levels: int,
    codebook_model_order: Optional[List[int]],
    passthrough_columns: List[str],
    num_proc: int = 1,
    tsv_audio_path_column: str = "src_audio",
    tsv_codes_column: str = "mimi_codes",
    tsv_codes_shape_column: str = "mimi_codes_shape",
):
    """Load one TSV or a list of TSVs into a ``datasets.Dataset`` of
    ``{audio_column: path, codes_column: [Q][F] codes, *passthrough_columns}``.

    Codes are cut to ``num_quantizers`` levels and, if ``codebook_model_order``
    is given, the first ``num_modeled_levels`` are reordered into the model's
    internal level order.
    """
    from datasets import concatenate_datasets, load_dataset

    def parse(example: Dict[str, Any]) -> Dict[str, Any]:
        audio_path = example.get(tsv_audio_path_column)
        if audio_path is None or str(audio_path).strip() == "":
            raise KeyError(f"TSV row is missing '{tsv_audio_path_column}'.")
        codes_text = example.get(tsv_codes_column)
        if codes_text is None:
            raise KeyError(f"TSV row is missing '{tsv_codes_column}'.")
        flat_codes = [int(tok) for tok in str(codes_text).strip().split() if tok != ""]
        if not flat_codes:
            raise ValueError(f"Encountered empty {tsv_codes_column} field.")
        shape_text = str(example.get(tsv_codes_shape_column, "")).strip().lower()
        shape_parts = [p for p in shape_text.replace(" ", "").split("x") if p != ""]
        if len(shape_parts) != 3 or int(shape_parts[0]) != 1:
            raise ValueError(f"Invalid {tsv_codes_shape_column}='{shape_text}'; expected '1x<levels>x<frames>'.")
        n_q_total, n_f_total = int(shape_parts[1]), int(shape_parts[2])
        if n_q_total * n_f_total != len(flat_codes):
            raise ValueError(
                f"Malformed {tsv_codes_column}: expected {n_q_total * n_f_total} tokens, got {len(flat_codes)}."
            )
        if num_quantizers > n_q_total:
            raise ValueError(f"Requested {num_quantizers} quantizers, but the row has only {n_q_total}.")
        codes_n_f = [flat_codes[q * n_f_total : (q + 1) * n_f_total] for q in range(n_q_total)][:num_quantizers]
        if codebook_model_order is not None:
            modeled = [codes_n_f[mimi_idx] for mimi_idx in codebook_model_order]
            codes_n_f = modeled + codes_n_f[num_modeled_levels:num_quantizers]
        parsed = {audio_column: str(audio_path), codes_column: codes_n_f}
        for column in passthrough_columns:
            value = example.get(column)
            if value is not None and str(value).strip() != "":
                parsed[column] = str(value)
        return parsed

    def load_one(path: str):
        ds = load_dataset("csv", data_files=path, delimiter="\t", split="train")
        remove_cols = [c for c in ds.column_names if c not in {audio_column, codes_column}]
        return ds.map(parse, remove_columns=remove_cols, num_proc=num_proc)

    if isinstance(sources, str):
        return load_one(sources)
    return concatenate_datasets([load_one(path) for path in sources])
