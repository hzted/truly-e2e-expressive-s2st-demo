#!/usr/bin/env python3
"""Precompute pooled Mimi source-acoustic embeddings for DirectS2ST.

This mirrors WhisperTemporalDepthTransformer._encode_mimi_source_acoustic_embedding:
  source wav -> Mimi encoder -> encoder_transformer -> optional downsample -> masked pooling.

Outputs, for each input TSV:
  - <stem>_<suffix>.npy: [num_rows, embedding_dim] float matrix
  - <stem>_<suffix>.tsv: original TSV plus embedding path/index metadata
  - <stem>_<suffix>.json: metadata for reproducibility
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchaudio
from transformers import AutoFeatureExtractor, MimiModel


def _read_tsv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"TSV has no header: {path}")
        rows = [dict(row) for row in reader]
        return rows, list(reader.fieldnames)


def _write_tsv(path: Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _resolve_audio_path(value: str, audio_root: Optional[Path]) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError("empty audio path")
    path = Path(text)
    if not path.is_absolute() and audio_root is not None:
        path = audio_root / path
    return path


def _load_audio_mono(path: Path, input_sample_rate: int, max_seconds: Optional[float]) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.numel() == 0:
        raise ValueError(f"empty audio: {path}")
    wav = wav.to(torch.float32)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    if sr != input_sample_rate:
        wav = torchaudio.functional.resample(wav, sr, input_sample_rate)
    if max_seconds is not None and max_seconds > 0:
        max_len = int(round(float(max_seconds) * float(input_sample_rate)))
        if wav.numel() > max_len:
            wav = wav[:max_len]
    return wav.contiguous()


def _pad_waveforms(wavs: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([max(1, int(w.numel())) for w in wavs], dtype=torch.long)
    max_len = int(lengths.max().item())
    batch = torch.zeros((len(wavs), max_len), dtype=torch.float32)
    for i, wav in enumerate(wavs):
        batch[i, : wav.numel()] = wav
    return batch, lengths


class MimiSourceAcousticPooler:
    def __init__(
        self,
        model_name: str,
        input_sample_rate: int,
        device: str,
        pooling: str,
    ) -> None:
        self.model_name = model_name
        self.input_sample_rate = int(input_sample_rate)
        self.device = torch.device(device)
        self.pooling = pooling

        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model_sample_rate = int(getattr(feature_extractor, "sampling_rate", self.input_sample_rate))
        self.model = MimiModel.from_pretrained(model_name).to(self.device)
        self.model.eval().requires_grad_(False)
        self.hidden_size = int(getattr(self.model.config, "hidden_size"))

    @torch.no_grad()
    def __call__(self, wavs_16k: torch.Tensor, wav_lens_16k: torch.Tensor) -> torch.Tensor:
        wavs = wavs_16k.to(device=self.device, dtype=torch.float32)
        wav_lens = wav_lens_16k.to(device=self.device, dtype=torch.long).clamp_min(1)

        if self.input_sample_rate != self.model_sample_rate:
            resampled: List[torch.Tensor] = []
            new_lens: List[int] = []
            for i in range(wavs.size(0)):
                wav_len = int(wav_lens[i].item())
                wav_i = wavs[i, :wav_len]
                wav_i = torchaudio.functional.resample(wav_i, self.input_sample_rate, self.model_sample_rate)
                resampled.append(wav_i)
                new_lens.append(int(wav_i.numel()))
            max_len = max(new_lens)
            padded = wavs.new_zeros((len(resampled), max_len))
            for i, wav_i in enumerate(resampled):
                padded[i, : wav_i.numel()] = wav_i
            wavs = padded
            wav_lens = torch.tensor(new_lens, device=self.device, dtype=torch.long).clamp_min(1)

        input_values = wavs.unsqueeze(1)
        hidden = self.model.encoder(input_values)
        encoder_outputs = self.model.encoder_transformer(hidden.transpose(1, 2), return_dict=True)
        hidden = encoder_outputs[0].transpose(1, 2)
        if getattr(self.model, "downsample", None) is not None:
            hidden = self.model.downsample(hidden)

        encoded_lens = self.model.get_encoded_length(wav_lens).to(device=hidden.device)
        max_frames = int(hidden.size(-1))
        frame_ids = torch.arange(max_frames, device=hidden.device).unsqueeze(0)
        frame_mask = frame_ids < encoded_lens.clamp(min=1, max=max_frames).unsqueeze(1)
        weights = frame_mask.to(dtype=hidden.dtype).unsqueeze(1)
        denom = weights.sum(dim=-1).clamp_min(1.0)
        mean = (hidden * weights).sum(dim=-1) / denom

        if self.pooling == "mean":
            return mean
        centered = (hidden - mean.unsqueeze(-1)) * weights
        var = (centered.square()).sum(dim=-1) / denom
        std = torch.sqrt(var.clamp_min(1.0e-12))
        if self.pooling == "std":
            return std
        if self.pooling == "mean_std":
            return torch.cat([mean, std], dim=-1)
        raise ValueError(f"unsupported pooling: {self.pooling}")


def _batched_indices(n: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, n, batch_size):
        yield start, min(n, start + batch_size)


def process_tsv(
    input_tsv: Path,
    output_dir: Path,
    audio_col: str,
    id_col: Optional[str],
    audio_root: Optional[Path],
    pooler: MimiSourceAcousticPooler,
    batch_size: int,
    dtype: np.dtype,
    suffix: str,
    overwrite: bool,
    limit: Optional[int],
    allow_errors: bool,
    max_seconds: Optional[float],
    progress_every: int,
    resume: bool = False,
) -> None:
    rows, fieldnames = _read_tsv(input_tsv)
    if limit is not None and limit > 0:
        rows = rows[: int(limit)]
    if not rows:
        raise ValueError(f"no rows to process: {input_tsv}")
    if audio_col not in fieldnames:
        raise KeyError(f"audio column {audio_col!r} not found in {input_tsv}; columns={fieldnames}")

    stem = input_tsv.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    out_npy = output_dir / f"{stem}_{suffix}.npy"
    out_tsv = output_dir / f"{stem}_{suffix}.tsv"
    out_json = output_dir / f"{stem}_{suffix}.json"
    progress_path = output_dir / f"{stem}_{suffix}.progress.json"
    contract = dict(input_tsv=str(input_tsv.resolve()), rows=len(rows),
                    input_size=input_tsv.stat().st_size,
                    input_mtime_ns=input_tsv.stat().st_mtime_ns,
                    model=pooler.model_name, pooling=pooler.pooling,
                    input_sample_rate=pooler.input_sample_rate,
                    batch_size=batch_size, dtype=str(dtype),
                    max_seconds=max_seconds, audio_col=audio_col)
    start_index = 0
    if resume and allow_errors:
        raise ValueError('--resume requires fail-fast processing (no --allow-errors)')
    if resume and progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress['contract'] != contract:
            raise ValueError(f'Resume input/settings changed: {progress_path}')
        start_index = int(progress['next_index'])
        if not 0 <= start_index <= len(rows):
            raise ValueError(f'Invalid progress: {progress_path}')
        if start_index == len(rows) and out_tsv.exists() and out_json.exists():
            print(f'[resume] already complete: {out_tsv}', flush=True)
            return
    for path in (out_npy, out_tsv, out_json):
        if path.exists() and not overwrite and not (resume and progress_path.exists()):
            raise FileExistsError(f"output exists; pass --overwrite to replace: {path}")
    if resume and not progress_path.exists():
        progress_path.write_text(json.dumps(dict(contract=contract, next_index=0)))

    extra_fields = [
        "mimi_source_acoustic_embedding_path",
        "mimi_source_acoustic_embedding_index",
        "mimi_source_acoustic_embedding_dim",
        "mimi_source_acoustic_pooling",
        "mimi_source_acoustic_model",
        "mimi_source_acoustic_status",
        "mimi_source_acoustic_error",
    ]
    merged_fieldnames = list(fieldnames)
    for name in extra_fields:
        if name not in merged_fieldnames:
            merged_fieldnames.append(name)

    n = len(rows)
    mmap = None
    embedding_dim = None
    errors = 0
    t0 = time.time()
    if start_index:
        mmap = np.load(out_npy, mmap_mode='r+')
        embedding_dim = int(mmap.shape[1])
        if mmap.shape[0] != n or mmap.dtype != dtype:
            raise ValueError(f'Invalid resume array: {out_npy}')
        for i in range(start_index):
            rows[i].update(mimi_source_acoustic_embedding_path=str(out_npy),
                           mimi_source_acoustic_embedding_index=str(i),
                           mimi_source_acoustic_embedding_dim=str(embedding_dim),
                           mimi_source_acoustic_pooling=pooler.pooling,
                           mimi_source_acoustic_model=pooler.model_name,
                           mimi_source_acoustic_status='ok', mimi_source_acoustic_error='')
        print(f'[resume] {input_tsv.name}: {start_index}/{n}', flush=True)

    for start in range(start_index, n, batch_size):
        end = min(n, start + batch_size)
        batch_rows = rows[start:end]
        wavs: List[torch.Tensor] = []
        ok_positions: List[int] = []
        batch_errors: Dict[int, str] = {}

        for local_i, row in enumerate(batch_rows):
            global_i = start + local_i
            try:
                audio_path = _resolve_audio_path(row.get(audio_col, ""), audio_root)
                wav = _load_audio_mono(audio_path, pooler.input_sample_rate, max_seconds)
                wavs.append(wav)
                ok_positions.append(global_i)
            except Exception as exc:
                if not allow_errors:
                    raise RuntimeError(f"failed loading row={global_i} input={input_tsv}: {exc}") from exc
                errors += 1
                batch_errors[global_i] = repr(exc)

        if wavs:
            wav_batch, wav_lens = _pad_waveforms(wavs)
            embeddings = pooler(wav_batch, wav_lens).detach().cpu().numpy().astype(dtype, copy=False)
            if embedding_dim is None:
                embedding_dim = int(embeddings.shape[1])
                mmap = np.lib.format.open_memmap(out_npy, mode="w+", dtype=dtype, shape=(n, embedding_dim))
                mmap[:] = 0
            elif int(embeddings.shape[1]) != embedding_dim:
                raise RuntimeError(f"embedding dim changed: {embeddings.shape[1]} != {embedding_dim}")
            assert mmap is not None
            for emb_i, global_i in enumerate(ok_positions):
                mmap[global_i] = embeddings[emb_i]
                rows[global_i]["mimi_source_acoustic_status"] = "ok"
                rows[global_i]["mimi_source_acoustic_error"] = ""

        for global_i, error in batch_errors.items():
            rows[global_i]["mimi_source_acoustic_status"] = "error"
            rows[global_i]["mimi_source_acoustic_error"] = error

        if embedding_dim is not None:
            for global_i in range(start, end):
                rows[global_i]["mimi_source_acoustic_embedding_path"] = str(out_npy)
                rows[global_i]["mimi_source_acoustic_embedding_index"] = str(global_i)
                rows[global_i]["mimi_source_acoustic_embedding_dim"] = str(embedding_dim)
                rows[global_i]["mimi_source_acoustic_pooling"] = pooler.pooling
                rows[global_i]["mimi_source_acoustic_model"] = pooler.model_name

        if progress_every > 0 and (end == n or end % progress_every == 0):
            if resume and mmap is not None:
                mmap.flush()
                temp_progress = progress_path.with_suffix('.tmp')
                temp_progress.write_text(json.dumps(dict(contract=contract, next_index=end)))
                os.replace(temp_progress, progress_path)
            elapsed = time.time() - t0
            rate = (end - start_index) / max(elapsed, 1.0e-6)
            print(f"[{input_tsv.name}] processed={end}/{n} errors={errors} rate={rate:.2f} rows/s", flush=True)

    if mmap is None or embedding_dim is None:
        raise RuntimeError(f"no embeddings were produced for {input_tsv}")
    mmap.flush()
    _write_tsv(out_tsv, rows, merged_fieldnames)

    meta = {
        "input_tsv": str(input_tsv),
        "output_tsv": str(out_tsv),
        "output_npy": str(out_npy),
        "rows": n,
        "errors": errors,
        "audio_col": audio_col,
        "id_col": id_col,
        "model_name": pooler.model_name,
        "input_sample_rate": pooler.input_sample_rate,
        "model_sample_rate": pooler.model_sample_rate,
        "pooling": pooler.pooling,
        "hidden_size": pooler.hidden_size,
        "embedding_dim": embedding_dim,
        "dtype": str(np.dtype(dtype)),
        "max_seconds": max_seconds,
        "created_unix_time": time.time(),
    }
    out_json.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[done] {input_tsv} -> {out_npy} shape=({n},{embedding_dim}) errors={errors}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tsv", nargs="+", required=True, help="Input TSV(s) to augment.")
    parser.add_argument("--output-dir", required=True, help="Directory for .npy/.tsv/.json outputs.")
    parser.add_argument("--audio-col", default="src_audio", help="Source audio column in TSV.")
    parser.add_argument("--id-col", default="id", help="Optional id column, stored in metadata only.")
    parser.add_argument("--audio-root", default="", help="Root for relative audio paths.")
    parser.add_argument("--model-name", default="kyutai/mimi", help="Mimi model name/path.")
    parser.add_argument("--input-sample-rate", type=int, default=16000, help="Match training collator sample rate.")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pooling", choices=["mean", "std", "mean_std"], default="mean_std")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--suffix", default="mimi_source_acoustic_meanstd")
    parser.add_argument("--limit", type=int, default=0, help="Debug limit per TSV; 0 means all rows.")
    parser.add_argument("--max-seconds", type=float, default=30.0, help="Crop like training; <=0 disables crop.")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--allow-errors", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.resume and (args.overwrite or args.progress_every <= 0):
        raise ValueError('--resume requires positive --progress-every and no --overwrite')
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    torch.backends.cuda.matmul.allow_tf32 = True
    audio_root = Path(args.audio_root) if args.audio_root else None
    dtype = np.dtype(args.dtype)
    limit = int(args.limit) if int(args.limit) > 0 else None
    max_seconds = float(args.max_seconds) if float(args.max_seconds) > 0 else None

    pooler = MimiSourceAcousticPooler(
        model_name=args.model_name,
        input_sample_rate=args.input_sample_rate,
        device=args.device,
        pooling=args.pooling,
    )
    print(
        "[init]",
        {
            "model": args.model_name,
            "device": args.device,
            "input_sample_rate": pooler.input_sample_rate,
            "model_sample_rate": pooler.model_sample_rate,
            "hidden_size": pooler.hidden_size,
            "pooling": args.pooling,
        },
        flush=True,
    )

    for tsv_text in args.input_tsv:
        process_tsv(
            input_tsv=Path(tsv_text),
            output_dir=Path(args.output_dir),
            audio_col=args.audio_col,
            id_col=args.id_col or None,
            audio_root=audio_root,
            pooler=pooler,
            batch_size=int(args.batch_size),
            dtype=dtype,
            suffix=args.suffix,
            overwrite=bool(args.overwrite),
            limit=limit,
            allow_errors=bool(args.allow_errors),
            max_seconds=max_seconds,
            progress_every=int(args.progress_every),
            resume=bool(args.resume),
        )


if __name__ == "__main__":
    main()
