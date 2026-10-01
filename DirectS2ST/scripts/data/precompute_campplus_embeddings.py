#!/usr/bin/env python3
"""Precompute frozen CAMPPlus speaker embeddings for DirectS2ST TSVs.

Run once on the source audio (``--audio-col src_audio --embedding-column-prefix
campplus_source``) and once on the target audio (``--audio-col tgt_audio
--embedding-column-prefix campplus_target``); the latter supervises the depth
decoder's speaker-identity loss. Use the ``campplus_cn_en_common.pt`` weights
(iic/speech_campplus_sv_zh_en_16k-common_advanced) the model was trained with.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchaudio


def read_tsv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"TSV has no header: {path}")
        return [dict(row) for row in reader], list(reader.fieldnames)


def write_tsv(
    path: Path,
    rows: Sequence[Dict[str, str]],
    fieldnames: Sequence[str],
) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def write_json_atomic(path: Path, payload: Dict[str, object]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def resolve_audio_path(value: str, audio_root: Optional[Path]) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError("empty audio path")
    path = Path(text)
    if not path.is_absolute() and audio_root is not None:
        path = audio_root / path
    return path


def audio_to_fbank(
    path: Path,
    sample_rate: int,
    max_seconds: Optional[float],
) -> torch.Tensor:
    wav, source_rate = torchaudio.load(str(path))
    if wav.numel() == 0:
        raise ValueError(f"empty audio: {path}")
    wav = wav.to(torch.float32)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    if source_rate != sample_rate:
        wav = torchaudio.functional.resample(wav, source_rate, sample_rate)
    if max_seconds is not None:
        wav = wav[: int(round(max_seconds * sample_rate))]
    if wav.numel() < 400:
        wav = torch.nn.functional.pad(wav, (0, 400 - int(wav.numel())))
    features = torchaudio.compliance.kaldi.fbank(
        wav.unsqueeze(0),
        num_mel_bins=80,
        dither=0,
        sample_frequency=sample_rate,
    )
    return (features - features.mean(dim=0, keepdim=True)).contiguous()


class CampPlusExtractor:
    def __init__(
        self,
        model_root: Path,
        checkpoint: Path,
        embedding_size: int,
        device: str,
    ) -> None:
        root_text = str(model_root.resolve())
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        from modules.campplus.DTDNN import CAMPPlus

        self.device = torch.device(device)
        self.model = CAMPPlus(feat_dim=80, embedding_size=embedding_size)
        state = torch.load(str(checkpoint), map_location="cpu")
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False).to(self.device)
        self.embedding_size = int(embedding_size)
        self.checkpoint = checkpoint.resolve()

    @torch.inference_mode()
    def __call__(self, fbanks: Sequence[torch.Tensor]) -> torch.Tensor:
        # Do not pad variable-length utterances into one CAMPPlus batch. Its CAM
        # blocks pool over time before StatsPool, so padded frames can change the
        # embedding even when StatsPool receives valid lengths. This loop exactly
        # matches the online training path; I/O/fbank work remains parallelized.
        embeddings = torch.cat(
            [
                self.model(
                    features.unsqueeze(0).to(self.device, non_blocking=True)
                )
                for features in fbanks
            ],
            dim=0,
        )
        if embeddings.ndim != 2 or embeddings.size(1) != self.embedding_size:
            raise RuntimeError(
                f"Unexpected CAMPPlus output shape: {tuple(embeddings.shape)}"
            )
        return embeddings.float().cpu()


def process_tsv(
    input_tsv: Path,
    output_dir: Path,
    audio_col: str,
    audio_root: Optional[Path],
    extractor: CampPlusExtractor,
    sample_rate: int,
    max_seconds: Optional[float],
    batch_size: int,
    num_workers: int,
    dtype: np.dtype,
    suffix: str,
    embedding_column_prefix: str,
    limit: Optional[int],
    overwrite: bool,
    resume: bool,
    progress_every: int,
) -> None:
    rows, fieldnames = read_tsv(input_tsv)
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No rows to process: {input_tsv}")
    if audio_col not in fieldnames:
        raise KeyError(f"Missing audio column {audio_col!r}: {input_tsv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_tsv.stem
    output_npy = output_dir / f"{stem}_{suffix}.npy"
    output_tsv = output_dir / f"{stem}_{suffix}.tsv"
    output_meta = output_dir / f"{stem}_{suffix}.json"
    progress_path = output_dir / f"{stem}_{suffix}.progress.json"
    count = len(rows)
    shape = (count, extractor.embedding_size)

    start_index = 0
    if resume and output_npy.is_file() and progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if int(progress["rows"]) != count or int(progress["embedding_dim"]) != shape[1]:
            raise ValueError(f"Resume metadata does not match current input: {progress_path}")
        start_index = int(progress["next_index"])
        embeddings_mmap = np.load(output_npy, mmap_mode="r+")
        if tuple(embeddings_mmap.shape) != shape:
            raise ValueError(
                f"Resume array shape {embeddings_mmap.shape} != expected {shape}"
            )
        print(
            f"[campplus] resume {input_tsv.name} at {start_index}/{count}",
            flush=True,
        )
    else:
        existing = [
            path
            for path in (output_npy, output_tsv, output_meta, progress_path)
            if path.exists()
        ]
        if existing and not overwrite:
            raise FileExistsError(
                "Outputs exist; pass --resume or --overwrite: "
                + ", ".join(str(path) for path in existing)
            )
        embeddings_mmap = np.lib.format.open_memmap(
            output_npy,
            mode="w+",
            dtype=dtype,
            shape=shape,
        )
        embeddings_mmap[:] = 0
        embeddings_mmap.flush()
        write_json_atomic(
            progress_path,
            {
                "input_tsv": str(input_tsv),
                "rows": count,
                "embedding_dim": shape[1],
                "next_index": 0,
            },
        )

    if start_index >= count and output_tsv.is_file():
        print(f"[campplus] already complete: {output_tsv}", flush=True)
        return

    executor = ThreadPoolExecutor(max_workers=num_workers) if num_workers > 1 else None
    started = time.time()
    last_checkpoint = start_index

    try:
        for start in range(start_index, count, batch_size):
            end = min(count, start + batch_size)
            paths = [
                resolve_audio_path(rows[index].get(audio_col, ""), audio_root)
                for index in range(start, end)
            ]
            if executor is None:
                fbanks = [
                    audio_to_fbank(path, sample_rate, max_seconds)
                    for path in paths
                ]
            else:
                fbanks = list(
                    executor.map(
                        lambda path: audio_to_fbank(path, sample_rate, max_seconds),
                        paths,
                    )
                )
            batch_embeddings = extractor(fbanks).numpy().astype(dtype, copy=False)
            embeddings_mmap[start:end] = batch_embeddings

            should_checkpoint = end == count or end - last_checkpoint >= progress_every
            if should_checkpoint:
                embeddings_mmap.flush()
                write_json_atomic(
                    progress_path,
                    {
                        "input_tsv": str(input_tsv),
                        "rows": count,
                        "embedding_dim": shape[1],
                        "next_index": end,
                    },
                )
                elapsed = time.time() - started
                processed = end - start_index
                print(
                    f"[campplus] {input_tsv.name} {end}/{count} "
                    f"session_rate={processed / max(elapsed, 1e-6):.2f} rows/s",
                    flush=True,
                )
                last_checkpoint = end
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    embedding_path_key = f"{embedding_column_prefix}_embedding_path"
    embedding_index_key = f"{embedding_column_prefix}_embedding_index"
    embedding_dim_key = f"{embedding_column_prefix}_embedding_dim"
    embedding_model_key = f"{embedding_column_prefix}_embedding_model"
    embedding_status_key = f"{embedding_column_prefix}_embedding_status"
    for index, row in enumerate(rows):
        row[embedding_path_key] = str(output_npy)
        row[embedding_index_key] = str(index)
        row[embedding_dim_key] = str(extractor.embedding_size)
        row[embedding_model_key] = (
            f"CAMPPlus/{extractor.checkpoint.name}"
        )
        row[embedding_status_key] = "ok"

    extra_fields = [
        embedding_path_key,
        embedding_index_key,
        embedding_dim_key,
        embedding_model_key,
        embedding_status_key,
    ]
    merged_fields = list(fieldnames)
    for field in extra_fields:
        if field not in merged_fields:
            merged_fields.append(field)
    write_tsv(output_tsv, rows, merged_fields)
    write_json_atomic(
        output_meta,
        {
            "input_tsv": str(input_tsv),
            "output_tsv": str(output_tsv),
            "output_npy": str(output_npy),
            "rows": count,
            "audio_col": audio_col,
            "sample_rate": sample_rate,
            "max_seconds": max_seconds,
            "embedding_dim": extractor.embedding_size,
            "checkpoint": str(extractor.checkpoint),
            "embedding_column_prefix": embedding_column_prefix,
            "dtype": str(dtype),
            "created_unix_time": time.time(),
        },
    )
    print(f"[campplus] done: {output_tsv}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tsv", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audio-col", default="src_audio")
    parser.add_argument("--audio-root", default="")
    parser.add_argument(
        "--model-root",
        required=True,
        help="Local clone of https://github.com/Plachtaa/seed-vc (CAMPPlus model definition).",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="campplus_cn_en_common.pt from iic/speech_campplus_sv_zh_en_16k-common_advanced.",
    )
    parser.add_argument("--embedding-size", type=int, default=192)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--suffix", default="campplus192")
    parser.add_argument(
        "--embedding-column-prefix",
        default="campplus_source",
        help="Prefix for output TSV columns, e.g. campplus_source or campplus_target.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if not args.embedding_column_prefix.replace("_", "").isalnum():
        raise ValueError(
            "--embedding-column-prefix must contain only letters, digits, and underscores"
        )

    torch.backends.cuda.matmul.allow_tf32 = True
    extractor = CampPlusExtractor(
        model_root=Path(args.model_root),
        checkpoint=Path(args.checkpoint),
        embedding_size=args.embedding_size,
        device=args.device,
    )
    output_dir = Path(args.output_dir)
    audio_root = Path(args.audio_root) if args.audio_root else None
    max_seconds = args.max_seconds if args.max_seconds > 0 else None
    limit = args.limit if args.limit > 0 else None
    dtype = np.dtype(args.dtype)

    for input_value in args.input_tsv:
        process_tsv(
            input_tsv=Path(input_value),
            output_dir=output_dir,
            audio_col=args.audio_col,
            audio_root=audio_root,
            extractor=extractor,
            sample_rate=args.sample_rate,
            max_seconds=max_seconds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            dtype=dtype,
            suffix=args.suffix,
            embedding_column_prefix=args.embedding_column_prefix,
            limit=limit,
            overwrite=args.overwrite,
            resume=args.resume,
            progress_every=args.progress_every,
        )


if __name__ == "__main__":
    main()
