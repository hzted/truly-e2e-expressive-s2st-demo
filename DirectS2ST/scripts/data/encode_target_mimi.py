#!/usr/bin/env python3
"""Encode target speech into Mimi codes for DirectS2ST training TSVs.

Adds, for every row of ``--input-tsv``:

  mimi_audio_path       the encoded audio path
  mimi_num_quantizers   number of RVQ levels kept (32; the model predicts the first 16)
  mimi_codes_shape      "1x<levels>x<frames>"
  mimi_num_frames       number of 12.5 Hz frames
  mimi_codes            space-separated codes, flattened in [1, levels, frames] order
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import torch
from transformers import AutoFeatureExtractor, MimiModel


def serialize_codes(codes: torch.Tensor) -> str:
    return " ".join(str(int(x)) for x in codes.detach().cpu().reshape(-1).tolist())


def load_audio(audio_path: str, target_sr: int) -> torch.Tensor:
    wav, _ = librosa.load(audio_path, sr=target_sr, mono=True)
    return torch.tensor(wav, dtype=torch.float32)


def resolve_audio_path(audio_path: str, audio_root: Optional[str]) -> str:
    path = Path(str(audio_path))
    if path.is_absolute() or not audio_root:
        return str(path)
    return str(Path(audio_root) / path)


@torch.no_grad()
def encode_mimi(audio_path: str, feature_extractor, model, device: str, num_quantizers: int) -> Dict[str, str]:
    wav = load_audio(audio_path, int(feature_extractor.sampling_rate))
    inputs = feature_extractor(
        raw_audio=wav.numpy(),
        sampling_rate=int(feature_extractor.sampling_rate),
        return_tensors="pt",
    )
    codes = model.encode(
        inputs["input_values"].to(device),
        return_dict=True,
        num_quantizers=int(num_quantizers),
    ).audio_codes
    return {
        "mimi_audio_path": audio_path,
        "mimi_num_quantizers": str(int(num_quantizers)),
        "mimi_codes_shape": "x".join(str(int(x)) for x in codes.shape),
        "mimi_num_frames": str(int(codes.shape[-1])),
        "mimi_codes": serialize_codes(codes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-tsv", required=True)
    parser.add_argument("--output-tsv", required=True)
    parser.add_argument("--audio-col", default="tgt_audio", help="Target audio column.")
    parser.add_argument("--audio-root", default=None, help="Root for relative audio paths.")
    parser.add_argument("--model-id", default="kyutai/mimi")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-quantizers", type=int, default=32)
    parser.add_argument("--skip-bad", action="store_true", help="Drop rows whose audio fails to encode.")
    args = parser.parse_args()

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_id)
    model = MimiModel.from_pretrained(args.model_id).to(args.device).eval()

    with open(args.input_tsv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames: List[str] = list(reader.fieldnames or [])
        rows = list(reader)
    new_fields = ["mimi_audio_path", "mimi_num_quantizers", "mimi_codes_shape", "mimi_num_frames", "mimi_codes"]
    out_fields = fieldnames + [f for f in new_fields if f not in fieldnames]

    kept, failed = 0, 0
    Path(args.output_tsv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_tsv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for i, row in enumerate(rows):
            audio_path = resolve_audio_path(row[args.audio_col], args.audio_root)
            try:
                row.update(encode_mimi(audio_path, feature_extractor, model, args.device, args.num_quantizers))
            except Exception as exc:  # noqa: BLE001
                if not args.skip_bad:
                    raise
                failed += 1
                print(f"[mimi] skip {audio_path}: {exc}", flush=True)
                continue
            writer.writerow(row)
            kept += 1
            if (i + 1) % 500 == 0:
                print(f"[mimi] {i + 1}/{len(rows)}", flush=True)
    print(f"[mimi] wrote {kept} rows ({failed} skipped) to {args.output_tsv}")


if __name__ == "__main__":
    main()
