#!/usr/bin/env python3
"""Score source/target utterances with DNSMOSPro and write pair-level quality scores.

This wrapper intentionally uses a command template so it can work with different
DNSMOSPro checkouts. The command must contain ``{audio}`` and print at least one
numeric MOS-like score; the first parsed number is used.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm


FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")


def parse_score(text: str) -> float:
    match = FLOAT_RE.search(text)
    if not match:
        raise ValueError(f"No numeric DNSMOSPro score found in output: {text[:200]}")
    return float(match.group(0))


def score_audio(audio: str, command_template: str, timeout_sec: float) -> float:
    cmd = command_template.format(audio=shlex.quote(audio))
    proc = subprocess.run(
        cmd,
        shell=True,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_sec,
    )
    return parse_score(proc.stdout + "\n" + proc.stderr)


def combine_scores(src_score: float, tgt_score: float, mode: str) -> float:
    if mode == "mean":
        return (src_score + tgt_score) / 2.0
    if mode == "min":
        return min(src_score, tgt_score)
    raise ValueError(f"Unsupported combine mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tsv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--id-col", default="sample_id")
    parser.add_argument("--output-id-col", default="sample_id")
    parser.add_argument("--src-audio-col", default="pre_src")
    parser.add_argument("--tgt-audio-col", default="pre_tgt")
    parser.add_argument("--combine", choices=["mean", "min"], default="mean")
    parser.add_argument("--dnsmospro-cmd", required=True, help="Command template containing {audio}.")
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_tsv, sep="\t", low_memory=False)
    required = [args.id_col, args.src_audio_col, args.tgt_audio_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    audio_cache: dict[str, float] = {}
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="DNSMOSPro pairs"):
        src_audio = str(row[args.src_audio_col])
        tgt_audio = str(row[args.tgt_audio_col])
        for audio in (src_audio, tgt_audio):
            if audio not in audio_cache:
                audio_cache[audio] = score_audio(audio, args.dnsmospro_cmd, args.timeout_sec)
        src_score = audio_cache[src_audio]
        tgt_score = audio_cache[tgt_audio]
        rows.append(
            {
                args.output_id_col: row[args.id_col],
                args.src_audio_col: src_audio,
                args.tgt_audio_col: tgt_audio,
                "src_dnsmospro": src_score,
                "tgt_dnsmospro": tgt_score,
                "combined_dnsmospro": combine_scores(src_score, tgt_score, args.combine),
            }
        )

    scores = pd.DataFrame(rows)
    score_tsv = out_dir / "dnsmospro_quality_pairs.tsv"
    scores.to_csv(score_tsv, sep="\t", index=False)

    summary = {
        "input_tsv": args.input_tsv,
        "score_tsv": str(score_tsv),
        "num_pairs": int(len(scores)),
        "num_unique_audio": int(len(audio_cache)),
        "combine": args.combine,
        "combined_dnsmospro_mean": float(scores["combined_dnsmospro"].mean()) if len(scores) else None,
        "combined_dnsmospro_median": float(scores["combined_dnsmospro"].median()) if len(scores) else None,
    }
    (out_dir / "dnsmospro_quality_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote scores: {score_tsv}")
    print(f"Wrote summary: {out_dir / 'dnsmospro_quality_summary.json'}")


if __name__ == "__main__":
    main()
