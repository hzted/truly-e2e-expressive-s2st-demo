#!/usr/bin/env python3
"""Score source/target utterances with DNSMOSPro for construction filtering.

The DNSMOSPro output parser is explicit by design: pass either --score-key for
JSON output or --score-regex for named text output. The script never uses "first
number in stdout" parsing, because that can accidentally read version strings or
progress counters instead of the MOS/naturalness score.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm


def nested_get(obj: Any, dotted_key: str) -> Any:
    cur = obj
    for part in dotted_key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def parse_json_score(text: str, key: str) -> float | None:
    try:
        data: Any = json.loads(text)
    except Exception:
        return None
    if isinstance(data, list):
        for item in data:
            val = nested_get(item, key)
            if val is not None:
                return float(val)
    val = nested_get(data, key)
    if val is not None:
        return float(val)
    return None


def parse_score(text: str, score_key: str, score_regex: str) -> float:
    if score_key:
        parsed = parse_json_score(text, score_key)
        if parsed is not None:
            return parsed
    if score_regex:
        match = re.search(score_regex, text, flags=re.MULTILINE)
        if match:
            group = match.group(1) if match.groups() else match.group(0)
            return float(group)
    raise ValueError(
        "Could not parse DNSMOSPro score. Provide a JSON --score-key or a named "
        "--score-regex that captures the intended score field."
    )


def score_audio(audio: str, command_template: str, timeout_sec: float, score_key: str, score_regex: str) -> float:
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
    return parse_score(proc.stdout + "\n" + proc.stderr, score_key=score_key, score_regex=score_regex)


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
    parser.add_argument("--combine", choices=["mean", "min"], required=True)
    parser.add_argument("--dnsmospro-cmd", required=True, help="Command template containing {audio}.")
    parser.add_argument("--score-key", default="", help="JSON key or dotted key, e.g. nat or scores.nat.")
    parser.add_argument("--score-regex", default="", help="Regex for named text output; first capture group is used.")
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.score_key and not args.score_regex:
        raise ValueError("Pass --score-key or --score-regex; implicit first-number parsing is disabled.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_tsv, sep="\t", low_memory=False)
    required = [args.id_col, args.src_audio_col, args.tgt_audio_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[args.id_col].duplicated().any():
        dupes = df.loc[df[args.id_col].duplicated(), args.id_col].head(10).tolist()
        raise ValueError(f"Duplicate sample IDs in input manifest: {dupes}")

    audio_cache: dict[str, float] = {}
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="DNSMOSPro pairs"):
        src_audio = str(row[args.src_audio_col])
        tgt_audio = str(row[args.tgt_audio_col])
        for audio in (src_audio, tgt_audio):
            if audio not in audio_cache:
                audio_cache[audio] = score_audio(
                    audio,
                    args.dnsmospro_cmd,
                    args.timeout_sec,
                    score_key=args.score_key,
                    score_regex=args.score_regex,
                )
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
        "score_key": args.score_key,
        "score_regex": args.score_regex,
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
