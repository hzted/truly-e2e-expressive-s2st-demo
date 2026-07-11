#!/usr/bin/env python3
"""Score generated audio with DNSMOSPro naturalness/MOS.

The script is intentionally command-template based so it can wrap different
DNSMOSPro checkouts. The command must contain ``{audio}`` and print either JSON
or text containing a numeric naturalness/MOS value.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd

from _wrapper_utils import write_dry_outputs


def parse_json_score(text: str, key: str) -> float | None:
    try:
        data: Any = json.loads(text)
    except Exception:
        return None
    if isinstance(data, dict):
        value = data.get(key)
        if value is not None:
            return float(value)
        for nested in data.values():
            if isinstance(nested, dict) and key in nested:
                return float(nested[key])
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and key in item:
                return float(item[key])
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
        "Could not parse DNSMOSPro/NAT score. Provide --score-key for JSON output "
        "or --score-regex for named text output; first-number fallback is disabled."
    )


def score_audio(audio: str, command_template: str, timeout_sec: float, score_key: str, score_regex: str) -> tuple[float, str]:
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
    raw = (proc.stdout + "\n" + proc.stderr).strip()
    return parse_score(raw, score_key=score_key, score_regex=score_regex), raw


def ci95_margin(vals: pd.Series) -> float | None:
    vals = pd.to_numeric(vals, errors="coerce").dropna()
    n = len(vals)
    if n <= 1:
        return None
    # Normal approximation is sufficient for this reporting helper; downstream
    # tables can recompute t-intervals if scipy is available.
    return float(1.96 * vals.std(ddof=1) / math.sqrt(n))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Evaluation manifest TSV.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--id-col", default="sample_id")
    parser.add_argument("--audio-col", default="hypo_audio")
    parser.add_argument("--status-col", default="status")
    parser.add_argument("--ok-status", default="ok")
    parser.add_argument("--keep-non-ok", action="store_true")
    parser.add_argument("--dnsmospro-cmd", required=True, help="Command template containing {audio}.")
    parser.add_argument("--score-key", default="", help="Optional JSON key to parse, e.g. nat or mos.")
    parser.add_argument("--score-regex", default="", help="Optional regex; first capture group is used when present.")
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        write_dry_outputs(
            args.manifest,
            out_dir,
            args.id_col,
            {"dnsmospro_nat": 3.5},
            "dnsmospro_nat_summary.json",
            "dnsmospro_per_example.tsv",
        )
        return
    if not args.score_key and not args.score_regex:
        raise ValueError("Real DNSMOSPro evaluation requires --score-key or --score-regex.")

    df = pd.read_csv(args.manifest, sep="\t", low_memory=False)
    missing = [c for c in [args.id_col, args.audio_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    filtered = df.copy()
    if args.status_col in filtered.columns and not args.keep_non_ok:
        filtered = filtered[filtered[args.status_col].astype(str) == args.ok_status].copy()
    if args.max_rows > 0:
        filtered = filtered.head(args.max_rows).copy()

    rows = []
    cache: dict[str, float] = {}
    for _, row in filtered.iterrows():
        sample_id = str(row[args.id_col])
        audio = str(row[args.audio_col])
        start = time.time()
        try:
            if audio not in cache:
                score, raw = score_audio(
                    audio,
                    command_template=args.dnsmospro_cmd,
                    timeout_sec=args.timeout_sec,
                    score_key=args.score_key,
                    score_regex=args.score_regex,
                )
                cache[audio] = score
            else:
                score = cache[audio]
                raw = ""
            rows.append(
                {
                    args.id_col: sample_id,
                    args.audio_col: audio,
                    "dnsmospro_nat": score,
                    "status": "ok",
                    "elapsed_sec": time.time() - start,
                    "raw_output": raw,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    args.id_col: sample_id,
                    args.audio_col: audio,
                    "dnsmospro_nat": None,
                    "status": "error",
                    "elapsed_sec": time.time() - start,
                    "raw_output": repr(exc),
                }
            )

    scores = pd.DataFrame(rows)
    score_path = out_dir / "dnsmospro_nat_scores.tsv"
    per_example_path = out_dir / "dnsmospro_per_example.tsv"
    summary_path = out_dir / "dnsmospro_nat_summary.json"
    scores.to_csv(score_path, sep="\t", index=False)
    scores.to_csv(per_example_path, sep="\t", index=False)

    vals = pd.to_numeric(scores["dnsmospro_nat"], errors="coerce").dropna()
    summary = {
        "manifest": args.manifest,
        "score_tsv": str(score_path),
        "audio_col": args.audio_col,
        "num_rows": int(len(scores)),
        "num_ok": int((scores["status"] == "ok").sum()) if len(scores) else 0,
        "num_error": int((scores["status"] == "error").sum()) if len(scores) else 0,
        "dnsmospro_nat_mean": None if len(vals) == 0 else float(vals.mean()),
        "dnsmospro_nat_std": None if len(vals) <= 1 else float(vals.std(ddof=1)),
        "dnsmospro_nat_ci95": ci95_margin(vals),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote scores: {score_path}")
    print(f"Wrote summary: {summary_path}")
    if summary["num_rows"] > 0 and summary["num_ok"] == 0:
        raise RuntimeError(
            "DNSMOSPro produced zero valid scores. Check --dnsmospro-cmd and "
            "--score-key/--score-regex; refusing to report a successful run."
        )


if __name__ == "__main__":
    main()
