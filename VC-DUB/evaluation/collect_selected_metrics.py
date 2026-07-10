#!/usr/bin/env python3
"""Collect only the VC-DUB output-evaluation metrics reported in the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


SELECTED_KEYS = [
    "blaser2_qe_audio_mean",
    "blaser2_ref_mean",
    "autopcp_mean",
    "vsim_mean",
    "sc_0p2_compliance",
    "sc_0p4_compliance",
    "speech_rate_syllable_pearson",
    "pause_wmean_duration_score",
    "dnsmospro_nat_mean",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-tsv", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.eval_root)

    merged: dict[str, Any] = {}
    sources = {
        "blaser2": root / "blaser2_audio" / "blaser2_audio_summary.json",
        "stopes": root / "stopes_metrics" / "summary.json",
        "dnsmospro_nat": root / "dnsmospro_nat" / "dnsmospro_nat_summary.json",
    }
    for name, path in sources.items():
        data = read_json(path)
        merged[f"{name}_summary_path"] = str(path)
        for key, value in data.items():
            if key in SELECTED_KEYS:
                merged[key] = value

    # Make missing fields explicit instead of silently dropping them.
    for key in SELECTED_KEYS:
        merged.setdefault(key, None)

    out_json = Path(args.out_json) if args.out_json else root / "selected_metrics_summary.json"
    out_tsv = Path(args.out_tsv) if args.out_tsv else root / "selected_metrics_summary.tsv"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pd.DataFrame([merged]).to_csv(out_tsv, sep="\t", index=False)
    print(f"Wrote selected summary JSON: {out_json}")
    print(f"Wrote selected summary TSV: {out_tsv}")


if __name__ == "__main__":
    main()
