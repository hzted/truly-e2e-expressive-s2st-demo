#!/usr/bin/env python3
"""Optional Whisper-large-v3 ASR evaluation entry point.

This is deliberately disabled unless the caller passes --enabled. It is not a
VC-DUB construction dependency.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--id-col", default="sample_id")
    p.add_argument("--enabled", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.enabled:
        (out_dir / "WHISPER_ASR_DISABLED.txt").write_text(
            "Whisper ASR was not enabled. This is expected unless ASR-based evaluation is reported.\n",
            encoding="utf-8",
        )
        return
    df = pd.read_csv(args.manifest, sep="\t", dtype=str, keep_default_na=False, low_memory=False)
    if args.id_col not in df.columns:
        raise ValueError(f"Missing ID column: {args.id_col}")
    if args.dry_run:
        out = df[[args.id_col]].copy()
        out["whisper_transcript"] = "<dry-run transcript>"
        out.to_csv(out_dir / "whisper_transcripts.tsv", sep="\t", index=False)
        return
    raise RuntimeError(
        "Wire this wrapper to the exact Whisper-large-v3 ASR script used for the paper "
        "before enabling real ASR evaluation."
    )


if __name__ == "__main__":
    main()
