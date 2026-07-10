#!/usr/bin/env python3
"""Select a clean-pool manifest from DNSMOSPro pair-level quality scores.

The selector also writes per-example keep/drop decisions so construction-time
quality filtering can be audited without rerunning DNSMOSPro.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-tsv", required=True)
    parser.add_argument("--score-tsv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--output-id-col", default="sample_id")
    parser.add_argument("--score-col", default="combined_dnsmospro")
    parser.add_argument("--cutoff", type=float, default=None)
    parser.add_argument("--target-keep-pairs", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest_tsv, sep="\t", low_memory=False)
    scores = pd.read_csv(args.score_tsv, sep="\t", low_memory=False)
    if args.id_col not in manifest.columns or args.id_col not in scores.columns:
        raise ValueError(f"Both manifest and score TSVs must contain id column: {args.id_col}")
    if args.score_col not in scores.columns:
        raise ValueError(f"Score TSV missing score column: {args.score_col}")

    score_cols = [c for c in ["src_dnsmospro", "tgt_dnsmospro", "combined_dnsmospro"] if c in scores.columns]
    if args.score_col not in score_cols:
        score_cols.append(args.score_col)
    score_view = scores[[args.id_col] + score_cols].copy()
    merged = manifest.merge(score_view, on=args.id_col, how="inner")
    if args.cutoff is not None:
        keep = merged[merged[args.score_col] >= args.cutoff].copy()
        selection_rule = f"{args.score_col} >= {args.cutoff}"
        drop_reason = "below_cutoff"
    elif args.target_keep_pairs > 0:
        keep = merged.sort_values(args.score_col, ascending=False).head(args.target_keep_pairs).copy()
        selection_rule = f"top_{args.target_keep_pairs}_by_{args.score_col}"
        drop_reason = "below_top_n_boundary"
    else:
        raise ValueError("Provide either --cutoff or --target-keep-pairs.")

    dropped = merged[~merged[args.id_col].isin(set(keep[args.id_col]))].copy()
    kept_ids = set(keep[args.id_col].astype(str))
    decisions = merged[[args.id_col] + score_cols].copy()
    decisions["selected"] = decisions[args.id_col].astype(str).map(lambda x: x in kept_ids)
    decisions["drop_reason"] = decisions["selected"].map(lambda x: "" if x else drop_reason)
    if args.id_col != args.output_id_col:
        decisions = decisions.rename(columns={args.id_col: args.output_id_col})

    keep_path = out_dir / "dnsmospro_filtered_manifest.tsv"
    drop_path = out_dir / "dnsmospro_dropped_manifest.tsv"
    decision_path = out_dir / "dnsmospro_quality_pairs.tsv"
    keep.to_csv(keep_path, sep="\t", index=False)
    dropped.to_csv(drop_path, sep="\t", index=False)
    decisions.to_csv(decision_path, sep="\t", index=False)

    cutoff = args.cutoff
    if cutoff is None and len(keep):
        cutoff = float(keep[args.score_col].min())
    summary = {
        "manifest_tsv": args.manifest_tsv,
        "score_tsv": args.score_tsv,
        "score_col": args.score_col,
        "selection_rule": selection_rule,
        "cutoff": cutoff,
        "input_pairs": int(len(merged)),
        "kept_pairs": int(len(keep)),
        "dropped_pairs": int(len(dropped)),
        "output_manifest": str(keep_path),
        "decision_tsv": str(decision_path),
    }
    (out_dir / "dnsmospro_quality_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote kept manifest: {keep_path}")
    print(f"Wrote dropped manifest: {drop_path}")
    print(f"Wrote decisions: {decision_path}")


if __name__ == "__main__":
    main()
