#!/usr/bin/env python3
"""Aggregate VC-DUB paper evaluation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import stats


AGGREGATE_KEYS = [
    "blaser2_qe_audio_mean",
    "blaser2_ref_mean",
    "autopcp_mean",
    "dc_0p2_compliance_mean",
    "dc_0p4_compliance_mean",
    "speech_rate_syllable_spearman_mean",
    "pause_wmean_duration_score_mean",
    "vsim_mean",
    "dnsmospro_nat_mean",
]

PAPER_TABLE_FIELDS = [
    ("BLASER2_QE", "blaser2_qe_audio_mean"),
    ("BLASER2_ref", "blaser2_ref_mean"),
    ("A_PCP", "autopcp_mean"),
    ("SLC_0p2", "dc_0p2_compliance_mean"),
    ("SLC_0p4", "dc_0p4_compliance_mean"),
    ("SpeechRate", "speech_rate_syllable_spearman_mean"),
    ("Pause", "pause_wmean_duration_score_mean"),
    ("Vsim", "vsim_mean"),
    ("DNSMOSPro_Nat", "dnsmospro_nat_mean"),
]

PER_EXAMPLE_FILES = [
    "blaser2_audio/blaser2_per_example.tsv",
    "apcp/apcp_per_example.tsv",
    "isochrony/isochrony_per_example.tsv",
    "vsim/vsim_per_example.tsv",
    "dnsmospro/dnsmospro_per_example.tsv",
]

SUMMARY_FILES = [
    "blaser2_audio/blaser2_audio_summary.json",
    "apcp/summary.json",
    "isochrony/summary.json",
    "vsim/summary.json",
    "dnsmospro/dnsmospro_nat_summary.json",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_key(key: str) -> str:
    aliases = {
        "speech_rate_syllable_pearson": "speech_rate_syllable_pearson",
        "speech_rate_syllable_spearman": "speech_rate_syllable_spearman",
    }
    return aliases.get(key, key)


def add_summary_metrics(merged: dict[str, Any], data: dict[str, Any]) -> None:
    for key, value in data.items():
        nkey = normalize_key(key)
        if nkey.endswith("_mean") or nkey in {
            "blaser2_qe_audio_mean",
            "blaser2_ref_mean",
            "autopcp_mean",
            "vsim_mean",
            "dnsmospro_nat_mean",
            "dc_0p2_compliance",
            "dc_0p4_compliance",
            "sc_0p2_compliance",
            "sc_0p4_compliance",
            "speech_rate_syllable_spearman",
            "pause_wmean_duration_score",
        }:
            out_key = nkey if nkey.endswith("_mean") else f"{nkey}_mean"
            merged[out_key] = value


def merge_per_example(root: Path, id_col: str, manifest: Path | None) -> pd.DataFrame:
    if manifest is not None and manifest.is_file():
        base = pd.read_csv(manifest, sep="\t", dtype=str, keep_default_na=False, low_memory=False)
        if id_col in base.columns:
            out = base[[id_col]].copy()
        else:
            out = pd.DataFrame(columns=[id_col])
    else:
        out = pd.DataFrame(columns=[id_col])

    for rel in PER_EXAMPLE_FILES:
        path = root / rel
        if not path.is_file():
            continue
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, low_memory=False)
        if id_col not in df.columns:
            continue
        if out.empty:
            out = df.copy()
        else:
            out = out.merge(df, on=id_col, how="left")
    return out


def scalar_ci95_margin(values: pd.Series) -> float | None:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if len(vals) <= 1:
        return None
    mean = float(vals.mean())
    sem = float(vals.sem())
    low, high = stats.t.interval(0.95, df=len(vals) - 1, loc=mean, scale=sem)
    return float(max(mean - low, high - mean))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-root", required=True)
    p.add_argument("--manifest", default="")
    p.add_argument("--id-col", default="sample_id")
    p.add_argument("--out-json", default="")
    p.add_argument("--out-tsv", default="")
    p.add_argument("--out-paper-json", default="")
    p.add_argument("--out-paper-tsv", default="")
    p.add_argument("--out-per-example", default="")
    p.add_argument(
        "--uncertainty",
        choices=["none", "std", "sem", "ci95"],
        default="none",
        help="Optional plus/minus convention for per-example scalar metrics.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.eval_root)
    manifest = Path(args.manifest) if args.manifest else None
    per_example = merge_per_example(root, args.id_col, manifest)

    aggregate: dict[str, Any] = {
        "eval_root": str(root),
        "manifest": args.manifest,
        "id_col": args.id_col,
        "uncertainty": args.uncertainty,
    }
    for rel in SUMMARY_FILES:
        data = read_json(root / rel)
        aggregate[f"{Path(rel).stem}_summary_path"] = str(root / rel)
        add_summary_metrics(aggregate, data)

    if not per_example.empty:
        for col in per_example.columns:
            if col == args.id_col:
                continue
            vals = pd.to_numeric(per_example[col], errors="coerce").dropna()
            if len(vals) == 0:
                continue
            aggregate.setdefault(f"{col}_mean", float(vals.mean()))
            if args.uncertainty == "std" and len(vals) > 1:
                aggregate[f"{col}_pm"] = float(vals.std(ddof=1))
            elif args.uncertainty == "sem" and len(vals) > 1:
                aggregate[f"{col}_pm"] = float(vals.sem())
            elif args.uncertainty == "ci95":
                aggregate[f"{col}_pm"] = scalar_ci95_margin(vals)

    for key in AGGREGATE_KEYS:
        aggregate.setdefault(key, None)

    paper_table: dict[str, Any] = {
        "eval_root": str(root),
        "manifest": args.manifest,
        "id_col": args.id_col,
        "uncertainty": args.uncertainty,
    }
    for paper_key, source_key in PAPER_TABLE_FIELDS:
        paper_table[paper_key] = aggregate.get(source_key)
        pm_key = source_key.replace("_mean", "_pm")
        if pm_key in aggregate:
            paper_table[f"{paper_key}_pm"] = aggregate[pm_key]

    out_json = Path(args.out_json) if args.out_json else root / "aggregate_metrics.json"
    out_tsv = Path(args.out_tsv) if args.out_tsv else root / "aggregate_metrics.tsv"
    out_paper_json = Path(args.out_paper_json) if args.out_paper_json else root / "paper_table_metrics.json"
    out_paper_tsv = Path(args.out_paper_tsv) if args.out_paper_tsv else root / "paper_table_metrics.tsv"
    out_per = Path(args.out_per_example) if args.out_per_example else root / "per-example_metrics.tsv"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pd.DataFrame([aggregate]).to_csv(out_tsv, sep="\t", index=False)
    out_paper_json.write_text(json.dumps(paper_table, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pd.DataFrame([paper_table]).to_csv(out_paper_tsv, sep="\t", index=False)
    per_example.to_csv(out_per, sep="\t", index=False)
    print(f"Wrote per-example metrics: {out_per}")
    print(f"Wrote aggregate JSON: {out_json}")
    print(f"Wrote aggregate TSV: {out_tsv}")
    print(f"Wrote paper-table JSON: {out_paper_json}")
    print(f"Wrote paper-table TSV: {out_paper_tsv}")


if __name__ == "__main__":
    main()
