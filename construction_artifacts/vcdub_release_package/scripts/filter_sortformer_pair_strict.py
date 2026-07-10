#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


def truthy(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Filter Sortformer pair results using strict single-speaker logic and write a final pass TSV."
    )
    ap.add_argument("--pair-results", required=True)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--id-col", default="id")
    ap.add_argument(
        "--pass-column",
        default="both_strict_single",
        help="Boolean column to use for pass/fail. Defaults to both_strict_single.",
    )
    ap.add_argument(
        "--manifest-only",
        action="store_true",
        help="If set, pass TSV drops Sortformer diagnostic columns and keeps only the original manifest-style columns.",
    )
    args = ap.parse_args()

    pair_path = Path(args.pair_results)
    if not pair_path.is_file():
        raise FileNotFoundError(pair_path)

    out_dir = Path(args.out_dir) if args.out_dir else pair_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pair_path, sep="\t", low_memory=False)
    if args.pass_column not in df.columns:
        raise ValueError(f"Missing pass column: {args.pass_column}")
    if args.id_col not in df.columns:
        raise ValueError(f"Missing id column: {args.id_col}")

    pass_mask = df[args.pass_column].map(truthy)
    fail_mask = ~pass_mask

    pass_df = df.loc[pass_mask].copy()
    fail_df = df.loc[fail_mask].copy()

    manifest_cols = [
        c for c in [
            args.id_col,
            "src",
            "tgt",
            "out",
            "pre_src",
            "pre_tgt",
            "status",
            "rtf",
            "total_elapsed_s",
            "demucs_preprocess",
            "demucs_on",
            "demucs_model",
            "clearvoice_denoise",
            "clearvoice_on",
            "clearvoice_model",
            "source_clearvoice_ok",
            "target_clearvoice_ok",
            "source_demucs_ok",
            "target_demucs_ok",
            "preprocess_gate_ok",
            "pre_src_checked",
            "src_pred_lang",
            "src_pred_score",
            "src_match",
            "src_status",
            "src_error",
            "pre_tgt_checked",
            "tgt_pred_lang",
            "tgt_pred_score",
            "tgt_match",
            "tgt_status",
            "tgt_error",
            "both_match",
            "either_mismatch",
        ] if c in df.columns
    ]

    pass_tsv = out_dir / "sortformer_pair_pass_strict.tsv"
    fail_tsv = out_dir / "sortformer_pair_fail_strict.tsv"
    pass_ids = out_dir / "sortformer_pair_pass_strict_ids.txt"
    fail_ids = out_dir / "sortformer_pair_fail_strict_ids.txt"
    summary_json = out_dir / "sortformer_pair_strict_summary.json"
    summary_tsv = out_dir / "sortformer_pair_strict_summary.tsv"

    write_pass_df = pass_df[manifest_cols].copy() if args.manifest_only and manifest_cols else pass_df
    write_fail_df = fail_df[manifest_cols].copy() if args.manifest_only and manifest_cols else fail_df

    write_pass_df.to_csv(pass_tsv, sep="\t", index=False)
    write_fail_df.to_csv(fail_tsv, sep="\t", index=False)
    pass_ids.write_text("\n".join(pass_df[args.id_col].astype(str)) + ("\n" if len(pass_df) else ""), encoding="utf-8")
    fail_ids.write_text("\n".join(fail_df[args.id_col].astype(str)) + ("\n" if len(fail_df) else ""), encoding="utf-8")

    summary = {
        "pair_results": str(pair_path),
        "pass_column": args.pass_column,
        "manifest_only": bool(args.manifest_only),
        "n_pairs": int(len(df)),
        "pass_pairs": int(len(pass_df)),
        "fail_pairs": int(len(fail_df)),
        "pass_ratio": float(len(pass_df) / len(df)) if len(df) else 0.0,
        "files": {
            "pass_tsv": str(pass_tsv),
            "fail_tsv": str(fail_tsv),
            "pass_ids": str(pass_ids),
            "fail_ids": str(fail_ids),
        },
    }

    pd.DataFrame([summary]).to_csv(summary_tsv, sep="\t", index=False)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""
source {USER_HOME}/.bashrc
conda activate stopes_eval_a100

python -u {EXPRESSIVE_S2ST_ROOT}/verify_scripts/filter_sortformer_pair_strict.py \
  --pair-results {EXPRESSIVE_S2ST_ROOT}/es_en/seedvc_outputs_netflix_denoised/mms_lid_preprocessed_filter/sortformer_pair_filter/sortformer_pair_results.tsv \
  --manifest-only

"""
