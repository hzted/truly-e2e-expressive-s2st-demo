#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CORE_COLS = ["path", "sentence", "translation", "src_audio", "tgt_audio"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a VC-DUB ASR metadata TSV while preserving ASR-only columns "
            "in *_ar.tsv outputs."
        )
    )
    parser.add_argument("--input-tsv", "--input_tsv", type=str, required=True)
    parser.add_argument("--out-dir", "--out_dir", type=str, required=True)
    parser.add_argument(
        "--join-manifest-tsv",
        type=str,
        default="",
        help=(
            "Optional original manifest to join by id/path. Useful when the ASR meta "
            "does not carry pre_tgt/pre_src but split outputs need those paths."
        ),
    )
    parser.add_argument("--input-id-col", type=str, default="path")
    parser.add_argument("--join-id-col", type=str, default="id")
    parser.add_argument("--source-text-col", type=str, default="sentence")
    parser.add_argument("--target-text-col", type=str, default="translation")
    parser.add_argument("--source-audio-col", type=str, default="src_audio")
    parser.add_argument("--target-audio-col", type=str, default="tgt_audio")
    parser.add_argument(
        "--dev-test-ratio",
        type=float,
        default=0.15,
        help="Fraction of rows reserved for dev+test together.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=500,
        help="Fixed number of rows sampled for test.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--drop-duplicates",
        action="store_true",
        help="Drop duplicate path rows before splitting.",
    )
    parser.add_argument(
        "--drop-missing-audio",
        action="store_true",
        help="Drop rows whose src_audio or tgt_audio file does not exist.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def write_audio_list(path: Path, values: pd.Series) -> None:
    text = "\n".join(values.astype(str).tolist())
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_tsv = Path(args.input_tsv)
    out_dir = Path(args.out_dir)

    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {out_dir}. Pass --overwrite to replace split files."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_tsv, sep="\t", dtype=str, keep_default_na=False)
    require_columns(df, [args.input_id_col, args.source_text_col, args.target_text_col])

    if args.join_manifest_tsv:
        join_df = pd.read_csv(
            args.join_manifest_tsv,
            sep="\t",
            dtype=str,
            keep_default_na=False,
        )
        require_columns(join_df, [args.join_id_col])
        join_cols = [c for c in join_df.columns if c != args.join_id_col]
        rename_map = {c: f"manifest_{c}" for c in join_cols if c in df.columns}
        join_df = join_df.rename(columns=rename_map)
        df = df.merge(
            join_df,
            left_on=args.input_id_col,
            right_on=args.join_id_col,
            how="left",
            suffixes=("", "_joined"),
        )

    require_columns(df, [args.source_audio_col, args.target_audio_col])

    # Canonical raw split format used by the downstream dubbing builder.
    core_df = pd.DataFrame(
        {
            "path": df[args.input_id_col],
            "sentence": df[args.source_text_col],
            "translation": df[args.target_text_col],
            "src_audio": df[args.source_audio_col],
            "tgt_audio": df[args.target_audio_col],
        }
    )

    extra_cols = [c for c in df.columns if c not in core_df.columns]
    df = pd.concat([core_df, df[extra_cols]], axis=1)
    require_columns(df, CORE_COLS)

    before_rows = len(df)
    df = df.dropna(subset=CORE_COLS).reset_index(drop=True)

    if args.drop_duplicates:
        df = df.drop_duplicates(subset=["path"]).reset_index(drop=True)

    if args.drop_missing_audio:
        mask = df["src_audio"].map(lambda p: Path(p).exists()) & df["tgt_audio"].map(
            lambda p: Path(p).exists()
        )
        df = df[mask].reset_index(drop=True)

    n_rows = len(df)
    holdout_rows = round(n_rows * args.dev_test_ratio)
    if args.test_size <= 0:
        raise ValueError("--test-size must be positive")
    if holdout_rows <= args.test_size:
        raise ValueError(
            f"dev+test rows ({holdout_rows}) must be larger than test rows ({args.test_size})"
        )

    test_rows = args.test_size
    dev_rows = holdout_rows - test_rows
    train_rows = n_rows - holdout_rows

    shuffled = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    test_df = shuffled.iloc[:test_rows].reset_index(drop=True)
    dev_df = shuffled.iloc[test_rows : test_rows + dev_rows].reset_index(drop=True)
    train_df = shuffled.iloc[test_rows + dev_rows :].reset_index(drop=True)

    splits = {"train": train_df, "dev": dev_df, "test": test_df}

    df.to_csv(out_dir / "all_asr_filtered.tsv", sep="\t", index=False)
    df[CORE_COLS].to_csv(out_dir / "all.tsv", sep="\t", index=False)

    files: dict[str, str] = {}
    for split_name, split_df in splits.items():
        asr_path = out_dir / f"{split_name}_asr.tsv"
        core_path = out_dir / f"{split_name}.tsv"
        audio_list_path = out_dir / f"nar_{split_name}_audio_list.txt"
        tgt_audio_list_path = out_dir / f"nar_{split_name}_tgt_audio_list.txt"

        split_df.to_csv(asr_path, sep="\t", index=False)
        split_df[CORE_COLS].to_csv(core_path, sep="\t", index=False)
        write_audio_list(audio_list_path, split_df["tgt_audio"])
        write_audio_list(tgt_audio_list_path, split_df["tgt_audio"])

        files[f"{split_name}_asr"] = str(asr_path)
        files[split_name] = str(core_path)
        files[f"nar_{split_name}_audio_list"] = str(audio_list_path)
        files[f"nar_{split_name}_tgt_audio_list"] = str(tgt_audio_list_path)

    summary = {
        "input_tsv": str(input_tsv),
        "out_dir": str(out_dir),
        "seed": args.seed,
        "dev_test_ratio": args.dev_test_ratio,
        "test_size": args.test_size,
        "input_rows": int(before_rows),
        "rows_after_filter": int(n_rows),
        "train_rows": int(len(train_df)),
        "dev_rows": int(len(dev_df)),
        "test_rows": int(len(test_df)),
        "dropped_rows": int(before_rows - n_rows),
        "drop_duplicates": bool(args.drop_duplicates),
        "drop_missing_audio": bool(args.drop_missing_audio),
        "files": files,
    }

    (out_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    flat_summary = {k: v for k, v in summary.items() if k != "files"}
    pd.DataFrame([flat_summary]).to_csv(
        out_dir / "split_summary.tsv",
        sep="\t",
        index=False,
    )

    print(
        json.dumps(
            {
                "rows": n_rows,
                "train": len(train_df),
                "dev": len(dev_df),
                "test": len(test_df),
                "out_dir": str(out_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""
python {EXPRESSIVE_S2ST_ROOT}/utils/build_vcdub_splits.py \
  --input-tsv {EXPRESSIVE_S2ST_ROOT}/es_en/seedvc_outputs_netflix_denoised/mms_lid_preprocessed_filter/sortformer_pair_filter/quality_selection/granite_asr/vcdub_text_meta.tsv \
  --join-manifest-tsv {EXPRESSIVE_S2ST_ROOT}/es_en/seedvc_outputs_netflix_denoised/mms_lid_preprocessed_filter/sortformer_pair_filter/quality_selection/dnsmospro_filtered_manifest.tsv \
  --input-id-col path \
  --join-id-col id \
  --source-text-col out_sentence \
  --target-text-col translation \
  --source-audio-col tgt_audio \
  --target-audio-col pre_tgt \
  --out-dir {EXPRESSIVE_S2ST_ROOT}/es_en/splits \
  --dev-test-ratio 0.12 \
  --test-size 504 \
  --seed 42 \
  --overwrite
"""

"""
python3 {EXPRESSIVE_S2ST_ROOT}/utils/build_vcdub_splits.py \
  --input-tsv {EXPRESSIVE_S2ST_ROOT}/de_en/seedvc_outputs_netflix_denoised/mms_lid_preprocessed_filter/sortformer_pair_filter/quality_selection/granite_asr/vcdub_text_meta.tsv \
  --join-manifest-tsv {EXPRESSIVE_S2ST_ROOT}/de_en/seedvc_outputs_netflix_denoised/mms_lid_preprocessed_filter/sortformer_pair_filter/quality_selection/dnsmospro_filtered_manifest.tsv \
  --input-id-col path \
  --join-id-col id \
  --source-text-col out_sentence \
  --target-text-col translation \
  --source-audio-col tgt_audio \
  --target-audio-col pre_tgt \
  --out-dir {EXPRESSIVE_S2ST_ROOT}/de_en/splits \
  --dev-test-ratio 0.15 \
  --test-size 500 \
  --seed 42 \
  --overwrite
"""