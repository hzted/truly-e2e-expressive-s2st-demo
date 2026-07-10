#!/usr/bin/env python3
"""Create VC-DUB train/dev/test construction manifests.

This script intentionally does not depend on ASR output.  It reads the selected
construction manifest after quality selection, optionally joins original aligned
metadata by sample ID, removes ASR-specific columns if present, and writes
metadata and voice-conversion split manifests.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd


ASR_SPECIFIC_COLUMNS = {"out" + "_sentence"}


def read_tsv(path: Path) -> pd.DataFrame:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            return pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False, low_memory=False)
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, low_memory=False)


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-manifest-tsv", required=True)
    parser.add_argument(
        "--aligned-metadata-tsv",
        default="",
        help="Optional original aligned-corpus metadata TSV to join by sample ID.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--id-col", default="sample_id")
    parser.add_argument("--aligned-id-col", default="")
    parser.add_argument("--output-id-col", default="sample_id")
    parser.add_argument("--source-audio-col", default="pre_src")
    parser.add_argument("--target-audio-col", default="pre_tgt")
    parser.add_argument("--source-text-col", default="src_text")
    parser.add_argument("--target-text-col", default="tgt_text")
    parser.add_argument("--split-col", default="split")
    parser.add_argument(
        "--use-existing-split-col",
        action="store_true",
        help="Use an existing split column instead of assigning new random splits.",
    )
    parser.add_argument("--dev-test-ratio", type=float, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop-duplicates", action="store_true")
    parser.add_argument("--drop-missing-audio", action="store_true")
    parser.add_argument(
        "--keep-asr-specific-columns",
        action="store_true",
        help="Do not drop columns whose names indicate ASR/Whisper metadata.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_id_col(df: pd.DataFrame, requested: str) -> str:
    if requested in df.columns:
        return requested
    if requested == "sample_id" and "id" in df.columns:
        return "id"
    raise ValueError(f"Missing required ID column: {requested}")


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def drop_asr_specific_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    drop_cols = []
    for col in df.columns:
        low = col.lower()
        if col in ASR_SPECIFIC_COLUMNS or "whisper" in low or low.endswith("_asr") or "_asr_" in low:
            drop_cols.append(col)
    return df.drop(columns=drop_cols), drop_cols


def join_aligned_metadata(
    selected: pd.DataFrame,
    selected_id_col: str,
    aligned_path: str,
    aligned_id_col: str,
) -> pd.DataFrame:
    if not aligned_path:
        return selected

    aligned = read_tsv(Path(aligned_path))
    aligned_id_col = aligned_id_col or selected_id_col
    aligned_id_col = resolve_id_col(aligned, aligned_id_col)

    rename_map = {
        c: f"aligned_{c}"
        for c in aligned.columns
        if c != aligned_id_col and c in selected.columns
    }
    aligned = aligned.rename(columns=rename_map)
    return selected.merge(
        aligned,
        left_on=selected_id_col,
        right_on=aligned_id_col,
        how="left",
        suffixes=("", "_aligned"),
    )


def add_missing_text_columns(df: pd.DataFrame, source_text_col: str, target_text_col: str) -> pd.DataFrame:
    for col in (source_text_col, target_text_col):
        if col not in df.columns:
            df[col] = ""
    return df


def assign_random_splits(
    df: pd.DataFrame,
    dev_test_ratio: float,
    test_size: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    n_rows = len(df)
    if dev_test_ratio is None or test_size is None:
        raise ValueError("--dev-test-ratio and --test-size must be explicit when assigning random splits.")
    holdout_rows = round(n_rows * dev_test_ratio)
    if test_size <= 0:
        raise ValueError("--test-size must be positive")
    if holdout_rows <= test_size:
        raise ValueError(
            f"dev+test rows ({holdout_rows}) must be larger than test rows ({test_size}); "
            "use --use-existing-split-col for already assigned split manifests."
        )

    dev_rows = holdout_rows - test_size
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = shuffled.iloc[:test_size].copy()
    dev_df = shuffled.iloc[test_size : test_size + dev_rows].copy()
    train_df = shuffled.iloc[test_size + dev_rows :].copy()

    train_df["split"] = "train"
    dev_df["split"] = "dev"
    test_df["split"] = "test"
    return {
        "train": train_df.reset_index(drop=True),
        "dev": dev_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def split_from_existing_column(df: pd.DataFrame, split_col: str) -> dict[str, pd.DataFrame]:
    require_columns(df, [split_col])
    out = {}
    for split_name in ("train", "dev", "test"):
        split_df = df[df[split_col].astype(str) == split_name].copy().reset_index(drop=True)
        split_df["split"] = split_name
        out[split_name] = split_df
    return out


def vc_columns(df: pd.DataFrame, sample_id_col: str, source_audio_col: str, target_audio_col: str) -> pd.DataFrame:
    require_columns(df, [sample_id_col, source_audio_col, target_audio_col])
    return df[[sample_id_col, source_audio_col, target_audio_col]].copy()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {out_dir}. Pass --overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_path = Path(args.selected_manifest_tsv)
    df = read_tsv(selected_path)
    id_col = resolve_id_col(df, args.id_col)
    require_columns(df, [id_col, args.source_audio_col, args.target_audio_col])

    before_rows = len(df)
    if id_col != args.output_id_col:
        df = df.rename(columns={id_col: args.output_id_col})
        id_col = args.output_id_col

    df = join_aligned_metadata(df, id_col, args.aligned_metadata_tsv, args.aligned_id_col)
    df = add_missing_text_columns(df, args.source_text_col, args.target_text_col)

    dropped_asr_cols: list[str] = []
    if not args.keep_asr_specific_columns:
        df, dropped_asr_cols = drop_asr_specific_columns(df)

    required = [id_col, args.source_audio_col, args.target_audio_col]
    df = df.dropna(subset=required).reset_index(drop=True)
    if args.drop_duplicates:
        df = df.drop_duplicates(subset=[id_col]).reset_index(drop=True)
    if args.drop_missing_audio:
        mask = df[args.source_audio_col].map(lambda p: Path(str(p)).is_file()) & df[
            args.target_audio_col
        ].map(lambda p: Path(str(p)).is_file())
        df = df[mask].reset_index(drop=True)

    if args.use_existing_split_col:
        splits = split_from_existing_column(df, args.split_col)
    else:
        splits = assign_random_splits(df, args.dev_test_ratio, args.test_size, args.seed)

    all_path = out_dir / "all_metadata.tsv"
    write_tsv(df, all_path)

    files: dict[str, str] = {"all_metadata": str(all_path)}
    for split_name, split_df in splits.items():
        metadata_path = out_dir / f"{split_name}_metadata.tsv"
        vc_path = out_dir / f"{split_name}_vc.tsv"
        write_tsv(split_df, metadata_path)
        write_tsv(vc_columns(split_df, id_col, args.source_audio_col, args.target_audio_col), vc_path)
        files[f"{split_name}_metadata"] = str(metadata_path)
        files[f"{split_name}_vc"] = str(vc_path)

    summary = {
        "selected_manifest_tsv": str(selected_path),
        "aligned_metadata_tsv": args.aligned_metadata_tsv,
        "out_dir": str(out_dir),
        "id_col": id_col,
        "source_audio_col": args.source_audio_col,
        "target_audio_col": args.target_audio_col,
        "source_text_col": args.source_text_col,
        "target_text_col": args.target_text_col,
        "seed": args.seed,
        "dev_test_ratio": args.dev_test_ratio,
        "test_size": args.test_size,
        "use_existing_split_col": bool(args.use_existing_split_col),
        "input_rows": int(before_rows),
        "rows_after_filter": int(len(df)),
        "train_rows": int(len(splits["train"])),
        "dev_rows": int(len(splits["dev"])),
        "test_rows": int(len(splits["test"])),
        "dropped_rows": int(before_rows - len(df)),
        "dropped_asr_specific_columns": dropped_asr_cols,
        "files": files,
    }
    (out_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame([{k: v for k, v in summary.items() if k != "files"}]).to_csv(
        out_dir / "split_summary.tsv",
        sep="\t",
        index=False,
    )
    print(json.dumps({k: summary[k] for k in ("rows_after_filter", "train_rows", "dev_rows", "test_rows")}, indent=2))


if __name__ == "__main__":
    main()
