#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Select a subset by low UTMOS removal until reaching a target total duration."
    )
    ap.add_argument("--utmos-pairs", required=True)
    ap.add_argument("--manifest-tsv", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--utmos-id-col", default="sample_id")
    ap.add_argument("--manifest-id-col", default="id")
    ap.add_argument("--src-audio-col", default="src_audio")
    ap.add_argument("--tgt-audio-col", default="tgt_audio")
    ap.add_argument("--src-utmos-col", default="pre_src_utmos")
    ap.add_argument("--tgt-utmos-col", default="pre_tgt_utmos")
    ap.add_argument(
        "--score-mode",
        choices=["both_mean", "both_min", "src", "tgt"],
        default="both_mean",
        help="How to rank low-quality pairs for removal. both_mean usually best matches 'both sides are bad'.",
    )
    ap.add_argument(
        "--target-combined-hours",
        type=float,
        default=160.0,
        help="Target sum of src+tgt hours after filtering. Ignored when --target-keep-pairs is set.",
    )
    ap.add_argument(
        "--target-keep-pairs",
        type=int,
        default=0,
        help="If > 0, keep the top-N pairs by UTMOS rank_score after optional missing-UTMOS removal, instead of trimming by target hours.",
    )
    ap.add_argument(
        "--drop-missing-utmos",
        action="store_true",
        help="Drop rows with missing src/tgt UTMOS before duration-based trimming. Recommended.",
    )
    ap.add_argument(
        "--duration-cache-tsv",
        default="",
        help="Optional cache of audio_path -> duration_sec. Defaults under out-dir.",
    )
    return ap.parse_args()


def summarize_series(s: pd.Series, prefix: str) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    out = {}
    if len(s) == 0:
        return out
    out[f"{prefix}_n"] = int(len(s))
    out[f"{prefix}_mean"] = float(s.mean())
    out[f"{prefix}_median"] = float(s.median())
    out[f"{prefix}_std"] = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    out[f"{prefix}_min"] = float(s.min())
    out[f"{prefix}_max"] = float(s.max())
    out[f"{prefix}_p10"] = float(s.quantile(0.10))
    out[f"{prefix}_p25"] = float(s.quantile(0.25))
    out[f"{prefix}_p75"] = float(s.quantile(0.75))
    out[f"{prefix}_p90"] = float(s.quantile(0.90))
    return out


def load_duration_cache(path: Path) -> dict[str, float]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    df = pd.read_csv(path, sep="\t")
    if "audio_path" not in df.columns or "duration_sec" not in df.columns:
        return {}
    out = {}
    for _, row in df.iterrows():
        try:
            out[str(row["audio_path"])] = float(row["duration_sec"])
        except Exception:
            continue
    return out


def append_duration_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists() or path.stat().st_size == 0
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False, mode="a", header=header)


def get_duration_sec(audio_path: str) -> float:
    info = sf.info(audio_path)
    if info.samplerate <= 0:
        raise ValueError(f"Invalid samplerate for {audio_path}")
    return float(info.frames) / float(info.samplerate)


def build_duration_map(paths: list[str], cache_path: Path) -> dict[str, float]:
    cache = load_duration_cache(cache_path)
    missing = [p for p in paths if p not in cache]
    append_rows = []
    for p in tqdm(missing, desc="Reading durations"):
        dur = get_duration_sec(p)
        cache[p] = dur
        append_rows.append({"audio_path": p, "duration_sec": dur})
        if len(append_rows) >= 512:
            append_duration_rows(cache_path, append_rows)
            append_rows = []
    append_duration_rows(cache_path, append_rows)
    return cache


def main() -> None:
    args = parse_args()

    utmos_path = Path(args.utmos_pairs)
    if not utmos_path.is_file():
        raise FileNotFoundError(utmos_path)

    out_dir = Path(args.out_dir) if args.out_dir else utmos_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    duration_cache_path = Path(args.duration_cache_tsv) if args.duration_cache_tsv else (out_dir / "utmos_duration_cache.tsv")

    pairs = pd.read_csv(utmos_path, sep="\t", low_memory=False)
    required = [args.utmos_id_col, args.src_audio_col, args.tgt_audio_col, args.src_utmos_col, args.tgt_utmos_col]
    missing = [c for c in required if c not in pairs.columns]
    if missing:
        raise ValueError(f"Missing required columns in UTMOS pairs TSV: {missing}")

    pairs[args.utmos_id_col] = pairs[args.utmos_id_col].astype(str)
    pairs[args.src_utmos_col] = pd.to_numeric(pairs[args.src_utmos_col], errors="coerce")
    pairs[args.tgt_utmos_col] = pd.to_numeric(pairs[args.tgt_utmos_col], errors="coerce")
    pairs["utmos_mean"] = pairs[[args.src_utmos_col, args.tgt_utmos_col]].mean(axis=1)
    pairs["utmos_min"] = pairs[[args.src_utmos_col, args.tgt_utmos_col]].min(axis=1)
    pairs["utmos_max"] = pairs[[args.src_utmos_col, args.tgt_utmos_col]].max(axis=1)
    pairs["utmos_missing"] = pairs[[args.src_utmos_col, args.tgt_utmos_col]].isna().any(axis=1)

    if args.score_mode == "both_mean":
        pairs["rank_score"] = pairs["utmos_mean"]
    elif args.score_mode == "both_min":
        pairs["rank_score"] = pairs["utmos_min"]
    elif args.score_mode == "src":
        pairs["rank_score"] = pairs[args.src_utmos_col]
    elif args.score_mode == "tgt":
        pairs["rank_score"] = pairs[args.tgt_utmos_col]
    else:
        raise ValueError(args.score_mode)

    unique_paths = pd.unique(pd.concat([pairs[args.src_audio_col].astype(str), pairs[args.tgt_audio_col].astype(str)], ignore_index=True)).tolist()
    duration_map = build_duration_map(unique_paths, duration_cache_path)
    pairs["src_duration_sec"] = pairs[args.src_audio_col].astype(str).map(duration_map)
    pairs["tgt_duration_sec"] = pairs[args.tgt_audio_col].astype(str).map(duration_map)
    pairs["combined_duration_sec"] = pairs["src_duration_sec"] + pairs["tgt_duration_sec"]

    pairs["keep"] = True
    pairs["drop_reason"] = ""

    if args.drop_missing_utmos:
        miss_mask = pairs["utmos_missing"]
        pairs.loc[miss_mask, "keep"] = False
        pairs.loc[miss_mask, "drop_reason"] = "missing_utmos"

    current_keep_mask = pairs["keep"]
    current_combined_sec = float(pairs.loc[current_keep_mask, "combined_duration_sec"].sum())
    target_combined_sec = float(args.target_combined_hours * 3600.0)

    selection_mode = "top_n" if args.target_keep_pairs > 0 else "target_hours"
    need_remove_sec = 0.0
    removed_sec = 0.0
    removed_rows = 0
    cutoff_score = None

    if args.target_keep_pairs > 0:
        candidates = pairs.loc[pairs["keep"]].copy()
        candidates = candidates.sort_values(["rank_score", "utmos_min", "combined_duration_sec"], ascending=[False, False, False]).reset_index()
        target_keep = min(int(args.target_keep_pairs), len(candidates))
        keep_indices = set(candidates.head(target_keep)["index"].astype(int).tolist())
        if target_keep < len(candidates):
            dropped = candidates.iloc[target_keep:]
            for _, row in dropped.iterrows():
                idx = int(row["index"])
                pairs.at[idx, "keep"] = False
                pairs.at[idx, "drop_reason"] = "low_utmos_rank_trim"
                removed_sec += float(row["combined_duration_sec"])
                removed_rows += 1
            if target_keep > 0:
                cutoff_val = candidates.iloc[target_keep - 1]["rank_score"]
                cutoff_score = float(cutoff_val) if pd.notna(cutoff_val) else None
    elif current_combined_sec > target_combined_sec:
        need_remove_sec = current_combined_sec - target_combined_sec
        candidates = pairs.loc[pairs["keep"]].copy()
        candidates = candidates.sort_values(["rank_score", "utmos_min", "combined_duration_sec"], ascending=[True, True, False]).reset_index()
        for _, row in candidates.iterrows():
            if removed_sec >= need_remove_sec:
                break
            idx = int(row["index"])
            pairs.at[idx, "keep"] = False
            pairs.at[idx, "drop_reason"] = "low_utmos_trim"
            removed_sec += float(row["combined_duration_sec"])
            removed_rows += 1
            cutoff_score = float(row["rank_score"]) if pd.notna(row["rank_score"]) else None

    keep_df = pairs.loc[pairs["keep"]].copy().reset_index(drop=True)
    drop_df = pairs.loc[~pairs["keep"]].copy().reset_index(drop=True)

    keep_pairs_tsv = out_dir / "utmos_keep_pairs.tsv"
    drop_pairs_tsv = out_dir / "utmos_drop_pairs.tsv"
    keep_ids_txt = out_dir / "utmos_keep_ids.txt"
    drop_ids_txt = out_dir / "utmos_drop_ids.txt"
    summary_json = out_dir / "utmos_filter_summary.json"
    summary_tsv = out_dir / "utmos_filter_summary.tsv"
    final_manifest_tsv = out_dir / "utmos_filtered_manifest.tsv"
    dropped_manifest_tsv = out_dir / "utmos_dropped_manifest.tsv"

    keep_df.to_csv(keep_pairs_tsv, sep="\t", index=False)
    drop_df.to_csv(drop_pairs_tsv, sep="\t", index=False)
    keep_ids_txt.write_text("\n".join(keep_df[args.utmos_id_col].astype(str)) + ("\n" if len(keep_df) else ""), encoding="utf-8")
    drop_ids_txt.write_text("\n".join(drop_df[args.utmos_id_col].astype(str)) + ("\n" if len(drop_df) else ""), encoding="utf-8")

    if args.manifest_tsv:
        manifest = pd.read_csv(args.manifest_tsv, sep="\t", low_memory=False)
        if args.manifest_id_col not in manifest.columns:
            raise ValueError(f"Missing manifest id column: {args.manifest_id_col}")
        manifest[args.manifest_id_col] = manifest[args.manifest_id_col].astype(str)
        keep_ids = set(keep_df[args.utmos_id_col].astype(str))
        drop_ids = set(drop_df[args.utmos_id_col].astype(str))
        manifest_keep = manifest[manifest[args.manifest_id_col].isin(keep_ids)].copy()
        manifest_drop = manifest[manifest[args.manifest_id_col].isin(drop_ids)].copy()
        manifest_keep.to_csv(final_manifest_tsv, sep="\t", index=False)
        manifest_drop.to_csv(dropped_manifest_tsv, sep="\t", index=False)

    kept_combined_hours = float(keep_df["combined_duration_sec"].sum() / 3600.0) if len(keep_df) else 0.0
    kept_src_hours = float(keep_df["src_duration_sec"].sum() / 3600.0) if len(keep_df) else 0.0
    kept_tgt_hours = float(keep_df["tgt_duration_sec"].sum() / 3600.0) if len(keep_df) else 0.0
    dropped_combined_hours = float(drop_df["combined_duration_sec"].sum() / 3600.0) if len(drop_df) else 0.0

    summary = {
        "utmos_pairs": str(utmos_path),
        "manifest_tsv": args.manifest_tsv,
        "score_mode": args.score_mode,
        "selection_mode": selection_mode,
        "target_keep_pairs": int(args.target_keep_pairs),
        "target_combined_hours": float(args.target_combined_hours),
        "initial_pairs": int(len(pairs)),
        "kept_pairs": int(len(keep_df)),
        "dropped_pairs": int(len(drop_df)),
        "dropped_missing_utmos_pairs": int((drop_df["drop_reason"] == "missing_utmos").sum()) if len(drop_df) else 0,
        "dropped_low_utmos_trim_pairs": int((drop_df["drop_reason"] == "low_utmos_trim").sum()) if len(drop_df) else 0,
        "dropped_low_utmos_rank_trim_pairs": int((drop_df["drop_reason"] == "low_utmos_rank_trim").sum()) if len(drop_df) else 0,
        "initial_combined_hours": float(current_combined_sec / 3600.0),
        "kept_combined_hours": kept_combined_hours,
        "kept_src_hours": kept_src_hours,
        "kept_tgt_hours": kept_tgt_hours,
        "dropped_combined_hours": dropped_combined_hours,
        "requested_remove_hours": float(need_remove_sec / 3600.0),
        "actual_removed_hours": float(removed_sec / 3600.0),
        "low_utmos_cutoff_score": cutoff_score,
        "files": {
            "keep_pairs_tsv": str(keep_pairs_tsv),
            "drop_pairs_tsv": str(drop_pairs_tsv),
            "keep_ids_txt": str(keep_ids_txt),
            "drop_ids_txt": str(drop_ids_txt),
            "duration_cache_tsv": str(duration_cache_path),
            "final_manifest_tsv": str(final_manifest_tsv) if args.manifest_tsv else "",
            "dropped_manifest_tsv": str(dropped_manifest_tsv) if args.manifest_tsv else "",
        },
    }
    summary.update(summarize_series(keep_df[args.src_utmos_col], f"kept_{args.src_utmos_col}"))
    summary.update(summarize_series(keep_df[args.tgt_utmos_col], f"kept_{args.tgt_utmos_col}"))
    summary.update(summarize_series(drop_df[args.src_utmos_col], f"dropped_{args.src_utmos_col}"))
    summary.update(summarize_series(drop_df[args.tgt_utmos_col], f"dropped_{args.tgt_utmos_col}"))

    pd.DataFrame([summary]).to_csv(summary_tsv, sep="\t", index=False)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""
source {USER_HOME}/.bashrc
conda activate stopes_eval_a100

python -u {EXPRESSIVE_S2ST_ROOT}/verify_scripts/select_utmos_subset_by_hours.py \
  --utmos-pairs {EXPRESSIVE_S2ST_ROOT}/de_en/seedvc_outputs_netflix_denoised/mms_lid_preprocessed_filter/sortformer_pair_filter/utmos_pre_src_pre_tgt/utmos_pairs.tsv \
  --manifest-tsv {EXPRESSIVE_S2ST_ROOT}/de_en/seedvc_outputs_netflix_denoised/mms_lid_preprocessed_filter/sortformer_pair_filter/sortformer_pair_pass_strict.tsv \
  --target-combined-hours 300 \
  --score-mode both_mean \
  --drop-missing-utmos
"""

"""
source {USER_HOME}/.bashrc
conda activate stopes_eval_a100

python -u {EXPRESSIVE_S2ST_ROOT}/verify_scripts/select_utmos_subset_by_hours.py \
  --utmos-pairs {EXPRESSIVE_S2ST_ROOT}/es_en/seedvc_outputs_netflix_denoised/mms_lid_preprocessed_filter/sortformer_pair_filter/utmos_pre_src_pre_tgt/utmos_pairs.tsv \
  --manifest-tsv {EXPRESSIVE_S2ST_ROOT}/es_en/seedvc_outputs_netflix_denoised/mms_lid_preprocessed_filter/sortformer_pair_filter/sortformer_pair_pass_strict.tsv \
  --target-keep-pairs 90000 \
  --score-mode both_min \
  --drop-missing-utmos

"""