#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import soundfile as sf


STAGES = [
    "Raw dubbing/VC manifest",
    "After ClearVoice + Demucs success",
    "After MMS-LID language filtering",
    "After Sortformer single-speaker gate",
    "After scale-matched quality selection",
    "Training split used in Table 1",
]


def truthy(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y", "done", "ok"}


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t", low_memory=False)


def parse_vad_segments(v) -> list[tuple[float, float]]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    segments: list[tuple[float, float]] = []
    for token in str(v).strip().split():
        if ":" not in token:
            continue
        start_s, end_s = token.split(":", 1)
        try:
            start = float(start_s)
            end = float(end_s)
        except ValueError:
            continue
        if end > start:
            segments.append((start, end))
    return segments


def vad_span_duration(v) -> Optional[float]:
    segments = parse_vad_segments(v)
    if not segments:
        return None
    return max(end for _, end in segments) - min(start for start, _ in segments)


def load_duration_cache(path: Path) -> dict[str, float]:
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    df = pd.read_csv(path, sep="\t")
    if "audio_path" not in df.columns or "duration_sec" not in df.columns:
        return {}
    cache: dict[str, float] = {}
    for _, row in df.iterrows():
        audio_path = str(row["audio_path"])
        try:
            cache[audio_path] = float(row["duration_sec"])
        except Exception:
            continue
    return cache


def write_duration_cache(path: Path, cache: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"audio_path": p, "duration_sec": d} for p, d in sorted(cache.items())]
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def audio_duration_sec(path: str) -> Optional[float]:
    try:
        info = sf.info(path)
        if info.samplerate <= 0:
            return None
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return None


def fill_audio_durations(
    paths: Iterable[str],
    cache: dict[str, float],
    label: str,
    cache_path: Optional[Path] = None,
) -> None:
    unique_paths = [p for p in pd.unique(pd.Series([str(p) for p in paths if str(p)])) if p not in cache]
    total = len(unique_paths)
    if total:
        print(f"[duration] {label}: reading {total} uncached audio headers")
    for i, path in enumerate(unique_paths, start=1):
        dur = audio_duration_sec(path)
        if dur is not None:
            cache[path] = dur
        if i % 10000 == 0 or i == total:
            print(f"[duration] {label}: {i}/{total}")
            if cache_path is not None:
                write_duration_cache(cache_path, cache)


def load_split_vad_map(split_dir: Path) -> dict[str, float]:
    """Map split audio paths to VAD-span durations from train/dev/test_ar.tsv."""
    out: dict[str, float] = {}
    for split in ["train", "dev", "test"]:
        path = split_dir / f"{split}_ar.tsv"
        if not path.is_file():
            continue
        df = read_tsv(path)
        for _, row in df.iterrows():
            src_path = str(row.get("src_path", ""))
            tgt_path = str(row.get("tgt_path", ""))
            src_dur = vad_span_duration(row.get("src_vad"))
            tgt_dur = vad_span_duration(row.get("tgt_vad"))
            if src_path and src_dur is not None:
                out[src_path] = src_dur
            if tgt_path and tgt_dur is not None:
                out[tgt_path] = tgt_dur
    return out


def series_from_audio_paths(
    df: pd.DataFrame,
    audio_col: str,
    cache: dict[str, float],
    label: str,
    cache_path: Optional[Path] = None,
) -> pd.Series:
    if audio_col not in df.columns:
        raise KeyError(f"Missing audio column {audio_col} for {label}")
    paths = df[audio_col].astype(str)
    fill_audio_durations(paths, cache, label, cache_path=cache_path)
    return paths.map(cache).astype("float64")


def series_from_split_vad_map(
    df: pd.DataFrame,
    audio_col: str,
    vad_map: dict[str, float],
) -> pd.Series:
    if audio_col not in df.columns:
        raise KeyError(f"Missing audio column {audio_col}")
    return df[audio_col].astype(str).map(vad_map).astype("float64")


def summarize_stage(
    language_pair: str,
    stage: str,
    df: pd.DataFrame,
    manifest_path: str,
    source_sec: pd.Series,
    target_sec: pd.Series,
    duration_source: str,
    prev_num_pairs: Optional[int],
    force_no_retention: bool = False,
) -> dict:
    num_pairs = int(len(df))
    src_sum = float(pd.to_numeric(source_sec, errors="coerce").sum())
    tgt_sum = float(pd.to_numeric(target_sec, errors="coerce").sum())
    retention = None
    if not force_no_retention and prev_num_pairs and prev_num_pairs > 0:
        retention = float(num_pairs / prev_num_pairs)
    return {
        "language_pair": language_pair,
        "stage": stage,
        "num_pairs": num_pairs,
        "source_hours": src_sum / 3600.0,
        "target_hours": tgt_sum / 3600.0,
        "total_hours": (src_sum + tgt_sum) / 3600.0,
        "avg_source_duration_sec": src_sum / num_pairs if num_pairs else 0.0,
        "avg_target_duration_sec": tgt_sum / num_pairs if num_pairs else 0.0,
        "retention_from_previous_stage": retention,
        "duration_source_columns_used": duration_source,
        "manifest_path": manifest_path,
    }


def preprocess_success_mask(df: pd.DataFrame) -> pd.Series:
    required = [
        "preprocess_gate_ok",
        "source_clearvoice_ok",
        "target_clearvoice_ok",
        "source_demucs_ok",
        "target_demucs_ok",
    ]
    mask = pd.Series(True, index=df.index)
    for col in required:
        if col in df.columns:
            mask &= df[col].map(truthy)
    if "status" in df.columns:
        mask &= df["status"].map(truthy)
    return mask


def build_language_rows(
    language_pair: str,
    root: Path,
    split_dir: Path,
    duration_cache: dict[str, float],
    cache_path: Path,
    expected_train: Optional[Tuple[int, float, float, float]],
    strict_table1_check: bool,
) -> list[dict]:
    raw_path = root / "manifests" / "vc_manifest.tsv"
    lid_path = root / "mms_lid_preprocessed_filter" / "lid_pass_manifest.tsv"
    sort_path = root / "mms_lid_preprocessed_filter" / "sortformer_pair_filter" / "sortformer_pair_pass_strict.tsv"
    quality_path = root / "mms_lid_preprocessed_filter" / "sortformer_pair_filter" / "quality_selection" / "dnsmospro_filtered_manifest.tsv"
    train_path = split_dir / "train_ar.tsv"

    raw = read_tsv(raw_path)
    pre = raw.loc[preprocess_success_mask(raw)].copy()
    lid = read_tsv(lid_path)
    sort = read_tsv(sort_path)
    quality = read_tsv(quality_path)
    train = read_tsv(train_path)
    split_vad = load_split_vad_map(split_dir)

    rows: list[dict] = []
    prev: Optional[int] = None

    for stage, df, path in [
        (STAGES[0], raw, raw_path),
        (STAGES[1], pre, f"{raw_path} [filter: preprocessing success]"),
        (STAGES[2], lid, lid_path),
        (STAGES[3], sort, sort_path),
    ]:
        src = series_from_audio_paths(df, "out", duration_cache, f"{language_pair} {stage} src", cache_path)
        tgt = series_from_audio_paths(df, "pre_tgt", duration_cache, f"{language_pair} {stage} tgt", cache_path)
        row = summarize_stage(
            language_pair,
            stage,
            df,
            str(path),
            src,
            tgt,
            "audio_file_duration:out,pre_tgt",
            prev,
        )
        rows.append(row)
        prev = int(len(df))

    # The clean pool is exactly split into train/dev/test AR manifests, so use
    # the same VAD-span definition as Table 1 when all paths are covered.
    q_src = series_from_split_vad_map(quality, "out", split_vad)
    q_tgt = series_from_split_vad_map(quality, "pre_tgt", split_vad)
    if q_src.notna().all() and q_tgt.notna().all():
        duration_source = "splits train/dev/test_ar.tsv VAD-span:out,pre_tgt"
    else:
        q_src_audio = series_from_audio_paths(quality, "out", duration_cache, f"{language_pair} clean pool src fallback", cache_path)
        q_tgt_audio = series_from_audio_paths(quality, "pre_tgt", duration_cache, f"{language_pair} clean pool tgt fallback", cache_path)
        q_src = q_src.fillna(q_src_audio)
        q_tgt = q_tgt.fillna(q_tgt_audio)
        duration_source = "mixed:split_vad_span_when_available+audio_file_duration_fallback"
    rows.append(
        summarize_stage(
            language_pair,
            STAGES[4],
            quality,
            str(quality_path),
            q_src,
            q_tgt,
            duration_source,
            prev,
        )
    )

    train_src = train["src_vad"].map(vad_span_duration).astype("float64")
    train_tgt = train["tgt_vad"].map(vad_span_duration).astype("float64")
    train_row = summarize_stage(
        language_pair,
        STAGES[5],
        train,
        str(train_path),
        train_src,
        train_tgt,
        "src_vad,tgt_vad VAD-span",
        None,
        force_no_retention=True,
    )
    rows.append(train_row)

    if expected_train:
        exp_pairs, exp_src, exp_tgt, exp_total = expected_train
        diffs = {
            "num_pairs": abs(train_row["num_pairs"] - exp_pairs),
            "source_hours": abs(train_row["source_hours"] - exp_src),
            "target_hours": abs(train_row["target_hours"] - exp_tgt),
            "total_hours": abs(train_row["total_hours"] - exp_total),
        }
        print(f"[sanity] {language_pair} train row diffs vs Table 1 targets: {diffs}")
        if strict_table1_check:
            if diffs["num_pairs"] != 0 or any(diffs[k] > 0.02 for k in ["source_hours", "target_hours", "total_hours"]):
                raise RuntimeError(f"{language_pair} training split sanity check failed: {diffs}")

    return rows


def write_outputs(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    out_tsv = out_dir / "vcdub_filtering_stage_stats.tsv"
    out_json = out_dir / "vcdub_filtering_stage_stats.json"
    df.to_csv(out_tsv, sep="\t", index=False)
    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote TSV: {out_tsv}")
    print(f"Wrote JSON: {out_json}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute stage-wise VC-DUB filtering duration statistics.")
    ap.add_argument("--es-root", default="{EXPRESSIVE_S2ST_ROOT}/es_en/seedvc_outputs_netflix_denoised")
    ap.add_argument("--de-root", default="{EXPRESSIVE_S2ST_ROOT}/de_en/seedvc_outputs_netflix_denoised")
    ap.add_argument("--es-split-dir", default="{EXPRESSIVE_S2ST_ROOT}/es_en/splits")
    ap.add_argument("--de-split-dir", default="{EXPRESSIVE_S2ST_ROOT}/de_en/splits")
    ap.add_argument("--out-dir", default="{EXPRESSIVE_S2ST_ROOT}/data_stats/vcdub_filtering_stage_stats")
    ap.add_argument("--duration-cache", default="", help="Optional audio duration cache TSV. Defaults under --out-dir.")
    ap.add_argument("--strict-table1-check", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    cache_path = Path(args.duration_cache) if args.duration_cache else out_dir / "vcdub_filtering_duration_cache.tsv"
    duration_cache = load_duration_cache(cache_path)

    rows: list[dict] = []
    rows.extend(
        build_language_rows(
            "En-Es",
            Path(args.es_root),
            Path(args.es_split_dir),
            duration_cache,
            cache_path,
            expected_train=(79200, 84.61, 86.52, 171.13),
            strict_table1_check=args.strict_table1_check,
        )
    )
    rows.extend(
        build_language_rows(
            "En-De",
            Path(args.de_root),
            Path(args.de_split_dir),
            duration_cache,
            cache_path,
            expected_train=(131399, 127.41, 135.82, 263.23),
            strict_table1_check=args.strict_table1_check,
        )
    )

    write_duration_cache(cache_path, duration_cache)
    print(f"Wrote duration cache: {cache_path}")
    write_outputs(rows, out_dir)


if __name__ == "__main__":
    main()
