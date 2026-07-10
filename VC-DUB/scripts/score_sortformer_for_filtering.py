#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

PROGRESS_COLUMNS = [
    "row_id",
    "side",
    "audio_path",
    "converted_audio_path",
    "num_speakers",
    "num_segments",
    "speaker_switches",
    "total_labeled_duration",
    "dominant_speaker",
    "dominant_duration",
    "secondary_duration",
    "secondary_ratio",
    "speaker_durations_json",
    "segments_json",
    "strict_single_speaker",
    "relaxed_single_speaker",
    "status",
    "error",
]


def truthy(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def load_progress(progress_path: Path, rerun_errors: bool) -> dict[str, dict]:
    if not progress_path.exists() or progress_path.stat().st_size == 0:
        return {}
    df = pd.read_csv(progress_path, sep="\t", dtype=str).fillna("")
    done: dict[str, dict] = {}
    for _, row in df.iterrows():
        key = f"{row['row_id']}::{row['side']}"
        status = str(row.get("status", "done"))
        if status == "done" or (status == "error" and not rerun_errors):
            done[key] = row.to_dict()
    return done


def append_progress(progress_path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    header = not progress_path.exists() or progress_path.stat().st_size == 0
    out_df = pd.DataFrame([{k: r.get(k, "") for k in PROGRESS_COLUMNS} for r in rows])
    out_df.to_csv(progress_path, sep="\t", index=False, mode="a", header=header)


def convert_to_mono_16k(in_path: str, out_path: Path, target_sr: int = 16000) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wav, sr = sf.read(in_path, always_2d=True, dtype="float32")
    wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 1.0:
        wav = wav / peak
    sf.write(str(out_path), wav.astype(np.float32), sr)
    return str(out_path.resolve())


def parse_one_segment(seg: Any):
    if isinstance(seg, dict):
        start = seg.get("start", seg.get("begin", seg.get("start_time", None)))
        end = seg.get("end", seg.get("end_time", None))
        speaker = seg.get("speaker", seg.get("speaker_id", seg.get("label", None)))
        if start is not None and end is not None and speaker is not None:
            return float(start), float(end), str(speaker)

    if isinstance(seg, (list, tuple)) and len(seg) >= 3:
        return float(seg[0]), float(seg[1]), str(seg[2])

    if isinstance(seg, str):
        parts = seg.strip().replace(",", " ").split()
        if len(parts) >= 3:
            return float(parts[0]), float(parts[1]), str(parts[2])

    raise ValueError(f"Cannot parse segment: {repr(seg)}")


def normalize_predicted_segments(pred: Any) -> list[tuple[float, float, str]]:
    if pred is None:
        return []
    if isinstance(pred, list) and len(pred) == 1 and isinstance(pred[0], list):
        pred = pred[0]

    segments = []
    if isinstance(pred, (list, tuple)):
        for x in pred:
            if x is None:
                continue
            try:
                segments.append(parse_one_segment(x))
            except Exception:
                if isinstance(x, (list, tuple)):
                    for y in x:
                        segments.append(parse_one_segment(y))
                else:
                    raise
    else:
        segments.append(parse_one_segment(pred))

    cleaned = []
    for start, end, speaker in segments:
        if end <= start:
            continue
        cleaned.append((float(start), float(end), str(speaker)))
    return sorted(cleaned, key=lambda x: (x[0], x[1], x[2]))


def analyze_segments(segments: list[tuple[float, float, str]]) -> dict:
    speaker_durations = defaultdict(float)
    for start, end, speaker in segments:
        speaker_durations[str(speaker)] += max(0.0, float(end - start))
    total_labeled_duration = float(sum(speaker_durations.values()))
    num_speakers = int(len(speaker_durations))

    dominant_speaker = None
    dominant_duration = 0.0
    if speaker_durations:
        dominant_speaker, dominant_duration = max(speaker_durations.items(), key=lambda x: x[1])

    secondary_duration = float(total_labeled_duration - dominant_duration)
    secondary_ratio = float(secondary_duration / total_labeled_duration) if total_labeled_duration > 0 else 0.0

    speaker_switches = 0
    prev_speaker = None
    for _, _, speaker in segments:
        if prev_speaker is not None and speaker != prev_speaker:
            speaker_switches += 1
        prev_speaker = speaker

    return {
        "num_speakers": num_speakers,
        "num_segments": int(len(segments)),
        "speaker_switches": int(speaker_switches),
        "total_labeled_duration": total_labeled_duration,
        "dominant_speaker": dominant_speaker,
        "dominant_duration": float(dominant_duration),
        "secondary_duration": secondary_duration,
        "secondary_ratio": secondary_ratio,
        "speaker_durations_json": json.dumps(dict(speaker_durations), ensure_ascii=False),
        "segments_json": json.dumps([{"start": s, "end": e, "speaker": spk} for s, e, spk in segments], ensure_ascii=False),
    }


def build_pair_results(df: pd.DataFrame, progress_df: pd.DataFrame, args) -> pd.DataFrame:
    # Drop any pre-existing pair-side result columns from upstream filters (e.g. MMS LID)
    # so pandas merge does not suffix away the exact column names we expect here.
    conflict_cols = [
        args.src_audio_col + "_checked",
        args.tgt_audio_col + "_checked",
        "src_converted_audio_path", "tgt_converted_audio_path",
        "src_num_speakers", "tgt_num_speakers",
        "src_num_segments", "tgt_num_segments",
        "src_speaker_switches", "tgt_speaker_switches",
        "src_total_labeled_duration", "tgt_total_labeled_duration",
        "src_dominant_speaker", "tgt_dominant_speaker",
        "src_dominant_duration", "tgt_dominant_duration",
        "src_secondary_duration", "tgt_secondary_duration",
        "src_secondary_ratio", "tgt_secondary_ratio",
        "src_speaker_durations_json", "tgt_speaker_durations_json",
        "src_segments_json", "tgt_segments_json",
        "src_strict_single_speaker", "tgt_strict_single_speaker",
        "src_relaxed_single_speaker", "tgt_relaxed_single_speaker",
        "src_status", "tgt_status",
        "src_error", "tgt_error",
        "both_done", "both_relaxed_single", "both_strict_single",
        "either_multi_or_error", "pair_pass", "pair_reject",
    ]
    df = df.drop(columns=[c for c in conflict_cols if c in df.columns], errors="ignore")

    src = progress_df[progress_df["side"] == "source"].copy().rename(columns={
        "audio_path": args.src_audio_col + "_checked",
        "converted_audio_path": "src_converted_audio_path",
        "num_speakers": "src_num_speakers",
        "num_segments": "src_num_segments",
        "speaker_switches": "src_speaker_switches",
        "total_labeled_duration": "src_total_labeled_duration",
        "dominant_speaker": "src_dominant_speaker",
        "dominant_duration": "src_dominant_duration",
        "secondary_duration": "src_secondary_duration",
        "secondary_ratio": "src_secondary_ratio",
        "speaker_durations_json": "src_speaker_durations_json",
        "segments_json": "src_segments_json",
        "strict_single_speaker": "src_strict_single_speaker",
        "relaxed_single_speaker": "src_relaxed_single_speaker",
        "status": "src_status",
        "error": "src_error",
    })
    tgt = progress_df[progress_df["side"] == "target"].copy().rename(columns={
        "audio_path": args.tgt_audio_col + "_checked",
        "converted_audio_path": "tgt_converted_audio_path",
        "num_speakers": "tgt_num_speakers",
        "num_segments": "tgt_num_segments",
        "speaker_switches": "tgt_speaker_switches",
        "total_labeled_duration": "tgt_total_labeled_duration",
        "dominant_speaker": "tgt_dominant_speaker",
        "dominant_duration": "tgt_dominant_duration",
        "secondary_duration": "tgt_secondary_duration",
        "secondary_ratio": "tgt_secondary_ratio",
        "speaker_durations_json": "tgt_speaker_durations_json",
        "segments_json": "tgt_segments_json",
        "strict_single_speaker": "tgt_strict_single_speaker",
        "relaxed_single_speaker": "tgt_relaxed_single_speaker",
        "status": "tgt_status",
        "error": "tgt_error",
    })

    src_keep = [
        "row_id", args.src_audio_col + "_checked", "src_converted_audio_path", "src_num_speakers", "src_num_segments",
        "src_speaker_switches", "src_total_labeled_duration", "src_dominant_speaker", "src_dominant_duration",
        "src_secondary_duration", "src_secondary_ratio", "src_speaker_durations_json", "src_segments_json",
        "src_strict_single_speaker", "src_relaxed_single_speaker", "src_status", "src_error",
    ]
    tgt_keep = [
        "row_id", args.tgt_audio_col + "_checked", "tgt_converted_audio_path", "tgt_num_speakers", "tgt_num_segments",
        "tgt_speaker_switches", "tgt_total_labeled_duration", "tgt_dominant_speaker", "tgt_dominant_duration",
        "tgt_secondary_duration", "tgt_secondary_ratio", "tgt_speaker_durations_json", "tgt_segments_json",
        "tgt_strict_single_speaker", "tgt_relaxed_single_speaker", "tgt_status", "tgt_error",
    ]

    out = df.copy()
    out[args.id_col] = out[args.id_col].astype(str)
    out = out.merge(src[src_keep], left_on=args.id_col, right_on="row_id", how="left").drop(columns=["row_id"])
    out = out.merge(tgt[tgt_keep], left_on=args.id_col, right_on="row_id", how="left").drop(columns=["row_id"])

    for col in ["src_strict_single_speaker", "src_relaxed_single_speaker", "tgt_strict_single_speaker", "tgt_relaxed_single_speaker"]:
        out[col] = out[col].map(truthy)

    out["both_done"] = (out["src_status"] == "done") & (out["tgt_status"] == "done")
    out["both_relaxed_single"] = out["src_relaxed_single_speaker"] & out["tgt_relaxed_single_speaker"]
    out["both_strict_single"] = out["src_strict_single_speaker"] & out["tgt_strict_single_speaker"]
    out["either_multi_or_error"] = (~out["both_done"]) | (~out["both_relaxed_single"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run NVIDIA Sortformer diarization on both source and target columns and make pair-level keep/drop decisions.")
    ap.add_argument("--input-tsv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--id-col", default="sample_id")
    ap.add_argument("--src-audio-col", default="pre_src")
    ap.add_argument("--tgt-audio-col", default="pre_tgt")
    ap.add_argument("--model-id", default="nvidia/diar_sortformer_4spk-v1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-secondary-ratio", type=float, default=0.05)
    ap.add_argument("--max-secondary-duration", type=float, default=1.0)
    ap.add_argument("--rerun-errors", action="store_true")
    ap.add_argument("--keep-converted", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    converted_dir = out_dir / "converted_16k_mono"
    progress_path = out_dir / "sortformer_audio_progress.tsv"
    audio_results_path = out_dir / "sortformer_audio_results.tsv"
    pair_results_path = out_dir / "sortformer_pair_results.tsv"
    bad_rows_path = out_dir / "sortformer_pair_bad_rows.tsv"
    good_rows_path = out_dir / "sortformer_pair_good_rows.tsv"
    bad_ids_path = out_dir / "sortformer_pair_bad_ids.txt"
    good_ids_path = out_dir / "sortformer_pair_good_ids.txt"
    summary_json = out_dir / "sortformer_pair_summary.json"
    summary_tsv = out_dir / "sortformer_pair_summary.tsv"

    df = pd.read_csv(args.input_tsv, sep="\t", low_memory=False)
    required = [args.id_col, args.src_audio_col, args.tgt_audio_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df[args.id_col] = df[args.id_col].astype(str)

    print(f"[INFO] loading Sortformer: {args.model_id}")
    from nemo.collections.asr.models import SortformerEncLabelModel

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU")
        args.device = "cpu"

    model = SortformerEncLabelModel.from_pretrained(args.model_id)
    model.eval()
    model = model.to(torch.device(args.device))

    done = load_progress(progress_path, rerun_errors=args.rerun_errors)
    work: List[dict] = []
    for _, row in df.iterrows():
        row_id = str(row[args.id_col])
        for side, audio_col in [("source", args.src_audio_col), ("target", args.tgt_audio_col)]:
            key = f"{row_id}::{side}"
            if key in done:
                continue
            work.append({"row_id": row_id, "side": side, "audio_path": str(row[audio_col])})

    rows: List[dict] = []
    pending: List[str] = []
    pending_meta: List[dict] = []

    # seed rows from existing progress
    if done:
        for rec in done.values():
            rows.append(rec)

    def build_error_result(meta: dict, conv_path: str | None, exc: Exception) -> dict:
        return {
            "row_id": meta["row_id"],
            "side": meta["side"],
            "audio_path": meta["audio_path"],
            "converted_audio_path": conv_path or "",
            "num_speakers": "",
            "num_segments": "",
            "speaker_switches": "",
            "total_labeled_duration": "",
            "dominant_speaker": "",
            "dominant_duration": "",
            "secondary_duration": "",
            "secondary_ratio": "",
            "speaker_durations_json": "",
            "segments_json": "",
            "strict_single_speaker": "",
            "relaxed_single_speaker": "",
            "status": "error",
            "error": repr(exc),
        }

    def flush_batch() -> None:
        nonlocal pending, pending_meta, rows
        if not pending:
            return
        batch_rows: List[dict] = []
        try:
            preds = model.diarize(audio=pending, batch_size=args.batch_size)
            preds_per_file = [preds] if len(pending) == 1 else preds
            if len(preds_per_file) != len(pending):
                preds_per_file = [model.diarize(audio=p, batch_size=1) for p in pending]

            for meta, conv_path, pred in zip(pending_meta, pending, preds_per_file):
                segments = normalize_predicted_segments(pred)
                metrics = analyze_segments(segments)
                strict_single = metrics["num_speakers"] <= 1
                relaxed_single = strict_single or (
                    metrics["secondary_ratio"] <= args.max_secondary_ratio
                    and metrics["secondary_duration"] <= args.max_secondary_duration
                )
                result = {
                    "row_id": meta["row_id"],
                    "side": meta["side"],
                    "audio_path": meta["audio_path"],
                    "converted_audio_path": conv_path,
                    **metrics,
                    "strict_single_speaker": bool(strict_single),
                    "relaxed_single_speaker": bool(relaxed_single),
                    "status": "done",
                    "error": "",
                }
                batch_rows.append(result)
        except Exception as exc:
            for meta, conv_path in zip(pending_meta, pending):
                try:
                    pred = model.diarize(audio=conv_path, batch_size=1)
                    segments = normalize_predicted_segments(pred)
                    metrics = analyze_segments(segments)
                    strict_single = metrics["num_speakers"] <= 1
                    relaxed_single = strict_single or (
                        metrics["secondary_ratio"] <= args.max_secondary_ratio
                        and metrics["secondary_duration"] <= args.max_secondary_duration
                    )
                    result = {
                        "row_id": meta["row_id"],
                        "side": meta["side"],
                        "audio_path": meta["audio_path"],
                        "converted_audio_path": conv_path,
                        **metrics,
                        "strict_single_speaker": bool(strict_single),
                        "relaxed_single_speaker": bool(relaxed_single),
                        "status": "done",
                        "error": "",
                    }
                except Exception as e2:
                    result = build_error_result(meta, conv_path, e2)
                batch_rows.append(result)

        append_progress(progress_path, batch_rows)
        rows.extend(batch_rows)
        pending = []
        pending_meta = []

    for item in tqdm(work, total=len(work), desc="Sortformer pair"):
        meta = {"row_id": item["row_id"], "side": item["side"], "audio_path": item["audio_path"]}
        try:
            conv_name = f"{item['row_id']}__{item['side']}.wav"
            conv_path = converted_dir / conv_name
            conv_path_str = convert_to_mono_16k(item["audio_path"], conv_path, target_sr=16000)
            pending.append(conv_path_str)
            pending_meta.append(meta)
            if len(pending) >= args.batch_size:
                flush_batch()
        except Exception as e:
            result = build_error_result(meta, None, e)
            append_progress(progress_path, [result])
            rows.append(result)

    flush_batch()

    audio_df = pd.DataFrame(rows)
    if len(audio_df):
        audio_df = audio_df.sort_values(["row_id", "side"]).reset_index(drop=True)
    audio_df.to_csv(audio_results_path, sep="\t", index=False)

    pair_df = build_pair_results(df, audio_df, args)
    pair_df.to_csv(pair_results_path, sep="\t", index=False)

    bad_df = pair_df[pair_df["either_multi_or_error"]].copy()
    good_df = pair_df[(pair_df["both_done"]) & (pair_df["both_relaxed_single"])].copy()
    bad_df.to_csv(bad_rows_path, sep="\t", index=False)
    good_df.to_csv(good_rows_path, sep="\t", index=False)
    bad_ids_path.write_text("\n".join(bad_df[args.id_col].astype(str).tolist()) + ("\n" if len(bad_df) else ""), encoding="utf-8")
    good_ids_path.write_text("\n".join(good_df[args.id_col].astype(str).tolist()) + ("\n" if len(good_df) else ""), encoding="utf-8")

    summary = {
        "input_tsv": args.input_tsv,
        "out_dir": str(out_dir),
        "model_id": args.model_id,
        "device": args.device,
        "batch_size": args.batch_size,
        "src_audio_col": args.src_audio_col,
        "tgt_audio_col": args.tgt_audio_col,
        "n_pairs": int(len(pair_df)),
        "n_audio_checks": int(len(audio_df)),
        "done_audio_checks": int((audio_df["status"] == "done").sum()) if len(audio_df) else 0,
        "error_audio_checks": int((audio_df["status"] == "error").sum()) if len(audio_df) else 0,
        "both_done_pairs": int(pair_df["both_done"].sum()),
        "both_relaxed_single_pairs": int(pair_df["both_relaxed_single"].sum()),
        "both_strict_single_pairs": int(pair_df["both_strict_single"].sum()),
        "either_multi_or_error_pairs": int(pair_df["either_multi_or_error"].sum()),
        "files": {
            "progress": str(progress_path),
            "audio_results": str(audio_results_path),
            "pair_results": str(pair_results_path),
            "bad_rows": str(bad_rows_path),
            "good_rows": str(good_rows_path),
            "bad_ids": str(bad_ids_path),
            "good_ids": str(good_ids_path),
        },
    }
    pd.DataFrame([summary]).to_csv(summary_tsv, sep="\t", index=False)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if not args.keep_converted:
        try:
            if converted_dir.exists():
                for wav in converted_dir.glob('*.wav'):
                    try:
                        wav.unlink()
                    except FileNotFoundError:
                        pass
                try:
                    converted_dir.rmdir()
                except OSError:
                    pass
        except Exception as e:
            print(f"[WARN] failed to clean converted dir: {e!r}")


if __name__ == "__main__":
    main()


"""
source {USER_HOME}/.bashrc
conda activate nemo_diar

python -u scripts/score_sortformer_for_filtering.py \
  --input-tsv {WORK_ROOT}/mms_lid/lid_pass_manifest.tsv \
  --out-dir {WORK_ROOT}/sortformer \
  --id-col sample_id \
  --src-audio-col pre_src \
  --tgt-audio-col pre_tgt \
  --device cuda \
  --batch-size 1


source {USER_HOME}/.bashrc
conda activate nemo_diar

python -u scripts/score_sortformer_for_filtering.py \
  --input-tsv {WORK_ROOT}/mms_lid/lid_pass_manifest.tsv \
  --out-dir {WORK_ROOT}/sortformer \
  --id-col sample_id \
  --src-audio-col pre_src \
  --tgt-audio-col pre_tgt \
  --device cuda \
  --batch-size 1

"""
