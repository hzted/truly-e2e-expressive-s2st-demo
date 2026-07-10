#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import torch
from tqdm import tqdm
import utmosv2
from utmosv2.dataset._schema import DatasetItem
from utmosv2.utils import get_dataset


def summarize_series(xs, prefix: str) -> dict:
    s = pd.Series(xs, dtype="float64").dropna()
    out = {}
    if len(s) == 0:
        return out

    out[f"{prefix}_n"] = int(len(s))
    out[f"{prefix}_mean"] = float(s.mean())
    out[f"{prefix}_median"] = float(s.median())
    out[f"{prefix}_std"] = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    if len(s) > 1:
        sem = float(s.sem())
        ci_low, ci_high = stats.t.interval(0.95, df=len(s) - 1, loc=float(s.mean()), scale=sem)
        out[f"{prefix}_sem"] = sem
        out[f"{prefix}_ci95_low"] = float(ci_low)
        out[f"{prefix}_ci95_high"] = float(ci_high)
    else:
        out[f"{prefix}_sem"] = 0.0
        out[f"{prefix}_ci95_low"] = float(s.mean())
        out[f"{prefix}_ci95_high"] = float(s.mean())
    out[f"{prefix}_min"] = float(s.min())
    out[f"{prefix}_max"] = float(s.max())
    out[f"{prefix}_p10"] = float(s.quantile(0.10))
    out[f"{prefix}_p25"] = float(s.quantile(0.25))
    out[f"{prefix}_p75"] = float(s.quantile(0.75))
    out[f"{prefix}_p90"] = float(s.quantile(0.90))
    return out


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def load_existing_progress(
    progress_path: Path,
    rerun_errors: bool,
) -> tuple[dict[str, float | None], dict[str, str | None]]:
    score_cache: dict[str, float | None] = {}
    error_cache: dict[str, str | None] = {}
    if not progress_path.exists() or progress_path.stat().st_size == 0:
        return score_cache, error_cache

    df = pd.read_csv(progress_path, sep="\t")
    if "audio_path" not in df.columns:
        return score_cache, error_cache

    for _, row in df.iterrows():
        audio_path = str(row["audio_path"])
        score = None if pd.isna(row.get("utmos")) else float(row["utmos"])
        error = None if pd.isna(row.get("error")) else str(row["error"])
        status = "" if pd.isna(row.get("status")) else str(row["status"])
        if rerun_errors and status == "error":
            continue
        score_cache[audio_path] = score
        error_cache[audio_path] = error

    return score_cache, error_cache


def append_progress_rows(progress_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not progress_path.exists() or progress_path.stat().st_size == 0
    df.to_csv(progress_path, sep="\t", index=False, mode="a", header=header)


def _make_dataset_items(paths: list[str], predict_dataset: str) -> list[DatasetItem]:
    return [DatasetItem(file_path=Path(p), dataset_name=predict_dataset) for p in paths]


def _score_chunk(
    model,
    paths: list[str],
    device: str,
    predict_dataset: str,
    remove_silent_section: bool,
    batch_size: int,
    num_workers: int,
    num_repetitions: int,
    verbose: bool,
) -> list[float]:
    initial_state = getattr(model._cfg.dataset, "remove_silent_section", None)
    model._cfg.dataset.remove_silent_section = remove_silent_section
    dataset = get_dataset(model._cfg, _make_dataset_items(paths, predict_dataset), model._cfg.phase)
    model._cfg.dataset.remove_silent_section = initial_state

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=bool(num_workers > 0),
    )

    preds = model._predict_impl(
        dataloader=dataloader,
        num_repetitions=num_repetitions,
        device=device,
        verbose=verbose,
    )
    return [float(x) for x in np.asarray(preds).tolist()]


def _score_paths_recursive(
    model,
    paths: list[str],
    device: str,
    predict_dataset: str,
    remove_silent_section: bool,
    batch_size: int,
    num_workers: int,
    num_repetitions: int,
    verbose: bool,
    score_cache: dict[str, float | None],
    error_cache: dict[str, str | None],
) -> None:
    if not paths:
        return

    try:
        preds = _score_chunk(
            model=model,
            paths=paths,
            device=device,
            predict_dataset=predict_dataset,
            remove_silent_section=remove_silent_section,
            batch_size=batch_size,
            num_workers=num_workers,
            num_repetitions=num_repetitions,
            verbose=verbose,
        )
        for path, pred in zip(paths, preds):
            score_cache[path] = pred
            error_cache[path] = None
        return
    except Exception as exc:  # noqa: BLE001
        if len(paths) == 1:
            score_cache[paths[0]] = None
            error_cache[paths[0]] = repr(exc)
            return

    mid = len(paths) // 2
    _score_paths_recursive(
        model=model,
        paths=paths[:mid],
        device=device,
        predict_dataset=predict_dataset,
        remove_silent_section=remove_silent_section,
        batch_size=batch_size,
        num_workers=num_workers,
        num_repetitions=num_repetitions,
        verbose=False,
        score_cache=score_cache,
        error_cache=error_cache,
    )
    _score_paths_recursive(
        model=model,
        paths=paths[mid:],
        device=device,
        predict_dataset=predict_dataset,
        remove_silent_section=remove_silent_section,
        batch_size=batch_size,
        num_workers=num_workers,
        num_repetitions=num_repetitions,
        verbose=False,
        score_cache=score_cache,
        error_cache=error_cache,
    )


def score_audio_paths(
    model,
    paths: list[str],
    device: str,
    predict_dataset: str,
    remove_silent_section: bool,
    batch_size: int,
    num_workers: int,
    num_repetitions: int,
    chunk_size: int,
    progress_path: Path,
    score_cache: dict[str, float | None],
    error_cache: dict[str, str | None],
) -> tuple[dict[str, float | None], dict[str, str | None]]:
    invalid_rows = []

    valid_paths = []
    for audio_path in paths:
        if audio_path in score_cache:
            continue
        path = Path(audio_path)
        if not path.exists():
            score_cache[audio_path] = None
            error_cache[audio_path] = f"missing_file:{path}"
            invalid_rows.append(
                {
                    "audio_path": audio_path,
                    "utmos": None,
                    "error": error_cache[audio_path],
                    "status": "error",
                }
            )
            continue
        if path.suffix.lower() != ".wav":
            score_cache[audio_path] = None
            error_cache[audio_path] = f"unsupported_suffix:{path.suffix}"
            invalid_rows.append(
                {
                    "audio_path": audio_path,
                    "utmos": None,
                    "error": error_cache[audio_path],
                    "status": "error",
                }
            )
            continue
        valid_paths.append(audio_path)

    append_progress_rows(progress_path, invalid_rows)

    total_chunks = (len(valid_paths) + chunk_size - 1) // chunk_size if valid_paths else 0
    for chunk_idx, start in enumerate(range(0, len(valid_paths), chunk_size), start=1):
        chunk = valid_paths[start : start + chunk_size]
        desc = f"Scoring UTMOS chunk {chunk_idx}/{total_chunks}"
        for _ in tqdm(range(1), desc=desc):
            _score_paths_recursive(
                model=model,
                paths=chunk,
                device=device,
                predict_dataset=predict_dataset,
                remove_silent_section=remove_silent_section,
                batch_size=batch_size,
                num_workers=num_workers,
                num_repetitions=num_repetitions,
                verbose=True,
                score_cache=score_cache,
                error_cache=error_cache,
            )
        append_progress_rows(
            progress_path,
            [
                {
                    "audio_path": path,
                    "utmos": score_cache.get(path),
                    "error": error_cache.get(path),
                    "status": "ok" if error_cache.get(path) is None else "error",
                }
                for path in chunk
            ],
        )

    return score_cache, error_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tsv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--id-col", default="sample_id")
    parser.add_argument("--src-audio-col", default="src_audio")
    parser.add_argument("--tgt-audio-col", default="hypo_audio")
    parser.add_argument("--src-label", default="src")
    parser.add_argument("--tgt-label", default="tgt")
    parser.add_argument("--config", default="fusion_stage3")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--predict-dataset", default="sarulab")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--keep-silence", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-repetitions", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--rerun-errors", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "utmos_audio_progress.tsv"

    device = resolve_device(args.device)
    remove_silent_section = not args.keep_silence

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    df = pd.read_csv(args.input_tsv, sep="\t")
    if args.src_audio_col not in df.columns:
        raise KeyError(f"Missing column: {args.src_audio_col}")
    if args.tgt_audio_col not in df.columns:
        raise KeyError(f"Missing column: {args.tgt_audio_col}")

    model = utmosv2.create_model(
        pretrained=True,
        config=args.config,
        fold=args.fold,
        checkpoint_path=args.checkpoint_path,
        seed=args.seed,
        device=device,
    )

    score_cache, error_cache = load_existing_progress(progress_path, rerun_errors=args.rerun_errors)
    unique_paths = []
    seen = set()
    for col in [args.src_audio_col, args.tgt_audio_col]:
        for p in df[col].astype(str):
            if p not in seen:
                seen.add(p)
                unique_paths.append(p)

    score_cache, error_cache = score_audio_paths(
        model=model,
        paths=unique_paths,
        device=device,
        predict_dataset=args.predict_dataset,
        remove_silent_section=remove_silent_section,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_repetitions=args.num_repetitions,
        chunk_size=args.chunk_size,
        progress_path=progress_path,
        score_cache=score_cache,
        error_cache=error_cache,
    )

    rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building pair table"):
        sample_id = str(row[args.id_col]) if args.id_col in df.columns else str(idx)
        src_audio = str(row[args.src_audio_col])
        tgt_audio = str(row[args.tgt_audio_col])
        src_utmos = score_cache.get(src_audio)
        tgt_utmos = score_cache.get(tgt_audio)

        delta = None
        reverse_delta = None
        if src_utmos is not None and tgt_utmos is not None:
            delta = float(tgt_utmos - src_utmos)
            reverse_delta = float(src_utmos - tgt_utmos)

        rows.append(
            {
                "sample_id": sample_id,
                "src_audio": src_audio,
                "tgt_audio": tgt_audio,
                f"{args.src_label}_utmos": src_utmos,
                f"{args.tgt_label}_utmos": tgt_utmos,
                f"{args.tgt_label}_minus_{args.src_label}_utmos": delta,
                f"{args.src_label}_minus_{args.tgt_label}_utmos": reverse_delta,
                f"{args.src_label}_error": error_cache.get(src_audio),
                f"{args.tgt_label}_error": error_cache.get(tgt_audio),
            }
        )

    out_df = pd.DataFrame(rows)
    out_pairs = out_dir / "utmos_pairs.tsv"
    out_df.to_csv(out_pairs, sep="\t", index=False)

    delta_col = f"{args.tgt_label}_minus_{args.src_label}_utmos"
    reverse_delta_col = f"{args.src_label}_minus_{args.tgt_label}_utmos"
    src_col = f"{args.src_label}_utmos"
    tgt_col = f"{args.tgt_label}_utmos"

    summary = {
        "input_tsv": args.input_tsv,
        "config": args.config,
        "fold": args.fold,
        "seed": args.seed,
        "predict_dataset": args.predict_dataset,
        "device": device,
        "remove_silent_section": remove_silent_section,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "num_repetitions": args.num_repetitions,
        "chunk_size": args.chunk_size,
        "progress_file": str(progress_path),
        "rerun_errors": args.rerun_errors,
        "n_pairs": int(len(out_df)),
        "n_unique_audio_paths": int(len(unique_paths)),
    }
    summary.update(summarize_series(out_df[src_col], src_col))
    summary.update(summarize_series(out_df[tgt_col], tgt_col))
    summary.update(summarize_series(out_df[delta_col], delta_col))
    summary.update(summarize_series(out_df[reverse_delta_col], reverse_delta_col))

    valid_pairs = out_df[[src_col, tgt_col]].dropna()
    summary["valid_pairs"] = int(len(valid_pairs))
    summary["failed_pairs"] = int(len(out_df) - len(valid_pairs))
    summary["failed_rate"] = float((len(out_df) - len(valid_pairs)) / len(out_df)) if len(out_df) else 0.0

    if len(valid_pairs):
        delta_series = pd.to_numeric(out_df[delta_col], errors="coerce").dropna()
        summary[f"{args.tgt_label}_lower_than_{args.src_label}_rate"] = float((delta_series < 0).mean())
        summary[f"{args.tgt_label}_higher_than_{args.src_label}_rate"] = float((delta_series > 0).mean())
        summary[f"{args.tgt_label}_equal_{args.src_label}_rate"] = float((delta_series == 0).mean())

    out_json = out_dir / "utmos_summary.json"
    out_tsv = out_dir / "utmos_summary.tsv"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(out_tsv, sep="\t", index=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote pair-level file to: {out_pairs}")
    print(f"Wrote summary json to: {out_json}")
    print(f"Wrote summary tsv to: {out_tsv}")


if __name__ == "__main__":
    main()


"""
source {USER_HOME}/.bashrc
conda activate utmos_a100

python -u {EXPRESSIVE_S2ST_ROOT}/verify_scripts/eval_utmos.py \
  --input-tsv {EXPRESSIVE_S2ST_ROOT}/de_en/seedvc_outputs_netflix_denoised/mms_lid_preprocessed_filter/sortformer_pair_filter/sortformer_pair_pass_strict.tsv \
  --out-dir {EXPRESSIVE_S2ST_ROOT}/de_en/seedvc_outputs_netflix_denoised/mms_lid_preprocessed_filter/sortformer_pair_filter/utmos_pre_src_pre_tgt \
  --id-col id \
  --src-audio-col pre_src \
  --tgt-audio-col pre_tgt \
  --src-label pre_src \
  --tgt-label pre_tgt \
  --device cuda \
  --batch-size 128 \
  --num-workers 8 \
  --chunk-size 8192
"""

"""
source {USER_HOME}/.bashrc
conda activate utmos_a100

python -u {EXPRESSIVE_S2ST_ROOT}/verify_scripts/eval_utmos.py \
  --input-tsv {EXPRESSIVE_S2ST_ROOT}/es_en/seedvc_outputs_netflix_denoised/mms_lid_preprocessed_filter/sortformer_pair_filter/sortformer_pair_pass_strict.tsv \
  --out-dir {EXPRESSIVE_S2ST_ROOT}/es_en/seedvc_outputs_netflix_denoised/mms_lid_preprocessed_filter/sortformer_pair_filter/utmos_pre_src_pre_tgt \
  --id-col id \
  --src-audio-col pre_src \
  --tgt-audio-col pre_tgt \
  --src-label pre_src \
  --tgt-label pre_tgt \
  --device cuda \
  --batch-size 256 \
  --num-workers 8 \
  --chunk-size 8192
"""