#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from scipy import stats


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def file_nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def write_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_hydra_speech_units(units: str) -> str:
    units = str(units).strip()
    if units.startswith("[") and units.endswith("]"):
        inner = units[1:-1]
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        return "[" + ",".join(parts) + "]"
    return units


def _utterance_is_valid(raw: object) -> bool:
    if pd.isna(raw):
        return False
    try:
        utt = json.loads(str(raw))
    except Exception:
        return False

    words = utt.get("words")
    text = utt.get("text")

    if text is None:
        return False
    if not isinstance(words, list) or len(words) == 0:
        return False
    if words == [""]:
        return False
    if all((w is None) or (str(w).strip() == "") for w in words):
        return False
    return True


def filter_bad_local_prosody_rows(
    src_path: Path,
    tgt_path: Path,
    src_filtered_path: Path,
    tgt_filtered_path: Path,
) -> dict:
    src_df = pd.read_csv(src_path, sep="\t")
    tgt_df = pd.read_csv(tgt_path, sep="\t")
    src_df, tgt_df, key = choose_merge_key(src_df, tgt_df)

    src_df = src_df.copy()
    tgt_df = tgt_df.copy()
    src_df["_src_valid"] = src_df["utterance"].map(_utterance_is_valid)
    tgt_df["_tgt_valid"] = tgt_df["utterance"].map(_utterance_is_valid)

    merged_keys = src_df[[key, "_src_valid"]].merge(
        tgt_df[[key, "_tgt_valid"]],
        on=key,
        how="inner",
    )
    keep_keys = merged_keys.loc[
        merged_keys["_src_valid"] & merged_keys["_tgt_valid"], key
    ].drop_duplicates()

    src_filtered = src_df[src_df[key].isin(keep_keys)].drop(columns=["_src_valid"]).reset_index(drop=True)
    tgt_filtered = tgt_df[tgt_df[key].isin(keep_keys)].drop(columns=["_tgt_valid"]).reset_index(drop=True)

    src_filtered.to_csv(src_filtered_path, sep="\t", index=False)
    tgt_filtered.to_csv(tgt_filtered_path, sep="\t", index=False)

    src_matched = int(src_df[key].isin(merged_keys[key]).sum())
    tgt_matched = int(tgt_df[key].isin(merged_keys[key]).sum())
    return {
        "local_prosody_src_rows_raw": int(len(src_df)),
        "local_prosody_tgt_rows_raw": int(len(tgt_df)),
        "local_prosody_src_rows_matched": src_matched,
        "local_prosody_tgt_rows_matched": tgt_matched,
        "local_prosody_rows_kept": int(len(keep_keys)),
        "local_prosody_rows_dropped": int(max(src_matched, tgt_matched) - len(keep_keys)),
        "local_prosody_src_invalid_rows": int((~src_df["_src_valid"]).sum()),
        "local_prosody_tgt_invalid_rows": int((~tgt_df["_tgt_valid"]).sum()),
        "local_prosody_unmatched_src_rows": int(len(src_df) - src_matched),
        "local_prosody_unmatched_tgt_rows": int(len(tgt_df) - tgt_matched),
    }


def run_cmd(
    cmd: str,
    env: Optional[dict] = None,
    name: str = "step",
    output_path: Optional[Path] = None,
) -> dict:
    prefix = f"[{name}]"
    if output_path is not None and file_nonempty(output_path):
        print(f"\n{prefix} SKIP (exists): {output_path}\n")
        return {
            "status": "skipped",
            "output": str(output_path),
            "elapsed_sec": 0.0,
        }

    print(f"\n{prefix} START")
    print("[CMD]\n" + cmd + "\n")
    t0 = time.time()
    subprocess.run(cmd, shell=True, check=True, env=env)
    dt = time.time() - t0
    print(f"{prefix} DONE in {dt/60:.2f} min\n")

    out = {
        "status": "done",
        "elapsed_sec": dt,
    }
    if output_path is not None:
        out["output"] = str(output_path)
    return out


def sanitize_ctc_aligner_text(series: pd.Series, column_name: str) -> pd.Series:
    """Replace underscores retained by word-character cleanup but absent from CTC vocab."""
    text = series.fillna("").astype(str)
    affected = int(text.str.contains("_", regex=False).sum())
    if affected:
        print(
            f"[TEXT] Replaced underscores with spaces in {affected} "
            f"{column_name} rows for CTC forced alignment."
        )
    return text.str.replace("_", " ", regex=False)


def split_manifest_tsv(manifest_path: Path, shard_dir: Path, num_shards: int) -> list[Path]:
    if num_shards <= 1:
        return [manifest_path]

    ensure_dir(shard_dir)
    df = pd.read_csv(manifest_path, sep="\t")

    shard_paths: list[Path] = []
    for shard_idx in range(num_shards):
        shard_df = df.iloc[shard_idx::num_shards].reset_index(drop=True)
        if len(shard_df) == 0:
            continue
        shard_path = shard_dir / f"{manifest_path.stem}.shard{shard_idx:03d}.tsv"
        shard_df.to_csv(shard_path, sep="\t", index=False)
        shard_paths.append(shard_path)
    return shard_paths


def merge_text_outputs(parts: list[Path], merged_path: Path) -> None:
    with merged_path.open("w", encoding="utf-8") as fout:
        first = True
        for part in parts:
            if not part.exists():
                continue
            text = part.read_text(encoding="utf-8", errors="ignore")
            if not text:
                continue
            if not first and not text.startswith("\n"):
                fout.write("\n")
            fout.write(text)
            first = False


def merge_tsv_outputs(parts: list[Path], merged_path: Path) -> None:
    dfs = []
    for part in parts:
        if part.exists() and file_nonempty(part):
            dfs.append(pd.read_csv(part, sep="\t"))
    if not dfs:
        return
    pd.concat(dfs, ignore_index=True).to_csv(merged_path, sep="\t", index=False)


def cleanup_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass


def run_cmd_once(
    cmd: str,
    env: Optional[dict] = None,
    name: str = "step",
) -> tuple[int, float]:
    prefix = f"[{name}]"
    print(f"\n{prefix} START")
    print("[CMD]\n" + cmd + "\n")
    t0 = time.time()
    ret = subprocess.run(cmd, shell=True, env=env).returncode
    dt = time.time() - t0
    if ret == 0:
        print(f"{prefix} DONE in {dt/60:.2f} min\n")
    else:
        print(f"{prefix} FAILED in {dt/60:.2f} min (exit={ret})\n")
    return ret, dt


def _extract_override_path(cmd: str, key: str) -> Optional[Path]:
    m = re.search(rf"(?:^|\s)\+?{re.escape(key)}=([^\s]+)", cmd)
    if not m:
        return None
    value = m.group(1).strip().strip("\"'")
    return Path(value)


def _replace_override_path(cmd: str, key: str, new_path: Path) -> str:
    repl = f"+{key}={new_path}"
    cmd, n = re.subn(
        rf"(\+?{re.escape(key)}=)([^\s]+)",
        lambda m: m.group(1) + str(new_path),
        cmd,
        count=1,
    )
    if n == 0:
        raise ValueError(f"Could not replace override {key} in command: {cmd}")
    return cmd


def _split_dataframe_file(path: Path, left_path: Path, right_path: Path) -> tuple[int, int]:
    df = pd.read_csv(path, sep="\t")
    if len(df) == 0:
        pd.DataFrame(columns=df.columns).to_csv(left_path, sep="\t", index=False)
        pd.DataFrame(columns=df.columns).to_csv(right_path, sep="\t", index=False)
        return 0, 0
    mid = max(1, len(df) // 2)
    df.iloc[:mid].reset_index(drop=True).to_csv(left_path, sep="\t", index=False)
    df.iloc[mid:].reset_index(drop=True).to_csv(right_path, sep="\t", index=False)
    return mid, len(df) - mid


def run_tsv_job_with_fallback(
    name: str,
    cmd: str,
    out: Path,
    env: Optional[dict],
    shard_dir: Path,
    mode: str,
    depth: int = 0,
) -> dict:
    if file_nonempty(out):
        print(f"\n[{name}] SKIP (exists): {out}\n")
        return {
            "status": "skipped",
            "output": str(out),
            "elapsed_sec": 0.0,
            "name": name,
            "fallback_depth": depth,
            "failed_rows": 0,
        }

    ret, dt = run_cmd_once(cmd, env=env, name=name)
    if ret == 0:
        return {
            "status": "done",
            "output": str(out),
            "elapsed_sec": dt,
            "name": name,
            "fallback_depth": depth,
            "failed_rows": 0,
        }

    if mode == "annot":
        data_path = _extract_override_path(cmd, "data_path")
        if data_path is None or not data_path.exists():
            return {
                "status": "failed",
                "output": str(out),
                "elapsed_sec": dt,
                "name": name,
                "fallback_depth": depth,
                "failed_rows": 0,
                "error": f"annotation input missing for failed shard: {data_path}",
            }
        df = pd.read_csv(data_path, sep="\t")
        row_count = len(df)
        if row_count <= 1:
            print(f"[{name}] FALLBACK exhausted at single row; skipping this row.")
            return {
                "status": "failed",
                "output": str(out),
                "elapsed_sec": dt,
                "name": name,
                "fallback_depth": depth,
                "failed_rows": row_count,
                "failed_input": str(data_path),
            }
        child_results = []
        child_outs = []
        failed_rows = 0
        print(f"[{name}] FALLBACK to per-row annotation for {row_count} rows.")
        for row_idx in range(row_count):
            child_name = f"{name}.row{row_idx:05d}"
            child_data = shard_dir / f"{data_path.stem}.row{row_idx:05d}.tsv"
            child_out = out.with_name(f"{out.stem}.row{row_idx:05d}{out.suffix}")
            df.iloc[[row_idx]].reset_index(drop=True).to_csv(child_data, sep="\t", index=False)
            child_cmd = _replace_override_path(cmd, "data_path", child_data)
            child_cmd = _replace_override_path(child_cmd, "result_path", child_out)
            res = run_tsv_job_with_fallback(
                name=child_name,
                cmd=child_cmd,
                out=child_out,
                env=env,
                shard_dir=shard_dir,
                mode=mode,
                depth=depth + 1,
            )
            child_results.append(res)
            if res["status"] in {"done", "partial", "skipped"} and file_nonempty(child_out):
                child_outs.append(child_out)
            failed_rows += int(res.get("failed_rows", 0))

        if child_outs:
            merge_tsv_outputs(child_outs, out)
        status = "partial" if failed_rows > 0 else "done"
        return {
            "status": status,
            "output": str(out),
            "elapsed_sec": dt,
            "name": name,
            "fallback_depth": depth,
            "failed_rows": failed_rows,
            "child_results": child_results,
        }

    if mode == "compare":
        src_path = _extract_override_path(cmd, "src_path")
        tgt_path = _extract_override_path(cmd, "tgt_path")
        if src_path is None or tgt_path is None or not src_path.exists() or not tgt_path.exists():
            return {
                "status": "failed",
                "output": str(out),
                "elapsed_sec": dt,
                "name": name,
                "fallback_depth": depth,
                "failed_rows": 0,
                "error": f"compare inputs missing for failed shard: src={src_path}, tgt={tgt_path}",
            }
        src_df = pd.read_csv(src_path, sep="\t")
        tgt_df = pd.read_csv(tgt_path, sep="\t")
        if len(src_df) != len(tgt_df):
            return {
                "status": "failed",
                "output": str(out),
                "elapsed_sec": dt,
                "name": name,
                "fallback_depth": depth,
                "failed_rows": min(len(src_df), len(tgt_df)),
                "error": f"compare input length mismatch: src={len(src_df)} tgt={len(tgt_df)}",
            }
        row_count = len(src_df)
        if row_count <= 1:
            print(f"[{name}] FALLBACK exhausted at single row; skipping this pair.")
            return {
                "status": "failed",
                "output": str(out),
                "elapsed_sec": dt,
                "name": name,
                "fallback_depth": depth,
                "failed_rows": row_count,
                "failed_src_input": str(src_path),
                "failed_tgt_input": str(tgt_path),
            }
        child_results = []
        child_outs = []
        failed_rows = 0
        print(f"[{name}] FALLBACK to per-row compare for {row_count} rows.")
        for row_idx in range(row_count):
            child_name = f"{name}.row{row_idx:05d}"
            child_src = shard_dir / f"{src_path.stem}.row{row_idx:05d}.tsv"
            child_tgt = shard_dir / f"{tgt_path.stem}.row{row_idx:05d}.tsv"
            child_out = out.with_name(f"{out.stem}.row{row_idx:05d}{out.suffix}")
            src_df.iloc[[row_idx]].reset_index(drop=True).to_csv(child_src, sep="\t", index=False)
            tgt_df.iloc[[row_idx]].reset_index(drop=True).to_csv(child_tgt, sep="\t", index=False)
            child_cmd = _replace_override_path(cmd, "src_path", child_src)
            child_cmd = _replace_override_path(child_cmd, "tgt_path", child_tgt)
            child_cmd = _replace_override_path(child_cmd, "result_path", child_out)
            res = run_tsv_job_with_fallback(
                name=child_name,
                cmd=child_cmd,
                out=child_out,
                env=env,
                shard_dir=shard_dir,
                mode=mode,
                depth=depth + 1,
            )
            child_results.append(res)
            if res["status"] in {"done", "partial", "skipped"} and file_nonempty(child_out):
                child_outs.append(child_out)
            failed_rows += int(res.get("failed_rows", 0))

        if child_outs:
            merge_tsv_outputs(child_outs, out)
        status = "partial" if failed_rows > 0 else "done"
        return {
            "status": status,
            "output": str(out),
            "elapsed_sec": dt,
            "name": name,
            "fallback_depth": depth,
            "failed_rows": failed_rows,
            "child_results": child_results,
        }

    raise ValueError(f"Unsupported fallback mode: {mode}")


def run_cmds_parallel(
    jobs: list[tuple[str, str, Path]],
    env: Optional[dict],
    max_parallel: int,
    shard_dir: Optional[Path] = None,
    fallback_mode: Optional[str] = None,
) -> list[dict]:
    if max_parallel <= 1:
        results = []
        for name, cmd, out in jobs:
            if fallback_mode is not None and shard_dir is not None:
                results.append(
                    run_tsv_job_with_fallback(
                        name=name,
                        cmd=cmd,
                        out=out,
                        env=env,
                        shard_dir=shard_dir,
                        mode=fallback_mode,
                    )
                )
            else:
                results.append(run_cmd(cmd, env=env, name=name, output_path=out))
        return results

    queue = deque(jobs)
    running: list[tuple[str, str, Path, subprocess.Popen, float]] = []
    results: list[dict] = []

    while queue or running:
        while queue and len(running) < max_parallel:
            name, cmd, out = queue.popleft()
            prefix = f"[{name}]"
            if file_nonempty(out):
                print(f"\n{prefix} SKIP (exists): {out}\n")
                results.append(
                    {
                        "status": "skipped",
                        "output": str(out),
                        "elapsed_sec": 0.0,
                        "name": name,
                    }
                )
                continue

            print(f"\n{prefix} START")
            print("[CMD]\n" + cmd + "\n")
            proc = subprocess.Popen(cmd, shell=True, env=env)
            running.append((name, cmd, out, proc, time.time()))

        time.sleep(1.0)
        still_running: list[tuple[str, str, Path, subprocess.Popen, float]] = []
        for name, cmd, out, proc, t0 in running:
            ret = proc.poll()
            if ret is None:
                still_running.append((name, cmd, out, proc, t0))
                continue

            dt = time.time() - t0
            if ret != 0:
                if fallback_mode is not None and shard_dir is not None:
                    print(f"[{name}] FAILED in parallel run; falling back to smaller shards.")
                    results.append(
                        run_tsv_job_with_fallback(
                            name=name,
                            cmd=cmd,
                            out=out,
                            env=env,
                            shard_dir=shard_dir,
                            mode=fallback_mode,
                        )
                    )
                    continue
                raise subprocess.CalledProcessError(ret, cmd)

            print(f"[{name}] DONE in {dt/60:.2f} min\n")
            results.append(
                {
                    "status": "done",
                    "output": str(out),
                    "elapsed_sec": dt,
                    "name": name,
                }
            )
        running = still_running

    return results


def sample_df(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    if frac <= 0 or frac > 1:
        raise ValueError("sample_frac must be in (0, 1].")
    if frac == 1.0:
        return df.reset_index(drop=True)
    n = max(1, int(round(len(df) * frac)))
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def extract_numeric_series_from_text(path: Path) -> list[float]:
    values: list[float] = []
    if not path.exists():
        return values
    float_pat = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        nums = [float(x) for x in float_pat.findall(line)]
        if nums:
            values.append(nums[-1])
    return values


def add_ci_fields(out: dict, xs, prefix: str) -> None:
    s = pd.to_numeric(pd.Series(xs), errors="coerce").dropna()
    if len(s) == 0:
        return
    mean = float(s.mean())
    if len(s) <= 1:
        out[f"{prefix}_sem"] = 0.0
        out[f"{prefix}_ci95_low"] = mean
        out[f"{prefix}_ci95_high"] = mean
        out[f"{prefix}_ci95_half_width"] = 0.0
        return
    sem = float(s.sem())
    if sem == 0.0 or not math.isfinite(sem):
        ci_low, ci_high = mean, mean
    else:
        ci_low, ci_high = stats.t.interval(0.95, df=len(s) - 1, loc=mean, scale=sem)
    if not math.isfinite(float(ci_low)) or not math.isfinite(float(ci_high)):
        ci_low, ci_high = mean, mean
    out[f"{prefix}_sem"] = sem
    out[f"{prefix}_ci95_low"] = float(ci_low)
    out[f"{prefix}_ci95_high"] = float(ci_high)
    out[f"{prefix}_ci95_half_width"] = float((ci_high - ci_low) / 2.0)


def summarize_scalar_distribution(xs: list[float], prefix: str) -> dict:
    out = {}
    if not xs:
        return out

    s = pd.Series(xs, dtype="float64").dropna()
    if len(s) == 0:
        return out

    out[f"{prefix}_n"] = int(s.shape[0])
    out[f"{prefix}_mean"] = float(s.mean())
    out[f"{prefix}_median"] = float(s.median())
    out[f"{prefix}_std"] = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    add_ci_fields(out, s, prefix)
    out[f"{prefix}_min"] = float(s.min())
    out[f"{prefix}_max"] = float(s.max())
    out[f"{prefix}_p10"] = float(s.quantile(0.10))
    out[f"{prefix}_p25"] = float(s.quantile(0.25))
    out[f"{prefix}_p75"] = float(s.quantile(0.75))
    out[f"{prefix}_p90"] = float(s.quantile(0.90))
    return out


def compute_rank(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def pearson_corr(x: list[float], y: list[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx == 0 or vy == 0:
        return None
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return cov / (vx ** 0.5 * vy ** 0.5)


def spearman_corr(x: list[float], y: list[float]) -> Optional[float]:
    return pearson_corr(compute_rank(x), compute_rank(y))


def choose_merge_key(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    for cand in ["sample_id", "id", "path"]:
        if cand in df1.columns and cand in df2.columns:
            return df1, df2, cand

    df1 = df1.copy()
    df2 = df2.copy()
    df1["__idx__"] = range(len(df1))
    df2["__idx__"] = range(len(df2))
    return df1, df2, "__idx__"


def find_best_rate_column(cols: Iterable[str]) -> Optional[str]:
    cols = list(cols)
    cols_lower = {c: c.lower() for c in cols}

    unit_priority = ["syll", "word", "char", "phoneme", "vowel"]

    for unit in unit_priority:
        candidates = [
            c for c in cols
            if "speech_rate" in cols_lower[c] and unit in cols_lower[c]
        ]
        if candidates:
            return candidates[0]

    for unit in unit_priority:
        candidates = [
            c for c in cols
            if "rate" in cols_lower[c] and unit in cols_lower[c]
        ]
        if candidates:
            return candidates[0]

    generic = [c for c in cols if "speech_rate" in cols_lower[c]]
    if generic:
        return generic[0]

    generic = [c for c in cols if "rate" in cols_lower[c]]
    if generic:
        return generic[0]

    return None


def summarize_rate(src_annot: Path, tgt_annot: Path, out_dir: Path) -> dict:
    out = {}
    if not src_annot.exists() or not tgt_annot.exists():
        return out

    src_df = pd.read_csv(src_annot, sep="\t")
    tgt_df = pd.read_csv(tgt_annot, sep="\t")
    src_df, tgt_df, key = choose_merge_key(src_df, tgt_df)

    rate_corr_rows = []
    for unit in ["word", "syllable", "char"]:
        col = f"speech_rate_{unit}"
        if col not in src_df.columns or col not in tgt_df.columns:
            continue
        unit_df = src_df[[key, col]].rename(columns={col: "rate_src"}).merge(
            tgt_df[[key, col]].rename(columns={col: "rate_tgt"}),
            on=key,
            how="inner",
        )
        unit_df["rate_src"] = pd.to_numeric(unit_df["rate_src"], errors="coerce")
        unit_df["rate_tgt"] = pd.to_numeric(unit_df["rate_tgt"], errors="coerce")
        unit_df = unit_df.dropna()
        if len(unit_df) < 2:
            continue
        x_unit = unit_df["rate_src"].astype(float).tolist()
        y_unit = unit_df["rate_tgt"].astype(float).tolist()
        pearson = pearson_corr(x_unit, y_unit)
        spearman = spearman_corr(x_unit, y_unit)
        rate_corr_rows.append(
            {
                "speech_rate_unit": col,
                "n": int(len(unit_df)),
                "pearson": pearson,
                "spearman": spearman,
            }
        )
        out[f"{col}_n"] = int(len(unit_df))
        out[f"{col}_pearson"] = pearson
        out[f"{col}_spearman"] = spearman

    if rate_corr_rows:
        rate_corr_path = out_dir / "speech_rate_correlations.tsv"
        pd.DataFrame(rate_corr_rows).to_csv(rate_corr_path, sep="\t", index=False)
        out["speech_rate_correlations_tsv"] = str(rate_corr_path)

    rate_col_src = find_best_rate_column(src_df.columns)
    rate_col_tgt = find_best_rate_column(tgt_df.columns)
    if rate_col_src is None or rate_col_tgt is None:
        return out

    merged = src_df[[key, rate_col_src]].rename(columns={rate_col_src: "rate_src"}).merge(
        tgt_df[[key, rate_col_tgt]].rename(columns={rate_col_tgt: "rate_tgt"}),
        on=key,
        how="inner",
    ).copy()

    merged["rate_src"] = pd.to_numeric(merged["rate_src"], errors="coerce")
    merged["rate_tgt"] = pd.to_numeric(merged["rate_tgt"], errors="coerce")
    merged = merged.dropna().reset_index(drop=True)

    if len(merged) == 0:
        return out

    x = merged["rate_src"].astype(float).tolist()
    y = merged["rate_tgt"].astype(float).tolist()

    diffs = [b - a for a, b in zip(x, y)]
    abs_diffs = [abs(d) for d in diffs]
    sq_diffs = [d * d for d in diffs]

    log_ratio = []
    for a, b in zip(x, y):
        if a > 0 and b > 0:
            log_ratio.append(abs(math.log(b / a)))

    out["rate_column_src"] = rate_col_src
    out["rate_column_tgt"] = rate_col_tgt
    out["rate_n"] = len(merged)
    out["rate_pearson"] = pearson_corr(x, y)
    out["rate_spearman"] = spearman_corr(x, y)
    out["rate_mae"] = float(sum(abs_diffs) / len(abs_diffs))
    out["rate_rmse"] = float((sum(sq_diffs) / len(sq_diffs)) ** 0.5)
    out["rate_mean_signed_diff"] = float(sum(diffs) / len(diffs))
    out["rate_mean_abs_log_ratio"] = float(sum(log_ratio) / len(log_ratio)) if log_ratio else None

    merged.to_csv(out_dir / "rate_pairs.csv", index=False)
    return out


def summarize_pause(compare_path: Path, out_dir: Path) -> dict:
    out = {}
    if not compare_path.exists():
        return out

    df = pd.read_csv(compare_path, sep="\t")
    cols_lower = [c.lower() for c in df.columns]

    native_pause_cols = {
        "pause_wmean_duration_score": "wmean_duration_score",
        "pause_wmean_alignment_score": "wmean_alignment_score",
        "pause_wmean_joint_score": "wmean_joint_score",
        "pause_mean_duration_score": "mean_duration_score",
        "pause_mean_alignment_score": "mean_alignment_score",
        "pause_mean_joint_score": "mean_joint_score",
    }
    for out_key, col_name in native_pause_cols.items():
        if col_name in df.columns:
            vals = pd.to_numeric(df[col_name], errors="coerce").dropna()
            if len(vals):
                out[out_key] = float(vals.iloc[0]) if len(vals) == 1 else float(vals.mean())
                add_ci_fields(out, vals, out_key)

    joint_cols = [
        df.columns[i]
        for i, c in enumerate(cols_lower)
        if "joint" in c and "score" in c
    ]
    weight_cols = [
        df.columns[i]
        for i, c in enumerate(cols_lower)
        if "weight" in c
    ]

    if not joint_cols:
        return out

    joint_col = joint_cols[0]
    joint = pd.to_numeric(df[joint_col], errors="coerce").dropna()

    out["pause_joint_column"] = joint_col
    out["pause_n"] = int(joint.shape[0])
    if len(joint):
        out["pause_mean_joint"] = float(joint.mean())
        out["pause_median_joint"] = float(joint.median())
        out["pause_std_joint"] = float(joint.std(ddof=1)) if len(joint) > 1 else 0.0
        add_ci_fields(out, joint, "pause_joint")
        out["pause_p10_joint"] = float(joint.quantile(0.10))
        out["pause_p25_joint"] = float(joint.quantile(0.25))
        out["pause_p75_joint"] = float(joint.quantile(0.75))
        out["pause_p90_joint"] = float(joint.quantile(0.90))

    if weight_cols:
        weight_col = weight_cols[0]
        weights = pd.to_numeric(df[weight_col], errors="coerce")
        tmp = pd.DataFrame(
            {"joint": pd.to_numeric(df[joint_col], errors="coerce"), "weight": weights}
        ).dropna()
        tmp = tmp[tmp["weight"] > 0]
        if len(tmp):
            out["pause_weight_column"] = weight_col
            out["pause_weighted_mean_joint"] = float(
                (tmp["joint"] * tmp["weight"]).sum() / tmp["weight"].sum()
            )

    df.to_csv(out_dir / "pause_scores_copy.csv", index=False)
    return out


def read_audio_duration_sec(path_str: str) -> Optional[float]:
    p = str(path_str)
    try:
        import soundfile as sf
        return float(sf.info(p).duration)
    except Exception:
        pass

    try:
        import torchaudio
        info = torchaudio.info(p)
        if info.sample_rate and info.num_frames:
            return float(info.num_frames) / float(info.sample_rate)
    except Exception:
        pass

    return None


def compliance_score(a: float, b: float) -> Optional[float]:
    if a is None or b is None:
        return None
    if a <= 0 or b <= 0:
        return None
    lo = min(a, b)
    hi = max(a, b)
    if hi <= 0:
        return None
    return float(lo / hi)


def summarize_dc_sc(
    manifest_path: Path,
    src_annot: Path,
    tgt_annot: Path,
    out_dir: Path,
) -> dict:
    out = {}
    if not manifest_path.exists():
        return out

    manifest = pd.read_csv(manifest_path, sep="\t")
    if not {"src_audio", "hypo_audio"}.issubset(set(manifest.columns)):
        return out

    if "sample_id" not in manifest.columns:
        manifest = manifest.copy()
        manifest["sample_id"] = range(len(manifest))

    pairs = manifest[["sample_id", "src_audio", "hypo_audio"]].copy()
    pairs["src_duration_sec"] = pairs["src_audio"].apply(read_audio_duration_sec)
    pairs["tgt_duration_sec"] = pairs["hypo_audio"].apply(read_audio_duration_sec)
    pairs["dc_score"] = [
        compliance_score(a, b)
        for a, b in zip(pairs["src_duration_sec"], pairs["tgt_duration_sec"])
    ]

    dc = pd.to_numeric(pairs["dc_score"], errors="coerce").dropna()
    out["dc_n"] = int(dc.shape[0])
    if len(dc):
        out["dc_mean"] = float(dc.mean())
        out["dc_median"] = float(dc.median())
        out["dc_std"] = float(dc.std(ddof=1)) if len(dc) > 1 else 0.0
        add_ci_fields(out, dc, "dc")
        out["dc_min"] = float(dc.min())
        out["dc_max"] = float(dc.max())
        dc_0p2 = (dc >= 0.8).astype("float64")
        dc_0p4 = (dc >= 0.6).astype("float64")
        out["dc_ge_0_8"] = float(dc_0p2.mean())
        out["dc_ge_0_6"] = float(dc_0p4.mean())
        # More explicit aliases: compliance within 20% / 40% of the longer duration.
        out["dc_0p2_compliance"] = out["dc_ge_0_8"]
        out["dc_0p4_compliance"] = out["dc_ge_0_6"]
        add_ci_fields(out, dc_0p2, "dc_0p2_compliance")
        add_ci_fields(out, dc_0p4, "dc_0p4_compliance")
        out["dc_p10"] = float(dc.quantile(0.10))
        out["dc_p25"] = float(dc.quantile(0.25))
        out["dc_p75"] = float(dc.quantile(0.75))
        out["dc_p90"] = float(dc.quantile(0.90))
        out["src_duration_mean_sec"] = float(pd.to_numeric(pairs["src_duration_sec"], errors="coerce").dropna().mean())
        out["tgt_duration_mean_sec"] = float(pd.to_numeric(pairs["tgt_duration_sec"], errors="coerce").dropna().mean())

    if src_annot.exists() and tgt_annot.exists():
        src_df = pd.read_csv(src_annot, sep="\t")
        tgt_df = pd.read_csv(tgt_annot, sep="\t")
        src_df, tgt_df, key = choose_merge_key(src_df, tgt_df)

        rate_col_src = find_best_rate_column(src_df.columns)
        rate_col_tgt = find_best_rate_column(tgt_df.columns)

        if rate_col_src is not None and rate_col_tgt is not None:
            rate_pairs = src_df[[key, rate_col_src]].rename(columns={rate_col_src: "rate_src"}).merge(
                tgt_df[[key, rate_col_tgt]].rename(columns={rate_col_tgt: "rate_tgt"}),
                on=key,
                how="inner",
            ).copy()

            if key != "sample_id":
                if "sample_id" in src_df.columns and "sample_id" in tgt_df.columns:
                    rate_pairs = src_df[[key, "sample_id", rate_col_src]].rename(columns={rate_col_src: "rate_src"}).merge(
                        tgt_df[[key, rate_col_tgt]].rename(columns={rate_col_tgt: "rate_tgt"}),
                        on=key,
                        how="inner",
                    )
                else:
                    rate_pairs["sample_id"] = rate_pairs[key]
            else:
                rate_pairs["sample_id"] = rate_pairs[key]

            rate_pairs["rate_src"] = pd.to_numeric(rate_pairs["rate_src"], errors="coerce")
            rate_pairs["rate_tgt"] = pd.to_numeric(rate_pairs["rate_tgt"], errors="coerce")
            rate_pairs = rate_pairs.dropna().reset_index(drop=True)

            rate_pairs["sc_score"] = [
                compliance_score(a, b)
                for a, b in zip(rate_pairs["rate_src"], rate_pairs["rate_tgt"])
            ]

            rate_pairs = pairs.merge(
                rate_pairs[["sample_id", "rate_src", "rate_tgt", "sc_score"]],
                on="sample_id",
                how="left",
            )

            sc = pd.to_numeric(rate_pairs["sc_score"], errors="coerce").dropna()
            out["sc_rate_column_src"] = rate_col_src
            out["sc_rate_column_tgt"] = rate_col_tgt
            out["sc_n"] = int(sc.shape[0])
            if len(sc):
                sc_tmp = rate_pairs[["sc_score", "src_duration_sec", "tgt_duration_sec"]].copy()
                sc_tmp["sc_score"] = pd.to_numeric(sc_tmp["sc_score"], errors="coerce")
                sc_tmp["duration_weight"] = pd.concat(
                    [
                        pd.to_numeric(sc_tmp["src_duration_sec"], errors="coerce"),
                        pd.to_numeric(sc_tmp["tgt_duration_sec"], errors="coerce"),
                    ],
                    axis=1,
                ).max(axis=1)
                sc_tmp = sc_tmp.dropna()
                sc_tmp = sc_tmp[sc_tmp["duration_weight"] > 0]
                out["sc_mean"] = float(sc.mean())
                out["sc_median"] = float(sc.median())
                out["sc_std"] = float(sc.std(ddof=1)) if len(sc) > 1 else 0.0
                add_ci_fields(out, sc, "sc")
                out["sc_min"] = float(sc.min())
                out["sc_max"] = float(sc.max())
                sc_0p2 = (sc >= 0.8).astype("float64")
                sc_0p4 = (sc >= 0.6).astype("float64")
                out["sc_ge_0_8"] = float(sc_0p2.mean())
                out["sc_ge_0_6"] = float(sc_0p4.mean())
                out["sc_0p2_compliance"] = out["sc_ge_0_8"]
                out["sc_0p4_compliance"] = out["sc_ge_0_6"]
                add_ci_fields(out, sc_0p2, "sc_0p2_compliance")
                add_ci_fields(out, sc_0p4, "sc_0p4_compliance")
                if len(sc_tmp):
                    out["sc_weighted_mean"] = float(
                        (sc_tmp["sc_score"] * sc_tmp["duration_weight"]).sum()
                        / sc_tmp["duration_weight"].sum()
                    )
                out["sc_p10"] = float(sc.quantile(0.10))
                out["sc_p25"] = float(sc.quantile(0.25))
                out["sc_p75"] = float(sc.quantile(0.75))
                out["sc_p90"] = float(sc.quantile(0.90))

            rate_pairs.to_csv(out_dir / "dc_sc_pairs.csv", index=False)
            return out

    pairs.to_csv(out_dir / "dc_pairs.csv", index=False)
    return out


def write_summary_tsv(summary: dict, path: Path) -> None:
    df = pd.DataFrame([summary])
    df.to_csv(path, sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stopes expressive metrics on a sampled TSV subset.")
    parser.add_argument("--input-tsv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sample-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--src-lang", required=True, help="Stopes language code, e.g. eng, deu")
    parser.add_argument("--tgt-lang", required=True, help="Stopes language code, e.g. deu, eng")

    parser.add_argument("--src-audio-col", default="src_audio")
    parser.add_argument("--tgt-audio-col", default="tgt_audio")
    parser.add_argument("--src-text-col", default="sentence")
    parser.add_argument("--tgt-text-col", default="translation")
    parser.add_argument("--id-col", default="path")

    parser.add_argument("--run-autopcp", action="store_true")
    parser.add_argument("--run-vsim", action="store_true")
    parser.add_argument("--run-local-prosody", action="store_true")

    parser.add_argument("--wavlm-ckpt", default=None, help="Required when --run-vsim")
    parser.add_argument("--forced-aligner", default="ctc_wav2vec2-xlsr-multilingual-56")
    parser.add_argument("--pause-min-duration", type=float, default=0.1)
    parser.add_argument("--speech-units", default="[word,syllable,char,phoneme,vowel]")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--parallel-jobs", type=int, default=1)
    parser.add_argument("--keep-shard-files", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument(
        "--autopcp-cpu-only",
        action="store_true",
        help=(
            "Run only AutoPCP on CPU. Unlike --cpu-only, this keeps CUDA visible "
            "for VSim and local prosody forced alignment."
        ),
    )
    parser.add_argument(
        "--vsim-cpu-only",
        action="store_true",
        help=(
            "Run only vocal-style similarity on CPU. This avoids CUDA contention "
            "with the training process while leaving local prosody free to use CUDA."
        ),
    )

    args = parser.parse_args()

    # 不再默认全开，改成显式开关制
    enabled_metrics = []
    if args.run_autopcp:
        enabled_metrics.append("autopcp")
    if args.run_vsim:
        enabled_metrics.append("vsim")
    if args.run_local_prosody:
        enabled_metrics.append("local_prosody")

    if not enabled_metrics:
        raise ValueError(
            "No metric selected. Use one or more of: "
            "--run-autopcp --run-vsim --run-local-prosody"
        )

    print(f"[METRICS] enabled: {enabled_metrics}")

    if args.run_vsim and not args.wavlm_ckpt:
        raise ValueError("--wavlm-ckpt is required when --run-vsim is enabled.")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.parallel_jobs < 1:
        raise ValueError("--parallel-jobs must be >= 1")

    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)

    progress_path = out_dir / "progress.json"

    df = pd.read_csv(args.input_tsv, sep="\t")
    needed = [args.src_audio_col, args.tgt_audio_col, args.src_text_col, args.tgt_text_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input TSV: {missing}")

    sampled = sample_df(df, frac=args.sample_frac, seed=args.seed)
    sampled_raw_path = out_dir / "sampled_raw.tsv"
    sampled.to_csv(sampled_raw_path, sep="\t", index=False)

    manifest = pd.DataFrame({
        "sample_id": sampled[args.id_col].astype(str) if args.id_col in sampled.columns else sampled.index.astype(str),
        "src_audio": sampled[args.src_audio_col].astype(str),
        "hypo_audio": sampled[args.tgt_audio_col].astype(str),
        "src_text": sanitize_ctc_aligner_text(sampled[args.src_text_col], "source-text"),
        "s2t_out": sanitize_ctc_aligner_text(sampled[args.tgt_text_col], "target-text"),
    })
    manifest_path = out_dir / "stopes_eval_manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)
    shard_dir = out_dir / "_tmp_shards"
    shard_manifest_paths = split_manifest_tsv(manifest_path, shard_dir, args.num_shards)

    pair_tsv = out_dir / "audio_pairs_noheader.tsv"
    manifest[["src_audio", "hypo_audio"]].to_csv(pair_tsv, sep="\t", header=False, index=False)

    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    if args.cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = ""

    import stopes  # late import

    stopes_root = Path(inspect.getfile(stopes)).resolve().parent
    annotate_script = stopes_root / "eval/local_prosody/annotate_utterances.py"
    compare_script = stopes_root / "eval/local_prosody/compare_utterances.py"
    compat_dir = Path(__file__).resolve().parent / "_compat"
    pythonpath_parts = [str(compat_dir), str(stopes_root.parent)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["TRANSFORMERS_ALLOW_UNSAFE_TORCH_LOAD_FOR_STOPES"] = "1"
    stopes_runtime_dir = out_dir / "_stopes_runtime"
    stopes_cache_dir = stopes_runtime_dir / "stopes_cache"
    stopes_tmp_dir = stopes_runtime_dir / "tmp"
    hydra_run_dir = stopes_runtime_dir / "hydra_outputs"
    ensure_dir(stopes_cache_dir)
    ensure_dir(stopes_tmp_dir)
    ensure_dir(hydra_run_dir)
    env["TMPDIR"] = str(stopes_tmp_dir)
    env["TEMP"] = str(stopes_tmp_dir)
    env["TMP"] = str(stopes_tmp_dir)
    # Hydra otherwise writes relative outputs/YYYY-MM-DD/... under the caller cwd,
    # which is often not writable on shared cluster filesystems.
    hydra_override = f"hydra.run.dir={hydra_run_dir} hydra.output_subdir=null "
    cache_override = f"{hydra_override}launcher.cache.caching_dir={stopes_cache_dir} "

    def stage_env(force_cpu: bool) -> dict:
        """Hide GPUs for stage-specific CPU-only runs.

        Some STOPES dependencies inspect CUDA availability independently of the
        Hydra `use_cuda=false` flag. If CUDA remains visible, tensors can still
        land on GPU while model weights stay on CPU.
        """
        stage_specific_env = env.copy()
        if force_cpu:
            stage_specific_env["CUDA_VISIBLE_DEVICES"] = ""
        return stage_specific_env

    steps: list[tuple[str, Optional[Path], Optional[str]]] = []
    if args.run_autopcp:
        steps.append(("autopcp", out_dir / "autopcp_result.txt", None))
    if args.run_vsim:
        steps.append(("vsim", out_dir / "vocal_style_sim_result.txt", None))
    if args.run_local_prosody:
        steps.append((f"{args.src_lang}_annot", out_dir / f"{args.src_lang}_speech_rate_pause_annotation.tsv", None))
        steps.append((f"{args.tgt_lang}_annot", out_dir / f"{args.tgt_lang}_speech_rate_pause_annotation.tsv", None))
        steps.append(("pause_compare", out_dir / f"{args.src_lang}_{args.tgt_lang}_pause_scores.tsv", None))

    print("\n[PLAN]")
    print(f"total metric stages: {len(steps)}")
    for i, (name, path, _) in enumerate(steps, 1):
        print(f"  {i}. {name} -> {path}")

    summary = {
        "input_tsv": str(Path(args.input_tsv).resolve()),
        "sample_frac": args.sample_frac,
        "sampled_rows": len(manifest),
        "sampled_raw_tsv": str(sampled_raw_path),
        "stopes_eval_manifest": str(manifest_path),
        "audio_pairs_noheader": str(pair_tsv),
        "src_lang": args.src_lang,
        "tgt_lang": args.tgt_lang,
        "num_shards": args.num_shards,
        "parallel_jobs": args.parallel_jobs,
        "keep_shard_files": args.keep_shard_files,
        "cpu_only": args.cpu_only,
        "autopcp_cpu_only": args.autopcp_cpu_only,
        "vsim_cpu_only": args.vsim_cpu_only,
    }

    progress = {
        "input_tsv": str(Path(args.input_tsv).resolve()),
        "out_dir": str(out_dir),
        "sampled_rows": len(manifest),
        "stages_total": len(steps),
        "stages": {},
    }
    write_json(progress, progress_path)

    stage_idx = 0

    if args.run_autopcp:
        stage_idx += 1
        autopcp_out = out_dir / "autopcp_result.txt"
        autopcp_force_cpu = args.cpu_only or args.autopcp_cpu_only
        use_cuda_flag = "false" if autopcp_force_cpu else "true"
        name = f"{stage_idx}/{len(steps)} autopcp"
        if args.num_shards <= 1:
            cmd = (
                f"{sys.executable} -m stopes.modules +compare_audios=AutoPCP_multilingual_v2 "
                f"launcher.cluster=local "
                f"{cache_override}"
                f"+compare_audios.input_file={manifest_path} "
                f"compare_audios.src_audio_column=src_audio "
                f"compare_audios.tgt_audio_column=hypo_audio "
                f"+compare_audios.named_columns=true "
                f"+compare_audios.output_file={autopcp_out} "
                f"compare_audios.use_cuda={use_cuda_flag}"
            )
            progress["stages"]["autopcp"] = run_cmd(
                cmd,
                env=stage_env(autopcp_force_cpu),
                name=name,
                output_path=autopcp_out,
            )
        else:
            shard_outs = [shard_dir / f"autopcp_result.shard{i:03d}.txt" for i in range(len(shard_manifest_paths))]
            jobs = []
            for i, (shard_manifest, shard_out) in enumerate(zip(shard_manifest_paths, shard_outs)):
                shard_name = f"{name}.shard{i:03d}"
                shard_cmd = (
                    f"{sys.executable} -m stopes.modules +compare_audios=AutoPCP_multilingual_v2 "
                    f"launcher.cluster=local "
                    f"{cache_override}"
                    f"+compare_audios.input_file={shard_manifest} "
                    f"compare_audios.src_audio_column=src_audio "
                    f"compare_audios.tgt_audio_column=hypo_audio "
                    f"+compare_audios.named_columns=true "
                    f"+compare_audios.output_file={shard_out} "
                    f"compare_audios.use_cuda={use_cuda_flag}"
                )
                jobs.append((shard_name, shard_cmd, shard_out))
            shard_results = run_cmds_parallel(
                jobs,
                env=stage_env(autopcp_force_cpu),
                max_parallel=args.parallel_jobs,
            )
            merge_text_outputs(shard_outs, autopcp_out)
            if not args.keep_shard_files:
                cleanup_paths(shard_outs)
            progress["stages"]["autopcp"] = {
                "status": "done",
                "output": str(autopcp_out),
                "sharded": True,
                "num_shards": len(shard_manifest_paths),
                "parallel_jobs": args.parallel_jobs,
                "shard_results": shard_results,
            }
        write_json(progress, progress_path)

        vals = extract_numeric_series_from_text(autopcp_out)
        pd.DataFrame({"autopcp": vals}).to_csv(out_dir / "autopcp_values.csv", index=False)

        summary["autopcp_file"] = str(autopcp_out)
        summary.update(summarize_scalar_distribution(vals, "autopcp"))

    if args.run_vsim:
        stage_idx += 1
        vsim_out = out_dir / "vocal_style_sim_result.txt"
        vsim_force_cpu = args.cpu_only or args.vsim_cpu_only
        vsim_use_cuda_flag = "false" if vsim_force_cpu else "true"
        name = f"{stage_idx}/{len(steps)} vsim"
        if args.num_shards <= 1:
            cmd = (
                f"{sys.executable} -m stopes.modules +vocal_style_similarity=base "
                f"launcher.cluster=local "
                f"{cache_override}"
                f"vocal_style_similarity.model_type=valle "
                f"+vocal_style_similarity.model_path={Path(args.wavlm_ckpt).resolve()} "
                f"+vocal_style_similarity.input_file={manifest_path} "
                f"+vocal_style_similarity.output_file={vsim_out} "
                f"vocal_style_similarity.named_columns=true "
                f"vocal_style_similarity.src_audio_column=src_audio "
                f"vocal_style_similarity.tgt_audio_column=hypo_audio "
                f"vocal_style_similarity.use_cuda={vsim_use_cuda_flag}"
            )
            progress["stages"]["vsim"] = run_cmd(
                cmd,
                env=stage_env(vsim_force_cpu),
                name=name,
                output_path=vsim_out,
            )
        else:
            shard_outs = [shard_dir / f"vocal_style_sim_result.shard{i:03d}.txt" for i in range(len(shard_manifest_paths))]
            jobs = []
            for i, (shard_manifest, shard_out) in enumerate(zip(shard_manifest_paths, shard_outs)):
                shard_name = f"{name}.shard{i:03d}"
                shard_cmd = (
                    f"{sys.executable} -m stopes.modules +vocal_style_similarity=base "
                    f"launcher.cluster=local "
                    f"{cache_override}"
                    f"vocal_style_similarity.model_type=valle "
                    f"+vocal_style_similarity.model_path={Path(args.wavlm_ckpt).resolve()} "
                    f"+vocal_style_similarity.input_file={shard_manifest} "
                    f"+vocal_style_similarity.output_file={shard_out} "
                    f"vocal_style_similarity.named_columns=true "
                    f"vocal_style_similarity.src_audio_column=src_audio "
                    f"vocal_style_similarity.tgt_audio_column=hypo_audio "
                    f"vocal_style_similarity.use_cuda={vsim_use_cuda_flag}"
                )
                jobs.append((shard_name, shard_cmd, shard_out))
            shard_results = run_cmds_parallel(
                jobs,
                env=stage_env(vsim_force_cpu),
                max_parallel=args.parallel_jobs,
            )
            merge_text_outputs(shard_outs, vsim_out)
            if not args.keep_shard_files:
                cleanup_paths(shard_outs)
            progress["stages"]["vsim"] = {
                "status": "done",
                "output": str(vsim_out),
                "sharded": True,
                "num_shards": len(shard_manifest_paths),
                "parallel_jobs": args.parallel_jobs,
                "shard_results": shard_results,
            }
        write_json(progress, progress_path)

        vals = extract_numeric_series_from_text(vsim_out)
        pd.DataFrame({"vsim": vals}).to_csv(out_dir / "vsim_values.csv", index=False)

        summary["vsim_file"] = str(vsim_out)
        summary.update(summarize_scalar_distribution(vals, "vsim"))

    if args.run_local_prosody:
        # CPU-only is useful for AutoPCP/VSim, but STOPES local prosody's
        # Silero VAD currently calls .cuda() internally. Keep CUDA visible for
        # local prosody while forcing the heavier aligner to CPU below.
        local_prosody_env = os.environ.copy()
        local_prosody_env["HYDRA_FULL_ERROR"] = "1"
        local_prosody_env["PYTHONPATH"] = env["PYTHONPATH"]
        local_prosody_env["TRANSFORMERS_ALLOW_UNSAFE_TORCH_LOAD_FOR_STOPES"] = "1"
        local_prosody_env["TMPDIR"] = str(stopes_tmp_dir)
        local_prosody_env["TEMP"] = str(stopes_tmp_dir)
        local_prosody_env["TMP"] = str(stopes_tmp_dir)

        src_annot = out_dir / f"{args.src_lang}_speech_rate_pause_annotation.tsv"
        tgt_annot = out_dir / f"{args.tgt_lang}_speech_rate_pause_annotation.tsv"
        src_annot_filtered = out_dir / f"{args.src_lang}_speech_rate_pause_annotation.filtered.tsv"
        tgt_annot_filtered = out_dir / f"{args.tgt_lang}_speech_rate_pause_annotation.filtered.tsv"
        compare_out = out_dir / f"{args.src_lang}_{args.tgt_lang}_pause_scores.tsv"
        speech_units_arg = shlex.quote(normalize_hydra_speech_units(args.speech_units))
        forced_aligner_arg = str(args.forced_aligner)
        if args.cpu_only and forced_aligner_arg == "ctc_wav2vec2-xlsr-multilingual-56":
            cpu_aligner_config = out_dir / "ctc_wav2vec2-xlsr-multilingual-56.cpu.yaml"
            cpu_aligner_config.write_text(
                "aligner_type: Wav2Vec2\n\n"
                "config:\n"
                "  model_name: voidful/wav2vec2-xlsr-multilingual-56\n"
                "  device: cpu\n"
            )
            forced_aligner_arg = str(cpu_aligner_config)
        forced_aligner_arg = shlex.quote(forced_aligner_arg)
        stage_idx += 1
        name = f"{stage_idx}/{len(steps)} {args.src_lang}_annot"
        if args.num_shards <= 1:
            cmd_src = (
                f"{sys.executable} {annotate_script} "
                f"{hydra_override}"
                f"+data_path={manifest_path} "
                f"+result_path={src_annot} "
                f"+audio_column=src_audio "
                f"+text_column=src_text "
                f"+speech_units={speech_units_arg} "
                f"+vad=true +net=true "
                f"+lang={args.src_lang} "
                f"+forced_aligner={forced_aligner_arg}"
            )
            progress["stages"][f"{args.src_lang}_annot"] = run_cmd(
                cmd_src, env=local_prosody_env, name=name, output_path=src_annot
            )
        else:
            src_annot_parts = [shard_dir / f"{args.src_lang}_annot.shard{i:03d}.tsv" for i in range(len(shard_manifest_paths))]
            jobs = []
            for i, (shard_manifest, shard_out) in enumerate(zip(shard_manifest_paths, src_annot_parts)):
                shard_name = f"{name}.shard{i:03d}"
                shard_cmd = (
                    f"{sys.executable} {annotate_script} "
                    f"{hydra_override}"
                    f"+data_path={shard_manifest} "
                    f"+result_path={shard_out} "
                    f"+audio_column=src_audio "
                    f"+text_column=src_text "
                    f"+speech_units={speech_units_arg} "
                    f"+vad=true +net=true "
                    f"+lang={args.src_lang} "
                    f"+forced_aligner={forced_aligner_arg}"
                )
                jobs.append((shard_name, shard_cmd, shard_out))
            shard_results = run_cmds_parallel(
                jobs,
                env=local_prosody_env,
                max_parallel=args.parallel_jobs,
                shard_dir=shard_dir,
                fallback_mode="annot",
            )
            merge_tsv_outputs(src_annot_parts, src_annot)
            src_failed_rows = int(sum(x.get("failed_rows", 0) for x in shard_results))
            progress["stages"][f"{args.src_lang}_annot"] = {
                "status": "partial" if src_failed_rows > 0 else "done",
                "output": str(src_annot),
                "sharded": True,
                "num_shards": len(shard_manifest_paths),
                "parallel_jobs": args.parallel_jobs,
                "failed_rows": src_failed_rows,
                "shard_results": shard_results,
            }
        write_json(progress, progress_path)
        stage_idx += 1
        name = f"{stage_idx}/{len(steps)} {args.tgt_lang}_annot"
        if args.num_shards <= 1:
            cmd_tgt = (
                f"{sys.executable} {annotate_script} "
                f"{hydra_override}"
                f"+data_path={manifest_path} "
                f"+result_path={tgt_annot} "
                f"+audio_column=hypo_audio "
                f"+text_column=s2t_out "
                f"+speech_units={speech_units_arg} "
                f"+vad=true +net=true "
                f"+lang={args.tgt_lang} "
                f"+forced_aligner={forced_aligner_arg}"
            )
            progress["stages"][f"{args.tgt_lang}_annot"] = run_cmd(
                cmd_tgt, env=local_prosody_env, name=name, output_path=tgt_annot
            )
        else:
            tgt_annot_parts = [shard_dir / f"{args.tgt_lang}_annot.shard{i:03d}.tsv" for i in range(len(shard_manifest_paths))]
            jobs = []
            for i, (shard_manifest, shard_out) in enumerate(zip(shard_manifest_paths, tgt_annot_parts)):
                shard_name = f"{name}.shard{i:03d}"
                shard_cmd = (
                    f"{sys.executable} {annotate_script} "
                    f"{hydra_override}"
                    f"+data_path={shard_manifest} "
                    f"+result_path={shard_out} "
                    f"+audio_column=hypo_audio "
                    f"+text_column=s2t_out "
                    f"+speech_units={speech_units_arg} "
                    f"+vad=true +net=true "
                    f"+lang={args.tgt_lang} "
                    f"+forced_aligner={forced_aligner_arg}"
                )
                jobs.append((shard_name, shard_cmd, shard_out))
            shard_results = run_cmds_parallel(
                jobs,
                env=local_prosody_env,
                max_parallel=args.parallel_jobs,
                shard_dir=shard_dir,
                fallback_mode="annot",
            )
            merge_tsv_outputs(tgt_annot_parts, tgt_annot)
            tgt_failed_rows = int(sum(x.get("failed_rows", 0) for x in shard_results))
            progress["stages"][f"{args.tgt_lang}_annot"] = {
                "status": "partial" if tgt_failed_rows > 0 else "done",
                "output": str(tgt_annot),
                "sharded": True,
                "num_shards": len(shard_manifest_paths),
                "parallel_jobs": args.parallel_jobs,
                "failed_rows": tgt_failed_rows,
                "shard_results": shard_results,
            }
        write_json(progress, progress_path)

        if args.num_shards <= 1:
            filter_stats = filter_bad_local_prosody_rows(
                src_path=src_annot,
                tgt_path=tgt_annot,
                src_filtered_path=src_annot_filtered,
                tgt_filtered_path=tgt_annot_filtered,
            )
            compare_src_parts = [src_annot_filtered]
            compare_tgt_parts = [tgt_annot_filtered]
        else:
            src_filtered_parts = [shard_dir / f"{args.src_lang}_annot.filtered.shard{i:03d}.tsv" for i in range(len(shard_manifest_paths))]
            tgt_filtered_parts = [shard_dir / f"{args.tgt_lang}_annot.filtered.shard{i:03d}.tsv" for i in range(len(shard_manifest_paths))]
            stats_list = []
            compare_src_parts = []
            compare_tgt_parts = []
            for src_part, tgt_part, src_filtered_part, tgt_filtered_part in zip(
                src_annot_parts, tgt_annot_parts, src_filtered_parts, tgt_filtered_parts
            ):
                if not file_nonempty(src_part) or not file_nonempty(tgt_part):
                    continue
                stats_list.append(
                    filter_bad_local_prosody_rows(
                        src_path=src_part,
                        tgt_path=tgt_part,
                        src_filtered_path=src_filtered_part,
                        tgt_filtered_path=tgt_filtered_part,
                    )
                )
                if file_nonempty(src_filtered_part) and file_nonempty(tgt_filtered_part):
                    compare_src_parts.append(src_filtered_part)
                    compare_tgt_parts.append(tgt_filtered_part)
            merge_tsv_outputs(src_filtered_parts, src_annot_filtered)
            merge_tsv_outputs(tgt_filtered_parts, tgt_annot_filtered)
            filter_stats = {
                "local_prosody_src_rows_raw": int(sum(x["local_prosody_src_rows_raw"] for x in stats_list)),
                "local_prosody_tgt_rows_raw": int(sum(x["local_prosody_tgt_rows_raw"] for x in stats_list)),
                "local_prosody_rows_kept": int(sum(x["local_prosody_rows_kept"] for x in stats_list)),
                "local_prosody_rows_dropped": int(sum(x["local_prosody_rows_dropped"] for x in stats_list)),
                "local_prosody_src_invalid_rows": int(sum(x["local_prosody_src_invalid_rows"] for x in stats_list)),
                "local_prosody_tgt_invalid_rows": int(sum(x["local_prosody_tgt_invalid_rows"] for x in stats_list)),
            }
        progress["stages"]["local_prosody_filter"] = {
            "status": "done",
            "src_filtered": str(src_annot_filtered),
            "tgt_filtered": str(tgt_annot_filtered),
            **filter_stats,
        }
        write_json(progress, progress_path)
        stage_idx += 1
        name = f"{stage_idx}/{len(steps)} pause_compare"
        if args.num_shards <= 1:
            cmd_cmp = (
                f"{sys.executable} {compare_script} "
                f"{hydra_override}"
                f"+src_path={src_annot_filtered} "
                f"+tgt_path={tgt_annot_filtered} "
                f"+result_path={compare_out} "
                f"+pause_min_duration={args.pause_min_duration}"
            )
            progress["stages"]["pause_compare"] = run_cmd(
                cmd_cmp, env=env, name=name, output_path=compare_out
            )
        else:
            compare_parts = [shard_dir / f"pause_compare.shard{i:03d}.tsv" for i in range(len(compare_src_parts))]
            jobs = []
            for i, (src_part, tgt_part, cmp_out) in enumerate(zip(compare_src_parts, compare_tgt_parts, compare_parts)):
                shard_name = f"{name}.shard{i:03d}"
                shard_cmd = (
                    f"{sys.executable} {compare_script} "
                    f"{hydra_override}"
                    f"+src_path={src_part} "
                    f"+tgt_path={tgt_part} "
                    f"+result_path={cmp_out} "
                    f"+pause_min_duration={args.pause_min_duration}"
                )
                jobs.append((shard_name, shard_cmd, cmp_out))
            if jobs:
                shard_results = run_cmds_parallel(
                    jobs,
                    env=env,
                    max_parallel=args.parallel_jobs,
                    shard_dir=shard_dir,
                    fallback_mode="compare",
                )
                merge_tsv_outputs(compare_parts, compare_out)
                compare_failed_rows = int(sum(x.get("failed_rows", 0) for x in shard_results))
                progress["stages"]["pause_compare"] = {
                    "status": "partial" if compare_failed_rows > 0 else "done",
                    "output": str(compare_out),
                    "sharded": True,
                    "num_shards": len(compare_parts),
                    "parallel_jobs": args.parallel_jobs,
                    "failed_rows": compare_failed_rows,
                    "shard_results": shard_results,
                }
            else:
                progress["stages"]["pause_compare"] = {
                    "status": "failed",
                    "output": str(compare_out),
                    "sharded": True,
                    "num_shards": 0,
                    "parallel_jobs": args.parallel_jobs,
                    "failed_rows": 0,
                    "error": "No valid filtered shard pairs available for pause_compare.",
                }
        write_json(progress, progress_path)

        summary["local_prosody_src_annot"] = str(src_annot)
        summary["local_prosody_tgt_annot"] = str(tgt_annot)
        summary["local_prosody_src_annot_filtered"] = str(src_annot_filtered)
        summary["local_prosody_tgt_annot_filtered"] = str(tgt_annot_filtered)
        summary["local_prosody_compare"] = str(compare_out)
        summary.update(filter_stats)
        if args.num_shards > 1:
            summary["local_prosody_src_failed_rows"] = int(
                progress["stages"].get(f"{args.src_lang}_annot", {}).get("failed_rows", 0)
            )
            summary["local_prosody_tgt_failed_rows"] = int(
                progress["stages"].get(f"{args.tgt_lang}_annot", {}).get("failed_rows", 0)
            )
            summary["local_prosody_compare_failed_rows"] = int(
                progress["stages"].get("pause_compare", {}).get("failed_rows", 0)
            )

        summary.update(summarize_rate(src_annot_filtered, tgt_annot_filtered, out_dir))
        summary.update(summarize_pause(compare_out, out_dir))
        summary.update(summarize_dc_sc(manifest_path, src_annot_filtered, tgt_annot_filtered, out_dir))

        if args.num_shards > 1 and not args.keep_shard_files:
            cleanup_paths(
                [
                    *src_annot_parts,
                    *tgt_annot_parts,
                    *compare_src_parts,
                    *compare_tgt_parts,
                    *compare_parts,
                ]
            )

    summary_path = out_dir / "summary.json"
    summary_tsv_path = out_dir / "summary.tsv"

    write_json(summary, summary_path)
    write_summary_tsv(summary, summary_tsv_path)

    progress["summary_json"] = str(summary_path)
    progress["summary_tsv"] = str(summary_tsv_path)
    write_json(progress, progress_path)

    if args.num_shards > 1 and not args.keep_shard_files:
        cleanup_paths(shard_manifest_paths)

    print("\n[SUMMARY]\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote summary to: {summary_path}")
    print(f"Wrote summary TSV to: {summary_tsv_path}")
    print(f"Wrote progress to: {progress_path}")


if __name__ == "__main__":
    main()
    
    
"""
python -u \
VC-DUB/evaluation/scripts/impl/eval_stopes_switch.py \
  --input-tsv /path/to/eval_manifest.tsv \
  --out-dir /path/to/stopes_metrics \
  --sample-frac 1.0 \
  --seed 42 \
  --src-lang eng \
  --tgt-lang deu \
  --src-audio-col original_en_audio \
  --tgt-audio-col src_audio \
  --src-text-col src_text \
  --tgt-text-col s2t_out \
  --id-col sample_id \
  --run-autopcp \
  --run-vsim \
  --run-local-prosody \
  --speech-units "[word,syllable]" \
  --wavlm-ckpt /path/to/wavlm_large_finetune.pth
"""
