#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_manifest(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, low_memory=False)


def write_json(data: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_dry_outputs(
    manifest: str | Path,
    out_dir: str | Path,
    id_col: str,
    metric_values: dict[str, float],
    summary_name: str,
    per_example_name: str,
) -> None:
    out = Path(out_dir)
    ensure_dir(out)
    df = load_manifest(manifest)
    if id_col not in df.columns:
        raise ValueError(f"Missing ID column for dry-run: {id_col}")
    per = df[[id_col]].copy()
    for key, value in metric_values.items():
        per[key] = value
    per.to_csv(out / per_example_name, sep="\t", index=False)
    summary = {f"{key}_mean": float(value) for key, value in metric_values.items()}
    summary.update({"num_rows": int(len(per)), "dry_run": True})
    write_json(summary, out / summary_name)
    pd.DataFrame([summary]).to_csv(out / summary_name.replace(".json", ".tsv"), sep="\t", index=False)


def run_command(cmd: list[str]) -> None:
    print("[CMD] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def copy_metric_file(src: Path, dst: Path) -> None:
    if src.is_file():
        ensure_dir(dst.parent)
        dst.write_text(src.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")


def sampled_ids_for_outputs(
    manifest: str | Path,
    out_dir: str | Path,
    id_col: str,
    expected_rows: int,
) -> pd.DataFrame:
    """Return IDs in the exact order consumed by the STOPES wrapper.

    eval_stopes_switch.py writes sampled_raw.tsv after applying --sample-frac.
    The metric text outputs preserve that sampled order, so mapping to the
    original manifest head would be wrong whenever sampling is enabled.
    """
    out = Path(out_dir)
    sampled = out / "sampled_raw.tsv"
    source = sampled if sampled.is_file() else Path(manifest)
    df = load_manifest(source)
    if id_col not in df.columns:
        raise ValueError(f"Missing ID column in {source}: {id_col}")
    ids = df[[id_col]].head(expected_rows).reset_index(drop=True)
    if len(ids) != expected_rows:
        raise ValueError(
            f"Metric output has {expected_rows} rows but {source} has {len(ids)} IDs."
        )
    return ids
