#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _wrapper_utils import load_manifest, run_command, write_dry_outputs


def default_impl_root() -> str:
    return str(Path(__file__).resolve().parent / "impl")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SLC, speech-rate, and pause metrics through eval_stopes_switch.py.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--implementation-root", default=default_impl_root())
    p.add_argument("--verify-scripts-root", default=None, help="Deprecated alias for --implementation-root.")
    p.add_argument("--python", default="python")
    p.add_argument("--id-col", default="sample_id")
    p.add_argument("--src-lang", default="eng")
    p.add_argument("--tgt-lang", default="spa")
    p.add_argument("--src-audio-col", default="source_audio")
    p.add_argument("--tgt-audio-col", default="hypo_audio")
    p.add_argument("--src-text-col", default="source_text")
    p.add_argument("--tgt-text-col", default="hypo_text")
    p.add_argument("--speech-units", default="[syllable]")
    p.add_argument("--forced-aligner", default="ctc_wav2vec2-xlsr-multilingual-56")
    p.add_argument("--num-shards", default="1")
    p.add_argument("--parallel-jobs", default="1")
    p.add_argument("--sample-frac", default="1.0")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def write_isochrony_per_example(manifest: str, out_dir: Path, id_col: str) -> None:
    sampled = out_dir / "sampled_raw.tsv"
    base_path = sampled if sampled.is_file() else Path(manifest)
    base = load_manifest(base_path)
    if id_col not in base.columns:
        raise ValueError(f"Missing ID column in {base_path}: {id_col}")
    per = base[[id_col]].rename(columns={id_col: "sample_id"}).copy()

    dc_sc_path = out_dir / "dc_sc_pairs.csv"
    if dc_sc_path.is_file():
        dc_sc = pd.read_csv(dc_sc_path)
        if "sample_id" in dc_sc.columns:
            keep_cols = ["sample_id"]
            for col in ["dc_score", "sc_score", "rate_src", "rate_tgt"]:
                if col in dc_sc.columns:
                    keep_cols.append(col)
            per = per.merge(dc_sc[keep_cols], on="sample_id", how="left")
            if "dc_score" in per.columns:
                dc = pd.to_numeric(per["dc_score"], errors="coerce")
                per["dc_0p2_compliance"] = (dc >= 0.8).astype("float64").where(dc.notna())
                per["dc_0p4_compliance"] = (dc >= 0.6).astype("float64").where(dc.notna())
            if "sc_score" in per.columns:
                sc = pd.to_numeric(per["sc_score"], errors="coerce")
                per["sc_0p2_compliance"] = (sc >= 0.8).astype("float64").where(sc.notna())
                per["sc_0p4_compliance"] = (sc >= 0.6).astype("float64").where(sc.notna())

    pause_path = out_dir / "pause_scores_copy.csv"
    if pause_path.is_file():
        pause = pd.read_csv(pause_path)
        pause_cols = [c for c in ["sample_id", "wmean_duration_score", "mean_duration_score"] if c in pause.columns]
        if "wmean_duration_score" in pause.columns:
            if "sample_id" in pause_cols:
                pause_part = pause[pause_cols].rename(columns={"wmean_duration_score": "pause_wmean_duration_score"})
                per = per.merge(pause_part, on="sample_id", how="left")
            elif len(pause) == len(per):
                per["pause_wmean_duration_score"] = pd.to_numeric(
                    pause["wmean_duration_score"], errors="coerce"
                ).reset_index(drop=True)

    per = per.rename(columns={"sample_id": id_col})
    per.to_csv(out_dir / "isochrony_per_example.tsv", sep="\t", index=False)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if str(args.num_shards) != "1":
        raise ValueError(
            "Multi-shard isochrony evaluation is disabled in the reviewer release "
            "until per-example ID ordering is fully audited. Use --num-shards 1."
        )
    if args.dry_run:
        write_dry_outputs(
            args.manifest,
            out_dir,
            args.id_col,
            {
                "dc_0p2_compliance": 0.5,
                "dc_0p4_compliance": 0.5,
                "sc_0p2_compliance": 0.5,
                "sc_0p4_compliance": 0.5,
                "speech_rate_syllable_spearman": 0.5,
                "pause_wmean_duration_score": 0.5,
            },
            "summary.json",
            "isochrony_per_example.tsv",
        )
        return
    impl_root = Path(args.verify_scripts_root or args.implementation_root)
    script = impl_root / "eval_stopes_switch.py"
    if not script.is_file():
        raise FileNotFoundError(f"Missing Stopes metric implementation: {script}")
    cmd = [
        args.python,
        str(script),
        "--input-tsv",
        args.manifest,
        "--out-dir",
        str(out_dir),
        "--src-lang",
        args.src_lang,
        "--tgt-lang",
        args.tgt_lang,
        "--src-audio-col",
        args.src_audio_col,
        "--tgt-audio-col",
        args.tgt_audio_col,
        "--src-text-col",
        args.src_text_col,
        "--tgt-text-col",
        args.tgt_text_col,
        "--id-col",
        args.id_col,
        "--run-local-prosody",
        "--speech-units",
        args.speech_units,
        "--forced-aligner",
        args.forced_aligner,
        "--num-shards",
        str(args.num_shards),
        "--parallel-jobs",
        str(args.parallel_jobs),
        "--sample-frac",
        str(args.sample_frac),
    ]
    run_command(cmd)
    write_isochrony_per_example(args.manifest, out_dir, args.id_col)


if __name__ == "__main__":
    main()
