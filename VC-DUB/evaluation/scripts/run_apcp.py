#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _wrapper_utils import run_command, sampled_ids_for_outputs, write_dry_outputs


def default_impl_root() -> str:
    return str(Path(__file__).resolve().parent / "impl")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run AutoPCP through eval_stopes_switch.py.")
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
    p.add_argument("--num-shards", default="1")
    p.add_argument("--parallel-jobs", default="1")
    p.add_argument("--sample-frac", default="1.0")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.dry_run:
        write_dry_outputs(args.manifest, out_dir, args.id_col, {"autopcp": 0.5}, "summary.json", "apcp_per_example.tsv")
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
        "--run-autopcp",
        "--num-shards",
        str(args.num_shards),
        "--parallel-jobs",
        str(args.parallel_jobs),
        "--sample-frac",
        str(args.sample_frac),
    ]
    run_command(cmd)
    values = out_dir / "autopcp_values.csv"
    if values.is_file():
        vals = pd.read_csv(values)
        ids = sampled_ids_for_outputs(args.manifest, out_dir, args.id_col, len(vals))
        pd.concat([ids.reset_index(drop=True), vals.reset_index(drop=True)], axis=1).to_csv(
            out_dir / "apcp_per_example.tsv", sep="\t", index=False
        )


if __name__ == "__main__":
    main()
