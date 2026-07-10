#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _wrapper_utils import run_command, write_dry_outputs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SLC, speech-rate, and pause metrics through eval_stopes_switch.py.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--verify-scripts-root", default="/export/fs06/hzhan276/Expressive_S2ST/verify_scripts")
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
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.dry_run:
        write_dry_outputs(
            args.manifest,
            out_dir,
            args.id_col,
            {
                "dc_0p2_compliance": 0.5,
                "dc_0p4_compliance": 0.5,
                "speech_rate_syllable_spearman": 0.5,
                "pause_wmean_duration_score": 0.5,
            },
            "summary.json",
            "isochrony_per_example.tsv",
        )
        return
    script = Path(args.verify_scripts_root) / "eval_stopes_switch.py"
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
    ]
    run_command(cmd)


if __name__ == "__main__":
    main()
