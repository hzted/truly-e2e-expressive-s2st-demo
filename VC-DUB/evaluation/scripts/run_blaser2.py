#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _wrapper_utils import copy_metric_file, run_command, write_dry_outputs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run BLASER 2.0 audio evaluation via the project implementation.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--verify-scripts-root", default="/export/fs06/hzhan276/Expressive_S2ST/verify_scripts")
    p.add_argument("--python", default="python")
    p.add_argument("--id-col", default="sample_id")
    p.add_argument("--source-audio-col", default="source_audio")
    p.add_argument("--hypo-audio-col", default="hypo_audio")
    p.add_argument("--reference-audio-col", default="reference_audio")
    p.add_argument("--reference-text-col", default="reference_translation")
    p.add_argument("--target-lang-col", default="target_lang")
    p.add_argument("--source-lang", default="eng")
    p.add_argument("--batch-size", default="32")
    p.add_argument("--device", default="cuda")
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
            {"blaser2_qe_audio": 3.5, "blaser2_ref": 3.5},
            "blaser2_audio_summary.json",
            "blaser2_per_example.tsv",
        )
        return
    script = Path(args.verify_scripts_root) / "eval_blaser2_audio.py"
    cmd = [
        args.python,
        str(script),
        "--manifest",
        args.manifest,
        "--output-dir",
        str(out_dir),
        "--id-col",
        args.id_col,
        "--source-audio-col",
        args.source_audio_col,
        "--hypo-audio-col",
        args.hypo_audio_col,
        "--reference-audio-col",
        args.reference_audio_col,
        "--reference-text-col",
        args.reference_text_col,
        "--target-lang-col",
        args.target_lang_col,
        "--source-lang",
        args.source_lang,
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
    ]
    run_command(cmd)
    copy_metric_file(out_dir / "blaser2_audio_scores.tsv", out_dir / "blaser2_per_example.tsv")


if __name__ == "__main__":
    main()
