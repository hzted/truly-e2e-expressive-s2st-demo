#!/usr/bin/env python3
"""Optional Whisper-large-v3 ASR evaluation entry point.

This is deliberately disabled unless the caller passes --enabled. It is not a
VC-DUB construction dependency.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _wrapper_utils import run_command


def default_impl_script() -> str:
    return str(Path(__file__).resolve().parent / "impl" / "transcribe_whisper_largev3.py")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--id-col", default="sample_id")
    p.add_argument("--implementation-script", default=default_impl_script())
    p.add_argument("--python", default="python")
    p.add_argument("--source-audio-col", default="source_audio")
    p.add_argument("--hypo-audio-col", default="hypo_audio")
    p.add_argument("--asr-source-field", default="", help="Defaults to --source-audio-col when omitted.")
    p.add_argument("--asr-hypo-field", default="", help="Defaults to --hypo-audio-col when omitted.")
    p.add_argument("--source-language", default="english")
    p.add_argument("--hypo-language", default="spanish")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", default="16")
    p.add_argument("--chunk-length-s", default="0")
    p.add_argument("--max-new-tokens", default="440")
    p.add_argument("--skip-missing-audio", action="store_true")
    p.add_argument("--rerun-errors", action="store_true")
    p.add_argument("--enabled", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.enabled:
        (out_dir / "WHISPER_ASR_DISABLED.txt").write_text(
            "Whisper ASR was not enabled. This is expected unless ASR-based evaluation is reported.\n",
            encoding="utf-8",
        )
        return
    df = pd.read_csv(args.manifest, sep="\t", dtype=str, keep_default_na=False, low_memory=False)
    if args.id_col not in df.columns:
        raise ValueError(f"Missing ID column: {args.id_col}")
    for col in [args.source_audio_col, args.hypo_audio_col]:
        if col not in df.columns:
            raise ValueError(f"Missing audio column: {col}")
    if args.dry_run:
        out = df[[args.id_col]].copy()
        out["whisper_transcript"] = "<dry-run transcript>"
        out.to_csv(out_dir / "whisper_transcripts.tsv", sep="\t", index=False)
        meta = pd.DataFrame(
            {
                "path": df[args.id_col],
                "sentence": "<dry-run source transcript>",
                "translation": "<dry-run hypothesis transcript>",
                "src_audio": df[args.source_audio_col],
                "tgt_audio": df[args.hypo_audio_col],
            }
        )
        meta.to_csv(out_dir / "transvip_text_meta.tsv", sep="\t", index=False)
        return
    script = Path(args.implementation_script)
    if not script.is_file():
        raise FileNotFoundError(f"Missing Whisper implementation: {script}")
    normalized_manifest = out_dir / "whisper_input_manifest.tsv"
    normalized = df.copy()
    normalized["id"] = normalized[args.id_col]
    if "status" not in normalized.columns:
        normalized["status"] = "ok"
    normalized.to_csv(normalized_manifest, sep="\t", index=False)
    asr_source_field = args.asr_source_field or args.source_audio_col
    asr_hypo_field = args.asr_hypo_field or args.hypo_audio_col
    cmd = [
        args.python,
        str(script),
        "--manifest",
        str(normalized_manifest),
        "--out-dir",
        str(out_dir),
        "--src-audio-field",
        args.source_audio_col,
        "--tgt-audio-field",
        args.hypo_audio_col,
        "--asr-src-field",
        asr_source_field,
        "--asr-tgt-field",
        asr_hypo_field,
        "--src-language",
        args.source_language,
        "--tgt-language",
        args.hypo_language,
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--chunk-length-s",
        str(args.chunk_length_s),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.skip_missing_audio:
        cmd.append("--skip-missing-audio")
    if args.rerun_errors:
        cmd.append("--rerun-errors")
    run_command(cmd)


if __name__ == "__main__":
    main()
