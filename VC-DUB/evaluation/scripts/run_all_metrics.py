#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run(cmd: list[str]) -> None:
    print("[CMD] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def base_args(args: argparse.Namespace, out_dir: Path, metric_subdir: str) -> list[str]:
    return [
        "--manifest",
        args.manifest,
        "--out-dir",
        str(out_dir / metric_subdir),
        "--id-col",
        args.id_col,
    ]


def default_impl_root() -> str:
    return str(Path(__file__).resolve().parent / "impl")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run all selected VC-DUB paper evaluation metrics.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--config", default="evaluation/configs/evaluation_config.json")
    p.add_argument("--python", default="python")
    p.add_argument("--implementation-root", default=default_impl_root())
    p.add_argument("--verify-scripts-root", default=None, help="Deprecated alias for --implementation-root.")
    p.add_argument("--id-col", default="sample_id")
    p.add_argument("--source-lang", default="")
    p.add_argument("--hypo-lang", default="")
    p.add_argument("--wavlm-ckpt", default="", help="Required for real Vsim evaluation.")
    p.add_argument("--dnsmospro-cmd", default="")
    p.add_argument("--dnsmospro-score-key", default="")
    p.add_argument("--dnsmospro-score-regex", default="")
    p.add_argument("--num-shards", default="1")
    p.add_argument("--parallel-jobs", default="1")
    p.add_argument("--uncertainty", choices=["none", "std", "sem", "ci95"], default="none")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(Path(args.config))
    source_lang = args.source_lang or config.get("source_lang", "eng")
    hypo_lang = args.hypo_lang or config.get("hypo_lang", "spa")
    impl_root = args.verify_scripts_root or args.implementation_root

    common_stopes = [
        "--implementation-root",
        impl_root,
        "--python",
        args.python,
        "--src-lang",
        source_lang,
        "--tgt-lang",
        hypo_lang,
        "--num-shards",
        str(args.num_shards),
        "--parallel-jobs",
        str(args.parallel_jobs),
    ]
    dry = ["--dry-run"] if args.dry_run else []

    metrics = config.get("metrics", {})
    if metrics.get("blaser2", {}).get("enabled", False):
        run([
            args.python,
            str(root / "scripts" / "run_blaser2.py"),
            *base_args(args, out_dir, "blaser2_audio"),
            "--implementation-root",
            impl_root,
            "--python",
            args.python,
            "--source-lang",
            source_lang,
            *dry,
        ])
    if metrics.get("apcp", {}).get("enabled", False):
        run([args.python, str(root / "scripts" / "run_apcp.py"), *base_args(args, out_dir, "apcp"), *common_stopes, *dry])
    if metrics.get("isochrony", {}).get("enabled", False):
        run([args.python, str(root / "scripts" / "run_isochrony_metrics.py"), *base_args(args, out_dir, "isochrony"), *common_stopes, *dry])
    if metrics.get("vsim", {}).get("enabled", False):
        if not args.wavlm_ckpt and not args.dry_run:
            raise ValueError("Real Vsim evaluation requires --wavlm-ckpt.")
        run([
            args.python,
            str(root / "scripts" / "run_vsim.py"),
            *base_args(args, out_dir, "vsim"),
            *common_stopes,
            "--wavlm-ckpt",
            args.wavlm_ckpt,
            *dry,
        ])
    if metrics.get("dnsmospro", {}).get("enabled", False):
        if not args.dnsmospro_cmd and not args.dry_run:
            raise ValueError("Real DNSMOSPro evaluation requires --dnsmospro-cmd; dry-run is the only mode that may use a placeholder.")
        if not args.dry_run and not (args.dnsmospro_score_key or args.dnsmospro_score_regex):
            raise ValueError("Real DNSMOSPro evaluation requires --dnsmospro-score-key or --dnsmospro-score-regex.")
        dnsmos_cmd = args.dnsmospro_cmd or "echo 3.5"
        dnsmos_parse_args = []
        if args.dnsmospro_score_key:
            dnsmos_parse_args += ["--score-key", args.dnsmospro_score_key]
        if args.dnsmospro_score_regex:
            dnsmos_parse_args += ["--score-regex", args.dnsmospro_score_regex]
        run([
            args.python,
            str(root / "scripts" / "run_dnsmospro.py"),
            *base_args(args, out_dir, "dnsmospro"),
            "--dnsmospro-cmd",
            dnsmos_cmd,
            *dnsmos_parse_args,
            *dry,
        ])
    if metrics.get("whisper_asr", {}).get("enabled", False):
        run([args.python, str(root / "scripts" / "run_whisper_asr.py"), *base_args(args, out_dir, "whisper_asr"), "--enabled", *dry])

    run([
        args.python,
        str(root / "scripts" / "aggregate_results.py"),
        "--eval-root",
        str(out_dir),
        "--manifest",
        args.manifest,
        "--id-col",
        args.id_col,
        "--uncertainty",
        args.uncertainty,
    ])


if __name__ == "__main__":
    main()
