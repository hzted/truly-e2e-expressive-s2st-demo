#!/usr/bin/env python3
"""Collect path-sanitized VC-DUB construction manifests for anonymous release."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXP_ROOT = Path(os.environ.get("EXPRESSIVE_S2ST_ROOT", "{EXPRESSIVE_S2ST_ROOT}"))
ALIGNED_DUBBING_ROOT = Path(os.environ.get("ALIGNED_DUBBING_ROOT", "{ALIGNED_DUBBING_ROOT}"))

LANG_ROOTS = {
    "en_es": {
        "work_root": Path(os.environ.get("VC_DUB_EN_ES_WORK_ROOT", str(EXP_ROOT / "en_es" / "work"))),
        "split_root": Path(os.environ.get("VC_DUB_EN_ES_SPLIT_ROOT", str(EXP_ROOT / "en_es" / "splits"))),
        "release_subdir": "en_es",
    },
    "en_de": {
        "work_root": Path(os.environ.get("VC_DUB_EN_DE_WORK_ROOT", str(EXP_ROOT / "en_de" / "work"))),
        "split_root": Path(os.environ.get("VC_DUB_EN_DE_SPLIT_ROOT", str(EXP_ROOT / "en_de" / "splits"))),
        "release_subdir": "en_de",
    },
}


@dataclass(frozen=True)
class ReleaseFile:
    language_pair: str
    stage: str
    source: Path
    target_rel: Path
    kind: str  # tsv, json, plain
    required: bool = True


def build_specs() -> list[ReleaseFile]:
    specs: list[ReleaseFile] = []
    for pair, roots in LANG_ROOTS.items():
        work = roots["work_root"]
        split = roots["split_root"]
        rel = Path(roots["release_subdir"])

        specs.extend(
            [
                ReleaseFile(pair, "aligned_pair_manifest", work / "manifests" / "aligned_pair_manifest.tsv", rel / "filtering" / "stage_00_aligned_pair_manifest.tsv.gz", "tsv"),
                ReleaseFile(pair, "mms_lid_pass", work / "mms_lid" / "lid_pass_manifest.tsv", rel / "filtering" / "stage_01_mms_lid_pass_manifest.tsv.gz", "tsv"),
                ReleaseFile(pair, "sortformer_single_speaker_pass", work / "sortformer" / "sortformer_pair_pass_strict.tsv", rel / "filtering" / "stage_02_sortformer_single_speaker_pass.tsv.gz", "tsv"),
                ReleaseFile(pair, "scale_matched_quality_selection", work / "quality_selection" / "dnsmospro_filtered_manifest.tsv", rel / "filtering" / "stage_03_dnsmospro_quality_selected_manifest.tsv.gz", "tsv"),
                ReleaseFile(pair, "mms_lid_summary", work / "mms_lid" / "mms_lid_summary.json", rel / "summaries" / "mms_lid_summary.json", "json", required=False),
                ReleaseFile(pair, "sortformer_summary", work / "sortformer" / "sortformer_pair_summary.json", rel / "summaries" / "sortformer_summary.json", "json", required=False),
                ReleaseFile(pair, "dnsmospro_quality_summary", work / "quality_selection" / "dnsmospro_quality_summary.json", rel / "summaries" / "dnsmospro_quality_summary.json", "json", required=False),
            ]
        )

        for split_name in ("train", "dev", "test"):
            specs.extend(
                [
                    ReleaseFile(pair, f"{split_name}_metadata_split", split / f"{split_name}_metadata.tsv", rel / "splits" / f"{split_name}_metadata.tsv.gz", "tsv"),
                    ReleaseFile(pair, f"{split_name}_vc_split", split / f"{split_name}_vc.tsv", rel / "splits" / f"{split_name}_vc.tsv.gz", "tsv", required=False),
                ]
            )
        specs.extend(
            [
                ReleaseFile(pair, "all_metadata", split / "all_metadata.tsv", rel / "splits" / "all_metadata.tsv.gz", "tsv", required=False),
                ReleaseFile(pair, "dnsmospro_quality_pairs", work / "quality_selection" / "dnsmospro_quality_pairs.tsv", rel / "filtering" / "dnsmospro_quality_pairs.tsv.gz", "tsv", required=False),
                ReleaseFile(pair, "split_summary", split / "split_summary.json", rel / "summaries" / "split_summary.json", "json", required=False),
            ]
        )

    stats_root = EXP_ROOT / "data_stats" / "vcdub_filtering_stage_stats"
    specs.extend(
        [
            ReleaseFile("all", "filtering_stage_stats", stats_root / "vcdub_filtering_stage_stats.tsv", Path("global") / "vcdub_filtering_stage_stats.tsv.gz", "tsv", required=False),
            ReleaseFile("all", "filtering_stage_stats_json", stats_root / "vcdub_filtering_stage_stats.json", Path("global") / "vcdub_filtering_stage_stats.json", "json", required=False),
        ]
    )

    threshold_root = EXP_ROOT / "threshold_samples" / "dnsmospro_critical"
    for name in (
        "all_dnsmospro_threshold_critical_40.tsv",
        "en_es_dnsmospro_threshold_critical_20.tsv",
        "en_es_dnsmospro_threshold_critical_20_audio.tsv",
        "en_es_dnsmospro_threshold_critical_summary.tsv",
        "en_de_dnsmospro_threshold_critical_20.tsv",
        "en_de_dnsmospro_threshold_critical_20_audio.tsv",
        "en_de_dnsmospro_threshold_critical_summary.tsv",
    ):
        specs.append(
            ReleaseFile("all", "threshold_samples", threshold_root / name, Path("global") / "threshold_samples" / f"{name}.gz", "tsv", required=False)
        )
    return specs


def replacement_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = [
        (str(EXP_ROOT / "en_es" / "work"), "{VC_DUB_ROOT}/en_es/work"),
        (str(EXP_ROOT / "en_de" / "work"), "{VC_DUB_ROOT}/en_de/work"),
        (str(EXP_ROOT / "en_es" / "splits"), "{VC_DUB_MANIFEST_ROOT}/en_es/splits"),
        (str(EXP_ROOT / "en_de" / "splits"), "{VC_DUB_MANIFEST_ROOT}/en_de/splits"),
        (str(ALIGNED_DUBBING_ROOT / "en_es"), "{ALIGNED_DUBBING_ROOT}/en_es"),
        (str(ALIGNED_DUBBING_ROOT / "en_de"), "{ALIGNED_DUBBING_ROOT}/en_de"),
        (str(ALIGNED_DUBBING_ROOT), "{ALIGNED_DUBBING_ROOT}"),
        (str(EXP_ROOT), "{EXPRESSIVE_S2ST_ROOT}"),
        ("{USER_WORK_ROOT}", "{USER_WORK_ROOT}"),
        ("{USER_HOME}", "{USER_HOME}"),
    ]
    # Longest paths first prevents a short prefix from masking a more precise path.
    return sorted(pairs, key=lambda x: len(x[0]), reverse=True)


def sanitize_text(text: str, replacements: list[tuple[str, str]]) -> str:
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def sanitize_obj(obj: Any, replacements: list[tuple[str, str]]) -> Any:
    if isinstance(obj, str):
        return sanitize_text(obj, replacements)
    if isinstance(obj, list):
        return [sanitize_obj(x, replacements) for x in obj]
    if isinstance(obj, dict):
        return {str(k): sanitize_obj(v, replacements) for k, v in obj.items()}
    return obj


def copy_tsv_gz(src: Path, dst: Path, replacements: list[tuple[str, str]]) -> tuple[int, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with src.open("rt", encoding="utf-8", errors="replace", newline="") as fin:
        with gzip.open(dst, "wt", encoding="utf-8", newline="") as fout:
            for line in fin:
                fout.write(sanitize_text(line, replacements))
                rows += 1
    return max(rows - 1, 0), dst.stat().st_size


def copy_json(src: Path, dst: Path, replacements: list[tuple[str, str]]) -> tuple[int, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as fin:
        data = json.load(fin)
    with dst.open("w", encoding="utf-8") as fout:
        json.dump(sanitize_obj(data, replacements), fout, ensure_ascii=False, indent=2, sort_keys=True)
        fout.write("\n")
    return 1, dst.stat().st_size


def copy_plain(src: Path, dst: Path, replacements: list[tuple[str, str]]) -> tuple[int, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = sanitize_text(src.read_text(encoding="utf-8", errors="replace"), replacements)
    dst.write_text(text, encoding="utf-8")
    return len(text.splitlines()), dst.stat().st_size


def write_inventory(records: list[dict[str, Any]], output_root: Path) -> None:
    out = output_root / "manifest_inventory.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "language_pair",
        "stage",
        "source_path",
        "release_path",
        "kind",
        "required",
        "status",
        "rows_or_items",
        "bytes",
        "note",
    ]
    with out.open("w", encoding="utf-8", newline="") as f:
        f.write("\t".join(fields) + "\n")
        for rec in records:
            f.write("\t".join(str(rec.get(field, "")) for field in fields) + "\n")

    out_json = output_root / "manifest_inventory.json"
    out_json.write_text(json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("manifests"))
    parser.add_argument("--skip-existing", action="store_true", help="Do not rewrite already packaged release files.")
    parser.add_argument("--fail-on-missing-required", action="store_true", default=True)
    parser.add_argument("--no-fail-on-missing-required", dest="fail_on_missing_required", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = args.output_root
    reps = replacement_pairs()
    records: list[dict[str, Any]] = []
    missing_required: list[Path] = []

    for spec in build_specs():
        dst = out_root / spec.target_rel
        rec: dict[str, Any] = {
            "language_pair": spec.language_pair,
            "stage": spec.stage,
            "source_path": str(spec.source),
            "release_path": str(dst),
            "kind": spec.kind,
            "required": spec.required,
            "status": "pending",
            "rows_or_items": "",
            "bytes": "",
            "note": "",
        }
        if not spec.source.exists():
            rec["status"] = "missing"
            rec["note"] = "missing optional file" if not spec.required else "missing required file"
            if spec.required:
                missing_required.append(spec.source)
            records.append(rec)
            continue

        if args.skip_existing and dst.exists():
            rec["status"] = "existing"
            rec["bytes"] = dst.stat().st_size
            rec["note"] = "kept existing packaged file"
            records.append(rec)
            continue

        if spec.kind == "tsv":
            rows, size = copy_tsv_gz(spec.source, dst, reps)
        elif spec.kind == "json":
            rows, size = copy_json(spec.source, dst, reps)
        else:
            rows, size = copy_plain(spec.source, dst, reps)
        rec["status"] = "written"
        rec["rows_or_items"] = rows
        rec["bytes"] = size
        records.append(rec)
        print(f"[written] {dst} ({size / 1024 / 1024:.2f} MiB)")

    write_inventory(records, out_root)
    print(f"Wrote inventory: {out_root / 'manifest_inventory.tsv'}")

    if missing_required and args.fail_on_missing_required:
        print("Missing required files:")
        for path in missing_required:
            print(f"  {path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
