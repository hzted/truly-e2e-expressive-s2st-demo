#!/usr/bin/env python3
"""Build SeedVC-style pair TSVs from a VC-DUB construction manifest.

Input manifests may be `*_metadata.tsv`, `*_vc.tsv`, or a selected stage-04
construction manifest. They are expected to contain cleaned source and target
utterance paths.
The output TSV has the columns expected by the bundled SeedVC batch runner:

id<TAB>source<TAB>target<TAB>output
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import pandas as pd


def read_tsv(path: Path) -> pd.DataFrame:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            return pd.read_csv(f, sep="\t", low_memory=False)
    return pd.read_csv(path, sep="\t", low_memory=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tsv", required=True, help="VC-DUB metadata/VC manifest, optionally .gz")
    parser.add_argument("--output-tsv", required=True, help="SeedVC pair TSV to write.")
    parser.add_argument("--output-audio-root", required=True, help="Directory for generated VC wavs.")
    parser.add_argument("--id-col", default="sample_id")
    parser.add_argument("--source-audio-col", default="pre_src")
    parser.add_argument("--target-audio-col", default="pre_tgt")
    parser.add_argument("--output-prefix", default="")
    parser.add_argument("--output-ext", default=".wav")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_tsv(Path(args.input_tsv))
    required = [args.id_col, args.source_audio_col, args.target_audio_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in split manifest: {missing}")

    out_audio_root = Path(args.output_audio_root)
    out_audio_root.mkdir(parents=True, exist_ok=True)
    out_tsv = Path(args.output_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, row in df.iterrows():
        sample_id = str(row[args.id_col])
        src = str(row[args.source_audio_col])
        tgt = str(row[args.target_audio_col])
        out_wav = out_audio_root / f"{args.output_prefix}{sample_id}{args.output_ext}"
        rows.append({"id": sample_id, "source": src, "target": tgt, "output": str(out_wav)})

    pd.DataFrame(rows).to_csv(out_tsv, sep="\t", index=False)
    print(f"Wrote SeedVC pair TSV: {out_tsv}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
