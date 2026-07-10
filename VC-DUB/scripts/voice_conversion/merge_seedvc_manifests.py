import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard-root', required=True)
    ap.add_argument('--output-tsv', required=True)
    args = ap.parse_args()

    shard_root = Path(args.shard_root)
    out_path = Path(args.output_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    shard_manifest_paths = sorted(shard_root.glob('shard*/manifests/vc_manifest.tsv'))
    if not shard_manifest_paths:
        raise SystemExit(f'No shard manifests found under {shard_root}')

    header = None
    rows = []
    for mp in shard_manifest_paths:
        with mp.open('r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            shard_header = next(reader, None)
            if shard_header is None:
                continue
            if header is None:
                header = shard_header
            elif shard_header != header:
                raise ValueError(f'Header mismatch in {mp}')
            for row in reader:
                if row:
                    rows.append(row)

    def sort_key(row):
        try:
            return int(str(row[0]))
        except Exception:
            return str(row[0])

    rows.sort(key=sort_key)

    with out_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        writer.writerows(rows)

    print(out_path)
    print('rows', len(rows))
