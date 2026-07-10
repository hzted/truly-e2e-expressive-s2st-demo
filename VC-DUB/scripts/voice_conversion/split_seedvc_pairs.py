import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-tsv', required=True)
    ap.add_argument('--shard-root', required=True)
    ap.add_argument('--num-shards', type=int, default=2)
    ap.add_argument('--output-subdir', default='vc_wavs')
    args = ap.parse_args()

    shard_root = Path(args.shard_root)
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_tsv_dir = shard_root / 'pair_tsvs'
    shard_tsv_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.input_tsv, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader, None)
        if header is None or [h.strip().lower() for h in header[:3]] != ['source', 'target', 'output']:
            raise ValueError(f'Expected header [source, target, output], got {header}')
        for row in reader:
            if row and len(row) >= 3:
                rows.append(row[:3])

    shard_files = []
    writers = []
    handles = []
    try:
        for shard_idx in range(args.num_shards):
            shard_name = f'shard{shard_idx:03d}'
            shard_dir = shard_root / shard_name
            shard_dir.mkdir(parents=True, exist_ok=True)
            tsv_path = shard_tsv_dir / f'{shard_name}.tsv'
            h = tsv_path.open('w', encoding='utf-8', newline='')
            w = csv.writer(h, delimiter='\t')
            w.writerow(['source', 'target', 'output'])
            handles.append(h)
            writers.append(w)
            shard_files.append(tsv_path)

        for idx, row in enumerate(rows):
            shard_idx = idx % args.num_shards
            shard_name = f'shard{shard_idx:03d}'
            src, tgt, out = row
            out_name = Path(out).name
            shard_out = shard_root / shard_name / args.output_subdir / out_name
            writers[shard_idx].writerow([src, tgt, str(shard_out)])
    finally:
        for h in handles:
            h.close()

    for p in shard_files:
        print(p)


if __name__ == '__main__':
    main()
