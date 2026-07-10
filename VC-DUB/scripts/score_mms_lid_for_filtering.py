#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import soundfile as sf
import torch
import torchaudio
from tqdm.auto import tqdm
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification


PROGRESS_COLUMNS = [
    'row_id',
    'side',
    'audio_path',
    'expected_lang',
    'pred_lang',
    'pred_score',
    'top3_langs',
    'top3_scores',
    'match',
    'status',
    'error',
]

LANG_ALIASES = {
    'en': 'eng',
    'es': 'spa',
    'de': 'deu',
    'fr': 'fra',
    'it': 'ita',
    'pt': 'por',
    'nl': 'nld',
    'ca': 'cat',
    'gl': 'glg',
}


def parse_args():
    ap = argparse.ArgumentParser(description='Filter language-mismatched rows using facebook/mms-lid-126.')
    ap.add_argument('--input-tsv', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--id-col', default='sample_id')
    ap.add_argument('--src-audio-col', default='pre_src')
    ap.add_argument('--tgt-audio-col', default='pre_tgt')
    ap.add_argument('--expected-src-lang', default='eng')
    ap.add_argument('--expected-tgt-lang', default='deu')
    ap.add_argument('--model-id', default='facebook/mms-lid-126')
    ap.add_argument('--cache-dir', default='')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--rerun-errors', action='store_true')
    ap.add_argument('--write-filtered-manifest', action='store_true')
    return ap.parse_args()


def normalize_lang(v: str) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ''
    key = str(v).strip().lower()
    return LANG_ALIASES.get(key, key)


def truthy(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return False
    return str(v).strip().lower() in {'1', 'true', 'yes', 'y'}


def load_audio_16k(path: str):
    wav, sr = sf.read(path)
    wav = torch.tensor(wav, dtype=torch.float32)
    if wav.ndim == 2:
        wav = wav.mean(dim=1)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, 16000).squeeze(0)
    return wav.numpy()


def progress_key(row_id: str, side: str) -> str:
    return f'{row_id}::{side}'


def load_progress(progress_path: Path) -> Dict[str, Dict[str, str]]:
    if not progress_path.is_file() or progress_path.stat().st_size == 0:
        return {}
    df = pd.read_csv(progress_path, sep='\t', dtype=str).fillna('')
    return {progress_key(str(r['row_id']), str(r['side'])): r.to_dict() for _, r in df.iterrows()}


def append_progress(progress_path: Path, row: Dict[str, str]):
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not progress_path.exists() or progress_path.stat().st_size == 0
    out_df = pd.DataFrame([{k: row.get(k, '') for k in PROGRESS_COLUMNS}])
    out_df.to_csv(progress_path, sep='\t', index=False, mode='a', header=write_header)


def classify_batch(model, processor, audios: List, device: str):
    inputs = processor(audios, sampling_rate=16000, return_tensors='pt', padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        top_scores, top_ids = torch.topk(probs, k=3, dim=-1)
    return top_scores.cpu(), top_ids.cpu()


def build_pair_results(df: pd.DataFrame, progress_df: pd.DataFrame, args) -> pd.DataFrame:
    src = progress_df[progress_df['side'] == 'source'][['row_id', 'audio_path', 'pred_lang', 'pred_score', 'match', 'status', 'error']].rename(
        columns={
            'audio_path': args.src_audiocol_out,
            'pred_lang': 'src_pred_lang',
            'pred_score': 'src_pred_score',
            'match': 'src_match',
            'status': 'src_status',
            'error': 'src_error',
        }
    )
    tgt = progress_df[progress_df['side'] == 'target'][['row_id', 'audio_path', 'pred_lang', 'pred_score', 'match', 'status', 'error']].rename(
        columns={
            'audio_path': args.tgt_audiocol_out,
            'pred_lang': 'tgt_pred_lang',
            'pred_score': 'tgt_pred_score',
            'match': 'tgt_match',
            'status': 'tgt_status',
            'error': 'tgt_error',
        }
    )
    merged = df.copy()
    merged[args.id_col] = merged[args.id_col].astype(str)
    merged = merged.merge(src, left_on=args.id_col, right_on='row_id', how='left').drop(columns=['row_id'])
    merged = merged.merge(tgt, left_on=args.id_col, right_on='row_id', how='left').drop(columns=['row_id'])
    merged['src_match'] = merged['src_match'].map(truthy)
    merged['tgt_match'] = merged['tgt_match'].map(truthy)
    merged['both_match'] = merged['src_match'] & merged['tgt_match']
    merged['either_mismatch'] = (~merged['src_match']) | (~merged['tgt_match'])
    return merged


def main():
    args = parse_args()
    args.src_audiocol_out = args.src_audio_col + '_checked'
    args.tgt_audiocol_out = args.tgt_audio_col + '_checked'
    args.expected_src_lang = normalize_lang(args.expected_src_lang)
    args.expected_tgt_lang = normalize_lang(args.expected_tgt_lang)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / 'mms_lid_progress.tsv'
    results_path = out_dir / 'mms_lid_results.tsv'
    bad_rows_path = out_dir / 'mms_lid_bad_rows.tsv'
    good_rows_path = out_dir / 'mms_lid_good_rows.tsv'
    summary_json = out_dir / 'mms_lid_summary.json'
    summary_tsv = out_dir / 'mms_lid_summary.tsv'
    filtered_manifest = out_dir / 'vc_manifest_lang_filtered.tsv'
    canonical_pass_manifest = out_dir / 'lid_pass_manifest.tsv'

    df = pd.read_csv(args.input_tsv, sep='\t')
    required = [args.id_col, args.src_audio_col, args.tgt_audio_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    df[args.id_col] = df[args.id_col].astype(str)

    progress = load_progress(progress_path)
    work = []
    for _, row in df.iterrows():
        row_id = str(row[args.id_col])
        for side, audio_col, expected_lang in [
            ('source', args.src_audio_col, args.expected_src_lang),
            ('target', args.tgt_audio_col, args.expected_tgt_lang),
        ]:
            key = progress_key(row_id, side)
            prior = progress.get(key)
            if prior and prior.get('status') == 'done' and not args.rerun_errors:
                continue
            if prior and prior.get('status') == 'error' and not args.rerun_errors:
                continue
            work.append({
                'row_id': row_id,
                'side': side,
                'audio_path': str(row[audio_col]),
                'expected_lang': expected_lang,
            })

    model_kwargs = {}
    if args.cache_dir:
        model_kwargs['cache_dir'] = args.cache_dir
    processor = AutoFeatureExtractor.from_pretrained(args.model_id, **model_kwargs)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(args.model_id, **model_kwargs).to(args.device)
    model.eval()

    batch_starts = range(0, len(work), args.batch_size)
    batch_iter = tqdm(
        batch_starts,
        total=(len(work) + args.batch_size - 1) // args.batch_size,
        desc='MMS LID',
        unit='batch',
    )
    for batch_start in batch_iter:
        batch_items = work[batch_start: batch_start + args.batch_size]
        batch_audios = []
        batch_kept = []
        for item in batch_items:
            row = {
                'row_id': item['row_id'],
                'side': item['side'],
                'audio_path': item['audio_path'],
                'expected_lang': item['expected_lang'],
                'pred_lang': '',
                'pred_score': '',
                'top3_langs': '',
                'top3_scores': '',
                'match': '',
                'status': '',
                'error': '',
            }
            try:
                if not Path(item['audio_path']).is_file():
                    raise FileNotFoundError(item['audio_path'])
                audio = load_audio_16k(item['audio_path'])
                batch_audios.append(audio)
                batch_kept.append((item, row))
            except Exception as e:
                row['status'] = 'error'
                row['error'] = repr(e)
                append_progress(progress_path, row)

        if not batch_kept:
            continue

        try:
            top_scores, top_ids = classify_batch(model, processor, batch_audios, args.device)
            for i, ((item, row), scores, ids) in enumerate(zip(batch_kept, top_scores, top_ids)):
                langs = [model.config.id2label[int(x)] for x in ids.tolist()]
                vals = [float(x) for x in scores.tolist()]
                pred_lang = normalize_lang(langs[0])
                row['pred_lang'] = pred_lang
                row['pred_score'] = f'{vals[0]:.6f}'
                row['top3_langs'] = json.dumps(langs, ensure_ascii=False)
                row['top3_scores'] = json.dumps([round(v, 6) for v in vals])
                row['match'] = str(pred_lang == item['expected_lang'])
                row['status'] = 'done'
                append_progress(progress_path, row)
        except Exception as e:
            for item, row in batch_kept:
                row['status'] = 'error'
                row['error'] = repr(e)
                append_progress(progress_path, row)

    progress_df = pd.read_csv(progress_path, sep='\t', dtype=str).fillna('')
    progress_df = progress_df.sort_values(['row_id', 'side'])
    progress_df.to_csv(progress_path, sep='\t', index=False)

    pair_df = build_pair_results(df, progress_df, args)
    pair_df.to_csv(results_path, sep='\t', index=False)

    bad_df = pair_df[pair_df['either_mismatch']].copy()
    good_df = pair_df[pair_df['both_match']].copy()
    bad_df.to_csv(bad_rows_path, sep='\t', index=False)
    good_df.to_csv(good_rows_path, sep='\t', index=False)

    if args.write_filtered_manifest:
        good_df[df.columns].to_csv(filtered_manifest, sep='\t', index=False)
        good_df[df.columns].to_csv(canonical_pass_manifest, sep='\t', index=False)

    summary = {
        'input_tsv': args.input_tsv,
        'model_id': args.model_id,
        'device': args.device,
        'src_audio_col': args.src_audio_col,
        'tgt_audio_col': args.tgt_audio_col,
        'expected_src_lang': args.expected_src_lang,
        'expected_tgt_lang': args.expected_tgt_lang,
        'n_rows': int(len(df)),
        'n_audio_checks': int(len(progress_df)),
        'done_checks': int((progress_df['status'] == 'done').sum()),
        'error_checks': int((progress_df['status'] == 'error').sum()),
        'src_mismatch_rows': int((~pair_df['src_match']).sum()),
        'tgt_mismatch_rows': int((~pair_df['tgt_match']).sum()),
        'either_mismatch_rows': int(pair_df['either_mismatch'].sum()),
        'both_match_rows': int(pair_df['both_match'].sum()),
        'progress_tsv': str(progress_path),
        'results_tsv': str(results_path),
        'bad_rows_tsv': str(bad_rows_path),
        'good_rows_tsv': str(good_rows_path),
        'filtered_manifest_tsv': str(filtered_manifest) if args.write_filtered_manifest else '',
        'canonical_pass_manifest_tsv': str(canonical_pass_manifest) if args.write_filtered_manifest else '',
    }
    pd.DataFrame([summary]).to_csv(summary_tsv, sep='\t', index=False)
    with summary_json.open('w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

"""
source {USER_HOME}/.bashrc
conda activate stopes_eval_a100
export HF_HOME={USER_WORK_ROOT}/.cache/huggingface

python -u scripts/score_mms_lid_for_filtering.py \
  --input-tsv {WORK_ROOT}/manifests/preprocessed_pair_manifest.tsv \
  --out-dir {WORK_ROOT}/mms_lid \
  --id-col sample_id \
  --src-audio-col pre_src \
  --tgt-audio-col pre_tgt \
  --expected-src-lang eng \
  --expected-tgt-lang spa \
  --device cuda \
  --batch-size 16 \
  --write-filtered-manifest
"""
