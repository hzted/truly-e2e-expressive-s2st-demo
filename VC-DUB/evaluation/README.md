# Evaluation utilities

Scripts for scoring generated speech against its source. They are separate from
VC-Dub construction and are not used to filter or split construction data.

## Metrics

### Used in the paper (Table 3)

| Paper column | Metric | In this package |
| --- | --- | --- |
| ASR-BLEU | Whisper-large-v3 transcripts scored with SacreBLEU | Transcription only (`run_whisper_asr.py`); BLEU scoring is not included |
| BLASER | BLASER 2.0-QE | `BLASER2_QE` |
| A.PCP | AutoPCP (STOPES) | `A_PCP` |
| Rate | Spearman correlation of source and target syllable speech rates (STOPES) | `SpeechRate` |
| Pause | Pause alignment (STOPES): per-utterance duration score weighted by pause length (`wmean_duration_score`), averaged over utterances | `Pause` |
| Vsim | Vocal-style similarity (STOPES, WavLM) | `Vsim` |
| NAT | NISQA-TTS | Not included |

DNSMOSPro appears in the paper only as a speech-quality score of the training
data (Table 2); it is not the NAT column.

### Additionally supported

- `BLASER2_ref`: reference-based BLASER 2.0
- `SLC_0p2`, `SLC_0p4`: duration speech-length compliance
- `DNSMOSPro_Nat`: DNSMOSPro naturalness

### Not included

NISQA-TTS scoring and ASR-BLEU scoring (BLEU, WER, CER).

## Input manifest

A single TSV keyed by `sample_id`; see `examples/manifest_schema.md`.

Required columns:

```text
sample_id
source_audio
hypo_audio
source_text
hypo_text
source_lang
hypo_lang
target_lang
status
```

Optional columns:

```text
reference_audio
reference_text
reference_translation
```

## Smoke test

From the `VC-DUB` directory:

```bash
bash evaluation/tests/test_smoke.sh
```

This runs with `--dry-run`: it checks the command plumbing and aggregation, not
metric values, and needs no checkpoints or audio.

## Running

From the `VC-DUB` directory:

```bash
python -u evaluation/scripts/run_all_metrics.py \
  --manifest /path/to/eval_manifest.tsv \
  --out-dir /path/to/evaluation_outputs \
  --config evaluation/configs/evaluation_config.json \
  --python /path/to/python \
  --implementation-root evaluation/scripts/impl \
  --source-lang eng \
  --hypo-lang spa \
  --wavlm-ckpt /path/to/wavlm_large_finetune.pth \
  --dnsmospro-cmd 'python /path/to/DNSMOSPro/infer.py --audio {audio}' \
  --dnsmospro-score-key <json_score_key> \
  --num-shards 1 \
  --parallel-jobs 1 \
  --sample-frac 1.0
```

Only `--num-shards 1` is supported. Outputs:

```text
per-example_metrics.tsv
aggregate_metrics.json
aggregate_metrics.tsv
paper_table_metrics.json
paper_table_metrics.tsv
```

`--uncertainty {std,sem,ci95}` adds `*_pm` fields; without it none are added.

## Requirements

The wrappers call the implementations in `evaluation/scripts/impl` and need the
metric backends: STOPES, SONAR/BLASER 2.0, the WavLM checkpoint used by Vsim,
DNSMOSPro (for `DNSMOSPro_Nat`), and a matching PyTorch/audio stack.
DNSMOSPro output is parsed only through `--dnsmospro-score-key` or
`--dnsmospro-score-regex`.
