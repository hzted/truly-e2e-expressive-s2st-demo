# VC-DUB Paper Evaluation Package

This package is separate from VC-DUB construction/cleaning. It evaluates final
generated audio and must not be used to decide which construction examples are
kept, dropped, ordered, or assigned to train/dev/test.

## Metrics

The package covers only paper-facing metrics:

- Content: BLASER 2.0
- Prosody: A.PCP
- Isochrony: SLC at `p = 0.2`, SLC at `p = 0.4`, syllable speech-rate correlation, pause weighted-mean duration score
- Speaker identity: Vsim
- Quality: DNSMOSPro, only when reported as an evaluation metric
- ASR: Whisper large-v3, only when an ASR-based evaluation metric is explicitly enabled

Whisper large-v3 is not a BLASER, DNSMOSPro, Vsim, A.PCP, or VC-DUB cleaning
dependency.

## Input Manifest

Use a single TSV keyed by `sample_id`. See `examples/manifest_schema.md`.

Required core columns:

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

Recommended optional columns:

```text
reference_audio
reference_text
reference_translation
```

## One-Command Smoke Test

From the `VC-DUB` directory:

```bash
bash evaluation/tests/test_smoke.sh
```

This uses `--dry-run`, so it does not require model checkpoints or audio files.

## Real Evaluation Command

From the `VC-DUB` directory:

```bash
python -u evaluation/scripts/run_all_metrics.py \
  --manifest /path/to/eval_manifest.tsv \
  --out-dir /path/to/evaluation_outputs \
  --config evaluation/configs/evaluation_config.json \
  --python /path/to/python \
  --verify-scripts-root /export/fs06/hzhan276/Expressive_S2ST/verify_scripts \
  --source-lang eng \
  --hypo-lang spa \
  --wavlm-ckpt /path/to/wavlm_large_finetune.pth \
  --dnsmospro-cmd 'python /path/to/DNSMOSPro/infer.py --audio {audio}' \
  --num-shards 1 \
  --parallel-jobs 1
```

Outputs:

```text
per-example_metrics.tsv
aggregate_metrics.json
aggregate_metrics.tsv
```

## Plus/Minus Reporting

The aggregator does not silently choose what table `±` means. Pass one of:

```text
--uncertainty std
--uncertainty sem
--uncertainty ci95
```

If omitted, no `*_pm` fields are added. The paper table caption should explicitly
state whether `±` is standard deviation, standard error, or a 95% confidence
interval margin.

## Implementation Notes

The wrappers call the project implementations under `verify_scripts` rather than
reimplementing metric proxies. If a model checkpoint, dependency version, or
official implementation commit is missing, treat it as a blocker and fill it from
the original experiment environment instead of guessing.
