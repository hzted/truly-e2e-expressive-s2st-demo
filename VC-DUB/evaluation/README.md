# VC-DUB Paper Evaluation Package

This package is separate from VC-DUB construction/cleaning. It evaluates final
generated audio and must not be used to decide which construction examples are
kept, dropped, ordered, or assigned to train/dev/test.

## Metrics

The package covers only paper-facing metrics:

- Content: BLASER 2.0
- Prosody: A.PCP
- Isochrony: duration SLC at `p = 0.2`, duration SLC at `p = 0.4`,
  syllable speech-rate correlation, pause weighted-mean duration score
- Speaker identity: Vsim
- Quality: DNSMOSPro, only when reported as an evaluation metric
- ASR: Whisper large-v3, only when an ASR-based evaluation metric is explicitly enabled

Whisper large-v3 is not a BLASER, DNSMOSPro, Vsim, A.PCP, or VC-DUB cleaning
dependency.

## Paper Table Field Mapping

The Stopes/local-prosody implementation exposes several internal columns. The
paper-facing table should use only the following mapping:

| Paper column | Aggregated output key | Underlying implementation field |
| --- | --- | --- |
| `BLASER2_QE` | `BLASER2_QE` | `blaser2_qe_audio_mean` |
| `BLASER2_ref` | `BLASER2_ref` | `blaser2_ref_mean` |
| `A_PCP` | `A_PCP` | `autopcp_mean` |
| `SLC_0p2` | `SLC_0p2` | `dc_0p2_compliance_mean` |
| `SLC_0p4` | `SLC_0p4` | `dc_0p4_compliance_mean` |
| `SpeechRate` | `SpeechRate` | `speech_rate_syllable_spearman_mean` |
| `Pause` | `Pause` | `pause_wmean_duration_score_mean` |
| `Vsim` | `Vsim` | `vsim_mean` |
| `DNSMOSPro_Nat` | `DNSMOSPro_Nat` | `dnsmospro_nat_mean` |

`sc_0p2_compliance` and `sc_0p4_compliance` may still appear in intermediate
debug files, but they are not part of the default paper-table output.

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
It validates command plumbing and aggregation only; it does not validate metric
numerical equivalence.

## Real Evaluation Command

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
  --dnsmospro-score-key <confirmed_json_score_key> \
  --num-shards 1 \
  --parallel-jobs 1 \
  --sample-frac 1.0
```

Use `--num-shards 1` in the reviewer release. Multi-shard evaluation is disabled
until per-example ID ordering has been fully audited.

Outputs:

```text
per-example_metrics.tsv
aggregate_metrics.json
aggregate_metrics.tsv
paper_table_metrics.json
paper_table_metrics.tsv
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

The wrappers call the vendored project implementations under
`evaluation/scripts/impl` rather than fixed-value dry-run outputs. Real-mode
execution still requires the original metric backends: Stopes, SONAR/BLASER 2.0,
the WavLM checkpoint used by Vsim, DNSMOSPro, and the matching PyTorch/audio
stack. If a model checkpoint, dependency version, or official implementation
commit is missing, treat it as a blocker and fill it from the original
experiment environment instead of guessing.

DNSMOSPro evaluation also requires explicit parsing through
`--dnsmospro-score-key` or `--dnsmospro-score-regex`; implicit first-number
parsing is disabled.

The optional Whisper wrapper only produces transcripts. It does not compute or
aggregate WER, CER, ASR-BLEU, or normalized ASR-BLEU in this release.
