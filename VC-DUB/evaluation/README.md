# VC-DUB Output Evaluation

This folder documents the selected output-audio metrics used for generated
speech evaluation. It is separate from the data-cleaning pipeline: cleaning uses
Whisper large-v3 only for ASR/text metadata, while this folder evaluates final
generated audio.

## Metrics Included

Only the following metrics are included:

- BLASER2.0 audio score (`blaser2_qe_audio_mean`, and `blaser2_ref_mean` when a reference audio column is provided).
- AutoPCP (`autopcp_mean`).
- Speech-rate compliance within 20% and 40% (`sc_0p2_compliance`, `sc_0p4_compliance`).
- Syllable speech-rate Pearson correlation (`speech_rate_syllable_pearson`).
- Pause weighted-mean duration score (`pause_wmean_duration_score`).
- Vocal-style similarity (`vsim_mean`).
- DNSMOSPro naturalness/MOS (`dnsmospro_nat_mean`).

No ASR-BLEU, EmoCos, F0, WER/CER, or additional metrics are run here.

## Manifest

See `eval_manifest_schema.md`. The expected TSV columns are:

```text
id
source_audio
hypo_audio
source_text
hypo_text
source_lang
hypo_lang
reference_audio
reference_text
status
```

The text columns should come from the same Whisper large-v3 ASR/text-metadata
logic used elsewhere in the project when generated-audio transcripts are needed.

## Command Template

```bash
EVAL_MANIFEST=/path/to/eval_manifest.tsv \
OUT_DIR=/path/to/evaluation_outputs \
VERIFY_SCRIPTS_ROOT=/path/to/Expressive_S2ST/verify_scripts \
PYTHON=/path/to/python \
SOURCE_LANG=eng \
HYPO_LANG=spa \
WAVLM_CKPT=/path/to/wavlm_large_finetune.pth \
DNSMOSPRO_CMD='python /path/to/DNSMOSPro/infer.py --audio {audio}' \
bash evaluation/run_selected_metrics_template.sh
```

The final paper-facing output is:

```text
${OUT_DIR}/selected_metrics_summary.tsv
${OUT_DIR}/selected_metrics_summary.json
```

These files contain only the metric whitelist listed above, even if the underlying
Stopes or BLASER scripts write additional diagnostic outputs.
