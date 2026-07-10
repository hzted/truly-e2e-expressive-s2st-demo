# VC-DUB Construction Artifacts

This directory contains the reviewer-facing artifacts for reproducing the VC-DUB
supervision construction procedure. VC-DUB is released as a construction method,
not as a fixed standalone audio dataset.

The GitHub package intentionally contains code, configuration, schemas, and small
synthetic examples only. Complete per-example manifests are too large for an
anonymous GitHub mirror and may contain source-corpus text or derived metadata, so
they should be distributed through a separate artifact package only when licensing
permits it.

## Repository Contents

- `configs/`: model choices, filtering criteria, and observed thresholds.
- `scripts/`: construction, filtering, splitting, and release helper scripts.
- `scripts/voice_conversion/`: final local VC materialization wrapper.
- `evaluation/`: selected generated-audio evaluation templates and schema.
- `small_example_manifests/`: synthetic toy manifests that exercise the expected schemas.
- `manifest_schema.md`: required columns for each construction stage.
- `examples/`: small non-sensitive statistics and command examples.
- `DATA_LICENSE.md`: data-release and redistribution guidance.
- `requirements.txt` and `environment.yml`: lightweight runtime dependencies for utility scripts.

## Pipeline Order

The public reconstruction order is:

1. Obtain aligned source/target speech segments.
2. Denoise both sides with ClearVoice-Studio.
3. Extract vocal stems with Demucs.
4. Filter language pairs with MMS-LID.
5. Remove multi-speaker or overlapped segments with Sortformer diarization.
6. Score both sides with DNSMOSPro and apply scale-matched quality selection.
7. Run ASR/text-metadata preparation with Whisper large-v3 when text-bearing split manifests are needed.
8. Create train/dev/test split manifests.
9. Run voice conversion locally as the final materialization step.

Voice conversion is therefore the last step in this release. The filtering stages
operate on aligned, cleaned source/target utterance pairs and their metadata. The
generated VC waveforms are not redistributed.

## Environment

Install the Python dependencies used by the released utility scripts:

```bash
python -m pip install -r requirements.txt
```

Several stages require external model repositories or framework-specific
environments: ClearVoice-Studio for denoising, Demucs for vocal extraction, NeMo
for Sortformer diarization, DNSMOSPro for quality scoring, and SeedVC for the
final voice conversion materialization. Install those components from their
upstream projects and point the command templates to the local checkouts.

## Input Requirement

VC-DUB assumes an aligned bilingual dubbing corpus. Users should first collect or
prepare source/target speech segment pairs with corresponding text metadata.

If starting from parallel speech documents rather than pre-aligned utterances, an
embedding-based speech alignment method such as
[Speech Vecalign: an Embedding-based Method for Aligning Parallel Speech Documents](https://aclanthology.org/2025.emnlp-main.833.pdf)
can be used to obtain aligned speech segments. Equivalent alignment tools can also
be used. VC-DUB only requires that the result is converted into utterance-level
source/target audio pairs.

See `manifest_schema.md` for the expected columns. The synthetic examples under
`small_example_manifests/` use placeholder paths and do not contain source-corpus
text.

## DNSMOSPro Quality Selection

DNSMOSPro is used as the quality predictor. We score both source and target
utterances, compute a source-target combined quality score, and retain pairs above
the empirical quality cutoff. The cutoff characterizes the quality floor of the
retained clean pool while keeping the final VC-DUB training split comparable in
scale to CVSS-T.

Observed retained/dropped boundaries:

| Pair | Combined quality column | Approx. cutoff | Retained clean pool |
| --- | --- | ---: | ---: |
| En-Es | `combined_dnsmospro` | 3.57 | 90,000 |
| En-De | `combined_dnsmospro` | 3.60 | 147,639 |

## Voice Conversion Materialization

After filtering and splitting, run voice conversion locally from a selected
`*_asr.tsv` split manifest. The `*_asr.tsv` files retain the `id`, `pre_src`, and
`pre_tgt` columns required by the VC wrapper.

Example:

```bash
SPLIT_TSV=/path/to/VC-DUB/small_example_manifests/en_es/splits/train_asr.tsv \
SEEDVC_ROOT=/path/to/seed-vc-main \
OUTPUT_ROOT=/path/to/vcdub_vc_outputs/en_es/train \
PYTHON=/path/to/python \
NUM_SHARDS=1 \
MAX_PARALLEL=1 \
CUDA_DEV=0 \
bash scripts/voice_conversion/run_voice_conversion_materialization.sh
```

The wrapper writes the SeedVC-style pair TSV and, after local inference, a merged
materialization manifest:

```text
${OUTPUT_ROOT}/pair_tsvs/all_pairs.tsv
${OUTPUT_ROOT}/merged/vc_manifest.tsv
```

## Full Manifest Artifacts

The full manifests are not stored in this GitHub tree. If licensing permits
redistribution, package them separately, for example:

```text
VC-DUB_full_manifests.tar.gz
SHA256SUMS
```

During local preparation, keep the full archive and its `SHA256SUMS` outside the
Git repository, then upload them to the chosen anonymous artifact host.

If the source corpus license does not permit redistribution of text, translations,
or codec tokens, release only sanitized metadata such as `sample_id`, `split`,
stage decisions, scores, and placeholder reference IDs.

## Output Evaluation

Generated-audio evaluation commands are separated under `evaluation/`. The
included template runs only BLASER2.0, AutoPCP, speech-rate compliance at 20% and
40%, syllable speech-rate Pearson, pause weighted-mean duration score, vocal-style
similarity, and DNSMOSPro naturalness/MOS. It does not run ASR-BLEU, EmoCos, F0,
WER/CER, or any other MOS variants.
