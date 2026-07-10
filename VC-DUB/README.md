# VC-DUB Construction Artifacts

This directory contains scripts, configuration files, and manifest packaging utilities
for reproducing the VC-DUB supervision construction pipeline used in the paper.

VC-DUB is released here as a construction procedure rather than a fixed audio
dataset. The original aligned dubbing audio and generated voice-converted audio are
not included. Instead, this package records the exact filtering stages, thresholds,
model choices, and per-example manifests/splits needed to re-materialize the
supervision locally from an aligned dubbing corpus.

## Pipeline Order

The release package treats voice conversion as the final local materialization step:

0. Obtain an aligned bilingual dubbing corpus and format it as VC-DUB input manifests.
1. Start from an aligned dubbing manifest with source/target audio and text.
2. Denoise source and target utterances with ClearVoice-Studio.
3. Extract vocals with Demucs.
4. Filter language pairs with MMS-LID.
5. Remove multi-speaker or overlapped segments with Sortformer diarization.
6. Apply scale-matched quality selection with DNSMOSPro.
7. Create train/dev/test splits and downstream split manifests.
8. Run voice conversion locally on the selected split manifests.

This ordering keeps the released artifacts reusable without redistributing audio,
and avoids treating VC-generated waveforms as a standalone benchmark resource.

## Contents

- `configs/vcdub_construction_config.json`: model names, thresholds, and filtering criteria.
- `scripts/`: copied construction/filtering scripts plus release helper scripts.
- `scripts/voice_conversion/`: final local VC materialization wrapper and SeedVC batch helpers.
- `manifests/`: generated, path-sanitized, compressed manifests and split files.
- `docs/`: detailed inventory and notes for appendix/reporting.

## Environment

Install the Python dependencies used by the released utility scripts:

```bash
python -m pip install -r requirements.txt
```

Several pipeline stages also require external model repositories or framework-specific
environments: ClearVoice-Studio for denoising, Demucs for vocal extraction, NeMo for
Sortformer diarization, DNSMOSPro for quality scoring, and SeedVC for the final voice
conversion materialization. Those components should be installed following their
upstream instructions and then wired into the command templates in `scripts/`.

## Input Requirement

VC-DUB assumes an existing aligned dubbing corpus. In other words, before running
the construction scripts, users should collect or prepare source/target dubbing
utterance pairs with corresponding text metadata.

If starting from parallel speech documents rather than pre-aligned utterances,
an embedding-based alignment method such as
[Speech Vecalign: an Embedding-based Method for Aligning Parallel Speech Documents](https://aclanthology.org/2025.emnlp-main.833.pdf)
can be used to obtain aligned speech segments. Equivalent alignment tools can also
be used. VC-DUB only requires that the result is converted into utterance-level
source/target audio pairs.

The first materialization step expects a pair TSV with:

```text
source    target    output
```

where `source` is the source/content utterance path, `target` is the target/reference
voice utterance path, and `output` is the desired local path for the generated
voice-converted waveform. The batch VC step writes `manifests/vc_manifest.tsv`,
which is then consumed by the filtering and splitting scripts.

The released split manifests use the cleaned-audio columns:

```text
id    pre_src    pre_tgt
```

Additional text, ASR, duration, and bookkeeping columns are preserved when present.

## Build The Release Manifests

Run from the repository root:

```bash
EXPRESSIVE_S2ST_ROOT=/path/to/Expressive_S2ST \
python -u scripts/collect_release_manifests.py \
  --output-root manifests
```

The script writes `.tsv.gz` and `.json` files with absolute local paths replaced by
portable placeholders such as `{VC_DUB_ROOT}` and `{ALIGNED_DUBBING_ROOT}`.

## Voice Conversion Materialization

After filtering and splitting, run voice conversion locally from a selected split
manifest. Example:

```bash
SPLIT_TSV=/path/to/VC-DUB/manifests/en_es/splits/train.tsv.gz \
SEEDVC_ROOT=/path/to/seed-vc-main \
OUTPUT_ROOT=/path/to/vcdub_vc_outputs/en_es/train \
PYTHON=/path/to/python \
NUM_SHARDS=8 \
MAX_PARALLEL=1 \
CUDA_DEV=0 \
bash scripts/voice_conversion/run_voice_conversion_materialization.sh
```

The wrapper writes a merged materialization manifest to:

```text
${OUTPUT_ROOT}/merged/vc_manifest.tsv
```

## Audio Path Placeholders

The packaged manifests preserve the column names and row identities used during
construction, but local machine paths are anonymized. Before re-running the pipeline,
map placeholders to your local paths:

- `{VC_DUB_ROOT}/es_en`
- `{VC_DUB_ROOT}/de_en`
- `{ALIGNED_DUBBING_ROOT}/en_es`
- `{ALIGNED_DUBBING_ROOT}/de_en`
