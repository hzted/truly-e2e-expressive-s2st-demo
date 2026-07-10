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

1. Start from an aligned dubbing manifest with source/target audio and text.
2. Denoise source and target utterances with ClearVoice-Studio.
3. Extract vocals with Demucs.
4. Filter language pairs with MMS-LID.
5. Remove multi-speaker or overlapped segments with Sortformer diarization.
6. Apply scale-matched quality selection with UTMOS v2.
7. Create train/dev/test splits and downstream split manifests.
8. Run voice conversion locally on the selected split manifests.

This ordering keeps the released artifacts reusable without redistributing audio,
and avoids treating VC-generated waveforms as a standalone benchmark resource.

## Contents

- `configs/vcdub_construction_config.json`: model names, thresholds, and filtering criteria.
- `scripts/`: copied construction/filtering scripts plus release helper scripts.
- `manifests/`: generated, path-sanitized, compressed manifests and split files.
- `docs/`: detailed inventory and notes for appendix/reporting.

## Build The Release Manifests

Run from the repository root:

```bash
EXPRESSIVE_S2ST_ROOT=/path/to/Expressive_S2ST \
python -u scripts/collect_release_manifests.py \
  --output-root manifests
```

The script writes `.tsv.gz` and `.json` files with absolute local paths replaced by
portable placeholders such as `{VC_DUB_ROOT}` and `{ALIGNED_DUBBING_ROOT}`.

## Audio Path Placeholders

The packaged manifests preserve the column names and row identities used during
construction, but local machine paths are anonymized. Before re-running the pipeline,
map placeholders to your local paths:

- `{VC_DUB_ROOT}/es_en`
- `{VC_DUB_ROOT}/de_en`
- `{ALIGNED_DUBBING_ROOT}/en_es`
- `{ALIGNED_DUBBING_ROOT}/de_en`

## Notes

The final training split rows in the statistics are reported for comparison with
Table 1 and are not additional filtering gates.

The early filtering-stage duration statistics use audio-file duration because the
raw and intermediate cleaning manifests do not store VAD-span columns. Final clean
pool and training split duration statistics use the VAD-span columns from the
downstream split manifests.
