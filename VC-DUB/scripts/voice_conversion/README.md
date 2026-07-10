# Voice Conversion Materialization

VC-DUB treats voice conversion as the final local materialization step. The
released manifests identify which cleaned utterance pairs are retained, while the
generated voice-converted waveforms are intentionally not redistributed.

## Inputs

- A selected VC-DUB `*_metadata.tsv`, `*_vc.tsv`, or stage-03 construction
  manifest, for example `small_example_manifests/en_es/splits/train_metadata.tsv`.
- A local SeedVC checkout containing `inference.py` and its model dependencies.
- Local access to the cleaned source/target audio paths referenced by the manifest.

## Build And Run

```bash
SPLIT_TSV=/path/to/VC-DUB/small_example_manifests/en_es/splits/train_metadata.tsv \
SEEDVC_ROOT=/path/to/seed-vc-main \
OUTPUT_ROOT=/path/to/vcdub_vc_outputs/en_es/train \
PYTHON=/path/to/python \
NUM_SHARDS=8 \
MAX_PARALLEL=1 \
CUDA_DEV=0 \
bash scripts/voice_conversion/run_voice_conversion_materialization.sh
```

The wrapper first converts the split manifest into a SeedVC pair TSV with columns:

```text
source    target    output
```

It then runs SeedVC per shard and writes a merged `vc_manifest.tsv` under:

```text
${OUTPUT_ROOT}/merged/vc_manifest.tsv
```

## Audio Direction

The default released setup uses:

- `pre_src` as the content/source utterance.
- `pre_tgt` as the target/reference voice utterance.
- generated `output` as the local VC materialization.

The bundled wrapper calls the release package's
`scripts/voice_conversion/inference_batch_denoise_both.py` with `PYTHONPATH`
pointing at `SEEDVC_ROOT`, so the normal SeedVC checkout does not need to contain
the custom batch script.

The source and target columns can be changed through:

```bash
SOURCE_AUDIO_COL=pre_src
TARGET_AUDIO_COL=pre_tgt
ID_COL=sample_id
```
