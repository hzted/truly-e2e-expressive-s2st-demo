# VC-Dub construction

Scripts, configuration and schemas for building VC-Dub training pairs from
aligned professional dubbing: audio cleaning, language and speaker filtering,
quality selection, train/dev/test splitting, and voice conversion of the dubbed
side. VC-Dub is released as a construction method; the dubbing audio itself is
not redistributed.

## Contents

- `configs/`: model choices and filtering criteria.
- `scripts/`: construction, filtering and splitting scripts.
- `scripts/voice_conversion/`: voice-conversion step (Seed-VC).
- `evaluation/`: utilities for scoring generated speech.
- `small_example_manifests/`: synthetic manifests illustrating the schemas.
- `manifest_schema.md`: columns of each construction stage.
- `docs/reproducibility_limitations.md`: settings that are not pinned in this release.
- `docs/model_dependencies.md`: external models and tools.
- `examples/`: stage-wise statistics.
- `DATA_LICENSE.md`: data redistribution notes.

## Construction pipeline

1. aligned-pair metadata preparation
2. ClearVoice/Demucs preprocessing
3. MMS-LID filtering
4. Sortformer speaker filtering (exactly one active speaker in each utterance)
5. DNSMOSPro quality scoring and selection
6. train/dev/test split assignment
7. voice conversion of the dubbed side

Filtering and splitting operate on aligned, cleaned source/target pairs; voice
conversion is applied last. The runner wraps the individual stages; stages 1-2
(alignment metadata and ClearVoice/Demucs preprocessing) are run in the user's
own environment before the filtering scripts consume the resulting manifest.
Whisper is not used in construction.

## Environment

```bash
python -m pip install -r requirements.txt
```

ClearVoice-Studio, Demucs, NeMo (Sortformer), DNSMOSPro and Seed-VC are installed
from their upstream projects; see `docs/model_dependencies.md`.

## Input

An aligned bilingual dubbing corpus: utterance-level source/target audio pairs
with text metadata. Parallel speech documents can be aligned with
[Speech Vecalign](https://aclanthology.org/2025.emnlp-main.833/) or an
equivalent tool. See `manifest_schema.md` for the expected columns; the
synthetic examples under `small_example_manifests/` use placeholder paths.

## DNSMOSPro quality selection

Pairs are scored with DNSMOSPro on both sides and selected on a combined
source-target score. The retained/dropped boundaries in the paper's
instantiations are approximately 3.57 (En-Es) and 3.60 (En-De); these are
observed boundaries after selection rather than preset cutoffs. The scoring
script requires an explicit score field:

```bash
python -u scripts/score_dnsmospro_for_filtering.py \
  --input-tsv /path/to/sortformer_pair_pass_strict.tsv \
  --out-dir /path/to/dnsmospro \
  --id-col sample_id \
  --src-audio-col pre_src \
  --tgt-audio-col pre_tgt \
  --combine <min_or_mean> \
  --dnsmospro-cmd 'python /path/to/DNSMOSPro/infer.py --audio {audio}' \
  --score-key <json_score_key>
```

`--score-regex` can replace `--score-key` when DNSMOSPro prints text output.

## Splits

The split builder reads the selected manifest and optional aligned metadata.
Settings that give the paper's split sizes:

| Pair | Clean pool | Dev+test fraction | Test pairs | Train pairs |
| --- | ---: | ---: | ---: | ---: |
| En-Es | 90,000 | 0.12 | 504 | 79,200 |
| En-De | 147,639 | 0.11 | 504 | 131,399 |

```bash
python -u scripts/build_vcdub_splits.py \
  --selected-manifest-tsv small_example_manifests/en_es/filtering/stage_04_quality_selected_manifest.tsv \
  --aligned-metadata-tsv small_example_manifests/en_es/filtering/stage_00_aligned_pair_manifest.tsv \
  --out-dir /tmp/vcdub_example_splits \
  --id-col sample_id \
  --source-audio-col pre_src \
  --target-audio-col pre_tgt \
  --source-text-col src_text \
  --target-text-col tgt_text \
  --dev-test-ratio 0.50 \
  --test-size 1 \
  --seed 42 \
  --overwrite
```

Outputs: `all_metadata.tsv`, `{train,dev,test}_metadata.tsv`,
`{train,dev,test}_vc.tsv`, `split_summary.json`.

## Voice conversion

Run on `*_metadata.tsv`, `*_vc.tsv`, or a selected stage-04 manifest containing
`sample_id`, `pre_src` and `pre_tgt`:

```bash
SPLIT_TSV=small_example_manifests/en_es/splits/train_metadata.tsv \
SEEDVC_ROOT=/path/to/seed-vc \
OUTPUT_ROOT=/tmp/vcdub_vc_outputs/en_es/train \
PYTHON=/path/to/python \
NUM_SHARDS=1 \
MAX_PARALLEL=1 \
CUDA_DEV=0 \
bash scripts/voice_conversion/run_voice_conversion_materialization.sh
```

Outputs: `${OUTPUT_ROOT}/pair_tsvs/all_pairs.tsv` and
`${OUTPUT_ROOT}/merged/vc_manifest.tsv`.

## Construction manifests

Per-example construction manifests are available on Figshare:
<https://figshare.com/s/06a010b1ab7f2d0e0486> (`VC-DUB_full_manifests.tar.gz`
with `SHA256SUMS`). They cover the aligned-pair, preprocessing, MMS-LID,
Sortformer and selected-pool stages, the train/dev/test splits (metadata and
voice-conversion inputs), and stage-wise count and duration statistics. They do
not include audio (original, cleaned or voice-converted) or per-example
DNSMOSPro scores.

## Release status

Released: construction and splitting scripts, model choices and filtering
criteria, observed DNSMOSPro boundaries, synthetic schema examples, stage-wise
statistics, the per-example manifests above, and the DirectS2ST checkpoints
(see `../DirectS2ST`).

Not released: original, cleaned or voice-converted audio, and per-example
DNSMOSPro score tables. Some DNSMOSPro settings are not pinned, so the stages
cannot yet be rerun bit for bit; see `docs/reproducibility_limitations.md`.

## Evaluation

See `evaluation/README.md`.
