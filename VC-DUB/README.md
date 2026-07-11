# VC-DUB Construction Artifacts

This directory contains reviewer-facing artifacts documenting the VC-DUB
supervision construction procedure. VC-DUB is released as a construction method,
not as a fixed standalone audio dataset.

The GitHub package intentionally contains code, configuration, schemas, and small
synthetic examples only. Complete per-example manifests are distributed through a
separate artifact package only when licensing permits it.

## Repository Contents

- `configs/`: model choices, filtering criteria, and confirmed/unconfirmed settings.
- `scripts/`: construction, filtering, splitting, and release helper scripts.
- `scripts/voice_conversion/`: optional final local VC materialization wrapper.
- `evaluation/`: paper evaluation package for generated audio.
- `small_example_manifests/`: synthetic toy manifests that exercise the schemas.
- `manifest_schema.md`: required columns for each construction stage.
- `docs/blockers.md`: configuration conflicts that require experiment-log confirmation.
- `docs/model_dependencies.md`: upstream model/tool links and local preparation notes.
- `examples/`: small non-sensitive statistics and command examples.
- `DATA_LICENSE.md`: data-release and redistribution guidance.

## Construction Pipeline

Construction is organized around these stages:

1. aligned-pair metadata preparation
2. ClearVoice/Demucs preprocessing
3. MMS-LID filtering
4. Sortformer speaker filtering
5. DNSMOSPro quality scoring and selection
6. train/dev/test split assignment
7. optional local voice-conversion materialization

Voice conversion is the last step in this release. Filtering and split assignment
operate on aligned, cleaned source/target utterance pairs and their construction
metadata. Generated VC waveforms are not redistributed.

The provided runner is a stage wrapper, not a bit-for-bit end-to-end
reconstruction script. Stages 1--2 require users to prepare aligned metadata and
run ClearVoice/Demucs in their local environment before the downstream filtering
wrappers can consume the resulting manifest. Exact DNSMOSPro scoring/selection
settings remain blockers unless confirmed from the original experiment logs.

Whisper large-v3 is not a construction dependency. It must not affect sample
retention, deletion, ordering, or train/dev/test split assignment. If ASR-based
paper evaluation is run, it lives under `evaluation/`.

## Environment

Install the lightweight Python dependencies used by the released utility scripts:

```bash
python -m pip install -r requirements.txt
```

Several stages require external model repositories or framework-specific
environments: ClearVoice-Studio for denoising, Demucs for vocal extraction, NeMo
for Sortformer diarization, DNSMOSPro for quality scoring, and SeedVC for optional
final VC materialization. Install those components from their upstream projects
and point the command templates to the local checkouts.
See `docs/model_dependencies.md` for links and preparation notes.

## Input Requirement

VC-DUB assumes an aligned bilingual dubbing corpus. Users should first collect or
prepare source/target speech segment pairs with corresponding text metadata.

If starting from parallel speech documents rather than pre-aligned utterances, an
embedding-based speech alignment method such as
[Speech Vecalign: an Embedding-based Method for Aligning Parallel Speech Documents](https://aclanthology.org/2025.emnlp-main.833.pdf)
can be used to obtain aligned speech segments. Equivalent alignment tools can
also be used. VC-DUB only requires that the result is converted into
utterance-level source/target audio pairs.

See `manifest_schema.md` for the expected columns. The synthetic examples under
`small_example_manifests/` use placeholder paths and do not contain source-corpus
text.

## DNSMOSPro Quality Selection

DNSMOSPro is the construction-time quality-selection criterion. The auditable
artifact is:

```text
dnsmospro_quality_pairs.tsv.gz
```

It must contain at least:

```text
sample_id
src_dnsmospro
tgt_dnsmospro
combined_dnsmospro
selected
drop_reason
```

The paper text reports retained/dropped boundaries of approximately 3.57 for
En-Es and 3.60 for En-De. These are observed boundaries after selection, not
necessarily preset cutoffs. The exact implementation commit, checkpoint, score
field, source-target combination rule, and cutoff/duration-matching rule require
confirmation from the original experiment logs. These unresolved items are listed
in `docs/blockers.md`; the public package does not guess them.

The DNSMOSPro scoring wrapper requires explicit parsing:

```bash
python -u scripts/score_dnsmospro_for_filtering.py \
  --input-tsv /path/to/sortformer_pair_pass_strict.tsv \
  --out-dir /path/to/dnsmospro \
  --id-col sample_id \
  --src-audio-col pre_src \
  --tgt-audio-col pre_tgt \
  --combine <confirmed_min_or_mean> \
  --dnsmospro-cmd 'python /path/to/DNSMOSPro/infer.py --audio {audio}' \
  --score-key <confirmed_json_score_key>
```

Use `--score-regex` instead of `--score-key` only when the DNSMOSPro checkout
prints named text output. The wrapper deliberately rejects implicit "first
number in stdout" parsing.

## Build Splits

The split builder reads the selected construction manifest and optional aligned
metadata. It does not read ASR metadata.

The settings matching the currently released split counts are:

| Pair | Clean pool | Dev+test fraction | Test pairs | Train pairs |
| --- | ---: | ---: | ---: | ---: |
| En-Es | 90,000 | 0.12 | 504 | 79,200 |
| En-De | 147,639 | 0.11 | 504 | 131,399 |

These values reproduce the released split counts, but exact rerunnable
construction still depends on resolving the DNSMOSPro blockers in
`docs/blockers.md`.

```bash
cd /path/to/truly-e2e-expressive-s2st-demo/VC-DUB

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

The outputs are:

```text
all_metadata.tsv
train_metadata.tsv
dev_metadata.tsv
test_metadata.tsv
train_vc.tsv
dev_vc.tsv
test_vc.tsv
split_summary.json
```

## Voice Conversion Materialization

After filtering and splitting, run voice conversion locally from `*_metadata.tsv`,
`*_vc.tsv`, or a selected stage-04 manifest containing `sample_id`, `pre_src`,
and `pre_tgt`.

```bash
cd /path/to/truly-e2e-expressive-s2st-demo/VC-DUB

SPLIT_TSV=small_example_manifests/en_es/splits/train_metadata.tsv \
SEEDVC_ROOT=/path/to/seed-vc-main \
OUTPUT_ROOT=/tmp/vcdub_vc_outputs/en_es/train \
PYTHON=/path/to/python \
NUM_SHARDS=1 \
MAX_PARALLEL=1 \
CUDA_DEV=0 \
bash scripts/voice_conversion/run_voice_conversion_materialization.sh
```

The wrapper writes:

```text
${OUTPUT_ROOT}/pair_tsvs/all_pairs.tsv
${OUTPUT_ROOT}/merged/vc_manifest.tsv
```

## Reproducibility Artifacts

The full reproducibility artifact is hosted through an anonymous Figshare
private link:

```text
https://figshare.com/s/06a010b1ab7f2d0e0486
```

The Figshare package currently contains:

- per-example VC-DUB construction manifests for the aligned-pair, preprocessing,
  MMS-LID, Sortformer, and selected clean-pool stages;
- train/dev/test split manifests, including metadata and VC-materialization
  inputs;
- stage-wise count and duration statistics;
- `SHA256SUMS` integrity checks for the uploaded archive.

The artifact does not redistribute original audio, denoised/vocal-extracted
audio, or generated voice-converted waveforms.
The current artifact does not include audited per-example DNSMOSPro
score/decision tables; the selected clean-pool manifests and provenance blockers
are included instead.

The expected downloadable files are:

```text
VC-DUB_full_manifests.tar.gz
SHA256SUMS
```

Checkpoint artifacts will be added separately if released.

## Paper Evaluation

Generated-audio evaluation commands are separated under `evaluation/`. These are
wrappers around the project metric implementations; real-mode execution still
requires the matching external metric backends, checkpoints, and environment.
The package covers only paper-reported metrics:

- BLASER 2.0
- A.PCP
- SLC at `p = 0.2` and `p = 0.4`
- speech-rate compliance at `p = 0.2` and `p = 0.4`
- syllable speech-rate correlation
- pause weighted-mean duration score
- Vsim
- DNSMOSPro when reported as a quality metric
- optional ASR-based evaluation only when explicitly enabled

See `evaluation/README.md`.
