# VC-DUB Manifest Schemas

The public GitHub package uses synthetic examples only. Full manifests, when
redistributable, should follow the same schemas.

## Aligned Pair Input

Required columns:

```text
sample_id
src_audio
tgt_audio
src_text
tgt_text
src_lang
tgt_lang
```

`src_audio` is the source/content utterance. `tgt_audio` is the aligned
speaker-reference utterance used by optional voice-conversion materialization.
The schema does not treat target audio as a prosody label; any prosodic transfer
risk should be evaluated separately in the paper-facing metrics.

## Preprocessed Pair Manifest

Required columns:

```text
sample_id
pre_src
pre_tgt
src_text
tgt_text
clearvoice_src_success
clearvoice_tgt_success
demucs_src_success
demucs_tgt_success
```

`pre_src` and `pre_tgt` are the denoised and vocal-extracted source/target audio
paths.

## MMS-LID Output

Required columns:

```text
sample_id
pre_src
pre_tgt
src_lid_label
tgt_lid_label
src_lid_score
tgt_lid_score
lid_pass
```

The released setting uses top-1 label matching and no confidence threshold.

## Sortformer Output

Required columns:

```text
sample_id
pre_src
pre_tgt
src_num_speakers
tgt_num_speakers
src_segments_json
tgt_segments_json
sortformer_single_speaker_pass
```

The strict gate keeps pairs where source and target each have exactly one
detected active speaker. Rows with zero detected speakers are treated as
non-passing because the diarizer did not confirm single-speaker speech.

## DNSMOSPro Quality Output

Required columns:

```text
sample_id
pre_src
pre_tgt
src_dnsmospro
tgt_dnsmospro
combined_dnsmospro
selected
drop_reason
```

This is a construction-time artifact. Exact score combination and selection
settings must be confirmed from experiment logs; see `docs/blockers.md`.

## Metadata Split Manifest

Required columns:

```text
sample_id
pre_src
pre_tgt
src_text
tgt_text
split
```

Additional construction metadata columns may be retained. ASR-specific fields do
not belong in construction split manifests.

## VC Split Manifest

Required columns:

```text
sample_id
pre_src
pre_tgt
```

These are the minimal columns needed by optional final voice-conversion
materialization.

## SeedVC Pair TSV

The final local materialization step converts a metadata or VC manifest into:

```text
id
source
target
output
```

`id` is the original `sample_id`, `source` is the content/source utterance path,
`target` is the target/reference voice utterance path, and `output` is the
desired local path for the generated voice-converted waveform.

## Stage-Wise Statistics

`scripts/compute_vcdub_filtering_stage_stats.py` writes:

```text
language_pair
stage
num_pairs
source_hours
target_hours
total_hours
avg_source_duration_sec
avg_target_duration_sec
retention_from_previous_stage
duration_source_columns_used
manifest_path
```
