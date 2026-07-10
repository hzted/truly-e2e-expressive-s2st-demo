# VC-DUB Manifest Schemas

The public GitHub package uses synthetic examples only. Full manifests, when
redistributable, should follow the same schemas.

## Aligned Pair Input

Required columns:

```text
id
src_audio
tgt_audio
src_text
tgt_text
src_lang
tgt_lang
```

`src_audio` is the source/content utterance. `tgt_audio` is the aligned target
utterance used as the reference voice/prosody side.

## Preprocessed Pair Manifest

Required columns:

```text
id
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
id
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
id
pre_src
pre_tgt
src_num_speakers
tgt_num_speakers
src_segments_json
tgt_segments_json
sortformer_single_speaker_pass
```

The strict gate keeps pairs where source and target each have at most one detected
active speaker.

## DNSMOSPro Quality Output

Required columns:

```text
id
pre_src
pre_tgt
src_dnsmospro
tgt_dnsmospro
combined_dnsmospro
dnsmospro_quality_pass
```

The empirical retained/dropped boundary is approximately `3.57` for En-Es and
`3.60` for En-De.

## Text-Bearing Split Manifest

The VC materialization wrapper expects the `*_asr.tsv` split format:

```text
id
pre_src
pre_tgt
sentence
translation
src_asr
tgt_asr
split
```

Only `id`, `pre_src`, and `pre_tgt` are required for voice conversion. The text
columns are preserved for downstream organization and evaluation.

## SeedVC Pair TSV

The final local materialization step converts a split manifest into:

```text
source
target
output
```

`source` is the content/source utterance path, `target` is the target/reference
voice utterance path, and `output` is the desired local path for the generated
voice-converted waveform.
