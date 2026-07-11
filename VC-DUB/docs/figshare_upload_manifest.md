# Figshare Upload Manifest

Use this checklist for the external reviewer artifact. The GitHub mirror should
keep only code, schemas, synthetic examples, and non-sensitive summaries.

## Required Package Files

Upload these files together and generate `SHA256SUMS` after packaging:

```text
VC-DUB_full_manifests.tar.gz
SHA256SUMS
README_ARTIFACT.txt
manifest_inventory.tsv
manifest_inventory.json
```

`VC-DUB_full_manifests.tar.gz` should contain:

```text
global/vcdub_filtering_stage_stats.tsv.gz
global/vcdub_filtering_stage_stats.json
global/threshold_samples/
en_es/filtering/stage_00_aligned_pair_manifest.tsv.gz
en_es/filtering/stage_01_mms_lid_pass_manifest.tsv.gz
en_es/filtering/stage_02_sortformer_single_speaker_pass.tsv.gz
en_es/filtering/stage_03_dnsmospro_quality_selected_manifest.tsv.gz
en_es/filtering/dnsmospro_quality_pairs.tsv.gz
en_es/splits/all_metadata.tsv.gz
en_es/splits/train_metadata.tsv.gz
en_es/splits/dev_metadata.tsv.gz
en_es/splits/test_metadata.tsv.gz
en_es/splits/train_vc.tsv.gz
en_es/splits/dev_vc.tsv.gz
en_es/splits/test_vc.tsv.gz
en_es/summaries/
en_de/filtering/stage_00_aligned_pair_manifest.tsv.gz
en_de/filtering/stage_01_mms_lid_pass_manifest.tsv.gz
en_de/filtering/stage_02_sortformer_single_speaker_pass.tsv.gz
en_de/filtering/stage_03_dnsmospro_quality_selected_manifest.tsv.gz
en_de/filtering/dnsmospro_quality_pairs.tsv.gz
en_de/splits/all_metadata.tsv.gz
en_de/splits/train_metadata.tsv.gz
en_de/splits/dev_metadata.tsv.gz
en_de/splits/test_metadata.tsv.gz
en_de/splits/train_vc.tsv.gz
en_de/splits/dev_vc.tsv.gz
en_de/splits/test_vc.tsv.gz
en_de/summaries/
```

`dnsmospro_quality_pairs.tsv.gz` must include at least:

```text
sample_id
src_dnsmospro
tgt_dnsmospro
combined_dnsmospro
selected
drop_reason
```

## Sanitized Alternative

If the source corpus license does not allow redistribution of utterance text,
translations, codec tokens, or derived full manifests, upload a sanitized
artifact instead. It should preserve reproducibility of filtering decisions while
omitting copyrighted text/audio-derived content:

```text
sample_id
language_pair
split
mms_lid_pass
sortformer_single_speaker_pass
src_dnsmospro
tgt_dnsmospro
combined_dnsmospro
selected
drop_reason
anonymized_src_ref
anonymized_tgt_ref
source_duration_sec
target_duration_sec
```

## Optional Evaluation Artifacts

If the rebuttal claims that paper evaluation outputs are released, include:

```text
evaluation/per_example_metrics.tsv.gz
evaluation/aggregate_metrics.json
evaluation/aggregate_metrics.tsv
evaluation/evaluation_config.json
```

Only include ASR transcripts if the source-corpus license permits releasing the
text. Otherwise release aggregate ASR-BLEU summaries only.

## Optional Checkpoints

Only upload checkpoints if the paper response claims checkpoint release:

```text
VC-DUB_en-es_checkpoint.tar.gz
VC-DUB_en-de_checkpoint.tar.gz
training_commands.txt
checkpoint_inventory.json
```

Add all checkpoint files to `SHA256SUMS`.

## After Upload

Send the private Figshare or Zenodo reviewer URL back to the maintainer, then
update:

```text
VC-DUB/examples/full_artifact_package.json
VC-DUB/README.md
README.md
```
