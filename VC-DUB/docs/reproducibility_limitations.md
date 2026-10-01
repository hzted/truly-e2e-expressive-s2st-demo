# Reproducibility limitations

The following construction settings are not pinned in this release. Without
them the stages can be rerun, but not bit for bit.

1. DNSMOSPro score combination: some historical configuration text refers to
   `both_min` / `both_mean`, while the released selector implements `min` /
   `mean`.
2. Historical configuration cutoffs differ from the observed retained/dropped
   boundaries of approximately 3.57 (En-Es) and 3.60 (En-De).
3. The En-De notes describe duration-matched quality selection; the released
   selector implements an explicit cutoff or top-N selection.
4. The DNSMOSPro implementation commit, model checkpoint, score field and
   source-target combination rule are not pinned.
5. Per-example DNSMOSPro score tables (`dnsmospro_quality_pairs.tsv.gz` with
   `sample_id`, `src_dnsmospro`, `tgt_dnsmospro`, `combined_dnsmospro`,
   `selected`, `drop_reason`) are not included in the released manifests.
6. The split settings (En-Es: `dev_test_fraction=0.12`, `test_size=504`;
   En-De: `dev_test_fraction=0.11`, `test_size=504`) reproduce the released
   split sizes.
