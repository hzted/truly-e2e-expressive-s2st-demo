# Configuration Blockers Requiring Experiment Confirmation

The release package does not fabricate missing construction settings.  The
following items must be confirmed against the exact experiment logs before using
the package to claim bit-for-bit reconstruction of the paper manifests.

1. DNSMOSPro score-combination naming is inconsistent across historical files:
   some configuration text refers to `both_min` / `both_mean`, while the released
   selector script implements `min` / `mean`.
2. Historical configuration cutoffs differ from the reported retained/dropped
   boundaries of approximately 3.57 for En-Es and 3.60 for En-De.
3. The En-De notes describe duration-matched quality selection, while the
   currently released selector implements explicit cutoff or top-N selection.
4. The exact DNSMOSPro implementation commit, model checkpoint, score field, and
   source-target combination rule must be pinned before rerunning construction.
5. Stage-03 selected manifests should be accompanied by per-example
   `dnsmospro_quality_pairs.tsv.gz` files containing `sample_id`,
   `src_dnsmospro`, `tgt_dnsmospro`, `combined_dnsmospro`, `selected`, and
   `drop_reason`. If those files are absent from the external artifact, they
   must be regenerated from the original experiment logs, not inferred.
6. The split settings currently match the released split counts
   (`En-Es: dev_test_fraction=0.12, test_size=504`;
   `En-De: dev_test_fraction=0.11, test_size=504`) and should still be
   confirmed against original experiment logs before claiming exact rerunnable
   reconstruction.
7. The external full-manifest artifact URL is present in this GitHub mirror, but
   the current artifact does not include checkpoints or audited per-example
   DNSMOSPro score/decision tables. Do not claim checkpoint release or exact
   DNSMOSPro score reconstruction until those files and their SHA256 checks are
   added.

These blockers do not change sample IDs, split assignments, or paper-reported
results. They document the remaining provenance checks required for exact
reproducibility.
