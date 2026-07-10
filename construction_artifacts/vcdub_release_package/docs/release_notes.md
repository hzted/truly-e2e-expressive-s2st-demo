# VC-DUB Release Notes

## What Is Released

This package is intended for anonymous review and reproducibility. It includes:

- End-to-end filtering and construction scripts.
- Model names and filtering criteria.
- Observed thresholds used for the En-Es and En-De experimental instantiations.
- Per-example manifests for intermediate filtering stages and train/dev/test splits.
- Stage-wise duration/count statistics.

## What Is Not Released

Audio waveforms are not included:

- Original aligned dubbing audio is subject to the source corpus license.
- ClearVoice/Demucs-enhanced waveforms are derived audio.
- Voice-converted waveforms are generated materializations and should be recreated locally.

## Recommended Reviewer-Facing Description

VC-DUB is a supervision-construction method. The En-De and En-Es corpora used in
the paper are experimental instantiations for validating the construction method,
not proposed as fixed standalone datasets.

The released manifests allow reviewers to inspect exact row identities, filtering
decisions, splits, thresholds, and downstream training inputs without redistributing
restricted audio.
