# VC-DUB Release Notes

## What Is Released In GitHub

This package is intended for anonymous review and method reproducibility. It
includes:

- End-to-end construction, filtering, splitting, and VC materialization scripts.
- Model names and filtering criteria.
- Observed DNSMOSPro quality cutoffs for the En-Es and En-De experimental
  instantiations.
- Synthetic small manifests documenting the expected schemas.
- Non-sensitive stage-wise duration/count statistics.

## What Is Not Released In GitHub

The GitHub mirror does not include:

- Original aligned dubbing audio.
- ClearVoice/Demucs-enhanced audio.
- Voice-converted waveforms.
- Full per-example manifests containing source-corpus text, translations, codec
  tokens, or derived row-level metadata.
- Checkpoints.

Full manifests should be uploaded to a separate anonymous artifact host only when
the source-corpus license permits redistribution. Otherwise, share a sanitized
metadata table and let authorized users join it against their local corpus copy.

## Reviewer-Facing Description

VC-DUB is a supervision-construction method. The En-De and En-Es corpora used in
the paper are experimental instantiations for validating the construction method,
not proposed as fixed standalone datasets.

Voice conversion is treated as the final local materialization step after
filtering and splitting. DNSMOSPro is the quality predictor used for
scale-matched filtering.
