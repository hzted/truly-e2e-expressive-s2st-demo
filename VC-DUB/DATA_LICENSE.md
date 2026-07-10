# Data Redistribution Notice

This repository does not redistribute the original dubbing audio, enhanced audio,
voice-converted audio, or full source-corpus manifests.

The files under `small_example_manifests/` are synthetic examples with placeholder
paths and dummy text. They are included only to document the expected file schemas.

Full VC-DUB manifests may contain source-corpus text, translations, derived audio
metadata, or codec-token metadata. Those artifacts should be shared only if the
underlying source-corpus license permits redistribution. If redistribution is not
permitted, release a sanitized table containing only non-sensitive fields such as:

- `sample_id`
- `split`
- filtering-stage pass/drop flags
- LID, diarization, and DNSMOSPro scores/decisions
- anonymized source/target reference IDs

Researchers with lawful access to the source corpus can then join the sanitized
metadata back to their local copy of the original data.
