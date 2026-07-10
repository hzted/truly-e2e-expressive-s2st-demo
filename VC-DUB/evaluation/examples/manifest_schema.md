# Evaluation Manifest Schema

The evaluation manifest is keyed by `sample_id`.

Required columns:

```text
sample_id
source_audio
hypo_audio
source_text
hypo_text
source_lang
hypo_lang
target_lang
status
```

Optional columns:

```text
reference_audio
reference_text
reference_translation
```

Column meaning:

- `source_audio`: input/source utterance audio used for source-vs-hypothesis metrics.
- `hypo_audio`: generated system output audio.
- `source_text`: source-side text or transcript used by Stopes local prosody.
- `hypo_text`: generated-side text or transcript used by Stopes local prosody.
- `target_lang`: language tag consumed by BLASER/SONAR wrappers.
- `reference_audio`: optional target/reference audio for reference-aware metrics.
- `reference_translation`: optional text reference for BLASER reference mode.

Whisper transcript columns are not required unless an ASR-based evaluation metric
is explicitly enabled.
