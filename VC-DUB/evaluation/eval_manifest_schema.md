# VC-DUB Output Evaluation Manifest

The output-evaluation scripts expect a TSV with one row per generated utterance.

Required columns:

```text
id
source_audio
hypo_audio
source_text
hypo_text
source_lang
hypo_lang
```

Recommended optional columns:

```text
reference_audio
reference_text
status
```

Column meanings:

- `source_audio`: source/content speech used as input to S2ST or VC materialization.
- `hypo_audio`: generated output audio to evaluate.
- `source_text`: source-side text or ASR transcript used by Stopes local prosody.
- `hypo_text`: output-side text or ASR transcript used by Stopes local prosody.
- `source_lang` and `hypo_lang`: Stopes/SONAR language tags such as `eng`, `spa`, or `deu`.
- `reference_audio`: optional reference target audio for BLASER2 reference mode.
- `reference_text`: optional reference target text.

For ASR-derived text metadata, use Whisper large-v3 outputs.
