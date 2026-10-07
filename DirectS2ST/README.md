# DirectS2ST

Inference code for DirectS2ST, the end-to-end expressive speech-to-speech
translation model from *Holistic Parallel Supervision for Expressive
Speech-to-Speech Translation*. Given English source speech, it predicts all 16
Mimi codec streams of the German or Spanish target speech directly: a frozen
w2v-BERT 2.0 encoder, an AR temporal decoder (source text, target text and
codec stream c0 with one shared head), and an NAR depth decoder (c1-c15)
conditioned on a CAMPPlus speaker prefix and a source codec prompt.

## Contents

- `inference/infer.py`: command-line and Python inference.
- `inference/model_core.py`: model definition.
- `training/train.py`, `training/data.py`: training; configurations in `training/configs/`.
- `training/tokenizers/`: the En-De and En-Es SentencePiece-8k text tokenizers.
- `scripts/data/`: builds training manifests (target Mimi codes, Mimi source prompts, CAMPPlus embeddings).
- `scripts/clean_checkpoint.py`: turns a training checkpoint into the release layout.
- `docs/model_dependencies.md`: external models and how to obtain them.

## Setup

Tested with Python 3.9 and the versions in `requirements.txt` (torch 2.2.2,
torchaudio 2.2.2, transformers 4.57.6, datasets 4.5.0, accelerate 1.10.1):

```
pip install -r requirements.txt
```

w2v-BERT 2.0 and Mimi download from the Hugging Face Hub on first use. CAMPPlus needs two local pieces, described in
`docs/model_dependencies.md`: the Apache-2.0 weights
(`campplus_cn_en_common.pt`) and a clone of
[Seed-VC](https://github.com/Plachtaa/seed-vc), whose GPL-3.0 model definition
is not redistributed here.

Checkpoints (En-De and En-Es) are on the Hugging Face Hub at
[TedZhangHao/DirectS2ST](https://huggingface.co/TedZhangHao/DirectS2ST):

```
hf download TedZhangHao/DirectS2ST --include "en-es/*" --local-dir checkpoints
```

Each checkpoint directory contains `pytorch_model.bin`, `model_config.json`
and `text_tokenizer/`.

## Translate one utterance

```
python inference/infer.py \
  --checkpoint-dir checkpoints/en-es \
  --campplus-model-root /path/to/seed-vc \
  --campplus-checkpoint-path /path/to/campplus_cn_en_common.pt \
  --source-wav input_en.wav --output-wav output_es.wav
```

`--manifest file.tsv --output-dir out/` translates a TSV of utterances
(`id` and `audio_path` columns by default; `--audio-root` resolves relative
paths) and writes `<id>.wav` plus `generation.tsv`. Consecutive rows are
encoded in batches of `--batch-size` (default 30, as in training-time
evaluation); the Mimi source prompt and CAMPPlus embedding are computed over
the zero-padded batch, so outputs depend on the batch composition.
`--batch-size 1` translates each utterance on its own, as `--source-wav` does.

## Training

### 1. Build the manifests

Start from a TSV per split with `src_audio`, `tgt_audio`, a source-text column
(`sentence`) and a target-text column (`translation`), e.g. the VC-Dub pairs
from `../VC-DUB`. Then add the precomputed columns:

```
python scripts/data/encode_target_mimi.py --input-tsv train.tsv --output-tsv train_mimi.tsv
python scripts/data/precompute_mimi_source_acoustic_embeddings.py \
  --input-tsv train_mimi.tsv --output-dir feats/ --audio-col src_audio
python scripts/data/precompute_campplus_embeddings.py \
  --input-tsv feats/train_mimi_mimi_source_acoustic_meanstd.tsv --output-dir feats/ \
  --audio-col src_audio --embedding-column-prefix campplus_source \
  --model-root /path/to/seed-vc --checkpoint /path/to/campplus_cn_en_common.pt
python scripts/data/precompute_campplus_embeddings.py \
  --input-tsv feats/train_mimi_mimi_source_acoustic_meanstd_campplus192.tsv --output-dir feats/ \
  --audio-col tgt_audio --embedding-column-prefix campplus_target --suffix campplus192_target \
  --model-root /path/to/seed-vc --checkpoint /path/to/campplus_cn_en_common.pt
```

and point `data.train_tsv` / `data.dev_tsv` in the config at the final TSVs.

### 2. Train

```
torchrun --nproc_per_node=1 training/train.py --config training/configs/es.yaml \
  --campplus-model-root /path/to/seed-vc --campplus-checkpoint-path /path/to/campplus_cn_en_common.pt
```

`training/configs/{de,es}.yaml` follow the paper: the temporal and depth
decoders are trained from random initialisation with the w2v-BERT 2.0 encoder,
Mimi and CAMPPlus frozen, on one GPU with batch size 8 in bf16, and a
polynomially decayed learning rate starting at 1e-4. Loss weights: source text
0.4, target text 0.6, C0 1.0, depth 3.0; the training-only text-to-text and
text-to-C0 objectives 0.3 and 0.5, and the speaker-identity loss 0.1.
`scripts/clean_checkpoint.py` turns a training checkpoint into the layout
`inference/infer.py` loads.
