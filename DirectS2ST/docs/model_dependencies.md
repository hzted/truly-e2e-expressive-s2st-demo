# External models

| Component | Source | Notes |
| --- | --- | --- |
| Speech encoder: w2v-BERT 2.0 | https://huggingface.co/facebook/w2v-bert-2.0 | Downloaded automatically. |
| Codec: Mimi | https://huggingface.co/kyutai/mimi | Downloaded automatically. Used for the source prompt and for decoding the generated codes. |
| Speaker encoder weights: CAM++ (`campplus_cn_en_common.pt`) | https://modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced | Apache-2.0. Pass the local file with `--campplus-checkpoint-path`. |
| Speaker encoder code: CAM++ DTDNN | https://github.com/Plachtaa/seed-vc (`modules/campplus/DTDNN.py`) | GPL-3.0, not included. Clone Seed-VC and pass its root with `--campplus-model-root`. |
| Text tokenizers (SentencePiece, 8k) | this repository | `training/tokenizers/`, and `text_tokenizer/` inside each checkpoint. |

The checkpoints are on the Hugging Face Hub:
https://huggingface.co/TedZhangHao/DirectS2ST

## Checkpoint export

`scripts/clean_checkpoint.py` turns a training checkpoint into the layout
`inference/infer.py` loads (`pytorch_model.bin`, `model_config.json`,
`text_tokenizer/`):

```
python scripts/clean_checkpoint.py \
  --source-checkpoint-dir outputs/es/checkpoint-NNNNN \
  --output-dir checkpoints/en-es
```
