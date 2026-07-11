# Model And Dependency Preparation

This file records the upstream model/tool sources used by the construction and
evaluation wrappers. The reviewer GitHub mirror does not redistribute model
weights or checkpoints. Users should install/download each component locally and
pass paths/model IDs through the documented command-line flags.

## Construction

| Stage | Tool / model | Upstream source | Local preparation |
| --- | --- | --- | --- |
| Denoising | ClearerVoice-Studio, `MossFormer2_SE_48K` | https://github.com/modelscope/ClearerVoice-Studio and https://huggingface.co/alibabasglab/MossFormer2_SE_48K | Clone/install ClearerVoice-Studio, allow it to download `MossFormer2_SE_48K` or pre-download the HF weights, then run preprocessing outside the template runner. |
| Vocal extraction | Demucs, `htdemucs` | https://github.com/facebookresearch/demucs | Install Demucs and let `demucs`/`demucs.pretrained.get_model("htdemucs")` download the model cache locally. |
| Language ID | MMS-LID, `facebook/mms-lid-126` | https://huggingface.co/facebook/mms-lid-126 | Install `transformers`/`torch`; the script loads the HF model ID unless your environment uses a local HF cache. |
| Diarization | Sortformer, `nvidia/diar_sortformer_4spk-v1` | https://huggingface.co/nvidia/diar_sortformer_4spk-v1 | Install NVIDIA NeMo ASR and log in/accept HF terms if required by the model card; the script calls `SortformerEncLabelModel.from_pretrained`. |
| Quality selection | DNSMOSPro | https://github.com/fcumlin/DNSMOSPro | Clone the exact DNSMOSPro commit used in the experiment, download its checkpoint according to that repo, and pass an explicit `DNSMOSPRO_CMD` plus `DNSMOSPRO_SCORE_KEY`/`DNSMOSPRO_SCORE_REGEX`. The exact commit/checkpoint/score key are still blockers for bit-for-bit reruns. |
| Final VC materialization | Seed-VC | https://github.com/Plachtaa/seed-vc | Clone Seed-VC locally, install its dependencies/checkpoints, and set `SEEDVC_ROOT=/path/to/seed-vc`. VC is optional and runs after filtering/splitting. |

## Evaluation

| Metric | Tool / model | Upstream source | Local preparation |
| --- | --- | --- | --- |
| BLASER 2.0 | SONAR / BLASER 2.0 | https://github.com/facebookresearch/SONAR and https://facebookresearch.github.io/stopes/docs/eval/blaser | Install SONAR/fairseq2-compatible dependencies. The wrapper loads SONAR speech/text encoders and BLASER models from the local model cache. |
| A.PCP, Vsim, local prosody | Stopes expressive evaluation | https://github.com/facebookresearch/stopes | Install the Stopes checkout/version used for the paper. The reviewer wrapper calls Stopes modules directly. |
| Vsim checkpoint | WavLM-derived fine-tuned checkpoint | https://huggingface.co/microsoft/wavlm-large for the base model | The paper wrapper requires `--wavlm-ckpt /path/to/wavlm_large_finetune.pth`. This fine-tuned checkpoint is not included in the current artifact. |
| DNSMOSPro quality metric | DNSMOSPro | https://github.com/fcumlin/DNSMOSPro | Same as construction, but run on generated audio with `evaluation/scripts/run_dnsmospro.py`. The command fails if all samples fail to produce a valid score. |
| Optional ASR | Whisper large-v3 | https://huggingface.co/openai/whisper-large-v3 | Install `transformers`, `torch`, `librosa`, and `soundfile`. Whisper is disabled by default and is not a construction dependency. The bundled wrapper only generates transcripts; WER/CER/ASR-BLEU aggregation must be added separately if those metrics are claimed. |

## Pinning Rule

Before claiming exact reproducibility, record the exact git commit, package
version, model ID/checkpoint path, and score field for every external component.
Do not infer missing DNSMOSPro settings from aggregate counts.
