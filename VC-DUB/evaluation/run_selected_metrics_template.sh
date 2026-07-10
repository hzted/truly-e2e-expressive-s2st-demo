#!/usr/bin/env bash
set -euo pipefail

# Run the selected VC-DUB output-audio metrics only:
#   BLASER2.0, AutoPCP, speech-rate compliance 0.2/0.4,
#   syllable speech-rate Pearson, pause_wmean_duration_score, VSim,
#   and DNSMOSPro naturalness/MOS.
#
# This template assumes the evaluation scripts from the project workspace are
# available through VERIFY_SCRIPTS_ROOT. It does not run ASR-BLEU, EmoCos, F0,
# WER/CER, or any additional metrics.

EVAL_MANIFEST="${EVAL_MANIFEST:?Set EVAL_MANIFEST to an output evaluation TSV.}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR.}"
VERIFY_SCRIPTS_ROOT="${VERIFY_SCRIPTS_ROOT:-/path/to/Expressive_S2ST/verify_scripts}"
PYTHON="${PYTHON:-python}"

SOURCE_LANG="${SOURCE_LANG:-eng}"
HYPO_LANG="${HYPO_LANG:-spa}"
SAMPLE_FRAC="${SAMPLE_FRAC:-1.0}"
SEED="${SEED:-42}"
NUM_SHARDS="${NUM_SHARDS:-1}"
PARALLEL_JOBS="${PARALLEL_JOBS:-1}"
WAVLM_CKPT="${WAVLM_CKPT:-/path/to/wavlm_large_finetune.pth}"
DNSMOSPRO_CMD="${DNSMOSPRO_CMD:-python /path/to/DNSMOSPro/infer.py --audio {audio}}"
DNSMOSPRO_SCORE_KEY="${DNSMOSPRO_SCORE_KEY:-}"
DNSMOSPRO_SCORE_REGEX="${DNSMOSPRO_SCORE_REGEX:-}"

SOURCE_AUDIO_COL="${SOURCE_AUDIO_COL:-source_audio}"
HYPO_AUDIO_COL="${HYPO_AUDIO_COL:-hypo_audio}"
REFERENCE_AUDIO_COL="${REFERENCE_AUDIO_COL:-reference_audio}"
SOURCE_TEXT_COL="${SOURCE_TEXT_COL:-source_text}"
HYPO_TEXT_COL="${HYPO_TEXT_COL:-hypo_text}"
ID_COL="${ID_COL:-id}"

mkdir -p "${OUT_DIR}"

echo "[1/4] BLASER2.0"
"${PYTHON}" "${VERIFY_SCRIPTS_ROOT}/eval_blaser2_audio.py" \
  --manifest "${EVAL_MANIFEST}" \
  --output-dir "${OUT_DIR}/blaser2_audio" \
  --source-audio-col "${SOURCE_AUDIO_COL}" \
  --hypo-audio-col "${HYPO_AUDIO_COL}" \
  --reference-audio-col "${REFERENCE_AUDIO_COL}" \
  --id-col "${ID_COL}" \
  --source-lang "${SOURCE_LANG}"

echo "[2/4] Stopes selected metrics: AutoPCP, VSim, local prosody"
"${PYTHON}" "${VERIFY_SCRIPTS_ROOT}/eval_stopes_switch.py" \
  --input-tsv "${EVAL_MANIFEST}" \
  --out-dir "${OUT_DIR}/stopes_metrics" \
  --sample-frac "${SAMPLE_FRAC}" \
  --seed "${SEED}" \
  --num-shards "${NUM_SHARDS}" \
  --parallel-jobs "${PARALLEL_JOBS}" \
  --src-lang "${SOURCE_LANG}" \
  --tgt-lang "${HYPO_LANG}" \
  --src-audio-col "${SOURCE_AUDIO_COL}" \
  --tgt-audio-col "${HYPO_AUDIO_COL}" \
  --src-text-col "${SOURCE_TEXT_COL}" \
  --tgt-text-col "${HYPO_TEXT_COL}" \
  --id-col "${ID_COL}" \
  --speech-units "[syllable]" \
  --run-autopcp \
  --run-vsim \
  --run-local-prosody \
  --wavlm-ckpt "${WAVLM_CKPT}"

echo "[3/4] DNSMOSPro NAT"
"${PYTHON}" "$(dirname "$0")/score_dnsmospro_nat.py" \
  --manifest "${EVAL_MANIFEST}" \
  --out-dir "${OUT_DIR}/dnsmospro_nat" \
  --id-col "${ID_COL}" \
  --audio-col "${HYPO_AUDIO_COL}" \
  --dnsmospro-cmd "${DNSMOSPRO_CMD}" \
  --score-key "${DNSMOSPRO_SCORE_KEY}" \
  --score-regex "${DNSMOSPRO_SCORE_REGEX}"

echo "[4/4] Collect selected metrics only"
"${PYTHON}" "$(dirname "$0")/collect_selected_metrics.py" \
  --eval-root "${OUT_DIR}"

echo "Selected metrics written under: ${OUT_DIR}"
