#!/usr/bin/env bash
set -euo pipefail

# Template runner for the VC-DUB construction procedure.
#
# This script is intentionally a template: external stages such as ClearVoice,
# Demucs, DNSMOSPro, Whisper, NeMo/Sortformer, and SeedVC should be installed in
# their own environments. Fill in the paths below, then run the stages you need.

PAIR="${PAIR:-en_es}"  # en_es or en_de
WORK_ROOT="${WORK_ROOT:-/path/to/vcdub_work/${PAIR}}"
PREPROCESSED_MANIFEST="${PREPROCESSED_MANIFEST:-/path/to/preprocessed_pair_manifest.tsv}"
ASR_META_TSV="${ASR_META_TSV:-/path/to/whisper_largev3_text_meta.tsv}"
DNSMOSPRO_CMD="${DNSMOSPRO_CMD:-python /path/to/DNSMOSPro/infer.py --audio {audio}}"

SRC_LID="${SRC_LID:-eng}"
TGT_LID="${TGT_LID:-spa}"
if [ "${PAIR}" = "en_de" ]; then
  TGT_LID="${TGT_LID:-deu}"
fi

mkdir -p "${WORK_ROOT}"

echo "[1/7] MMS-LID filtering"
python -u scripts/eval_mms_lid_filter.py \
  --input-tsv "${PREPROCESSED_MANIFEST}" \
  --out-dir "${WORK_ROOT}/mms_lid" \
  --src-audio-col pre_src \
  --tgt-audio-col pre_tgt \
  --expected-src-lang "${SRC_LID}" \
  --expected-tgt-lang "${TGT_LID}"

echo "[2/7] Sortformer diarization"
python -u scripts/eval_sortformer_pair_filter.py \
  --input-tsv "${WORK_ROOT}/mms_lid/lid_pass_manifest.tsv" \
  --out-dir "${WORK_ROOT}/sortformer" \
  --src-audio-col pre_src \
  --tgt-audio-col pre_tgt

python -u scripts/filter_sortformer_pair_strict.py \
  --pair-results "${WORK_ROOT}/sortformer/sortformer_pair_results.tsv" \
  --out-dir "${WORK_ROOT}/sortformer"

echo "[3/7] DNSMOSPro scoring"
python -u scripts/eval_dnsmospro_quality.py \
  --input-tsv "${WORK_ROOT}/sortformer/sortformer_pair_pass_strict.tsv" \
  --out-dir "${WORK_ROOT}/dnsmospro" \
  --src-audio-col pre_src \
  --tgt-audio-col pre_tgt \
  --dnsmospro-cmd "${DNSMOSPRO_CMD}"

echo "[4/7] Scale-matched quality selection"
if [ "${PAIR}" = "en_es" ]; then
  TARGET_KEEP="${TARGET_KEEP:-90000}"
else
  TARGET_KEEP="${TARGET_KEEP:-147639}"
fi
python -u scripts/select_dnsmospro_quality_subset.py \
  --manifest-tsv "${WORK_ROOT}/sortformer/sortformer_pair_pass_strict.tsv" \
  --score-tsv "${WORK_ROOT}/dnsmospro/dnsmospro_quality_pairs.tsv" \
  --out-dir "${WORK_ROOT}/quality_selection" \
  --score-col combined_dnsmospro \
  --target-keep-pairs "${TARGET_KEEP}"

echo "[5/7] Prepare Whisper large-v3 text metadata"
echo "      Produce ASR_META_TSV=${ASR_META_TSV} with id/text/audio columns before splitting."

echo "[6/7] Build train/dev/test manifests"
python -u scripts/build_vcdub_splits.py \
  --input-tsv "${ASR_META_TSV}" \
  --join-manifest-tsv "${WORK_ROOT}/quality_selection/dnsmospro_filtered_manifest.tsv" \
  --input-id-col path \
  --join-id-col id \
  --source-text-col out_sentence \
  --target-text-col translation \
  --source-audio-col pre_src \
  --target-audio-col pre_tgt \
  --out-dir "${WORK_ROOT}/splits" \
  --dev-test-ratio 0.15 \
  --test-size 500 \
  --seed 42 \
  --overwrite

echo "[7/7] Voice conversion as final local materialization"
echo "      Example:"
echo "      SPLIT_TSV=${WORK_ROOT}/splits/train_asr.tsv SEEDVC_ROOT=/path/to/seed-vc-main OUTPUT_ROOT=${WORK_ROOT}/vc_outputs/train bash scripts/voice_conversion/run_voice_conversion_materialization.sh"
