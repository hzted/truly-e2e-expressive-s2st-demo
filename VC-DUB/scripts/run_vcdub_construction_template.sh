#!/usr/bin/env bash
set -euo pipefail

# Template runner for VC-DUB construction.
#
# This script covers only data construction/cleaning. Whisper ASR and generated
# audio metrics belong to VC-DUB/evaluation and are intentionally not called here.

PAIR="${PAIR:-en_es}"  # en_es or en_de
WORK_ROOT="${WORK_ROOT:-/path/to/vcdub_work/${PAIR}}"
PREPROCESSED_MANIFEST="${PREPROCESSED_MANIFEST:-/path/to/preprocessed_pair_manifest.tsv}"
ALIGNED_METADATA_TSV="${ALIGNED_METADATA_TSV:-/path/to/aligned_pair_metadata.tsv}"
DNSMOSPRO_CMD="${DNSMOSPRO_CMD:-python /path/to/DNSMOSPro/infer.py --audio {audio}}"
DNSMOSPRO_SCORE_KEY="${DNSMOSPRO_SCORE_KEY:-}"
DNSMOSPRO_SCORE_REGEX="${DNSMOSPRO_SCORE_REGEX:-}"
DNSMOSPRO_COMBINE="${DNSMOSPRO_COMBINE:-}"  # min or mean; must match original experiment logs
DNSMOSPRO_SELECTION_MODE="${DNSMOSPRO_SELECTION_MODE:-}"  # cutoff or target_keep_pairs
DNSMOSPRO_CUTOFF="${DNSMOSPRO_CUTOFF:-}"
DNSMOSPRO_TARGET_KEEP_PAIRS="${DNSMOSPRO_TARGET_KEEP_PAIRS:-}"

SRC_LID="${SRC_LID:-eng}"
if [ "${PAIR}" = "en_de" ]; then
  TGT_LID="${TGT_LID:-deu}"
  REPORTED_TARGET_KEEP_PAIRS=147639
  DEV_TEST_RATIO="${DEV_TEST_RATIO:-0.11}"
  TEST_SIZE="${TEST_SIZE:-504}"
else
  TGT_LID="${TGT_LID:-spa}"
  REPORTED_TARGET_KEEP_PAIRS=90000
  DEV_TEST_RATIO="${DEV_TEST_RATIO:-0.12}"
  TEST_SIZE="${TEST_SIZE:-504}"
fi

mkdir -p "${WORK_ROOT}"

echo "[1/7] aligned-pair metadata preparation"
echo "      Input preprocessed manifest: ${PREPROCESSED_MANIFEST}"
echo "      Input aligned metadata:      ${ALIGNED_METADATA_TSV}"

echo "[2/7] ClearVoice/Demucs preprocessing"
echo "      Run ClearVoice + Demucs externally, then write PREPROCESSED_MANIFEST."

echo "[3/7] MMS-LID filtering"
python -u scripts/score_mms_lid_for_filtering.py \
  --input-tsv "${PREPROCESSED_MANIFEST}" \
  --out-dir "${WORK_ROOT}/mms_lid" \
  --src-audio-col pre_src \
  --tgt-audio-col pre_tgt \
  --expected-src-lang "${SRC_LID}" \
  --expected-tgt-lang "${TGT_LID}" \
  --write-filtered-manifest

echo "[4/7] Sortformer speaker filtering"
python -u scripts/score_sortformer_for_filtering.py \
  --input-tsv "${WORK_ROOT}/mms_lid/lid_pass_manifest.tsv" \
  --out-dir "${WORK_ROOT}/sortformer" \
  --src-audio-col pre_src \
  --tgt-audio-col pre_tgt

python -u scripts/filter_sortformer_pair_strict.py \
  --pair-results "${WORK_ROOT}/sortformer/sortformer_pair_results.tsv" \
  --out-dir "${WORK_ROOT}/sortformer"

echo "[5/7] DNSMOSPro scoring and quality selection"
dnsmos_parse_args=()
if [ -n "${DNSMOSPRO_SCORE_KEY}" ]; then
  dnsmos_parse_args+=(--score-key "${DNSMOSPRO_SCORE_KEY}")
fi
if [ -n "${DNSMOSPRO_SCORE_REGEX}" ]; then
  dnsmos_parse_args+=(--score-regex "${DNSMOSPRO_SCORE_REGEX}")
fi
if [ "${#dnsmos_parse_args[@]}" -eq 0 ]; then
  echo "ERROR: set DNSMOSPRO_SCORE_KEY or DNSMOSPRO_SCORE_REGEX so DNSMOSPro parsing is explicit." >&2
  exit 2
fi
if [ -z "${DNSMOSPRO_COMBINE}" ]; then
  echo "ERROR: set DNSMOSPRO_COMBINE=min or DNSMOSPRO_COMBINE=mean after confirming the original experiment logs." >&2
  exit 2
fi
selection_args=()
case "${DNSMOSPRO_SELECTION_MODE}" in
  cutoff)
    if [ -z "${DNSMOSPRO_CUTOFF}" ]; then
      echo "ERROR: DNSMOSPRO_SELECTION_MODE=cutoff requires DNSMOSPRO_CUTOFF." >&2
      exit 2
    fi
    selection_args+=(--cutoff "${DNSMOSPRO_CUTOFF}")
    ;;
  target_keep_pairs)
    if [ -z "${DNSMOSPRO_TARGET_KEEP_PAIRS}" ]; then
      echo "ERROR: DNSMOSPRO_SELECTION_MODE=target_keep_pairs requires DNSMOSPRO_TARGET_KEEP_PAIRS." >&2
      echo "       Reported retained counts were ${REPORTED_TARGET_KEEP_PAIRS} for ${PAIR}, but do not use them as reconstruction rules until confirmed." >&2
      exit 2
    fi
    selection_args+=(--target-keep-pairs "${DNSMOSPRO_TARGET_KEEP_PAIRS}")
    ;;
  *)
    echo "ERROR: set DNSMOSPRO_SELECTION_MODE to cutoff or target_keep_pairs after confirming the original experiment logs." >&2
    exit 2
    ;;
esac

python -u scripts/score_dnsmospro_for_filtering.py \
  --input-tsv "${WORK_ROOT}/sortformer/sortformer_pair_pass_strict.tsv" \
  --out-dir "${WORK_ROOT}/dnsmospro" \
  --id-col sample_id \
  --src-audio-col pre_src \
  --tgt-audio-col pre_tgt \
  --combine "${DNSMOSPRO_COMBINE}" \
  --dnsmospro-cmd "${DNSMOSPRO_CMD}" \
  "${dnsmos_parse_args[@]}"

python -u scripts/select_dnsmospro_quality_subset.py \
  --manifest-tsv "${WORK_ROOT}/sortformer/sortformer_pair_pass_strict.tsv" \
  --score-tsv "${WORK_ROOT}/dnsmospro/dnsmospro_quality_pairs.tsv" \
  --out-dir "${WORK_ROOT}/quality_selection" \
  --id-col sample_id \
  --score-col combined_dnsmospro \
  "${selection_args[@]}"

echo "[6/7] train/dev/test split assignment"
python -u scripts/build_vcdub_splits.py \
  --selected-manifest-tsv "${WORK_ROOT}/quality_selection/dnsmospro_filtered_manifest.tsv" \
  --aligned-metadata-tsv "${ALIGNED_METADATA_TSV}" \
  --id-col sample_id \
  --source-audio-col pre_src \
  --target-audio-col pre_tgt \
  --source-text-col src_text \
  --target-text-col tgt_text \
  --out-dir "${WORK_ROOT}/splits" \
  --dev-test-ratio "${DEV_TEST_RATIO}" \
  --test-size "${TEST_SIZE}" \
  --seed 42 \
  --overwrite

echo "[7/7] Optional final voice-conversion materialization"
echo "      SPLIT_TSV=${WORK_ROOT}/splits/train_metadata.tsv SEEDVC_ROOT=/path/to/seed-vc-main OUTPUT_ROOT=${WORK_ROOT}/vc_outputs/train bash scripts/voice_conversion/run_voice_conversion_materialization.sh"
