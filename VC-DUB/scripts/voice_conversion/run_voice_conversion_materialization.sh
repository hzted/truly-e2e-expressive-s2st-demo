#!/usr/bin/env bash
set -euo pipefail

# Materialize VC-DUB voice-converted audio from a selected split manifest.
#
# Required inputs:
#   SPLIT_TSV: VC-DUB split manifest with pre_src/pre_tgt columns.
#   SEEDVC_ROOT: local SeedVC checkout containing inference_batch_denoise_both.py.
#   OUTPUT_ROOT: output directory for pair TSVs, wav shards, logs, and merged manifest.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SPLIT_TSV="${SPLIT_TSV:?Set SPLIT_TSV to train/dev/test split manifest.}"
SEEDVC_ROOT="${SEEDVC_ROOT:?Set SEEDVC_ROOT to the local SeedVC repository.}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT for generated VC materialization outputs.}"

PYTHON="${PYTHON:-python}"
NUM_SHARDS="${NUM_SHARDS:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
CUDA_DEV="${CUDA_DEV:-0}"
PREFETCH_WORKERS="${PREFETCH_WORKERS:-4}"
PREFETCH_DEPTH="${PREFETCH_DEPTH:-16}"
SOURCE_AUDIO_COL="${SOURCE_AUDIO_COL:-pre_src}"
TARGET_AUDIO_COL="${TARGET_AUDIO_COL:-pre_tgt}"
ID_COL="${ID_COL:-id}"
OUTPUT_SR="${OUTPUT_SR:-16000}"
DEMUX_PREPROCESS="${DEMUX_PREPROCESS:-false}"
DEMUX_ON="${DEMUX_ON:-both}"
CLEARVOICE_DENOISE="${CLEARVOICE_DENOISE:-false}"
CLEARVOICE_ON="${CLEARVOICE_ON:-both}"
SAVE_PREPROCESSED_AUDIO="${SAVE_PREPROCESSED_AUDIO:-false}"

PAIR_TSV="${OUTPUT_ROOT}/pair_tsvs/all_pairs.tsv"
SHARD_ROOT="${OUTPUT_ROOT}/shards"
LOG_DIR="${OUTPUT_ROOT}/logs"
VC_AUDIO_ROOT="${OUTPUT_ROOT}/vc_wavs"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}" "${VC_AUDIO_ROOT}"

"${PYTHON}" "${SCRIPT_DIR}/build_voice_conversion_pairs.py" \
  --input-tsv "${SPLIT_TSV}" \
  --output-tsv "${PAIR_TSV}" \
  --output-audio-root "${VC_AUDIO_ROOT}" \
  --id-col "${ID_COL}" \
  --source-audio-col "${SOURCE_AUDIO_COL}" \
  --target-audio-col "${TARGET_AUDIO_COL}"

"${PYTHON}" "${SCRIPT_DIR}/split_seedvc_pairs.py" \
  --input-tsv "${PAIR_TSV}" \
  --shard-root "${SHARD_ROOT}" \
  --num-shards "${NUM_SHARDS}" \
  --output-subdir vc_wavs

pids=()
active=0
for shard_tsv in "${SHARD_ROOT}"/pair_tsvs/shard*.tsv; do
  shard_name="$(basename "${shard_tsv}" .tsv)"
  shard_dir="${SHARD_ROOT}/${shard_name}"
  mkdir -p "${shard_dir}"
  (
    export CUDA_VISIBLE_DEVICES="${CUDA_DEV}"
    cd "${SEEDVC_ROOT}"
    "${PYTHON}" inference_batch_denoise_both.py \
      --pair_tsv "${shard_tsv}" \
      --output_dir "${shard_dir}/vc_wavs" \
      --output_sr "${OUTPUT_SR}" \
      --demucs-preprocess "${DEMUX_PREPROCESS}" \
      --demucs-on "${DEMUX_ON}" \
      --clearvoice-denoise "${CLEARVOICE_DENOISE}" \
      --clearvoice-on "${CLEARVOICE_ON}" \
      --save_preprocessed_audio "${SAVE_PREPROCESSED_AUDIO}" \
      --prefetch-workers "${PREFETCH_WORKERS}" \
      --prefetch-depth "${PREFETCH_DEPTH}" \
      --skip_existing \
      > "${LOG_DIR}/${shard_name}.log" 2>&1
  ) &
  pids+=("$!")
  active=$((active + 1))
  if [ "${active}" -ge "${MAX_PARALLEL}" ]; then
    wait -n
    active=$((active - 1))
  fi
done
wait

"${PYTHON}" "${SCRIPT_DIR}/merge_seedvc_manifests.py" \
  --shard-root "${SHARD_ROOT}" \
  --output-tsv "${OUTPUT_ROOT}/merged/vc_manifest.tsv"

echo "Voice conversion materialization complete: ${OUTPUT_ROOT}/merged/vc_manifest.tsv"
