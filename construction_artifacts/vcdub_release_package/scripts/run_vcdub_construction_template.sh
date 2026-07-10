#!/usr/bin/env bash
set -euo pipefail

# Template runner for VC-DUB construction.
# Fill in local roots before running. The release package does not redistribute audio.

ALIGNED_DUBBING_ROOT="${ALIGNED_DUBBING_ROOT:-/path/to/aligned_dubbing}"
WORK_ROOT="${WORK_ROOT:-/path/to/vcdub_work}"
PAIR="${PAIR:-en_es}"  # en_es or en_de

echo "[1/8] Prepare aligned source/target manifest for ${PAIR}"
echo "      Input root: ${ALIGNED_DUBBING_ROOT}"
echo "      Work root : ${WORK_ROOT}"

echo "[2/8] Run ClearVoice-Studio denoising locally"
echo "      Model: MossFormer2_SE_48K"

echo "[3/8] Run Demucs vocal extraction locally"
echo "      Model: htdemucs"

echo "[4/8] Run MMS-LID filtering"
python -u scripts/eval_mms_lid_filter.py --help

echo "[5/8] Run Sortformer diarization filtering"
python -u scripts/eval_sortformer_pair_filter.py --help
python -u scripts/filter_sortformer_pair_strict.py --help

echo "[6/8] Run DNSMOSPro quality scoring and scale-matched selection"
python -u scripts/eval_dnsmospro_quality.py --help
python -u scripts/select_dnsmospro_quality_subset.py --help

echo "[7/8] Build train/dev/test split manifests"
python -u scripts/build_vcdub_splits.py --help

echo "[8/8] Run voice conversion locally as final materialization"
echo "      Use the selected split manifests as input; generated audio is not part of this release."
