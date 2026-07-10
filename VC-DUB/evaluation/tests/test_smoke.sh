#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python -u evaluation/scripts/run_all_metrics.py \
  --manifest evaluation/examples/example_manifest.tsv \
  --out-dir /tmp/vcdub_evaluation_smoke \
  --config evaluation/configs/evaluation_config.json \
  --python python \
  --dry-run

test -s /tmp/vcdub_evaluation_smoke/per-example_metrics.tsv
test -s /tmp/vcdub_evaluation_smoke/aggregate_metrics.json
test -s /tmp/vcdub_evaluation_smoke/aggregate_metrics.tsv
