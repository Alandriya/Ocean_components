#!/usr/bin/env bash
# Eight historical input channels: sensible, latent, sst, sp, u10, v10, t2m, d2m.
# Outputs remain the two future fluxes over 14 days.
# Usage from the cluster code root:
#   bash Forecast_1d/jobs/run_era5_8ch.sh train
#   bash Forecast_1d/jobs/run_era5_8ch.sh val
#   bash Forecast_1d/jobs/run_era5_8ch.sh test
# Optional arguments are passed on (e.g. --points P3 P5 P7 P9).

set -euo pipefail
CODE_ROOT=/home/aosipova/Ocean_Components
cd "$CODE_ROOT"

if [[ $# -lt 1 ]]; then
  echo 'Usage: run_era5_8ch.sh {train|val|test} [extra options]' >&2
  exit 2
fi
ACTION="$1"
shift

COMMON=(
  --model gru
  --feature-set era5
  --target-mode anomaly
  --run-id seed42
  --seed 42
  --in-len 60
  --out-len 14
  --val-days 730
  --test-days 365
  --climatology-radius 15
  --device cuda
  --amp
)

case "$ACTION" in
  train)
    exec bash Forecast_1d/jobs/submit_features.sh train \
      "${COMMON[@]}" \
      --epochs 100 \
      --batch-size 128 \
      --learning-rate 0.0003 \
      --weight-decay 0.0001 \
      --patience 12 \
      --min-delta 0.00001 \
      --scheduler plateau \
      --lr-patience 3 \
      --lr-factor 0.5 \
      --min-lr 0.000001 \
      --grad-clip 1.0 \
      --dropout 0.1 \
      --hidden-size 64 \
      "$@"
    ;;
  val|test)
    exec bash Forecast_1d/jobs/submit_features.sh test \
      "${COMMON[@]}" \
      --split "$ACTION" \
      "$@"
    ;;
  *)
    echo "Unknown action: $ACTION. Expected train|val|test." >&2
    exit 2
    ;;
esac
