#!/bin/bash
# Launch the submit features workflow using the configured environment.
# Submit multivariate experiments using the established A100 Docker runtime.
# bash Forecast_1d/jobs/submit_features.sh train --model gru --feature-set all
set -euo pipefail
MODE="${1:-}"
if [[ "$MODE" != "train" && "$MODE" != "test" ]]; then
  echo "Usage: bash Forecast_1d/jobs/submit_features.sh {train|test} [model arguments...]" >&2
  exit 2
fi
shift
CODE_ROOT=/home/aosipova/Ocean_Components
LOG_DIR=/home/aosipova/OceanFull/Forecast/logs
mkdir -p "$LOG_DIR"
exec sbatch --output="$LOG_DIR/%x_%j.out" --error="$LOG_DIR/%x_%j.err" \
  "$CODE_ROOT/Forecast_1d/jobs/${MODE}_features.sbatch" "$@"
