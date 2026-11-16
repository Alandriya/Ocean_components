#!/bin/bash
# Create the shared log directory on the login node before Slurm opens log files.
# Usage:
#   bash Forecast_1d/jobs/submit_1d.sh train --model sarima
#   bash Forecast_1d/jobs/submit_1d.sh test --model all --split test
#   bash Forecast_1d/jobs/submit_1d.sh plot --model sarima --split test

set -euo pipefail

CODE_ROOT="/home/aosipova/Ocean_Components"
DATA_ROOT="/home/aosipova/OceanFull"
LOG_DIR="$DATA_ROOT/Forecast/logs"
JOB_DIR="$CODE_ROOT/Forecast_1d/jobs"

if [[ ! -d "$CODE_ROOT" ]]; then
    echo "[error] Code directory does not exist: $CODE_ROOT" >&2
    exit 2
fi

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
    echo "Usage: bash Forecast_1d/jobs/submit_1d.sh {train|test|plot} [job arguments...]" >&2
    exit 2
fi
shift

case "$MODE" in
    train|test|plot)
        JOB_SCRIPT="$JOB_DIR/$MODE.sbatch"
        ;;
    *)
        echo "[error] Unknown mode '$MODE'. Choose train, test, or plot." >&2
        exit 2
        ;;
esac

if [[ ! -f "$JOB_SCRIPT" ]]; then
    echo "[error] Slurm script not found: $JOB_SCRIPT" >&2
    exit 2
fi

# Slurm opens stdout/stderr files before the batch script starts.
# Therefore this directory must be created before calling sbatch.
mkdir -p "$LOG_DIR"

echo "[submit] mode=$MODE"
echo "[submit] logs=$LOG_DIR"
exec sbatch \
    --output="$LOG_DIR/%x_%j.out" \
    --error="$LOG_DIR/%x_%j.err" \
    "$JOB_SCRIPT" "$@"
