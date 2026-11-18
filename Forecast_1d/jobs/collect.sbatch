#!/bin/bash
#SBATCH --job-name=forecast_1d_collect
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/home/aosipova/OceanFull/Forecast/logs/forecast_1d_collect_%j.out
#SBATCH --error=/home/aosipova/OceanFull/Forecast/logs/forecast_1d_collect_%j.err
set -euo pipefail

CODE_ROOT=/home/aosipova/Ocean_Components
RESULTS_ROOT=/home/aosipova/OceanFull/Forecast/Results/Forecast_1d/atlantic15
cd "$CODE_ROOT"

# CPU-only collector: use host Python and its standard library, not Docker.
# -E ignores PYTHON* environment variables; -S skips site-packages.
check_python() {
    "$1" -E -S -c 'import sys, argparse, csv, json, math, pathlib; sys.exit(0 if sys.version_info >= (3, 7) else 1)' >/dev/null 2>&1
}

PYTHON_BIN=""
if [[ -n "${PYTHON_1D:-}" ]]; then
    if ! check_python "$PYTHON_1D"; then
        echo "[error] PYTHON_1D=$PYTHON_1D is not a working Python 3.7+ interpreter" >&2
        exit 127
    fi
    PYTHON_BIN="$PYTHON_1D"
else
    for candidate in python3 /usr/bin/python3 /usr/local/bin/python3 /home/aosipova/venv2/bin/python /home/aosipova/venv2/bin/python3 /opt/conda/bin/python python; do
        if check_python "$candidate"; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
fi
if [[ -z "$PYTHON_BIN" ]]; then
    echo '[error] No working Python 3.7+ interpreter on this CPU node. Set PYTHON_1D to its executable path.' >&2
    exit 127
fi

printf '[collect] job=%s host=%s python=%s\n' "${SLURM_JOB_ID:-interactive}" "$(hostname)" "$PYTHON_BIN"
"$PYTHON_BIN" -E -S --version
# Pass the base results path explicitly to avoid importing project_paths.py.
# The collector adds h14 (or another requested horizon) itself.
exec "$PYTHON_BIN" -E -S -u -m Forecast_1d.collect_results \
    --results-dir "$RESULTS_ROOT" "$@"
