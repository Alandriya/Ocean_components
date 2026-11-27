#!/usr/bin/env bash
# Submit the extract points workflow to Slurm.
#SBATCH --job-name=forecast_1d_extract
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/home/aosipova/OceanFull/Forecast/logs/forecast_1d_extract_%j.out
#SBATCH --error=/home/aosipova/OceanFull/Forecast/logs/forecast_1d_extract_%j.err

# Extract the 15-point dataset on a CPU node.
# NumPy is required by Forecast_1d.extract_points; no GPU or Docker is used.

set -euo pipefail

CODE_ROOT="/home/aosipova/Ocean_Components"
DATA_ROOT="/home/aosipova/OceanFull"

cd "$CODE_ROOT"

resolve_python() {
    local candidate="$1"
    if [[ "$candidate" == */* ]]; then
        [[ -x "$candidate" ]] && printf '%s\n' "$candidate"
    else
        command -v "$candidate" 2>/dev/null || true
    fi
}

PYTHON_BIN=""

if [[ -n "${PYTHON_1D:-}" ]]; then
    resolved="$(resolve_python "$PYTHON_1D")"
    if [[ -z "$resolved" ]]; then
        echo "[error] PYTHON_1D does not point to an executable: $PYTHON_1D" >&2
        exit 127
    fi
    if ! "$resolved" -c 'import sys, numpy; assert sys.version_info >= (3, 8)' >/dev/null 2>&1; then
        echo "[error] PYTHON_1D=$resolved must provide Python >=3.8 and NumPy." >&2
        exit 127
    fi
    PYTHON_BIN="$resolved"
else
    candidates=(
        "/home/aosipova/venv2/bin/python"
        "/home/aosipova/venv2/bin/python3"
        "python3"
        "/usr/bin/python3"
        "/usr/local/bin/python3"
    )

    for candidate in "${candidates[@]}"; do
        resolved="$(resolve_python "$candidate")"
        [[ -n "$resolved" ]] || continue

        if "$resolved" -c 'import sys, numpy; assert sys.version_info >= (3, 8)' >/dev/null 2>&1; then
            PYTHON_BIN="$resolved"
            break
        fi
    done
fi

if [[ -z "$PYTHON_BIN" ]]; then
    cat >&2 <<'EOF'
[error] No host Python >=3.8 with NumPy was found on this CPU node.

Forecast_1d.extract_points requires NumPy to read and write the .npy arrays.
This job deliberately does not start Docker and does not request a GPU.
Set PYTHON_1D to an existing Python executable that already has NumPy, for example:

  PYTHON_1D=/path/to/python bash Forecast_1d/jobs/submit_1d.sh extract --dry-run

If no such environment is available on the CPU nodes, run extraction on the
local computer as before and copy the resulting atlantic15_native_1979-2026
directory to:

  /home/aosipova/OceanFull/DATA/Forecast_1d/
EOF
    exit 127
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "[extract] job=${SLURM_JOB_ID:-interactive}"
echo "[extract] node=$(hostname)"
echo "[extract] partition=${SLURM_JOB_PARTITION:-unknown}"
echo "[extract] python=$PYTHON_BIN"
"$PYTHON_BIN" -c 'import sys, numpy; print("[extract] Python", sys.version.split()[0]); print("[extract] NumPy", numpy.__version__)'
echo "[extract] DATA_ROOT=$DATA_ROOT"
echo "[extract] points=$CODE_ROOT/Forecast_1d/points.json"
echo "[extract] output=$DATA_ROOT/DATA/Forecast_1d/atlantic15_native_1979-2026"

exec srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" \
    "$PYTHON_BIN" -u -m Forecast_1d.extract_points \
    --data-root "$DATA_ROOT" \
    --expected-data-root "$DATA_ROOT" \
    "$@"
