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
cd "$CODE_ROOT"

docker-id -w "$CODE_ROOT" nvcr.io/nvidia/pytorch:25.02-py3 \
  /bin/bash "$CODE_ROOT/Forecast_1d/jobs/run_1d.docker" collect "$@"
