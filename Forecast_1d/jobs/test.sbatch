#!/bin/bash
#SBATCH --job-name=forecast_1d_test
#SBATCH --partition=intel-a100-pci5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/home/aosipova/OceanFull/Forecast/logs/forecast_1d_test_%j.out
#SBATCH --error=/home/aosipova/OceanFull/Forecast/logs/forecast_1d_test_%j.err
set -euo pipefail

CODE_ROOT=/home/aosipova/Ocean_Components
cd "$CODE_ROOT"

docker-id --gpus 1 -w "$CODE_ROOT" nvcr.io/nvidia/pytorch:25.02-py3 \
  /bin/bash "$CODE_ROOT/Forecast_1d/jobs/run_1d.docker" test "$@"
