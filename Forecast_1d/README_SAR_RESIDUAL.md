# Forecast_1d: Seasonal AR(1) residual forecasting, 60 -> 14 days

This is a minimal overlay on the existing **Forecast_1d_anomaly_patch.zip** code.
The new experiment is a diagnostic alternative to anomaly forecasting. It does
**not** replace the existing raw/anomaly modes or the Level-0 reference models.
No extraction, SSH, Docker, Slurm, or new launcher scripts are added.

## Definition

For each point and channel, fit the existing, train-only seasonal AR(1):

    baseline(t,h) = mu_train(calendar(t+h))
        + phi_train**h * [x(t) - mu_train(calendar(t))]

The Level-1 MLP, TCN or GRU observes the previous 60 days of *seasonal anomalies*
and forecasts the correction for each future lead and channel:

    z(t-L+1:t) = x(t-L+1:t) - mu_train(calendar(t-L+1:t))
    correction(t+1:t+H) = NN(z(t-L+1:t)) * OOF_residual_scale
    xhat(t+h) = baseline(t,h) + correction(t,h)

The neural outputs have **zero residual offset**; input anomalies are standardized
using a train-only mean and standard deviation. Corrections are **scaled but not
centered**, so the exact zero network output reproduces the baseline.
The last linear layer is zero-initialized for MLP, TCN and GRU. The untrained
zero-correction network is checkpoint candidate **epoch 0**. When
`--monitor sar` is used, an epoch is selected only if its aggregate validation
MSE-to-SAR ratio is below the best previous candidate (initially exactly 1.0).
The best validation checkpoint can thus be the baseline itself (`best_epoch=0`).
This does not guarantee improvement over SAR on the independent test period.

### Why train targets are not obtained from a SAR fit on the same targets

Fitting SAR on all train and then computing its in-sample residuals would train
NNs to correct a baseline that has already seen the target values via its fitted
parameters. This patch instead constructs **expanding out-of-fold SAR forecasts**
inside train only:

1. Reserve the first 33% of the training interval for the initial SAR fit.
2. Split the remaining 67% into four sequential folds.
3. For each fold, fit climatology and AR(1) strictly before the fold start;
   construct corrections for origins whose full horizon lies in that fold.
4. Form network inputs from the same fold's past-fitted climatology.
5. Fit the neural corrections on those OOF labels; validate with SAR fitted only
   through the original train boundary (2023-10-31 on the Atlantic15 dataset).

This prevents targets in any OOF fold from leaking into that fold's SAR fit.
Only **training** values are used for normalization. Validation and test never
contribute to fitted SAR, climatology, scaling, optimization, or training labels.
The fold-wise and full-train SAR climatologies are not identical; that is a
legitimate source of train/validation distribution shift to be measured.

The `seasonal_ar1` reference is the existing `models/baselines.py` implementation;
it is **not** the `sarima` CLI model, which currently defaults to ordinary AR(1).
There is **no** prerequisite `train --model sarima` command.

## Frozen training settings

- `in-len 60`, `out-len 14`.
- AdamW: `learning-rate 0.0003`, `weight-decay 0.0001`.
- `scheduler plateau`, `lr-patience 3`, `lr-factor 0.5`, `min-lr 1e-6`.
- `patience 12`, `epochs 100`, `batch-size 128`, `dropout 0.1`.
- `grad-clip 1.0`, `seed 42` plus point index.
- `climatology-radius 15`, `residual-oof-folds 4`.
- `monitor sar` **explicit in the commands**. The objective remains normalized
  correction MSE, but checkpoint selection and scheduler compare equal-weighted
  per-channel/per-lead relative MSE against the exact SAR baseline on validation.
- All 15 points are selected automatically when `--points` is omitted.

## Install from the project root on the cluster

    cd /home/aosipova/Ocean_Components
    unzip -o Forecast_1d_sar_residual_patch.zip

The existing prepared dataset is used without modification:

    /home/aosipova/OceanFull/DATA/Forecast_1d/atlantic15_native_1979-2026

### Full Atlantic15 — training (3 Slurm GPU jobs)

    cd /home/aosipova/Ocean_Components
    BASE=/home/aosipova/OceanFull/Forecast/Results/Forecast_1d/atlantic15_sar_residual_run1
    for model in mlp tcn gru; do
        bash Forecast_1d/jobs/submit_1d.sh train \
            --model "$model" --target-mode sar_residual --monitor sar \
            --results-dir "$BASE" --device cuda
    done

**Wait for all three Slurm jobs to finish successfully.** `sbatch` only submits
jobs; it does not make the loop wait. Monitor `squeue -u aosipova` or log files.

### Full Atlantic15 — validation (sequential Slurm GPU jobs)

    bash Forecast_1d/jobs/submit_1d.sh test \
        --model level0_new --target-mode sar_residual \
        --split val --results-dir "$BASE" --device cpu

**Wait for completion** before the next job, because both update a shared
`metrics.json`. The `test.sbatch` script still runs in `intel-a100-pci5` even
when `--device cpu` is passed to Python.

    bash Forecast_1d/jobs/submit_1d.sh test \
        --model level1 --target-mode sar_residual \
        --split val --results-dir "$BASE" --device cuda

`level0_new` includes `seasonal_ar1`, `seasonal_climatology`, `ridge_var`,
`persistence`, and `input_mean`. `level1` includes the three neural models and
the two trivial baselines. The resulting scores are on identical forecast
origins and targets in **original flux units**.

### Independent test (after choosing the protocol on validation)

    bash Forecast_1d/jobs/submit_1d.sh test \
        --model level0_new --target-mode sar_residual \
        --split test --results-dir "$BASE" --device cpu

**Wait for completion**, then:

    bash Forecast_1d/jobs/submit_1d.sh test \
        --model level1 --target-mode sar_residual \
        --split test --results-dir "$BASE" --device cuda

### Gather rankings without scheduling the busy `intel` partition

On the login node, only if light standard-library JSON processing is permitted:

    python3 -E -S -m Forecast_1d.collect_results \
        --target-mode sar_residual --split val --results-dir "$BASE"
    python3 -E -S -m Forecast_1d.collect_results \
        --target-mode sar_residual --split test --results-dir "$BASE"

The `collect` launcher is unchanged; scheduling `submit_1d.sh collect` would
use the busy `intel` partition, so invoke the light Python collector only if
local cluster policy permits it.

### Optional pilot on P3/P5/P7/P9

Use a different `BASE`, and **add exactly** `--points P3 P5 P7 P9` to each
train and test command. Add `--points P3 P5 P7 P9` to collection as well.
Do not mix results of different point selections in one `metrics.json`.

### Optional diagnostic plots

    bash Forecast_1d/jobs/submit_1d.sh plot \
        --model gru --target-mode sar_residual --split val \
        --point P12 --show-baselines --results-dir "$BASE"

## Result paths and checks

    $BASE/h14/sar_residual/mlp/P1/best.pt
    $BASE/h14/sar_residual/mlp/P1/train_metadata.json
    $BASE/h14/sar_residual/gru/P12/history.json
    $BASE/h14/sar_residual/evaluations/val/P1_P2_..._P15/metrics.json
    $BASE/h14/sar_residual/evaluations/test/P1_P2_..._P15/metrics.json
    $BASE/h14/sar_residual/summary/*_ranking.csv

For each checkpoint inspect `target_mode`, `sar_residual.oof_fold_edges`,
`best_epoch`, `best_validation_sar_skill`, `training_config`, and
`dataset_fingerprint_sha256`. A zero best epoch means the SAR baseline was
retained instead of a learned correction. Existing `h14/anomaly/` and raw
`h14/mlp/`, `h14/tcn/`, `h14/gru/` paths remain unaffected.

Keep the same point selection, data split, context length, horizon,
climatology radius, and results root during training and evaluation. The
metadata fingerprint and radius are checked at inference. Never pass
`--overwrite` on a first run. Use a fresh `BASE` for an experimental variation
(seed, regularization, number of OOF folds, architecture size).

## Validation and caveats

The patch was tested on synthetic data, including zero-head equivalence to
Seasonal AR(1), four past-only OOF folds, anomaly/raw compatibility, train/test
metadata, and CPU collector. Real-cluster execution is not yet confirmed.
This setup tests whether past flux anomalies carry learnable information about
**SAR's conditional errors**. The residual model cannot create new information
about future wind, air temperature or humidity, which strongly control sensible
and latent surface fluxes. Small improvements, including no improvement, are
scientifically plausible.

References:
- Project source: https://github.com/Alandriya/Ocean_components
- NOAA flux methods: https://www.pmel.noaa.gov/ocs/flux-documentation/
- PyTorch AdamW: https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
- PyTorch ReduceLROnPlateau: https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
