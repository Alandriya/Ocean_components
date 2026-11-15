# Forecast_1d: Initial point forecasting models

This folder contains a separate training, evaluation and plotting framework for
**ten offshore point series** generated with `Forecast_1d/extract_points.py`.
All model output is in the original physical units; no future values are
used for parameter fitting.

## Paths

The code is in `/home/aosipova/Ocean_Components/Forecast_1d/` on the cluster.
The data, after local extraction and upload, is under
`/home/aosipova/OceanFull/DATA/Forecast_1d/atlantic10_native_1979-2026/`.

The `DATA_ROOT` in the cluster's `project_paths.py` must be set to
`/home/aosipova/OceanFull/` and point to the transferred data, not to the laptop.
All model output is written to
`DATA_ROOT/Forecast/Results/Forecast_1d/`.

No changes are made to `Forecast_nn`, `main.py`, or existing 1D extraction logic.

## Models

- `persistence`: repeat the last observed day for all forecast leads.
- `input_mean`: repeat the mean of the last 30 observed days for all leads.
- `sarima`: independent univariate SARIMA per location and flux channel.

The default specification is `order=(1,0,0)`, `seasonal_order=(0,0,0,0)`,
`trend='c'`: a linear AR(1) with intercept.
For this **default** specification, `--backend auto` uses a NumPy
**conditional least-squares** solver, not statsmodels maximum likelihood.
It is statistically equivalent in model structure, but the fitted coefficients
need not be numerically identical to statsmodels SARIMAX MLE.
For a true statsmodels SARIMAX fit of the same specification, use
`--backend statsmodels`, which requires the `statsmodels` package.
For any other order/seasonal specification, the `statsmodels` backend is
selected automatically and must be installed in the runtime environment.

The two training-free baselines are always evaluated by `test.py` and require
no checkpoint or model fitting. Ten locations and both channels share exactly
the same forecast origins and scoring semantics.

## Chronological protocol

The final 365 days are `test`, the preceding 730 days are `val`, and all earlier
observations are `train` by default. For example, with data through 2026-10-30,
`test` covers the last 365 target days. To change these split lengths,
pass matching `--val-days` / `--test-days` / `--in-len` / `--out-len` arguments
to both `train` and `test`.

AR/SARIMA **parameters** are estimated only from `train`; evaluation updates
filter state with observations available by the forecast origin. A 3-day
prediction at time `t` does not use observations at `t+1` or `t+2`.
Forecast origins near the start/end of the held-out split are chosen so all
forecast targets lie inside that split. Use the model's fixed 1/2/3-day
forecast, not a one-step forecast using the true previous future day.

Input mean baseline uses the most recent `--in-len` observed days, including
origin `t`. Training is never refitted during validation/test.

## Local or container CLI

Run the following commands from the Git repository root (where
`project_paths.py` is located):

```shell
python -m Forecast_1d.train --model sarima
python -m Forecast_1d.test --model all --split val
python -m Forecast_1d.test --model all --split test
python -m Forecast_1d.plotting.plot_results --model sarima --split test
python -m Forecast_1d.plotting.plot_results --model sarima --split test --point P3 --horizon 1 3 --show-baselines
```

Custom SARIMA example:

```shell
python -m Forecast_1d.train --model sarima --order 2,0,1 --seasonal-order 1,0,0,7 --backend statsmodels --overwrite
python -m Forecast_1d.test --model all --split test --overwrite
```

If overwriting model parameters, rerun `test` with `--overwrite`. Existing
results are not overwritten otherwise.

## Cluster GPU launch (optional)

These launchers explicitly **request an A100 GPU** in the `intel-a100-pci5`
partition, satisfying the partition policy. The initial SARIMA/AR(1) calculation
is CPU-based even inside the GPU-enabled PyTorch container; the GPU itself
is not used by these statistical models. These launchers are provided to match
the project's working container environment, and GPU models can be added later.

Before submitting, ensure `/home/aosipova/Ocean_Components/Cluster/` exists;
Slurm stores stdout/stderr there. The cluster's `docker-id` must expose both
`Ocean_Components` and `OceanFull` inside the container.

```shell
mkdir -p /home/aosipova/Ocean_Components/Cluster
sbatch Forecast_1d/jobs/train.sbatch --model sarima
sbatch Forecast_1d/jobs/test.sbatch --model all --split test
sbatch Forecast_1d/jobs/plot.sbatch --model sarima --split test
```

The default AR(1) backend needs only NumPy. The plotting step requires
Matplotlib. Custom SARIMA requires Statsmodels; inspect/install it in the
container before selecting a custom specification.

## Saved results

```text
DATA_ROOT/Forecast/Results/Forecast_1d/
├── sarima/
│   ├── params.npz
│   └── train_metadata.json
├── evaluations/
│   ├── val/
│   │   ├── persistence_predictions.npz
│   │   ├── input_mean_predictions.npz
│   │   ├── sarima_predictions.npz
│   │   └── metrics.json
│   └── test/
│       ├── persistence_predictions.npz
│       ├── input_mean_predictions.npz
│       ├── sarima_predictions.npz
│       └── metrics.json
└── plots/
    └── test/
        └── sarima/
            ├── P1_sarima_h1_test.png
            ├── P1_sarima_h2_test.png
            └── ...
```

Each plot shows **two vertically stacked time series**: sensible above latent.
Black is the actual observed series; red is the chosen model's rolling
prediction on its target date. `--show-baselines` adds dashed persistence and
input-window mean curves for comparison. Separate figures are produced for
horizons 1, 2 and 3, and for each selected geographic point.

The test report contains joint RMSE, MAE, bias, and MSE skill versus persistence,
plus metrics per channel, location and lead time.

## Tests

```shell
python -m unittest Forecast_1d.tests.test_forecasting -v
```

The tests use synthetic data and include checks against future-target leakage,
checkpoint/split consistency, and end-to-end training/testing/plotting.

## References

- [SARIMAX, statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)
- [NumPy NPY/NPZ storage](https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html)
- [Slurm sbatch options](https://slurm.schedmd.com/sbatch.html)
