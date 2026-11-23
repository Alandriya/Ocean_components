# Forecast_1d: training on historical ERA5 and flux statistics

## 1. Data placement (cluster)

Your Windows build writes:

`D:\Nastya\Data\OceanFull\DATA\Forecast_1d\atlantic15_native_1979-2026\features_era5\`

Transfer the **whole `features_era5` folder**, without renaming files, to:

`/home/aosipova/OceanFull/DATA/Forecast_1d/atlantic15_native_1979-2026/features_era5/`

The folder must contain `features.npy`, `targets.npy`, `exogenous.npy`, `dates.npy`, `manifest.json`, and `points.json`. Existing files one directory **above** `features_era5` (`centers.npy`, `dates.npy`, `points.json`, etc.) must also be present on the cluster; the scripts use them to check date alignment and to anchor the existing evaluation periods. If the original centers differ from newly assembled targets, the scripts stop instead of silently shifting data.

For example, connect with WinSCP and copy the whole Windows folder into the parent `atlantic15_native_1979-2026` directory on the cluster. **No original large gridded ERA5 arrays or GRIB files are needed on the cluster for this training**; the feature extraction is complete.

Check files on the cluster:

```bash
ls -lh /home/aosipova/OceanFull/DATA/Forecast_1d/atlantic15_native_1979-2026/features_era5/
```

The code lives in `/home/aosipova/Ocean_Components/Forecast_1d/`, **not** under the data root. Extract this ZIP into `/home/aosipova/Ocean_Components/`, preserving its `Forecast_1d/` structure. The old `train.py`, `test.py`, `jobs/submit_1d.sh`, legacy results and checkpoints are untouched.

In the cluster copy of `project_paths.py`, `DATA_ROOT` must be `/home/aosipova/OceanFull`.

## 2. Inputs and target

Default input is `[batch,60,20]` and target is `[batch,14,2]` (sensible/latent). The first two input features are the **history** of both heat fluxes; the next six are SST, surface pressure (`sp`), `u10`, `v10`, `t2m`, `d2m`; the remaining twelve are the existing `mean/std` of sensible/latent patches 3×3, 7×7 and 15×15. All 20 channels are used if `--feature-set all` (default). `--feature-set era5` uses the first 8, `--feature-set flux` uses the first 2.

The output is always **only 2 future flux channels**. No actual future ERA5 observations are used as network inputs. Target seasonal anomalies are fitted using the training partition only; test predictions are reconstructed into original flux units. Every feature is normalized **per point/channel using the training period only**. Models predict all 14 leads directly (no autoregressive reuse of future ground truth).

`--target-mode anomaly` (default) allows a direct comparison with previous Anomaly GRU/TCN/MLP. `--target-mode raw` is available too. The earlier SAR-residual mode is not implemented in these scripts; it remains available in the original train/test pipeline. Also, existing `patchtst` and `dlinear` implementations explicitly require 2 input channels, so they are deliberately **not** advertised as 20-channel models. The new scripts support GRU, TCN, MLP, N-BEATS, N-HiTS and TiDE, all via existing `models/neural.py`.

## 3. Slurm/Docker commands

All commands below run **on the cluster** from `/home/aosipova/Ocean_Components/` and use the **existing** `docker-id` + NGC PyTorch A100 environment. `submit_features.sh` creates the log directory automatically.

Pilot for 4 points, 20 input channels, default 60→14 and 100 epochs:

```bash
bash Forecast_1d/jobs/submit_features.sh train \
  --model gru --feature-set all --target-mode anomaly \
  --points P3 P5 P7 P9 --device cuda --amp
```

After the train job completes, validation:

```bash
bash Forecast_1d/jobs/submit_features.sh test \
  --model gru --feature-set all --target-mode anomaly \
  --points P3 P5 P7 P9 --split val --device cuda --amp
```

For controlled **same-architecture, same-split** comparisons, train separate GRU models with:

```bash
for selected in flux era5 all; do
  bash Forecast_1d/jobs/submit_features.sh train \
    --model gru --feature-set "$selected" --target-mode anomaly \
    --points P3 P5 P7 P9 --device cuda --amp
done
```

Wait for all jobs to complete before validating each. Each feature set has its own result directory. Run the `test` command with the matching `--feature-set`.

For all 15 points, omit `--points`. For a subsequent validation-controlled test, use `--split test` (but do not use the test split to choose the feature set or hyperparameters).

## 4. Storage and reports

Default experiment directory:

`/home/aosipova/OceanFull/Forecast/Results/Forecast_1d/atlantic15/h14/anomaly/multivar/all/seed42/gru/`

Individual models: `P1/best.pt`, `P1/last.pt`, `P1/train_metadata.json`, `P1/history.csv` (etc.). Evaluation: `.../evaluations/gru/val/P3_P5_P7_P9/metrics.json` and `predictions.npz`.

Every evaluation reports the model, persistence and **train-fitted Seasonal AR(1)** on exactly the same forecast origins, as well as per-horizon scores and MSE skill against SAR. Comparison with older result files is only valid for the **same dates and forecast-origin support**; the new data may end one day earlier than the legacy series. Original train/validation/test date boundaries are fixed from the legacy centers and do not shift when ERA5 data are trimmed.

The model checkpoint stores a SHA256 digest of the feature matrix, dates, metadata and sites; evaluation refuses mismatching data. For runs with different hidden sizes or learning rates, pass a new `--run-id` (for example `--run-id gru128_lr1e4`), and reuse it when testing. Running without `--overwrite` prevents accidental replacement of prior checkpoints or scores.

## 5. Local verification and compatibility

```bash
python -m unittest Forecast_1d.tests.test_feature_experiment
python -m Forecast_1d.train_features --help
python -m Forecast_1d.test_features --help
```

The default training and test scripts and the original Slurm submission tool are intentionally untouched. These are complete files, not patches. The script library requires Python, NumPy, PyTorch and the original `Forecast_1d` modules. In case of `CUDA` availability issues, review the existing A100 partition and `docker-id` setup before changing the model.

Source code contract: [`Forecast_1d/data.py`](https://github.com/Alandriya/Ocean_components/blob/main/Forecast_1d/data.py), [`Forecast_1d/models/neural.py`](https://github.com/Alandriya/Ocean_components/blob/main/Forecast_1d/models/neural.py), [`Forecast_1d/models/seasonal.py`](https://github.com/Alandriya/Ocean_components/blob/main/Forecast_1d/models/seasonal.py) and [Copernicus ERA5 documentation](https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5%2Bdata%2Bdocumentation).
