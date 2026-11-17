# Forecast_1d: level-1 neural forecasting

## Experimental hierarchy

- **Level 0:** persistence, 30-day input mean, and the existing independent SARIMA/AR(1) baseline.
- **Level 1:** independent MLP, TCN, and GRU models at four selected ocean points.
- **Level 2:** more advanced neural forecasting architectures.
- **Level 3:** large pretrained time-series models and SOTA comparisons.
- **Level 4:** SDE-informed architectures and physics-guided objectives.

The level-1 experiment is intentionally location-specific. It does not pool observations
from different geographical points.

## Selected locations

The default point selection is `P3 P5 P7 P9`:

- P3: 40 N, 55 W — eastern extension of the Gulf Stream.
- P5: 28 N, 60 W — Sargasso Sea.
- P7: 15 N, 25 W — eastern tropical Atlantic.
- P9: 57 N, 37 W — subpolar Atlantic, Irminger region.

Each neural model is trained separately at each point. Within a point, sensible and
latent heat fluxes are two jointly modelled channels. Thus each architecture produces
four independent checkpoints, each predicting both fluxes.

## Level-1 architectures

- `mlp`: flatten 60 daily observations of two channels; hidden layers 128 and 64.
- `tcn`: causal dilated temporal convolutions, 32 channels, kernel size 3,
  dilations 1, 2, 4, 8, four residual blocks.
- `gru`: one recurrent layer with 64 hidden units.

All models use a direct 3-day forecast head. The model classes expose a configurable
`input_channels` argument (currently 2), while retaining a two-channel flux output.
This makes a later extension to historical SDE features (2 drift + 3 diffusion
covariance channels) straightforward without changing the forecasting heads. The
initial loss is normalized-space MSE.
Normalization means and scales are estimated using training data only and saved with
each point-specific checkpoint. Early stopping uses the chronological validation split.

Default neural training settings: 60 input days, 3 forecast days, batch size 128,
AdamW learning rate 1e-3, weight decay 1e-4, dropout 0.1, maximum 100 epochs,
patience 15. The exact configuration is stored in each model's `train_metadata.json`.

## Training commands

Run from `/home/aosipova/Ocean_Components`:

```bash
bash Forecast_1d/jobs/submit_1d.sh train --model mlp --points P3 P5 P7 P9
bash Forecast_1d/jobs/submit_1d.sh train --model tcn --points P3 P5 P7 P9
bash Forecast_1d/jobs/submit_1d.sh train --model gru --points P3 P5 P7 P9
```

Each job trains the four independent point models sequentially on the requested A100.
To change training settings:

```bash
bash Forecast_1d/jobs/submit_1d.sh train --model tcn --points P3 P5 P7 P9 \
  --in-len 60 --out-len 3 --epochs 100 --batch-size 128 --patience 15
```

Model checkpoints are saved separately:

```text
DATA_ROOT/Forecast/Results/Forecast_1d/
├── mlp/P3/best.pt
├── mlp/P3/train_metadata.json
├── mlp/P3/history.json
├── ...
├── tcn/P9/best.pt
└── gru/P9/best.pt
```

The existing SARIMA model remains the level-0 model and continues to fit all ten
points independently. For evaluation, its predictions are subset to the same four
selected points.

## Evaluation

If the SARIMA checkpoint has not yet been created, fit it first:

```bash
bash Forecast_1d/jobs/submit_1d.sh train --model sarima
```

After training all three neural architectures:

```bash
# Level 0: persistence, input mean, and SARIMA.
bash Forecast_1d/jobs/submit_1d.sh test --model level0 --split test

# Level 1: the same two baselines plus MLP, TCN, and GRU.
bash Forecast_1d/jobs/submit_1d.sh test --model level1 --split test

```

Level-0 and level-1 evaluations use the same four points and forecast target dates. The legacy `--model all` remains an alias for level 0. Run level 0 and level 1 separately; the metrics collector reads the merged result file.
The input-mean baseline always averages the last 30 observed days. Neural models use
60 input days. SARIMA parameters remain fitted on train only; its filter state is
updated only with observations available at each forecast origin.

To evaluate all ten points explicitly, pass:

```bash
--points P1 P2 P3 P4 P5 P6 P7 P8 P9 P10
```

The default selected points are P3, P5, P7, and P9.

## Plotting

```bash
bash Forecast_1d/jobs/submit_1d.sh plot --model tcn --split test --point P3
bash Forecast_1d/jobs/submit_1d.sh plot --model gru --split test --point all --show-baselines
```

Each figure contains sensible and latent series in separate panels. Forecast horizons
are plotted separately.

## CPU-only result collection

`collect_results.py` uses only Python's standard library: `argparse`, `csv`, `json`,
`pathlib`. It does not import NumPy, PyTorch, pandas, or Matplotlib.

```bash
bash Forecast_1d/jobs/submit_1d.sh collect --split test
```

The collector runs in the `intel` CPU partition without a GPU request and writes:

```text
DATA_ROOT/Forecast/Results/Forecast_1d/summary/
├── forecast_1d_test_P3_P5_P7_P9.json
└── forecast_1d_test_P3_P5_P7_P9.csv
```

JSON contains a list of tidy result rows with model level, point, flux channel,
forecast horizon, RMSE, MAE, bias, MSE, sample count, and persistence MSE skill.
CSV is provided for convenient inspection in spreadsheet software.

## Result paths

All evaluation artefacts are stored under:

`DATA_ROOT/Forecast/Results/Forecast_1d/evaluations/<split>/P3_P5_P7_P9/`

The `metrics.json` file is merged across level-0 and level-1 evaluation runs.
Existing baseline prediction files are reused when level 1 is evaluated, so the
baselines do not need to be recalculated or overwritten.

## SDE roadmap

The level-1 models are deliberately trained without SDE inputs. The saved metadata
records each point, normalization, architecture configuration, and data fingerprint.
The next controlled experiment can add historical drift and diffusion features
(2 drift components and 3 unique entries of the symmetric diffusion covariance),
followed by a drift-based residual formulation. The unmodified level-1 models remain
the reference baseline for measuring the contribution of SDE information.
