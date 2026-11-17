# Forecast_1d: Level 0 extensions and Level 2 models

Install on top of the working `Forecast_1d_horizon14_patch.zip` version. The
patch does not change extraction, the dataset, the existing Level 1 checkpoint
architecture, project paths, or Slurm launchers. All results remain under
`DATA_ROOT/Forecast/Results/Forecast_1d/h14/`.

## Fixed protocol

- Points: P3, P5, P7, P9 (each point has an independent fitted model).
- Input: 60 observed days, sensible and latent heat fluxes.
- Output: 14 daily values of both fluxes, direct multi-horizon predictions.
- Chronological validation: last 730 days before the final 365-day test.
- Normalization: per point and channel, train-only; MSE in normalized units.
- Same forecast origins and finite scoring support as the existing Level 1 runs.
- Default optimizer: AdamW, learning rate 0.001, batch size 128, maximum 100
  epochs, early stopping patience 15. Validation inference is batched.
- No additional weather variables, SDE features, or pretrained weights.

## Level 0 additions

These three fitted baselines are computed inside `test.py`; no separate train
job or checkpoint is required. Parameters are recomputed deterministically from
train only for each evaluation split. The existing SARIMA checkpoint is not
changed. `input_mean` remains the established 30-day baseline.

* `seasonal_climatology`: train-only calendar-day mean, pooled observations in a
  circular +/-15-day calendar window. Calendar days map to 366 month-day slots;
  March 1 is consistently assigned to the same slot in leap/non-leap years.
  If a calendar window contains no training data, the channel's overall training
  mean is used. Future calendar dates are known, not future observations.
* `seasonal_ar1`: a zero-intercept AR(1) of anomalies from the above climatology.
  Fit phi by least squares on valid train pairs separately per point/channel,
  clip to [-0.999, 0.999], then add phi**h times the observed origin anomaly
  to the climatology at the target date. The fitted phi values are recorded
  in `metrics.json` under `baseline_config`.
* `ridge_var`: direct regularized multivariate lag regression (not a recursive
  classical VAR). Both 60-day histories (120 features) are used to jointly
  predict both channels at all 14 leads (28 outputs). Train-only scaling and
  an unpenalized intercept; fixed alpha=0.01 for the objective
  ||Y-XW-b||_F^2 / N + alpha * ||W||_F^2. No validation/test tuning in this patch.

## Level 2 architectures

These are compact PyTorch reimplementations/adaptations of published model
families, not exact official repositories or claims of published SOTA.
All use the existing [batch, time, 2] -> [batch, horizon, 2] interface.

| Name | Configuration | Channel treatment |
| --- | --- | --- |
| `nbeats` | Generic identity basis, four backcast/forecast residual blocks, three 128-unit hidden layers each | Joint |
| `nhits` | Three scales, pool sizes 4/2/1, forecast coefficient downsampling 4/2/1, linear interpolation, width 128 | Joint |
| `patchtst` | Patch 8, stride 4, 64-dimensional encoder, 4 heads, 2 Transformer layers, reversible window normalization | Channel-independent with shared encoder/head |
| `dlinear` | Moving average kernel 25, independent trend and residual linear maps per channel | Channel-independent, individual maps |
| `tide` | Width 128 residual dense encoder/decoder, temporal decoder, global linear shortcut, no exogenous covariates | Joint |

PatchTST and DLinear predict both outputs within one point checkpoint, but do
not mix sensible and latent input channels. This preserves a key architectural
control. The other three mix the two channels. DLinear is a learned linear
model, included in Level 2 as a useful comparison rather than a nonlinear net.
Future SDE input integration needs model-specific choices (including an explicit
cross-channel/covariate variant for PatchTST and DLinear).

References:
- N-BEATS: https://arxiv.org/abs/1905.10437
- N-HiTS: https://doi.org/10.1609/aaai.v37i6.25854
- PatchTST: https://arxiv.org/abs/2211.14730
- DLinear: https://doi.org/10.1609/aaai.v37i9.26317
- TiDE: https://arxiv.org/abs/2304.08424

## Cluster quick start

From `/home/aosipova/Ocean_Components`:

```bash
unzip -o Forecast_1d_level2_patch.zip
# Add the baselines without reevaluating the already saved models.
for split in val test; do
  bash Forecast_1d/jobs/submit_1d.sh test --model level0_new --split "$split" --out-len 14
done
# Submit five independent architecture jobs (4 point models per job).
for model in nbeats nhits patchtst dlinear tide; do
  bash Forecast_1d/jobs/submit_1d.sh train --model "$model" --in-len 60 --out-len 14 --device cuda
done
```

Wait for these jobs to finish successfully before submitting evaluation. Do not
run two evaluations writing to the same split/point directory simultaneously:
`metrics.json` is a merged report. Validation and test have separate directories.

```bash
for split in val test; do
  bash Forecast_1d/jobs/submit_1d.sh test --model level2 --split "$split" --out-len 14 --device cuda
done
```

`level0_new` evaluates the three new baselines plus the two train-free ones;
`level2` evaluates the five new architectures plus the train-free ones.
Existing SARIMA and Level 1 metric entries are retained. `--model all` now
requests all 14 models; it is unnecessary for this incremental update and
will reject existing predictions unless `--overwrite` is explicitly supplied.
An already completed single-model re-evaluation likewise requires `--overwrite`.

After evaluation is finished, submit the CPU-only summaries:

```bash
for split in val test; do
  bash Forecast_1d/jobs/submit_1d.sh collect --split "$split" --out-len 14
done
```

The existing CPU submission uses `intel`, no GPU request. The collector imports
only Python standard-library modules and the project's stdlib-only registry
and horizon helpers. Alternatively use Python directly with an explicit path:

```bash
python -S -m Forecast_1d.collect_results --split val --out-len 14 \
  --results-dir /home/aosipova/OceanFull/Forecast/Results/Forecast_1d
```

## Plot selected new models

```bash
bash Forecast_1d/jobs/submit_1d.sh plot --model nhits --split test --point P3 \
  --horizon 1 3 7 14 --compare seasonal_ar1 ridge_var gru
```

`--show-baselines` overlays available Level 0 models; `--compare` lets you
specify only the references of interest. The default plotted leads are 1, 3, 7,
14; `--all-horizons` plots all 14.

## Output

`DATA_ROOT/Forecast/Results/Forecast_1d/h14/<model>/<point>/` stores `best.pt`,
`train_metadata.json` (including model level, parameter count, hyperparameters
and train-only scales), and `history.json`. Evaluation NPZ files and merged
metrics retain the same layout in `h14/evaluations/{val,test}/P3_P5_P7_P9/`.

The summary directory contains a detailed CSV/JSON, a new `_ranking.csv`, and
`ranking_by_rmse` plus `missing_models` in the JSON. In addition to RMSE, MAE,
bias, and MSE skill against persistence/SARIMA, the collector computes skill
against `seasonal_climatology`, `seasonal_ar1`, and `ridge_var`. Macro skill is
the unweighted mean over 4 points x 2 channels x 14 leads (112 cells); a missing
reference or undefined skill yields a null macro score, not an artificial zero.
The pooled RMSE instead weights squared errors by observation counts.

Select architectures on validation; test scores are descriptive at this stage
because the test period has already been inspected in earlier experiments.
For confirmatory SOTA claims, preserve another unused period and quantify paired
uncertainty accounting for overlapping forecast windows.

## Local checks

From the repository root (with NumPy, PyTorch and matplotlib installed):

```bash
python -m unittest discover -s Forecast_1d/tests
```

Tests cover output shape, gradients, calendar mapping, causal new baselines,
independent channel behavior, incremental evaluation and CPU-only collection.
The integration tests use synthetic data: they do not validate ocean forecast
quality or actual cluster GPU behavior.
