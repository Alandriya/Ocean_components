# Forecast_1d: 14-day experiment

This patch extends the existing Level 0 and Level 1 workflows. It requires the
previous `Forecast_1d` extraction and Level 1 modules. It does not change the
original extracted arrays, neural architecture definitions, point selection,
cluster launchers, or log directory.

## Experimental protocol

- Default input: 60 observed days for MLP, TCN and GRU.
- Default output: 14 future daily values for both sensible and latent fluxes.
- Default points: P3, P5, P7, P9, with independent neural weights at each point.
- Persistence repeats the last observed value at every lead. Input mean repeats
  the mean of the preceding 30 observed days. SARIMA defaults to AR(1), fitted
  independently to each point/channel and iterated for all 14 leads.
- Unchanged chronological fitting and evaluation intervals: final 365 days test,
  preceding 730 days validation, earlier days train. All target days must lie
  inside their respective interval; therefore 352 test origins and 717 validation
  origins are available on the current dataset.
- Normalization is estimated on train only. Neural models use direct multi-lead
  prediction and equally weighted normalized MSE over both channels and all
  14 leads. No future observations are provided at inference.
- Evaluation uses a fixed common finite-target and baseline support for each
  split. A model producing non-finite values on that support fails evaluation.

## Installation and training

Extract the archive into `/home/aosipova/Ocean_Components`, replacing the files
it contains. Existing job launchers are not modified. All text is stored with LF
line endings. From the repository root:

```bash
bash Forecast_1d/jobs/submit_1d.sh train --model sarima --out-len 14
bash Forecast_1d/jobs/submit_1d.sh train --model mlp --out-len 14
bash Forecast_1d/jobs/submit_1d.sh train --model tcn --out-len 14
bash Forecast_1d/jobs/submit_1d.sh train --model gru --out-len 14
```

The three neural jobs may run in parallel. Each fits its four independent models
sequentially. SARIMA is cheap to refit and uses the same train interval as before.
It still fits all ten points; evaluation defaults to the four selected points.
The neural models require new weights because the forecast head has 28 outputs
instead of 6. The new horizon is also the CLI default.

## Evaluation and plots

After all four training jobs have completed successfully:

```bash
bash Forecast_1d/jobs/submit_1d.sh test --model all --split val --out-len 14
```

Use validation to select settings. Once frozen, evaluate test:

```bash
bash Forecast_1d/jobs/submit_1d.sh test --model all --split test --out-len 14
```

`all` now explicitly evaluates all six models. `level0` evaluates the statistical
baselines only; `level1` evaluates the three networks and two constant baselines.
Use one evaluation job per split and point selection to avoid concurrent writes
into the same metrics file. Existing model predictions require `--overwrite` to
replace them. That flag affects only the selected horizon's result directory.

```bash
bash Forecast_1d/jobs/submit_1d.sh plot --model gru --split test --out-len 14 --show-baselines
bash Forecast_1d/jobs/submit_1d.sh plot --model tcn --split test --out-len 14 --point P3 --horizon 1 7 14 --show-baselines
```

Without `--horizon`, plots show leads 1, 3, 7 and 14 at all selected points.
Use `--all-horizons` for 56 figures per model. The optional baseline overlays
include persistence, input mean, and SARIMA (if evaluated).

## CPU-only collection

```bash
bash Forecast_1d/jobs/submit_1d.sh collect --split val --out-len 14
bash Forecast_1d/jobs/submit_1d.sh collect --split test --out-len 14
```

Collection uses only the Python standard library and the existing CPU job.
CSV/JSON include every horizon, point and channel, plus pooled results and
MSE skill against persistence and SARIMA. A missing SARIMA score is represented
as null/blank, never as zero. SARIMA-relative skill is computed when counts
match; the new evaluator enforces common scoring support.

## Result directories

```text
DATA_ROOT/Forecast/Results/Forecast_1d/
├── sarima/                     # Legacy 3-day files, untouched
├── mlp/                        # Legacy 3-day files, untouched
├── tcn/
├── gru/
├── evaluations/
├── plots/
├── summary/
└── h14/
    ├── sarima/params.npz
    ├── mlp/P3/best.pt
    ├── tcn/P3/best.pt
    ├── gru/P3/best.pt
    ├── evaluations/val/P3_P5_P7_P9/metrics.json
    ├── evaluations/test/P3_P5_P7_P9/metrics.json
    ├── plots/test/gru/P3_P5_P7_P9/
    └── summary/forecast_1d_test_P3_P5_P7_P9_h14.csv
```

Use `--out-len 3` to access original 3-day paths. Other positive horizons use
`hN/`. `--results-dir` always means the base result directory, before adding
`hN`. For different input windows or other hyperparameter configurations at the
same horizon, use separate base result directories via `--results-dir` in every
stage. Existing artifacts are checked for split and dataset compatibility.

Logs are unchanged: `DATA_ROOT/Forecast/logs/`.

## Tests

```bash
python -m unittest Forecast_1d.tests.test_forecasting Forecast_1d.tests.test_level1_models Forecast_1d.tests.test_horizon14 -v
```

Synthetic tests cover 12 independent point/model checkpoints, all six models,
14-day chronological split boundaries, future-observation invariance, plotting,
legacy output preservation, and collection with `python -S`.

## Interpretation

A 14-day network is trained with a different objective from a 3-day network.
Comparing the first three leads of the two experiments should use the same
forecast origins (the original experiment contains 11 additional late-year
origins). Longer-horizon prediction can converge toward climatology; adding a
training-only seasonal climatology benchmark is recommended before making claims
of long-range predictive skill.
