# Forecast_1d: train-only anomaly forecasting (60 -> 14 days)

This is a small overlay on the existing Atlantic15 / Level 0–2 implementation.
No new launchers, dependencies, extraction scripts, model architectures, or Slurm
partitions are introduced. Only training, evaluation, plotting, result collection,
and experiment path selection are updated. All code and comments remain in English.

## Model definition and leakage boundaries

For each location and channel, fit a smoothed daily climatology using **only the
training interval** (the same 366-bin calendar and ±15-day circular window used
by `models/baselines.py`):

    mu(d) = train-only seasonal climatology for calendar date d
    z(t)  = x(t) - mu(d(t))
    zhat(t+1:t+14) = NN(z(t-59:t))
    xhat(t+h) = zhat(t+h) + mu(d(t+h))

The model **does not consume or predict Seasonal AR(1) output**. The AR(1)
coefficient `phi` is not used in anomaly training/inference. Seasonal AR(1)
remains an independently evaluated statistical reference, not a network base.
No observed future flux is used in prediction; future calendar dates are known.

Normalization is performed on the **training anomalies only** (per point/channel).
Input windows are causal. Checkpoints keep the complete training protocol,
including `target_mode=anomaly`, climatology radius, split boundary,
normalization statistics, and dataset fingerprint. Evaluation verifies metadata
and rebuilds the climatology from the training interval only, using the same
fitted data and settings. All final predictions and scores are saved in original
physical units, not normalized anomaly units.

## Default training configuration

The *optimizer* defaults were updated after review of learning histories:

| Option | Default |
| --- | --- |
| `--in-len` | `60` (neural) |
| `--out-len` | `14` |
| `--learning-rate` | `0.0003` |
| `--scheduler` | `plateau` (`ReduceLROnPlateau`) |
| `--lr-patience` | `3` |
| `--lr-factor` | `0.5` |
| `--min-lr` | `0.000001` |
| `--patience` | `12` |
| `--epochs` | `100` |
| `--batch-size` | `128` |
| `--weight-decay` | `0.0001` |
| `--dropout` | `0.1` |
| `--grad-clip` | `1.0` |
| `--monitor` | `normalized` (validation normalized anomaly MSE) |
| `--seed` | `42` (+ point index) |
| `--climatology-radius` | `15` |

**Compatibility:** `--target-mode` continues to default to `raw` so existing
scripts and original raw checkpoints work as before. For this experiment,
**always explicitly pass `--target-mode anomaly` to train, test, plot, and collect**.
The code isolates anomaly artifacts under `h14/anomaly`, without touching
`h14/{mlp,tcn,gru,...}` or previous evaluations. The separate test baseline
scores in anomaly mode are computed in original units.

## One-time installation

Extract `Forecast_1d_anomaly_patch.zip` in the repo root:

```bash
cd /home/aosipova/Ocean_Components
unzip -o Forecast_1d_anomaly_patch.zip
```

The dataset already resides at:

    /home/aosipova/OceanFull/DATA/Forecast_1d/atlantic15_native_1979-2026

No re-extraction or data upload is required.

## Pilot: P3, P5, P7, P9 (recommended first)

Use a *new*, independent results root and allow all three Slurm training jobs to
finish **before** submitting evaluation jobs:

```bash
cd /home/aosipova/Ocean_Components
BASE=/home/aosipova/OceanFull/Forecast/Results/Forecast_1d/atlantic15_anomaly_pilot

for model in mlp tcn gru; do
    bash Forecast_1d/jobs/submit_1d.sh train \
        --model "$model" --target-mode anomaly \
        --points P3 P5 P7 P9 --in-len 60 --out-len 14 \
        --results-dir "$BASE" --device cuda
done
```

Each training job fits one independent model per point; each model predicts both
sensible and latent fluxes jointly for all fourteen leads.

### Validation

After **all three** training jobs finish, evaluate **sequentially**; separate
submissions to one split must not overlap because they update one metrics file.

```bash
bash Forecast_1d/jobs/submit_1d.sh test \
    --model level0_new --target-mode anomaly --split val \
    --points P3 P5 P7 P9 --results-dir "$BASE" --device cpu
```

After that job completes:

```bash
bash Forecast_1d/jobs/submit_1d.sh test \
    --model level1 --target-mode anomaly --split val \
    --points P3 P5 P7 P9 --results-dir "$BASE" --device cuda
```

`level0_new` supplies seasonal climatology, Seasonal AR(1), Ridge-VAR and the
trivial baselines; it does not require training additional networks.
`level1` supplies MLP, TCN and GRU with the same baseline support.
This yields a single comparable validation `metrics.json` containing all of
these methods. The training-monitor criterion is `normalized` by default;
SAR is only a separate baseline used when evaluating the final models.

If you need the ordinary AR(1) (`sarima`) as a comparison too, fit it once in
the original raw root using `submit_1d.sh train --model sarima`, then evaluate
`submit_1d.sh test --model sarima --target-mode anomaly ...` sequentially. The
raw fitted AR(1) parameters are shared across target modes.

### Final test (only after selecting the protocol on validation)

```bash
bash Forecast_1d/jobs/submit_1d.sh test \
    --model level0_new --target-mode anomaly --split test \
    --points P3 P5 P7 P9 --results-dir "$BASE" --device cpu
```

After that job finishes:

```bash
bash Forecast_1d/jobs/submit_1d.sh test \
    --model level1 --target-mode anomaly --split test \
    --points P3 P5 P7 P9 --results-dir "$BASE" --device cuda
```

The evaluation metrics are written to:

    $BASE/h14/anomaly/evaluations/val/P3_P5_P7_P9/metrics.json
    $BASE/h14/anomaly/evaluations/test/P3_P5_P7_P9/metrics.json

### Optional collection (no NumPy, GPU, Docker, or new Slurm job)

If the cluster permits light commands on its login node, run:

```bash
python3 -E -S -m Forecast_1d.collect_results \
    --split val --target-mode anomaly \
    --points P3 P5 P7 P9 --results-dir "$BASE"
python3 -E -S -m Forecast_1d.collect_results \
    --split test --target-mode anomaly \
    --points P3 P5 P7 P9 --results-dir "$BASE"
```

The collector uses only the Python standard library. If login-node compute is
not permitted, instead submit it through the existing `collect` launcher
(which uses the `intel` CPU partition).

## Full Atlantic15 run

After assessing the pilot on validation, train *all 15 points* by **omitting
`--points`**:

```bash
for model in mlp tcn gru; do
    bash Forecast_1d/jobs/submit_1d.sh train \
        --model "$model" --target-mode anomaly --device cuda
done
```

The defaults are 60-day history, 14-day forecast, 15 loaded points, and the
optimizer defaults above. Wait for every training job to complete.

Then evaluate separately, **sequentially**, for each split:

```bash
bash Forecast_1d/jobs/submit_1d.sh test \
    --model level0_new --target-mode anomaly --split val --device cpu
# Wait for completion.
bash Forecast_1d/jobs/submit_1d.sh test \
    --model level1 --target-mode anomaly --split val --device cuda
# After protocol selection, run the same two jobs for --split test.
```

Full-dataset output is isolated at:

    /home/aosipova/OceanFull/Forecast/Results/Forecast_1d/atlantic15/h14/anomaly/

If you change the climatology radius, network context, seed, or another
experimental factor, use an independent `--results-dir`; never mix predictions
from incompatible experiments. Pass the same `--climatology-radius` during
training and evaluation. Evaluation detects a mismatch and fails safely.

## Reproducing legacy raw forecasts

Old raw checkpoints and metrics remain in `h14/` as before. The original
training defaults can still be specified explicitly:

```bash
bash Forecast_1d/jobs/submit_1d.sh train \
    --model gru --target-mode raw --learning-rate 0.001 \
    --scheduler none --patience 15 --device cuda
```

Existing data and checkpoints are never automatically overwritten. No `--overwrite`
is needed for a first anomaly experiment; use it only when explicitly replacing
an anomaly run.

## Validation scope

Synthetic tests cover train-only climatology fitting, anomaly training for
MLP/GRU/TCN, reconstruction of the original-unit forecast, evaluation and
collection, all fifteen default points, optional SAR monitor, and raw/anomaly
checkpoint isolation. Existing forecasting tests were also rerun.

Cluster execution on real Atlantic15 data has not yet been verified by this
patch. Stronger anomaly forecasts are a hypothesis to be tested, not a claimed
result. References: PyTorch `AdamW`, `ReduceLROnPlateau` official documentation;
the statistical baseline formulation is in `Forecast_1d/models/baselines.py`.
