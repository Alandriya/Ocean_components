# Forecast_1d: Atlantic15 and training diagnostics

This patch extends the 14-day Level 0–2 experiment to P1–P15. It is an overlay
for the existing Level 2 code and CPU collector fix. No change to project_paths.py
or the original gridded flux arrays is required. The old ten-point configuration
is available as `points_10.json`; P1–P10 retain their original grid coordinates.
All 15 default centers passed the 21x21 all-water check on the supplied mask.
The region labels are descriptive, not a diagnosis of instantaneous current positions.

## Artifact isolation

Default dataset: `DATA_ROOT/DATA/Forecast_1d/atlantic15_native_1979-2026`.
Default results base: `DATA_ROOT/Forecast/Results/Forecast_1d/atlantic15`.
The horizon directory `h14` is appended automatically. The original
`Forecast_1d/h14` results and `atlantic10_native_1979-2026` data are preserved.
Old checkpoints are bound to their original dataset fingerprint and must not be
copied to the new experiment without a separate, verified migration.
Train/test/plot derive the default point set from the loaded data. The CPU
collector defaults to P1–P15 and reads only JSON. For a custom subset pass the
same ordered `--points` to training, test, plotting and collection. To use old
artifacts specify the old `--dataset-dir` and `--results-dir` (for collection
only `--results-dir`), and explicitly select the original point subset.

## Extract and run the 15-point baselines

From repository root:

```bash
bash Forecast_1d/jobs/submit_1d.sh extract --dry-run
# Wait for the dry-run to finish successfully, then:
bash Forecast_1d/jobs/submit_1d.sh extract
# Wait for extraction to finish, then:
bash Forecast_1d/jobs/submit_1d.sh train --model sarima
# Wait for AR(1) fitting to finish, then:
for split in val test; do
    bash Forecast_1d/jobs/submit_1d.sh test --model level0 --split "$split" --device cpu
done
```

The extraction job uses the established CPU environment with NumPy and the
collector uses the CPU host Python without Docker. The training/test/plot launchers
retain their existing scheduler and container settings; `--device cpu` changes
Python computation but does not change the Slurm partition or resource request.
Do not submit concurrent evaluation jobs for the same experiment and split.
The collector can run after evaluation completes:

```bash
bash Forecast_1d/jobs/submit_1d.sh collect --split val
```

## Training diagnostics: opt-in controls

Default optimizer, initial learning rate, dropout, hidden sizes, training
objective and epoch budget remain as in Level 2. Checkpoints now track the exact
best selection score; `min_delta` only determines patience resets. Added flags:

- `--scheduler plateau`: reduce LR on a validation plateau (default: `none`).
- `--lr-patience 3 --lr-factor 0.5 --min-lr 1e-6`.
- `--monitor sar`: minimize mean per-channel/per-lead validation MSE ratio to
  train-fitted Seasonal AR(1). `--monitor normalized` retains normalized MSE
  selection (default). This does not change the training objective. SAR
  monitoring rejects incomplete validation targets, nonfinite origin values or
  zero SAR MSE rather than comparing unequal scoring supports.
- `--grad-clip 1.0`: gradient clipping threshold, unchanged by default.

History includes learning rate before/after scheduler, normalized train/val
MSE, selection score, SAR skill when enabled, mean pre-clipping gradient norm
and fraction of batches clipped. Metadata includes full optimizer settings,
stop reason, selection criterion and best recorded score.

Example validation-only pilot on the original four points (separate results root):

```bash
BASE=/home/aosipova/OceanFull/Forecast/Results/Forecast_1d/atlantic15_pilot_lr
for model in mlp tcn gru; do
    bash Forecast_1d/jobs/submit_1d.sh train --model "$model" \
        --points P3 P5 P7 P9 --results-dir "$BASE" \
        --learning-rate 0.0003 --scheduler plateau --lr-patience 3 \
        --patience 12 --epochs 100 --monitor sar --seed 42 --device cuda
done
# After all three jobs finish:
bash Forecast_1d/jobs/submit_1d.sh test --model level0_new --split val \
    --points P3 P5 P7 P9 --results-dir "$BASE" --device cpu
# Wait for the baseline job before starting evaluation of neural methods:
bash Forecast_1d/jobs/submit_1d.sh test --model level1 --split val \
    --points P3 P5 P7 P9 --results-dir "$BASE"
# After evaluation finishes:
bash Forecast_1d/jobs/submit_1d.sh collect --split val \
    --points P3 P5 P7 P9 --results-dir "$BASE"
```

Results are not automatically comparable across changed context lengths unless
forecast origins and target support agree. The pilot is intended for tuning on
validation, not repeated test selection. Compare one factor at a time with a
matched control and use a separate output directory per configuration/seed.
The collector writes the original RMSE ranking and an additional
`*_sar_ranking.csv`, sorted by macro Seasonal AR(1) skill (larger is better).
Missing baseline skill values sort last. Ridge-VAR remains a direct linear model.
Anomaly, calendar-informed and SAR-residual training are planned follow-ups,
not implemented by this patch.

## Verification

Run from the repository root with NumPy, PyTorch and Matplotlib installed:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -m unittest discover -s Forecast_1d/tests
```

Synthetic tests cover extraction/recovery at 15 points, legacy subset operation,
all 120 point-specific neural fits (one epoch each), all 14 model predictions,
SAR checkpoint selection and actual LR reduction, plotting P15, and CPU-only
collection with Python `-E -S`. Real cluster execution and real-data learning
quality are not validated by these tests.
