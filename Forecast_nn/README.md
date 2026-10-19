# Forecast_nn — common framework + SDE-informed models

The folder keeps the common interface:

```python
x:          [B, T_in, C, H, W]
mask:       [1, 1, H, W] or [B, 1, H, W]
prediction: [B, T_out, C, H, W]
```

The original model implementations remain in `models/`. Three additional informed versions are registered without removing the base versions.

## Main structure

```text
Forecast_nn/
├── config.py
├── data.py
├── sde.py                       # a(x), b(x) lookup and conditioning features
├── losses.py                    # MSE + optional dynamic + optional SDE loss
├── evaluation.py                # RMSE/MAE/MSE/SSIM + temporal diagnostics
├── train.py
├── test.py
├── aggregate_metrics.py         # all metrics_*.xlsx -> one comparison workbook
├── models/
│   ├── unet_convlstm.py
│   ├── simvp.py
│   ├── fno.py
│   ├── unet_convlstm_sde.py
│   ├── simvp_sde.py
│   ├── fno_sde.py
│   ├── sde_conditioning.py
│   └── ... preserved legacy models
└── plotting/
    ├── training_curve.py
    └── forecast_comparison.py
```

## Registered models

Base models:

```text
unet
unet_convlstm
simvp
fno
```

SDE-informed models:

```text
unet_convlstm_sde
simvp_sde
fno_sde
```

### U-Net + ConvLSTM + SDE

For every historical day, the spatial U-Net bottleneck is conditioned by local `a(x), b(x)` maps through a small FiLM module before entering ConvLSTM:

```text
X_t -> U-Net encoder -> bottleneck -----> FiLM -----> ConvLSTM
                                      ^
                                      |
                               a(X_t), b(X_t)
```

### SimVP + SDE

The same FiLM conditioning is applied to every encoded historical bottleneck before the SimVP spatiotemporal CNN blocks. This is the recommended first informed experiment because base SimVP currently preserves temporal motion best.

### FNO + SDE

The SDE coefficient fields are treated as physical/operator coefficient fields. By default the complete history of `a(X_t), b(X_t)` is concatenated with the ordered flux history before the FNO lifting layer.

```text
[X_1...X_30, a_1...a_30, b_1...b_30, mask, Y, X]
                         -> 1x1 lifting -> Fourier blocks -> output
```

Set `sde_history=False` in `config.py` if only the last input-day coefficients should be supplied.

## SDE statistics format

The preferred input is the NPZ already created by the earlier SDE-Energy pipeline:

```text
bin_edges : [C, bins+1]
a         : [C, bins]
b         : [C, bins]
```

Those statistics were estimated in normalized state space, so they match the current normalized model tensors directly.

The loader also accepts generic NPZ keys:

```text
values / x / centers / bin_centers
+
a / drift
+
b / variance / diffusion
```

Paths are configured in `config.py`:

```python
cfg.sde_sources = {
    'sensible': {
        'file': cfg.root_path / 'DATA' / 'Fluxes' /
                'sde_energy_stats_sensible_ds2_stride_1_bins_500.npz'
    },
    'latent': {...},
}
```

If your files have different names, only change these paths.

If the coefficients are stored as three separate NumPy arrays instead of one NPZ, use:

```python
cfg.sde_sources['sensible'] = {
    'value_file': r'D:/.../flux_values.npy',
    'a_file': r'D:/.../a.npy',
    'b_file': r'D:/.../b.npy',
}
```

These arrays must describe the same one-dimensional state lookup (one row per field channel), not a future target map.


### Meaning of `b`

For the older statistics:

```python
cfg.sde_b_kind = 'variance'
```

because `b=Var[dX|X]` was saved.

If your coefficient is the Ito diffusion amplitude in

```text
dX = a(X) dt + b(X) dW
```

use:

```python
cfg.sde_b_kind = 'diffusion'
```

Then the loss uses `b^2 * dt` as conditional variance.

## SDE loss

The informed versions use both architectural conditioning and the optional article-style SDE regularizer:

```text
loss = masked_MSE
     + dynamic_weight * dynamic_loss
     + sde_weight * sde_energy
```

For each forecast horizon, the SDE term is evaluated from the previous state. The first transition uses the last true input map; later transitions use the previous model prediction.

Default:

```python
cfg.sde_weight = 1e-4
cfg.sde_dt = 1.0
cfg.dynamic_weight = 0.0
```

For an ablation with conditioning only:

```bat
python -m Forecast_nn.train --model simvp_sde --sde-weight 0
```

For SDE-loss-only with the original SimVP architecture:

```bat
python -m Forecast_nn.train --model simvp --sde-weight 1e-4
```

## Recommended informed runs

SimVP:

```bat
python -m Forecast_nn.train ^
  --model simvp_sde ^
  --variable sensible ^
  --sde-weight 1e-4 ^
  --batch-size 16 ^
  --num-workers 4 ^
  --amp
```

U-Net + ConvLSTM:

```bat
python -m Forecast_nn.train ^
  --model unet_convlstm_sde ^
  --variable sensible ^
  --sde-weight 1e-4 ^
  --batch-size 16 ^
  --num-workers 4 ^
  --amp
```

FNO:

```bat
python -m Forecast_nn.train ^
  --model fno_sde ^
  --variable sensible ^
  --sde-weight 1e-4 ^
  --batch-size 16 ^
  --num-workers 4 ^
  --amp
```

The CLI can override the table path:

```bat
python -m Forecast_nn.train --model simvp_sde --sde-file "D:/.../my_sde_stats.npz"
```

## Testing and SSIM

Run as before:

```bat
python -m Forecast_nn.test --model simvp_sde --variable sensible
```

Each XLSX now contains:

```text
MSE
MAE
RMSE
SSIM
Delta RMSE
Delta correlation
Predicted mean |delta|
True mean |delta|
```

SSIM is computed locally with an 11x11 Gaussian window and averaged only over valid ocean pixels. It is exported as overall, per-horizon and per-sample SSIM.

Old checkpoints and old metric workbooks remain usable; old workbooks simply have no SSIM column.

## Unified metrics workbook

After testing several models:

```bat
python -m Forecast_nn.aggregate_metrics
```

The script recursively finds all `metrics_*.xlsx` below `cfg.results_root` and saves:

```text
<results_root>/_summary/all_models_metrics.xlsx
```

Sheets:

```text
summary       one row per metrics workbook / method
per_horizon   one row per horizon
read_errors   any old/corrupt workbook that could not be parsed
```

The summary includes architecture parameters and SDE settings when present, so base and informed versions can be compared in one table.

## Resume

```bat
python -m Forecast_nn.train --model simvp_sde --resume --epochs 15 --sde-weight 1e-4
```

The checkpoint contains the model, optimizer, AMP scaler, epoch counter and complete loss history.

## Time-aligned semiparametric A/B coefficient maps

The SDE-informed models now read the coefficient maps directly from:

```text
{files_path_prefix}/Components/sensible-latent/A1_1-17106.npy  # sensible drift A
{files_path_prefix}/Components/sensible-latent/B1_1-17106.npy  # sensible diffusion B
{files_path_prefix}/Components/sensible-latent/A2_1-17106.npy  # latent drift A
{files_path_prefix}/Components/sensible-latent/B2_1-17106.npy  # latent diffusion B
```

`files_path_prefix` defaults to `cfg.root_path` and can be changed in `config.py`.
The files may use the same supported layouts as the flux arrays: `(space,time)`,
`(time,space)`, `(time,H,W)` or `(time,C,H,W)`.

### Causal time alignment

The coefficient series begins at **t=1**. Coefficient element 0 is interpreted as
`A_1/B_1`, estimated from the already observed transition `X_1-X_0`.
Consequently:

```text
X_0 -> no A/B map is available (zero conditioning only)
X_1 -> A/B element 0
X_2 -> A/B element 1
...
X_t -> A/B element t-1
```

Only A/B maps aligned with the historical input sequence are read. The target
coefficient maps `A_{t+1}, B_{t+1}, ...` are **not** read because they were
estimated using future increments and would leak target information.

The default SDE regularizer uses only the last observed `A_t/B_t` for the first
forecast transition (`sde_loss_mode='first_step'`). An optional
`last_observed_all` mode reuses the same known `A_t/B_t` for all short-horizon
increments without reading any future coefficient map.

### B interpretation

For coefficients produced by `semiparametric.py`, use:

```python
cfg.sde_b_kind = 'diffusion'
```

The saved object is named `B`, and the vector implementation subsequently forms
`B B^T`. For normalized model state `Z=(X-mean)/std`, the loader therefore uses
`B_Z = B_X/std`, and the SDE loss uses variance `B_Z**2 * dt`.

### Example

```bat
python -m Forecast_nn.train ^
    --model simvp_sde ^
    --variable sensible ^
    --sde-b-kind diffusion ^
    --sde-loss-mode first_step ^
    --sde-weight 1e-4 ^
    --batch-size 16 ^
    --num-workers 4 ^
    --amp
```

---

## Chronological train / validation / test (updated pipeline)

The effective daily series is cut at **2023-12-31 inclusive**, regardless of
whether the input NPY contains observations from 2024 or later. The default
calendar start is `1979-01-01`, with index 0 for that day, hence:

| Index | Date |
| ---: | --- |
| 0 | 1979-01-01 |
| 1 | 1979-01-02 (first available SDE A/B estimate) |
| 16435 | 2023-12-31 (last included frame) |
| 16436 | 2024-01-01 (excluded) |
| 17106 | 2025-11-01 **if interpreted as a zero-based daily index** |

The `1-17106` file-name suffix is retained as-is. Coefficient element 0 maps
to flux index 1. For SDE runs, coefficient coverage is also checked; no future
A/B maps are supplied to the network. If those file indices use a different
calendar convention, correct their *alignment* first, not the flux dates.

Defaults in `config.py`:

- **Train**: 1979-01-01 ... 2020-12-31, target indices `[0, 15341)`
- **Validation**: 2021-01-01 ... 2022-12-31, `[15341, 16071)`
- **Test**: 2023-01-01 ... 2023-12-31, `[16071, 16436)`

These are **target** intervals. Validation and test inputs may include the
immediately preceding 30 days from the previous interval, because those days
would have been observed at forecast initialization. Predictions are kept
strictly within their own target intervals. Models receive no future fields.

Normalization and SDE-feature statistics are estimated from train only.
The fixed **PSNR reference range** is independently estimated per output
channel from uniformly sampled *training* frames using the 0.1 and 99.9
percentiles (default: up to 256 frames). No validation/test targets contribute
to this range. PSNR is calculated in **physical units**, masked over ocean,
and aggregated from MSE *before* converting to dB. Multi-channel overall PSNR
is the arithmetic mean of the channel-level PSNRs, not a PSNR based on one
shared range. Note that PSNR may be +infinity for exact predictions.

### New training commands

From the repository root:

```bash
python -m Forecast_nn.train --model simvp --variable sensible \
  --epochs 100 --batch-size 16 --num-workers 4 --amp \
  --end-date 2023-12-31 --val-steps 730 --test-steps 365 --patience 12
```

For the current scalar SDE-informed SimVP:

```bash
python -m Forecast_nn.train --model simvp_sde --variable sensible \
  --epochs 100 --amp --end-date 2023-12-31 --val-steps 730 --test-steps 365
```

`best.pth` is saved when the **validation RMSE in W/m²** improves, `last.pth`
stores the latest epoch, and `checkpoint.pth` duplicates the latest epoch for
compatibility. `--resume` restores the latest checkpoint and the best metric.
`--patience 0` disables early stopping; `--min-delta` is the minimum physical
RMSE improvement required to reset the patience counter. The output directory
includes `end..._val..._test...` to keep new checkpoints separate from older
experiments.

Training logs and `training_history.npz` now include train/validation RMSE and
PSNR (dB), including per-horizon PSNR arrays. Plots are written to
`training_curve.png` and `training_curve_psnr.png`.

### Testing

```bash
python -m Forecast_nn.test --model simvp --variable sensible --amp
# Or evaluate the validation interval using the same selected checkpoint:
python -m Forecast_nn.test --model simvp --variable sensible --split val --amp
# Explicit checkpoint (including older ones):
python -m Forecast_nn.test --model simvp --checkpoint /path/to/best.pth
```

The test runner selects `best.pth` by default. It reads split boundaries and
PSNR reference ranges from the checkpoint, so the evaluation matches training.
For legacy checkpoints without PSNR metadata, the fixed range is reconstructed
from **their training period only**. If a legacy checkpoint was trained on
2023/2024 data, its results should **not** be considered a valid independent
2023 test; retrain under the new split.

Excel outputs include overall, per-horizon, per-channel,
per-horizon-per-channel and per-sample PSNR. `aggregate_metrics.py` also retains
PSNR when merging the workbooks.

The current `*_sde` architectures are **scalar SDE-informed models**, not yet
the full coupled `2 x 1` drift / `2 x 2` diffusion implementation. The
chronological infrastructure is ready for this later extension, which will
need its own joint-field data loader and covariance-aware training objective.

Regression tests with synthetic data (no ERA5 download needed):

```bash
python -m pytest -q Forecast_nn/tests/test_pipeline_integration.py
```
