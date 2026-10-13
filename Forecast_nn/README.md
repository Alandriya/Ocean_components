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
