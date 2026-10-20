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

## Text logs (train and test)

Each invocation of `python -m Forecast_nn.train` or `python -m Forecast_nn.test`
prints to the terminal **and simultaneously** writes its stdout and stderr to
`Forecast_nn/logs/`. The folder is created automatically. No new CLI options
are needed.

By default, a sensible-flux U-Net + ConvLSTM run writes:

```text
Forecast_nn/logs/train_unet_convlstm_ds2_i30_o3_b16_d3.txt
Forecast_nn/logs/test_unet_convlstm_ds2_i30_o3_b16_d3.txt
```

For `--variable latent` or `--variable joint`, `_latent` or `_joint` is appended
before `.txt` so their logs cannot collide. `test.py --split val` appends `_val`.
Existing compact run tags, including optional `--run-id`, model size and SDE
suffixes, are reused verbatim. Paths to saved model checkpoints are unchanged.

Logs are **appended** on subsequent starts and `--resume`, rather than overwritten.
Every invocation starts with a timestamp, the command line and the log path;
uncaught Python tracebacks are also logged. The log is flushed during execution,
so it can be viewed live on the cluster with `tail -f`.

## Compact experiment folder names

Results continue to live under `results_root/<model>/<variable>/<run_tag>/`.
The run tag now contains **only short, relevant hyperparameters**. For example:

- U-Net + ConvLSTM: `ds2_i30_o3_b16_d3`
- SimVP: `ds2_i30_o3_b16_d3_tb4`
- FNO: `ds2_i30_o3_w48_my14_mx14_l3`
- SimVP + SDE: `ds2_i30_o3_b16_d3_tb4_sw1e-4`

Here `ds` is downsampling, `i` input history, `o` output horizon,
`b` base channels, `d` depth, `tb` temporal blocks, `w` FNO width,
`my/mx` Fourier modes, `l` FNO blocks, `sw` SDE-loss weight.
Stride is included only if it is not 1. Non-default dropout, kernel sizes,
mask/coordinate flags and other architecture switches appear only when changed.

**Date/split information (`_end..._val..._test...`) is no longer added to names.**
The full settings (including train/validation/test boundaries, dates, and
model hyperparameters) are still stored in `best.pth` and `last.pth`.
When an existing folder has a different chronological split, training refuses
to overwrite it. To keep independent runs with different splits, random seeds or
optimizer settings, use a short explicit `--run-id`:

```bash
python -m Forecast_nn.train --model simvp --run-id seed2
python -m Forecast_nn.test --model simvp --run-id seed2
```

Existing experiments with older long names are **not automatically moved or
renamed**. Pass `--checkpoint /path/to/old/best.pth` to evaluate an old run.

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
RMSE improvement required to reset the patience counter. The output directory uses compact architecture tags; chronological split
boundaries are stored inside each checkpoint rather than in the directory name.

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
new `joint` data mode supports two simultaneous flux channels but **not yet**
the full coupled 2x2 stochastic drift/diffusion loss. That is a separate model extension.

Regression tests with synthetic data (no ERA5 download needed):

```bash
python -m pytest -q Forecast_nn/tests/test_pipeline_integration.py
```


## New heavy deterministic model: `swin_gru` (CNN + attention + ConvGRU)

Implementation: [`models/swin_gru.py`](models/swin_gru.py). This is a new,
standalone hybrid **inspired by**, but NOT an exact reimplementation of,
SwinLSTM or Swin Transformer. There are no downloaded pretrained weights.

Design:

1. A shared CNN encoder encodes **every** historical frame at scales 1/2,
   1/4, 1/8, including a mask/coordinate option and masking of land features.
2. Learned **temporal cross-attention** compresses all `in_len` historical
   bottleneck frames to `temporal_tokens` ordered latent states without
   making tokens for every full-resolution space-time pixel.
3. A **ConvGRU** recurrent cell consumes these states; local window self-attention
   with alternating regular and shifted windows mixes spatial information.
4. U-Net-like skip features from the **last observed frame** are decoded once
   for each requested horizon, with horizon embeddings, to direct residual
   predictions relative to the last observed day.
5. Predicts deterministically: no SDE regularization, future inputs, sampling,
   ERA5 forecasts or physical coefficient maps are used.

The default size is about **7.5 million parameters** in the `joint` case.
Long sequences and spatial attention are computationally expensive; use an
initial per-GPU batch size of **2–4**, rather than the old global default 16,
then measure GPU memory and increase if possible. The trainer currently uses
**one GPU per invocation**; it does **not** contain distributed data-parallel
logic. Gradient accumulation lets you reach a larger effective batch size on
one GPU.

### Option A: the original single-field input

```bash
python -m Forecast_nn.train --model swin_gru --variable sensible \
  --epochs 100 --lr 1e-4 --batch-size 2 --grad-accum 8 \
  --num-workers 4 --grad-clip 1 --patience 15 --amp

python -m Forecast_nn.test --model swin_gru --variable sensible --batch-size 2 --amp
```

### Option B: both historical fluxes, **joint** prediction (recommended)

```bash
python -m Forecast_nn.train --model swin_gru --variable joint \
  --epochs 100 --lr 1e-4 --batch-size 2 --grad-accum 8 \
  --num-workers 4 --grad-clip 1 --patience 15 --amp

python -m Forecast_nn.test --model swin_gru --variable joint --batch-size 2 --amp
```

`joint` reads the existing `sensible` and `latent` files given in `config.py`
without duplication or temporal re-interpolation. Both files must have the
same number of frames and identical grids; the loader channels are
`[sensible, latent]`. Means, standard deviations and PSNR ranges are calculated
**per channel from training data only**. The last date is still 2023-12-31.
The test and persistence baselines use the same two channels and dates.
Do not use `--variable joint` with the legacy `*_sde` models: those use a
single-component SDE and must be generalized separately.

### Architectural experiments from the CLI

All size overrides are **specific to** `--model swin_gru`:

```bash
python -m Forecast_nn.train --model swin_gru --variable joint \
  --base-channels 48 --temporal-tokens 6 --attention-depth 2 \
  --heads 8 --window-size 4 --run-id compact \
  --lr 1e-4 --batch-size 4 --grad-accum 4 --epochs 100 --amp

python -m Forecast_nn.test --model swin_gru --variable joint \
  --base-channels 48 --temporal-tokens 6 --attention-depth 2 \
  --heads 8 --window-size 4 --run-id compact --batch-size 4 --amp
```

For experiments with altered architecture settings, provide the same
architectural flags and `--run-id` during testing, or specify
`--checkpoint path/to/best.pth` explicitly. Architecture arguments are stored
inside the checkpoint and are not inferred from the current config when
loading a checkpoint explicitly.

Example default compact tag:
`ds2_i30_o3_b64_tt8_ad2_h8_ws4` inside
`Results/swin_gru/joint/`.

Gradient accumulation (`--grad-accum`, default 1) and gradient clipping
(`--grad-clip`, default 1) are also available for existing models.
The effective batch has approximately `batch-size * grad-accum` examples.

### Tests

```bash
python -m pytest -q Forecast_nn/tests
```

The tests include masked/odd-sized shape checks, gradient flow through temporal
and window attention, a two-field reader, an end-to-end joint-field
training/validation/test loop, PSNR and persisted `best.pth` metadata.
They use synthetic data, not ERA5; production speed, GPU RAM, and forecast
accuracy must still be measured on the actual cluster.

Original methodology references:
- [Liu et al., Swin Transformer, ICCV 2021](https://arxiv.org/abs/2103.14030)
- [Tang et al., SwinLSTM, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Tang_SwinLSTM_Improving_Spatiotemporal_Prediction_Accuracy_using_Swin_Transformer_and_LSTM_ICCV_2023_paper.html)
- [PyTorch Scaled Dot Product Attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
