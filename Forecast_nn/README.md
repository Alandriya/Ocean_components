# Forecast_nn — cleaned common framework

This folder replaces the duplicated launchers from the previous version with one common interface.

## Structure

```text
Forecast_nn/
├── config.py                  # experiment configuration
├── data.py                    # mmap reader, mask, normalization, sequence dataset
├── losses.py                  # common masked losses
├── evaluation.py              # metrics, XLSX and saved plot samples
├── runtime.py                 # CUDA / DataLoader helpers
├── train.py                   # common training launcher
├── test.py                    # common testing + baselines launcher
├── models/
│   ├── registry.py            # common-interface model registry
│   ├── unet_convlstm.py       # Model 1: U-Net + ConvLSTM forecaster
│   ├── unet.py                # lightweight U-Net control model
│   └── ...                    # preserved old model implementations
└── plotting/
    ├── training_curve.py
    └── forecast_comparison.py
```

Old duplicated top-level launchers, `__pycache__`, experimental handlers and obsolete plotting files were intentionally removed. Existing model implementations from the uploaded folder were preserved in `models/` unchanged.

Preserved legacy model sources: `SDE_HNN.py`, `attetion_unet.py`, `conv_gru.py`, `convlstm.py`, `dense_unet.py`, `encoder_decoder.py`, `ms_lstm.py`, `ms_predrnn.py`, `rbf_kan.py`, `residual_head.py`, and `sde_energy.py`. The uploaded archive contained only `.pyc` files (no recoverable `.py` source) for `predrnn_v2` and `sde_energy_v2`, so those two were not reconstructed from bytecode.

## Model 1: U-Net + ConvLSTM

`UNetConvLSTMForecaster` is the primary model in this iteration. Each historical map is passed through the same U-Net encoder, the bottleneck maps are processed chronologically by ConvLSTM, and the decoder uses skip features from the last observed map. The model accepts `[B, T_in, C, H, W]` and returns `[B, T_out, C, H, W]`.

Each input day is encoded separately with shared spatial weights, and ConvLSTM processes the bottleneck features in chronological order. The model can append the ocean mask and normalized x/y coordinates as static channels to every frame. Intermediate U-Net features can also be masked at every spatial scale. By default the output is residual relative to the last observed map, and the output layer is initialized to zero, so the initial model is exactly the persistence forecast.

A lightweight `unet` model is also registered as a control baseline. Future SimVP and FNO models can be added to `models/registry.py` without changing `train.py` or `test.py`.

## Configuration

Edit `Forecast_nn/config.py`. The important settings are:

```python
cfg.model_name = 'unet_convlstm'
cfg.variable = 'sensible'
cfg.downsample = 2
cfg.in_len = 30
cfg.out_len = 3
cfg.stride = 1
cfg.test_steps = 365

cfg.model_params['unet_convlstm'] = {
    'base_channels': 16,
    'depth': 3,
    'dropout': 0.05,
    'residual': True,
    'use_mask_channel': True,
    'use_coord_channels': True,
    'mask_features': True,
    'lstm_kernel': 3,
}
```

The default data files are:

```text
D:/Nastya/Data/OceanFull/DATA/Fluxes/sensible_grouped_1979-2024.npy
D:/Nastya/Data/OceanFull/DATA/Fluxes/latent_grouped_1979-2024.npy
```

The `(29141, time)` layout is read directly via numpy mmap. `downsample=2` converts each native `161x181` field to `81x91` using `[::2, ::2]` without creating a second dataset file.

## Training

From the repository root:

```bat
python -m Forecast_nn.train
```

Useful overrides:

```bat
python -m Forecast_nn.train --model unet_convlstm --variable sensible --epochs 50 --batch-size 4 --amp --num-workers 2
```

To continue the same run for another 15 epochs:

```bat
python -m Forecast_nn.train --resume --epochs 15
```

A checkpoint is saved after every epoch and includes model, optimizer, AMP scaler and complete history, so resume training is a real continuation rather than loading model weights only.

## Testing

```bat
python -m Forecast_nn.test
```

The test script compares:

- the selected neural network;
- historical mean of the input sequence;
- persistence (last observed field copied forward).

Each method gets a separate XLSX with overall, per-horizon and per-sample errors. The summary also contains temporal diagnostics:

- delta RMSE;
- correlation between predicted and true temporal increments;
- mean absolute predicted and true temporal change.

These metrics help detect the nearly-constant forecast collapse that occurred in the earlier architecture.

## Plots

Training curve:

```bat
python -m Forecast_nn.plotting.training_curve --input ".../training_history.npz"
```

Forecast comparison:

```bat
python -m Forecast_nn.plotting.forecast_comparison --input ".../forecast_samples_....npz" --method all
```

The comparison plot uses the requested three-row format: truth, forecast and absolute error.

## Adding the next model

Implement a class with the common interface:

```python
prediction = model(x, mask)
# x:          B,T_in,C,H,W
# prediction: B,T_out,C,H,W
```

Then add it to `Forecast_nn/models/registry.py`. The common train/test/evaluation/plotting code does not need to change.
