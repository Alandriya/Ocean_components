"""Central configuration for all Forecast_nn experiments.

Edit this file for normal runs. The train/test scripts also expose a few CLI overrides
(model, variable, epochs, batch size, learning rate, etc.) for quick experiments.
"""
from pathlib import Path


class Config:
    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    root_path = Path(r'D:/Nastya/Data/OceanFull')
    variable = 'sensible'  # 'sensible' or 'latent'
    data_files = {
        'sensible': root_path / 'DATA' / 'Fluxes' / 'sensible_grouped_1979-2024.npy',
        'latent': root_path / 'DATA' / 'Fluxes' / 'latent_grouped_1979-2024.npy',
    }
    mask_file = root_path / 'DATA' / 'mask'

    # Native source grid is 161x181. downsample=2 reproduces the old [::2, ::2]
    # preprocessing and gives 81x91 maps.
    downsample = 2
    in_len = 30
    out_len = 3
    stride = 1

    # The last test_steps are never used for training statistics or optimization.
    test_steps = 365

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    # Registered models can be selected without changing train.py/test.py.
    model_name = 'unet_convlstm'

    # Parameters are stored per model so future architectures can live behind the
    # same train/test interface. New architectures only need a registry entry.
    model_params = {
        'unet_convlstm': {
            'base_channels': 16,
            'depth': 3,
            'dropout': 0.05,
            'residual': True,
            'use_mask_channel': True,
            'use_coord_channels': True,
            'mask_features': True,
            'lstm_kernel': 3,
        },
        'unet': {
            'base_channels': 16,
            'depth': 3,
            'dropout': 0.05,
            'residual': True,
            'use_mask_channel': True,
            'use_coord_channels': True,
            'mask_features': True,
        },
    }

    # -------------------------------------------------------------------------
    # Optimization
    # -------------------------------------------------------------------------
    epochs = 50
    batch_size = 4
    learning_rate = 3.5e-4
    weight_decay = 1e-4
    num_workers = 2
    amp = True
    fast_cuda = True
    seed = 2025
    log_every = 50

    # Start with plain masked MSE for an interpretable architecture comparison.
    # Set dynamic_weight > 0 only if you explicitly want to penalize temporal
    # increments in addition to forecast values.
    dynamic_weight = 0.0

    # Save an always-resumable checkpoint after each epoch.
    checkpoint_every = 1

    # -------------------------------------------------------------------------
    # Evaluation / plotting
    # -------------------------------------------------------------------------
    plot_samples = (0,)  # zero-based sample IDs from the test dataset
    base_date = '1979-01-01'
    plot_language = 'ru'

    # All outputs from the common framework are grouped here.
    results_root = root_path / 'Forecast' / 'Results'
    stats_root = root_path / 'DATA' / 'Forecast_nn_stats'


cfg = Config()

# -----------------------------------------------------------------------------
# Legacy-model compatibility
# -----------------------------------------------------------------------------
# The old model files are preserved under Forecast_nn/models as requested. They
# were written against the former cfg object, so the most common attributes are
# kept here to make later adaptation easier. They are not used by the new U-Net.
try:
    import torch.nn as nn
    cfg.LSTM_conv = nn.Conv2d
except Exception:
    cfg.LSTM_conv = None
cfg.channels = 1
cfg.features_amount = 1
cfg.batch = cfg.batch_size
cfg.lstm_hidden_state = 32
cfg.LSTM_layers = 6
cfg.kernel_size = 2
