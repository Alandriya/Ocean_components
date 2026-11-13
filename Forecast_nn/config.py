"""Central configuration for Forecast_nn experiments on grouped ERA5 1979--2026.

The default forecast targets are sensible and latent. Other solo targets
(Flux, SST and Pressure) are already registered for future experiments.
Data length is inferred from the actual NPY header: the 2026 portion may be
incomplete, and ``2026`` in a filename is *not* a last observation date.
"""
import os
from pathlib import Path

from Forecast_nn.source_paths import (
    BASE_DATE, PERIOD, data_paths, grouped_length, joint_sde_source,
    last_available_date, scalar_sde_source,
)


class Config:
    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    # Optionally set OCEAN_DATA_ROOT on a Linux cluster or another computer.
    root_path = Path(os.environ.get('OCEAN_DATA_ROOT', r'D:/Nastya/Data/OceanFull'))
    variable = 'sensible'  # Primary targets: 'sensible', 'latent', or 'joint'.
    period = PERIOD
    data_files = data_paths(root_path, period)
    # The Russian alternatives are retained for reference:
    # variable = 'sensible'  # Основные цели: явный/скрытый тепловые потоки.
    mask_file = root_path / 'DATA' / 'mask'

    downsample = 2
    in_len = 30
    out_len = 3
    stride = 1
    base_date = BASE_DATE
    # Last actual observed day, not a hardcoded assumed full-year endpoint.
    data_last_date = last_available_date(data_files, variables=('sensible', 'latent'),
                                         base_date=BASE_DATE)
    val_steps = 730
    test_steps = 365
    early_stopping_patience = 12
    early_stopping_min_delta = 0.0
    psnr_percentiles = (0.1, 99.9)
    psnr_max_sample_frames = 256

    # -------------------------------------------------------------------------
    # Ito-SDE coefficients
    # -------------------------------------------------------------------------
    files_path_prefix = root_path
    sde_coefficient_method = 'sp'   # 'sp' or 'np'
    sde_1d_bins = 144              # Must match 1D coefficient calculation.
    sde_2d_bins = 12              # Documented for 2D bridge generation.
    sde_sources = {}  # Filled below, after class creation (Python class scope).
    # Available for the joint SimVP/GAN SDE-loss branch. The latter expects
    # diagonal-first B from Coefficients/(1979-2026)/sensible_latent.
    joint_sde_sources = joint_sde_source(root_path, names=('sensible', 'latent'),
                                         method=sde_coefficient_method)
    sde_joint_a_file = joint_sde_sources['a_file']
    sde_joint_b_file = joint_sde_sources['b_file']
    sde_joint_b_layout = joint_sde_sources['b_layout']

    # B in the scalar coefficient files is a diffusion *amplitude*.
    sde_b_kind = 'diffusion'  # 'diffusion' or 'variance'
    sde_weight = 1e-4
    sde_dt = 1.0
    sde_eps = 1e-6
    sde_feature_clip = 5.0

    # Use the last already-observed coefficient A_t/B_t, never A_(t+1).
    sde_loss_mode = 'first_step'  # 'first_step' or 'last_observed_all'

    # -------------------------------------------------------------------------
    # Models (unchanged defaults)
    # -------------------------------------------------------------------------
    model_name = 'unet_convlstm'

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
        'unet_convlstm_sde': {
            'base_channels': 16,
            'depth': 3,
            'dropout': 0.05,
            'residual': True,
            'use_mask_channel': True,
            'use_coord_channels': True,
            'mask_features': True,
            'lstm_kernel': 3,
            'sde_condition_scale': 0.10,
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
        'simvp': {
            'base_channels': 16,
            'depth': 3,
            'dropout': 0.05,
            'residual': True,
            'use_mask_channel': True,
            'use_coord_channels': True,
            'mask_features': True,
            'temporal_blocks': 4,
            'temporal_kernel': 3,
            'spatial_kernel': 3,
            'horizon_embedding': True,
        },
        'simvp_sde': {
            'base_channels': 16,
            'depth': 3,
            'dropout': 0.05,
            'residual': True,
            'use_mask_channel': True,
            'use_coord_channels': True,
            'mask_features': True,
            'temporal_blocks': 4,
            'temporal_kernel': 3,
            'spatial_kernel': 3,
            'horizon_embedding': True,
            'sde_condition_scale': 0.10,
        },
        'fno': {
            'width': 48,
            'modes_y': 14,
            'modes_x': 14,
            'layers': 3,
            'projection_width': 96,
            'dropout': 0.0,
            'residual': True,
            'use_mask_channel': True,
            'use_coord_channels': True,
            'mask_features': True,
            'padding': 2,
        },
        'swin_gru': {
            'base_channels': 64,
            'temporal_tokens': 8,
            'attention_depth': 2,
            'heads': 8,
            'window_size': 4,
            'dropout': 0.05,
            'residual': True,
            'use_mask_channel': True,
            'use_coord_channels': True,
            'mask_features': True,
        },
        'fno_sde': {
            'width': 48,
            'modes_y': 14,
            'modes_x': 14,
            'layers': 3,
            'projection_width': 96,
            'dropout': 0.0,
            'residual': True,
            'use_mask_channel': True,
            'use_coord_channels': True,
            'mask_features': True,
            'padding': 2,
            'sde_history': True,
        },
    }

    # -------------------------------------------------------------------------
    # Optimization
    # -------------------------------------------------------------------------
    epochs = 50
    batch_size = 16
    learning_rate = 3.5e-4
    weight_decay = 1e-4
    num_workers = 4
    amp = True
    fast_cuda = True
    seed = 2025
    log_every = 50
    dynamic_weight = 0.0
    checkpoint_every = 1

    # -------------------------------------------------------------------------
    # Evaluation / plotting
    # -------------------------------------------------------------------------
    plot_samples = (0,)
    plot_language = 'ru'
    ssim_window = 11
    ssim_sigma = 1.5
    results_root = root_path / 'Forecast' / 'Results'
    stats_root = root_path / 'DATA' / 'Forecast_nn_stats'


cfg = Config()
cfg.sde_sources = {
    name: scalar_sde_source(cfg.root_path, name, bins=cfg.sde_1d_bins,
                            method=cfg.sde_coefficient_method,
                            field_length=(grouped_length(path)
                                          if path.is_file() else None))
    for name, path in cfg.data_files.items()
}

# -----------------------------------------------------------------------------
# Legacy model compatibility (unmodified)
# -----------------------------------------------------------------------------
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
