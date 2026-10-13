"""Central configuration for all Forecast_nn experiments."""
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

    downsample = 2
    in_len = 30
    out_len = 3
    stride = 1
    test_steps = 365

    # -------------------------------------------------------------------------
    # Ito-SDE information
    # -------------------------------------------------------------------------
    # These defaults point to the statistics files produced by the earlier
    # SDE-Energy pipeline. Change the paths if your a/b tables live elsewhere.
    # Expected legacy keys: bin_edges, a, b. Generic tables with
    # values/centers + a/drift + b/variance/diffusion are also accepted.
    sde_sources = {
        'sensible': {
            'file': root_path / 'DATA' / 'Fluxes' / 'sde_energy_stats_sensible_ds2_stride_1_bins_500.npz',
        },
        'latent': {
            'file': root_path / 'DATA' / 'Fluxes' / 'sde_energy_stats_latent_ds2_stride_1_bins_500.npz',
        },
    }

    # The old SDE-Energy statistics store b=Var[dX|X] in normalized units.
    # If your b is the diffusion amplitude in dX=a dt+b dW, set 'diffusion'.
    sde_b_kind = 'variance'  # 'variance' or 'diffusion'
    sde_weight = 1e-4
    sde_dt = 1.0
    sde_eps = 1e-6
    sde_feature_clip = 5.0

    # -------------------------------------------------------------------------
    # Model
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
            # True: append a/b for all T_in days. False: only the last day.
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
    base_date = '1979-01-01'
    plot_language = 'ru'

    # SSIM: standard constants with a masked Gaussian window.
    ssim_window = 11
    ssim_sigma = 1.5

    results_root = root_path / 'Forecast' / 'Results'
    stats_root = root_path / 'DATA' / 'Forecast_nn_stats'


cfg = Config()

# -----------------------------------------------------------------------------
# Legacy-model compatibility
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
