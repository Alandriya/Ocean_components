"""Model registry used by the common train/test scripts."""
from Forecast_nn.models.unet import UNetForecaster
from Forecast_nn.models.unet_convlstm import UNetConvLSTMForecaster
from Forecast_nn.models.simvp import SimVPForecaster
from Forecast_nn.models.fno import FNOForecaster
from Forecast_nn.models.unet_convlstm_sde import SDEUNetConvLSTMForecaster
from Forecast_nn.models.simvp_sde import SDESimVPForecaster
from Forecast_nn.models.fno_sde import SDEFNOForecaster
from Forecast_nn.models.swin_gru import SwinGRUForecaster


_MODEL_REGISTRY = {
    'unet_convlstm': UNetConvLSTMForecaster,
    'unet_convlstm_sde': SDEUNetConvLSTMForecaster,
    'unet': UNetForecaster,
    'simvp': SimVPForecaster,
    'simvp_sde': SDESimVPForecaster,
    'fno': FNOForecaster,
    'fno_sde': SDEFNOForecaster,
    'swin_gru': SwinGRUForecaster,
}

_SDE_CONDITIONED_MODELS = {'unet_convlstm_sde', 'simvp_sde', 'fno_sde'}


def available_models():
    return tuple(sorted(_MODEL_REGISTRY))


def model_uses_sde_conditioning(name):
    return str(name).lower() in _SDE_CONDITIONED_MODELS


def build_model(name, in_len, out_len, in_channels, model_params):
    key = str(name).lower()
    if key not in _MODEL_REGISTRY:
        known = ', '.join(available_models())
        raise KeyError(f'Model {name!r} is not registered. Available models: {known}')
    return _MODEL_REGISTRY[key](
        in_len=in_len,
        out_len=out_len,
        in_channels=in_channels,
        **dict(model_params),
    )
