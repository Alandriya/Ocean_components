"""Model registry used by the common train/test scripts."""
from Forecast_nn.models.unet import UNetForecaster
from Forecast_nn.models.unet_convlstm import UNetConvLSTMForecaster


_MODEL_REGISTRY = {
    # Model 1: recommended map/time architecture.
    'unet_convlstm': UNetConvLSTMForecaster,
    # Lightweight U-Net-only control model using ordered time channels.
    'unet': UNetForecaster,
}


def available_models():
    return tuple(sorted(_MODEL_REGISTRY))


def build_model(name, in_len, out_len, in_channels, model_params):
    key = str(name).lower()
    if key not in _MODEL_REGISTRY:
        known = ', '.join(available_models())
        raise KeyError(f'Model {name!r} is not registered. Available common-interface models: {known}')
    return _MODEL_REGISTRY[key](in_len=in_len, out_len=out_len, in_channels=in_channels, **dict(model_params))
