"""Common losses for all forecasting architectures."""
import torch


def _valid_mask(mask, tensor):
    if mask.ndim == 2:
        mask = mask[None, None, None]
    elif mask.ndim == 3:
        mask = mask[None, None]
    elif mask.ndim == 4:
        mask = mask[:, None]
    return mask.to(device=tensor.device, dtype=torch.bool).expand_as(tensor)


def masked_mse(pred, target, mask):
    valid = _valid_mask(mask, pred)
    return (pred - target).square().masked_select(valid).mean()


def masked_dynamic_mse(pred, target, last_input, mask):
    pred_full = torch.cat([last_input[:, None], pred], dim=1)
    target_full = torch.cat([last_input[:, None], target], dim=1)
    pred_delta = pred_full[:, 1:] - pred_full[:, :-1]
    target_delta = target_full[:, 1:] - target_full[:, :-1]
    return masked_mse(pred_delta, target_delta, mask)


def forecast_loss(pred, target, last_input, mask, dynamic_weight=0.0):
    mse = masked_mse(pred.float(), target.float(), mask)
    dynamic = masked_dynamic_mse(pred.float(), target.float(), last_input.float(), mask)
    total = mse + float(dynamic_weight) * dynamic
    return total, {'mse': mse.detach(), 'dynamic': dynamic.detach()}
