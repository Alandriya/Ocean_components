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


def sde_energy_loss(pred, last_input, mask, sde_lookup, dt=1.0, eps=1e-6):
    """Article-style quadratic SDE penalty on predicted increments.

    For each horizon k:
        delta_k = x_hat_k - x_prev
        penalty = 0.5 * (delta_k - a(x_prev)*dt)^2 / variance

    If lookup.b_kind == 'variance', b is used as the one-step conditional
    variance and scales linearly with dt. If b_kind == 'diffusion', b is the
    diffusion amplitude from dX=a dt+b dW, hence variance=b^2*dt.

    The lookup is re-evaluated on the previous *predicted* state for horizons
    +2, +3, ...; only the first transition uses the last observed input map.
    """
    pred = pred.float()
    prev = last_input.float()
    valid = _valid_mask(mask, pred)
    energy_terms = []
    drift_terms = []
    dt = float(dt)
    eps = float(eps)

    for k in range(pred.size(1)):
        current = pred[:, k]
        # lookup expects B,T,C,H,W
        a, b = sde_lookup.lookup(prev[:, None])
        a = a[:, 0]
        b = b[:, 0]
        target_increment = a * dt
        delta = current - prev
        if sde_lookup.b_kind == 'variance':
            variance = b.clamp_min(0.0) * dt
        else:
            variance = b.square() * dt
        energy_terms.append(0.5 * (delta - target_increment).square() / (variance + eps))
        drift_terms.append((delta - target_increment).square())
        prev = current

    energy = torch.stack(energy_terms, dim=1).masked_select(valid).mean()
    drift = torch.stack(drift_terms, dim=1).masked_select(valid).mean()
    return energy, drift


def forecast_loss(pred, target, last_input, mask, dynamic_weight=0.0,
                  sde_lookup=None, sde_weight=0.0, sde_dt=1.0, sde_eps=1e-6):
    pred32 = pred.float()
    target32 = target.float()
    last32 = last_input.float()

    mse = masked_mse(pred32, target32, mask)
    dynamic = masked_dynamic_mse(pred32, target32, last32, mask)
    total = mse + float(dynamic_weight) * dynamic

    zero = mse.detach().new_zeros(())
    sde = zero
    sde_drift = zero
    if sde_lookup is not None and float(sde_weight) > 0.0:
        sde, sde_drift = sde_energy_loss(
            pred32, last32, mask, sde_lookup,
            dt=sde_dt, eps=sde_eps,
        )
        total = total + float(sde_weight) * sde

    return total, {
        'mse': mse.detach(),
        'dynamic': dynamic.detach(),
        'sde': sde.detach(),
        'sde_drift': sde_drift.detach(),
    }
