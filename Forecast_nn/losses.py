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


def _split_sde_coefficients(sde_last_coefficients, channels):
    if sde_last_coefficients is None:
        raise ValueError('sde_last_coefficients are required for time-aligned SDE loss.')
    if sde_last_coefficients.ndim != 4:
        raise ValueError(
            f'sde_last_coefficients must be [B,2*C,H,W], got {tuple(sde_last_coefficients.shape)}.'
        )
    if sde_last_coefficients.size(1) != 2 * channels:
        raise ValueError(
            f'Expected {2*channels} SDE coefficient channels, got {sde_last_coefficients.size(1)}.'
        )
    return (
        sde_last_coefficients[:, :channels].float(),
        sde_last_coefficients[:, channels:].float(),
    )


def sde_field_loss(pred, last_input, mask, sde_last_coefficients,
                   b_kind='diffusion', dt=1.0, eps=1e-6,
                   mode='first_step'):
    """Causal SDE penalty using only the last observed A_t/B_t map.

    The saved A_t/B_t maps are aligned with observed state X_t and were computed
    from X_t-X_{t-1}.  Using A_{t+1}/B_{t+1} during training would therefore use
    information from future target X_{t+1}.  This loss never reads those maps.

    Modes
    -----
    first_step (default):
        penalize only X_hat_{t+1}-X_t using the known A_t/B_t.
    last_observed_all:
        reuse the same known A_t/B_t for every forecast increment. This remains
        causal but assumes the local coefficient field is approximately constant
        over the short forecast horizon.
    """
    pred = pred.float()
    last_input = last_input.float()
    channels = pred.size(2)
    a, b = _split_sde_coefficients(sde_last_coefficients, channels)
    b_kind = str(b_kind).lower()
    if b_kind not in ('variance', 'diffusion'):
        raise ValueError("b_kind must be 'variance' or 'diffusion'.")
    mode = str(mode).lower()
    if mode not in ('first_step', 'last_observed_all'):
        raise ValueError("sde loss mode must be 'first_step' or 'last_observed_all'.")

    dt = float(dt)
    eps = float(eps)
    if b_kind == 'variance':
        variance = b.clamp_min(0.0) * dt
    else:
        variance = b.square() * dt
    target_increment = a * dt

    if mode == 'first_step':
        delta = pred[:, 0] - last_input
        valid = _valid_mask(mask, delta[:, None])[:, 0]
        sq = (delta - target_increment).square()
        energy = (0.5 * sq / (variance + eps)).masked_select(valid).mean()
        drift = sq.masked_select(valid).mean()
        return energy, drift

    prev = last_input
    energy_terms = []
    drift_terms = []
    for k in range(pred.size(1)):
        delta = pred[:, k] - prev
        sq = (delta - target_increment).square()
        energy_terms.append(0.5 * sq / (variance + eps))
        drift_terms.append(sq)
        prev = pred[:, k]
    valid = _valid_mask(mask, pred)
    energy = torch.stack(energy_terms, dim=1).masked_select(valid).mean()
    drift = torch.stack(drift_terms, dim=1).masked_select(valid).mean()
    return energy, drift


def forecast_loss(pred, target, last_input, mask, dynamic_weight=0.0,
                  sde_last_coefficients=None, sde_weight=0.0,
                  sde_b_kind='diffusion', sde_loss_mode='first_step',
                  sde_dt=1.0, sde_eps=1e-6):
    pred32 = pred.float()
    target32 = target.float()
    last32 = last_input.float()

    mse = masked_mse(pred32, target32, mask)
    dynamic = masked_dynamic_mse(pred32, target32, last32, mask)
    total = mse + float(dynamic_weight) * dynamic

    zero = mse.detach().new_zeros(())
    sde = zero
    sde_drift = zero
    if sde_last_coefficients is not None and float(sde_weight) > 0.0:
        sde, sde_drift = sde_field_loss(
            pred32,
            last32,
            mask,
            sde_last_coefficients.float(),
            b_kind=sde_b_kind,
            dt=sde_dt,
            eps=sde_eps,
            mode=sde_loss_mode,
        )
        total = total + float(sde_weight) * sde

    return total, {
        'mse': mse.detach(),
        'dynamic': dynamic.detach(),
        'sde': sde.detach(),
        'sde_drift': sde_drift.detach(),
    }
