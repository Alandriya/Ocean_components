"""Model-agnostic metrics, XLSX export and saved plotting samples."""
from pathlib import Path
import numpy as np


def _excel_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (tuple, list)):
        if len(value) == 2 and all(isinstance(v, (int, np.integer)) for v in value):
            return f'{int(value[0])}x{int(value[1])}'
        return ', '.join(str(_excel_value(v)) for v in value)
    if isinstance(value, dict):
        return ', '.join(f'{k}={_excel_value(v)}' for k, v in value.items())
    return value


def _short(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return f'{value:g}'.replace('+', '')
    return value


def build_run_tag(model_name, variable, downsample, in_len, out_len, stride, model_params):
    aliases = {
        'base_channels': 'base', 'depth': 'depth', 'dropout': 'drop',
        'residual': 'res', 'use_mask_channel': 'mask',
        'use_coord_channels': 'xy', 'mask_features': 'mf',
        'temporal_blocks': 'tb', 'temporal_kernel': 'tk', 'spatial_kernel': 'sk',
        'horizon_embedding': 'he', 'projection_width': 'proj',
        'sde_condition_scale': 'sdec', 'sde_history': 'sdehist',
    }
    parts = [f'ds{downsample}', f'in{in_len}', f'out{out_len}', f'st{stride}']
    for key, value in model_params.items():
        name = aliases.get(key, key)
        parts.append(f'{name}{_short(value)}')
    return '_'.join(map(str, parts))


def append_sde_run_tag(run_tag, sde_weight, b_kind):
    return f'{run_tag}_sdew{_short(float(sde_weight))}_sdeb{str(b_kind)}'


def _gaussian_kernel(size=11, sigma=1.5):
    size = int(size)
    if size < 3 or size % 2 == 0:
        raise ValueError('SSIM window must be an odd integer >= 3.')
    x = np.arange(size, dtype=np.float64) - size // 2
    k = np.exp(-(x * x) / (2.0 * float(sigma) ** 2))
    return k / k.sum()


def _separable_blur_2d(arr, kernel):
    """Reflect-padded separable convolution without scipy dependency."""
    arr = np.asarray(arr, dtype=np.float64)
    pad = len(kernel) // 2
    h, w = arr.shape
    vp = np.pad(arr, ((pad, pad), (0, 0)), mode='reflect')
    tmp = np.zeros_like(arr, dtype=np.float64)
    for i, weight in enumerate(kernel):
        tmp += float(weight) * vp[i:i + h]
    hp = np.pad(tmp, ((0, 0), (pad, pad)), mode='reflect')
    out = np.zeros_like(arr, dtype=np.float64)
    for i, weight in enumerate(kernel):
        out += float(weight) * hp[:, i:i + w]
    return out


def masked_ssim2d(pred, truth, mask, window=11, sigma=1.5):
    """Masked local SSIM for one 2-D physical field.

    Local moments are computed with a Gaussian window using only ocean pixels.
    The dynamic range is derived from the truth map, which keeps the comparison
    consistent across NN and baseline predictions for the same target.
    """
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if pred.shape != truth.shape or pred.shape != valid.shape:
        raise ValueError(f'SSIM shape mismatch: pred={pred.shape}, truth={truth.shape}, mask={valid.shape}.')
    if not np.any(valid):
        return float('nan')

    kernel = _gaussian_kernel(window, sigma)
    m = valid.astype(np.float64)
    support = _separable_blur_2d(m, kernel)
    denom = np.maximum(support, 1e-12)

    mu_x = _separable_blur_2d(pred * m, kernel) / denom
    mu_y = _separable_blur_2d(truth * m, kernel) / denom
    ex2 = _separable_blur_2d(pred * pred * m, kernel) / denom
    ey2 = _separable_blur_2d(truth * truth * m, kernel) / denom
    exy = _separable_blur_2d(pred * truth * m, kernel) / denom
    var_x = np.maximum(ex2 - mu_x * mu_x, 0.0)
    var_y = np.maximum(ey2 - mu_y * mu_y, 0.0)
    cov_xy = exy - mu_x * mu_y

    vals = truth[valid]
    data_range = float(np.nanmax(vals) - np.nanmin(vals))
    if not np.isfinite(data_range) or data_range < 1e-8:
        data_range = max(float(np.nanmax(np.abs(vals))), 1.0)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    num = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    ssim_map = num / np.maximum(den, 1e-30)

    # Require reasonable local ocean support to avoid coastline windows dominated
    # by reflected land/missing values.
    local_valid = valid & (support >= 0.5)
    if not np.any(local_valid):
        local_valid = valid
    return float(np.nanmean(ssim_map[local_valid]))


def masked_ssim(pred, truth, mask, window=11, sigma=1.5):
    """Average SSIM across channels for [C,H,W] fields."""
    pred = np.asarray(pred)
    truth = np.asarray(truth)
    if pred.ndim == 2:
        return masked_ssim2d(pred, truth, mask, window, sigma)
    scores = [masked_ssim2d(pred[c], truth[c], mask, window, sigma) for c in range(pred.shape[0])]
    return float(np.nanmean(scores))


class MetricAccumulator:
    """Masked value errors, SSIM and temporal-change diagnostics in physical units."""
    def __init__(self, out_len, ssim_window=11, ssim_sigma=1.5):
        self.out_len = int(out_len)
        self.ssim_window = int(ssim_window)
        self.ssim_sigma = float(ssim_sigma)
        self.sq_sum = np.zeros(out_len, dtype=np.float64)
        self.abs_sum = np.zeros(out_len, dtype=np.float64)
        self.count = np.zeros(out_len, dtype=np.int64)
        self.ssim_sum = np.zeros(out_len, dtype=np.float64)
        self.ssim_count = np.zeros(out_len, dtype=np.int64)
        self.sample_rows = []
        self.delta_sq_sum = 0.0
        self.delta_abs_pred_sum = 0.0
        self.delta_abs_true_sum = 0.0
        self.delta_count = 0
        self.dx_sum = self.dy_sum = self.dxx_sum = self.dyy_sum = self.dxy_sum = 0.0

    def update(self, pred, truth, last_input, mask, sample_offset):
        valid = mask.astype(bool)
        error = pred - truth
        for b in range(error.shape[0]):
            for t in range(self.out_len):
                e = error[b, t, :, valid].reshape(-1)
                self.sq_sum[t] += np.square(e, dtype=np.float64).sum()
                self.abs_sum[t] += np.abs(e).sum(dtype=np.float64)
                self.count[t] += e.size
                mse = float(np.mean(np.square(e, dtype=np.float64)))
                mae = float(np.mean(np.abs(e)))
                ssim = masked_ssim(
                    pred[b, t], truth[b, t], valid,
                    window=self.ssim_window, sigma=self.ssim_sigma,
                )
                if np.isfinite(ssim):
                    self.ssim_sum[t] += ssim
                    self.ssim_count[t] += 1
                self.sample_rows.append((sample_offset + b, t + 1, mse, mae, float(np.sqrt(mse)), ssim))

        pred_full = np.concatenate([last_input[:, None], pred], axis=1)
        truth_full = np.concatenate([last_input[:, None], truth], axis=1)
        dp = np.diff(pred_full, axis=1)[:, :, :, valid].reshape(-1).astype(np.float64)
        dt = np.diff(truth_full, axis=1)[:, :, :, valid].reshape(-1).astype(np.float64)
        self.delta_sq_sum += np.square(dp - dt).sum()
        self.delta_abs_pred_sum += np.abs(dp).sum()
        self.delta_abs_true_sum += np.abs(dt).sum()
        self.delta_count += dp.size
        self.dx_sum += dp.sum(); self.dy_sum += dt.sum()
        self.dxx_sum += np.square(dp).sum(); self.dyy_sum += np.square(dt).sum(); self.dxy_sum += (dp * dt).sum()

    def summary(self):
        mse = self.sq_sum / np.maximum(self.count, 1)
        mae = self.abs_sum / np.maximum(self.count, 1)
        ssim = self.ssim_sum / np.maximum(self.ssim_count, 1)
        overall_mse = float(self.sq_sum.sum() / max(self.count.sum(), 1))
        n = max(self.delta_count, 1)
        cov = self.dxy_sum - self.dx_sum * self.dy_sum / n
        var_x = self.dxx_sum - self.dx_sum ** 2 / n
        var_y = self.dyy_sum - self.dy_sum ** 2 / n
        corr = cov / np.sqrt(max(var_x * var_y, 1e-30))
        total_ssim_count = max(int(self.ssim_count.sum()), 1)
        return {
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse),
            'ssim': ssim,
            'overall_mse': overall_mse,
            'overall_mae': float(self.abs_sum.sum() / max(self.count.sum(), 1)),
            'overall_rmse': float(np.sqrt(overall_mse)),
            'overall_ssim': float(self.ssim_sum.sum() / total_ssim_count),
            'delta_rmse': float(np.sqrt(self.delta_sq_sum / n)),
            'pred_mean_abs_change': float(self.delta_abs_pred_sum / n),
            'true_mean_abs_change': float(self.delta_abs_true_sum / n),
            'delta_correlation': float(corr),
        }


def save_metrics_xlsx(path, method, variable, run_tag, metrics, checkpoint, sample_rows):
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
    except ImportError as exc:
        raise ImportError('XLSX export requires openpyxl: pip install openpyxl') from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = 'summary'
    fill = PatternFill('solid', fgColor='D9EAF7')
    ws.append(['Parameter', 'Value'])
    for cell in ws[1]:
        cell.font = Font(bold=True); cell.fill = fill

    rows = [
        ('Variable', variable), ('Method', method), ('Run tag', run_tag),
        ('Overall MSE', metrics['overall_mse']), ('Overall MAE', metrics['overall_mae']),
        ('Overall RMSE', metrics['overall_rmse']), ('Overall SSIM', metrics['overall_ssim']),
        ('Delta RMSE', metrics['delta_rmse']), ('Predicted mean |delta|', metrics['pred_mean_abs_change']),
        ('True mean |delta|', metrics['true_mean_abs_change']), ('Delta correlation', metrics['delta_correlation']),
    ]
    for key in (
        'model_name', 'downsample', 'grid_shape', 'in_len', 'out_len', 'stride',
        'train_end_idx', 'epochs', 'learning_rate', 'dynamic_weight', 'sde_enabled',
        'sde_weight', 'sde_b_kind', 'sde_dt', 'sde_source',
    ):
        if key in checkpoint:
            rows.append((key, checkpoint.get(key)))
    for key, value in checkpoint.get('model_kwargs', {}).items():
        rows.append((key, value))
    for row in rows:
        ws.append([_excel_value(v) for v in row])
    ws.column_dimensions['A'].width = 30; ws.column_dimensions['B'].width = 42

    wh = wb.create_sheet('per_horizon')
    wh.append(['Horizon', 'MSE', 'MAE', 'RMSE', 'SSIM'])
    for cell in wh[1]:
        cell.font = Font(bold=True); cell.fill = fill
    for t in range(len(metrics['mse'])):
        wh.append([
            t + 1, float(metrics['mse'][t]), float(metrics['mae'][t]),
            float(metrics['rmse'][t]), float(metrics['ssim'][t]),
        ])

    wsamp = wb.create_sheet('per_sample')
    wsamp.append(['Sample', 'Horizon', 'MSE', 'MAE', 'RMSE', 'SSIM'])
    for cell in wsamp[1]:
        cell.font = Font(bold=True); cell.fill = fill
    for row in sample_rows:
        wsamp.append([_excel_value(v) for v in row])

    for sheet in (ws, wh, wsamp):
        sheet.freeze_panes = 'A2'
        for row in sheet.iter_rows():
            for cell in row:
                cell.alignment = Alignment(vertical='center')
    wb.save(path)


def save_plot_samples(path, records, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: np.asarray(value) for key, value in records.items()}
    for key, value in metadata.items():
        payload[f'meta_{key}'] = np.asarray(value)
    np.savez_compressed(path, **payload)
