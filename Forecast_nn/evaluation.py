"""Model-agnostic metrics, XLSX export and saved plotting samples."""
from pathlib import Path
import os
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
    }
    parts = [model_name, variable, f'ds{downsample}', f'in{in_len}', f'out{out_len}', f'st{stride}']
    for key, value in model_params.items():
        name = aliases.get(key, key)
        parts.append(f'{name}{_short(value)}')
    return '_'.join(map(str, parts))


class MetricAccumulator:
    """Masked value errors plus temporal-change diagnostics in physical units."""
    def __init__(self, out_len):
        self.out_len = int(out_len)
        self.sq_sum = np.zeros(out_len, dtype=np.float64)
        self.abs_sum = np.zeros(out_len, dtype=np.float64)
        self.count = np.zeros(out_len, dtype=np.int64)
        self.sample_rows = []
        self.delta_sq_sum = 0.0
        self.delta_abs_pred_sum = 0.0
        self.delta_abs_true_sum = 0.0
        self.delta_count = 0
        self.dx_sum = self.dy_sum = self.dxx_sum = self.dyy_sum = self.dxy_sum = 0.0

    def update(self, pred, truth, last_input, mask, sample_offset):
        # Arrays are B,T,C,H,W in original physical units.
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
                self.sample_rows.append((sample_offset + b, t + 1, mse, mae, float(np.sqrt(mse))))

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
        overall_mse = float(self.sq_sum.sum() / max(self.count.sum(), 1))
        n = max(self.delta_count, 1)
        cov = self.dxy_sum - self.dx_sum * self.dy_sum / n
        var_x = self.dxx_sum - self.dx_sum ** 2 / n
        var_y = self.dyy_sum - self.dy_sum ** 2 / n
        corr = cov / np.sqrt(max(var_x * var_y, 1e-30))
        return {
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse),
            'overall_mse': overall_mse,
            'overall_mae': float(self.abs_sum.sum() / max(self.count.sum(), 1)),
            'overall_rmse': float(np.sqrt(overall_mse)),
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
        ('Overall MSE', metrics['overall_mse']), ('Overall MAE', metrics['overall_mae']), ('Overall RMSE', metrics['overall_rmse']),
        ('Delta RMSE', metrics['delta_rmse']), ('Predicted mean |delta|', metrics['pred_mean_abs_change']),
        ('True mean |delta|', metrics['true_mean_abs_change']), ('Delta correlation', metrics['delta_correlation']),
    ]
    for key in ('model_name', 'downsample', 'grid_shape', 'in_len', 'out_len', 'stride', 'train_end_idx', 'epochs', 'learning_rate', 'dynamic_weight'):
        rows.append((key, checkpoint.get(key)))
    for key, value in checkpoint.get('model_kwargs', {}).items():
        rows.append((key, value))
    for row in rows:
        ws.append([_excel_value(v) for v in row])
    ws.column_dimensions['A'].width = 30; ws.column_dimensions['B'].width = 36

    wh = wb.create_sheet('per_horizon')
    wh.append(['Horizon', 'MSE', 'MAE', 'RMSE'])
    for cell in wh[1]:
        cell.font = Font(bold=True); cell.fill = fill
    for t in range(len(metrics['mse'])):
        wh.append([t + 1, float(metrics['mse'][t]), float(metrics['mae'][t]), float(metrics['rmse'][t])])

    wsamp = wb.create_sheet('per_sample')
    wsamp.append(['Sample', 'Horizon', 'MSE', 'MAE', 'RMSE'])
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
    payload = {}
    for key, value in records.items():
        payload[key] = np.asarray(value)
    for key, value in metadata.items():
        payload[f'meta_{key}'] = np.asarray(value)
    np.savez_compressed(path, **payload)
