"""Plot saved forecast examples in the three-row format used by the reference figure.

Rows are: real values, forecast, absolute difference. Columns are forecast horizons. The same value scale is
used for truth and forecast, land is shown in light green, and every panel has its own colorbar. When --method all
is used, one figure per forecast method is saved with common color limits so the methods are visually comparable.
"""
import argparse
import ast
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from Forecast_nn.config import cfg
from Forecast_nn.sde_energy_dataloader import load_full_mask

METHODS = ('sde_energy', 'historical_mean', 'persistence')
METHOD_LABELS = {
    'ru': {'sde_energy': 'SDE-Energy', 'historical_mean': 'Среднее по истории', 'persistence': 'Копирование последнего дня'},
    'en': {'sde_energy': 'SDE-Energy', 'historical_mean': 'Historical mean', 'persistence': 'Persistence'},
}
ROW_LABELS = {
    'ru': ('Реальные значения', 'Прогноз', 'Абсолютная разность'),
    'en': ('Real values', 'Forecast', 'Absolute difference'),
}
VARIABLE_LABELS = {
    'ru': {'latent': 'скрытого потока', 'sensible': 'явного потока'},
    'en': {'latent': 'latent heat flux', 'sensible': 'sensible heat flux'},
}


def _scalar(data, key, default=None):
    name = f'meta_{key}'
    if name in data.files:
        value = data[name]
        return value.item() if value.ndim == 0 else value.flat[0].item()
    # Backward compatibility with v4 plot files.
    if 'metadata' in data.files:
        try:
            meta = ast.literal_eval(str(data['metadata'].flat[0]))
            return meta.get(key, default)
        except (ValueError, SyntaxError, AttributeError):
            pass
    return default


def _date_labels(data, sample_pos, out_len, base_date_override=None):
    base_date_text = base_date_override or _scalar(data, 'base_date', '1979-01-01')
    base_date = datetime.strptime(str(base_date_text), '%Y-%m-%d')
    stride = int(_scalar(data, 'stride', 1))
    if 'forecast_start_index' not in data.files:
        return None
    start_index = int(data['forecast_start_index'][sample_pos])
    return [base_date + timedelta(days=start_index + t * stride) for t in range(out_len)]


def _masked_values(array, mask):
    # array: T,H,W
    return array[:, mask].reshape(-1)


def _shared_limits(truth, forecasts, mask, vmin=None, vmax=None, error_max=None):
    values = [_masked_values(truth, mask)] + [_masked_values(v, mask) for v in forecasts]
    all_values = np.concatenate(values)
    all_values = all_values[np.isfinite(all_values)]
    lo = float(np.min(all_values)) if vmin is None else float(vmin)
    hi = float(np.max(all_values)) if vmax is None else float(vmax)
    if lo == hi:
        hi = lo + 1.0
    errors = [np.abs(v - truth)[:, mask].reshape(-1) for v in forecasts]
    error_values = np.concatenate(errors)
    error_values = error_values[np.isfinite(error_values)]
    err_hi = float(np.max(error_values)) if error_max is None else float(error_max)
    if err_hi <= 0:
        err_hi = 1.0
    return lo, hi, err_hi


def _value_norm(vmin, vmax):
    return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax) if vmin < 0.0 < vmax else Normalize(vmin=vmin, vmax=vmax)


def _title(variable, method, dates, language, show_method=True):
    method_label = METHOD_LABELS[language][method]
    variable_label = VARIABLE_LABELS[language].get(variable, variable)
    if language == 'ru':
        first = dates[0].strftime('%d.%m.%Y') if dates else '?'
        last = dates[-1].strftime('%d.%m.%Y') if dates else '?'
        method_text = f' ({method_label})' if show_method else ''
        return f'Прогноз {variable_label}{method_text}\nв интервале {first} - {last}'
    first = dates[0].strftime('%d.%m.%Y') if dates else '?'
    last = dates[-1].strftime('%d.%m.%Y') if dates else '?'
    method_text = f' ({method_label})' if show_method else ''
    return f'{variable_label.capitalize()} forecast{method_text}\n{first} - {last}'


def draw_reference_style(truth, forecast, mask, title, output, value_limits, error_max, dpi=180):
    out_len, height, width = truth.shape
    vmin, vmax = value_limits
    value_cmap = plt.get_cmap('seismic').copy(); value_cmap.set_bad('#90EE90')
    error_cmap = plt.get_cmap('Reds').copy(); error_cmap.set_bad('#90EE90')
    value_norm = _value_norm(vmin, vmax)
    error_norm = Normalize(vmin=0.0, vmax=error_max)
    error = np.abs(forecast - truth)

    # The reference uses a very wide canvas with one colorbar per panel.
    fig, axes = plt.subplots(3, out_len, figsize=(4.15 * out_len, 10.7), squeeze=False)
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.055, top=0.88, wspace=0.25, hspace=0.24)
    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.972)

    row_names = ROW_LABELS['ru'] if title.startswith('Прогноз') else ROW_LABELS['en']
    rows = (truth, forecast, error)
    for r, values in enumerate(rows):
        center = out_len // 2
        axes[r, center].set_title(row_names[r], fontsize=13, fontweight='bold', pad=11)
        for t in range(out_len):
            field = np.ma.array(values[t], mask=~mask)
            cmap, norm = (value_cmap, value_norm) if r < 2 else (error_cmap, error_norm)
            im = axes[r, t].imshow(field, origin='upper', cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
            axes[r, t].set_xlim(-0.5, width - 0.5); axes[r, t].set_ylim(height - 0.5, -0.5)
            axes[r, t].set_xticks(np.arange(0, width, 50)); axes[r, t].set_yticks(np.arange(0, height, 20))
            axes[r, t].tick_params(labelsize=8)
            cb = fig.colorbar(im, ax=axes[r, t], fraction=0.046, pad=0.055)
            cb.ax.tick_params(labelsize=8)

    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='forecast_samples_*.npz produced by nn_test_sde_energy.py')
    parser.add_argument('--sample-index', type=int, default=0, help='Position inside the saved sample list, not the dataset ID.')
    parser.add_argument('--channel', type=int, default=0)
    parser.add_argument('--method', choices=METHODS + ('all',), default='all')
    parser.add_argument('--language', choices=('ru', 'en'), default='ru')
    parser.add_argument('--show-method-in-title', action=argparse.BooleanOptionalAction, default=True, help='Use --no-show-method-in-title for a title matching the reference image even more closely.')
    parser.add_argument('--mask-file', default=os.path.join(cfg.root_path, 'DATA', 'mask'))
    parser.add_argument('--base-date', default=None, help='Override data epoch, YYYY-MM-DD. Default comes from the test file.')
    parser.add_argument('--vmin', type=float, default=None)
    parser.add_argument('--vmax', type=float, default=None)
    parser.add_argument('--error-max', type=float, default=None)
    parser.add_argument('--dpi', type=int, default=180)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--output', default=None, help='Exact output filename; only valid when one method is selected.')
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    downsample = int(_scalar(data, 'downsample', 1))
    mask = load_full_mask(mask_file=args.mask_file, downsample=downsample)
    i, c = args.sample_index, args.channel
    truth = np.asarray(data['truth'][i, :, c], dtype=np.float32)
    methods = METHODS if args.method == 'all' else (args.method,)
    forecasts = {method: np.asarray(data[method][i, :, c], dtype=np.float32) for method in methods}
    variable = str(_scalar(data, 'variable', 'latent'))
    dates = _date_labels(data, i, truth.shape[0], args.base_date)

    # For --method all, scales are shared across all three methods. This makes visual comparison meaningful.
    vmin, vmax, error_max = _shared_limits(truth, list(forecasts.values()), mask, args.vmin, args.vmax, args.error_max)
    sample_id = int(data['sample_id'][i]) if 'sample_id' in data.files else i
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.input).parent

    print(f'Input: {args.input}')
    print(f'Sample: saved position={i}, dataset sample id={sample_id}, variable={variable}, horizons={truth.shape[0]}, downsample={downsample}, grid={mask.shape[0]}x{mask.shape[1]}')
    print(f'Common color limits: values=[{vmin:.6g}, {vmax:.6g}], absolute error=[0, {error_max:.6g}]')
    for method in methods:
        if args.output and len(methods) == 1:
            output = args.output
        else:
            output = output_dir / f'forecast_comparison_{method}_{variable}_sample{sample_id}.png'
        title = _title(variable, method, dates, args.language, args.show_method_in_title)
        draw_reference_style(truth, forecasts[method], mask, title, str(output), (vmin, vmax), error_max, args.dpi)
        print(f'Saved: {output}')


if __name__ == '__main__':
    main()
