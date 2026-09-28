"""Draw truth / forecast / absolute-error maps from Forecast_nn.test outputs."""
import argparse
import ast
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm

from Forecast_nn.data import load_mask


def _scalar(data, key, default=None):
    name = f'meta_{key}'
    if name not in data.files:
        return default
    value = data[name]
    if value.ndim == 0:
        return value.item()
    return value.tolist()


def _methods(data):
    value = _scalar(data, 'methods', None)
    if value is not None:
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except Exception:
                value = [value]
        return list(value)
    excluded = {'sample_id', 'forecast_start_index', 'input_last', 'truth'}
    return [key for key in data.files if not key.startswith('meta_') and key not in excluded]


def _value_norm(vmin, vmax):
    return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax) if vmin < 0 < vmax else Normalize(vmin=vmin, vmax=vmax)


def _date_labels(data, sample_pos, out_len):
    base = datetime.strptime(str(_scalar(data, 'base_date', '1979-01-01')), '%Y-%m-%d')
    stride = int(_scalar(data, 'stride', 1))
    start = int(data['forecast_start_index'][sample_pos])
    return [base + timedelta(days=start + k * stride) for k in range(out_len)]


def draw(truth, forecast, mask, title, output, vmin, vmax, error_max, dates=None, dpi=180):
    out_len = truth.shape[0]
    cmap_values = plt.get_cmap('seismic').copy(); cmap_values.set_bad('#90EE90')
    cmap_error = plt.get_cmap('Reds').copy(); cmap_error.set_bad('#90EE90')
    value_norm = _value_norm(vmin, vmax)
    error_norm = Normalize(0, error_max)
    error = np.abs(forecast - truth)

    fig, axes = plt.subplots(3, out_len, figsize=(4.2 * out_len, 10.5), squeeze=False)
    fig.subplots_adjust(left=0.04, right=0.985, bottom=0.055, top=0.89, wspace=0.25, hspace=0.27)
    fig.suptitle(title, fontsize=19, fontweight='bold', y=0.975)
    row_names = ('Реальные значения', 'Прогноз', 'Абсолютная разность')
    rows = (truth, forecast, error)

    for r, values in enumerate(rows):
        axes[r, out_len // 2].set_title(row_names[r], fontsize=13, fontweight='bold', pad=10)
        for t in range(out_len):
            field = np.ma.array(values[t], mask=~mask)
            cmap, norm = (cmap_values, value_norm) if r < 2 else (cmap_error, error_norm)
            im = axes[r, t].imshow(field, cmap=cmap, norm=norm, origin='upper', aspect='auto', interpolation='nearest')
            if r == 0 and dates:
                axes[r, t].text(0.5, 1.02, dates[t].strftime('%d.%m.%Y'), transform=axes[r, t].transAxes,
                                ha='center', va='bottom', fontsize=9)
            axes[r, t].tick_params(labelsize=8)
            cb = fig.colorbar(im, ax=axes[r, t], fraction=0.046, pad=0.055)
            cb.ax.tick_params(labelsize=8)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--sample-index', type=int, default=0)
    p.add_argument('--channel', type=int, default=0)
    p.add_argument('--method', default='all')
    p.add_argument('--output-dir', default=None)
    p.add_argument('--vmin', type=float, default=None)
    p.add_argument('--vmax', type=float, default=None)
    p.add_argument('--error-max', type=float, default=None)
    p.add_argument('--dpi', type=int, default=180)
    args = p.parse_args()

    data = np.load(args.input, allow_pickle=True)
    methods = _methods(data)
    selected = methods if args.method == 'all' else [args.method]
    for method in selected:
        if method not in methods:
            raise KeyError(f'Method {method!r} not found. Available: {methods}')

    downsample = int(_scalar(data, 'downsample', 1))
    mask_file = str(_scalar(data, 'mask_file'))
    mask = load_mask(mask_file, downsample)
    i, c = args.sample_index, args.channel
    truth = np.asarray(data['truth'][i, :, c], dtype=np.float32)
    forecasts = {m: np.asarray(data[m][i, :, c], dtype=np.float32) for m in selected}

    all_values = [truth[:, mask].reshape(-1)] + [v[:, mask].reshape(-1) for v in forecasts.values()]
    values = np.concatenate(all_values); values = values[np.isfinite(values)]
    vmin = float(values.min()) if args.vmin is None else args.vmin
    vmax = float(values.max()) if args.vmax is None else args.vmax
    errors = np.concatenate([np.abs(v - truth)[:, mask].reshape(-1) for v in forecasts.values()])
    errors = errors[np.isfinite(errors)]
    error_max = float(errors.max()) if args.error_max is None else args.error_max
    if error_max <= 0: error_max = 1.0

    variable = str(_scalar(data, 'variable', 'variable'))
    dates = _date_labels(data, i, truth.shape[0])
    sample_id = int(data['sample_id'][i]) if 'sample_id' in data.files else i
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.input).parent

    for method in selected:
        title = f'{variable}: {method}\n{dates[0].strftime("%d.%m.%Y")} - {dates[-1].strftime("%d.%m.%Y")}'
        output = output_dir / f'forecast_comparison_{method}_{variable}_sample{sample_id}.png'
        draw(truth, forecasts[method], mask, title, output, vmin, vmax, error_max, dates, args.dpi)
        print(f'Saved: {output}')


if __name__ == '__main__':
    main()
