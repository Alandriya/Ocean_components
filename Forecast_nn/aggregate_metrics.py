"""Collect all Forecast_nn metrics_*.xlsx files into one comparison workbook.

Usage from repository root:
    python -m Forecast_nn.aggregate_metrics

Optional:
    python -m Forecast_nn.aggregate_metrics --root "D:/.../Forecast/Results" --output "D:/.../all_models.xlsx"

The script is backward-compatible with older metric files that do not yet contain SSIM/PSNR.
"""
import argparse
from pathlib import Path

from Forecast_nn.config import cfg


def _read_summary(ws):
    result = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or row[0] is None:
            continue
        result[str(row[0])] = row[1] if len(row) > 1 else None
    return result


def _read_per_horizon(ws):
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return []
    headers = [str(v) if v is not None else '' for v in rows[0]]
    result = []
    for values in rows[1:]:
        if not values or values[0] is None:
            continue
        result.append({headers[i]: values[i] for i in range(min(len(headers), len(values)))})
    return result


def _infer_model(path, root):
    try:
        rel = path.relative_to(root)
        return rel.parts[0] if rel.parts else ''
    except Exception:
        return ''


def _preferred_columns(all_keys):
    preferred = [
        'Variable', 'model_name', 'Method', 'Run tag',
        'Overall RMSE', 'Overall MAE', 'Overall MSE', 'Overall SSIM', 'Overall PSNR (dB)',
        'Delta RMSE', 'Delta correlation', 'Predicted mean |delta|', 'True mean |delta|',
        'downsample', 'grid_shape', 'in_len', 'out_len', 'stride', 'epochs',
        'learning_rate', 'dynamic_weight', 'sde_enabled', 'sde_weight', 'sde_b_kind', 'sde_dt',
    ]
    present = [k for k in preferred if k in all_keys]
    extra = sorted(k for k in all_keys if k not in set(present) and k != 'Source file' and not k.startswith('_'))
    return present + extra + ['Source file']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default=str(cfg.results_root), help='Root directory containing model result folders.')
    p.add_argument('--output', default=None, help='Output XLSX. Default: <root>/_summary/all_models_metrics.xlsx')
    args = p.parse_args()

    try:
        from openpyxl import Workbook, load_workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError as exc:
        raise ImportError('This script requires openpyxl (already used by Forecast_nn.evaluation).') from exc

    root = Path(args.root)
    output = Path(args.output) if args.output else root / '_summary' / 'all_models_metrics.xlsx'
    files = sorted(p for p in root.rglob('metrics_*.xlsx') if p.resolve() != output.resolve())
    if not files:
        raise FileNotFoundError(f'No metrics_*.xlsx files found below {root}')

    summary_rows = []
    horizon_rows = []
    failed = []
    for path in files:
        try:
            wb = load_workbook(path, read_only=True, data_only=True)
            if 'summary' not in wb.sheetnames:
                raise ValueError('missing summary sheet')
            summary = _read_summary(wb['summary'])
            summary.setdefault('model_name', _infer_model(path, root))
            summary['Source file'] = str(path)
            summary_rows.append(summary)

            if 'per_horizon' in wb.sheetnames:
                for row in _read_per_horizon(wb['per_horizon']):
                    item = {
                        'Variable': summary.get('Variable'),
                        'model_name': summary.get('model_name'),
                        'Method': summary.get('Method'),
                        'Run tag': summary.get('Run tag'),
                        **row,
                        'Source file': str(path),
                    }
                    horizon_rows.append(item)
            wb.close()
        except Exception as exc:
            failed.append((str(path), str(exc)))

    if not summary_rows:
        raise RuntimeError(f'No readable metrics workbooks. Failures: {failed[:5]}')

    all_keys = set().union(*(row.keys() for row in summary_rows))
    columns = _preferred_columns(all_keys)

    # Sort: variable first, then neural/baseline method and increasing RMSE.
    def sort_key(row):
        rmse = row.get('Overall RMSE')
        try:
            rmse = float(rmse)
        except Exception:
            rmse = float('inf')
        return (str(row.get('Variable', '')), rmse, str(row.get('model_name', '')), str(row.get('Method', '')))

    summary_rows.sort(key=sort_key)
    horizon_rows.sort(key=lambda r: (
        str(r.get('Variable', '')), str(r.get('model_name', '')), str(r.get('Method', '')),
        int(r.get('Horizon') or 0),
    ))

    out = Workbook()
    ws = out.active
    ws.title = 'summary'
    wh = out.create_sheet('per_horizon')
    we = out.create_sheet('read_errors')

    header_fill = PatternFill('solid', fgColor='D9EAF7')
    neural_fill = PatternFill('solid', fgColor='E2F0D9')

    ws.append(columns)
    for row in summary_rows:
        ws.append([row.get(c) for c in columns])
    for cell in ws[1]:
        cell.font = Font(bold=True); cell.fill = header_fill
    for r in range(2, ws.max_row + 1):
        method = ws.cell(r, columns.index('Method') + 1).value if 'Method' in columns else None
        model = ws.cell(r, columns.index('model_name') + 1).value if 'model_name' in columns else None
        if method and model and str(method) == str(model):
            for cell in ws[r]:
                cell.fill = neural_fill

    hcols = ['Variable', 'model_name', 'Method', 'Run tag', 'Horizon', 'MSE', 'MAE', 'RMSE', 'SSIM', 'PSNR (dB)', 'Source file']
    wh.append(hcols)
    for row in horizon_rows:
        wh.append([row.get(c) for c in hcols])
    for cell in wh[1]:
        cell.font = Font(bold=True); cell.fill = header_fill

    we.append(['Source file', 'Error'])
    for item in failed:
        we.append(list(item))
    for cell in we[1]:
        cell.font = Font(bold=True); cell.fill = header_fill

    for sheet in (ws, wh, we):
        sheet.freeze_panes = 'A2'
        sheet.auto_filter.ref = sheet.dimensions
        for row in sheet.iter_rows():
            for cell in row:
                cell.alignment = Alignment(vertical='top')
        for col_idx in range(1, sheet.max_column + 1):
            max_len = 0
            for cell in sheet.iter_cols(min_col=col_idx, max_col=col_idx, max_row=min(sheet.max_row, 300)):
                for c in cell:
                    max_len = max(max_len, len(str(c.value)) if c.value is not None else 0)
            sheet.column_dimensions[get_column_letter(col_idx)].width = min(max(max_len + 2, 10), 42)

    output.parent.mkdir(parents=True, exist_ok=True)
    out.save(output)
    print(f'Found metric files: {len(files)}')
    print(f'Summary rows: {len(summary_rows)}')
    print(f'Per-horizon rows: {len(horizon_rows)}')
    print(f'Read errors: {len(failed)}')
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
