"""Aggregate JSON metric reports from Forecast_nn, without openpyxl.

Run from project root:
    python -m Forecast_nn.aggregate_metrics
    python -m Forecast_nn.aggregate_metrics --root /path/to/Results

Produces _summary/all_models_metrics.json and three sibling CSVs.
Legacy .xlsx reports are not parsed; re-run test.py for new JSON reports.
"""
import argparse
import json
from pathlib import Path

from Forecast_nn.config import cfg
from Forecast_nn.evaluation import to_jsonable, write_csv


def _flatten(report, path):
    meta = report.get('checkpoint', {})
    row = {
        'Variable': report.get('variable'),
        'model_name': meta.get('model_name') or path.parents[3].name,
        'Method': report.get('method'),
        'Run tag': report.get('run_tag'),
        'Source file': str(path),
    }
    metric_names = {
        'overall_rmse': 'Overall RMSE',
        'overall_mae': 'Overall MAE',
        'overall_mse': 'Overall MSE',
        'overall_ssim': 'Overall SSIM',
        'overall_psnr': 'Overall PSNR (dB)',
        'delta_rmse': 'Delta RMSE',
        'delta_correlation': 'Delta correlation',
        'pred_mean_abs_change': 'Predicted mean |delta|',
        'true_mean_abs_change': 'True mean |delta|',
    }
    for key, alias in metric_names.items():
        row[alias] = report.get('summary', {}).get(key)
    for key, value in meta.items():
        if key != 'model_kwargs':
            row[key] = value
    for key, value in meta.get('model_kwargs', {}).items():
        row[f'model_{key}'] = value
    row['psnr_ranges'] = report.get('psnr_ranges')
    return row


def _sort_rmse(row):
    value = row.get('Overall RMSE')
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('inf')


def collect(root):
    root = Path(root)
    files = sorted(root.rglob('metrics_*.json'))
    summary, horizons, errors = [], [], []
    for path in files:
        try:
            with path.open(encoding='utf-8') as stream:
                report = json.load(stream)
            if report.get('schema_version') != 1 or not isinstance(report.get('summary'), dict):
                raise ValueError('unsupported or incomplete JSON metrics format')
            row = _flatten(report, path)
            summary.append(row)
            for h in report.get('per_horizon', []):
                horizons.append({
                    'Variable': row['Variable'],
                    'model_name': row['model_name'],
                    'Method': row['Method'],
                    'Run tag': row['Run tag'],
                    **h,
                    'Source file': str(path),
                })
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            errors.append({'Source file': str(path), 'Error': str(exc)})
    summary.sort(key=lambda r: (str(r.get('Variable')), _sort_rmse(r), str(r.get('Method'))))
    horizons.sort(key=lambda r: (str(r.get('Variable')), str(r.get('model_name')),
                                 str(r.get('Method')), int(r.get('horizon', 0))))
    return files, summary, horizons, errors


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(cfg.results_root),
                        help='Root of results tree with metrics_*.json files.')
    parser.add_argument('--output', default=None,
                        help='Output JSON; default: <root>/_summary/all_models_metrics.json.')
    args = parser.parse_args(argv)
    root = Path(args.root)
    output = (Path(args.output) if args.output else
              root / '_summary' / 'all_models_metrics.json')
    if output.suffix.lower() != '.json':
        parser.error('--output must end in .json')
    files, summary, horizons, errors = collect(root)
    if not files:
        raise FileNotFoundError(f'No metrics_*.json files found below {root}. '
                                'Older XLSX reports require re-running test.py.')
    if not summary:
        raise ValueError('Found JSON files but none could be read successfully.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as stream:
        json.dump(to_jsonable({'schema_version': 1, 'summary': summary,
                               'per_horizon': horizons, 'read_errors': errors}),
                  stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')

    headers = [
        'Variable', 'model_name', 'Method', 'Run tag', 'Overall RMSE',
        'Overall MAE', 'Overall MSE', 'Overall SSIM', 'Overall PSNR (dB)',
        'Delta RMSE', 'Delta correlation', 'Predicted mean |delta|',
        'True mean |delta|',
    ]
    remaining = sorted(set().union(*(set(row) for row in summary)) - set(headers) - {'Source file'})
    headers += remaining + ['Source file']
    hheaders = ['Variable', 'model_name', 'Method', 'Run tag',
                'horizon', 'mse', 'mae', 'rmse', 'ssim', 'psnr_db', 'Source file']
    paths = [
        write_csv(output.with_name(f'{output.stem}_summary.csv'), headers,
                  ([row.get(k) for k in headers] for row in summary)),
        write_csv(output.with_name(f'{output.stem}_per_horizon.csv'), hheaders,
                  ([row.get(k) for k in hheaders] for row in horizons)),
        write_csv(output.with_name(f'{output.stem}_read_errors.csv'),
                  ['Source file', 'Error'],
                  ([r['Source file'], r['Error']] for r in errors)),
    ]
    print(f'Found: {len(files)} JSON files | Models/baselines: {len(summary)} | '
          f'Rows by horizon: {len(horizons)} | Errors: {len(errors)}')
    for path in [output, *paths]:
        print(f'Saved: {path}')


if __name__ == '__main__':
    main()
