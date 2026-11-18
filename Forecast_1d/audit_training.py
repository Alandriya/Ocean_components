"""Inspect saved Forecast_1d training traces without third-party dependencies.
Reads only; writes a JSON report at the explicitly requested output path.
"""
import argparse
import json
import math
from pathlib import Path


def finite(value):
    return isinstance(value, (float, int)) and math.isfinite(value)


def audit(root):
    records = []
    warnings = []
    for meta_path in sorted(root.glob('*/P*/train_metadata.json')):
        path = meta_path.parent
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            history = json.loads((path / 'history.json').read_text(encoding='utf-8'))
            if not isinstance(history, list) or not history:
                raise ValueError('Empty or unsupported history')
            valid = [r for r in history if finite(r.get('val_loss')) and finite(r.get('train_loss'))]
            if not valid:
                raise ValueError('No finite train/validation pairs')
            best = min(valid, key=lambda r: r['val_loss'])
            saved = next((r for r in valid if r['epoch'] == meta.get('best_epoch')), None)
            first, last = valid[0], valid[-1]
            records.append({
                'model': meta.get('model', path.parent.name),
                'point_id': meta.get('point_id', path.name),
                'input_length': meta.get('input_length'),
                'horizon': meta.get('horizon'),
                'dataset_fingerprint_sha256': meta.get('dataset_fingerprint_sha256'),
                'seed': meta.get('seed'), 'parameter_count': meta.get('parameter_count'),
                'model_config': meta.get('model_config'),
                'epochs_logged': len(history), 'finite_epochs': len(valid),
                'first_train_loss': first['train_loss'], 'first_val_loss': first['val_loss'],
                'last_epoch': last['epoch'], 'last_train_loss': last['train_loss'],
                'last_val_loss': last['val_loss'], 'best_observed_epoch': best['epoch'],
                'best_observed_val_loss': best['val_loss'],
                'saved_best_epoch': meta.get('best_epoch'),
                'saved_best_validation_loss_normalized': meta.get('best_validation_loss_normalized'),
                'train_loss_at_saved_best': saved['train_loss'] if saved else None,
                'val_loss_at_saved_best': saved['val_loss'] if saved else None,
                'last_minus_best_val_loss': last['val_loss'] - best['val_loss'],
                'epochs_after_saved_best': last['epoch'] - saved['epoch'] if saved else None,
                'history': history,
            })
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            warnings.append({'path': str(path), 'error': str(exc)})
    return {'results_dir': str(root), 'record_count': len(records), 'warnings': warnings,
            'note': 'Losses are normalized. Best observed epoch may differ from saved epoch due to min_delta. Stop reason cannot be inferred reliably from history alone.',
            'records': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True, help='Existing h14 result directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.results_dir.resolve()
    if not root.is_dir():
        parser.error(f'Results directory not found: {root}')
    report = audit(root)
    if not report['record_count']:
        parser.error(f'No readable point-specific training histories: {report["warnings"]}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Saved {report["record_count"]} training traces to {args.output}; warnings={len(report["warnings"])}')


if __name__ == '__main__':
    main()
