"""No-Excel export and aggregation tests (only standard library + numpy)."""
import csv
import json
import numpy as np

from Forecast_nn.evaluation import save_metrics_json
from Forecast_nn.aggregate_metrics import collect, main as aggregate_main


def test_save_metrics_and_aggregate(tmp_path):
    root = tmp_path / 'Results'
    out = root / 'swin_gru' / 'joint' / 'ds2_i30_o3_b64_tt8_ad2_h8_ws4' / 'test'
    metrics = {
        'overall_mse': 2.0, 'overall_mae': 1.0, 'overall_rmse': np.sqrt(2.0),
        'overall_ssim': 0.7, 'overall_psnr': 20.0,
        'delta_rmse': 1.2, 'delta_correlation': 0.3,
        'pred_mean_abs_change': 3.0, 'true_mean_abs_change': 4.0,
        'mse': np.array([4.0, 2.0]), 'mae': np.array([2.0, 1.0]),
        'rmse': np.array([2.0, np.sqrt(2.)]), 'ssim': np.array([0.8, 0.6]),
        'psnr': np.array([20.0, 21.0]),
        'psnr_ranges': np.array([20.0, 50.0]),
        'psnr_per_channel': np.array([21.0, 22.0]),
        'psnr_per_horizon_channel': np.array([[20., 21.], [22., 23.]]),
    }
    checkpoint = {
        'model_name': 'swin_gru', 'model_kwargs': {'base_channels': 64},
        'best_epoch': 15, 'mean': np.array([0.5, 0.1]),
    }
    path = out / 'metrics_swin_gru_joint.json'
    save_metrics_json(path, 'swin_gru', 'joint', 'ds2_i30_o3_b64_tt8_ad2_h8_ws4',
                      metrics, checkpoint,
                      [(0, 1, 4., 2., 2., .8, 20.), (0, 2, 2., 1., 1.414, .6, float('inf'))])
    report = json.loads(path.read_text(encoding='utf-8'), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    assert report['summary']['overall_psnr'] == 20.0
    assert report['per_horizon'][0]['psnr_db'] == 20.0
    assert len(report['per_channel']) == 2
    assert report['checkpoint']['model_kwargs']['base_channels'] == 64
    assert 'mean' not in report['checkpoint']
    samples = path.with_name(report['per_sample_csv'])
    with samples.open(newline='', encoding='utf-8') as stream:
        entries = list(csv.DictReader(stream))
    assert len(entries) == 2 and entries[1]['psnr_db'] == ''

    files, summary, horizon, failures = collect(root)
    assert len(files) == 1 and not failures
    assert summary[0]['model_name'] == 'swin_gru'
    assert summary[0]['Overall PSNR (dB)'] == 20.0
    assert len(horizon) == 2
    aggregate_main(['--root', str(root)])
    folder = root / '_summary'
    combined = json.loads((folder / 'all_models_metrics.json').read_text(encoding='utf-8'))
    assert len(combined['summary']) == 1
    assert len(combined['per_horizon']) == 2
    with (folder / 'all_models_metrics_summary.csv').open(newline='', encoding='utf-8') as stream:
        sr = list(csv.DictReader(stream))
    assert sr[0]['Overall PSNR (dB)'] == '20.0'
    assert (folder / 'all_models_metrics_read_errors.csv').is_file()
