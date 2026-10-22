"""Offline synthetic integration test for train/val/test and PSNR, no ERA5 files."""
import sys
import json
import tempfile
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import torch

from Forecast_nn.config import cfg
from Forecast_nn.splits import resolve_splits
from Forecast_nn.data import OceanSequenceDataset, load_or_create_psnr_ranges
from Forecast_nn.evaluation import (OnlineErrorMetrics, MetricAccumulator, psnr_from_mse,
                                    build_run_tag, append_sde_run_tag, add_run_id)


def test_splits_and_metrics():
    s = resolve_splits(16436, '1979-01-01', '2023-12-31', 730, 365, 30, 3)
    assert (s.train_end, s.test_start, s.data_end) == (15341, 16071, 16436)
    assert s.dataset_bounds('val', 30) == (15311, 16071)
    assert s.dataset_bounds('test', 30) == (16041, 16436)
    assert s.dataset_bounds('train', 30) == (0, 15341)
    assert np.isclose(psnr_from_mse(4.0, 20.0), 20.0)
    mask = np.array([[True, False], [True, True]])
    p = np.array([[[[[3., 300.], [3., 3.]]]]])
    y = np.array([[[[[1., -999.], [1., 1.]]]]])
    acc = MetricAccumulator(1, psnr_ranges=[20.])
    acc.update(p, y, y[:, 0], mask, 0)
    assert np.isclose(acc.summary()['overall_psnr'], 20.0)
    assert np.isclose(acc.summary()['overall_mse'], 4.0)
    assert np.isclose(acc.summary()['psnr'][0], 20.0)
    assert np.isclose(acc.sample_rows[0][-1], 20.0)
    am = OnlineErrorMetrics(1, [2.], [20.])
    am.update(torch.tensor(p / 2, dtype=torch.float32), torch.tensor(y / 2, dtype=torch.float32), torch.tensor(mask[None, None]))
    assert np.isclose(am.summary()['overall_psnr'], 20.0)


def test_compact_run_names():
    base = dict(base_channels=16, depth=3, dropout=0.05, residual=True,
                use_mask_channel=True, use_coord_channels=True, mask_features=True,
                lstm_kernel=3)
    tag = build_run_tag('unet_convlstm', 'sensible', 2, 30, 3, 1, base)
    assert tag == 'ds2_i30_o3_b16_d3'
    assert 'val' not in tag and 'test' not in tag and 'end' not in tag
    assert build_run_tag('unet_convlstm', 'sensible', 2, 30, 3, 1,
                         dict(base, dropout=0.1, lstm_kernel=5)).endswith('_p0.1_k5')
    assert build_run_tag('simvp', 'latent', 2, 30, 3, 1,
                         {'base_channels': 16, 'depth': 3, 'temporal_blocks': 4}) == 'ds2_i30_o3_b16_d3_tb4'
    assert build_run_tag('fno', 'latent', 2, 30, 3, 1,
                         {'width': 48, 'modes_y': 14, 'modes_x': 14, 'layers': 3,
                          'dropout': 0., 'projection_width': 96, 'padding': 2}) == 'ds2_i30_o3_w48_my14_mx14_l3'
    assert append_sde_run_tag('abc', 1e-4, 'diffusion', 'first_step') == 'abc_sw1e-4'
    assert append_sde_run_tag('abc', 0, 'diffusion', 'first_step') == 'abc'
    assert append_sde_run_tag('abc', 1e-4, 'variance', 'last_observed_all').endswith('_bv_ma')
    assert add_run_id(tag, 'seed-2') == tag + '_seed-2'
    try:
        add_run_id(tag, '../oops')
    except ValueError:
        pass
    else:
        raise AssertionError('Unsafe run-id was accepted.')


def fixture(root):
    root = Path(root)
    n = 75
    h, w = 21, 23
    mask = np.ones((161, 181), dtype=np.bool_)
    mask[:, :8] = False
    (root / 'mask').write_bytes(mask.tobytes())
    rng = np.random.default_rng(2026)
    flux = (np.sin(np.arange(n)[:, None, None] * 0.1) * 10 +
            rng.standard_normal((n, h, w)) * 0.7 +
            np.arange(w)[None, None, :] / 8).astype('float32')
    np.save(root / 'flux.npy', flux)
    a = np.ones((n - 1, h, w), dtype='float32') * 0.12
    b = np.ones((n - 1, h, w), dtype='float32') * 0.3
    np.save(root / 'a.npy', a); np.save(root / 'b.npy', b)
    cfg.data_files = {'sensible': root / 'flux.npy', 'latent': root / 'flux.npy'}
    cfg.mask_file = root / 'mask'
    cfg.stats_root = root / 'stats'
    cfg.results_root = root / 'results'
    cfg.base_date = '1979-01-01'
    cfg.data_last_date = str(date(1979, 1, 1) + timedelta(days=n - 1))
    cfg.in_len, cfg.out_len, cfg.stride = 4, 3, 1
    cfg.val_steps, cfg.test_steps = 10, 10
    cfg.psnr_max_sample_frames = 20
    cfg.model_params['unet']['base_channels'] = 4
    cfg.model_params['unet']['depth'] = 2
    cfg.model_params['unet']['dropout'] = 0.0
    cfg.model_params['simvp_sde']['base_channels'] = 4
    cfg.model_params['simvp_sde']['depth'] = 2
    cfg.model_params['simvp_sde']['temporal_blocks'] = 1
    cfg.model_params['simvp_sde']['dropout'] = 0.0
    cfg.sde_sources = {'sensible': {'a_file': root / 'a.npy', 'b_file': root / 'b.npy', 'time_offset': 1},
                       'latent': {'a_file': root / 'a.npy', 'b_file': root / 'b.npy', 'time_offset': 1}}
    cfg.log_every=1000
    cfg.checkpoint_every=1
    return n


def run(model='unet'):
    from Forecast_nn import train as trainmod
    from Forecast_nn import test as testmod
    sys.argv = ['train', '--model', model, '--variable', 'sensible', '--end-date', cfg.data_last_date,
                '--epochs', '1', '--batch-size', '8', '--num-workers', '0', '--downsample', '8', '--patience', '2', '--no-amp']
    trainmod.main()
    sys.argv = ['test', '--model', model, '--variable', 'sensible', '--end-date', cfg.data_last_date,
                '--batch-size', '8', '--num-workers', '0', '--downsample', '8', '--no-amp', '--plot-samples', '0']
    testmod.main()
    # Check the validation period is also accessible from the same best checkpoint.
    sys.argv += ['--split', 'val']
    testmod.main()


def test_end_to_end_train_eval():
    with tempfile.TemporaryDirectory() as temp:
        fixture(temp)
        run('unet')
        # SDE feature alignment and causal loss stay compatible.
        run('simvp_sde')
        files = list((Path(temp) / 'results').rglob('best.pth'))
        assert len(files) == 2
        from Forecast_nn.runtime import load_checkpoint
        for f in files:
            c = load_checkpoint(f)
            assert c['epoch'] == 1 and c['best_epoch'] == 1
            assert c['data_end_idx'] == 75
            assert c['val_start_idx'] == 55 and c['test_start_idx'] == 65
            assert len(c['psnr_ranges']) == 1
            assert len(c['history']['val_psnr_horizons']) == 1
        outputs = list((Path(temp) / 'results').rglob('metrics_*.json'))
        assert len(outputs) == 12
        for f in outputs:
            report = json.loads(f.read_text(encoding='utf-8'))
            assert report['schema_version'] == 1
            assert 'overall_psnr' in report['summary']
            assert 'psnr_db' in report['per_horizon'][0]
            assert f.with_name(report['per_sample_csv']).is_file()



if __name__ == '__main__':
    test_splits_and_metrics()
    test_end_to_end_train_eval()
    print('ALL SYNTHETIC TESTS PASSED')
