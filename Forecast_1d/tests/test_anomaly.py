"""End-to-end checks for train-only seasonal anomaly forecasting."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from Forecast_1d.data import PointDataset
from Forecast_1d.experiment import DEFAULT_TARGET_MODE, target_root
from Forecast_1d.models.baselines import calendar_index, fit_seasonal
from Forecast_1d import train, test, collect_results


class AnomalyForecastTests(unittest.TestCase):
    """Regression tests covering Anomaly Forecast."""
    def setUp(self):
        """Prepare the test fixtures before running the assertions."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.dataset_dir = self.base / 'dataset'
        self.dataset_dir.mkdir()
        self.result_dir = self.base / 'results'
        n_days = 220
        dates = np.datetime64('2000-01-01', 'D') + np.arange(n_days).astype('timedelta64[D]')
        rng = np.random.default_rng(1729)
        data = np.empty((n_days, 15, 2), dtype=np.float32)
        day = np.arange(n_days)
        for p in range(15):
            for c in range(2):
                data[:, p, c] = (15 * (c + 1) + 4 * p +
                                 (7 + p) * np.sin(2*np.pi*day/365.25 + c) +
                                 rng.normal(0, 1.5, n_days)).astype('float32')
        np.save(self.dataset_dir / 'centers.npy', data)
        np.save(self.dataset_dir / 'dates.npy', dates)
        points_path = Path(__file__).resolve().parents[1] / 'points.json'
        (self.dataset_dir / 'points.json').write_bytes(points_path.read_bytes())
        self.data = PointDataset(self.dataset_dir)

    def _train(self, model='mlp', monitor='normalized'):
        """Compute train for Anomaly Forecast Tests from ``model``, ``monitor``."""
        a = train.build_parser().parse_args([
            '--model', model, '--target-mode', 'anomaly', '--points', 'P3',
            '--dataset-dir', str(self.dataset_dir), '--results-dir', str(self.result_dir),
            '--val-days', '40', '--test-days', '40', '--in-len', '15', '--out-len', '14',
            '--epochs', '2', '--patience', '1', '--scheduler', 'none',
            '--batch-size', '32', '--device', 'cpu', '--monitor', monitor,
        ])
        train.run(a)
        return self.result_dir / 'h14' / 'anomaly' / model / 'P3'

    def _test(self, model='mlp'):
        """Compute test for Anomaly Forecast Tests from ``model``."""
        a = test.build_parser().parse_args([
            '--model', model, '--target-mode', 'anomaly', '--points', 'P3',
            '--dataset-dir', str(self.dataset_dir), '--results-dir', str(self.result_dir),
            '--val-days', '40', '--test-days', '40', '--in-len', '15', '--out-len', '14',
            '--split', 'test', '--batch-size', '128', '--device', 'cpu',
        ])
        return test.run(a)

    def test_only_train_values_contribute_to_climatology(self):
        """Verify that only train values contribute to climatology."""
        values = np.asarray(self.data.values[:, 2:3, :]).copy()
        train_end = 140
        clim_a, phi_a = fit_seasonal(values, self.data.dates, train_end, radius=15)
        values[train_end:, :, :] += 100000
        clim_b, phi_b = fit_seasonal(values, self.data.dates, train_end, radius=15)
        np.testing.assert_array_equal(clim_a, clim_b)
        np.testing.assert_array_equal(phi_a, phi_b)

    def test_mlps_train_and_evaluate_in_original_units(self):
        """Verify that mlps train and evaluate in original units."""
        ckpt = self._train('mlp')
        metadata = json.loads((ckpt/'train_metadata.json').read_text())
        self.assertEqual(metadata['target_mode'], 'anomaly')
        self.assertEqual(metadata['anomaly_climatology']['fit_end_exclusive'], 140)
        self.assertEqual(metadata['anomaly_climatology']['radius_days'], 15)
        self.assertEqual(metadata['normalization']['fit_interval'], [0, 140])
        self.assertEqual(metadata['training_config']['learning_rate'], 3e-4)
        # A network forced to output zero normalized anomalies must reconstruct
        # the train-only seasonal climatology plus the fitted anomaly mean.
        import torch
        weight_path = ckpt / 'best.pt'
        weights = torch.load(weight_path, map_location='cpu', weights_only=True)
        weights['state_dict'] = {k: torch.zeros_like(v) for k, v in weights['state_dict'].items()}
        torch.save(weights, weight_path)
        dest = self._test()
        self.assertIn('/anomaly/evaluations/', str(dest))
        with np.load(dest/'mlp_predictions.npz') as z:
            pred = z['prediction']
            target = z['target']
            self.assertEqual(pred.shape, (27, 1, 2, 14))
            self.assertEqual(pred.shape, target.shape)
            # Both forecasts and target are in original non-zero flux units.
            self.assertGreater(float(np.mean(pred)), 15)
            self.assertGreater(float(np.mean(target)), 15)
            clim, _ = fit_seasonal(
                self.data.values[:, 2:3, :], self.data.dates, 140, radius=15
            )
            future_dates = self.data.dates[z['origins'][:, None] + np.arange(1, 15)]
            expected = (clim[calendar_index(future_dates), 0, :] +
                        np.asarray(metadata['normalization']['mean'])[None, None, :])
            np.testing.assert_allclose(pred, expected.transpose(0, 2, 1)[:, None, :, :], atol=1e-5)
        report = json.loads((dest/'metrics.json').read_text())
        self.assertEqual(report['target_mode'], 'anomaly')
        self.assertIn('mlp', report['models'])

        # Baseline scoring stays in original units and does not rely on SAR
        self._test('level0_new')
        args = collect_results.build_parser().parse_args([
            '--split','test','--points','P3',
            '--out-len','14','--target-mode','anomaly',
            '--results-dir',str(self.result_dir),
        ])
        collect_results.collect(args)
        ranking = self.result_dir/'h14'/'anomaly'/'summary'/'forecast_1d_test_P3_h14_ranking.csv'
        self.assertTrue(ranking.exists())
        self.assertIn('mlp', ranking.read_text())

    def test_anomaly_checkpoint_cannot_be_used_as_raw(self):
        """Verify that anomaly checkpoint cannot be used as raw."""
        self._train('gru')
        bad = test.build_parser().parse_args([
            '--model', 'gru', '--target-mode', 'raw', '--points', 'P3',
            '--dataset-dir', str(self.dataset_dir), '--results-dir', str(self.result_dir),
            '--val-days', '40', '--test-days', '40', '--in-len', '15', '--out-len', '14',
            '--split', 'test', '--device', 'cpu',
        ])
        with self.assertRaises(FileNotFoundError):
            test.run(bad)

    def test_optional_sar_monitor_measures_anomalies_consistently(self):
        """Verify that optional sar monitor measures anomalies consistently."""
        ckpt = self._train('tcn', monitor='sar')
        metadata = json.loads((ckpt/'train_metadata.json').read_text())
        self.assertEqual(metadata['selection_metric'], 'sar')
        self.assertIsInstance(metadata['best_validation_sar_skill'], float)
        self._test('tcn')

    def test_default_point_selection_is_all15_in_anomaly_mode(self):
        """Verify that default point selection is all15 in anomaly mode."""
        args = train.build_parser().parse_args([
            '--model', 'mlp', '--target-mode', 'anomaly',
            '--dataset-dir', str(self.dataset_dir), '--results-dir', str(self.result_dir),
            '--val-days', '40', '--test-days', '40', '--in-len', '15', '--out-len', '14',
            '--epochs', '1', '--batch-size', '64', '--device', 'cpu',
            '--hidden-size', '8',
        ])
        train.run(args)
        ckpt_root = self.result_dir/'h14'/'anomaly'/'mlp'
        for i in range(1, 16):
            self.assertTrue((ckpt_root/f'P{i}'/'best.pt').is_file())
        ev = test.build_parser().parse_args([
            '--model', 'mlp', '--target-mode', 'anomaly',
            '--dataset-dir', str(self.dataset_dir), '--results-dir', str(self.result_dir),
            '--val-days', '40', '--test-days', '40', '--in-len', '15', '--out-len', '14',
            '--split', 'val', '--batch-size', '128', '--device', 'cpu',
        ])
        dest = test.run(ev)
        self.assertEqual(len(dest.name.split('_')), 15)
        with np.load(dest/'mlp_predictions.npz') as output:
            self.assertEqual(output['prediction'].shape, (27, 15, 2, 14))
        coll = collect_results.build_parser().parse_args([
            '--split', 'val', '--target-mode', 'anomaly',
            '--results-dir', str(self.result_dir),
        ])
        collect_results.collect(coll)
        self.assertTrue((self.result_dir/'h14'/'anomaly'/'summary'/
                         ('forecast_1d_val_'+'_'.join(f'P{i}' for i in range(1, 16))+'_h14_ranking.csv')).is_file())

    def test_mode_paths_are_isolated(self):
        """Verify that mode paths are isolated."""
        base = Path('/root/forecast')
        self.assertEqual(target_root(base, 14, 'raw'), base/'h14')
        self.assertEqual(target_root(base, 14, 'anomaly'), base/'h14'/'anomaly')
        self.assertEqual(DEFAULT_TARGET_MODE, 'raw')
        self.assertEqual(train.build_parser().get_default('learning_rate'), 3e-4)
        self.assertEqual(train.build_parser().get_default('scheduler'), 'plateau')
        self.assertEqual(train.build_parser().get_default('patience'), 12)


if __name__ == '__main__':
    unittest.main()
