"""Run: python -m unittest Forecast_1d.tests.test_feature_experiment"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from Forecast_1d.feature_experiment import FeatureExperiment, PointTransform


class TestFeatureAlignment(unittest.TestCase):
    """Regression tests covering Test Feature Alignment."""
    def setUp(self):
        """Prepare the test fixtures before running the assertions."""
        self.ctx = tempfile.TemporaryDirectory()
        self.addCleanup(self.ctx.cleanup)
        root = Path(self.ctx.name)
        self.flux = root / 'atlantic15_native_1979-2026'
        self.features = self.flux / 'features_era5'
        self.features.mkdir(parents=True)
        self.N = 1400
        self.T = self.N - 1  # A trimmed ERA5 tail must NOT move train/val/test dates
        self.dates = np.datetime64('1979-01-01', 'D') + np.arange(self.N).astype('timedelta64[D]')
        x = np.arange(self.N, dtype=np.float32)
        original = np.empty((self.N, 2, 2), dtype=np.float32)
        original[..., 0] = 20 + np.sin(x[:, None]/17) + np.arange(2)[None, :]
        original[..., 1] = 35 + np.cos(x[:, None]/19) + np.arange(2)[None, :]
        values = np.zeros((self.T, 2, 20), dtype=np.float32)
        values[..., :2] = original[:-1]
        values[..., 2:8] = 1
        values[..., 8:] = 2
        values[-1,:,4] = 8000.0  # test-only extreme must not affect train normalization
        points = {'points': [{'id':'P1'},{'id':'P2'}]}
        names = ['sensible','latent','sst','sp','u10','v10','t2m','d2m'] + [f'stat{i}' for i in range(12)]
        (self.flux / 'points.json').write_text(json.dumps(points))
        (self.features / 'points.json').write_text(json.dumps(points))
        (self.features / 'manifest.json').write_text(json.dumps({'feature_names': names, 'target_names': ['sensible','latent']}))
        np.save(self.flux / 'centers.npy', original)
        np.save(self.flux / 'dates.npy', self.dates)
        np.save(self.features / 'features.npy', values)
        np.save(self.features / 'targets.npy', values[..., :2])
        np.save(self.features / 'dates.npy', self.dates[:-1])

    def test_trimmed_last_day_preserves_original_split(self):
        """Verify that trimmed last day preserves original split."""
        exp = FeatureExperiment(features_directory=self.features, flux_directory=self.flux)
        self.assertEqual(exp.split.train_end, self.N - 1095)
        self.assertEqual(exp.split.test_end, self.N-1)
        self.assertEqual(len(exp.selected_names('flux')), 2)
        self.assertEqual(len(exp.selected_names('era5')), 8)
        self.assertEqual(len(exp.selected_names('all')), 20)
        tr = PointTransform(exp, 0, 'all', 'anomaly', 15)
        origins = exp.split.origins('val')
        inputs, targets = tr.windows(origins[:3], 60, 14)
        self.assertEqual(inputs.shape, (3, 60, 20))
        self.assertEqual(targets.shape, (3, 14, 2))
        np.testing.assert_array_equal(inputs[0,-1,:], tr.x_norm[exp.split.val_start-1])
        np.testing.assert_array_equal(targets[0,0,:], tr.y_norm[exp.split.val_start])
        self.assertEqual(float(tr.x_mean[4]), 1.0)
        self.assertEqual(tr.unscale(targets, origins[:3]).shape, (3,14,2))

    def test_shifted_features_rejected(self):
        """Verify that shifted features rejected."""
        dates = np.load(self.features/'dates.npy')
        np.save(self.features/'dates.npy', dates + np.timedelta64(1,'D'))
        with self.assertRaises(ValueError):
            FeatureExperiment(features_directory=self.features, flux_directory=self.flux)

    def test_target_mismatch_rejected(self):
        """Verify that target mismatch rejected."""
        t = np.load(self.features/'targets.npy')
        t[7,0,0] += 1
        np.save(self.features/'targets.npy', t)
        with self.assertRaises(ValueError):
            FeatureExperiment(features_directory=self.features, flux_directory=self.flux)

if __name__ == '__main__': unittest.main()
