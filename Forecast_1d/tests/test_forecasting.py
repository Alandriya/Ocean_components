"""Synthetic tests for 1D chronological forecasting and persistence baselines."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from Forecast_1d.data import PointDataset, split_dates
from Forecast_1d.models.baselines import persistence, input_mean
from Forecast_1d.models.sarima import (fit_ar1, forecast_ar1, fast_ar1_supported,
                                      parse_order)
from Forecast_1d.train import run as train_model
from Forecast_1d.test import run as test_model
from Forecast_1d.metrics import score_predictions


class ForecastUnitTests(unittest.TestCase):
    """Regression tests covering Forecast Unit."""
    def test_baselines_never_include_target(self):
        """Verify that baselines never include target."""
        series = np.zeros((50, 10, 2), dtype=np.float64)
        series[:, :, :] = np.arange(50)[:, None, None]
        origins = np.array([30, 35])
        persistence_predictions = persistence(series, origins, 3)
        mean_predictions = input_mean(series, origins, 3, in_len=5)
        self.assertEqual(float(persistence_predictions[0, 0, 0, 2]), 30.0)
        self.assertEqual(float(mean_predictions[0, 0, 0, 2]), 28.0)
        series[36:] = 10**8
        np.testing.assert_array_equal(persistence_predictions, persistence(series, origins, 3))
        np.testing.assert_array_equal(mean_predictions, input_mean(series, origins, 3, 5))

    def test_ar1_learns_only_prior_data(self):
        """Verify that AR(1) learns only prior data."""
        rng = np.random.default_rng(52)
        values = np.zeros((165, 10, 2), dtype=np.float64)
        for t in range(1, len(values)):
            values[t] = 2.5 + 0.72 * values[t-1] + rng.normal(scale=0.3, size=(10, 2))
        params = fit_ar1(values, 0, 110)
        origins = np.array([120, 125])
        before = forecast_ar1(values, origins, 3, params)
        altered = values.copy()
        altered[126:] = 1000000
        after = forecast_ar1(altered, origins, 3, params)
        np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(params, fit_ar1(altered, 0, 110))

    def test_splits_only_use_targets_inside_split(self):
        """Verify that splits only use targets inside split."""
        split = split_dates(170, val_days=30, test_days=30, in_len=10, out_len=3)
        self.assertEqual(split.train_end, 110)
        for part in ("val", "test"):
            origins = split.origins(part)
            start, end = split.slice_for(part)
            self.assertGreaterEqual(origins.min() + 1, start)
            self.assertLess(origins.max() + 3, end)
            self.assertGreaterEqual(origins.min() - 9, 0)
        self.assertEqual(split.origins("test").size, 28)

    def test_sarima_spec_and_errors(self):
        """Verify that SARIMA spec and errors."""
        self.assertTrue(fast_ar1_supported((1, 0, 0), (0, 0, 0, 0), "c"))
        self.assertFalse(fast_ar1_supported((1, 0, 0), (1, 0, 0, 7), "c"))
        self.assertEqual(parse_order("0,1,1", 3), (0, 1, 1))
        with self.assertRaises(ValueError):
            parse_order("1,-1,0", 3)

    def test_scores_zero_and_skill(self):
        """Verify that scores zero and skill."""
        truth = np.full((2, 10, 2, 2), 10., dtype=np.float64)
        persistence_pred = np.zeros_like(truth)
        forecast = np.full_like(truth, 5.)
        scores = score_predictions(forecast, truth, baseline=persistence_pred)
        self.assertAlmostEqual(scores["1"]["persistence_mse_skill"], 0.75)
        self.assertEqual(scores["1"]["joint"]["rmse"], 5.0)


class ForecastIntegrationTests(unittest.TestCase):
    """Regression tests covering Forecast Integration."""
    def setUp(self):
        """Prepare the test fixtures before running the assertions."""
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.data_path = root / "dataset"
        self.result_path = root / "results"
        self.data_path.mkdir()
        rng = np.random.default_rng(24)
        series = np.empty((170, 10, 2), dtype=np.float32)
        series[0] = 0
        for t in range(1, len(series)):
            series[t] = 0.2 + 0.83 * series[t-1] + rng.normal(0, 0.8, (10, 2))
        np.save(self.data_path / "centers.npy", series)
        np.save(self.data_path / "dates.npy", np.datetime64("1979-01-01") +
                np.arange(len(series)).astype("timedelta64[D]"))
        points = {"points": [{"id": f"P{i}", "region": f"Test location {i}",
                              "lat_deg": 33.0, "lon_deg": -45.0} for i in range(1, 11)]}
        (self.data_path / "points.json").write_text(json.dumps(points), encoding="utf-8")

    def tearDown(self):
        """Release the test fixtures and temporary resources."""
        self.temp.cleanup()

    def _args(self, **kwargs):
        """Compute args for Forecast Integration Tests."""
        default = dict(model="sarima", dataset_dir=self.data_path,
                       results_dir=self.result_path,
                       val_days=30, test_days=30, in_len=10, out_len=3,
                       order="1,0,0", seasonal_order="0,0,0,0", trend="c",
                       backend="auto", fit_days=0, maxiter=100, overwrite=False,
                       split="test")
        default.update(kwargs)
        return SimpleNamespace(**default)

    def test_train_test_plot_and_no_refit(self):
        """Verify that train test plot and no refit."""
        dataset = PointDataset(self.data_path)
        train_model(self._args())
        result_dir = test_model(self._args(model="level0"))
        with np.load(result_dir / "sarima_predictions.npz", allow_pickle=False) as file:
            pred = file["prediction"]
            origins = file["origins"]
            targets = file["target"]
        self.assertEqual(pred.shape, (28, 10, 2, 3))
        np.testing.assert_array_equal(targets[..., 2], dataset.values[origins + 3])
        # The first origin is the day immediately before the test interval.
        self.assertEqual(int(origins[0]), 139)
        with self.assertRaises(FileExistsError):
            test_model(self._args(model="level0"))
        # Forecast values are independent of actual future targets.
        with np.load(self.result_path / "sarima" / "params.npz") as f:
            params = f["params"]
        altered = np.asarray(dataset.values).copy()
        altered[origins[0]+1:] = 1e6
        np.testing.assert_array_equal(
            forecast_ar1(dataset.values, origins[:1], 3, params),
            forecast_ar1(altered, origins[:1], 3, params),
        )
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            return
        from Forecast_1d.plotting.plot_results import run as plot_results
        outputs = plot_results(self._args(model="sarima", point="P1", horizon=[1, 3],
                                          show_baselines=True, dpi=60, output_dir=None))
        self.assertEqual(len(outputs), 2)
        for path in outputs:
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
