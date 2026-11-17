"""Synthetic shape and end-to-end tests for level-1 neural forecasters."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for neural model tests")
class NeuralArchitectureTests(unittest.TestCase):
    def test_all_architectures_return_direct_three_day_two_channel_forecasts(self):
        from Forecast_1d.models.neural import build_model, default_config

        x = torch.randn(8, 60, 2)
        for name in ("mlp", "tcn", "gru"):
            model = build_model(name, 60, 3, default_config(name))
            y = model(x)
            self.assertEqual(tuple(y.shape), (8, 3, 2))

    def test_tcn_is_causal(self):
        from Forecast_1d.models.neural import build_model, default_config

        torch.manual_seed(11)
        model = build_model("tcn", 60, 3, default_config("tcn")).eval()
        x1 = torch.randn(2, 60, 2)
        x2 = x1.clone()
        x2[:, 45:, :] += 1000.0
        # The final representation depends on the whole input history, so
        # changing late input may change forecasts. A separate prefix check
        # verifies causal padding preserves the prefix outputs of its blocks.
        block = model.blocks[0]
        a = torch.randn(1, 32, 60)
        b = a.clone()
        b[:, :, 45:] += 1000.0
        out_a, out_b = block(a), block(b)
        torch.testing.assert_close(out_a[:, :, :45], out_b[:, :, :45])


@unittest.skipIf(torch is None, "PyTorch is required for neural model tests")
class NeuralTrainingIntegrationTests(unittest.TestCase):
    def test_one_point_train_and_test(self):
        from Forecast_1d.train import run as train_run
        from Forecast_1d.test import run as test_run
        from Forecast_1d.data import PointDataset

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "dataset"
            results_dir = root / "results"
            data_dir.mkdir()
            rng = np.random.default_rng(101)
            values = np.zeros((170, 10, 2), dtype=np.float32)
            for t in range(1, len(values)):
                values[t] = 0.3 + 0.75 * values[t - 1] + rng.normal(
                    0.0, 0.2, size=(10, 2)
                )
            np.save(data_dir / "centers.npy", values)
            np.save(
                data_dir / "dates.npy",
                np.datetime64("1979-01-01") +
                np.arange(len(values)).astype("timedelta64[D]")
            )
            points = {
                "points": [
                    {"id": f"P{i}", "region": f"Test point {i}",
                     "lat_deg": 10.0 + i, "lon_deg": -20.0}
                    for i in range(1, 11)
                ]
            }
            (data_dir / "points.json").write_text(json.dumps(points), encoding="utf-8")

            for model_name in ("mlp", "tcn", "gru"):
                train_args = SimpleNamespace(
                    model=model_name, points=["P3"], dataset_dir=data_dir,
                    results_dir=results_dir, order="1,0,0",
                    seasonal_order="0,0,0,0", trend="c", backend="auto",
                    in_len=10, out_len=3, val_days=30, test_days=30,
                    epochs=2, batch_size=16, learning_rate=1e-3,
                    weight_decay=1e-4, patience=2, min_delta=0.0,
                    dropout=0.1, hidden_size=16, seed=3, num_workers=0,
                    device="cpu", overwrite=False,
                )
                train_run(train_args)
                test_args = SimpleNamespace(
                    model=model_name, points=["P3"], split="test",
                    dataset_dir=data_dir, results_dir=results_dir,
                    in_len=10, mean_days=30, out_len=3,
                    val_days=30, test_days=30, batch_size=32,
                    device="cpu", overwrite=False,
                )
                destination = test_run(test_args)
                with np.load(destination / f"{model_name}_predictions.npz", allow_pickle=False) as archive:
                    prediction = archive["prediction"]
                    target = archive["target"]
                    self.assertEqual(tuple(prediction.shape), (28, 1, 2, 3))
                    self.assertEqual(tuple(target.shape), (28, 1, 2, 3))
                metrics = json.loads((destination / "metrics.json").read_text(encoding="utf-8"))
                self.assertIn(model_name, metrics["models"])
                self.assertEqual(metrics["point_ids"], ["P3"])


if __name__ == "__main__":
    unittest.main()
