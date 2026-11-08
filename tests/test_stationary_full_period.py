"""Synthetic tests for full-period stationary preprocessing and labeling."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "Functional" / "stationary.py"
spec = importlib.util.spec_from_file_location("stationary_tested", MODULE_PATH)
stationary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stationary)


class FullPeriodTests(unittest.TestCase):
    """Validate data/coeff endpoint alignment and streaming group statistics."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.names = ("sensible", "latent")
        self.n = 70
        self.h, self.w = 5, 5
        rng = np.random.default_rng(42)
        self.obs = []
        for i, name in enumerate(self.names):
            folder = self.root / "DATA" / "Fluxes"
            folder.mkdir(parents=True, exist_ok=True)
            values = (rng.normal(size=(self.n + 1, self.h, self.w)) * 2
                      - i * 20 + np.arange(self.n + 1)[:, None, None] * 0.04)
            self.obs.append(values)
            np.save(folder / f"{name}_1979-2026_grouped.npy", values)
        parent = self.root / "Coefficients" / "1979-2026" / "sensible_latent"
        parent.mkdir(parents=True)
        a = np.empty((self.n, 2, self.h, self.w), dtype=float)
        a[:, 0] = 1.0
        a[:, 1] = -1.0
        b = np.zeros((self.n, 4, self.h, self.w), dtype=float)
        b[:, 0] = 2.0
        b[:, 1] = 4.0
        b[:, 2] = 3.0
        b[:, 3] = 5.0
        np.save(parent / "A_1979-2026_sp.npy", a)
        np.save(parent / "B_1979-2026_sp.npy", b)
        self.parent = parent
        self.mock_io = types.ModuleType("Data_processing.io")
        self.mock_io.field_candidates = lambda root, name, start, end: [
            Path(root) / "DATA" / "Fluxes" / f"{name}_{start}-{end}_grouped.npy"]
        self.mock_io.load_field = lambda root, name, start, end: np.load(
            self.mock_io.field_candidates(root, name, start, end)[0], mmap_mode="r")
        self.mock_io.read_ocean_mask = lambda root, spatial_shape: np.ones(spatial_shape, dtype=bool)
        self.mock_io.start_label = lambda year: (date(year, 1, 1) - date(1979, 1, 1)).days + 1
        data_pkg = types.ModuleType("Data_processing")
        data_pkg.__path__ = []
        self.module_patch = patch.dict(sys.modules, {
            "Data_processing": data_pkg, "Data_processing.io": self.mock_io})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def test_B_channel_order_yields_correct_marginal_variances(self):
        inputs, data = stationary._group_full_period(
            self.root, self.names, 1979, 2026, "sp", bins=30,
            trim_bins=3, chunk_days=7, block_size=1,
            sample_size=2000, refresh_grouped=True)
        self.assertEqual(data["sources"]["first_label"], 2)
        self.assertEqual(data["sources"]["last_label"], self.n + 1)
        for name, variance in (("sensible", 13.0), ("latent", 41.0)):
            self.assertGreater(len(inputs[name][0]), 10)
            np.testing.assert_allclose(inputs[name][2], np.log(variance), atol=1e-12)

    def test_BBT_preferred_over_B_and_cache_reused(self):
        bbt = np.zeros((self.n, 4, self.h, self.w), dtype=float)
        bbt[:, 0] = 17.0
        bbt[:, 1] = 29.0
        np.save(self.parent / "BBT_1979-2026_sp.npy", bbt)
        kwargs = dict(bins=30, trim_bins=3, chunk_days=7,
                      block_size=1, sample_size=2000, refresh_grouped=False)
        first, meta = stationary._group_full_period(
            self.root, self.names, 1979, 2026, "sp", **kwargs)
        np.testing.assert_allclose(first["sensible"][2], np.log(17.0))
        np.testing.assert_allclose(first["latent"][2], np.log(29.0))
        with patch.object(stationary, "_stream_stationary_chunks",
                          side_effect=AssertionError("Recalculation occurred")):
            second, _ = stationary._group_full_period(
                self.root, self.names, 1979, 2026, "sp", **kwargs)
        np.testing.assert_array_equal(first["latent"][1], second["latent"][1])
        self.assertTrue(Path(meta["cache"]).exists())

    def test_stale_BBT_falls_back_to_B(self):
        bbt_path = self.parent / "BBT_1979-2026_sp.npy"
        bbt = np.zeros((self.n, 4, self.h, self.w), dtype=float)
        bbt[:, 0] = 1700.0
        bbt[:, 1] = 2900.0
        np.save(bbt_path, bbt)
        b_path = self.parent / "B_1979-2026_sp.npy"
        old_time = b_path.stat().st_mtime_ns - 10_000_000_000
        os.utime(bbt_path, ns=(old_time, old_time))
        fields, a, d, details = stationary._assembled_sources(
            self.root, self.names, 1979, 2026, "sp")
        self.assertEqual(details["diffusion_kind"], "B")

    def test_nonparametric_output_uses_separate_cache(self):
        for key in ("A", "B"):
            original = self.parent / f"{key}_1979-2026_sp.npy"
            np.save(self.parent / f"{key}_1979-2026_np.npy", np.load(original))
        inputs, meta = stationary._group_full_period(
            self.root, self.names, 1979, 2026, "np", bins=30,
            trim_bins=3, chunk_days=7, block_size=1,
            sample_size=2000, refresh_grouped=True)
        self.assertIn("_np_", meta["cache"])
        np.testing.assert_allclose(inputs["latent"][2], np.log(41.0), atol=1e-12)

    def test_legacy_decade_groups_are_unchanged(self):
        prefix = self.root / "Functional" / "sensible-latent"
        prefix.mkdir(parents=True)
        q = np.arange(10, dtype=float)
        np.save(prefix / "1-3653_quantiles1.npy", q)
        np.save(prefix / "1-3653_quantiles2.npy", q + 1)
        np.save(prefix / "1-3653_a_grouped.npy", np.vstack([q, q * 2]))
        np.save(prefix / "1-3653_b_grouped_log_simple.npy", np.vstack([q * 3, q * 4]))
        saved = stationary._load_historical_grouped(
            self.root, self.names, 1, 3653, "simple")
        np.testing.assert_array_equal(saved["sensible"][0], q)
        np.testing.assert_array_equal(saved["latent"][1], q * 2)

    def test_first_flux_day_is_endpoint_observation(self):
        fields, a, d, details = stationary._assembled_sources(
            self.root, self.names, 1979, 2026, "sp")
        chunks = stationary._stream_stationary_chunks(
            fields, a, d, details, np.ones((5, 5), dtype=bool),
            chunk_days=7, block_size=1)
        result = next(chunks)
        np.testing.assert_allclose(result[0][0][:25], self.obs[0][1].ravel() - 1.0)
        np.testing.assert_allclose(result[1][0][:25], self.obs[1][1].ravel() + 1.0)

    def test_spatial_block_averaging_rejects_land(self):
        arr = np.array([[[1., 3.], [10., 11.]]])
        valid = np.array([[[True, True], [False, False]]])
        got = stationary._mean_in_spatial_blocks(arr, valid, 2)
        self.assertEqual(got.shape, (1, 1, 1))
        self.assertAlmostEqual(float(got[0, 0, 0]), 2.0)

    def test_incomplete_field_is_rejected(self):
        file = self.root / "DATA" / "Fluxes" / "latent_1979-2026_grouped.npy"
        np.save(file, self.obs[1][:-3])
        with self.assertRaisesRegex(ValueError, "Time alignment failed"):
            stationary._assembled_sources(self.root, self.names, 1979, 2026, "sp")

    def test_stationary_saves_historical_filename_and_prints(self):
        with patch.object(stationary, "_fit_log_diffusion", return_value={
            "x0": -15.0, "x1": -40.0, "c1": 0.1, "c2": 0.2,
            "c3": 0.3, "c4": 0.4, "rmse_log_b": 0.1,
        }), patch.object(stationary, "_stationary_density",
                 side_effect=lambda grid, params, variable:
                 (np.exp(-((grid + 30) / 90) ** 2) /
                  np.trapezoid(np.exp(-((grid + 30) / 90) ** 2), grid))):
            p = stationary.estimate_stationary_characteristics(
                self.root, 1979, 2026, method="sp", quantile_bins=30,
                trim_bins=3, chunk_days=7, spatial_block_size=1,
                sample_size=2000)
        self.assertEqual(p.name, "sensible-latent_1979-2026.npz")
        with np.load(p) as values:
            self.assertIn("pdf_sensible", values.files)
            self.assertIn("pdf_latent", values.files)
        metadata = json.loads(p.with_suffix(".json").read_text())
        self.assertEqual(metadata["provenance"]["sources"]["last_label"], self.n + 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
