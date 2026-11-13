"""Offline regression tests for the ERA5 hourly quantile diagnostic.

Run from the Ocean_components project root:
  python -m unittest discover -s tests -p 'test_quantile_hourly.py' -v
"""
from __future__ import annotations

import ast
import csv
import json
import tempfile
import unittest
from pathlib import Path
from datetime import date
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from Coefficients.quantile_hourly import (
    HourlySource, _as_hour, _daily_average, _group_for_mode,
    _infer_stride, _same_group_corr, run_hourly_quantile_diagnostics,
)
from Coefficients.common import _bin_ids


class HourlyQuantileTests(unittest.TestCase):
    def test_no_shift_and_exact_four_snapshots(self):
        vals = np.arange(24, dtype=float)[:, None, None] * np.ones((1, 2, 3))
        vals[0] = 10.0
        avg = _daily_average(vals)
        np.testing.assert_allclose(avg, (10.0 + 6.0 + 12.0 + 18.0) / 4.0)
        self.assertEqual(str(_as_hour("1979-01-01T00:00")), "1979-01-01T00")
        with self.assertRaises(ValueError):
            _as_hour("1979-01-01T00:30")

    def test_exact_quantile_group_assignment_with_repeated_values(self):
        state = np.array([0, 0, 0, 2, 2, 5, 6, 7, 8, 9, 10, 11], dtype=float).reshape(1, 3, 4)
        nxt = state + 1
        row, sizes, residual, groups = _group_for_mode(
            state, nxt, dim=1, n_bins=3, water=np.ones((3, 4), dtype=bool))
        correct, _ = _bin_ids(state.reshape(-1), 3)
        np.testing.assert_array_equal(sizes, np.bincount(correct, minlength=3))
        self.assertEqual(int(sizes.sum()), 12)
        self.assertEqual(row["valid"], 12)
        np.testing.assert_allclose(residual[np.isfinite(residual)], 0.0)
        self.assertEqual(groups.shape, (1, 3, 4))

    def test_2d_group_accounting_and_missing_values(self):
        prev = np.stack((np.arange(50).reshape(1, 5, 10),
                         np.arange(50)[::-1].reshape(1, 5, 10)), axis=-1).astype(float)
        curr = prev + (1.0, 2.0)
        prev[0, 0, 0, 0] = np.nan
        water = np.ones((5, 10), bool)
        water[0, 1] = False
        row, sizes, _, groups = _group_for_mode(prev, curr, dim=2, n_bins=4, water=water)
        self.assertEqual(row["valid"], 48)
        self.assertEqual(int(sizes.sum()), 48)
        self.assertEqual(int((groups >= 0).sum()), 48)
        self.assertEqual(len(sizes), 16)

    def test_same_group_correlations_use_only_matched_cells(self):
        residual = np.arange(4*8*9, dtype=float).reshape(4, 8, 9, 1)
        ids = np.zeros((4, 8, 9), dtype=int)
        corr, pairs = _same_group_corr(residual, ids, axis=0, lag=1, component=0)
        self.assertGreater(pairs, 30)
        self.assertAlmostEqual(corr, 1, places=10)
        ids[1] = 5
        _, restricted = _same_group_corr(residual, ids, axis=0, lag=1, component=0)
        self.assertLess(restricted, pairs)

    def _make_npy_sources(self, root, days=3, shape=(9, 10), bad_hour=None):
        dates = 24 * days
        hours = np.arange(dates, dtype=np.float64)[:, None, None]
        yy, xx = np.indices(shape)
        sensible = hours / 10.0 + yy[None] / 3.0 + xx[None] / 7.0
        latent = 3 * hours / 10.0 - yy[None] / 5.0 + xx[None] / 13.0
        if bad_hour is not None:
            # Missing observations are still an hourly time label: blank field.
            sensible[bad_hour] = np.nan
        np.save(root / "sensible_hourly.npy", sensible)
        np.save(root / "latent_hourly.npy", latent)
        return ["sensible_hourly.npy", "latent_hourly.npy"]

    def _run_small_npy(self, root, *, bad_hour=None, max_days=2):
        files = self._make_npy_sources(root, bad_hour=bad_hour)
        return run_hourly_quantile_diagnostics(
            files_path_prefix=root, hourly_files=files,
            mask=np.ones((9, 10), dtype=bool), start_year=1979, end_year=1980,
            first_date="1979-01-01", last_date="1979-01-02",
            day_stride=1, max_days=max_days,
            hourly_npy_start="1979-01-01T00:00",
            bins_1d=(2, 3), bins_2d=(2, 3),
            thresholds=(2, 10, 30), corr_every=1,
            corr_bin_1d=2, corr_bin_2d=2,
            spatial_lags=(1,), temporal_lags=(1, 6), progress_every=1,
        )

    def test_1d_2d_all_modes_and_24hour_lag(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            result = self._run_small_npy(root)
            self.assertEqual(result["processed_days"], 2)
            output = Path(result["output_dir"])
            with (output / "daily_summary.csv").open() as f:
                records = list(csv.DictReader(f))
            self.assertEqual(len(records), 2 * 5 * 3 * 2)
            for mode, count in (("daily_mean4", 90), ("hourly_4", 360),
                                ("hourly_8", 720), ("hourly_12", 1080),
                                ("hourly_24", 2160)):
                sub = [r for r in records if r["mode"] == mode and
                       r["dimension"] == "2" and r["N"] == "2" and
                       r["date"] == "1979-01-01"]
                self.assertEqual(len(sub), 1)
                self.assertEqual(int(sub[0]["valid"]), count)
            conf = json.loads((output / "config.json").read_text())
            self.assertEqual(conf["timestamp_shift_hours"], 0)
            self.assertEqual(conf["transition_lag_hours"], 24)
            self.assertEqual(conf["modes_hours"]["daily_mean4"], [0, 6, 12, 18])
            self.assertTrue((output / "by_mode_summary.csv").is_file())
            with (output / "moment_stability_daily.csv").open() as fp:
                stability = list(csv.DictReader(fp))
            self.assertEqual(len(stability), 2 * 4 * 3 * 2)
            self.assertTrue((output / "moment_stability_summary.csv").is_file())
            # The synthetic increments are constant over all grid pixels,
            # so groupwise conditional means agree at common query states.
            compared = [r for r in stability if int(r["matched_points"]) > 0]
            self.assertTrue(compared)
            self.assertLess(max(float(r["rms_mean_vs_hourly24"]) for r in compared), 1e-11)

    def test_missing_clock_hour_skips_pair(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            result = self._run_small_npy(root, bad_hour=25)
            # Blank cells do not imply missing timestamps: they reduce valid count.
            self.assertEqual(result["processed_days"], 2)
            with (Path(result["output_dir"]) / "daily_summary.csv").open() as f:
                records = list(csv.DictReader(f))
            row = next(r for r in records if r["mode"] == "hourly_24" and
                       r["date"] == "1979-01-01" and r["dimension"] == "1" and
                       r["variable"] == "sensible" and r["N"] == "2")
            self.assertEqual(int(row["valid"]), 23 * 90)

    def test_netcdf_time_coordinate_and_grid_stride(self):
        try:
            import xarray as xr
        except ImportError:
            self.skipTest("xarray not available")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data = np.arange(48 * 5 * 7, dtype=np.float32).reshape(48, 5, 7)
            time = np.datetime64("1979-01-01T00") + np.arange(48).astype("timedelta64[h]")
            ds = xr.Dataset({
                "msshf": (("time", "latitude", "longitude"), data),
                "mslhf": (("time", "latitude", "longitude"), data * 0.3),
            }, coords={"time": time, "latitude": np.linspace(0, 4, 5),
                       "longitude": np.linspace(0, 6, 7)})
            ds.to_netcdf(root / "both.nc", engine="scipy")
            with HourlySource(root=root, paths=("both.nc",), spatial_shape=(3, 4),
                              stride="auto") as source:
                self.assertEqual(source.spatial_stride, 2)
                day = source.day(date(1979, 1, 1))
                self.assertEqual(day["sensible"].shape, (24, 3, 4))
                np.testing.assert_allclose(day["sensible"][0], data[0, ::2, ::2])

    def test_main_contains_full_new_route_and_existing_routes(self):
        tree = ast.parse((REPO / "main.py").read_text(encoding="utf-8"))
        all_strings = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == "task":
                for operand in node.comparators:
                    if isinstance(operand, ast.Constant) and isinstance(operand.value, str):
                        all_strings.add(operand.value)
        for name in ("quantile_hourly", "quantile_diagnostics", "quantile_diagnostics_1d",
                     "coefficients", "bridge_coefficients", "stationary", "copulas",
                     "grib", "grib-check", "fokker_planck_2d_prepare"):
            self.assertIn(name, all_strings)

    def test_bad_stride_rejected(self):
        self.assertEqual(_infer_stride((161, 181), (161, 181), "auto"), 1)
        self.assertEqual(_infer_stride((321, 361), (161, 181), "auto"), 2)
        with self.assertRaises(ValueError):
            _infer_stride((9, 10), (3, 4), "auto")


if __name__ == "__main__":
    unittest.main()
