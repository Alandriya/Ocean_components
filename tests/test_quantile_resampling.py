"""Quantile resampling tests on synthetic hourly arrays; no ERA5 download needed."""
from __future__ import annotations

import csv
import importlib
import json
import sys
import tempfile
import types
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The released archive is a partial repository overlay. In the user's real
# repository these modules already exist; stand-ins only allow isolated tests.
if "Coefficients.common" not in sys.modules:
    try:
        import Coefficients.common
    except ModuleNotFoundError:
        cm = types.ModuleType("Coefficients.common")
        def _bin_ids(values, n_bins):
            edges = np.quantile(values, np.linspace(0, 1, n_bins+1))
            return np.clip(np.searchsorted(edges, values, side="right")-1, 0, n_bins-1), edges
        def load_water_mask(mask, shape):
            return np.ones(shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool).reshape(shape)
        cm._bin_ids = _bin_ids
        cm.load_water_mask = load_water_mask
        sys.modules["Coefficients.common"] = cm
        diag = types.ModuleType("Coefficients.quantile_diagnostics")
        diag._weighted_p10_group_size = lambda x: 0
        sys.modules["Coefficients.quantile_diagnostics"] = diag

from Coefficients.quantile_resampling import (
    _group_fit, _spatial_choices, _cache_matches, _open_source,
    run_quantile_hourly_resampling,
)
from Coefficients.quantile_hourly import _group_for_mode


class ResamplingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.shape = (6, 8)
        self.mask = np.ones(self.shape, bool)
        self.mask[0, 0] = False
        self.mask[3, 4] = False
        hours = 24*4
        Y, X = np.indices(self.shape)
        t = np.arange(hours, dtype=float)[:, None, None]
        sensible = 100 + (Y*X)[None] + 0.1*t + np.cos(t/8)
        latent = 120 + Y[None]*3 + X[None]*2 + .2*t + np.sin(t/6)
        sensible = sensible.astype(float)
        latent = latent.astype(float)
        np.save(self.root / "sensible.npy", sensible)
        np.save(self.root / "latent.npy", latent)
        self.opts = dict(
            files_path_prefix=self.root, hourly_files=("sensible.npy", "latent.npy"),
            mask=self.mask, start_year=1979, end_year=1989,
            first_date="1979-01-01", last_date="1979-01-02", day_stride=1,
            hours=(4, 8, 12), replicates=2, spatial_steps=(1, 2),
            bins_1d=(3,), bins_2d=(2, 3),
            hourly_npy_start="1979-01-01T00:00",
            compare_min_group_size=2,
            output_dir=self.root / "out", resume=True,
            progress_every=1,
        )

    def test_spatial_masks_matched_size_and_regular_step(self):
        rng = np.random.default_rng(72)
        picks = _spatial_choices(self.mask, 2, rng)
        self.assertEqual([x[0] for x in picks], ["grid", "random"])
        self.assertEqual(picks[0][1].sum(), picks[1][1].sum())
        ys, xs = np.nonzero(picks[0][1])
        self.assertEqual(len(set(ys % 2)), 1)
        self.assertEqual(len(set(xs % 2)), 1)
        np.testing.assert_array_equal(picks[0][1] & ~self.mask, np.zeros(self.shape,bool))

    def test_vectorized_fits_match_original_quantile_method(self):
        rng = np.random.default_rng(3)
        for dimension in (1, 2):
            prev = rng.normal(size=(3, 6, 8, dimension))
            curr = prev + rng.normal(size=prev.shape)
            if dimension == 1:
                old, sizes, *_ = _group_for_mode(prev[..., 0], curr[..., 0], dim=1,
                                                 n_bins=3, water=self.mask, return_fit=True)
                kept = np.broadcast_to(self.mask, prev.shape[:3])
            else:
                old, sizes, *_ = _group_for_mode(prev, curr, dim=2,
                                                 n_bins=3, water=self.mask, return_fit=True)
                kept = np.broadcast_to(self.mask, prev.shape[:3])
            states, increments = prev[kept], (curr-prev)[kept]
            fresh, fit = _group_fit(states, increments, dimension, 3)
            original_fit = _group_for_mode(
                prev[..., 0] if dimension == 1 else prev,
                curr[..., 0] if dimension == 1 else curr,
                dim=dimension, n_bins=3, water=self.mask, return_fit=True)[-1]
            np.testing.assert_allclose(fit["mean"], original_fit["mean"])
            np.testing.assert_allclose(fit["cov"], original_fit["cov"])
            self.assertEqual(old["valid"], fresh["valid"])
            self.assertEqual(old["occupied"], fresh["occupied"])
            np.testing.assert_array_equal(sizes, fit["sizes"])

    def test_full_run_has_expected_rows_and_nested_hours(self):
        result = run_quantile_hourly_resampling(**self.opts)
        self.assertEqual(result["processed_dates"], 2)
        with (self.root / "out" / "resampling_daily.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        # days x reps x hours x spatial variants x (2 scalar + 2 joint N)
        self.assertEqual(len(rows), 2*2*3*3*4)
        for day in ("1979-01-01", "1979-01-02"):
            for rep in (0, 1):
                subsets = {}
                for n in (4, 8, 12):
                    r = next(x for x in rows if x["date"]==day and
                             x["replicate"]==str(rep) and x["hours_used"]==str(n))
                    subsets[n] = set(map(int, r["selected_hours"].split()))
                self.assertTrue(subsets[4] <= subsets[8] <= subsets[12])
        # Within a (day,rep,hours,step) pair, grid and random select same n.
        target = [r for r in rows if r["spatial_step"] == "2" and r["dimension"] == "2"]
        a, b = target[:2]
        self.assertEqual(a["spatial_points"], b["spatial_points"])
        self.assertTrue((self.root/"out"/"resampling_summary.csv").exists())
        self.assertTrue((self.root/"out"/"spatial_grid_vs_random_summary.csv").exists())
        self.assertEqual(json.loads((self.root/"out"/"config.json").read_text())
                         ["time_shift_hours"], 0)

    def test_resume_keeps_complete_days_and_recomputes_partial(self):
        first = run_quantile_hourly_resampling(**self.opts)
        output = Path(first["daily"])
        with output.open(newline="") as f:
            before = list(csv.DictReader(f))
        same = run_quantile_hourly_resampling(**self.opts)
        with output.open(newline="") as f:
            after = list(csv.DictReader(f))
        self.assertEqual(before, after)
        # Simulate crash during second date: remove some lines.
        with output.open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(before[0]))
            wr.writeheader()
            wr.writerows([r for r in before if r["date"] == "1979-01-01"])
            wr.writerows([r for r in before if r["date"] == "1979-01-02"][:3])
        run_quantile_hourly_resampling(**self.opts)
        with output.open(newline="") as f:
            rebuilt = list(csv.DictReader(f))
        self.assertEqual(before, rebuilt)

    def test_extend_from_smoke_to_full_without_losing_days(self):
        smoke = dict(self.opts, max_days=1)
        run_quantile_hourly_resampling(**smoke)
        with (self.root / "out" / "resampling_daily.csv").open(newline="") as f:
            rows_first = list(csv.DictReader(f))
        self.assertEqual({r["date"] for r in rows_first}, {"1979-01-01"})
        run_quantile_hourly_resampling(**dict(self.opts, max_days=None))
        with (self.root / "out" / "resampling_daily.csv").open(newline="") as f:
            rows_final = list(csv.DictReader(f))
        self.assertEqual(rows_final[:len(rows_first)], rows_first)
        self.assertEqual(len(rows_final), 2*len(rows_first))

    def test_changed_seed_rejects_resume(self):
        run_quantile_hourly_resampling(**self.opts)
        other = dict(self.opts, random_seed=31)
        with self.assertRaisesRegex(ValueError, "settings or input files changed"):
            run_quantile_hourly_resampling(**other)

    def test_valid_old_cache_does_not_require_grib_seek(self):
        files = (self.root/"a.grib",self.root/"b.grib")
        for path in files:
            path.write_bytes(b"GRIBnot-real-data")
        cached = self.root/"Functional"/"Diagnostics"/"quantile_hourly"/"1979-1989"/"_grib_day_cache"
        days = {date(1979, 1, 1),date(1979, 1, 2)}
        sig = {"sources": [{"path":str(p.resolve()), "size":p.stat().st_size,
                             "mtime_ns":p.stat().st_mtime_ns} for p in files],
               "grid_shape":list(self.shape)}
        for v in ("sensible", "latent"):
            (cached/v).mkdir(parents=True)
            for day in days:
                np.save(cached/v/f"{day}.npy",np.zeros((24,*self.shape),np.float32))
        (cached/"complete_manifest.json").write_text(json.dumps({"signature":sig,
            "source_shape":list(self.shape),"spatial_stride":1,
            "complete_days":{v:[str(d) for d in days] for v in ("sensible","latent")}}))
        with _open_source(self.root,files,days,self.shape,"auto",None) as source:
            self.assertTrue(source.has_complete_day(date(1979,1,1)))
            self.assertEqual(source.day(date(1979,1,1))["latent"].shape,(24,*self.shape))
        self.assertTrue(_cache_matches(cached,files,days,self.shape,"auto"))
        files[0].write_bytes(b"updated source")
        self.assertFalse(_cache_matches(cached,files,days,self.shape,"auto"))

    def test_incompatible_settings(self):
        with self.assertRaisesRegex(ValueError, "<= 24"):
            run_quantile_hourly_resampling(**dict(self.opts, hours=(25,)))


if __name__ == "__main__":
    unittest.main()
