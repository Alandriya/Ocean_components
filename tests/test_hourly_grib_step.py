"""GRIB-style time/step regression tests for the hourly ERA5 loader.

These use xarray-backed synthetic NetCDF, with exactly the coordinates/dims
that cfgrib exposes. A real GRIB decoder is not needed to run them.
Run: python -m unittest discover -s tests -p 'test_hourly_grib_step.py' -v
"""
from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np

from Coefficients.quantile_hourly import HourlySource, _as_hour, run_hourly_quantile_diagnostics

try:
    import xarray as xr
except ImportError:
    xr = None


@unittest.skipIf(xr is None, "xarray is required for GRIB-style coordinate tests")
class HourlyGribStepTests(unittest.TestCase):
    @staticmethod
    def make_dataset(*, include_valid_time=True):
        # Daily forecast cycles, with steps 1..24 h. Validity 00 UTC of
        # Jan 1 is the *previous* cycle, step 24; no manual time correction.
        starts = np.datetime64("1978-12-31T00") + np.arange(4).astype("timedelta64[D]")
        leads = np.arange(1, 25).astype("timedelta64[h]")
        valids = starts[:, None] + leads[None, :]
        data = (1000 * np.arange(4)[:, None, None, None]
                + np.arange(1, 25)[None, :, None, None]
                + np.arange(5)[None, None, :, None]
                + np.arange(7)[None, None, None, :]).astype(np.float32)
        coordinates = {"time": starts, "step": leads,
                       "latitude": np.arange(5.), "longitude": np.arange(7.)}
        if include_valid_time:
            coordinates["valid_time"] = (("time", "step"), valids)
        return xr.Dataset({"msshf": (("time", "step", "latitude", "longitude"), data),
                           "mslhf": (("time", "step", "latitude", "longitude"), data * 2)},
                          coords=coordinates), data

    def test_non_singleton_step_mapped_by_valid_time(self):
        ds, data = self.make_dataset()
        da, times, coords, selectors = HourlySource._normalize_array(ds["msshf"])
        self.assertEqual(da.dims, ("time", "step", "latitude", "longitude"))
        self.assertEqual(da.shape, (4, 24, 5, 7))
        self.assertEqual(len(times), 4 * 24)
        self.assertEqual(len(selectors), len(times))
        self.assertEqual(selectors[23], (0, 23))
        self.assertEqual(str(np.datetime64(times[23], "h")), "1979-01-01T00")

        with tempfile.TemporaryDirectory() as d:
            file = Path(d) / "fluxes.nc"
            ds.to_netcdf(file, engine="scipy")
            with HourlySource(root=d, paths=("fluxes.nc",), spatial_shape=(5, 7)) as source:
                self.assertTrue(source.has_complete_day(date(1979, 1, 1)))
                self.assertTrue(source.has_complete_day(date(1979, 1, 2)))
                fields = source.day(date(1979, 1, 1))
                np.testing.assert_array_equal(fields["sensible"][0], data[0, 23])
                np.testing.assert_array_equal(fields["sensible"][1], data[1, 0])
                np.testing.assert_array_equal(fields["sensible"][12], data[1, 11])
                np.testing.assert_array_equal(fields["sensible"][23], data[1, 22])
                np.testing.assert_array_equal(fields["latent"][0], data[0, 23] * 2)

    def test_full_diagnostic_with_time_step_input(self):
        ds, _ = self.make_dataset()
        with tempfile.TemporaryDirectory() as d:
            file = Path(d) / "fluxes.nc"
            ds.to_netcdf(file, engine="scipy")
            result = run_hourly_quantile_diagnostics(
                files_path_prefix=d, hourly_files=("fluxes.nc",),
                mask=np.ones((5, 7), dtype=bool),
                start_year=1979, end_year=1980,
                first_date="1979-01-01", last_date="1979-01-01",
                modes=("daily_mean4", "hourly_24"),
                bins_1d=(2,), bins_2d=(2,), thresholds=(2, 10),
                corr_every=0, max_days=1, progress_every=1,
                compare_min_group_size=2,
            )
            self.assertEqual(result["processed_days"], 1)
            import csv
            with (Path(result["output_dir"]) / "daily_summary.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 6)  # 2 modes × (2 scalar + 1 bivariate)
            for row in rows:
                self.assertEqual(int(row["valid"]), 35 if row["mode"] == "daily_mean4" else 24*35)

    def test_missing_valid_time_derived_from_step_timedelta(self):
        ds, data = self.make_dataset(include_valid_time=False)
        field, times, _, selectors = HourlySource._normalize_array(ds["msshf"])
        self.assertEqual(field.shape[:2], (4, 24))
        self.assertEqual(str(np.datetime64(times[23], "h")), "1979-01-01T00")
        self.assertEqual(selectors[24], (1, 0))

    def test_non_hour_validity_and_nat_rejected(self):
        with self.assertRaises(ValueError):
            _as_hour(np.datetime64("NaT"))
        with self.assertRaises(ValueError):
            _as_hour(np.datetime64("1979-01-01T01:30"))

    def test_duplicate_valid_hour_is_not_silently_selected(self):
        ds, _ = self.make_dataset()
        # Two forecast references 12 h apart and steps 1 and 13 h
        # yield a duplicate validity hour, so the input is ambiguous.
        starts = np.datetime64("1979-01-01T00") + np.array([0, 12]).astype("timedelta64[h]")
        leads = np.array([1, 13]).astype("timedelta64[h]")
        data = np.ones((2, 2, 5, 7), dtype=np.float32)
        conflicting = xr.Dataset({
            "msshf": (("time", "step", "latitude", "longitude"), data),
            "mslhf": (("time", "step", "latitude", "longitude"), data),
        }, coords={"time": starts, "step": leads,
                   "valid_time": (("time", "step"), starts[:, None] + leads[None, :]),
                   "latitude": np.arange(5.), "longitude": np.arange(7.)})
        with tempfile.TemporaryDirectory() as d:
            conflicting.to_netcdf(Path(d) / "collision.nc", engine="scipy")
            with self.assertRaisesRegex(ValueError, "Duplicate valid UTC timestamp"):
                HourlySource(root=d, paths=("collision.nc",), spatial_shape=(5, 7))


if __name__ == "__main__":
    unittest.main()
