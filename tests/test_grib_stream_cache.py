"""Regression tests for GRIB streaming without source seek; no ERA5 download needed."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class FakeEcCodes(types.ModuleType):
    def __init__(self):
        super().__init__("eccodes")
        self.decode_calls = 0

    def codes_new_from_message(self, blob):
        self.decode_calls += 1
        return json.loads(blob[16:-4].decode("ascii"))

    def codes_get(self, handle, key):
        if key not in handle:
            raise KeyError(key)
        return handle[key]

    def codes_get_array(self, handle, key):
        if key != "values":
            raise KeyError(key)
        return np.asarray(handle[key], dtype=float)

    def codes_release(self, handle):
        pass


def fake_grib_message(var, day, hour, shift):
    vals = (np.arange(9) + shift + hour).tolist()
    obj = {"shortName": var, "validityDate": int(day.strftime("%Y%m%d")),
           "validityTime": 100 * hour, "gridType": "regular_ll",
           "Ni": 3, "Nj": 3, "values": vals}
    raw = json.dumps(obj).encode("ascii")
    size = 16 + len(raw) + 4
    header = b"GRIB" + bytes((0, 0, 0, 2)) + size.to_bytes(8, "big")
    return header + raw + b"7777"


class GribStreamingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ds = (date(1979, 1, 1), date(1979, 1, 2))
        self.file1 = self.root / "sensible.grib"
        self.file2 = self.root / "latent.grib"
        for path, var, shift in ((self.file1, "msshf", 0),
                                 (self.file2, "mslhf", 100)):
            with path.open("wb") as f:
                for day in self.ds:
                    for hour in range(24):
                        f.write(fake_grib_message(var, day, hour, shift))
        self.fake = FakeEcCodes()
        self.aliases = {"sensible": ("msshf",), "latent": ("mslhf",)}

    def convert(self):
        from Coefficients.grib_stream_cache import prepare_grib_day_cache
        return prepare_grib_day_cache(
            paths=(self.file1, self.file2), days=self.ds,
            cache_dir=self.root / "cache", grid_shape=(2, 2),
            spatial_stride="auto", aliases=self.aliases,
            progress_every=0)

    def test_two_variables_and_exact_valid_hours(self):
        with patch.dict(sys.modules, {"eccodes": self.fake}):
            mapping, shape, stride = self.convert()
        self.assertEqual(shape, (3, 3))
        self.assertEqual(stride, 2)
        self.assertEqual(self.fake.decode_calls, 96)
        a = np.load(mapping["sensible"][self.ds[0]], allow_pickle=False)
        b = np.load(mapping["latent"][self.ds[1]], allow_pickle=False)
        self.assertEqual(a.shape, (24, 2, 2))
        np.testing.assert_array_equal(a[0], [[0, 2], [6, 8]])
        np.testing.assert_array_equal(a[6], [[6, 8], [12, 14]])
        np.testing.assert_array_equal(b[23], [[123, 125], [129, 131]])

    def test_second_run_reuses_cache_without_decoder(self):
        with patch.dict(sys.modules, {"eccodes": self.fake}):
            self.convert()
        old = self.fake.decode_calls
        with patch.dict(sys.modules, {"eccodes": None}):
            self.convert()  # no import and no original-file reads required
        self.assertEqual(self.fake.decode_calls, old)

    def test_truncated_grib_record_rejected(self):
        from Coefficients.grib_stream_cache import _raw_messages
        with self.file1.open("ab") as f:
            f.write(b"GRIB\x00\x00\x00\x02" + (125).to_bytes(8, "big") + b"too_short")
        with self.assertRaises(EOFError):
            list(_raw_messages(self.file1))

    def test_duplicate_valid_hour_rejected(self):
        with self.file1.open("ab") as f:
            f.write(fake_grib_message("msshf", self.ds[0], 0, 1))
        with patch.dict(sys.modules, {"eccodes": self.fake}):
            with self.assertRaisesRegex(ValueError, "Duplicate sensible"):
                self.convert()

    def test_full_diagnostic_and_csv_resume(self):
        import csv
        from Coefficients.quantile_hourly import run_hourly_quantile_diagnostics
        opts = dict(
            files_path_prefix=self.root, hourly_files=(self.file1, self.file2),
            mask=np.ones((2, 2), dtype=bool),
            start_year=1979, end_year=1989, max_days=1, day_stride=1,
            bins_1d=(2,), bins_2d=(2,), thresholds=(2,),
            corr_every=0, daily_reference=False,
            output_dir=self.root / "diagnostics", progress_every=1,
            grib_cache_dir=self.root / "cache", resume=True,
        )
        with patch.dict(sys.modules, {"eccodes": self.fake}):
            first = run_hourly_quantile_diagnostics(**opts)
            with Path(first["daily_summary"]).open(encoding="utf-8") as f:
                before = list(csv.DictReader(f))
            self.assertEqual(len(before), 15)
            second = run_hourly_quantile_diagnostics(**opts)
            with Path(second["daily_summary"]).open(encoding="utf-8") as f:
                after = list(csv.DictReader(f))
            self.assertEqual(before, after)
            settings = json.loads((self.root / "diagnostics" / "config.json").read_text())
            self.assertEqual(settings["resumed_completed_days"], 1)
            self.assertEqual(settings["processed_days"], 1)

    def test_hourly_source_reads_cached_days(self):
        from Coefficients.quantile_hourly import HourlySource
        with patch.dict(sys.modules, {"eccodes": self.fake}):
            with HourlySource(
                root=self.root, paths=(self.file1, self.file2),
                spatial_shape=(2, 2), stride="auto", required_days=self.ds,
                grib_cache_dir=self.root / "cache"
            ) as source:
                self.assertTrue(source.has_complete_day(self.ds[0]))
                self.assertTrue(source.has_complete_day(self.ds[1]))
                block = source.day(self.ds[0])
                self.assertEqual(block["sensible"].shape, (24, 2, 2))
                np.testing.assert_array_equal(block["sensible"][0], [[0, 2], [6, 8]])
                np.testing.assert_array_equal(block["latent"][12], [[112,114],[118,120]])


if __name__ == "__main__":
    unittest.main()
