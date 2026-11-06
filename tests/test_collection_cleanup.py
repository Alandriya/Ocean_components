"""Regression tests for NP/SP aggregation, cleanup, and period output paths."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from Coefficients.collection import (
    coefficient_source,
    collect_daily_coefficients,
    collect_period_coefficients,
    load_coefficient_source,
    load_daily_or_collected,
    period_collections_complete,
    period_dir,
)


class CollectionCleanupTests(unittest.TestCase):
    """Exercise both coefficient methods without expensive numeric estimators."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.names = ("sensible", "latent")
        self.period = (2023, 2024)
        self.branch = (self.root / "Coefficients" / "2023-2024" /
                       "2d_bins_3" / "sensible_latent")
        self.first, self.last = 20, 23

    def create_daily(self, method, start=None, stop=None):
        start = self.first if start is None else start
        stop = self.last if stop is None else stop
        daily = self.branch / method / "daily"
        daily.mkdir(parents=True, exist_ok=True)
        fields = ("A", "B", "B_with_reg") if method.startswith("Semiparametric") else ("A", "B")
        for field in fields:
            channels = 2 if field == "A" else 4
            for label in range(start, stop + 1):
                offset = 1000 if method.startswith("Semiparametric") else 0
                np.save(daily / f"{field}_{label}.npy",
                        np.full((channels, 2, 3), label + offset, dtype=np.float32))
        return daily

    def collect(self, **kwargs):
        return collect_period_coefficients(
            str(self.root), names=self.names, start_year=2023, end_year=2024,
            start_label=self.first, end_label=self.last, n_bins_2d=3,
            run_1d=False, run_2d=True, methods=("np", "sp"),
            show_progress=False, **kwargs,
        )

    def test_both_methods_saved_then_daily_deleted(self):
        np_daily = self.create_daily("Nonparametric_2d")
        sp_daily = self.create_daily("Semiparametric_2d")
        result = self.collect(delete_daily=True)
        self.assertEqual(set(result), {"2d/np", "2d/sp"})
        self.assertEqual(set(result["2d/np"]), {"A", "B"})
        self.assertEqual(set(result["2d/sp"]), {"A", "B", "B_with_reg"})
        self.assertFalse(np_daily.exists())
        self.assertFalse(sp_daily.exists())
        for method, shift in (("np", 0), ("sp", 1000)):
            for field, file in result[f"2d/{method}"].items():
                array = np.load(file, mmap_mode="r")
                self.assertEqual(array.shape, (4, 2 if field == "A" else 4, 2, 3))
                np.testing.assert_array_equal(array[:, 0, 0, 0],
                                              np.arange(self.first, self.last + 1) + shift)

    def test_idempotent_after_cleanup(self):
        self.create_daily("Nonparametric_2d")
        self.create_daily("Semiparametric_2d")
        initial = self.collect(delete_daily=True)
        again = self.collect(delete_daily=True)
        self.assertEqual(initial, again)
        self.assertFalse((self.branch / "Nonparametric_2d" / "daily").exists())

    def test_no_deletion_if_sp_missing_daily_file(self):
        np_daily = self.create_daily("Nonparametric_2d")
        sp_daily = self.create_daily("Semiparametric_2d")
        (sp_daily / f"B_{self.last}.npy").unlink()
        with self.assertRaises(FileNotFoundError):
            self.collect(delete_daily=True)
        self.assertTrue((np_daily / f"A_{self.first}.npy").is_file())
        self.assertTrue((sp_daily / f"A_{self.first}.npy").is_file())

    def test_existing_merged_shape_mismatch_prevents_cleanup(self):
        np_daily = self.create_daily("Nonparametric_2d")
        sp_daily = self.create_daily("Semiparametric_2d")
        self.collect(delete_daily=False)
        np.save(np_daily / f"A_{self.first}.npy", np.zeros((2, 8, 8)))
        with self.assertRaisesRegex(ValueError, "differs from remaining daily"):
            self.collect(delete_daily=True)
        self.assertTrue((np_daily / f"B_{self.first}.npy").exists())
        self.assertTrue((sp_daily / f"B_{self.first}.npy").exists())

    def test_period_specs_recognize_aggregates_after_cleanup(self):
        self.create_daily("Nonparametric_2d")
        self.create_daily("Semiparametric_2d")
        self.collect(delete_daily=True)
        specs = [
            (self.branch / "Nonparametric_2d" / "daily", ("A", "B"),
             ((2, 2, 3), (4, 2, 3))),
            (self.branch / "Semiparametric_2d" / "daily", ("A", "B", "B_with_reg"),
             ((2, 2, 3), (4, 2, 3), (4, 2, 3))),
        ]
        self.assertTrue(period_collections_complete(specs, self.first, self.last))
        self.assertFalse(period_collections_complete(specs, self.first, self.last + 1))

    def test_collected_source_fallback_after_cleanup(self):
        self.create_daily("Nonparametric_2d")
        self.create_daily("Semiparametric_2d")
        self.collect(delete_daily=True)
        for method, shift in (("np", 0), ("sp", 1000)):
            source = coefficient_source(self.root, self.period, self.names, 3,
                                        method, "A", self.first + 2)
            self.assertEqual(source[1], 2)
            self.assertEqual(float(load_coefficient_source(source)[0, 0, 0]), self.first + 2 + shift)
            removed_daily = (self.branch / ("Nonparametric_2d" if method == "np" else "Semiparametric_2d") /
                             "daily" / f"A_{self.first+2}.npy")
            self.assertEqual(float(load_daily_or_collected(removed_daily)[0, 0, 0]), self.first + 2 + shift)

    def test_extra_daily_data_is_kept(self):
        np_daily = self.create_daily("Nonparametric_2d", start=self.first - 1)
        self.create_daily("Semiparametric_2d")
        self.collect(delete_daily=True)
        self.assertTrue((np_daily / f"A_{self.first-1}.npy").exists())
        self.assertTrue((np_daily / f"B_{self.first-1}.npy").exists())
        self.assertFalse((np_daily / f"A_{self.first}.npy").exists())

    def test_migrates_legacy_directory(self):
        old = self.root / "Coefficients" / "aligned_2023-2024"
        old.mkdir(parents=True)
        (old / "probe.txt").write_text("historic", encoding="utf-8")
        # Remove empty modern branch from fixture initialization (not created yet).
        new = period_dir(self.root, 2023, 2024)
        self.assertTrue((new / "probe.txt").exists())
        self.assertFalse(old.exists())

    def test_does_not_merge_ambiguous_period_directories(self):
        old = self.root / "Coefficients" / "aligned_2023-2024"
        new = self.root / "Coefficients" / "2023-2024"
        old.mkdir(parents=True)
        new.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            period_dir(self.root, 2023, 2024)

    def test_continuous_assembly_reads_collected_periods(self):
        """Bridge assembly must work when no period daily map survives."""
        from unittest.mock import patch
        from Coefficients import bridges
        from Data_processing.io import start_label

        boundary = start_label(2024)
        first_period, second_period = (2023, 2024), (2024, 2025)
        labels = {
            first_period: range(boundary - 2, boundary),
            second_period: range(boundary + 1, boundary + 3),
        }
        for period, values in labels.items():
            for method, folder in (("np", "Nonparametric_2d"),
                                   ("sp", "Semiparametric_2d")):
                parent = (self.root / "Coefficients" /
                          f"{period[0]}-{period[1]}" / "2d_bins_3" /
                          "sensible_latent" / folder)
                parent.mkdir(parents=True, exist_ok=True)
                fields = ("A", "B", "B_with_reg") if method == "sp" else ("A", "B")
                for field in fields:
                    n_channels = 2 if field == "A" else 4
                    data = np.stack([np.full((n_channels, 2, 3), label, dtype=np.float32)
                                     for label in values])
                    np.save(parent / f"{field}_{values.start}-{values.stop-1}.npy", data)
        for method, folder in (("np", "Nonparametric_2d"),
                               ("sp", "Semiparametric_2d")):
            bridge = (self.root / "Coefficients" / "bridges" / "2d_bins_3" /
                      "sensible_latent" / folder / "daily")
            bridge.mkdir(parents=True, exist_ok=True)
            for field in (("A", "B", "B_with_reg") if method == "sp" else ("A", "B")):
                n_channels = 2 if field == "A" else 4
                np.save(bridge / f"{field}_{boundary}.npy",
                        np.full((n_channels, 2, 3), boundary, dtype=np.float32))
        with patch.object(bridges, "_period_indices", side_effect=lambda root, period, names: labels[period]):
            result = bridges.assemble_continuous_coefficients(
                self.root, (first_period, second_period), self.names,
                n_bins_2d=3, methods=("np", "sp"),
            )
        self.assertEqual(set(result["methods"]), {"np", "sp"})
        for method in ("np", "sp"):
            series = np.load(result["methods"][method]["A"])
            np.testing.assert_array_equal(series[:, 0, 0, 0],
                                          [boundary - 2, boundary - 1, boundary,
                                           boundary + 1, boundary + 2])

    def test_standalone_nonparametric_1d_collection(self):
        daily = self.root / "scalar" / "Nonparametric" / "daily"
        daily.mkdir(parents=True)
        for field in ("A", "B"):
            for label in (2, 3):
                np.save(daily / f"{field}_{label}.npy", np.full((2, 3), label))
        result = collect_daily_coefficients(
            str(self.root), path="scalar", method="Nonparametric",
            start_label=2, end_label=3, show_progress=False, delete_daily=True,
        )
        self.assertEqual(np.load(result["A"]).shape, (2, 2, 3))
        self.assertFalse(daily.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
