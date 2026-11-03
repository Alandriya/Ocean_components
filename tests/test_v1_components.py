"""Offline regression tests for the portable Ocean_components v1 helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from Data_processing.mean_year import calculate_mean_year
from Data_processing.io import field_candidates, dates_from_labels, start_label
from Plotting.paths import plot_path, save_figure, translate_label
from apply_refactor import apply_refactor


class MeanYearTests(unittest.TestCase):
    """Check calendar alignment, masking, and tensor dimensions."""

    def test_leap_year_calendar_alignment(self):
        """Confirm March 1 is aligned in leap and common years."""
        dates = [date(2020, 2, 28), date(2020, 2, 29), date(2020, 3, 1),
                 date(2021, 2, 28), date(2021, 3, 1)]
        values = np.array([2., 100., 4., 6., 8.])
        mean, count = calculate_mean_year(values, dates)
        self.assertEqual(mean.shape, (365,))
        self.assertEqual(float(mean[58]), 4.)
        self.assertEqual(float(mean[59]), 6.)
        self.assertEqual(int(count[59]), 2)

    def test_channels_mask_and_missing_values(self):
        """Preserve the channel dimension and ignore masked ocean-grid cells."""
        values = np.arange(2 * 4 * 2 * 3., dtype=float).reshape(2, 4, 2, 3)
        values[0, 0, 0, 0] = np.nan
        ocean = np.array([[1, 0, 1], [1, 1, 0]], dtype=bool)
        average, coverage = calculate_mean_year(
            values, [date(2020, 1, 1), date(2021, 1, 1)], mask=ocean)
        self.assertEqual(average.shape, (365, 4, 2, 3))
        self.assertEqual(coverage.shape, average.shape)
        self.assertTrue(np.isnan(average[0, :, 0, 1]).all())
        self.assertEqual(int(coverage[0, 0, 0, 0]), 1)
        self.assertEqual(int(coverage[0, 1, 0, 0]), 2)

    def test_non_monotonic_dates_are_rejected(self):
        """Ensure chronological inputs are validated."""
        with self.assertRaises(ValueError):
            calculate_mean_year(np.ones((2,)), [date(2021, 1, 1), date(2020, 1, 1)])


class IOTests(unittest.TestCase):
    """Verify established filename and date labeling conventions."""

    def test_period_file_variants(self):
        """Include both original period-first and grouped-first file names."""
        paths = [str(p) for p in field_candidates("/tmp/root", "sensible", 1989, 1999)]
        self.assertTrue(any("sensible_1989-1999_grouped.npy" in name for name in paths))
        self.assertTrue(any("sensible_grouped_1989-1999.npy" in name for name in paths))

    def test_transition_label(self):
        """Check that a transition ending on 1989-01-01 has label 3654."""
        self.assertEqual(start_label(1989), 3654)
        self.assertEqual(dates_from_labels(3654, 1), [date(1989, 1, 1)])


class PlotPolicyTests(unittest.TestCase):
    """Ensure all image destinations are rooted under Plots."""

    def test_new_and_legacy_paths(self):
        """Verify both the explicit and migrated legacy save locations."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as folder:
            result = plot_path(folder, "Mean Year", "Variables", "sample.png")
            self.assertTrue(str(result).startswith(str(Path(folder) / "Plots")))
            fig, ax = plt.subplots()
            ax.set_xlabel("Longitude")
            saved = save_figure(fig, Path(folder) / "videos" / "Eigenvalues" / "pair" / "sample.png")
            self.assertEqual(saved, Path(folder) / "Plots" / "Eigenvalues" / "pair" / "sample.png")
            self.assertEqual(ax.get_xlabel(), "Долгота")
            self.assertTrue(saved.is_file())
            plt.close(fig)

    def test_invalid_plot_folder(self):
        """Reject path traversal outside the Plots root."""
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                plot_path(folder, "..", "escape.png")


class MigrationTests(unittest.TestCase):
    """Check the migration on an isolated synthetic checkout."""

    def test_migration_keeps_forecast_nn_untouched(self):
        """Preserve the forecast folder and remove legacy entry points."""
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder) / "Ocean_components"
            for name in ("Coefficients", "Data_processing", "Plotting", "Forecast_nn", "legacy_scripts"):
                (repo / name).mkdir(parents=True)
            (repo / "Forecast_nn" / "untouched.py").write_bytes(b"# Original forecast code\n")
            (repo / "main.py").write_text("def example():\n    pass\n")
            (repo / "main_parts.py").write_text("def read_ocean_mask():\n    return None\n")
            (repo / "Coefficients" / "runner.py").write_text(
                "def run_decade_coefficients(a):\n    return a\n")
            (repo / "Coefficients" / "nonparametric.py").write_text(
                "def estimate_np(data):\n    return data\n")
            (repo / "Coefficients" / "semiparametric.py").write_text(
                "def estimate_sp(data):\n    return data\n")
            (repo / "Data_processing" / "sde_conditions.py").write_text(
                "from datetime import date\nfrom typing import Any\nFIELD_PATHS = {}\n"
                "def estimate_period(first_day,\n"
                "                    fields: dict[str, tuple[str, ...]] = FIELD_PATHS) -> dict[str, Any]:\n"
                "    first_index = (first_day - date(1979, 1, 1)).days + 1\n"
                "    return {'first_index': first_index}\n"
                "def run_all(*,\n"
                "    fail_on_missing: bool = False,\n"
                ") -> list[dict[str, Any]]:\n"
                "    temp_dir = None\n"
                "    fields = {}\n"
                "    estimate_period(date(1979, 1, 1), temp_dir=temp_dir, fields=fields)\n"
                "    return []\n")
            (repo / "Plotting" / "video.py").write_text(
                "def create_video(a, b, c, d):\n    return None\n")
            (repo / "Plotting" / "sample.py").write_text(
                "def example(fig, root):\n"
                "    fig.savefig(root + 'videos/example.png')\n")
            (repo / "legacy_scripts" / "notes.txt").write_text("Old reference\n")
            report = apply_refactor(repo, PACKAGE)
            self.assertTrue(report["forecast_nn_unchanged"])
            self.assertFalse((repo / "main_parts.py").exists())
            self.assertFalse((repo / "legacy_scripts").exists())
            self.assertTrue((repo / "Coefficients" / "estimators.py").exists())
            self.assertIn("from Plotting.paths import save_figure", (repo / "Plotting" / "sample.py").read_text())
            self.assertIn('first_index = (first_day - date(1979, 1, 1)).days + 1 + coefficient_index_shift',
                          (repo / "Data_processing" / "sde_conditions.py").read_text())
            self.assertEqual((repo / "Forecast_nn" / "untouched.py").read_bytes(),
                             b"# Original forecast code\n")
            self.assertTrue((repo / ".ocean_refactor_v1_backup.zip").is_file())


if __name__ == "__main__":
    unittest.main()
