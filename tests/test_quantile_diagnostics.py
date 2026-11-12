"""Regression checks for 1D and 2D quantile diagnostics.

Run at the Ocean_components project root:
    python -m unittest discover -s tests -p 'test_quantile_diagnostics.py' -v
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from Coefficients.common import prepare_transition
from Coefficients.quantile_diagnostics import (
    diagnose_quantile_period,
    diagnose_quantile_period_1d,
    run_quantile_diagnostics_1d,
)


class QuantileDiagnosticTests(unittest.TestCase):
    @staticmethod
    def make_inputs():
        rng = np.random.default_rng(2026)
        a = rng.normal(size=(5, 10, 12))
        b = 0.5 * a + rng.normal(size=(5, 10, 12))
        mask = np.ones((10, 12), dtype=np.uint8)
        mask[0, 0] = 0
        a[2, 1, 1] = np.nan
        return a, b, mask

    def test_2d_unchanged(self):
        a, b, mask = self.make_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            result = diagnose_quantile_period(
                a, b, mask, tmp, start_year=1979, end_year=1989,
                bins=(3, 5), thresholds=(2, 10, 30),
                first_transition=1, stop_transition=4,
                corr_every=1, corr_bins=3, corr_lags=(1,),
            )
            self.assertEqual(result["transitions_analyzed"], 3)
            with (Path(tmp) / "daily_summary.csv").open(newline="") as fp:
                rows = list(csv.DictReader(fp))
            self.assertEqual(len(rows), 6)
            self.assertEqual(rows[0]["date"], "1979-01-02")
            for row in rows:
                n = int(row["N"])
                self.assertEqual(int(row["occupied"]) + int(row["empty"]), n*n)
                self.assertLessEqual(int(row["valid"]), int(mask.sum()))

    def test_1d_group_accounting_matches_np_sp_groups(self):
        a, _, mask = self.make_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            result = diagnose_quantile_period_1d(
                a, mask, tmp, variable="sensible",
                start_year=1979, end_year=1989,
                bins=(3, 5, 20), thresholds=(2, 10, 30),
                first_transition=1, stop_transition=4,
                corr_every=1, corr_bins=5, corr_lags=(1,),
            )
            self.assertEqual(result["transitions_analyzed"], 3)
            with (Path(tmp) / "daily_summary.csv").open(newline="") as fp:
                rows = list(csv.DictReader(fp))
            self.assertEqual(len(rows), 9)
            for row in rows:
                n, t = int(row["N"]), int(row["transition_t"])
                groups = prepare_transition(a[t-1], a[t], n, mask)
                sizes = np.asarray([len(g.increments) for g in groups.groups])
                valid = sum(sizes)
                self.assertEqual(int(row["valid"]), valid)
                self.assertEqual(int(row["occupied"]), len(groups.groups))
                self.assertEqual(int(row["empty"]), n - len(groups.groups))
                self.assertAlmostEqual(
                    float(row["coverage_ge_10_pct"]),
                    round(100 * sizes[sizes >= 10].sum() / valid, 4)
                    if valid else 0.0,
                    places=4,
                )
            self.assertEqual(rows[0]["date"], "1979-01-02")
            self.assertIn("resid_corr_lag1_row", rows[4])
            with (Path(tmp) / "by_bin_summary.csv").open(newline="") as fp:
                summaries = list(csv.DictReader(fp))
            self.assertEqual(len(summaries), 3)
            cfg = json.loads((Path(tmp) / "config.json").read_text())
            self.assertEqual(cfg["dimension"], 1)
            self.assertEqual(cfg["grouping"], "Coefficients.common.prepare_transition")

    def test_1d_tied_values_and_invalid_pairs(self):
        a = np.ones((3, 8, 10))
        mask = np.ones((8, 10), dtype=bool)
        mask[0, :] = 0
        a[1, 1, 1] = np.nan
        with tempfile.TemporaryDirectory() as tmp:
            diagnose_quantile_period_1d(
                a, mask, tmp, variable="sensible",
                start_year=2019, end_year=2026,
                bins=(5,), thresholds=(2, 30), corr_every=0,
            )
            with (Path(tmp) / "daily_summary.csv").open(newline="") as fp:
                rows = list(csv.DictReader(fp))
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(int(row["occupied"]), 1)
                self.assertEqual(int(row["empty"]), 4)
                self.assertEqual(int(row["zero_width_intervals"]), 5)
                self.assertEqual(int(row["distinct_state_values"]), 1)
                self.assertEqual(int(row["valid"]), 69)

    def test_variable_names_have_separate_outputs(self):
        a, b, mask = self.make_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "Plots" / "Diagnostics" / "quantile_cells_1d"
            with patch("Coefficients.quantile_diagnostics.load_field") as load:
                load.side_effect = lambda root_, name, *_: a if name == "sensible" else b
                for name in ("sensible", "latent"):
                    run_quantile_diagnostics_1d(
                        root, variable=name, periods=((1979, 1989),),
                        mask=mask, bins=(3,), corr_every=0,
                        out_dir=out / name,
                    )
            self.assertTrue((out / "sensible" / "1979-1989" / "daily_summary.csv").is_file())
            self.assertTrue((out / "latent" / "1979-1989" / "daily_summary.csv").is_file())
            self.assertTrue((out / "sensible" / "all_periods_summary.csv").is_file())
            self.assertTrue((out / "latent" / "all_periods_summary.csv").is_file())

    def test_invalid_corr_bins(self):
        a, _, mask = self.make_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                diagnose_quantile_period_1d(
                    a, mask, tmp, variable="sensible",
                    start_year=1979, end_year=1989,
                    bins=(8,), corr_every=30, corr_bins=15,
                )


if __name__ == "__main__":
    unittest.main()
