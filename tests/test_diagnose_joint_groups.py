"""Unit tests for 2D joint quantile-group diagnostics."""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Coefficients.common import prepare_pair_transition
from Coefficients.runner import diagnose_joint_groups


class JointGroupDiagnosticsTests(unittest.TestCase):
    def test_group_accounting_matches_estimator(self):
        rng = np.random.default_rng(2026)
        series0 = rng.normal(size=(5, 9, 10))
        series1 = rng.normal(size=(5, 9, 10))
        mask = np.ones((9, 10), dtype=np.uint8)
        mask[0, 0] = 0
        mask[3, 4] = 0
        series0[2, 1, 1] = np.nan
        with tempfile.TemporaryDirectory() as folder:
            daily, by_bin = diagnose_joint_groups(
                series0, series1, mask, folder,
                bins_list=(3, 5), n_transitions=3, index_offset=1,
                thresholds=(2, 10, 30), make_plots=False,
            )
            self.assertEqual(len(daily), 6)
            self.assertEqual(len(by_bin), 2)
            with (Path(folder) / "group_counts.csv").open(encoding="utf-8-sig", newline="") as f:
                groups = list(csv.DictReader(f))
            self.assertEqual(len(groups), 3 * (3*3 + 5*5))
            for entry in daily:
                n = entry["bins_per_axis"]
                t = entry["transition_t"]
                same = prepare_pair_transition(
                    series0[t-1], series0[t], series1[t-1], series1[t], n, mask
                )
                self.assertEqual(entry["valid_transition_cells"], same.n_valid_pairs)
                self.assertEqual(entry["water_cells"], 88)
                row_counts = [int(r["group_size"]) for r in groups
                              if int(r["bins_per_axis"]) == n
                              and int(r["transition_t"]) == t]
                self.assertEqual(sum(row_counts), same.n_valid_pairs)
                self.assertEqual(sum(c == 0 for c in row_counts), entry["empty_groups"])
                self.assertEqual(sum(c == 1 for c in row_counts), entry["singleton_groups"])
                self.assertEqual(sum(c for c in row_counts if c >= 2),
                                 entry["cells_in_groups_ge_2"])
                self.assertEqual(entry["output_day_index"], t + 1)

    def test_identical_variables_occupy_diagonal_only(self):
        rng = np.random.default_rng(17)
        series = rng.normal(size=(4, 20, 25))
        with tempfile.TemporaryDirectory() as folder:
            daily, _ = diagnose_joint_groups(
                series, series, np.ones((20,25)), folder,
                bins_list=(5,), n_transitions=2, make_plots=False,
            )
            for row in daily:
                self.assertEqual(row["occupied_groups"], 5)
                self.assertEqual(row["empty_groups"], 20)
                self.assertEqual(row["singleton_groups"], 0)
            with (Path(folder) / 'group_counts.csv').open(encoding="utf-8-sig", newline="") as stream:
                for item in csv.DictReader(stream):
                    if int(item['bin_first']) != int(item['bin_second']):
                        self.assertEqual(int(item['group_size']), 0)

    def test_flattened_first_series_and_normal_second_series(self):
        rng = np.random.default_rng(7)
        s0 = rng.normal(size=(4, 161, 181))
        s1 = rng.normal(size=(4, 161, 181))
        mask = np.ones((161,181), dtype=np.uint8)
        mask[:2] = 0
        with tempfile.TemporaryDirectory() as folder:
            path0, path1 = Path(folder) / 'first.npy', Path(folder) / 'second.npy'
            np.save(path0, s0.reshape(4, -1).T)
            np.save(path1, s1)
            daily, _ = diagnose_joint_groups(path0, path1, mask, Path(folder)/'diagnostics',
                                              bins_list=(6,), n_transitions=2, make_plots=False)
            self.assertEqual(daily[0]["water_cells"], 159 * 181)
            self.assertEqual(daily[0]["valid_transition_cells"], 159 * 181)
            self.assertEqual(daily[1]["possible_groups"], 36)

    def test_input_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            s = np.zeros((2, 3, 4))
            with self.assertRaises(ValueError):
                diagnose_joint_groups(s, s, np.ones((3,4)), folder,
                                      bins_list=(3,), n_transitions=2, make_plots=False)
            with self.assertRaises(ValueError):
                diagnose_joint_groups(s, s, np.zeros((3,4)), folder,
                                      bins_list=(3,), n_transitions=1, make_plots=False)


