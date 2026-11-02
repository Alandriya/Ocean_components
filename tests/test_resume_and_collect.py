"""Regression tests for resumable runs and memory-mapped coefficient merges."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Coefficients.runner import (
    run_decade_coefficients,
    collect_daily_coefficients,
    collect_period_coefficients,
)


class ResumeAndCollectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        gen = np.random.default_rng(824)
        self.x0 = gen.normal(size=(6, 12, 13))
        self.x1 = 0.4 * self.x0 + gen.normal(size=(6, 12, 13))
        self.mask = np.ones((12, 13), dtype=np.uint8)
        self.mask[0, 0] = 0
        self.kwargs = dict(
            files_path_prefix=str(self.root), array1=self.x0,
            array2=self.x1, mask=self.mask,
            names=('sensible', 'latent'), start_year=1979, end_year=1989,
            start_index=1, n_bins=4, n_bins_2d=3, n_components=2,
            n_init=2, min_group_size=2, run_1d=False, run_2d=True,
            chunk_transitions=2, overwrite=False,
        )
        self.base = (self.root / 'Coefficients' / 'aligned_1979-1989'
                     / '2d_bins_3' / 'sensible_latent')
        self.np_daily = self.base / 'Nonparametric_2d' / 'daily'
        self.sp_daily = self.base / 'Semiparametric_2d' / 'daily'

    def test_restart_begins_at_first_missing_sp_without_refitting_np(self):
        run_decade_coefficients(**self.kwargs)
        old_np = (self.np_daily / 'B_2.npy').stat().st_mtime_ns
        old_sp = (self.sp_daily / 'B_2.npy').stat().st_mtime_ns
        (self.sp_daily / 'B_with_reg_4.npy').unlink()
        with patch('Coefficients.runner.count_2d_nonparam', side_effect=AssertionError('NP called')):
            from Coefficients import runner
            original = runner.count_semi_2d_AB
            with patch('Coefficients.runner.count_semi_2d_AB', wraps=original) as sp:
                run_decade_coefficients(**self.kwargs)
                self.assertGreaterEqual(sp.call_count, 1)
                self.assertEqual(sp.call_args_list[0].args[3], 2)  # previous observation of label 4
        self.assertTrue((self.sp_daily / 'B_with_reg_4.npy').is_file())
        self.assertEqual((self.np_daily / 'B_2.npy').stat().st_mtime_ns, old_np)
        self.assertEqual((self.sp_daily / 'B_2.npy').stat().st_mtime_ns, old_sp)

    def test_malformed_file_is_recomputed(self):
        run_decade_coefficients(**self.kwargs)
        corrupted = self.sp_daily / 'B_3.npy'
        corrupted.write_bytes(b'bad-npy')
        run_decade_coefficients(**self.kwargs)
        self.assertEqual(np.load(corrupted).shape, (4, 12, 13))

    def test_multiple_corrupt_days_repaired_in_one_restart(self):
        run_decade_coefficients(**self.kwargs)
        broken_earlier = self.sp_daily / 'A_3.npy'
        broken_later = self.sp_daily / 'B_4.npy'
        broken_earlier.write_bytes(b'corrupt')
        broken_later.write_bytes(b'corrupt')
        run_decade_coefficients(**self.kwargs)
        self.assertEqual(np.load(broken_earlier).shape, (2, 12, 13))
        self.assertEqual(np.load(broken_later).shape, (4, 12, 13))

    def test_collect_2d_selected_range_and_correct_order(self):
        run_decade_coefficients(**self.kwargs)
        outputs = collect_daily_coefficients(
            str(self.root), path=str(self.base.relative_to(self.root)),
            method='Semiparametric_2d', start_label=3, end_label=5,
        )
        self.assertEqual(set(outputs), {'A', 'B', 'B_with_reg'})
        for coefficient, nch in (('A', 2), ('B', 4), ('B_with_reg', 4)):
            target = outputs[coefficient]
            self.assertEqual(target.parent, self.sp_daily.parent)
            self.assertEqual(target.name, f'{coefficient}_3-5.npy')
            stack = np.load(target, mmap_mode='r')
            self.assertEqual(stack.shape, (3, nch, 12, 13))
            for i, day in enumerate((3, 4, 5)):
                np.testing.assert_array_equal(
                    stack[i], np.load(self.sp_daily / f'{coefficient}_{day}.npy'))
        baseline = outputs['B'].stat().st_mtime_ns
        second = collect_daily_coefficients(
            str(self.root), path=str(self.base.relative_to(self.root)),
            method='Semiparametric_2d', start_label=3, end_label=5,
        )
        self.assertEqual(second['B'].stat().st_mtime_ns, baseline)

    def test_collect_all_enabled_methods(self):
        run_decade_coefficients(**self.kwargs)
        result = collect_period_coefficients(
            str(self.root), names=('sensible','latent'),
            start_year=1979, end_year=1989,
            start_label=2, end_label=6, n_bins_2d=3,
            run_1d=False, run_2d=True,
        )
        self.assertEqual(set(result), {'2d/np', '2d/sp'})
        self.assertEqual(np.load(result['2d/np']['A'], mmap_mode='r').shape, (5,2,12,13))
        self.assertEqual(np.load(result['2d/sp']['B'], mmap_mode='r').shape, (5,4,12,13))

    def test_missing_day_prevents_partial_merge(self):
        run_decade_coefficients(**self.kwargs)
        (self.np_daily / 'A_4.npy').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'A_4.npy'):
            collect_daily_coefficients(
                str(self.root), path=str(self.base.relative_to(self.root)),
                method='Nonparametric_2d', start_label=2, end_label=6,
            )
        self.assertFalse((self.base / 'Nonparametric_2d' / 'B_2-6.npy').exists())
        self.assertFalse((self.base / 'Nonparametric_2d' / 'A_2-6.npy').exists())

    def test_collect_1d_arrays(self):
        from Coefficients.nonparametric import count_1d_nonparam
        count_1d_nonparam(str(self.root), self.x0, 0, 6,
                          path='scalar_case', start_index=1,
                          quantiles_amount=4, mask=self.mask)
        merged = collect_daily_coefficients(
            str(self.root), path='scalar_case', method='Nonparametric',
            start_label=2, end_label=4,
        )
        self.assertEqual(np.load(merged['A'], mmap_mode='r').shape, (3, 12, 13))
        self.assertEqual(np.load(merged['B'], mmap_mode='r').shape, (3, 12, 13))

    def test_all_complete_restart_does_not_invoke_estimators(self):
        run_decade_coefficients(**self.kwargs)
        with patch('Coefficients.runner.count_2d_nonparam', side_effect=AssertionError('NP called')):
            with patch('Coefficients.runner.count_semi_2d_AB', side_effect=AssertionError('SP called')):
                run_decade_coefficients(**self.kwargs)


if __name__ == '__main__':
    unittest.main(verbosity=2)

class GenericRunnerResumeTests(unittest.TestCase):
    def test_direct_2d_runner_resumes_only_missing_sp(self):
        from Coefficients.runner import run_2d_coefficients
        rng = np.random.default_rng(778)
        x0 = rng.normal(size=(6, 9, 11))
        x1 = rng.normal(size=(6, 9, 11))
        with tempfile.TemporaryDirectory() as td:
            kwargs = dict(
                array1=x0, array2=x1, mask=np.ones((9, 11), dtype=int),
                files_path_prefix=td, path='myrun', n_components=2,
                n_init=2, min_group_size=2, quantiles_amount=3,
            )
            run_2d_coefficients(**kwargs)
            daily = Path(td)/'myrun'/'Semiparametric_2d'/'daily'
            (daily/'B_3.npy').unlink()
            with patch('Coefficients.runner.count_2d_nonparam',
                       side_effect=AssertionError('NP refitted')):
                run_2d_coefficients(**kwargs)
            self.assertEqual(np.load(daily/'B_3.npy').shape, (4, 9, 11))
