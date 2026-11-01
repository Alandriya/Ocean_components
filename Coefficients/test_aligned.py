"""Небольшие воспроизводимые тесты общего разбиения и двух алгоритмов.

Запуск: python test_aligned.py
"""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from aligned_common import prepare_transition
from nonparametric_aligned import estimate_np, count_1d_nonparam
from semiparametric_aligned import estimate_sp, count_semi_1d_AB
from compare_aligned import run_comparison


def demo_data(seed=2026):
    rng = np.random.default_rng(seed)
    T, H, W = 5, 8, 11
    arr = np.zeros((T, H, W), dtype=float)
    arr[0] = rng.standard_normal((H, W))
    for t in range(1, T):
        # Дрейф нелинейный, шум неоднороден: распределение не обязано быть нормальным.
        eps = np.where(rng.random((H, W)) < 0.2,
                       rng.normal(0, 0.45, (H, W)),
                       rng.normal(0, 0.10, (H, W)))
        arr[t] = arr[t-1] + (-0.1*arr[t-1] + 0.15*np.sin(arr[t-1])) + eps
    arr[:, 0, 0] = np.nan
    return arr


class AlignedTests(unittest.TestCase):
    def test_exact_same_input_moments(self):
        data = demo_data()
        for N in (4, 12, 40, 100):
            for t in (1, 2, 3):
                group = prepare_transition(data[t-1], data[t], N)
                an, bn = estimate_np(group)
                ag, bg, bg_reg, info = estimate_sp(group, n_init=2, random_state=17)
                assert np.array_equal(np.isfinite(an), np.isfinite(ag))
                assert np.array_equal(np.isfinite(bn), np.isfinite(bg))
                valid = np.isfinite(an)
                if valid.any():
                    np.testing.assert_allclose(an[valid], ag[valid], rtol=1e-8, atol=1e-10)
                    np.testing.assert_allclose(bn[valid], bg[valid], rtol=1e-7, atol=1e-8)
                    np.testing.assert_allclose(bg_reg[valid]**2, bn[valid]**2 + 1e-6,
                                               rtol=1e-7, atol=1e-9)
                assert info['non_converged_bins'] == 0

    def test_nan_mask_and_minimum_group(self):
        x = np.array([[1., 2., 3.], [4., 5., np.nan]])
        y = x + 1.
        excluded = np.zeros_like(x, bool)
        excluded[1, 1] = True
        g = prepare_transition(x, y, quantiles_amount=10, mask=excluded)
        self.assertEqual(g.n_valid_pairs, 4)
        self.assertEqual(sum(len(z.increments) for z in g.groups), 4)
        an, bn = estimate_np(g, min_group_size=2)
        self.assertTrue(np.isnan(an[1, 1]) and np.isnan(bn[1, 2]))

    def test_wrappers_save_all_days(self):
        data = demo_data()
        with tempfile.TemporaryDirectory() as d:
            count_1d_nonparam(d, data, 0, 4, quantiles_amount=4, overwrite=True)
            count_semi_1d_AB(d, data, 0, 4, quantiles_amount=4, n_init=2, overwrite=True)
            for t in (1, 2, 3):
                p = Path(d)/'Synthetic'/'Semi_aligned'/'daily'/f'B_{t}.npy'
                self.assertTrue(p.exists(), str(p))
                a = np.load(Path(d)/'Synthetic'/'Bel_aligned'/'daily'/f'A_{t}.npy')
                b = np.load(Path(d)/'Synthetic'/'Semi_aligned'/'daily'/f'A_{t}.npy')
                np.testing.assert_allclose(a, b, atol=1e-10, equal_nan=True)

    def test_runner_outputs(self):
        data = demo_data()
        with tempfile.TemporaryDirectory() as d:
            _, summary = run_comparison(data, Path(d), (4, 12, 60), 0, 3, n_init=2, save_maps=False)
            self.assertTrue((Path(d)/'summary.csv').exists())
            self.assertEqual(len(summary), 3)
            for x in summary:
                if x['n_total_comparable']:
                    self.assertLess(x['max_abs_a'], 1e-9)
                    self.assertLess(x['max_abs_b'], 1e-7)


if __name__ == '__main__':
    unittest.main(verbosity=2)
