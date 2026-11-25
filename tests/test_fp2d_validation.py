"""Run from repo root: python -m unittest discover -s tests -p 'test_fp2d_validation.py'."""
import tempfile
import unittest
import numpy as np

from Functional.fp2d_validation import (
    centered_full_tensor_fv_generator,
    run_synthetic,
)


class TestFiniteVolumeOperator(unittest.TestCase):
    def test_constant_state_zero_current_and_conservation(self):
        n = 7
        zeros = np.zeros((n, n))
        ones = np.ones((n, n))
        op = centered_full_tensor_fv_generator(
            zeros, zeros, ones, 1.3 * ones, 0.2 * ones, 0.1, 0.1,
        )
        self.assertLess(np.max(np.abs(op @ np.ones(n * n))), 1e-10)
        self.assertLess(np.max(np.abs(np.asarray(op.sum(axis=0)).ravel())), 1e-10)

    def test_uncorrelated_diffusion_gives_nonnegative_off_diagonal_fv(self):
        n = 7
        zeros = np.zeros((n, n))
        ones = np.ones((n, n))
        op = centered_full_tensor_fv_generator(
            zeros, zeros, ones, ones, zeros, 0.1, 0.1,
        )
        off = op.tolil(copy=True)
        off.setdiag(0)
        off = off.tocsr()
        off.eliminate_zeros()
        self.assertTrue(np.all(off.data >= -1e-12))

    def test_correlated_ou_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = run_synthetic(tmp, sizes=(40, 48), radius=9)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(z["fv_positivity_valid"] for z in rows))
        self.assertTrue(all(z["markov_TV_vs_exact"] < 0.01 for z in rows))
        self.assertLess(rows[-1]["markov_TV_vs_exact"], rows[0]["markov_TV_vs_exact"])
        self.assertLess(rows[-1]["fv_TV_vs_exact"], rows[0]["fv_TV_vs_exact"])


if __name__ == "__main__":
    unittest.main()
