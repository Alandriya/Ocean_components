"""Fast unit tests for the FP-Markov baseline comparison (no real ocean data)."""
import unittest
import numpy as np

from Forecast_nn.fp_markov_baselines import (
    ar1_forecast, evaluate, fit_ar1, trailing_mean,
)


class FPBaselineTests(unittest.TestCase):
    """Regression tests covering FPBaseline."""
    def test_trailing_means_use_no_future(self):
        """Verify that trailing means use no future."""
        blocks = np.arange(40., dtype=float).reshape(40, 1, 1, 1)
        blocks = np.repeat(blocks, 2, axis=-1)
        origins = np.array([29, 30])
        before7 = trailing_mean(blocks, origins, 7)
        before30 = trailing_mean(blocks, origins, 30)
        self.assertAlmostEqual(before7[0, 0, 0, 0], 26.)
        self.assertAlmostEqual(before30[0, 0, 0, 1], 14.5)
        blocks[32:] = 999999
        np.testing.assert_array_equal(before7, trailing_mean(blocks, origins, 7))
        np.testing.assert_array_equal(before30, trailing_mean(blocks, origins, 30))

    def test_ar1_predict_uses_recursion(self):
        """Verify that AR(1) predict uses recursion."""
        x = np.array([[[3., 4.]]])
        intercept = np.array([1., -2.])
        phi = np.array([.5, -.25])
        result = ar1_forecast(x, intercept, phi, 3)
        correct = x.copy()
        for _ in range(3):
            correct = intercept + phi * correct
        np.testing.assert_allclose(result, correct)

    def test_ar1_fit_local_and_fallback(self):
        """Verify that AR(1) fit local and fallback."""
        rng = np.random.default_rng(23)
        blocks = np.zeros((140, 2, 2, 2))
        blocks[0] = rng.normal(size=(2, 2, 2))
        for t in range(1, 140):
            blocks[t] = np.array([2., -3.]) + np.array([.6, .7])*blocks[t-1] + rng.normal(0, .8, (2, 2, 2))
        blocks[:, 0, 0, :] = np.nan
        fitted = fit_ar1(blocks, min_pairs=60)
        self.assertTrue(np.isfinite(fitted['local_phi']).all())
        self.assertTrue(np.isfinite(fitted['local_intercept']).all())
        self.assertEqual(fitted['n_local_ar1_fit'], 6)
        np.testing.assert_allclose(fitted['local_phi'][0, 0], fitted['global_phi'])
        np.testing.assert_allclose(fitted['local_intercept'][0, 0], fitted['global_intercept'])

    def test_equal_mask_for_all_candidates(self):
        """Verify that equal mask for all candidates."""
        lookup = {
            'domain_mask': np.array([[True]]),
            'x_edges': np.array([-2., 2.]),
            'y_edges': np.array([-2., 2.]),
            'x_centers': np.array([0.]),
            'y_centers': np.array([0.]),
            'mean': np.zeros((4, 1, 2)),
            'stationary_mean': np.array([0., 0.]),
        }
        rng = np.random.default_rng(99)
        blocks = rng.normal(0, .3, size=(85, 1, 1, 2))
        fit = fit_ar1(blocks[:45], min_pairs=25)
        report = evaluate(blocks, np.arange(45, 82), lookup, fit, bootstrap_reps=10)
        for horizon in ('1', '2', '3'):
            self.assertEqual(report[horizon]['n_valid_block_forecasts'], 37)
            for score in report[horizon]['models'].values():
                self.assertTrue(np.isfinite(score['rmse_joint']))
                self.assertIn('mse_skill_ci95_low', score)
            self.assertAlmostEqual(report[horizon]['models']['persistence']['mse_skill_vs_persistence'], 0.)


if __name__ == '__main__':
    unittest.main()
