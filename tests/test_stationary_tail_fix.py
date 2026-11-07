"""Regression tests for the article-defined stationary plotting tails."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / 'Functional' / 'stationary.py'
spec = importlib.util.spec_from_file_location('stationary_tail_fix', MODULE_PATH)
stationary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stationary)


def fake_formula(unstable_tail="left"):
    """Return a formula with unstable extrapolation outside fit support."""
    fake = types.ModuleType('Functional.func_estimation')
    fake.I = lambda x, args: (
        500.0 * (abs(x) / 1000.0) ** 4
        if (x < 0 if unstable_tail == "left" else x > 0)
        else -50.0 * (abs(x) / 1000.0) ** 2
    )
    fake.log_b2 = lambda x, args: 0.012 * abs(x)
    fake.count_constants = lambda params: [0.0] * 10
    fake.Phi = lambda x, c, k: x
    fake.Psi_minus = lambda x, c, k: x
    fake.Psi_plus = lambda x, c, k: x
    return fake


def parameters(xmin=-250.0, xmax=150.0, x0=-10.0):
    """Construct a ten-element fit parameter record for the tests."""
    return [x0, -40.0, np.zeros(5), xmin, xmax, 0.1, 0.2, 0.3, 0.4, 1]


class TailContinuationTests(unittest.TestCase):
    """Compare the clipped drift integral with the old, unstable extension."""

    def setUp(self):
        self.grid = np.linspace(-2000.0, 1500.0, 3501)

    def test_latent_left_tail_continues_to_zero(self):
        """The far-left latent tail must not absorb the probability mass."""
        with patch.dict(sys.modules, {'Functional.func_estimation': fake_formula()}):
            density = stationary._stationary_density(self.grid, parameters(), 'latent')
        self.assertAlmostEqual(float(np.trapezoid(density, self.grid)), 1.0, places=9)
        mass = stationary._window_probability_mass(self.grid, density, -800.0, 300.0)
        self.assertGreater(mass, 0.98)
        self.assertLess(float(density[0]), float(density[np.searchsorted(self.grid, -500.0)]))
        self.assertGreater(float(np.max(density)), 0.0)

    def test_sensible_right_tail_continues_to_zero(self):
        """The far-right sensible tail must not form a second density mode."""
        with patch.dict(sys.modules, {'Functional.func_estimation': fake_formula('right')}):
            density = stationary._stationary_density(self.grid, parameters(), 'sensible')
        self.assertAlmostEqual(float(np.trapezoid(density, self.grid)), 1.0, places=9)
        self.assertLess(float(density[-1]), float(density[np.searchsorted(self.grid, 500.0)]))

    def test_diffusion_logarithm_uses_unclipped_x(self):
        """Only the I argument is capped; log_b2 sees the original coordinate."""
        seen_i, seen_log_b = [], []
        fake = fake_formula()
        fake.I = lambda x, args: seen_i.append(x) or 0.0
        fake.log_b2 = lambda x, args: seen_log_b.append(x) or abs(x) * 0.012
        grid = np.array([-1000.0, -200.0, 0.0, 100.0])
        with patch.dict(sys.modules, {'Functional.func_estimation': fake}):
            stationary._stationary_density(grid, parameters(), 'latent')
        self.assertEqual(seen_i, [-250.0, -200.0, 0.0, 100.0])
        self.assertEqual(seen_log_b, [-1000.0, -200.0, 0.0, 100.0])

    def test_positive_x0_has_separate_constant_formula(self):
        """The x0>0 branch must use Psi_plus at its right junction."""
        fake = fake_formula()
        fake.count_constants = lambda params: (_ for _ in ()).throw(AssertionError('wrong branch'))
        with patch.dict(sys.modules, {'Functional.func_estimation': fake}):
            constants = stationary._stationary_integration_constants(parameters(x0=5.0))
        self.assertEqual(len(constants), 4)
        self.assertTrue(np.isfinite(constants).all())

    def test_outside_window_diagnostic(self):
        """Probability fraction is measured on the true integration grid."""
        grid = np.linspace(-1, 1, 21)
        p = np.ones_like(grid) * 0.5
        self.assertAlmostEqual(stationary._window_probability_mass(grid, p, -0.5, 0.5), 0.5)

    def test_numeric_workflow_prints_and_saves_tail_diagnostics(self):
        """A complete fit preserves the previous console and JSON behavior."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / 'Functional' / 'sensible-latent'
            data.mkdir(parents=True)
            first, last = 1, 3653
            x = np.linspace(-200.0, 80.0, 10)
            np.save(data / f'{first}-{last}_quantiles1.npy', x)
            np.save(data / f'{first}-{last}_quantiles2.npy', x)
            np.save(data / f'{first}-{last}_a_grouped.npy',
                    np.stack((x * -0.01, x * -0.02)))
            np.save(data / f'{first}-{last}_b_grouped_log_simple.npy',
                    np.stack((0.03 * abs(x), 0.04 * abs(x))))

            fake_io = types.ModuleType('Data_processing.io')
            fake_io.start_label = lambda year: 1
            settings = {'x0': -10.0, 'x1': -40.0,
                        'c1': 0.1, 'c2': 0.2, 'c3': 0.3,
                        'c4': 0.4, 'rmse_log_b': 0.01}
            printed = io.StringIO()
            with patch.dict(sys.modules, {
                'Functional.func_estimation': fake_formula(),
                'Data_processing.io': fake_io,
            }), patch.object(stationary, '_fit_log_diffusion', return_value=settings), \
                    contextlib.redirect_stdout(printed):
                result = stationary.estimate_stationary_characteristics(
                    root, 1979, 1989, grid_points=101)

            self.assertTrue(result.exists())
            with np.load(result) as recorded:
                self.assertEqual(set(recorded.files), {'grid', 'pdf_sensible', 'pdf_latent'})
                for name in ('sensible', 'latent'):
                    self.assertAlmostEqual(float(np.trapezoid(recorded[f'pdf_{name}'],
                                                              recorded['grid'])), 1.0, places=9)
            meta = json.loads(result.with_suffix('.json').read_text())
            for name in ('sensible', 'latent'):
                self.assertIn('probability_mass_in_plot_window', meta['fits'][name])
                self.assertIn(f'Stationary parameters for {name}', printed.getvalue())


if __name__ == '__main__':
    unittest.main(verbosity=2)
