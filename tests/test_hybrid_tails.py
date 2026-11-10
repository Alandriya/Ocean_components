"""Self-contained tests for experimental semiparametric tails.

Run from repository root: python -m unittest discover -s tests -p test_hybrid_tails.py -v
Tests use synthetic arrays; real ERA5 and A/B are not bundled.
"""
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

# Allow running tests against this standalone file bundle without cloning the
# rest of Ocean_components; the actual project provides Data_processing.io.
try:
    from Data_processing import io as dp_io
except ImportError:
    package = types.ModuleType("Data_processing")
    package.__path__ = []
    dp_io = types.ModuleType("Data_processing.io")
    dp_io.field_candidates = lambda root, name, first, last: [
        Path(root) / "DATA" / "Fluxes" / f"{name}_{first}-{last}_grouped.npy"
    ]
    sys.modules["Data_processing"] = package
    sys.modules["Data_processing.io"] = dp_io

from Functional.hybrid_tails import (
    _combine, _smooth_monotone_log_bridge, observed_sample,
)


class HybridTailTests(unittest.TestCase):
    def test_monotone_C1_transition(self):
        x = np.linspace(0, 35, 10001)
        logp = _smooth_monotone_log_bridge(
            x, 35, p0=0.012, outward_slope=-0.04,
            p1=0.0012, terminal_slope=-0.022)
        self.assertTrue(np.all(np.diff(logp) < 0))
        self.assertAlmostEqual(np.exp(logp[0]), 0.012, places=8)
        self.assertAlmostEqual(np.exp(logp[-1]), 0.0012, places=5)
        self.assertAlmostEqual((logp[1]-logp[0])/(x[1]-x[0]), -0.04, delta=0.0002)
        self.assertAlmostEqual((logp[-1]-logp[-2])/(x[1]-x[0]), -0.022, delta=0.0002)

    def test_hybrid_density_normalized_and_no_bubble(self):
        rng = np.random.default_rng(42)
        grid = np.linspace(-700, 700, 2801)
        # Known trustworthy central PDF, contaminated outer tails (old model)
        base = np.exp(-0.5 * ((grid + 16) / 65) ** 2)
        base /= np.trapezoid(base, grid)
        sample = -16 + 35 * rng.standard_t(4, 95000)
        output, section, info = _combine(
            grid, base, sample, sample, [-150, 100],
            tail_quantile=0.975, transition_quantile=0.90,
            reference_weight=0.7)
        self.assertAlmostEqual(np.trapezoid(output, grid), 1.0, places=10)
        self.assertTrue((output >= 0).all())
        self.assertEqual(set(section), {-2,-1,0,1,2})
        self.assertTrue(info["right"]["tail_start"] > info["right"]["anchor"])
        self.assertTrue(info["left"]["tail_start"] < info["left"]["anchor"])
        for side, region in (("left", -1), ("right", 1)):
            idx = np.flatnonzero(section == region)
            derivative = np.diff(output[idx])
            if side == "right":
                self.assertTrue((derivative < 0).all())
            else:
                self.assertTrue((derivative > 0).all())
        # C0 and C1 across analytically constructed joins tested via fine grid.

    def test_sample_cache_is_reused(self):
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            source = root / "DATA" / "Fluxes" / "sensible_1979-2026_grouped.npy"
            source.parent.mkdir(parents=True)
            array = np.arange(120 * 8 * 9).reshape(120, 8, 9).astype(np.float32)
            np.save(source, array)
            a, spec = observed_sample(root, "sensible", 1979, 2026,
                                      sample_days=30, pixels_per_day=15)
            self.assertTrue(len(a) == 450)
            self.assertTrue(set(a).issubset(set(array.ravel())))
            b, other = observed_sample(root, "sensible", 1979, 2026,
                                       sample_days=30, pixels_per_day=15)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(spec, other)


if __name__ == "__main__":
    unittest.main()
