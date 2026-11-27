"""Run with: python -m unittest discover -s tests -p 'test_fp_followup_validation.py'"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import genpareto

from Functional.hybrid_tails import _smooth_monotone_log_bridge
from Functional.fp_followup_validation import (
    _bridge_replay_parameters, diagnose_hybrid_cache, run_wide_ou,
)


class FollowupValidationTests(unittest.TestCase):
    """Regression tests covering Followup Validation."""
    def test_independent_bridge_quadrature(self):
        """Verify that independent bridge quadrature."""
        values = _bridge_replay_parameters(.4, .6, .27, -1., -1.1)
        self.assertLess(values["independent_integral_residual"], 1e-5)
        self.assertLess(values["root_equation_residual"], 1e-9)

    def test_saved_hybrid_grid_replay(self):
        """Verify that saved hybrid grid replay."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stationary_dir = root / 'Functional' / 'Stationary'
            hybrid_dir = stationary_dir / 'Hybrid'
            hybrid_dir.mkdir(parents=True)
            x = np.linspace(-2., 2., 401)
            core = np.exp(-.5 * x * x)
            lograw = np.log(core).copy()
            section = np.zeros(x.shape, dtype=np.int8)
            diag = {}
            for side, sign in (("left", -1), ("right", 1)):
                anchor = float(sign * 1.)
                threshold = float(sign * 1.4)
                length = sign * (threshold - anchor)
                p0 = float(np.exp(-.5))
                p1 = .27
                k0, k1 = -1., -1.1
                d = sign * (x - anchor)
                bridge = (d > 0) & (d < length)
                tail = d >= length
                lograw[bridge] = _smooth_monotone_log_bridge(
                    d[bridge], length, p0, k0, p1, k1
                )
                lograw[tail] = (np.log(p1) +
                                genpareto.logpdf(d[tail] - length,
                                                c=.1, loc=0., scale=1.))
                section[bridge] = sign
                section[tail] = 2 * sign
                diag[side] = {
                    "anchor": anchor, "tail_start": threshold,
                    "core_outward_log_slope": k0,
                    "used_tail_density_at_threshold": p1,
                    "shape": .1, "scale": 1.,
                    "amplitude_ratio_to_empirical": .8,
                }
            dens = np.exp(lograw - np.max(lograw))
            dens /= np.trapezoid(dens, x)
            base = stationary_dir / 'sensible-latent_1979-2026.npz'
            np.savez_compressed(base, grid=x, pdf_sensible=core)
            saved = hybrid_dir / 'sensible_1979-2026_gpd_reference_1979-2026.npz'
            np.savez_compressed(saved, grid=x, pdf=dens, section=section)
            saved.with_suffix('.json').write_text(json.dumps({'diagnostics': diag}), encoding='utf-8')
            summary, rows = diagnose_hybrid_cache(root, 'sensible', 1979, 2026)
            self.assertEqual(len(rows), 2)
            self.assertLess(summary['maximum_absolute_saved_vs_replay_density'], 1e-10)
            self.assertEqual(summary['monotonicity_violation_pairs'], 0)
            self.assertLess(summary['maximum_independent_integral_residual'], 1e-5)

    def test_ou_is_actually_wide(self):
        """Verify that ou is actually wide."""
        with tempfile.TemporaryDirectory() as tmp:
            rows = run_wide_ou(Path(tmp), sizes=(40,), signs=(1,), max_stencil_radius=9)
            self.assertEqual(len(rows), 1)
            self.assertGreaterEqual(rows[0]['max_stencil_radius'], 2)
            self.assertGreater(rows[0]['fraction_radius_ge_2'], 0.0)
            self.assertLess(rows[0]['ctmc_TV_vs_exact'], .1)


if __name__ == '__main__':
    unittest.main()
