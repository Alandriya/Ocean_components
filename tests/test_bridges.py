"""Time-index regression tests for cross-period coefficient transitions."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from Coefficients import bridges
from Data_processing.io import start_label


class BridgeTests(unittest.TestCase):
    """Check that the original estimators receive the missing transition."""

    def test_bridge_index_and_shapes(self):
        """Require a last-to-first transition with no interpolation."""
        prior = (2023, 2024)
        following = (2024, 2025)
        days = 365
        old = np.zeros((days, 2, 2), dtype=float)
        new = np.ones((2, 2, 2), dtype=float)
        calls = []
        def load_field(_, name, first, last):
            return old if first == 2023 else new
        def prepare(first0, last0, first1, last1, bins, mask):
            calls.append((first0.copy(), last0.copy(), first1.copy(), last1.copy(), bins))
            return object()
        common = types.ModuleType('Coefficients.common')
        common.prepare_pair_transition = prepare
        npmod = types.ModuleType('Coefficients.nonparametric')
        npmod.estimate_np_2d = lambda grouping, min_group_size, dt: (
            np.ones((2, 2, 2)), np.ones((4, 2, 2)))
        spmod = types.ModuleType('Coefficients.semiparametric')
        spmod.estimate_sp_2d = lambda grouping, **kwargs: (
            np.ones((2, 2, 2)), np.ones((4, 2, 2)), np.ones((4, 2, 2)), {})
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(sys.modules, {'Coefficients.common': common,
                                         'Coefficients.nonparametric': npmod,
                                         'Coefficients.semiparametric': spmod}), \
                patch.object(bridges, 'load_field', load_field):
            result = bridges.calculate_boundary_transition(
                folder, prior, following, mask=np.ones((2, 2), bool))
            self.assertEqual(result['label'], start_label(2024))
            self.assertEqual(len(calls), 1)
            self.assertTrue(np.array_equal(calls[0][0], old[-1]))
            self.assertTrue(np.array_equal(calls[0][1], new[0]))
            for method in ('np', 'sp'):
                a = np.load(result['paths'][method]['A'])
                b = np.load(result['paths'][method]['B'])
                self.assertEqual(a.shape, (2, 2, 2))
                self.assertEqual(b.shape, (4, 2, 2))

    def test_assembly_includes_every_label_once(self):
        """Join daily coefficient maps and one boundary into a single time axis."""
        periods = ((2023, 2024), (2024, 2025))
        names = ('sensible', 'latent')
        first = start_label(2023)
        second = start_label(2024)
        old = np.zeros((365, 2, 2))
        new = np.zeros((2, 2, 2))
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(bridges, 'load_field', side_effect=lambda root, name, y1, y2:
                             old if y1 == 2023 else new):
            for period in periods:
                labels = range(start_label(period[0]) + 1,
                               start_label(period[0]) + (365 if period[0] == 2023 else 2))
                directory = bridges.period_daily_dir(folder, period, names, 12, 'np')
                directory.mkdir(parents=True)
                for label in labels:
                    np.save(directory / f'A_{label}.npy', np.full((2, 2, 2), label))
                    np.save(directory / f'B_{label}.npy', np.full((4, 2, 2), label))
            bridge = bridges.bridge_daily_dir(folder, names, 12, 'np')
            bridge.mkdir(parents=True)
            np.save(bridge / f'A_{second}.npy', np.full((2, 2, 2), second))
            np.save(bridge / f'B_{second}.npy', np.full((4, 2, 2), second))
            result = bridges.assemble_continuous_coefficients(folder, periods, names,
                                                               methods=('np',))
            a = np.load(result['methods']['np']['A'])
            self.assertEqual(len(a), 366)
            self.assertEqual(result['first_label'], first + 1)
            self.assertEqual(result['last_label'], second + 1)
            self.assertEqual(int(a[364, 0, 0, 0]), second)
            self.assertEqual(int(a[-1, 0, 0, 0]), second + 1)


if __name__ == '__main__':
    unittest.main()
