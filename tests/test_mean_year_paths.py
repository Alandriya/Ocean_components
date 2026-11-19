"""Offline regression tests: pair/method/bin-specific climatology and plots."""
import tempfile
import unittest
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np

from Data_processing.mean_year import mean_year_paths, save_mean_year
from Plotting.map_workflows import plot_mean_year


class MeanYearPathTests(unittest.TestCase):
    def test_isolation_and_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            years = (1979, 2026)
            dates = [date(1979, 1, 1), date(1980, 1, 1)]
            cases = [(('Flux', 'SST'), 'sp', 12),
                     (('Flux', 'SST'), 'np', 12),
                     (('Flux', 'Pressure'), 'sp', 12),
                     (('Flux', 'SST'), 'sp', 24),
                     (('SST', 'Flux'), 'sp', 12)]
            identities = [dict(pair=pair, method=method, n_bins_2d=bins)
                          for pair, method, bins in cases]
            paths = []
            for i, identity in enumerate(identities, 1):
                values = np.full((2, 2, 3, 4), float(i))
                mean, count = save_mean_year(root, 'Coefficients', 'A', values,
                                             dates, years=years, **identity)
                paths.extend([mean, count])
                self.assertEqual((mean, count), mean_year_paths(
                    root, 'Coefficients', 'A', years=years, **identity))
            self.assertEqual(len(set(paths)), 2 * len(cases))
            for i, (mean, count) in enumerate(zip(paths[::2], paths[1::2]), 1):
                np.testing.assert_equal(np.load(mean)[0], float(i))
                np.testing.assert_equal(np.load(count)[0], 2)
            # Real plotting: same component filenames must not collide.
            images = []
            for identity in identities[:2]:
                images.extend(plot_mean_year(root, 'Coefficients', 'A', *years,
                                            day_indices=(0,)*6, **identity))
            self.assertEqual(len(set(images)), 4)
            self.assertTrue(all(p.is_file() and p.stat().st_size > 1000 for p in images))
            self.assertTrue(all(p.parent == root/'Plots'/'Mean Year'/
                                'Coefficients'/'2d_bins_12'/'Flux_SST'/p.parent.name
                                for p in images))
            self.assertFalse((root/'Mean_year'/'Coefficients'/'A_1979-2026.npy').exists())

    def test_variables_derived_and_spectra(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            years = (1979, 2026)
            self.assertEqual(mean_year_paths(root, 'Variables', 'Pressure', years=years)[0],
                             root/'Mean_year'/'Variables'/'Pressure_1979-2026.npy')
            for name in ('C', 'F', 'FS'):
                p, coverage = mean_year_paths(root, 'Derived', name, years=years,
                    pair=('sensible','latent'), method='sp', n_bins_2d=12)
                self.assertEqual(p.parent, root/'Mean_year'/'Derived'/'2d_bins_12'/'sensible_latent'/'sp')
                self.assertEqual(coverage.parent, p.parent)
            for pair in (('Flux','Flux'), ('Flux','SST')):
                p, _ = mean_year_paths(root, 'Eigenvalues', 'spectrum', years=years, pair=pair)
                self.assertEqual(p.parent, root/'Mean_year'/'Eigenvalues'/'-'.join(pair))

    def test_no_silent_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root/'Mean_year'/'Coefficients'/'B_1979-2026.npy'
            old.parent.mkdir(parents=True)
            np.save(old, np.ones((365, 4, 3, 4)))
            with self.assertRaisesRegex(FileNotFoundError, 'Flux_SST'):
                plot_mean_year(root, 'Coefficients', 'B', 1979, 2026,
                               pair=('Flux','SST'), method='sp', n_bins_2d=12)
            with self.assertRaises(ValueError):
                mean_year_paths(root, 'Coefficients', 'A')
            with self.assertRaises(ValueError):
                mean_year_paths(root, 'Derived', 'C', pair=('..','latent'),
                                method='sp', n_bins_2d=12)


if __name__ == '__main__':
    unittest.main()
