"""Regression tests for scalar paths, endpoint labels and bounded-memory climatology."""
import contextlib
import io
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import numpy as np

from Data_processing.mean_year import (calculate_mean_year, load_scalar_mean_year_source,
                                       mean_year_paths, save_mean_year)
from Plotting.map_workflows import plot_mean_year


class MeanYearScalarTests(unittest.TestCase):
    def test_streaming_matches_reference(self):
        rng = np.random.default_rng(37)
        dates = [date(1979, 1, 1) + timedelta(days=i) for i in range(732)]
        for shape in ((732,), (732, 2, 3), (732, 4, 2, 3)):
            with self.subTest(shape=shape), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'array.npy'
                arr = rng.normal(size=shape).astype(np.float32)
                arr[0] = np.nan
                arr[20] = np.inf
                arr[424] = -np.inf  # Feb 29, 1980
                mask = np.array([[True, False, True], [True, True, False]]) if len(shape) > 1 else None
                np.save(path, arr)
                data = np.load(path, mmap_mode='r')
                expected = np.full((365, *shape[1:]), np.nan)
                counts = np.zeros(expected.shape, dtype=np.int32)
                for j in range(365):
                    day = date(2001, 1, 1) + timedelta(days=j)
                    indices = [i for i, d in enumerate(dates) if (d.month, d.day) == (day.month, day.day)]
                    vals = arr[indices].astype(float)
                    valid = np.isfinite(vals)
                    if mask is not None:
                        valid &= mask
                    counts[j] = valid.sum(axis=0)
                    np.divide(np.where(valid, vals, 0).sum(axis=0), counts[j],
                              out=expected[j:j+1], where=counts[j:j+1] > 0)
                # Fail if implementation converts/masks the entire source with a new allocation.
                orig_asarray = np.asarray
                orig_where = np.where
                def checked_asarray(a, *args, **kwargs):
                    dtype = args[0] if args else kwargs.get('dtype')
                    if getattr(a, 'shape', None) == shape and dtype is not None:
                        raise AssertionError('Full source dtype conversion')
                    return orig_asarray(a, *args, **kwargs)
                def checked_where(cond, *args):
                    if any(getattr(x, 'shape', None) == shape for x in args):
                        raise AssertionError('Full source masking')
                    return orig_where(cond, *args)
                with patch('Data_processing.mean_year.np.asarray', side_effect=checked_asarray), \
                     patch('Data_processing.mean_year.np.where', side_effect=checked_where):
                    mean, coverage = calculate_mean_year(data, dates, mask=mask, progress=False)
                np.testing.assert_allclose(mean, expected, equal_nan=True)
                np.testing.assert_array_equal(coverage, counts)
                np.testing.assert_array_equal(data, arr)

    def test_scalar_load_save_plot_and_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            years = (1979, 2026)
            for method, folder, variable, bins, value in [('sp','Semiparametric','sensible',144,4.),
                      ('np','Nonparametric','sensible',144,5.),
                      ('sp','Semiparametric','latent',144,6.),
                      ('sp','Semiparametric','sensible',100,7.)]:
                p = root/'Coefficients'/'1979-2026'/f'1d_bins_{bins}'/variable/folder
                p.mkdir(parents=True)
                np.save(p/'A_2-367.npy', np.full((366, 2, 3),value,dtype=np.float32))
                data, dates = load_scalar_mean_year_source(root,variable,'A',*years,
                    method=method,n_bins_1d=bins,spatial_shape=(2,3))
                self.assertIsInstance(data, np.memmap)
                self.assertEqual(dates[0], date(1979,1,2))
                self.assertEqual(dates[-1], date(1980,1,2))
                identity = dict(variable=variable,method=method,n_bins_1d=bins)
                mean,count = save_mean_year(root,'Coefficients_1d','A',data,dates,
                    years=years,progress=False,**identity)
                self.assertEqual(mean.parent, root/'Mean_year'/'Coefficients_1d'/f'1d_bins_{bins}'/variable/method)
                np.testing.assert_equal(np.load(mean)[1], value)
                np.testing.assert_equal(np.load(count)[1], 2)
                self.assertEqual((mean,count),mean_year_paths(root,'Coefficients_1d','A',years=years,**identity))
            for method in ('np','sp'):
                outputs = plot_mean_year(root,'Coefficients_1d','A',*years,
                    variable='sensible',method=method,n_bins_1d=144)
                self.assertEqual(len(outputs),1)
                self.assertTrue(outputs[0].is_file())
                self.assertEqual(outputs[0].parent,root/'Plots'/'Mean Year'/'Coefficients_1d'/'1d_bins_144'/'sensible'/method)
            self.assertEqual(len(list((root/'Mean_year').rglob('A_1979-2026.npy'))),4)

    def test_scalar_validation_and_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root/'Coefficients'/'1979-2026'/'1d_bins_144'/'Flux'/'Semiparametric'
            folder.mkdir(parents=True)
            np.save(folder/'B_with_reg_10-11.npy',np.ones((2,2,3)))
            data, dates = load_scalar_mean_year_source(root,'Flux','B_with_reg',1979,2026,
                          method='sp',n_bins_1d=144,spatial_shape=(2,3))
            with contextlib.redirect_stderr(io.StringIO()) as log:
                calculate_mean_year(data,dates,progress=True,description='scalar progress')
            self.assertIn('scalar progress',log.getvalue())
            self.assertIn('2/2',log.getvalue())
            with self.assertRaises(ValueError):
                load_scalar_mean_year_source(root,'Flux','B_with_reg',1979,2026,
                          method='np',n_bins_1d=144,spatial_shape=(2,3))
            np.save(folder/'B_10-12.npy',np.ones((2,2,3)))
            with self.assertRaisesRegex(ValueError,'Unexpected scalar array'):
                load_scalar_mean_year_source(root,'Flux','B',1979,2026,
                          method='sp',n_bins_1d=144,spatial_shape=(2,3))
            with self.assertRaises(FileNotFoundError):
                load_scalar_mean_year_source(root,'Flux','A',1979,2026,
                          method='sp',n_bins_1d=144,spatial_shape=(2,3))


if __name__ == '__main__':
    unittest.main()
