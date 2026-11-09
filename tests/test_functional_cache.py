"""Tests of cache-only plotting and stationary fit consistency checks."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'isolated_functional_fit_plots', ROOT / 'Plotting' / 'functional_fit_plots.py'
)
plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plots)


class CacheTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.original = {key: sys.modules.get(key) for key in
                         ('Data_processing', 'Data_processing.io',
                          'Functional', 'Functional.stationary')}
        dp = types.ModuleType('Data_processing')
        io = types.ModuleType('Data_processing.io')
        io.start_label = lambda year: 1
        fn = types.ModuleType('Functional')
        stationary = types.ModuleType('Functional.stationary')
        stationary._load_historical_grouped = lambda *args, **kwargs: None
        stationary._group_full_period = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError('Heavy regrouping must not run in normal plotting')
        )
        def fit(x, y, name):
            return {'x0': -3., 'x1': -50. if name=='sensible' else -40.,
                    'c1': .1, 'c2': .2, 'c3': 6., 'c4': .25, 'rmse_log_b': .04}
        stationary._fit_log_diffusion = fit
        sys.modules.update({
            'Data_processing': dp, 'Data_processing.io': io,
            'Functional': fn, 'Functional.stationary': stationary,
        })
        self.folder = self.root / 'Functional' / 'sensible-latent'
        self.folder.mkdir(parents=True)
        self.cache = self.folder / 'grouped_1979-2026_sp_2-17470.npz'
        self.args = {'quantile_bins':260, 'trim_bins':5, 'chunk_days':16,
                     'spatial_block_size':5, 'sample_size':400000}
        config = {'bins':260, 'trim_bins':5, 'chunk_days':16,
                  'block_size':5, 'sample_size':400000}
        self.cache.with_suffix('.json').write_text(json.dumps({'config':config}))
        x = np.linspace(-200, 20, 24)
        store = {}
        for n in ('sensible', 'latent'):
            store[f'quantiles_{n}']=x
            store[f'drift_{n}']=-.1*x
            store[f'log_variance_{n}']=np.ones_like(x)*8.0
        np.savez_compressed(self.cache, **store)

    def tearDown(self):
        for key, old in self.original.items():
            if old is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = old
        self.temp.cleanup()

    def read(self, **overrides):
        kw = {**self.args, **overrides}
        return plots._historical_grouped_inputs(
            self.root, ('sensible','latent'), 1979, 2026, 'sp', 'simple',
            kw['quantile_bins'], kw['trim_bins'], kw['chunk_days'],
            kw['spatial_block_size'], kw['sample_size'],
            kw.get('refresh_grouped', False)
        )

    def test_loads_small_cache_without_large_data(self):
        loaded = self.read()
        self.assertEqual(len(loaded['sensible'][0]),24)
        self.assertEqual(len(loaded['latent'][0]),24)

    def test_wrong_settings_fail_fast(self):
        with self.assertRaisesRegex(ValueError, 'bins: saved=260, requested=100'):
            self.read(quantile_bins=100)

    def test_missing_cache_fails_fast(self):
        self.cache.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "task='stationary'"):
            self.read()

    def test_consistency_verification_success(self):
        target = self.root / 'Functional' / 'Stationary'
        target.mkdir()
        expected = {}
        actual = {}
        for n in ('sensible','latent'):
            fit = {'x0':-3., 'x1':-50. if n=='sensible' else -40.,
                   'c1':.1,'c2':.2,'c3':6.,'c4':.25,
                   'drift_polynomial_ascending':[1.,2.,3.,4.,5.]}
            expected[n] = fit
            actual[n] = dict(fit)
        (target / 'sensible-latent_1979-2026.json').write_text(
            json.dumps({'method':'sp','provenance':{'input_type':'assembled_coefficients'},
                        'fits':expected}))
        self.assertTrue(plots._check_against_saved_stationary(
            self.root, ('sensible','latent'), 1979,2026,'sp',actual))

    def test_old_density_triggers_warning(self):
        target = self.root / 'Functional' / 'Stationary'
        target.mkdir()
        expected = {}
        actual = {}
        for n in ('sensible','latent'):
            fit = {'x0':-3., 'x1':-40., 'c1':.1,'c2':.2,'c3':6.,'c4':.25,
                   'drift_polynomial_ascending':[1.,2.,3.,4.,5.]}
            expected[n] = fit
            actual[n] = dict(fit, x1=-50. if n=='sensible' else -40.)
        (target / 'sensible-latent_1979-2026.json').write_text(
            json.dumps({'method':'sp','provenance':{'input_type':'assembled_coefficients'},
                        'fits':expected}))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.assertFalse(plots._check_against_saved_stationary(
                self.root, ('sensible','latent'), 1979,2026,'sp',actual))
        self.assertTrue(any('sensible/x1' in str(z.message) for z in w))

    def test_plot_draws_using_saved_cache(self):
        # No stationary JSON => informative note, not an expensive scan.
        import matplotlib
        matplotlib.use('Agg')
        result=plots.plot_functional_fits(self.root,1979,2026,**{
            'names': ('sensible','latent'), 'method':'sp',
            **self.args, 'verify_stationary':False,
        })
        self.assertTrue(result.exists())

    def test_plotter_passes_all_settings(self):
        source=(ROOT / 'main_plotter.py').read_text()
        tree=ast.parse(source)
        fns=[x for x in ast.walk(tree) if isinstance(x,ast.Call)
             and isinstance(x.func,ast.Name) and x.func.id=='plot_functional_fits']
        self.assertEqual(len(fns),1)
        keys={x.arg for x in fns[0].keywords}
        self.assertTrue({'quantile_bins','trim_bins','chunk_days',
                         'spatial_block_size','sample_size','refresh_grouped',
                         'verify_stationary'} <= keys)


if __name__=='__main__':
    unittest.main()
