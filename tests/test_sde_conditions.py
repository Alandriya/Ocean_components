import csv
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
from Data_processing.sde_conditions import (
    PairSource, PERIODS, _flat_coefficients, constants_from_bounds,
    estimate_period, pair_key, run_all,
)


class TestSDEConditions(unittest.TestCase):
    def test_scalar_bound_matches_thesis_formula(self):
        c = constants_from_bounds(np.array([-32.02]), np.array([40.77]),
                                  np.array([0.]), np.array([148.02]), .1)
        self.assertAlmostEqual(c['K1'], 2208.1)
        self.assertAlmostEqual(c['K2'], 23572.1133, places=3)
        self.assertEqual(c['K'], c['K2'])

    def test_vector_and_matrix_norms(self):
        c = constants_from_bounds([0, -1], [1, 1], [0,0,0,0], [0,0,3,4], .5)
        self.assertAlmostEqual(c['K1'], (np.sqrt(5)+5)/.5)
        self.assertAlmostEqual(c['K2'], 1+1+9+16)

    def test_first_last_layout_and_diagonal(self):
        last = np.arange(8.).reshape((1,2,2,2))
        first = np.moveaxis(last, (2,3), (0,1))
        np.testing.assert_array_equal(
            _flat_coefficients(last, n=2, spatial_shape=(1,2), layout='last', kind='b'),
            _flat_coefficients(first, n=2, spatial_shape=(1,2), layout='first', kind='b'))
        d = np.array([[[2., 3.], [4., 5.]]])
        diag = _flat_coefficients(d, n=2, spatial_shape=(1,2), layout='last',
                                  kind='b', diffusion='diagonal')
        np.testing.assert_allclose(diag, [[2,0,0,3],[4,0,0,5]])
        flattened = np.arange(8.).reshape(2,4)
        np.testing.assert_array_equal(
            _flat_coefficients(flattened, n=2, spatial_shape=(1,2), layout='last', kind='b'),
            flattened)
        np.testing.assert_array_equal(
            _flat_coefficients(flattened.T, n=2, spatial_shape=(1,2), layout='first', kind='b'),
            flattened)
        with self.assertRaises(ValueError):
            _flat_coefficients(d,n=2,spatial_shape=(1,2),layout='last',kind='b')

    def test_pooled_window_quantiles_and_time_index(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            period = '1979-1989'
            for name in ('sensible', 'latent'):
                d = root / 'DATA' / 'Fluxes'
                d.mkdir(parents=True, exist_ok=True)
                np.save(d / f'{name}_grouped_{period}.npy', np.zeros((2,5)))
            (root/'Coeff_data').mkdir()
            for t in range(1,5):
                a = np.zeros((1,2,2))
                b = np.zeros((1,2,2,2))
                a[0,:,0] = (t,t+2)
                a[0,:,1] = (t+4,t+6)
                b[0,:,0,0] = (t+8,t+10)
                b[0,:,0,1] = (t+12,t+14)
                b[0,:,1,0] = (t+16,t+18)
                b[0,:,1,1] = (t+20,t+22)
                np.save(root/'Coeff_data'/f'A_{t}.npy',a)
                np.save(root/'Coeff_data'/f'B_{t}.npy',b)
            r = estimate_period(root,('sensible','latent'), 1979,1989,
                                PairSource('Coeff_data/A_{index}.npy','Coeff_data/B_{index}.npy'),
                                spatial_shape=(1,2), window_days=3)
            self.assertEqual(r['n_coefficient_days'], 4)
            self.assertEqual(r['n_valid_points'], 8)
            self.assertEqual([w['start_index'] for w in r['windows']], [1,4])
            self.assertEqual([w['end_index_inclusive'] for w in r['windows']], [3,4])
            # In first window a_0 = [1,3,2,4,3,5], 3rd and 97th quantiles.
            expected_low = np.quantile([1.,3.,2.,4.,3.,5.],.03)
            expected_high = np.quantile([1.,3.,2.,4.,3.,5.],.97)
            self.assertAlmostEqual(r['windows'][0]['a_lower'][0],expected_low)
            self.assertAlmostEqual(r['windows'][0]['a_upper'][0],expected_high)
            self.assertEqual(r['first_coefficient_index'],1)
            self.assertEqual(r['last_coefficient_index'],4)
            self.assertEqual(r['observed_last_date'], '1979-01-05')

    def test_end_period_partial_uses_real_length(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            d=root/'DATA'/'SST';d.mkdir(parents=True)
            np.save(d/'SST_2019-2026_grouped.npy',np.ones((2,3)))
            t0=(date(2019,1,1)-date(1979,1,1)).days + 1
            (root/'Coeff_data'/'SST-SST').mkdir(parents=True)
            for t in (t0,t0+1):
                np.save(root/'Coeff_data'/'SST-SST'/f'A_{t}.npy',np.ones((1,2))*t)
                np.save(root/'Coeff_data'/'SST-SST'/f'B_{t}.npy',np.ones((1,2)))
            result=estimate_period(root,('SST','SST'),2019,2026,
                                   PairSource('Coeff_data/{pair}/A_{index}.npy',
                                              'Coeff_data/{pair}/B_{index}.npy'),
                                   spatial_shape=(1,2))
            self.assertEqual(result['first_coefficient_index'],t0)
            self.assertEqual(result['last_coefficient_index'],t0+1)
            self.assertEqual(result['n_coefficient_days'],2)

    def test_batch_outputs_explicit_missing(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            d=root/'DATA'/'Fluxes'; d.mkdir(parents=True)
            np.save(d/'sensible_grouped_1979-1989.npy',np.zeros((2,2)))
            np.save(d/'latent_grouped_1979-1989.npy',np.zeros((2,2)))
            rs=run_all(root,selected_pairs=[('sensible','latent')],periods=[(1979,1989)],
                       spatial_shape=(1,2))
            self.assertEqual(rs[0]['status'],'missing_input')
            file=root/'Conditions'/'sde_constants.csv'
            self.assertTrue(file.exists())
            self.assertTrue((root/'Conditions'/'details'/'sensible-latent'/'1979-1989.json').exists())

    def test_all_five_periods_with_configured_named_pair(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for start, end in PERIODS:
                period = f"{start}-{end}"
                for name in ("sensible", "SST"):
                    folder = root / "DATA" / ("Fluxes" if name == "sensible" else "SST")
                    folder.mkdir(parents=True, exist_ok=True)
                    filename = (f"sensible_grouped_{period}.npy" if name == "sensible"
                                else f"SST_{period}_grouped.npy")
                    np.save(folder / filename, np.ones((2, 2)))
                first = (date(start, 1, 1) - date(1979, 1, 1)).days + 1
                folder = root / "custom" / period / "sensible-SST"
                folder.mkdir(parents=True, exist_ok=True)
                np.save(folder / f"A_{first}.npy", np.array([[[1., -2.], [2., -3.]]]))
                b = np.zeros((1, 2, 2, 2))
                b[..., 0, 0] = 1.
                b[..., 1, 1] = 2.
                np.save(folder / f"B_{first}.npy", b)
            results = run_all(
                root, variables=("sensible", "SST"),
                selected_pairs=[("sensible", "SST")],
                coefficient_sources={
                    "sensible-SST": {
                        "a_pattern": "custom/{period}/{pair}/A_{index}.npy",
                        "b_pattern": "custom/{period}/{pair}/B_{index}.npy",
                    }
                },
                spatial_shape=(1,2),
            )
            self.assertEqual(len(results), 5)
            self.assertEqual([r["period"] for r in results],
                             [f"{a}-{b}" for a,b in PERIODS])
            self.assertTrue(all(r["status"] == "ok" for r in results))
            self.assertEqual(len((root / "Conditions" / "sde_constants.csv").read_text().splitlines()), 6)

    def test_default_source_is_overridable_without_json(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            field = root / "DATA" / "SST"
            field.mkdir(parents=True)
            np.save(field / "SST_2019-2026_grouped.npy", np.ones((2,2)))
            offset = (date(2019,1,1) - date(1979,1,1)).days + 1
            src = root / "custom"
            src.mkdir()
            np.save(src / f"A_{offset}.npy", np.ones((1,2)))
            np.save(src / f"B_{offset}.npy", 2*np.ones((1,2)))
            result = run_all(
                root, variables=("SST",),
                coefficient_sources={"default": {
                    "a_pattern": "custom/A_{index}.npy",
                    "b_pattern": "custom/B_{index}.npy",
                }}, periods=[(2019,2026)], spatial_shape=(1,2),
            )
            self.assertEqual(result[0]["status"], "ok")
            self.assertAlmostEqual(result[0]["K2"], 5.0)


if __name__ == '__main__':
    unittest.main()
