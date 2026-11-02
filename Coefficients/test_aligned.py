"""Regression tests for I/O, water masks, 1D/2D estimators, and C/F.

Run: python -m unittest -v test_aligned.py
"""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from aligned_common import (
    load_water_mask, load_series, load_pair_series, prepare_transition,
    prepare_pair_transition, unpack_b_matrix, covariance_to_diffusion,
)
from nonparametric_aligned import (
    estimate_np, estimate_np_2d, count_1d_nonparam, count_2d_nonparam,
)
from semiparametric_aligned import (
    estimate_sp, estimate_sp_2d, count_semi_1d_AB, count_semi_2d_AB,
)
from coefficients_cf import (
    compute_c, compute_f, compute_fs, count_cf_from_daily,
    count_c_coeff, count_fraction, count_f_separate_coeff,
)


def demo_pair(seed=12, shape=(9, 10), steps=5):
    rng = np.random.default_rng(seed)
    first = np.zeros((steps, *shape))
    second = np.zeros((steps, *shape))
    first[0] = rng.normal(size=shape)
    second[0] = rng.normal(size=shape)
    for t in range(1, steps):
        d0 = 0.20 - 0.1 * first[t-1] + rng.normal(scale=.2, size=shape)
        d1 = -0.15 + 0.1 * second[t-1] + .5 * d0 + rng.normal(scale=.15, size=shape)
        first[t] = first[t-1] + d0
        second[t] = second[t-1] + d1
    return first, second


class AlignedTests(unittest.TestCase):
    def test_mask_water_one_land_zero(self):
        mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        water = load_water_mask(mask, (2, 2))
        np.testing.assert_array_equal(water, [[True, False], [False, True]])
        with self.assertRaises(ValueError):
            load_water_mask(np.array([[1, 2], [0, 1]]), (2, 2))
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / 'mask'
            mask.tofile(raw)
            np.testing.assert_array_equal(load_water_mask(raw, (2, 2)), water)

    def test_input_orientations(self):
        t, h, w = 4, 161, 181
        data = np.arange(t*h*w, dtype=np.float32).reshape((t,h,w))
        flat_spatial_first = data.reshape((t,-1)).T
        flat_time_first = data.reshape((t,-1))
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name, value in [('a',data),('b',flat_spatial_first),('c',flat_time_first)]:
                p=Path(directory)/f'{name}.npy'
                np.save(p, value)
                paths.append(p)
            for p in paths:
                loaded = load_series(p)
                self.assertEqual(loaded.shape, data.shape)
                np.testing.assert_array_equal(loaded[2], data[2])
            first, second = load_pair_series(paths[0], paths[1])
            np.testing.assert_array_equal(first[0],second[0])
            with self.assertRaises(ValueError):
                load_series(np.zeros((7,12)))

    def test_scalar_np_sp_agreement(self):
        first, _ = demo_pair()
        group = prepare_transition(first[0], first[1], 4)
        an, bn = estimate_np(group)
        ag, bg, bgr, diag = estimate_sp(group, n_init=2)
        np.testing.assert_allclose(an, ag, atol=1e-11, equal_nan=True)
        np.testing.assert_allclose(bn, bg, atol=1e-9, equal_nan=True)
        self.assertEqual(diag['non_converged_bins'],0)

    def test_2d_covariance_and_channel_order(self):
        first, second = demo_pair()
        mask = np.ones(first.shape[1:], dtype=np.uint8)
        mask[0,0] = 0
        grouped = prepare_pair_transition(first[0],first[1],second[0],second[1],3,mask)
        an, bn = estimate_np_2d(grouped)
        ag, bg, br, diag = estimate_sp_2d(grouped, n_init=2)
        self.assertEqual(an.shape,(2,9,10))
        self.assertEqual(bn.shape,(4,9,10))
        np.testing.assert_allclose(an,ag,atol=1e-10,equal_nan=True)
        np.testing.assert_allclose(bn,bg,atol=1e-7,equal_nan=True)
        self.assertTrue(np.isnan(bn[:,0,0]).all())
        self.assertTrue(np.isnan(an[:,0,0]).all())
        self.assertEqual(diag['non_converged_bins'],0)
        for g in grouped.groups:
            if len(g.increments)<2:
                continue
            m = np.mean(g.increments,axis=0)
            d = g.increments-m
            cov = d.T@d/len(d)
            pos = np.unravel_index(g.flat_indices[0], (9,10))
            b = unpack_b_matrix(bn)[pos]
            self.assertAlmostEqual(bn[2][pos],bn[3][pos],places=10)
            np.testing.assert_allclose(b@b.T,cov,atol=1e-10)
            # The with-regularization covariance includes reg_covar*I.
            braw = unpack_b_matrix(br)[pos]
            np.testing.assert_allclose(braw@braw.T,cov+1e-6*np.eye(2),atol=1e-9)

    def test_2d_two_file_wrappers(self):
        x, y = demo_pair(steps=4)
        water = np.ones(x.shape[1:],dtype=int)
        water[1,2] = 0
        with tempfile.TemporaryDirectory() as d:
            count_2d_nonparam(d,x,y,0,4,quantiles_amount=3,mask=water)
            count_semi_2d_AB(d,x,y,0,4,quantiles_amount=3,mask=water,n_init=2)
            for t in (1,2,3):
                n = Path(d)/'Coefficients'/'Nonparametric_2d'/'daily'
                s = Path(d)/'Coefficients'/'Semiparametric_2d'/'daily'
                a = np.load(n/f'A_{t}.npy')
                b = np.load(n/f'B_{t}.npy')
                self.assertEqual(b.shape,(4,9,10))
                np.testing.assert_allclose(a,np.load(s/f'A_{t}.npy'),atol=1e-10,equal_nan=True)
                np.testing.assert_allclose(b,np.load(s/f'B_{t}.npy'),atol=1e-7,equal_nan=True)
            count_cf_from_daily(d,'Nonparametric_2d',0,4,correlation_window=3,mean_width=2,mask=water)
            folder = Path(d)/'Coeff_data_3d'
            self.assertEqual(np.load(folder/'1_C.npy').shape,(2,9,10))
            self.assertEqual(np.load(folder/'1_F.npy').shape,(9,10))
            self.assertEqual(np.load(folder/'1_FS.npy').shape,(2,9,10))
            self.assertTrue(np.isnan(np.load(folder/'1_F.npy')[1,2]))

    def test_cf_interpretation(self):
        a=np.zeros((3,2,2,2))
        b=np.zeros((3,4,2,2))
        for t in range(3):
            a[t,0] = t+1
            a[t,1] = 3-t
            b[t,0] = t+1
            b[t,1] = t+2
        mask = np.ones((2,2))
        mask[0,0]=0
        c=compute_c(a,b,mask)
        np.testing.assert_allclose(c[0,0,1],-1,atol=1e-12)
        np.testing.assert_allclose(c[1,0,1],1,atol=1e-12)
        self.assertTrue(np.isnan(c[:,0,0]).all())
        f=compute_f(a[-1],b[-1],mask)
        self.assertAlmostEqual(f[0,1], np.sqrt(10)/4)
        fs=compute_fs(a[-1],b[-1],mask)
        self.assertAlmostEqual(fs[0,0,1],3/3)
        self.assertAlmostEqual(fs[1,0,1],1/4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
