"""Integration tests for function-based 1D/2D period runners."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Coefficients.runner import run_decade_coefficients, run_2d_coefficients


def sample_pair(seed=17, shape=(12, 13), steps=5):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=(steps, *shape))
    x1 = 0.4 * x0 + rng.normal(size=(steps, *shape))
    return x0, x1


class RunnerTests(unittest.TestCase):
    def test_decade_only_2d_and_resume(self):
        x0, x1 = sample_pair()
        water = np.ones((12,13), dtype=int)
        water[0, 0] = 0
        with tempfile.TemporaryDirectory() as td:
            kwargs = dict(files_path_prefix=td, array1=x0, array2=x1,
                          mask=water, names=('sensible','latent'),
                          start_year=1979, end_year=1989, start_index=1,
                          n_bins=250, n_bins_2d=3, n_components=2,
                          n_init=2, chunk_transitions=2, run_1d=False,
                          run_2d=True, min_group_size=2)
            run_decade_coefficients(**kwargs)
            base = Path(td)/'Coefficients'/'aligned_1979-1989'/'2d_bins_3'/'sensible_latent'
            np_dir=base/'Nonparametric_2d'/'daily'
            sp_dir=base/'Semiparametric_2d'/'daily'
            self.assertEqual(len(list(np_dir.glob('A_*.npy'))), 4)
            self.assertEqual(len(list(sp_dir.glob('B_with_reg_*.npy'))), 4)
            self.assertFalse((Path(td)/'Coefficients'/'aligned_1979-1989'/'1d_bins_250').exists())
            np.testing.assert_allclose(np.load(np_dir/'B_2.npy'), np.load(sp_dir/'B_2.npy'),
                                       rtol=1e-6, atol=1e-7, equal_nan=True)
            baseline=(np_dir/'B_2.npy').stat().st_mtime_ns
            run_decade_coefficients(**kwargs)
            self.assertEqual((np_dir/'B_2.npy').stat().st_mtime_ns, baseline)

    def test_generic_2d_sp_only(self):
        x0, x1=sample_pair()
        with tempfile.TemporaryDirectory() as td:
            result=run_2d_coefficients(x0,x1,np.ones((12,13)),td,
                                       path='comparison', quantiles_amount=3,
                                       min_group_size=2, methods='sp', n_init=2)
            self.assertEqual(result['n_transitions'],4)
            self.assertTrue((Path(td)/'comparison'/'Semiparametric_2d'/'daily'/'A_1.npy').exists())
            self.assertFalse((Path(td)/'comparison'/'Nonparametric_2d').exists())

    def test_generic_2d_cf(self):
        x0, x1=sample_pair()
        with tempfile.TemporaryDirectory() as td:
            run_2d_coefficients(x0,x1,np.ones((12,13)),td,
                                path='comparison', quantiles_amount=3,
                                min_group_size=2, methods='np',
                                compute_cf=True, correlation_window=3, mean_width=2)
            self.assertTrue((Path(td)/'Coeff_data_3d'/'Nonparametric_2d'/'1_C.npy').exists())
            self.assertTrue((Path(td)/'Coeff_data_3d'/'Nonparametric_2d'/'1_FS.npy').exists())


