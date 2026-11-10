"""Integration tests of the replacement 1D dispatcher against small NP/SP stubs.

These stubs validate paths, labels, parallel dispatch and aggregation; they do
not replace the repository's statistical NP/SP implementations in production.
"""
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
import sys

import numpy as np


class TestScalarDispatcher(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.working = tempfile.TemporaryDirectory()
        cls.pkg = Path(cls.working.name)
        source = Path(__file__).resolve().parents[1] / 'Coefficients' / 'parallel.py'
        original_collection = Path(__file__).resolve().parent/'fixtures'/'collection.py'
        p = cls.pkg / 'Coefficients'
        p.mkdir()
        (p/'__init__.py').write_text('')
        (p/'parallel.py').write_bytes(source.read_bytes())
        (p/'collection.py').write_bytes(original_collection.read_bytes())
        (p/'common.py').write_text('''import numpy as np\ndef coerce_series(a):\n    a = np.asarray(a)\n    if a.ndim != 3: raise ValueError('need 3D')\n    return a\n''')
        (p/'runner.py').write_text('''import numpy as np\ndef _daily_file_complete(p, expected_shape):\n    try:\n        a = np.load(p, mmap_mode='r', allow_pickle=False)\n        return tuple(a.shape) == tuple(expected_shape)\n    except (OSError,ValueError,FileNotFoundError):return False\n''')
        (p/'nonparametric.py').write_text('''from pathlib import Path\nimport numpy as np\ndef count_1d_nonparam(files_path_prefix, flux, time_start, time_end, path, start_index, quantiles_amount, mask, min_group_size, overwrite, dt):\n    folder=Path(files_path_prefix)/path/'Nonparametric'/'daily'\n    folder.mkdir(parents=True,exist_ok=True)\n    for t in range(time_start+1,time_end):\n        a = folder/f'A_{t+start_index}.npy'\n        b = folder/f'B_{t+start_index}.npy'\n        if not overwrite and a.exists() and b.exists():continue\n        np.save(a,(flux[t]-flux[t-1])/dt)\n        np.save(b,np.ones_like(flux[t])*2)\n''')
        (p/'semiparametric.py').write_text('''from pathlib import Path\nimport numpy as np\ndef count_semi_1d_AB(files_path_prefix, data_array, time_start, time_end, path, start_index, quantiles_amount, mask, min_group_size, overwrite, dt, n_components, n_init, remove_reg_covar):\n    folder=Path(files_path_prefix)/path/'Semiparametric'/'daily'\n    folder.mkdir(parents=True,exist_ok=True)\n    for t in range(time_start+1,time_end):\n        a = folder/f'A_{t+start_index}.npy'\n        b = folder/f'B_{t+start_index}.npy'\n        c = folder/f'B_with_reg_{t+start_index}.npy'\n        if not overwrite and a.exists() and b.exists() and c.exists():continue\n        np.save(a,(data_array[t]-data_array[t-1])/dt)\n        np.save(b,np.ones_like(data_array[t])*3)\n        np.save(c,np.ones_like(data_array[t])*4)\n''')
        sys.path.insert(0,str(cls.pkg))
        import Coefficients.parallel as parallel
        cls.mod = parallel

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(cls.pkg))
        for key in list(sys.modules):
            if key == 'Coefficients' or key.startswith('Coefficients.'):
                del sys.modules[key]
        cls.working.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _input(self,name,n=6):
        path=self.root/f'{name}.npy'
        np.save(path,np.arange(n*12,dtype=float).reshape(n,3,4)+len(name))
        return str(path)

    def _run(self, names, files, **kwargs):
        args = dict(files_path_prefix=str(self.root),source_files=files,
                    mask=np.ones((3,4),dtype=bool),names=names,
                    start_year=1979,end_year=1989,start_index=1,
                    parallel_workers=1,methods=('np','sp'),n_bins=5,n_bins_2d=3,
                    n_components=2,n_init=1,chunk_transitions=2,dt=1.,
                    run_1d=True,run_2d=False,overwrite=False,
                    delete_daily_after_collect=True,min_group_size=2)
        args.update(kwargs)
        return self.mod.run_parallel_period(**args)

    def test_two_scalar_variables_both_methods(self):
        result=self._run(('sensible','latent'),(self._input('sensible'),self._input('latent')))
        self.assertEqual(set(result),{'1d/sensible/np','1d/sensible/sp','1d/latent/np','1d/latent/sp'})
        for name in ('sensible','latent'):
            for method in ('np','sp'):
                for key,path in result[f'1d/{name}/{method}'].items():
                    self.assertEqual(np.load(path,mmap_mode='r').shape,(5,3,4))
                    self.assertEqual(path.name,f'{key}_2-6.npy')
        self.assertEqual(len(list((self.root/'Coefficients'/'1979-1989').rglob('daily/*'))),0)

    def test_one_variable_twice_no_duplicate_collect(self):
        f=self._input('Flux',7)
        result=self._run(('Flux','Flux'),(f,f),methods=('sp',))
        self.assertEqual(set(result),{'1d/Flux/sp'})
        self.assertEqual(np.load(result['1d/Flux/sp']['A']).shape,(6,3,4))
        self.assertEqual(len(list(self.root.rglob('A_2-7.npy'))),1)

    def test_parallel_workers(self):
        f=self._input('Flux',10)
        result=self._run(('Flux',),(f,),methods=('np',),parallel_workers=2)
        self.assertEqual(np.load(result['1d/Flux/np']['A']).shape,(9,3,4))

    def test_one_variable_once(self):
        f=self._input('Flux')
        result=self._run(('Flux',),(f,),methods=('np',))
        self.assertEqual(set(result),{'1d/Flux/np'})

    def test_scalar_lengths_independent(self):
        result=self._run(('Flux','SST'),(self._input('Flux',7),self._input('SST',5)),methods=('np',))
        self.assertEqual(np.load(result['1d/Flux/np']['A']).shape[0],6)
        self.assertEqual(np.load(result['1d/SST/np']['A']).shape[0],4)

    def test_resume_after_daily_cleanup(self):
        f=self._input('Flux')
        first=self._run(('Flux',),(f,),methods=('sp',))
        a=first['1d/Flux/sp']['A']; stamp=a.stat().st_mtime_ns
        second=self._run(('Flux',),(f,),methods=('sp',))
        self.assertEqual(a.stat().st_mtime_ns,stamp)
        self.assertEqual(first,second)

    def test_preserve_existing_bivariate_results(self):
        p=self.root/'Coefficients'/'1979-1989'/'2d_bins_3'/'sensible_latent'/'Semiparametric_2d'
        p.mkdir(parents=True)
        x=p/'A_2-6.npy'
        np.save(x,np.ones((5,2,3,4)))
        self._run(('sensible','latent'),(self._input('sensible'),self._input('latent')),
                  methods=('np',))
        self.assertTrue(x.is_file())
        np.testing.assert_allclose(np.load(x),1)

    def test_repair_corrupt_daily_file(self):
        f = self._input('Flux')
        d = self.root/'Coefficients'/'1979-1989'/'1d_bins_5'/'Flux'/'Nonparametric'/'daily'
        d.mkdir(parents=True)
        np.save(d/'A_3.npy', np.zeros((4,4)))  # bad spatial shape
        result = self._run(('Flux',),(f,),methods=('np',))
        self.assertEqual(np.load(result['1d/Flux/np']['A']).shape,(5,3,4))

    def test_mixed_mode_rejects_duplicate_names_before_writing(self):
        f = self._input('Flux')
        with self.assertRaisesRegex(ValueError,'two distinct'):
            self._run(('Flux','Flux'),(f,f),run_2d=True)
        self.assertFalse((self.root/'Coefficients').exists())

    def test_explicit_recompute(self):
        f=self._input('Flux')
        first=self._run(('Flux',),(f,),methods=('np',))
        a=first['1d/Flux/np']['A']; np.save(a,np.zeros((5,3,4)))
        second=self._run(('Flux',),(f,),methods=('np',),overwrite=True)
        self.assertTrue(np.any(np.load(second['1d/Flux/np']['A']) != 0))


if __name__=='__main__':
    import unittest
    unittest.main()

# Extended multiprocessing smoke test is defined as a standalone script in README.
