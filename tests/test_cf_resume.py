"""Synthetic tests for restart, publication and original C/F/FS formula parity."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
from Coefficients import derived_workflows as dw
from Coefficients.coefficients_cf import compute_c, compute_f, compute_fs


class ResumeTests(unittest.TestCase):
    """Regression tests covering Resume."""
    def setUp(self):
        """Prepare the test fixtures before running the assertions."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        rng = np.random.default_rng(12)
        self.a = rng.normal(size=(12, 2, 3, 4))
        self.b = rng.normal(size=(12, 4, 3, 4))
        self.a[5, 0, 1, 1] = np.nan
        self.a[:, :, 2, 2] = 0
        self.b[:, :, 2, 3] = 0
        self.mask = np.ones((3, 4), bool)
        self.mask[0, 0] = False
        self.pa, self.pb = self.root / 'a.npy', self.root / 'b.npy'
        np.save(self.pa, self.a)
        np.save(self.pb, self.b)
        continuous = self.root / 'Coefficients' / 'continuous'
        self.parent = continuous / '2d_bins_12' / 'sensible_latent' / 'Semiparametric_2d'
        continuous.mkdir(parents=True)
        (continuous / 'manifest_sensible_latent_12.json').write_text(json.dumps({
            'first_label': 2, 'methods': {'sp': {'A': str(self.pa), 'B': str(self.pb)}}}))
        self.kw = dict(method='sp', mask=self.mask, correlation_window=4,
                       mean_width=3, overwrite=True, checkpoint_every=2)

    def execute(self, **kwargs):
        """Compute execute for Resume Tests."""
        with contextlib.redirect_stdout(io.StringIO()) as log:
            result = dw.run_continuous_cf(self.root, **dict(self.kw, **kwargs))
        return result, log.getvalue()

    def assert_original(self, result):
        """Assert original for the test cf resume workflow."""
        for key, width in [('C', 4), ('F', 3), ('FS', 3)]:
            expected = []
            for step in range(len(self.a)-width+1):
                aw, bw = self.a[step:step+width], self.b[step:step+width]
                if key == 'C':
                    v = compute_c(aw, bw, self.mask)
                else:
                    valid = np.all(np.isfinite(aw), axis=(0,1)) & np.all(np.isfinite(bw), axis=(0,1))
                    am, bm = aw.mean(axis=0), bw.mean(axis=0)
                    am[:, ~valid] = np.nan
                    bm[:, ~valid] = np.nan
                    v = (compute_f if key == 'F' else compute_fs)(am, bm, self.mask)
                expected.append(v)
            np.testing.assert_allclose(np.load(result[key]['path']), np.stack(expected), equal_nan=True, rtol=0, atol=0)
            self.assertEqual(result[key]['first_label'], 2+width-1)
            self.assertEqual(result[key]['last_label'], 13)

    def test_original_numerics_and_cache(self):
        """Verify that original numerics and cache."""
        r, log = self.execute()
        self.assert_original(r)
        self.assertIn('ETA(stage)=', log)
        with patch.object(dw, '_calculate', side_effect=AssertionError('Unexpected recalculation')):
            self.assertIn('SKIP verified complete', self.execute()[1])

    def test_interrupt_f_after_three_windows(self):
        """Verify that interrupt f after three windows."""
        original = dw._calculate
        def interrupt(key, a, b, width, step, mask):
            """Compute interrupt for test interrupt f after three windows from ``key``, ``a``, ``b``, ``width``, ``step``."""
            if key == 'F' and step == 3:
                raise KeyboardInterrupt()
            return original(key, a, b, width, step, mask)
        with patch.object(dw, '_calculate', side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
            self.execute()
        self.assertFalse((self.parent / 'F_4-13.npy').exists())
        self.assertEqual(json.loads((self.parent / '.F_4-13.npy.progress.json').read_text())['next_index'], 3)
        with patch.object(dw, '_calculate', wraps=original) as calls:
            r, log = self.execute()
        self.assertIn('RESUME: 3/10', log)
        self.assertEqual([(x.args[0], x.args[4]) for x in calls.call_args_list],
                         [('F', i) for i in range(3, 10)] + [('FS', i) for i in range(10)])
        self.assert_original(r)

    def test_legacy_output_rebuilt(self):
        """Verify that legacy output rebuilt."""
        self.parent.mkdir(parents=True)
        legacy = self.parent / 'C_5-13.npy'
        np.save(legacy, np.zeros((9,2,3,4)))
        with self.assertRaises(FileExistsError):
            self.execute(overwrite=False)
        r, log = self.execute()
        self.assertIn('unverified', log)
        self.assert_original(r)

    def test_restart_if_source_changes(self):
        """Verify that restart if source changes."""
        self.execute()
        self.a[0] += 10
        np.save(self.pa, self.a)
        with self.assertRaisesRegex(RuntimeError, 'inputs/settings changed'):
            self.execute()
        r, _ = self.execute(resume=False)
        self.assert_original(r)

    def test_recover_publication_after_atomic_rename(self):
        """Verify that recover publication after atomic rename."""
        orig = dw._atomic_json
        def break_complete(path, value):
            """Compute break complete for test recover publication after atomic rename from ``path``, ``value``."""
            if value.get('status') == 'complete' and 'C_' in path.name:
                raise OSError('simulated publication interruption')
            return orig(path, value)
        with patch.object(dw, '_atomic_json', side_effect=break_complete), self.assertRaises(OSError):
            self.execute()
        self.assertTrue((self.parent / 'C_5-13.npy').exists())
        r, log = self.execute()
        self.assertIn('[C] SKIP verified complete', log)
        self.assert_original(r)

    def test_hard_exit_restarts_last_committed_window(self):
        """Verify that hard exit restarts last committed window."""
        script = r"""
import os, sys
import numpy as np
from Coefficients import derived_workflows as dw
orig = dw._calculate
# First window is checkpointed, then every 2. Kill before computing index 4:
# index 3 was calculated but not committed, so it must be repeated.
def interrupted(key, a, b, width, step, mask):
    if key == 'C' and step == 4:
        os._exit(73)
    return orig(key, a, b, width, step, mask)
dw._calculate = interrupted
water = np.ones((3,4), bool)
water[0,0] = False
dw.run_continuous_cf(sys.argv[1], method='sp', mask=water,
    correlation_window=4, mean_width=3, overwrite=True, checkpoint_every=2)
"""
        proc = subprocess.run([sys.executable, '-c', script, str(self.root)],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 73, proc.stdout + proc.stderr)
        progress = self.parent / '.C_5-13.npy.progress.json'
        self.assertEqual(json.loads(progress.read_text())['next_index'], 3)
        self.assertFalse((self.parent / 'C_5-13.npy').exists())
        with patch.object(dw, '_calculate', wraps=dw._calculate) as calls:
            result, log = self.execute()
        self.assertIn('RESUME: 3/9', log)
        self.assertEqual(calls.call_args_list[0].args[4], 3)
        self.assert_original(result)

    def test_lock(self):
        """Verify that lock."""
        self.parent.mkdir(parents=True)
        with dw._exclusive_lock(self.parent / '.derived_cf.lock'):
            with self.assertRaises(RuntimeError):
                self.execute()


if __name__ == '__main__':
    unittest.main()
