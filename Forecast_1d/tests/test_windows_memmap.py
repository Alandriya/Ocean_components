"""Regression checks for Windows file locks and safe recovery."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from Forecast_1d import extract_points as ep
from Forecast_1d.recover_staging import recover, validate_staging

POINTS = Path(__file__).parents[1] / 'points.json'
MASK = Path(__file__).parent / 'fixtures' / 'mask'


class WindowsMemmapTest(unittest.TestCase):
    """Regression tests covering Windows Memmap."""
    def setUp(self):
        """Prepare the test fixtures before running the assertions."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'DATA' / 'Fluxes').mkdir(parents=True)
        (self.root / 'DATA' / 'mask').write_bytes(MASK.read_bytes())
        self.days = 35
        yy, xx = np.mgrid[:161,:181]
        base = yy + xx / 2
        for k, name in enumerate(('sensible', 'latent')):
            vals = np.stack([(base + i * (1 + k)).astype('float32')
                             for i in range(self.days)],axis=0)
            np.save(self.root / 'DATA' / 'Fluxes' / f'{name}_1979-2026_grouped.npy', vals)
        self.output = self.root / 'DATA' / 'Forecast_1d' / ep.DEFAULT_SUBDIR

    def test_all_writer_mappings_closed_before_rename(self):
        """Verify that all writer mappings closed before rename."""
        original_memmap = np.lib.format.open_memmap
        original_rename = ep._rename_with_retry
        held = []

        def track_memmap(*args, **kwargs):
            """Compute track memmap for test all writer mappings closed before rename."""
            arr = original_memmap(*args, **kwargs)
            if '.building-' in str(args[0]):
                held.append(arr)
            return arr

        def check_closed_before_publish(source, dest, **kwargs):
            """Check closed before publish for the test windows memmap workflow."""
            self.assertEqual(len(held),8)
            self.assertTrue(all(x._mmap.closed for x in held),
                            'Windows would lock an open output memmap')
            return original_rename(source, dest, **kwargs)

        with mock.patch.object(ep.np.lib.format, 'open_memmap', side_effect=track_memmap):
            with mock.patch.object(ep, '_rename_with_retry', side_effect=check_closed_before_publish):
                ep.extract(self.root, POINTS, self.output, chunk_days=9)
        self.assertTrue((self.output / 'std_15x15.npy').is_file())
        self.assertEqual(len(held), 8)

    def test_rename_retries_transient_access_denied(self):
        """Verify that rename retries transient access denied."""
        self.output.parent.mkdir(parents=True, exist_ok=True)
        source = self.output.parent / 'tempdir'
        source.mkdir()
        original_rename = Path.rename
        attempts = [0]

        def temporarily_locked(path, target):
            """Compute temporarily locked for test rename retries transient access denied from ``path``, ``target``."""
            attempts[0] += 1
            if attempts[0] <= 2:
                raise PermissionError('Simulated Windows lock')
            return original_rename(path, target)

        with mock.patch.object(Path, 'rename', autospec=True,
                               side_effect=temporarily_locked):
            with mock.patch.object(ep.time, 'sleep'):
                ep._rename_with_retry(source, self.output)
        self.assertEqual(attempts[0], 3)
        self.assertTrue(self.output.is_dir())

    def test_failed_publish_preserves_complete_staging_for_recovery(self):
        """Verify that failed publish preserves complete staging for recovery."""
        with mock.patch.object(ep, '_rename_with_retry', side_effect=PermissionError('WinError 5')):
            with self.assertRaises(PermissionError):
                ep.extract(self.root, POINTS, self.output, chunk_days=8)
        candidates = list(self.output.parent.glob('.atlantic15_native_1979-2026.building-*'))
        self.assertEqual(len(candidates), 1)
        good, msg = validate_staging(candidates[0])
        self.assertTrue(good, msg)
        self.assertFalse(self.output.exists())
        self.assertTrue(recover(destination=self.output, publish=True))
        self.assertTrue(self.output.is_dir())
        self.assertEqual(np.load(self.output / 'dates.npy').shape,(self.days,))

    def test_incomplete_staging_is_not_published(self):
        """Verify that incomplete staging is not published."""
        with mock.patch.object(ep, '_rename_with_retry', side_effect=PermissionError('WinError 5')):
            with self.assertRaises(PermissionError):
                ep.extract(self.root, POINTS, self.output, chunk_days=11)
        stage, = self.output.parent.glob('.atlantic15_native_1979-2026.building-*')
        (stage / 'patches_15x15.npy').unlink()
        good, msg = validate_staging(stage)
        self.assertFalse(good)
        self.assertIn('missing files',msg)
        with self.assertRaises(RuntimeError):
            recover(destination=self.output, publish=True)
        self.assertFalse(self.output.exists())

    def test_writer_mappings_closed_and_partial_removed_after_processing_error(self):
        """Verify that writer mappings closed and partial removed after processing error."""
        held=[]
        real_open = np.lib.format.open_memmap
        def track(*args, **kwargs):
            """Compute track for test writer mappings closed and partial removed after processing error."""
            a=real_open(*args,**kwargs)
            if '.building-' in str(args[0]):
                held.append(a)
            return a
        with mock.patch.object(ep.np.lib.format, 'open_memmap', side_effect=track):
            with mock.patch.object(ep, '_read_unique_pixels', side_effect=RuntimeError('Read failure')):
                with self.assertRaisesRegex(RuntimeError,'Read failure'):
                    ep.extract(self.root,POINTS,self.output)
        self.assertTrue(all(x._mmap.closed for x in held))
        self.assertFalse(list(self.output.parent.glob('.atlantic15_native_1979-2026.building-*')))

if __name__=='__main__':
    unittest.main()
