"""Synthetic checks. Run: python -m unittest discover -s Forecast_1d/tests -v"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from Forecast_1d.extract_points import (
    BASE_DATE, GRID, _inspect_source, _read_unique_pixels,
    _validate_points, _mask_from_bytes, _unique_pixels, extract,
)

POINTS_JSON = Path(__file__).parents[1] / 'points.json'
RECEIVED_MASK = Path(__file__).parent / 'fixtures' / 'mask'


class ExtractTest(unittest.TestCase):
    """Regression tests covering Extract."""
    def setUp(self):
        """Prepare the test fixtures before running the assertions."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        folder = self.root / 'DATA' / 'Fluxes'
        folder.mkdir(parents=True)
        self.mask_bytes = (
            RECEIVED_MASK.read_bytes()
            if RECEIVED_MASK.exists()
            else bytes([1]) * (GRID[0] * GRID[1])
        )
        (self.root / 'DATA' / 'mask').write_bytes(self.mask_bytes)
        self.days = 35
        ys, xs = np.meshgrid(np.arange(GRID[0]), np.arange(GRID[1]), indexing='ij')
        space = (2*ys + xs).astype(np.float32)
        self.sensible = np.stack([space + 10*t for t in range(self.days)]).astype('float32')
        self.latent = np.stack([100 - space + 3*t for t in range(self.days)]).astype('float32')

    def save_sources(self, sensible_layout, latent_layout):
        """Save sources for the test extract points workflow."""
        for channel, original, layout in [
            ('sensible', self.sensible, sensible_layout),
            ('latent', self.latent, latent_layout),
        ]:
            if layout == 'time_hw': data = original
            elif layout == 'space_time': data = original.reshape(self.days, -1).T
            elif layout == 'time_space': data = original.reshape(self.days, -1)
            else: raise ValueError(layout)
            np.save(self.root / 'DATA' / 'Fluxes' / f'{channel}_1979-2026_grouped.npy', data)

    def test_offshore_points_and_latlon(self):
        """Verify that offshore points and latlon."""
        ocean, digest = _mask_from_bytes(self.root/'DATA'/'mask')
        points = _validate_points(POINTS_JSON, ocean)
        self.assertEqual(len(points),15)
        if RECEIVED_MASK.exists():
            self.assertEqual(sum(bool(ocean[int(p['grid_y']),int(p['grid_x'])]) for p in points),15)

    def test_extract_mixed_legacy_layouts(self):
        """Verify that extract mixed legacy layouts."""
        self.save_sources('time_hw', 'space_time')
        out = self.root/'DATA'/'Forecast_1d'/'test'
        info = extract(self.root, POINTS_JSON, out, chunk_days=9)
        self.assertEqual(info['n_dates'], self.days)
        self.assertEqual(info['last_date'],'1979-02-04')
        arr = np.load(out/'patches_15x15.npy', mmap_mode='r')
        center = np.load(out/'centers.npy')
        self.assertEqual(arr.shape,(self.days,15,2,15,15))
        self.assertTrue(np.array_equal(arr[:,:,:,7,7],center))
        pts=json.loads(POINTS_JSON.read_text())['points']
        for i, p in enumerate(pts):
            y,x = p['grid_y'],p['grid_x']
            self.assertEqual(center[34,i,0],self.sensible[34,y,x])
            self.assertEqual(center[34,i,1],self.latent[34,y,x])
            np.testing.assert_array_equal(arr[0,i,0],self.sensible[0,y-7:y+8,x-7:x+8])
            np.testing.assert_array_equal(arr[0,i,1],self.latent[0,y-7:y+8,x-7:x+8])
            np.testing.assert_allclose(np.load(out/'mean_3x3.npy')[0,i,0],
                                       np.mean(self.sensible[0,y-1:y+2,x-1:x+2]))
            np.testing.assert_allclose(np.load(out/'std_7x7.npy')[0,i,1],
                                       np.std(self.latent[0,y-3:y+4,x-3:x+4]))
        dates=np.load(out/'dates.npy')
        self.assertEqual(str(dates[0]),'1979-01-01')
        self.assertEqual(str(dates[-1]),'1979-02-04')
        self.assertEqual(json.loads((out/'manifest.json').read_text())['channels'],['sensible','latent'])
        with self.assertRaises(FileExistsError):
            extract(self.root,POINTS_JSON,out,chunk_days=8)
        extract(self.root,POINTS_JSON,out,chunk_days=8,overwrite=True)
        self.assertTrue((out/'centers.npy').exists())

    def test_extract_time_space(self):
        """Verify that extract time space."""
        self.save_sources('space_time','time_space')
        out = self.root/'data'
        extract(self.root,POINTS_JSON,out,chunk_days=11)
        vals=np.load(out/'centers.npy')
        y,x=92,40
        self.assertAlmostEqual(float(vals[31,0,0]),float(self.sensible[31,y,x]))
        self.assertAlmostEqual(float(vals[31,0,1]),float(self.latent[31,y,x]))

    def test_input_dates_mismatch_fails_early(self):
        """Verify that input dates mismatch fails early."""
        self.save_sources('time_hw','space_time')
        p=self.root/'DATA'/'Fluxes'/'latent_1979-2026_grouped.npy'
        np.save(p, self.latent[:34])
        out=self.root/'DATA'/'Forecast_1d'/'mismatch'
        with self.assertRaisesRegex(ValueError,'dates do not align'):
            extract(self.root,POINTS_JSON,out)
        self.assertFalse(out.exists())

    def test_mask_invalid_cell(self):
        """Verify that mask invalid cell."""
        water=np.frombuffer(self.mask_bytes,dtype=np.uint8).reshape(GRID).copy()
        water[92,40]=0
        (self.root/'DATA'/'mask').write_bytes(water.tobytes())
        o,_=_mask_from_bytes(self.root/'DATA'/'mask')
        with self.assertRaisesRegex(ValueError,'all-water'):
            _validate_points(POINTS_JSON,o)


if __name__=='__main__':
    unittest.main()
