"""Offline numerical and rendering tests; no ERA5 data required."""
import json
import tempfile
import unittest
from pathlib import Path
from datetime import date
import matplotlib
matplotlib.use('Agg')
import numpy as np
from Data_processing.trends import (spatial_statistics, block_statistics, linear_fit,
    run_trends, result_path, day_label)
from Plotting.trend_plots import plot_trends


class TrendsTests(unittest.TestCase):
    def test_mask_windows_slope(self):
        water=np.array([[1,1],[0,1]],bool)
        field=np.array([[1,4],[999,np.nan]])
        np.testing.assert_allclose(spatial_statistics(field,water), [[4,1,2.5,3]])
        labels=np.array([1,2,4,5,6,7])  # day 3 missing
        vals=np.tile(labels[:,None,None],(1,1,4)).astype(float)
        starts, blocks, coverage=block_statistics(labels,vals,3,anchor=1,min_coverage=1)
        np.testing.assert_equal(starts,[1,4]); self.assertTrue(np.isnan(blocks[0]).all())
        np.testing.assert_equal(coverage[:,0,0],[2,3]); np.testing.assert_allclose(blocks[1],5)
        _, block,_=block_statistics(labels,vals,3,anchor=1,min_coverage=2/3,spectral_min=True)
        self.assertEqual(block[1,0,1],4); self.assertEqual(block[1,0,0],5)
        np.testing.assert_allclose(linear_fit(np.arange(1,11),2*np.arange(10)+5),[2,5,18,1,10])

    def test_cached_workflow(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); water=np.ones((3,4),bool); water[0,0]=False
            start=day_label(date(2019,1,1)); n=65
            arr=np.arange(n,dtype=float)[:,None,None]+np.arange(12).reshape(1,3,4)
            arr[:,0,0]=1e9
            for name, folder, mult in (('Flux','Fluxes',1),('SST','SST',2)):
                target=root/'DATA'/folder; target.mkdir(parents=True)
                np.save(target/f'{name}_2019-2026_grouped.npy',arr*mult)
            parent=root/'Coefficients'/'2019-2026'/'Flux_SST'; parent.mkdir(parents=True)
            cont=root/'Coefficients'/'continuous'; cont.mkdir(exist_ok=True)
            a=np.stack((arr,2*arr),axis=1); b=np.stack((arr,2*arr,3*arr,4*arr),axis=1)
            for key,data in [('A',a),('B',b)]: np.save(parent/f'{key}_2019-2026_sp.npy',data)
            manifest={'first_label':start+1,'last_label':start+n,'methods':{'sp':{
                k:f'D:/old/Coefficients/2019-2026/Flux_SST/{k}_2019-2026_sp.npy' for k in ('A','B')}}}
            (cont/'manifest_Flux_SST_12.json').write_text(json.dumps(manifest))
            derived=cont/'2d_bins_12'/'Flux_SST'/'Semiparametric_2d'; derived.mkdir(parents=True)
            np.save(derived/f'F_{start+1}-{start+n}.npy',arr)
            np.save(derived/f'C_{start+1}-{start+n}.npy',a)
            spectral=root/'Eigenvalues'/'Flux-SST'; spectral.mkdir(parents=True)
            for i in range(n-1):
                np.save(spectral/f'singular_values_{start+i}.npy',[2+i,1])
                np.save(spectral/f'left_singularvectors_{start+i}.npy',np.eye(2))
                np.save(spectral/f'right_singularvectors_{start+i}.npy',np.eye(2))
            options=dict(sources=('Variables','Coefficients','Derived','Eigenvalues'),variables=('Flux','SST'),
                         coefficient_pairs=(('Flux','SST'),),derived_pair=('Flux','SST'),
                         spectral_pairs=(('Flux','SST'),),method='sp',windows=(30,),mode_extrema=True)
            paths=run_trends(root,2019,2026,**options,mask=water,spectral_bins=2,spectral_periods=((2019,2026),))
            self.assertEqual(len(paths),9)
            with np.load(result_path(root,'Variables','Flux',2019,2026)) as z:
                self.assertEqual(z['daily'][0,0,0],11)
                self.assertAlmostEqual(z['fits_30'][0,0,0],1)
            with np.load(result_path(root,'Coefficients','Flux-SST_B',2019,2026)) as z:
                self.assertEqual(z['channels'].tolist(),['B00','B11','B01','B10'])
                np.testing.assert_allclose(z['fits_30'][:,0,0],np.array([1,2,3,4])*59/60)
            with np.load(result_path(root,'Eigenvalues','Flux-SST_spectrum',2019,2026)) as z:
                self.assertAlmostEqual(z['fits_30'][0,2,0],1)
            with np.load(result_path(root,'Eigenvalues','Flux-SST_contribution1',2019,2026)) as z:
                self.assertTrue(np.nanmin(z['daily'][:,:,1])>=0)
            images=plot_trends(root,2019,2026,**options,dpi=55)
            self.assertEqual(len(images),9)
            self.assertTrue(all(p.stat().st_size>1000 for p in images))


if __name__=='__main__': unittest.main()
