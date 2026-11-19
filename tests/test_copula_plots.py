"""Synthetic end-to-end smoke test for the consolidated copula plotter."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.axes
import numpy as np

from Plotting.copula_plotting import plot_copula_results


class TestRestoredCopulaPlots(unittest.TestCase):
    def test_saved_grids_legacy_palette_and_files(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            target=root/'Functional'/'Copulas'
            target.mkdir(parents=True)
            centers=(np.arange(60)+.5)/60
            check=(np.arange(50)+.5)/50
            bern=np.linspace(.005,.995,90)
            def copula(c,width):
                xx,yy=np.meshgrid(c,c)
                return 1 + 3*np.exp(-((xx-yy)/width)**2)
            x=np.linspace(-2000,1500,150)
            yy,xx=np.meshgrid(x,x,indexing='ij')
            a=np.exp(-((xx+16)/75)**2-((yy+90)/170)**2)
            data=dict(centers=centers,empirical=copula(centers,.13),gaussian=copula(centers,.18),
                      student_t=copula(centers,.10),checkerboard_centers=check,checkerboard=copula(check,.17),
                      bernstein_grid=bern,bernstein=copula(bern,.19),x1=x,x2=x,
                      joint_gaussian=a,joint_checkerboard=.8*a,joint_bernstein=1.1*a)
            file=target/'sensible-latent_1979-1989.npz'
            np.savez_compressed(file,**data)
            before=file.read_bytes()
            original=matplotlib.axes.Axes.pcolormesh
            kwargs_seen=[]
            axes_seen=[]
            def record(ax,*args,**kwargs):
                kwargs_seen.append(kwargs.copy())
                if kwargs.get('cmap')=='YlOrRd':
                    axes_seen.append(ax)
                return original(ax,*args,**kwargs)
            with patch.object(matplotlib.axes.Axes,'pcolormesh',record):
                figures=plot_copula_results(root,('sensible','latent'),1979,1989,language='en',dpi=75)
            self.assertEqual(len(figures),14)
            self.assertEqual(len(set(figures)),14)
            self.assertTrue(all(f.is_file() and f.stat().st_size>1000 for f in figures))
            self.assertEqual(before,file.read_bytes(),'Must not rewrite computed copula NPZ')
            self.assertTrue(any(x.get('cmap')=='YlOrRd' for x in kwargs_seen))
            self.assertGreaterEqual(len(kwargs_seen),10)
            self.assertTrue(all(f.parent==root/'Plots'/'Copulas'/'Figures'/'sensible-latent_1979-1989' for f in figures))
            expected={f'copula_{m}_contours_eng.png' for m in ('gaussian','student_t','checkerboard','bernstein')}
            self.assertTrue(expected.issubset({f.name for f in figures}))
            self.assertTrue(all(f.name.endswith('_eng.png') for f in figures))
            self.assertFalse((root/'Plots'/'Article').exists())
            # All three joint-density maps retain the existing physical display limits.
            physical=[ax for ax in axes_seen if ax.get_xlabel()=='Sensible heat flux']
            self.assertEqual(len(physical),3)
            for ax in physical:
                np.testing.assert_allclose(ax.get_xlim(),[-100,100])
                np.testing.assert_allclose(ax.get_ylim(),[-250,50])
            contour_paths = {p: p.read_bytes() for p in figures if p.name in expected}
            bw = plot_copula_results(root,('sensible','latent'),1979,1989,
                                     language='en',dpi=75,monochrome=True)
            self.assertEqual(len(bw), 14)
            self.assertEqual({p.name for p in bw if p.name.endswith('_bw.png')},
                             {n.replace('.png','_bw.png') for n in expected})
            self.assertEqual(before, file.read_bytes())
            self.assertTrue(all(p.read_bytes() == content for p, content in contour_paths.items()))


if __name__=='__main__':unittest.main()
