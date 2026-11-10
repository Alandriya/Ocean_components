"""Synthetic end-to-end smoke test for the restored copula plotter."""
import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.axes
import numpy as np


class TestRestoredCopulaPlots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project = tempfile.TemporaryDirectory()
        top = Path(cls.project.name)
        for name in ('Plotting', 'Data_processing'):
            (top/name).mkdir()
            (top/name/'__init__.py').write_text('')
        source = Path(__file__).resolve().parents[1]/'Plotting'/'workflows.py'
        (top/'Plotting'/'workflows.py').write_bytes(source.read_bytes())
        (top/'Plotting'/'paths.py').write_text('''from pathlib import Path

def plot_path(root,*parts):
    p=Path(root)/'Plots'
    for part in parts: p=p/part
    p.parent.mkdir(parents=True,exist_ok=True)
    return p

def save_figure(fig,path,**kwargs):
    fig.savefig(path,dpi=100,bbox_inches='tight')
    return path
''')
        (top/'Data_processing'/'io.py').write_text('''from datetime import date
BASE_DATE = date(1979,1,1)
def load_field(*args):raise NotImplementedError()
def load_spectral_field(*args):raise NotImplementedError()
def read_ocean_mask(*args):raise NotImplementedError()
def start_label(*args):raise NotImplementedError()
''')
        sys.path.insert(0,str(top))
        cls.module = importlib.import_module('Plotting.workflows')
        cls.original_dir = top

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(cls.original_dir))
        for k in list(sys.modules):
            if k=='Plotting' or k.startswith('Plotting.') or k=='Data_processing' or k.startswith('Data_processing.'):
                del sys.modules[k]
        cls.project.cleanup()

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
            def record(ax,*args,**kwargs):
                kwargs_seen.append(kwargs.copy())
                return original(ax,*args,**kwargs)
            with patch.object(matplotlib.axes.Axes,'pcolormesh',record):
                figures=self.module.plot_copula_results(root,('sensible','latent'),1979,1989)
            self.assertEqual(len(figures),10)
            self.assertTrue(all(f.is_file() and f.stat().st_size>1000 for f in figures))
            self.assertEqual(before,file.read_bytes(),'Must not rewrite computed copula NPZ')
            self.assertTrue(any(x.get('cmap')=='YlOrRd' for x in kwargs_seen))
            self.assertGreaterEqual(len(kwargs_seen),10)
            self.assertIn('Comparison',str(figures[0]))
            self.assertIn('Joint',str(figures[-1]))
            self.assertIn('empirical',str(figures[1]))


if __name__=='__main__':unittest.main()
