"""Self-contained smoke tests with simulated source arrays, no ERA5 download."""
from __future__ import annotations

import sys
import json
import tempfile
from pathlib import Path
from types import ModuleType

import numpy as np
import matplotlib
matplotlib.use('Agg')

BASE = Path(__file__).parents[1]
sys.path.insert(0, str(BASE))

# Stub only the existing project's file-IO layer. The numerical and plotting
# functions under test are the actual delivered files.
dp = ModuleType('Data_processing')
dp.__path__ = []
io = ModuleType('Data_processing.io')

def field_candidates(root, name, start, end):
    return [Path(root)/'DATA'/'Fluxes'/f'{name}_{start}-{end}_grouped.npy']

def load_field(root, name, start, end):
    return np.load(field_candidates(root,name,start,end)[0], mmap_mode='r')

def read_ocean_mask(root, shape=(40,50)):
    return np.load(Path(root)/'DATA'/'ocean_mask.npy')
io.field_candidates = field_candidates
io.load_field = load_field
io.read_ocean_mask = read_ocean_mask
sys.modules['Data_processing'] = dp
sys.modules['Data_processing.io'] = io

# The plotting module's numerical sibling is imported normally.
from Functional.article_diagnostics import (stationary_density,
    equal_probability_regions, classify_map, prepare_article_diagnostics)
from Plotting.article_figures import plot_article_figures, _copula_contours


def make_root(root):
    (root/'DATA'/'Fluxes').mkdir(parents=True)
    (root/'Functional'/'Stationary').mkdir(parents=True)
    water = np.ones((40, 50), bool)
    water[:10, :10] = 0
    np.save(root/'DATA'/'ocean_mask.npy', water)
    rng = np.random.default_rng(52)
    fields = {}
    for name, mean, scale in [('sensible', -20, 20), ('latent', -90, 35)]:
        field = (rng.normal(mean, scale, (18,40,50)) +
                 20*np.linspace(-1, 1, 50)[None,None,:] +
                 30*np.linspace(0,1,40)[None,:,None]).astype('float32')
        field[:,~water] = np.nan
        np.save(root/'DATA'/'Fluxes'/f'{name}_1979-2026_grouped.npy',field)
        grid = np.linspace(-450,450,1501)
        pdf = np.exp(-.5*((grid-mean)/scale)**2)
        pdf /= np.trapezoid(pdf,grid)
        fields[name] = (grid,pdf)
    np.savez_compressed(root/'Functional'/'Stationary'/'sensible-latent_1979-2026.npz',
                        grid=fields['sensible'][0],
                        pdf_sensible=fields['sensible'][1],
                        pdf_latent=fields['latent'][1])
    np.savez_compressed(root/'Functional'/'Stationary'/'sensible-latent_1979-1989.npz',
                        grid=fields['sensible'][0],
                        pdf_sensible=fields['sensible'][1],
                        pdf_latent=fields['latent'][1])
    return water


def test_pdf_quintiles():
    grid = np.linspace(-8,8,5001)
    pdf = np.exp(-grid**2/2)/np.sqrt(2*np.pi)
    q, mass = equal_probability_regions(grid,pdf)
    assert len(q)==4 and np.allclose(mass,0.2,atol=1e-8)
    mask = np.array([[True,False,True,True]])
    cls = classify_map(np.array([[q[0],0,np.nan,q[3]+1]]),q,mask)
    assert np.array_equal(cls,np.array([[1,0,0,5]],np.uint8))


def test_maps_and_reuse():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        water = make_root(root)
        c = prepare_article_diagnostics(root, ('sensible','latent'), 1979, 2026,
                                        density_source='fp', chunk_days=5)
        assert set(c) == {'sensible','latent'}
        with np.load(c['sensible']) as arr:
            assert arr['class_map'].shape == water.shape
            assert not np.any(arr['class_map'][~water])
            assert np.isclose(arr['region_masses'].sum(), 1)
            assert np.isnan(arr['anomaly'][~water]).all()
            first_q = arr['thresholds'].copy()
        c2 = prepare_article_diagnostics(root, ('sensible','latent'), 1979,2026,
                                         density_source='fp', chunk_days=5)
        assert c == c2
        assert np.array_equal(np.load(c2['sensible'])['thresholds'],first_q)
        images = plot_article_figures(root, ('sensible','latent'), 1979, 2026,
                                     density_source='fp', chunk_days=5,
                                     language='en', dpi=75)
        assert len(images) == 6
        assert all(p.stat().st_size > 1500 for p in images)
        assert len(list((root/'Plots'/'Article'/'1979-2026').glob('fig0*.png'))) == 6
        # Optional path for a legitimate external 2D anomaly.
        second = root/'second.npy'
        np.save(second,np.random.default_rng(1).normal(0,20,water.shape))
        z = plot_article_figures(root,('sensible',),1979,2026,
                                 density_source='fp',chunk_days=5,
                                 second_anomaly_files={'sensible': str(second)},
                                 figures=('isolines',),dpi=75)
        assert len(z)==1
        # Optional full-period local quintile frequencies must sum to 1 at sea.
        f = prepare_article_diagnostics(root,('sensible',),1979,2026,
                                        density_source='fp',chunk_days=5,
                                        compute_frequencies=True)
        with np.load(f['sensible']) as arr:
            assert np.allclose(arr['frequencies'][:,water].sum(axis=0), 1,atol=1e-10)
        return images


def test_copula_contours():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d)
        folder=root/'Functional'/'Copulas'
        folder.mkdir(parents=True)
        x=np.linspace(.01,.99,45)
        xx,yy=np.meshgrid(x,x)
        pdf=np.exp(-((xx-yy)/.14)**2)
        src=folder/'sensible-latent_1979-2026.npz'
        np.savez(src,centers=x,checkerboard_centers=x,bernstein_grid=x,
                 gaussian=pdf,student_t=pdf*.9,checkerboard=pdf*.8,bernstein=pdf*.7)
        out=_copula_contours(root,('sensible','latent'),(1979,2026),root,'en',75)
        assert len(out)==4 and all(p.exists() for p in out)


if __name__ == '__main__':
    test_pdf_quintiles()
    test_maps_and_reuse()
    test_copula_contours()
    print('3 test groups passed')
