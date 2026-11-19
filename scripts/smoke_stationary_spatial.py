"""Self-contained smoke tests with simulated source arrays, no ERA5 download."""
from __future__ import annotations

import sys
import json
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np
import matplotlib
import matplotlib.image as mpimg
matplotlib.use('Agg')

BASE = Path(__file__).parents[1]
sys.path.insert(0, str(BASE))

# Stub only the project's file-IO layer. The numerical and plotting functions
# under test are the actual delivered files.
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

from Functional import stationary_diagnostics as diagnostics
from Functional.stationary_diagnostics import (stationary_density,
    equal_probability_regions, classify_map, prepare_stationary_diagnostics)
from Plotting.stationary_spatial_plots import (plot_stationary_spatial,
    _class_grid_edges, _class_mask_path, _grid_boundary_segments)
from Plotting.copula_plotting import plot_copula_contours


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
    for period in ("1979-2026", "1979-1989"):
        np.savez_compressed(root/'Functional'/'Stationary'/f'sensible-latent_{period}.npz',
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
        kwargs = dict(density_source='fp', chunk_days=5)
        try:
            prepare_stationary_diagnostics(root, cache_only=True, **kwargs)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError('Cache-only mode must reject missing arrays')
        c = prepare_stationary_diagnostics(root, ('sensible','latent'), 1979, 2026, **kwargs)
        assert set(c) == {'sensible','latent'}
        with np.load(c['sensible']) as arr:
            assert arr['class_map'].shape == water.shape
            assert not np.any(arr['class_map'][~water])
            assert np.isclose(arr['region_masses'].sum(), 1)
            assert np.isnan(arr['anomaly'][~water]).all()
            first_q = arr['thresholds'].copy()
        c2 = prepare_stationary_diagnostics(root, ('sensible','latent'), 1979,2026, **kwargs)
        assert c == c2
        with np.load(c2['sensible']) as arr:
            assert np.array_equal(arr['thresholds'],first_q)
        # Emulate a previous Article cache, preserving its metadata exactly.
        legacy = root/'Functional'/'Article'/'1979-2026'
        legacy.mkdir(parents=True)
        for path in c.values():
            path.rename(legacy/path.name)
            path.with_suffix('.json').rename(legacy/path.with_suffix('.json').name)
        with patch.object(diagnostics, 'load_field', side_effect=AssertionError('Plotter must not read ERA5')):
            old = prepare_stationary_diagnostics(root, cache_only=True, **kwargs)
            assert all(p.parent == legacy for p in old.values())
            images = plot_stationary_spatial(root, ('sensible','latent'), 1979, 2026,
                                             **kwargs, language='en', dpi=75)
            assert len(images) == 6
            assert all(p.stat().st_size > 1500 for p in images)
            assert len(list((root/'Plots'/'Stationary'/'Spatial'/'1979-2026').glob('*.png'))) == 6
            second = root/'second.npy'
            np.save(second,np.random.default_rng(1).normal(0,20,water.shape))
            z = plot_stationary_spatial(root,('sensible',),1979,2026, **kwargs,
                                         second_anomaly_files={'sensible': str(second)},
                                         figures=('isolines',),dpi=75)
            assert len(z)==1
            color_before = {p: p.read_bytes() for p in images}
            bw = plot_stationary_spatial(root, ('sensible','latent'), 1979, 2026,
                                         **kwargs, language='en', dpi=75,
                                         monochrome=True,
                                         second_anomaly_files={'sensible': str(second)})
            assert len(bw) == 6 and set(bw).isdisjoint(images)
            assert all(p.name.endswith('_bw.png') for p in bw)
            assert all(p.read_bytes() == content for p, content in color_before.items())
            for path in bw:
                pixels = mpimg.imread(path)[..., :3]
                assert np.allclose(pixels[..., 0], pixels[..., 1])
                assert np.allclose(pixels[..., 1], pixels[..., 2])
            manifest = bw[0].parent/'figure_manifest_isolines-regions-classes_en_bw.json'
            assert json.loads(manifest.read_text())['monochrome'] is True
            assert (bw[0].parent/'figure_manifest_isolines-regions-classes_en.json').is_file()
        f = prepare_stationary_diagnostics(root,('sensible',),1979,2026,
                                            **kwargs,compute_frequencies=True)
        with np.load(f['sensible']) as arr:
            assert np.allclose(arr['frequencies'][:,water].sum(axis=0), 1,atol=1e-10)


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
        out=plot_copula_contours(root,('sensible','latent'),1979,2026,
                                 output_dir=root,language='en',dpi=75)
        assert len(out)==4 and all(p.exists() for p in out)
        saved = {p: p.read_bytes() for p in out}
        bw = plot_copula_contours(root,('sensible','latent'),1979,2026,
                                  output_dir=root,language='en',dpi=75,monochrome=True)
        assert len(bw)==4 and all(p.name.endswith('_bw.png') for p in bw)
        assert set(bw).isdisjoint(out)
        assert all(p.read_bytes() == content for p, content in saved.items())
        for path in bw:
            pixels = mpimg.imread(path)[..., :3]
            assert np.allclose(pixels[..., 0], pixels[..., 1])
            assert np.allclose(pixels[..., 1], pixels[..., 2])


def test_class_geometry():
    # Hatches must cover precisely the selected cells and no others.
    mask = np.array([[1, 1, 0], [0, 1, 0]], dtype=bool)
    x, y = _class_grid_edges(mask.shape)
    path = _class_mask_path(mask, x, y)
    xx, yy = np.meshgrid((x[:-1] + x[1:])/2, (y[:-1] + y[1:])/2)
    assert np.array_equal(path.contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(mask.shape), mask)
    assert _class_mask_path(np.zeros_like(mask), x, y) is None
    vertical = mask[:, :-1] != mask[:, 1:]
    horizontal = mask[:-1, :] != mask[1:, :]
    edges = _grid_boundary_segments(vertical, horizontal, x, y)
    assert len(edges) == vertical.sum() + horizontal.sum()
    assert np.isin(edges[..., 0], x).all() and np.isin(edges[..., 1], y).all()


if __name__ == '__main__':
    test_pdf_quintiles()
    test_maps_and_reuse()
    test_copula_contours()
    test_class_geometry()
    print('4 test groups passed')
