"""Synthetic regression tests for the experimental components task."""
from datetime import date, timedelta
from pathlib import Path
import csv

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from Coefficients.common import prepare_transition
from Coefficients.spatial_components import (fit_component_window, match_components,
                                             run_components, component_run_parts)
from Plotting.component_plots import plot_components


def _sample():
    """Compute sample for test spatial components."""
    rng = np.random.default_rng(413)
    shape = (20, 20)
    water = np.ones(shape, dtype=bool)
    water[:2, :3] = False
    z = np.indices(shape).sum(axis=0) % 2
    prev = rng.normal(0, 10, shape)
    fields = [prev]
    for _ in range(12):
        increment = np.where(z == 0, -4., 6.) + rng.normal(0,.5,shape)
        fields.append(fields[-1] + increment)
    # Use independent previous states below to avoid sorting classes into separate ranks.
    return np.asarray(fields), water, z


def test_mixture_moments_and_posteriors():
    """Verify that mixture moments and posteriors."""
    rng = np.random.default_rng(10)
    shape = (20,20)
    water = np.ones(shape,dtype=bool)
    water[0] = False
    z = rng.integers(0,2,shape)
    grouped = []
    for _ in range(5):
        prev = rng.uniform(-50,50,shape)
        inc = np.where(z==0,-4., 6.) + rng.normal(0,.45,shape)
        grouped.append(prepare_transition(prev,prev+inc,2,water))
    with threadpool_limits(limits=1):
        d = fit_component_window(grouped,water,n_bins=2,n_init=3,dt=2.,min_group_size=20)
    assert d['supported'].all()
    assert np.allclose(d['a'][:,0],-2,atol=.15)
    assert np.allclose(d['a'][:,1],3,atol=.15)
    assert np.allclose(d['Posterior'][:,water].sum(axis=0),1)
    assert np.all(np.isnan(d['A'][:,~water]))
    assert np.all(d['Posterior'][0,water & (z==0)]>.99)
    assert np.all(d['Posterior'][1,water & (z==1)]>.99)
    assert np.allclose(np.sum(d['P']*d['A'],axis=0)[water],d['A_mix'][water])
    # law of total variance; dt enters between-component term as well
    combined = np.sum(d['P']*(d['B']**2+2*(d['A']-d['A_mix'])**2),axis=0)
    assert np.allclose(combined[water],d['B_with_reg'][water]**2)
    assert d['daily_edges'].shape==(5,3)
    assert np.allclose(d['P'][:, water].sum(axis=0),1)


def test_matching_independent_of_em_order():
    """Verify that matching independent of EM order."""
    old = np.array([[-4.,.5],[6.,1.]])
    order,status,cost,margin = match_components(np.array([6.1,-3.9]),np.array([1.,.5]),np.array([.4,.6]),old)
    assert order.tolist()==[1,0] and status==2 and margin>0
    order,status,_,_ = match_components(np.array([3.,-1.]),np.array([1.,2.]),np.array([.5,.5]))
    assert order.tolist()==[1,0] and status==1


def test_constant_and_missing_data():
    """Verify that constant and missing data."""
    prev = np.ones((6,6))
    water = np.ones(prev.shape,dtype=bool)
    groups = [prepare_transition(prev,prev+2,2,water)]
    result = fit_component_window(groups,water,n_bins=2,min_group_size=10,n_init=1)
    assert not result['supported'].any()
    assert np.isnan(result['A']).all()
    assert np.sum(result['n_all'])==36


def test_disk_workflow_resume_and_plots(tmp_path):
    # Real project layout, including mask; small synthetic temporal slice of full grid.
    """Verify that disk workflow resume and plots."""
    rng=np.random.default_rng(122)
    shape=(161,181)
    water=np.ones(shape,dtype=bool); water[:20,:30]=False
    fields=[rng.normal(0,30,shape)]
    for t in range(8):
        sign=rng.integers(0,2,shape)
        fields.append(fields[-1]+np.where(sign==0,-4.,6.)+rng.normal(0,.5,shape))
    folder=tmp_path/'DATA'/'Fluxes'; folder.mkdir(parents=True)
    np.save(folder/'sensible_2019-2026_grouped.npy',np.asarray(fields,dtype=np.float32))
    (tmp_path/'DATA'/'mask').write_bytes(water.tobytes())
    args=dict(variable='sensible',n_bins=2,window_days=3,step_days=2,
              first_date='2019-01-01',last_date='2019-01-09',spatial_stride=4)
    out=run_components(tmp_path,2019,2026,**args,n_init=2,progress=False)
    windows=sorted(out.glob('window_*.npz'))
    assert len(windows)==3 and windows[0].stem=='window_2019-01-04'
    with np.load(windows[0],allow_pickle=False) as d:
        assert int(d['endpoint_label'])==(date(2019,1,4)-date(1979,1,1)).days+1
        assert d['A'].shape==(2,*shape)
        assert d['supported'].all()
        assert d['n_fit'].sum()<d['n_all'].sum()
    mtimes=[p.stat().st_mtime_ns for p in windows]
    run_components(tmp_path,2019,2026,**args,n_init=2,progress=False)
    assert [p.stat().st_mtime_ns for p in windows]==mtimes
    with pytest.raises(ValueError,match='Changed input/configuration'):
        run_components(tmp_path,2019,2026,**args,n_init=3,progress=False)
    windows[1].unlink()
    run_components(tmp_path,2019,2026,**args,n_init=2,progress=False)
    assert windows[0].stat().st_mtime_ns==mtimes[0]
    assert windows[2].stat().st_mtime_ns!=mtimes[2]
    with (out/'parameters.csv').open() as f:
        assert len(list(csv.DictReader(f)))==3*2*2
    results=plot_components(tmp_path,2019,2026,**args,plot_first_date='2019-01-04',
                            plot_last_date='2019-01-04',track_group=1,dpi=72,progress=False)
    assert len(results)==3 and all(p.stat().st_size>1000 for p in results)


def test_invalid_window():
    """Verify that invalid window."""
    with pytest.raises(ValueError):
        component_run_parts(2019,2026,'sensible',10,30,7,'2019-01-01','2019-01-15')
