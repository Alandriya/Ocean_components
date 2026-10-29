"""Small reproducibility tests; synthetic arrays, no ERA5 data required."""
from __future__ import annotations

from itertools import combinations_with_replacement
from pathlib import Path
import numpy as np
import pytest

from Data_processing.data_processing import scale_to_bins, quantile_bin_indices
from Eigenvalues.eigenvalues import (
    get_eig, _moment_matrix, count_eigenvalues_pair,
    reconstruct_b2_map, matrix_to_map,
)
from main_parts import run_eigenvalues_for_period


def test_scale_bins_constant_ties_and_bad_values():
    x = np.array([7, 7, 7, np.nan, np.inf, -np.inf], dtype=float)
    y, q = scale_to_bins(x, 4)
    assert y.dtype.kind == 'f' and len(q) == 5
    np.testing.assert_allclose(y[:3], [7,7,7])
    assert np.isnan(y[3:]).all()
    np.testing.assert_array_equal(quantile_bin_indices(x, q), [3,3,3,-1,-1,-1])


def test_scale_integer_quantiles_maximum_included():
    x = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    y, q = scale_to_bins(x, 2)
    np.testing.assert_allclose(q, [0,2,4])
    np.testing.assert_allclose(y, [1,1,3,3,3])
    np.testing.assert_array_equal(quantile_bin_indices([0,2,4,np.nan], q), [0,1,1,-1])
    with pytest.raises(ValueError): scale_to_bins([np.nan, np.inf])
    with pytest.raises(ValueError): scale_to_bins([1,2], bins=0)


def test_joint_matrix_and_missing_t_plus_one():
    x0=np.array([0.0, 0.0, 1.0, 1.0])
    x1=np.array([1.0, 3.0, 4.0, np.nan])
    # Finite increments: [1,3,3] => diag [5,9], not [5,nan].
    b, counts=_moment_matrix(x0,x1,x0,x1,[0,1,2],[0,1,2],dt=1)
    np.testing.assert_allclose(b, np.diag([5,9]))
    np.testing.assert_array_equal(counts, np.diag([2,1]))
    eig,vec = get_eig(b,('sensible','sensible'))
    np.testing.assert_allclose(eig, [9,5])
    np.testing.assert_allclose(vec.T@vec,np.eye(2),atol=1e-12)


def test_svd_not_square_eigenvalues():
    b=np.array([[0,6.0],[2,0]])
    sig,u = get_eig(b,('sensible','latent'))
    np.testing.assert_allclose(sig,[6,2])
    np.testing.assert_allclose(u.T@u,np.eye(2))


def test_pair_saves_complete_arrays_and_recovers_b2(tmp_path):
    x=np.array([[0,1,2],[0,1,2],[1,3,4],[1,2,4]],dtype=float)
    y=np.array([[1,2,3],[0,2,2],[1,1,2],[0,0,0]],dtype=float)
    _,qx=scale_to_bins(x,2)
    _,qy=scale_to_bins(y,2)
    count_eigenvalues_pair(str(tmp_path),x,x,qx,qx,2,1,('sensible','sensible'),(2,2),2,1,
                           save_counts=True)
    folder=tmp_path/'Eigenvalues'/'sensible-sensible'
    e=np.load(folder/'eigenvalues_1.npy')
    ev=np.load(folder/'eigenvectors_1.npy')
    c=np.load(folder/'Cn_1.npy')
    np.testing.assert_allclose(c,(ev[:,:1]*e[:1])@ev[:,:1].T)
    np.testing.assert_allclose(np.load(folder/'B_map_1.npy')**2,
                               np.load(folder/'B2_map_1.npy'),equal_nan=True)
    b2,amp=reconstruct_b2_map(e,ev,x[:,0],qx,(2,2),n_components=1)
    np.testing.assert_allclose(b2,np.load(folder/'B2_map_1.npy'),equal_nan=True)
    count_eigenvalues_pair(str(tmp_path),x,y,qx,qy,2,1,('sensible','latent'),(2,2),2,1)
    mixed=tmp_path/'Eigenvalues'/'sensible-latent'
    s=np.load(mixed/'singular_values_1.npy')
    u=np.load(mixed/'left_singularvectors_1.npy')
    v=np.load(mixed/'right_singularvectors_1.npy')
    bn=np.load(mixed/'Bn_1.npy')
    np.testing.assert_allclose(bn,(u[:,:1]*s[:1])@v[:,:1].T)


def test_main_runner_calculates_ten_pairs(tmp_path):
    # Four tiny fields, all same time axis, with 3 daily points.
    paths={
        'sensible': 'DATA/Fluxes/sensible_grouped_2019-2026.npy',
        'latent': 'DATA/Fluxes/latent_grouped_2019-2026.npy',
        'SST': 'DATA/SST/SST_2019-2026_grouped.npy',
        'Pressure': 'DATA/Pressure/PRESS_2019-2026_grouped.npy',
    }
    for k,(name,rel) in enumerate(paths.items()):
        path=tmp_path/rel;path.parent.mkdir(parents=True,exist_ok=True)
        a=np.array([[0,1,2],[1,2,3],[2,3,4],[3,4,5]],float)+k
        np.save(path,a)
    result=run_eigenvalues_for_period(str(tmp_path),2019,2026,2,1,spatial_shape=(2,2))
    assert result['first_index']==14611 and result['last_index']==14612
    assert len(result['pairs'])==10
    assert len(list((tmp_path/'Eigenvalues').iterdir()))==10
    assert (tmp_path/'Eigenvalues'/'SST-Pressure'/'Bn_14611.npy').exists()


def test_plotter_supports_mixed_singular_vectors(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    from Plotting.plot_eigenvalues import plot_eigenvalues
    arr = np.array([[0,1],[0,1],[1,2],[1,2]], float)
    _, q = scale_to_bins(arr,2)
    count_eigenvalues_pair(str(tmp_path), arr, arr + 1, q, (np.asarray(q)+1).tolist(),
                           2, 14611, ('sensible','latent'), (2,2), n_components=1)
    # array2 supplied in plotting corresponds to right singular vectors.
    plot_eigenvalues(str(tmp_path)+'/', 1, np.ones(4,bool), 0, 1, 14611,
                     arr+1, (np.asarray(q)+1).tolist(),
                     ('sensible','latent'), (2,2))
    assert (tmp_path/'videos/Eigenvalues/sensible-latent/Lambdas_14611.png').exists()
