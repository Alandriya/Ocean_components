"""Gaussian-mixture estimators for scalar and coupled two-variable SDEs.

NP and SP receive precisely the same grouped raw increments. Full-covariance
2D Gaussian mixtures represent cross-covariance; the estimated covariance
is converted to a canonical symmetric PSD diffusion root B with B B.T = Cov/dt.
"""
from pathlib import Path
import numpy as np
from sklearn.mixture import GaussianMixture
from Coefficients.aligned_common import (
    GroupedTransition, prepare_transition, prepare_pair_transition,
    check_series, check_pair_series, load_series, coerce_series, load_water_mask,
    empty_maps, covariance_to_diffusion, pack_b_matrix,
)
from Coefficients.coefficients_cf import count_c_coeff, count_fraction, count_f_separate_coeff


def _validate_settings(n_components, min_group_size, reg_covar, n_init, dt):
    if not isinstance(n_components, (int, np.integer)) or n_components < 1:
        raise ValueError('n_components must be >= 1')
    if min_group_size < 2:
        raise ValueError('min_group_size must be >= 2')
    if not 0 <= reg_covar < float('inf'):
        raise ValueError('reg_covar must be finite and nonnegative')
    if n_init < 1 or dt <= 0:
        raise ValueError('n_init and dt must be positive')


def _fit_gmm(y, n_components, reg_covar, n_init, random_state):
    """Fit a full Gaussian mixture to 1D or 2D increments."""
    fit = GaussianMixture(
        n_components=min(int(n_components), len(y)),
        covariance_type='full', tol=1e-4,
        max_iter=1000, init_params='random', n_init=n_init,
        random_state=random_state, reg_covar=reg_covar,
    )
    fit.fit(y)
    return fit


def estimate_sp(grouped: GroupedTransition, n_components: int = 2,
                min_group_size: int = 2, reg_covar: float = 1e-6,
                n_init: int = 10, random_state: int = 42,
                remove_reg_covar: bool = True, dt: float = 1.0):
    """Return (a, b, b_with_reg, diagnostics) for a 1D grouped transition."""
    if grouped.dimension != 1:
        raise ValueError('estimate_sp expects scalar groups')
    _validate_settings(n_components, min_group_size, reg_covar, n_init, dt)
    a, b = empty_maps(grouped.shape)
    b_reg = np.full(grouped.shape, np.nan)
    af, bf, brf = a.ravel(), b.ravel(), b_reg.ravel()
    non_converged = fitted = 0
    for group in grouped.groups:
        if len(group.increments) < min_group_size:
            continue
        y = group.increments.reshape(-1, 1)
        gm = _fit_gmm(y, n_components, reg_covar, n_init,
                      random_state + group.bin_id)
        non_converged += int(not gm.converged_)
        w, mu, var = gm.weights_, gm.means_[:, 0], gm.covariances_[:, 0, 0]
        mean = float(np.dot(w, mu))
        variance = max(0., float(np.dot(w, var + (mu - mean) ** 2)))
        corrected = max(0., variance - (reg_covar if remove_reg_covar else 0.))
        af[group.flat_indices] = mean / dt
        bf[group.flat_indices] = np.sqrt(corrected / dt)
        brf[group.flat_indices] = np.sqrt(variance / dt)
        fitted += 1
    return a, b, b_reg, {'fitted_bins': fitted, 'non_converged_bins': non_converged}


def estimate_sp_2d(grouped: GroupedTransition, n_components: int = 2,
                   min_group_size: int = 2, reg_covar: float = 1e-6,
                   n_init: int = 10, random_state: int = 42,
                   remove_reg_covar: bool = True, dt: float = 1.0):
    """Return (A,B,B_with_reg,diagnostics), with B channels [00,11,01,10].

    The mixture covariance is
        C = sum_k w_k [Sigma_k + (mu_k-mu)(mu_k-mu)^T].
    Full-covariance GMM contributes reg_covar*I. Subtract it when comparing
    with NP's empirical covariance (if remove_reg_covar is True).
    """
    if grouped.dimension != 2:
        raise ValueError('estimate_sp_2d expects bivariate groups')
    _validate_settings(n_components, min_group_size, reg_covar, n_init, dt)
    a, b = empty_maps(grouped.shape, channels=2)
    b_reg = np.full(b.shape, np.nan)
    af, bf, brf = a.reshape((2, -1)), b.reshape((4, -1)), b_reg.reshape((4, -1))
    non_converged = fitted = 0
    for group in grouped.groups:
        if len(group.increments) < min_group_size:
            continue
        gm = _fit_gmm(group.increments, n_components, reg_covar, n_init,
                      random_state + group.bin_id)
        non_converged += int(not gm.converged_)
        w = gm.weights_
        mu = np.einsum('k,kj->j', w, gm.means_)
        residual = gm.means_ - mu
        covariance = np.einsum('k,kij->ij', w, gm.covariances_)
        covariance += np.einsum('k,ki,kj->ij', w, residual, residual)
        corrected = covariance - (np.eye(2) * reg_covar if remove_reg_covar else 0.)
        root = covariance_to_diffusion(corrected, dt)
        root_reg = covariance_to_diffusion(covariance, dt)
        af[:, group.flat_indices] = (mu / dt)[:, None]
        bf[:, group.flat_indices] = pack_b_matrix(root)[:, None]
        brf[:, group.flat_indices] = pack_b_matrix(root_reg)[:, None]
        fitted += 1
    diagnostics = {'fitted_bins': fitted, 'non_converged_bins': non_converged}
    return a, b, b_reg, diagnostics


def count_semi_1d_AB(files_path_prefix: str, data_array, time_start: int,
                     time_end: int, path: str = 'Coefficients/',
                     quantiles_amount: int = 100, n_components: int = 2,
                     start_index: int = 0, mask=None, min_group_size: int = 2,
                     reg_covar: float = 1e-6, n_init: int = 10,
                     random_state: int = 42, remove_reg_covar: bool = True,
                     overwrite: bool = False, dt: float = 1.0):
    """Save scalar daily A_t, B_t, and B_with_reg_t under Semiparametric/daily."""
    series = coerce_series(data_array)
    check_series(series, time_start, time_end)
    keep = load_water_mask(mask, series.shape[1:])
    target = Path(files_path_prefix) / path / 'Semiparametric' / 'daily'
    target.mkdir(parents=True, exist_ok=True)
    for t in range(time_start + 1, time_end):
        pa = target / f'A_{t + start_index}.npy'
        pb = target / f'B_{t + start_index}.npy'
        pbr = target / f'B_with_reg_{t + start_index}.npy'
        if not overwrite and pa.exists() and pb.exists() and pbr.exists():
            continue
        grouped = prepare_transition(series[t - 1], series[t], quantiles_amount, keep)
        a, b, b_reg, _ = estimate_sp(
            grouped, n_components, min_group_size, reg_covar, n_init,
            random_state + t * 1000, remove_reg_covar, dt,
        )
        np.save(pa, a)
        np.save(pb, b)
        np.save(pbr, b_reg)


def count_semi_2d_AB(files_path_prefix: str, variable_0, variable_1,
                     time_start: int, time_end: int,
                     path: str = 'Coefficients/', quantiles_amount: int = 12,
                     n_components: int = 2, start_index: int = 0, mask=None,
                     min_group_size: int = 2, reg_covar: float = 1e-6,
                     n_init: int = 10, random_state: int = 42,
                     remove_reg_covar: bool = True, overwrite: bool = False,
                     dt: float = 1.0):
    """Save coupled 2D drift and diffusion from two aligned scalar series.

    Input .npy files may have shape (H*W,T) or (T,H,W).
    A_t.npy has shape (2,H,W); B_t.npy and B_with_reg_t.npy are (4,H,W)
    with channel order [00,11,01,10].
    """
    s0 = coerce_series(variable_0)
    s1 = coerce_series(variable_1)
    check_pair_series(s0, s1, time_start, time_end)
    keep = load_water_mask(mask, s0.shape[1:])
    target = Path(files_path_prefix) / path / 'Semiparametric_2d' / 'daily'
    target.mkdir(parents=True, exist_ok=True)
    for t in range(time_start + 1, time_end):
        pa = target / f'A_{t + start_index}.npy'
        pb = target / f'B_{t + start_index}.npy'
        pbr = target / f'B_with_reg_{t + start_index}.npy'
        if not overwrite and pa.exists() and pb.exists() and pbr.exists():
            continue
        grouped = prepare_pair_transition(s0[t - 1], s0[t], s1[t - 1],
                                          s1[t], quantiles_amount, keep)
        a, b, b_reg, _ = estimate_sp_2d(
            grouped, n_components, min_group_size, reg_covar, n_init,
            random_state + t * 1000, remove_reg_covar, dt,
        )
        np.save(pa, a)
        np.save(pb, b)
        np.save(pbr, b_reg)
