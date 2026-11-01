from pathlib import Path
from typing import Optional
import numpy as np
from sklearn.mixture import GaussianMixture
from aligned_common import GroupedTransition, prepare_transition, check_series, empty_maps


def estimate_sp(
    grouped: GroupedTransition,
    n_components: int = 2,
    min_group_size: int = 2,
    reg_covar: float = 1e-6,
    n_init: int = 10,
    random_state: int = 42,
    remove_reg_covar: bool = True,
):
    """returns (a, b, b_reg, diagnostics).

    b_reg — sqrt(var) from GMM with reg_covar.
    b — corrected for comparing with NP.
    Var(GMM) = Var_emp(dX) + reg_covar at M-step of GMM.
    """
    if not isinstance(n_components, int) or n_components < 1:
        raise ValueError("n_components >= 1")
    if min_group_size < 2:
        raise ValueError("min_group_size >= 2  for GMM")
    if reg_covar < 0:
        raise ValueError("reg_covar can't be negative")
    if n_init < 1:
        raise ValueError("n_init >= 1")
    a, b = empty_maps(grouped.shape)
    b_reg = np.full(grouped.shape, np.nan)
    af, bf, brf = a.ravel(), b.ravel(), b_reg.ravel()
    non_converged = 0
    fitted_bins = 0
    for group in grouped.groups:
        increments = group.increments
        if increments.size < min_group_size:
            continue
        k = min(int(n_components), int(increments.size))
        gm = GaussianMixture(
            n_components=k,
            covariance_type="spherical",
            tol=1e-4,
            max_iter=1000,
            init_params="random",
            n_init=n_init,
            random_state=random_state + group.bin_id,
            reg_covar=reg_covar,
        )
        try:
            gm.fit(increments.reshape(-1, 1))
        except (ValueError, FloatingPointError) as error:
            raise RuntimeError(
                f"GMM error for group {group.bin_id}, n={len(increments)}: {error}"
            ) from error
        if not gm.converged_:
            non_converged += 1

        w = gm.weights_.ravel()
        mu = gm.means_.ravel()
        var = gm.covariances_.ravel()
        mixture_mean = float(np.dot(w, mu))
        # Stable form or Var = sum(w * (var_k + (mu_k - mean)^2)).
        # Equivalent to E[dX^2] - E[dX]^2, but avoids a loss of precision in cases with large increment values and a small spread.
        variance_with_reg = max(0., float(np.dot(w, var + (mu - mixture_mean)**2)))
        variance_adjusted = max(0., variance_with_reg - (reg_covar if remove_reg_covar else 0.))

        af[group.flat_indices] = mixture_mean
        bf[group.flat_indices] = np.sqrt(variance_adjusted)
        brf[group.flat_indices] = np.sqrt(variance_with_reg)
        fitted_bins += 1
    diagnostics = {"fitted_bins": fitted_bins, "non_converged_bins": non_converged}
    return a, b, b_reg, diagnostics


def count_semi_1d_AB(
    files_path_prefix: str,
    data_array: np.ndarray,
    time_start: int,
    time_end: int,
    path: str = 'Coefficients/',
    quantiles_amount: int = 100,
    n_components: int = 2,
    start_index: int = 0,
    mask: Optional[np.ndarray] = None,
    min_group_size: int = 2,
    reg_covar: float = 1e-6,
    n_init: int = 10,
    random_state: int = 42,
    remove_reg_covar: bool = True,
    overwrite: bool = False,
):
    """Saving to Semiparametric/daily/A_t.npy, B_t.npy and B_with_reg_t.npy.

    Best to compare with NP corrected version of B (after subtracting reg_covar).
    B_with_reg — default version after EM with GMM regularization.
    """
    check_series(data_array, time_start, time_end)
    target = Path(files_path_prefix) / path / 'Semiparametric' / 'daily'
    target.mkdir(parents=True, exist_ok=True)
    for t in range(time_start + 1, time_end):
        pa = target / f'A_{t + start_index}.npy'
        pb = target / f'B_{t + start_index}.npy'
        pbr = target / f'B_with_reg_{t + start_index}.npy'
        if not overwrite and pa.exists() and pb.exists() and pbr.exists():
            continue
        grouped = prepare_transition(data_array[t-1], data_array[t], quantiles_amount, mask)
        a, b, b_reg, _ = estimate_sp(
            grouped, n_components=n_components, min_group_size=min_group_size,
            reg_covar=reg_covar, n_init=n_init, random_state=random_state + t * 1000,
            remove_reg_covar=remove_reg_covar,
        )
        np.save(pa, a)
        np.save(pb, b)
        np.save(pbr, b_reg)
    return None
