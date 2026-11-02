"""Launch aligned NP and SP estimation over an entire period with resumable chunks."""
from pathlib import Path
from typing import Tuple
import numpy as np
from Coefficients.aligned_common import coerce_series, load_water_mask
from Coefficients.nonparametric_aligned import count_1d_nonparam, count_2d_nonparam
from Coefficients.semiparametric_aligned import count_semi_1d_AB, count_semi_2d_AB


def run_decade_coefficients(
    files_path_prefix: str,
    array1: np.ndarray,
    array2: np.ndarray,
    mask: np.ndarray,
    names: Tuple[str, str],
    start_year: int,
    end_year: int,
    start_index: int,
    n_bins: int = 250,
    n_components: int = 3,
    dt: float = 1.0,
    overwrite: bool = False,
    chunk_transitions: int = 7,
    min_group_size: int = 2,
    n_init: int = 10,
    run_1d: bool = True,
    run_2d: bool = False,
    n_bins_2d: int = 12,
) -> None:
    """Estimate both scalar series and optionally their joint bivariate process.

    Both NP and SP use identical quantile groups and raw increments.
    An output label for transition X[t-1] -> X[t] is t + start_index.
    Existing complete daily results are skipped unless overwrite=True.
    """
    x0, x1 = coerce_series(array1), coerce_series(array2)
    if x0.shape != x1.shape:
        raise ValueError(f'Input series must have identical shapes, got {x0.shape} and {x1.shape}')
    if x0.shape[0] < 2:
        raise ValueError('At least two observations are required')
    if len(names) != 2 or not all(isinstance(n, str) and n for n in names):
        raise ValueError('names must contain two nonempty variable names')
    if names[0] == names[1]:
        raise ValueError('Variable names must differ to avoid overwriting 1D output paths')
    if not isinstance(chunk_transitions, int) or chunk_transitions < 1:
        raise ValueError('chunk_transitions must be a positive integer')
    if not run_1d and not run_2d:
        raise ValueError('At least one of run_1d or run_2d must be True')
    water = load_water_mask(mask, x0.shape[1:])
    if not water.any():
        raise ValueError('The mask contains no water cells (mask=1)')

    prefix = Path(files_path_prefix)
    base = Path('Coefficients') / f'aligned_{start_year}-{end_year}'
    print(f'Input: {x0.shape}; water cells: {int(water.sum())}/{water.size}', flush=True)
    print(f'Period: {start_year}-{end_year}; transitions: {x0.shape[0]-1}; '
          f'GMM components: {n_components}', flush=True)
    if run_1d:
        print(f'1D quantile groups: {n_bins}', flush=True)
    if run_2d:
        print(f'2D quantile groups: {n_bins_2d} per axis (up to {n_bins_2d**2} joint groups)', flush=True)

    total_transitions = x0.shape[0] - 1
    for chunk_start in range(0, total_transitions, chunk_transitions):
        chunk_end = min(x0.shape[0], chunk_start + chunk_transitions + 1)
        # The wrapper's [time_start+1, time_end) convention retains all transitions.
        first_label = chunk_start + 1 + start_index
        last_label = chunk_end - 1 + start_index
        print(f'Calculating transitions {first_label}..{last_label} '
              f'({last_label-start_index}/{total_transitions})', flush=True)

        if run_1d:
            for variable, series in ((names[0], x0), (names[1], x1)):
                relative_output = base / f'1d_bins_{n_bins}' / variable
                count_1d_nonparam(
                    str(prefix), series, chunk_start, chunk_end,
                    path=str(relative_output), start_index=start_index,
                    quantiles_amount=n_bins, mask=water,
                    min_group_size=min_group_size, overwrite=overwrite, dt=dt,
                )
                count_semi_1d_AB(
                    str(prefix), series, chunk_start, chunk_end,
                    path=str(relative_output), start_index=start_index,
                    quantiles_amount=n_bins, n_components=n_components,
                    mask=water, min_group_size=min_group_size,
                    n_init=n_init, remove_reg_covar=True,
                    overwrite=overwrite, dt=dt,
                )

        if run_2d:
            pair_dir = base / f'2d_bins_{n_bins_2d}' / f'{names[0]}_{names[1]}'
            count_2d_nonparam(
                str(prefix), x0, x1, chunk_start, chunk_end,
                path=str(pair_dir), start_index=start_index,
                quantiles_amount=n_bins_2d, mask=water,
                min_group_size=min_group_size, overwrite=overwrite, dt=dt,
            )
            count_semi_2d_AB(
                str(prefix), x0, x1, chunk_start, chunk_end,
                path=str(pair_dir), start_index=start_index,
                quantiles_amount=n_bins_2d, n_components=n_components,
                mask=water, min_group_size=min_group_size,
                n_init=n_init, remove_reg_covar=True,
                overwrite=overwrite, dt=dt,
            )

    print(f'All coefficient calculations finished. Output: {prefix/base}', flush=True)
