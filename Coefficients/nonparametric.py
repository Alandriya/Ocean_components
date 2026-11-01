from pathlib import Path
from typing import Optional
import numpy as np
from aligned_common import GroupedTransition, prepare_transition, check_series, empty_maps


def estimate_np(grouped: GroupedTransition, min_group_size: int = 2):
    """returns maps (a, b), where b=sqrt(E[(dX-E[dX])^2]); dt=1."""
    if min_group_size < 2:
        raise ValueError("min_group_size >= 2 для одинакового режима с GMM")
    a, b = empty_maps(grouped.shape)
    af, bf = a.ravel(), b.ravel()
    for group in grouped.groups:
        if group.increments.size < min_group_size:
            continue
        inc = group.increments
        af[group.flat_indices] = np.mean(inc)
        bf[group.flat_indices] = np.std(inc, ddof=0)
    return a, b


def count_1d_nonparam(
    files_path_prefix: str,
    flux: np.ndarray,
    time_start: int,
    time_end: int,
    path: str = 'Coefficients/',
    start_index: int = 0,
    quantiles_amount: int = 100,
    mask: Optional[np.ndarray] = None,
    min_group_size: int = 2,
    overwrite: bool = False,
):
    """Saving to Nonparametric/daily/A_t.npy and B_t.npy
    uses timestamps t=time_start+1,...,time_end-1.
    """
    check_series(flux, time_start, time_end)
    target = Path(files_path_prefix) / path / 'Nonparametric' / 'daily'
    target.mkdir(parents=True, exist_ok=True)
    for t in range(time_start + 1, time_end):
        pa, pb = target / f'A_{t + start_index}.npy', target / f'B_{t + start_index}.npy'
        if not overwrite and pa.exists() and pb.exists():
            continue
        grouped = prepare_transition(flux[t-1], flux[t], quantiles_amount, mask)
        a, b = estimate_np(grouped, min_group_size=min_group_size)
        np.save(pa, a)
        np.save(pb, b)
    return None
