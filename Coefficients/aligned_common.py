"""Unified preparation for controlled comparison of 1D NP and SP.

Conventions:
- data[t] has shape (height, width); time step dt=1;
- mask=True (or 1) means a KEPT point (water / reliable data);
  mask=False (or 0) means an EXCLUDED point (land / unreliable data);
- the split into N quantile groups is computed from X[t-1] ONLY
  at points where both X[t-1] and X[t] are finite;
- both estimators receive EXACTLY THE SAME increment arrays X[t]-X[t-1];
- groups with fewer than min_group_size observations are excluded
  in BOTH methods.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class Group:
    bin_id: int
    flat_indices: np.ndarray
    increments: np.ndarray


@dataclass(frozen=True)
class GroupedTransition:
    shape: Tuple[int, int]
    groups: Tuple[Group, ...]
    n_valid_pairs: int
    n_total_positions: int
    n_requested_bins: int
    n_occupied_bins: int
    n_singleton_bins: int
    edges: np.ndarray


def prepare_transition(
    x_prev: np.ndarray,
    x_curr: np.ndarray,
    quantiles_amount: int,
    mask: Optional[np.ndarray] = None,
) -> GroupedTransition:
    """Prepare paired data once for BOTH algorithms.

    Quantile edges may coincide when values are repeated.
    Empty intervals are allowed; boundary values are assigned unambiguously.

    Parameters
    ----------
    mask : optional array of the same shape as x_prev
        Water / keep indicator: 1 (True) = keep, 0 (False) = exclude
        (e.g. land or unreliable data).
    """
    previous = np.asarray(x_prev, dtype=np.float64)
    current = np.asarray(x_curr, dtype=np.float64)
    if previous.shape != current.shape or previous.ndim != 2:
        raise ValueError("X[t-1] and X[t] must be 2D arrays of the same shape")
    if (
        not isinstance(quantiles_amount, (int, np.integer))
        or isinstance(quantiles_amount, bool)
        or quantiles_amount < 1
    ):
        raise ValueError("quantiles_amount must be a positive integer")

    valid = np.isfinite(previous) & np.isfinite(current)
    if mask is not None:
        keep = np.asarray(mask, dtype=bool)
        if keep.shape != previous.shape:
            raise ValueError(
                "mask must have shape (height, width); "
                "1/True = water (keep), 0/False = land (exclude)"
            )
        valid &= keep

    prev_flat = previous.ravel()
    curr_flat = current.ravel()
    positions = np.flatnonzero(valid.ravel())
    if len(positions) == 0:
        return GroupedTransition(
            previous.shape, tuple(), 0, previous.size,
            int(quantiles_amount), 0, 0, np.array([]),
        )

    previous_valid = prev_flat[positions]
    increments = curr_flat[positions] - previous_valid
    # Extra safeguard against overflow when subtracting two finite floats.
    finite_delta = np.isfinite(increments)
    positions = positions[finite_delta]
    previous_valid = previous_valid[finite_delta]
    increments = increments[finite_delta]
    if len(positions) == 0:
        return GroupedTransition(
            previous.shape, tuple(), 0, previous.size,
            int(quantiles_amount), 0, 0, np.array([]),
        )

    edges = np.quantile(
        previous_valid, np.linspace(0.0, 1.0, quantiles_amount + 1)
    )
    bin_indices = np.searchsorted(edges, previous_valid, side="right") - 1
    bin_indices = np.clip(bin_indices, 0, quantiles_amount - 1)

    # A single sort replaces repeated searches over all points per bin.
    order = np.argsort(bin_indices, kind="stable")
    sorted_ids = bin_indices[order]
    ids, starts, counts = np.unique(
        sorted_ids, return_index=True, return_counts=True
    )
    groups = tuple(
        Group(
            bin_id=int(bin_id),
            flat_indices=positions[order[start:start + count]],
            increments=increments[order[start:start + count]],
        )
        for bin_id, start, count in zip(ids, starts, counts)
    )
    return GroupedTransition(
        shape=previous.shape,
        groups=groups,
        n_valid_pairs=int(len(positions)),
        n_total_positions=int(previous.size),
        n_requested_bins=int(quantiles_amount),
        n_occupied_bins=len(groups),
        n_singleton_bins=sum(len(g.increments) == 1 for g in groups),
        edges=edges,
    )


def check_series(data_array: np.ndarray, time_start: int, time_end: int) -> None:
    if len(data_array.shape) != 3:
        raise ValueError(
            "Expected data_array of shape [time_steps, height, width]"
        )
    if not 0 <= time_start < time_end <= data_array.shape[0]:
        raise ValueError(
            "Must satisfy 0 <= time_start < time_end <= number of time steps"
        )
    if time_end - time_start < 2:
        raise ValueError(
            "At least one transition X[t-1] -> X[t] is required"
        )


def empty_maps(shape):
    return (
        np.full(shape, np.nan, dtype=float),
        np.full(shape, np.nan, dtype=float),
    )