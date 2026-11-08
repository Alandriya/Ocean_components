"""Empirical SDE coefficient bounds for named 1D/2D variable pairs.

Computes *trimmed empirical bounds*, not a proof of global Lipschitz
continuity or existence/uniqueness.  Coefficient B must be the Ito diffusion
amplitude (NOT a conditional second moment, covariance, or B**2).

For pairs X=(X1,X2): ||a(x)-a(y)||_2 + ||B(x)-B(y)||_F
                     <= K1*||x-y||_2 for ||x-y|| >= epsilon.
We use the conservative componentwise oscillation bound on the selected
empirical coefficient range. The linear-growth bound is
||a(x)||_2**2 + ||B(x)||_F**2 <= K2*(1+||x||_2**2).

The 0.03 and 0.97 quantiles are computed for each coefficient component
across all finite, ocean-masked observations in successive 1000-day windows.
The period envelope is the minimum lower / maximum upper quantile across
windows, exactly matching the window-envelope construction in the thesis.
This discards tail observations; K1/K2 are therefore EMPIRICAL, not global.

The code never treats Eigenvalues/Bn_map as a diffusion matrix.
"""
from __future__ import annotations

import csv
import json
import math
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import combinations_with_replacement
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

PERIODS = ((1979, 1989), (1989, 1999), (1999, 2009),
           (2009, 2019), (2019, 2026))
# Consistent with main_parts.FIELD_PATHS (both historical directory layouts).
FIELD_PATHS = {
    "sensible": ("DATA/Fluxes/sensible_grouped_{period}.npy", "Fluxes/sensible_grouped_{period}.npy"),
    "latent": ("DATA/Fluxes/latent_grouped_{period}.npy", "Fluxes/latent_grouped_{period}.npy"),
    "SST": ("DATA/SST/SST_{period}_grouped.npy", "SST/SST_{period}_grouped.npy"),
    "Pressure": ("DATA/Pressure/PRESS_{period}_grouped.npy", "Pressure/PRESS_{period}_grouped.npy"),
}


@dataclass(frozen=True)
class PairSource:
    """Coefficient inputs relative to root.

    Pattern tokens: {pair}, {left}, {right}, {period}, {index},
    {first_label}, {last_label}. For storage='collected', the two file
    patterns point at one merged period array each (no /daily/ folder).
    Its leading time axis is indexed by [first_label, last_label] inclusive.
    state_dimension=2 handles 2D estimators even for X-X pairs.

    layout=last:  A [H,W,N], B [H,W,N,N] or [H,W,N*N]
    layout=first: A [N,H,W], B [N,N,H,W] or [N*N,H,W]
    For N=1, A/B [H,W] are accepted in either layout.
    diffusion='diagonal' also accepts B [H,W,N] / [N,H,W] and expands
    it to diagonal diffusion. It does NOT estimate cross-diffusion.
    """
    a_pattern: str
    b_pattern: str
    layout: str = "last"
    diffusion: str = "matrix"
    storage: str = "daily"
    state_dimension: int | None = None

    def __post_init__(self):
        if self.layout not in ("last", "first"):
            raise ValueError("layout must be 'last' or 'first'")
        if self.diffusion not in ("matrix", "diagonal"):
            raise ValueError("diffusion must be 'matrix' or 'diagonal'")
        if self.storage not in ("daily", "collected"):
            raise ValueError("storage must be 'daily' or 'collected'")
        if self.state_dimension is not None and (not isinstance(self.state_dimension, int)
                                                 or isinstance(self.state_dimension, bool)
                                                 or self.state_dimension < 1):
            raise ValueError("state_dimension must be a positive integer or None")


DEFAULT_SOURCE = PairSource(
    "Coeff_data/{pair}/A_{index}.npy",
    "Coeff_data/{pair}/B_{index}.npy",
)
# The original bivariate coefficient output is stored without a pair directory.
LEGACY_SENSIBLE_LATENT = PairSource("Coeff_data/A_{index}.npy", "Coeff_data/B_{index}.npy")


def pair_key(names: tuple[str, str]) -> str:
    if len(names) != 2 or not all(isinstance(x, str) and x for x in names):
        raise ValueError("pair requires two nonempty variable names")
    if any("/" in name or "\\" in name or name in (".", "..") for name in names):
        raise ValueError("unsafe variable name")
    return "-".join(names)


def _format_file(root: Path, pattern: str, names: tuple[str, str], period: str,
                 index: int, *, first_label: int | None = None,
                 last_label: int | None = None) -> Path:
    result = pattern.format(pair=pair_key(names), left=names[0], right=names[1],
                            period=period, index=index,
                            first_label=index if first_label is None else first_label,
                            last_label=index if last_label is None else last_label)
    path = Path(result)
    # Absolute paths can be used when deliberately configured by the caller.
    return path if path.is_absolute() else root / path


def _flat_coefficients(raw: np.ndarray, *, n: int, spatial_shape: tuple[int, int],
                       layout: str, kind: str, diffusion: str = "matrix") -> np.ndarray:
    """Returns [H*W, n] for drift or [H*W,n*n] for diffusion."""
    h, w = spatial_shape
    x = np.asarray(raw, dtype=np.float64)
    if n == 1 and x.shape == (h, w):
        return x.reshape(-1, 1)
    # Historical legacy files can be flattened [H*W,channels] / [channels,H*W].
    channels = n if kind == "a" or diffusion == "diagonal" else n*n
    if x.shape == (h*w, channels):
        flat = x
    elif x.shape == (channels, h*w):
        flat = x.T
    else:
        flat = None
    if flat is not None:
        if kind == "a" or diffusion == "matrix":
            return flat
        m = np.zeros((h*w, n, n), dtype=np.float64)
        for k in range(n):
            m[:, k, k] = flat[:, k]
        return m.reshape(h*w, n*n)
    if kind == "a":
        expected = (h, w, n) if layout == "last" else (n, h, w)
        if x.shape != expected:
            raise ValueError(f"drift: expected {expected}, got {x.shape}")
        return x.reshape(-1, n) if layout == "last" else np.moveaxis(x, 0, -1).reshape(-1, n)
    # B is an Ito diffusion AMPLITUDE; accepting 2D full matrix or explicit diagonal.
    if diffusion == "diagonal":
        expected = (h, w, n) if layout == "last" else (n, h, w)
        if x.shape != expected:
            raise ValueError(f"diagonal B: expected {expected}, got {x.shape}")
        diag = x.reshape(-1, n) if layout == "last" else np.moveaxis(x, 0, -1).reshape(-1, n)
        m = np.zeros((h*w, n, n), dtype=np.float64)
        for k in range(n):
            m[:, k, k] = diag[:, k]
        return m.reshape(h*w, n*n)
    expected = ((h, w, n, n), (h, w, n*n)) if layout == "last" else ((n, n, h, w), (n*n, h, w))
    if x.shape not in expected:
        raise ValueError(f"matrix B: expected one of {expected}, got {x.shape}")
    if layout == "last":
        return x.reshape(-1, n*n)
    if x.ndim == 4:
        return np.moveaxis(x, (0, 1), (-2, -1)).reshape(-1, n*n)
    return np.moveaxis(x, 0, -1).reshape(-1, n*n)


def _pair_dimension(names: tuple[str, str], source: PairSource) -> int:
    return (source.state_dimension if source.state_dimension is not None
            else (1 if names[0] == names[1] else 2))


def _prepare_pair_maps(a_raw: np.ndarray, b_raw: np.ndarray,
                       names: tuple[str, str], source: PairSource,
                       spatial_shape: tuple[int, int],
                       mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Validate one map slice and discard non-finite / masked points in sync."""
    n = _pair_dimension(names, source)
    a = _flat_coefficients(a_raw, n=n, spatial_shape=spatial_shape,
                           layout=source.layout, kind="a")
    b = _flat_coefficients(b_raw, n=n, spatial_shape=spatial_shape,
                           layout=source.layout, kind="b", diffusion=source.diffusion)
    keep = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    if mask is not None:
        if mask.shape != spatial_shape:
            raise ValueError(f"mask shape {mask.shape} != {spatial_shape}")
        keep &= mask.ravel()
    return a[keep], b[keep]


def load_pair_maps(root: Path, names: tuple[str, str], period: str, index: int,
                   source: PairSource, spatial_shape: tuple[int, int],
                   mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load one daily map (storage='daily'); kept for backward compatibility."""
    if source.storage != "daily":
        raise ValueError("load_pair_maps expects daily source; use estimate_period for collected data")
    ap = _format_file(root, source.a_pattern, names, period, index)
    bp = _format_file(root, source.b_pattern, names, period, index)
    if not ap.is_file() or not bp.is_file():
        raise FileNotFoundError(f"coefficient files not found at index {index}: {ap} ; {bp}")
    return _prepare_pair_maps(np.load(ap, allow_pickle=False),
                              np.load(bp, allow_pickle=False), names,
                              source, spatial_shape, mask)


def _open_collected_maps(root: Path, names: tuple[str, str], period: str,
                         first_label: int, last_label: int, source: PairSource,
                         spatial_shape: tuple[int, int]) -> tuple[np.memmap, np.memmap]:
    """Open each decade array once, without allocating its complete payload.

    Expected shapes for state_dimension=2, layout='first':
    A: (T, 2, H, W), B: (T, 4, H, W), where T=last-first+1.
    """
    ap = _format_file(root, source.a_pattern, names, period, first_label,
                      first_label=first_label, last_label=last_label)
    bp = _format_file(root, source.b_pattern, names, period, first_label,
                      first_label=first_label, last_label=last_label)
    if not ap.is_file() or not bp.is_file():
        raise FileNotFoundError(f"collected coefficients missing for {names} {period}: {ap} ; {bp}")
    a = np.load(ap, mmap_mode="r", allow_pickle=False)
    b = np.load(bp, mmap_mode="r", allow_pickle=False)
    count = last_label - first_label + 1
    if a.ndim < 3 or b.ndim < 3 or a.shape[0] != count or b.shape[0] != count:
        raise ValueError(
            f"Collected A/B must have T={count} daily maps for labels "
            f"{first_label}-{last_label}. Got A {a.shape} ({ap}), B {b.shape} ({bp})"
        )
    # Validate channel/spatial axes on the first map, not only the time axis.
    _prepare_pair_maps(a[0], b[0], names, source, spatial_shape, None)
    return a, b


def read_mask(root: Path, spatial_shape: tuple[int, int], policy: str = "auto") -> np.ndarray | None:
    if policy == "none":
        return None
    p = next((x for x in (root / "DATA" / "mask", root / "mask") if x.is_file()), None)
    if p is None:
        if policy == "required":
            raise FileNotFoundError("ocean mask missing")
        return None
    size = int(np.prod(spatial_shape))
    raw = p.read_bytes()
    if len(raw) < size:
        raise ValueError(f"mask {p} has fewer than {size} bytes")
    return np.frombuffer(raw[:size], dtype=np.bool_).copy().reshape(spatial_shape)


def _field_day_counts(root: Path, names: tuple[str, str], period: str,
                      spatial_shape: tuple[int, int],
                      fields: Mapping[str, Sequence[str]]) -> dict[str, int]:
    """Return the available time length of each field, without loading its values.

    All period files are assumed to start on the same date. If one ends sooner,
    only the common prefix is available for joint coefficient calculations.
    """
    counts = {}
    n_space = int(np.prod(spatial_shape))
    for name in dict.fromkeys(names):
        if name not in fields:
            raise ValueError(f"Missing raw data path configuration for {name!r}")
        opts = [root / fmt.format(period=period) for fmt in fields[name]]
        path = next((p for p in opts if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"raw field {name} {period} missing; tried {opts}")
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        if arr.ndim != 2:
            raise ValueError(f"{path}: expected 2D (space,time) or (time,space), got {arr.shape}")
        if arr.shape[0] == n_space:
            counts[name] = int(arr.shape[1])
        elif arr.shape[1] == n_space:
            counts[name] = int(arr.shape[0])
        else:
            raise ValueError(f"{path}: spatial extent {n_space} absent from shape {arr.shape}")
    if min(counts.values()) < 2:
        raise ValueError(f"at least 2 shared observations required for {names}: {counts}")
    return counts


def actual_days(root: Path, names: tuple[str, str], period: str,
                spatial_shape: tuple[int, int],
                fields: Mapping[str, Sequence[str]] = FIELD_PATHS) -> int:
    """Use the shortest available time series for a named pair.

    Longer fields are aligned by their shared prefix; excess trailing days are
    ignored, rather than raising ValueError on unequal observation counts.
    """
    return min(_field_day_counts(Path(root), names, period, spatial_shape, fields).values())


def constants_from_bounds(a_lo: np.ndarray, a_hi: np.ndarray,
                          b_lo: np.ndarray, b_hi: np.ndarray,
                          epsilon: float) -> dict[str, float]:
    """Conservative componentwise bound for retained coefficient values."""
    if not (math.isfinite(epsilon) and epsilon > 0):
        raise ValueError("epsilon must be finite and positive")
    arrays = [np.asarray(v, dtype=float).ravel() for v in (a_lo, a_hi, b_lo, b_hi)]
    al, ah, bl, bh = arrays
    if not (al.size == ah.size >= 1 and bl.size == bh.size >= 1):
        raise ValueError("lower/upper shapes must match")
    if not all(np.isfinite(x).all() for x in arrays) or (al > ah).any() or (bl > bh).any():
        raise ValueError("all component bounds must be finite and ordered")
    k1 = (np.linalg.norm(ah - al) + np.linalg.norm(bh - bl)) / epsilon
    k2 = np.sum(np.maximum(np.abs(al), np.abs(ah))**2) + np.sum(np.maximum(np.abs(bl), np.abs(bh))**2)
    return {"K1": float(k1), "K2": float(k2), "K": float(max(k1, k2))}


def _window_quantiles(root: Path, names: tuple[str, str], period: str,
                      start: int, stop: int, source: PairSource,
                      spatial_shape: tuple[int, int], mask: np.ndarray | None,
                      q_low: float, q_high: float, temp_dir: str | None,
                      collected: tuple[np.ndarray, np.ndarray] | None = None,
                      collected_first_label: int | None = None) -> dict[str, Any]:
    """Exact pooled quantiles with bounded RAM, writing temporary float64 streams.

    Uses disk space ~= (n+n*n)*8*N_valid values per 1000-day window.
    Each component is selected/quantiled separately to bound peak RAM.
    """
    n = _pair_dimension(names, source)
    na, nb = n, n*n
    used_days = 0
    used_points = 0
    with tempfile.TemporaryDirectory(dir=temp_dir) as td:
        with ExitStack() as stack:
            streams = [stack.enter_context(open(Path(td)/f"component_{i}.bin", "wb"))
                       for i in range(na+nb)]
            for index in range(start, stop):
                if collected is None:
                    a, b = load_pair_maps(root, names, period, index,
                                          source, spatial_shape, mask)
                else:
                    offset = index - collected_first_label
                    a, b = _prepare_pair_maps(collected[0][offset], collected[1][offset],
                                              names, source, spatial_shape, mask)
                if len(a) == 0:
                    continue
                for j in range(na):
                    np.ascontiguousarray(a[:, j]).tofile(streams[j])
                for j in range(nb):
                    np.ascontiguousarray(b[:, j]).tofile(streams[na+j])
                used_days += 1
                used_points += len(a)
        if used_points == 0:
            raise ValueError(f"No finite coefficient maps for {names} {period}, days {start}..{stop-1}")
        lo, hi = [], []
        for j in range(na+nb):
            x = np.memmap(Path(td)/f"component_{j}.bin", dtype=np.float64,
                          mode="r", shape=(used_points,))
            lo.append(float(np.quantile(x, q_low)))
            hi.append(float(np.quantile(x, q_high)))
            del x
    return {"start_index": start, "end_index_inclusive": stop-1,
            "n_days": used_days, "n_points": used_points,
            "a_lower": lo[:na], "a_upper": hi[:na],
            "b_lower": lo[na:], "b_upper": hi[na:]}


def estimate_period(root: Path | str, names: tuple[str, str], start_year: int, end_year: int,
                    source: PairSource, *, window_days: int = 1000,
                    q_low: float = 0.03, q_high: float = 0.97, epsilon: float = 0.1,
                    spatial_shape: tuple[int, int] = (161, 181),
                    mask: np.ndarray | None = None, temp_dir: str | None = None,
                    fields: dict[str, tuple[str, ...]] = FIELD_PATHS,
                    coefficient_index_shift: int = 0,
                    skip_last_day_latest: bool = True) -> dict[str, Any]:
    """Calculate each requested period independently, with pooled 1000-day quantiles.

    coefficient_index_shift=0 keeps legacy label A_1 for the first 1979
    transition. The current Coefficients.runner uses an end-observation label:
    set coefficient_index_shift=1 to match A_2-3653 for 1979-1989.

    The final 2019-2026 interval includes observations inside 2026 (not
    half-open at 2026-01-01). If skip_last_day_latest=True and input field
    lengths agree, the potentially incomplete last day is excluded. When
    lengths differ, use the shortest shared prefix without trimming another
    day; the longer field's extra last day is already ignored. This assumes
    that the first observations are aligned in calendar time. Collected A/B
    arrays retain their filenames even when longer than the shared prefix.
    """
    root = Path(root)
    pair_key(names)
    if not isinstance(window_days, int) or window_days <= 0:
        raise ValueError("window_days must be positive integer")
    if not isinstance(coefficient_index_shift, int) or isinstance(coefficient_index_shift, bool):
        raise ValueError("coefficient_index_shift must be an integer")
    if not (0 <= q_low < q_high <= 1):
        raise ValueError("require 0 <= q_low < q_high <= 1")
    if (start_year, end_year) not in PERIODS:
        raise ValueError(f"Unsupported period {(start_year,end_year)}; choose from {PERIODS}")
    period = f"{start_year}-{end_year}"
    field_day_counts = _field_day_counts(root, names, period, spatial_shape, fields)
    raw_n_days = min(field_day_counts.values())
    max_raw_n_days = max(field_day_counts.values())
    different_field_lengths = raw_n_days != max_raw_n_days
    if different_field_lengths:
        print(f"{period} {pair_key(names)}: field lengths {field_day_counts}; "
              f"using common minimum {raw_n_days} observations", flush=True)
    first_day = date(start_year, 1, 1)
    raw_observed_end = first_day + timedelta(days=raw_n_days - 1)
    latest_period = (start_year, end_year) == PERIODS[-1]
    # The label '2019-2026' includes all of 2026, unlike the earlier
    # [start_year, end_year) decade ranges.
    period_end_exclusive = date(end_year + 1 if latest_period else end_year, 1, 1)
    if raw_observed_end >= period_end_exclusive:
        raise ValueError(
            f"input field {period} covers {raw_observed_end}, "
            f"outside allowed interval ending before {period_end_exclusive}"
        )
    if not isinstance(skip_last_day_latest, bool):
        raise ValueError("skip_last_day_latest must be a boolean")
    # If the longest field has an extra incomplete final day, taking the
    # shortest prefix already omits it. Avoid silently dropping yet another
    # observation from the shorter field. With equal lengths the established
    # latest-period policy still skips the potentially incomplete last day.
    skipped_last_day = latest_period and skip_last_day_latest and not different_field_lengths
    n_days = raw_n_days - int(skipped_last_day)
    if n_days < 2:
        raise ValueError(
            f"{period}: only {raw_n_days} observations; cannot skip the "
            "last one and still calculate a transition. "
            "Set skip_last_day_latest=False if it is complete."
        )
    observed_end = first_day + timedelta(days=n_days - 1)
    first_index = (first_day - date(1979, 1, 1)).days + 1 + coefficient_index_shift
    stop_index = first_index + n_days - 1  # exclusive; n_days-1 transitions
    raw_stop_index = first_index + raw_n_days - 1
    collected = None
    if source.storage == "collected":
        # Prefer the original complete aggregate filename, even if we omit
        # the last observation from quantiles. Some datasets may already
        # have been assembled without this final transition, so accept
        # the shortened filename as a second choice.
        # Usually the aggregate ends at the shortest field's last
        # transition. Also accept a pre-existing aggregate whose filename
        # follows the longer field: its excess tail is not read.
        candidates = list(dict.fromkeys((
            raw_stop_index - 1, stop_index - 1,
            first_index + max_raw_n_days - 2,
            first_index + max_raw_n_days - 3,
        )))
        candidates = [label for label in candidates if label >= stop_index - 1]
        not_found = None
        for file_last_label in candidates:
            try:
                collected = _open_collected_maps(
                    root, names, period, first_index,
                    file_last_label, source, spatial_shape
                )
                break
            except FileNotFoundError as exc:
                not_found = exc
        if collected is None:
            raise not_found
    windows = []
    for start in range(first_index, stop_index, window_days):
        stop = min(start + window_days, stop_index)
        windows.append(_window_quantiles(root, names, period, start, stop, source,
                                         spatial_shape, mask, q_low, q_high, temp_dir,
                                         collected=collected,
                                         collected_first_label=first_index))
    al = np.min([w["a_lower"] for w in windows], axis=0)
    ah = np.max([w["a_upper"] for w in windows], axis=0)
    bl = np.min([w["b_lower"] for w in windows], axis=0)
    bh = np.max([w["b_upper"] for w in windows], axis=0)
    constants = constants_from_bounds(al, ah, bl, bh, epsilon)
    return {
        "status": "ok", "pair": list(names), "pair_key": pair_key(names), "period": period,
        "observed_first_date": first_day.isoformat(), "observed_last_date": observed_end.isoformat(),
        "raw_observed_last_date": raw_observed_end.isoformat(),
        "field_observed_days": field_day_counts,
        "truncated_field_days": {name: days - raw_n_days
                                 for name, days in field_day_counts.items()},
        "aligned_to_shortest_field": different_field_lengths,
        "skipped_last_observation": skipped_last_day,
        "first_coefficient_index": first_index, "last_coefficient_index": stop_index-1,
        "n_raw_observed_days": raw_n_days,
        "n_observed_days": n_days, "n_coefficient_days": sum(w["n_days"] for w in windows),
        "n_valid_points": sum(w["n_points"] for w in windows),
        "epsilon": epsilon, "q_low": q_low, "q_high": q_high, "window_days": window_days,
        "bound_method": "exact pooled quantile per time window; envelope across windows",
        "source": {"a_pattern": source.a_pattern, "b_pattern": source.b_pattern,
                   "layout": source.layout, "diffusion": source.diffusion,
                   "storage": source.storage,
                   "state_dimension": _pair_dimension(names, source)},
        "a_lower": al.tolist(), "a_upper": ah.tolist(),
        "b_lower": bl.tolist(), "b_upper": bh.tolist(),
        **constants, "windows": windows,
        "qualification": "empirical trimmed ranges only, not a global SDE theorem proof",
    }


def run_all(
    root: str | Path,
    *,
    variables: Sequence[str] = tuple(FIELD_PATHS),
    selected_pairs: Sequence[tuple[str, str]] | None = None,
    periods: Sequence[tuple[int, int]] = PERIODS,
    coefficient_sources: Mapping[str, PairSource | Mapping[str, str]] | None = None,
    field_paths: Mapping[str, Sequence[str]] | None = None,
    output: str | Path | None = None,
    window_days: int = 1000,
    epsilon: float = 0.1,
    q_low: float = 0.03,
    q_high: float = 0.97,
    spatial_shape: tuple[int, int] = (161, 181),
    mask_policy: str = "auto",
    temp_dir: str | None = None,
    fail_on_missing: bool = False,
    coefficient_index_shift: int = 0,
    skip_last_day_latest: bool = True,
) -> list[dict[str, Any]]:
    """Compute named pairs for multiple periods, called from main.run_selected_task.

    Source layouts and file paths are configured by the calling workflow, not a CLI.
    Missing files are recorded as missing_input unless fail_on_missing=True.
    Invalid coefficient shapes and corrupted inputs are always fatal errors.
    The final 2019-2026 interval includes 2026; by default its last raw day
    is excluded as potentially incomplete. The switch does not affect earlier
    periods. If the paired raw fields have different lengths, the shared
    minimum is used and any longer aggregate tail is ignored.
    """
    root = Path(root)
    variables = tuple(variables)
    if not variables or len(set(variables)) != len(variables):
        raise ValueError("variables must be nonempty and unique")
    for name in variables:
        pair_key((name, name))
    pairs = (list(combinations_with_replacement(variables, 2))
             if selected_pairs is None else [tuple(pair) for pair in selected_pairs])
    if not pairs:
        raise ValueError("select at least one pair")
    for pair in pairs:
        pair_key(pair)
        if not set(pair).issubset(variables):
            raise ValueError(f"unknown variable in pair {pair}; available: {variables}")
    all_periods = [tuple(period) for period in periods]
    if not all_periods or any(period not in PERIODS for period in all_periods):
        raise ValueError(f"periods must be selected from {PERIODS}")
    sources = coefficient_sources or {}
    fields = dict(FIELD_PATHS if field_paths is None else field_paths)
    mask = read_mask(root, spatial_shape, mask_policy)
    destination = Path(output) if output is not None else root / "Conditions"
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    for years in all_periods:
        for pair in pairs:
            key = pair_key(pair)
            opts = sources.get(key)
            if opts is None:
                opts = sources.get("default")
                if opts is None and pair == ("sensible", "latent"):
                    opts = LEGACY_SENSIBLE_LATENT
            if opts is None:
                source = DEFAULT_SOURCE
            elif isinstance(opts, PairSource):
                source = opts
            elif isinstance(opts, Mapping):
                source = PairSource(**{**DEFAULT_SOURCE.__dict__, **dict(opts)})
            else:
                raise TypeError(f"Invalid coefficient source for {key}: {type(opts).__name__}")
            print(f"Checking {key} / {years[0]}-{years[1]} ...", flush=True)
            try:
                result = estimate_period(root, pair, *years, source,
                                         window_days=window_days, q_low=q_low,
                                         q_high=q_high, epsilon=epsilon,
                                         spatial_shape=spatial_shape, mask=mask,
                                         temp_dir=temp_dir, fields=fields,
                                         coefficient_index_shift=coefficient_index_shift,
                                         skip_last_day_latest=skip_last_day_latest)
            except FileNotFoundError as exc:
                if fail_on_missing:
                    raise
                result = {"status": "missing_input", "pair": list(pair),
                          "pair_key": key, "period": f"{years[0]}-{years[1]}",
                          "error": str(exc), "K1": None, "K2": None, "K": None}
            report_file = destination / "details" / key / f"{years[0]}-{years[1]}.json"
            report_file.parent.mkdir(parents=True, exist_ok=True)
            report_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            results.append(result)
            print(f"  {result['status']}: K1={result.get('K1')}, K2={result.get('K2')}", flush=True)
    with (destination / "sde_constants.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ["pair_key", "period", "status", "K1", "K2", "K", "epsilon", "q_low",
                   "q_high", "n_coefficient_days", "n_valid_points",
                   "observed_first_date", "observed_last_date", "error"]
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    return results
