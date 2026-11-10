"""Empirical trimmed SDE coefficient bounds from collected NP/SP maps.

Drop-in replacement for Ocean_components/Data_processing/sde_conditions.py.

Version 3: discovers collected scalar coefficients in 1d_bins_* for X-X
pairs, keeps 2D pair coefficients for X-Y pairs, optionally uses an existing
reverse-order 2D pair with correct component permutations, and merges field
aliases (particularly Pressure/PRESS) passed by older versions of main.py.
Missing 2D pairs are reported, never inferred from separate 1D coefficients.

The quantities K1, K2, and K are bounds on trimmed *observed coefficient*
values; they do not prove global SDE existence, uniqueness or Lipschitz
continuity. B must be the Ito diffusion AMPLITUDE (not BBT or B**2).

Modern coefficient layout for two-dimensional pairs::

    Coefficients/<period>/2d_bins_<bins>/<left>_<right>/
        Semiparametric_2d/A_<first>-<last>.npy
        Semiparametric_2d/B_<first>-<last>.npy

Each aggregate has time-first layout A:(T,2,H,W), B:(T,4,H,W).
The module accepts the obsolete ``{index1}`` template still passed by some
versions of main.py, replacing it with the correct collected-file source.
No legacy Coeff_data/ fallback is used for sensible/latent.
"""
from __future__ import annotations

import csv
import json
import math
import re
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, replace
from datetime import date, timedelta
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any

import numpy as np

PERIODS = (
    (1979, 1989), (1989, 1999), (1999, 2009),
    (2009, 2019), (2019, 2026),
)
FIELD_PATHS = {
    "sensible": (
        "DATA/Fluxes/sensible_{period}_grouped.npy",
        "DATA/Fluxes/sensible_grouped_{period}.npy",
        "Fluxes/sensible_{period}_grouped.npy",
        "Fluxes/sensible_grouped_{period}.npy",
    ),
    "latent": (
        "DATA/Fluxes/latent_{period}_grouped.npy",
        "DATA/Fluxes/latent_grouped_{period}.npy",
        "Fluxes/latent_{period}_grouped.npy",
        "Fluxes/latent_grouped_{period}.npy",
    ),
    "Flux": (
        "DATA/Fluxes/Flux_{period}_grouped.npy",
        "DATA/Fluxes/Flux_grouped_{period}.npy",
        "Fluxes/Flux_{period}_grouped.npy",
        "Fluxes/Flux_grouped_{period}.npy",
    ),
    "SST": (
        "DATA/SST/SST_{period}_grouped.npy",
        "DATA/SST/SST_grouped_{period}.npy",
        "SST/SST_{period}_grouped.npy",
        "SST/SST_grouped_{period}.npy",
    ),
    "Pressure": (
        "DATA/Pressure/Pressure_{period}_grouped.npy",
        "DATA/Pressure/PRESS_{period}_grouped.npy",
        "DATA/Pressure/Pressure_grouped_{period}.npy",
        "DATA/Pressure/PRESS_grouped_{period}.npy",
        "Pressure/Pressure_{period}_grouped.npy",
        "Pressure/PRESS_{period}_grouped.npy",
    ),
}


@dataclass(frozen=True)
class PairSource:
    """Coefficient paths and shape.

    ``storage='collected'`` expects aggregated time-first arrays.
    ``storage='daily'`` remains available for explicit custom input.
    ``layout='first'`` means A:(2,H,W), B:(4,H,W) per day.
    """

    a_pattern: str
    b_pattern: str
    layout: str = "last"
    diffusion: str = "matrix"
    storage: str = "daily"
    state_dimension: int | None = None
    reverse_components: bool = False

    def __post_init__(self):
        if self.layout not in ("first", "last"):
            raise ValueError("layout must be 'first' or 'last'")
        if self.diffusion not in ("matrix", "diagonal"):
            raise ValueError("diffusion must be 'matrix' or 'diagonal'")
        if self.storage not in ("daily", "collected"):
            raise ValueError("storage must be 'daily' or 'collected'")
        if self.state_dimension is not None and (type(self.state_dimension) is not int
                                                  or self.state_dimension < 1):
            raise ValueError("state_dimension must be a positive integer")
        if type(self.reverse_components) is not bool:
            raise ValueError("reverse_components must be a boolean")


# The default is the modern collected SP 2D layout. The bin count can be
# overridden via coefficient_sources (the main.py settings specify 12).
DEFAULT_SOURCE = PairSource(
    "Coefficients/{period}/2d_bins_12/{left}_{right}/Semiparametric_2d/"
    "A_{first_label}-{last_label}.npy",
    "Coefficients/{period}/2d_bins_12/{left}_{right}/Semiparametric_2d/"
    "B_{first_label}-{last_label}.npy",
    layout="first", diffusion="matrix", storage="collected", state_dimension=2,
)


def pair_key(names: tuple[str, str]) -> str:
    if len(names) != 2 or not all(isinstance(n, str) and n for n in names):
        raise ValueError("pair requires two nonempty names")
    if any("/" in n or "\\" in n or n in (".", "..") for n in names):
        raise ValueError("unsafe variable name")
    return "-".join(names)


def _modern_collected_source(bins: int = 12, method: str = "sp") -> PairSource:
    if method not in ("sp", "np"):
        raise ValueError(f"Unknown coefficient method: {method}")
    if bins < 1:
        raise ValueError("bins must be positive")
    folder = "Semiparametric_2d" if method == "sp" else "Nonparametric_2d"
    base = f"Coefficients/{{period}}/2d_bins_{bins}/{{left}}_{{right}}/{folder}/"
    return PairSource(
        base + "A_{first_label}-{last_label}.npy",
        base + "B_{first_label}-{last_label}.npy",
        layout="first", diffusion="matrix", storage="collected", state_dimension=2,
    )


def _upgrade_obsolete_source(source: PairSource) -> PairSource:
    """Repair the malformed A_{index1}-{}.npy pattern from main.py.

    Use the bin count/method already embedded in main.py rather than assuming
    a fixed numerical configuration. This is a compatibility bridge, NOT a
    fallback to obsolete raw Coeff_data paths.
    """
    patterns = (source.a_pattern, source.b_pattern)
    if any("{index1}" in p for p in patterns) or any("{}" in p for p in patterns):
        match = re.search(r"2d_bins_(\d+)", source.a_pattern)
        bins = int(match.group(1)) if match else 12
        folder = (source.a_pattern + source.b_pattern).lower()
        method = "np" if "nonparametric_2d" in folder else "sp"
        new = _modern_collected_source(bins, method)
        print("SDE: repaired outdated index1 coefficient pattern; "
              f"using collected {method.upper()} 2D files under 2d_bins_{bins}",
              flush=True)
        return new
    return source


def _has_collected_ab(folder: Path) -> bool:
    """True when at least one matching A/B aggregate exists in this folder.

    File names alone are checked here. The exact interval and array shapes
    are checked by _open_collected_maps before any statistics are calculated.
    """
    if not folder.is_dir():
        return False
    a_ranges = set()
    b_ranges = set()
    regex = re.compile(r"^(A|B)_(\d+)-(\d+)\.npy$")
    for path in folder.glob("[AB]_*-*.npy"):
        match = regex.fullmatch(path.name)
        if not match:
            continue
        span = (int(match.group(2)), int(match.group(3)))
        if span[0] > span[1]:
            continue
        (a_ranges if match.group(1) == "A" else b_ranges).add(span)
    return bool(a_ranges & b_ranges)


def _resolve_source_for_pair(root: Path, names: tuple[str, str],
                             years: tuple[int, int], source: PairSource) -> PairSource:
    """Resolve modern collected A/B without inventing missing cross-pair maps.

    The old main.py supplies a 2D template for *every* pair. This is wrong
    for (Flux, Flux), (SST, SST), etc.: scalar A/B live under 1d_bins_*.
    For distinct pairs we accept the reversed folder only if matching A/B
    exist; its coefficient components are permuted into the requested order.
    The specified NP/SP method is never changed behind the user's back.
    """
    if source.storage != "collected":
        return source  # User-supplied daily coefficient source.
    if not ("2d_bins_" in source.a_pattern and
            "{left}_{right}" in source.a_pattern):
        return source  # A custom source is authoritative.

    period = f"{years[0]}-{years[1]}"
    method = ("np" if "Nonparametric_2d" in source.a_pattern
              else "sp" if "Semiparametric_2d" in source.a_pattern
              else None)
    if method is None:
        return source

    if names[0] == names[1]:
        method_dir = "Semiparametric" if method == "sp" else "Nonparametric"
        root_dir = root / "Coefficients" / period
        candidates = sorted(root_dir.glob(
            f"1d_bins_*/{names[0]}/{method_dir}"
        ))
        available = [d for d in candidates if _has_collected_ab(d)]
        if not available:
            searched = root_dir / "1d_bins_*" / names[0] / method_dir
            raise FileNotFoundError(
                f"Missing scalar {method.upper()} collected A/B for {names[0]} "
                f"in {period}. Searched {searched} (with A_<first>-<last>.npy "
                "and B_<first>-<last>.npy). A 2D directory like "
                f"{names[0]}_{names[0]} is not a substitute for 1D maps."
            )
        if len(available) != 1:
            raise ValueError(
                f"Ambiguous 1D {method.upper()} coefficient bin configurations "
                f"for {names[0]} {period}: {available}. Select one "
                "explicitly instead of silently mixing resolutions."
            )
        folder = available[0]
        relative = folder.relative_to(root).as_posix()
        result = PairSource(
            f"{relative}/A_{{first_label}}-{{last_label}}.npy",
            f"{relative}/B_{{first_label}}-{{last_label}}.npy",
            layout="first", diffusion="matrix", storage="collected",
            state_dimension=1,
        )
        print(f"  SDE source: 1D {method.upper()} {folder}", flush=True)
        return result

    # For distinct variables use 2D maps. Reverse ONLY when the forward
    # folder lacks a paired A/B aggregate. Never reverse just one of A/B.
    template = source.a_pattern
    try:
        forward = _format_file(root, template, names, period, 1,
                               first_label=1, last_label=1).parent
    except (KeyError, ValueError):
        return source
    if _has_collected_ab(forward):
        return source
    reverse_names = (names[1], names[0])
    reverse_folder = _format_file(root, template, reverse_names, period, 1,
                                  first_label=1, last_label=1).parent
    if _has_collected_ab(reverse_folder):
        reversed_source = replace(
            source,
            a_pattern=source.a_pattern.replace(
                "{left}_{right}", f"{names[1]}_{names[0]}"
            ),
            b_pattern=source.b_pattern.replace(
                "{left}_{right}", f"{names[1]}_{names[0]}"
            ),
            reverse_components=True,
        )
        print(f"  SDE source: reversed pair folder {reverse_folder}; "
              "reordering both A and B components", flush=True)
        return reversed_source
    return source


def _format_file(root: Path, pattern: str, names: tuple[str, str], period: str,
                 index: int, *, first_label: int | None = None,
                 last_label: int | None = None) -> Path:
    value = pattern.format(
        pair=pair_key(names), left=names[0], right=names[1], period=period,
        index=index,
        first_label=index if first_label is None else first_label,
        last_label=index if last_label is None else last_label,
    )
    path = Path(value)
    return path if path.is_absolute() else root / path


def _flat_coefficients(raw: np.ndarray, *, n: int,
                       spatial_shape: tuple[int, int], layout: str,
                       kind: str, diffusion: str = "matrix") -> np.ndarray:
    """Flatten one coefficient map to [H*W,n] or [H*W,n*n]."""
    h, w = spatial_shape
    x = np.asarray(raw, dtype=np.float64)
    if n == 1 and x.shape == (h, w):
        return x.reshape(-1, 1)
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
        result = np.zeros((h*w, n, n), dtype=np.float64)
        for k in range(n):
            result[:, k, k] = flat[:, k]
        return result.reshape(h*w, n*n)
    if kind == "a":
        expected = (h,w,n) if layout == "last" else (n,h,w)
        if x.shape != expected:
            raise ValueError(f"drift: expected {expected}, got {x.shape}")
        return (x.reshape(-1,n) if layout == "last"
                else np.moveaxis(x, 0, -1).reshape(-1,n))
    if diffusion == "diagonal":
        expected = (h,w,n) if layout == "last" else (n,h,w)
        if x.shape != expected:
            raise ValueError(f"diagonal diffusion: expected {expected}, got {x.shape}")
        diag = (x.reshape(-1,n) if layout == "last"
                else np.moveaxis(x, 0, -1).reshape(-1,n))
        result = np.zeros((h*w, n, n), dtype=np.float64)
        for k in range(n):
            result[:, k, k] = diag[:, k]
        return result.reshape(h*w, n*n)
    expected = ((h,w,n,n), (h,w,n*n)) if layout == "last" else ((n,n,h,w), (n*n,h,w))
    if x.shape not in expected:
        raise ValueError(f"matrix diffusion: expected {expected}, got {x.shape}")
    if layout == "last":
        return x.reshape(-1,n*n)
    if x.ndim == 4:
        return np.moveaxis(x, (0,1), (-2,-1)).reshape(-1,n*n)
    return np.moveaxis(x, 0, -1).reshape(-1,n*n)


def _pair_dimension(names: tuple[str,str], source: PairSource) -> int:
    return (source.state_dimension if source.state_dimension is not None
            else (1 if names[0] == names[1] else 2))


def _prepare_pair_maps(a_raw: np.ndarray, b_raw: np.ndarray,
                       names: tuple[str,str], source: PairSource,
                       spatial_shape: tuple[int,int],
                       mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    n = _pair_dimension(names, source)
    a = _flat_coefficients(a_raw, n=n, spatial_shape=spatial_shape,
                           layout=source.layout, kind="a")
    b = _flat_coefficients(b_raw, n=n, spatial_shape=spatial_shape,
                           layout=source.layout, kind="b", diffusion=source.diffusion)
    if source.reverse_components:
        if n != 2:
            raise ValueError("Reversed coefficient order requires 2D coefficients")
        a = a[:, [1, 0]]
        b = b.reshape(-1, 2, 2)[:, [1, 0], :][:, :, [1, 0]].reshape(-1, 4)
    keep = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    if mask is not None:
        if mask.shape != spatial_shape:
            raise ValueError(f"mask shape {mask.shape} != {spatial_shape}")
        keep &= mask.ravel()
    return a[keep], b[keep]


def load_pair_maps(root: Path, names: tuple[str,str], period: str, index: int,
                   source: PairSource, spatial_shape: tuple[int,int],
                   mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if source.storage != "daily":
        raise ValueError("load_pair_maps requires daily storage")
    ap = _format_file(root,source.a_pattern,names,period,index)
    bp = _format_file(root,source.b_pattern,names,period,index)
    if not ap.is_file() or not bp.is_file():
        raise FileNotFoundError(f"Missing daily maps at {index}: {ap}; {bp}")
    return _prepare_pair_maps(
        np.load(ap,allow_pickle=False),np.load(bp,allow_pickle=False),
        names,source,spatial_shape,mask,
    )


def _open_collected_maps(root: Path, names: tuple[str,str], period: str,
                         first_label: int, last_label: int, source: PairSource,
                         spatial_shape: tuple[int,int]) -> tuple[np.ndarray, np.ndarray, int, str, str]:
    """Find two collected arrays covering all requested labels.

    The source may cover MORE days than the selected observations; we read
    only the correct offsets. This accommodates the last incomplete day and
    already-assembled arrays without silently misaligning time axes.
    """
    a_example = _format_file(root,source.a_pattern,names,period,
                             first_label,first_label=first_label,last_label=last_label)
    b_example = _format_file(root,source.b_pattern,names,period,
                             first_label,first_label=first_label,last_label=last_label)
    # Discover merged files by their actual label ranges, not a guessed last day.
    def matches(example: Path, prefix: str) -> list[tuple[int,int,Path]]:
        regex = re.compile(rf"^{re.escape(prefix)}_(\d+)-(\d+)\.npy$")
        found = []
        for candidate in example.parent.glob(f"{prefix}_*-*.npy"):
            m = regex.fullmatch(candidate.name)
            if m:
                lo,hi = int(m.group(1)),int(m.group(2))
                if lo <= first_label and hi >= last_label:
                    found.append((lo,hi,candidate))
        return found
    amatches = matches(a_example,"A")
    bmatches = matches(b_example,"B")
    common = [(a0,a1,ap,bp) for a0,a1,ap in amatches
              for b0,b1,bp in bmatches if a0==b0 and a1==b1]
    if not common:
        raise FileNotFoundError(
            f"Missing matching collected A/B covering {first_label}-{last_label}: "
            f"{a_example.parent} (expected example {a_example.name} / {b_example.name})"
        )
    # Prefer closest available covering interval; refuse ambiguity.
    common.sort(key=lambda item: (item[1]-item[0],item[0]))
    if len(common)>1 and common[0][1]-common[0][0]==common[1][1]-common[1][0]:
        raise ValueError(f"Ambiguous collected A/B sources in {a_example.parent}")
    lo,hi,ap,bp = common[0]
    a = np.load(ap,mmap_mode="r",allow_pickle=False)
    b = np.load(bp,mmap_mode="r",allow_pickle=False)
    count=hi-lo+1
    if a.ndim < 3 or b.ndim < 3 or len(a)!=count or len(b)!=count:
        raise ValueError(f"Incorrect aggregate time axis: A {a.shape}, B {b.shape}; "
                         f"expected {count} maps ({ap} / {bp})")
    _prepare_pair_maps(a[first_label-lo], b[first_label-lo],names,source,spatial_shape,None)
    return a,b,lo,str(ap),str(bp)


def read_mask(root: Path, spatial_shape: tuple[int,int], policy: str = "auto") -> np.ndarray | None:
    if policy not in ("auto", "none", "required"):
        raise ValueError("mask_policy must be 'auto', 'none' or 'required'")
    if policy == "none":
        return None
    root=Path(root)
    path=next((p for p in (root/"DATA"/"mask",root/"mask") if p.is_file()),None)
    if path is None:
        if policy == "required":
            raise FileNotFoundError("Ocean mask is missing")
        return None
    size=int(np.prod(spatial_shape))
    data=path.read_bytes()
    if len(data)<size:
        raise ValueError(f"Ocean mask contains only {len(data)} cells, expected {size}")
    return np.frombuffer(data[:size],dtype=np.bool_).copy().reshape(spatial_shape)


def _field_day_counts(root: Path, names: tuple[str,str], period: str,
                      spatial_shape: tuple[int,int],
                      fields: Mapping[str,Sequence[str]]) -> dict[str,int]:
    """Read only .npy headers; accept (space,time), (time,space) and 3D arrays."""
    counts={}
    n_space=int(np.prod(spatial_shape))
    for name in dict.fromkeys(names):
        patterns=fields.get(name)
        if not patterns:
            raise ValueError(f"No field paths configured for {name!r}")
        candidates=[root / fmt.format(period=period) for fmt in patterns]
        path=next((p for p in candidates if p.is_file()),None)
        if path is None:
            raise FileNotFoundError(f"Missing input field {name} {period}; tried {candidates}")
        x=np.load(path,mmap_mode="r",allow_pickle=False)
        if x.ndim==3 and x.shape[1:]==spatial_shape:
            counts[name]=int(x.shape[0])
        elif x.ndim==3 and x.shape[:2]==spatial_shape:
            counts[name]=int(x.shape[2])
        elif x.ndim==2 and x.shape[0]==n_space:
            counts[name]=int(x.shape[1])
        elif x.ndim==2 and x.shape[1]==n_space:
            counts[name]=int(x.shape[0])
        else:
            raise ValueError(f"Invalid {name} shape {x.shape} in {path}; "
                             f"expected (T,H,W), (H,W,T), (space,time) or (time,space)")
        del x
    if min(counts.values())<2:
        raise ValueError(f"At least 2 observations required: {counts}")
    return counts


def actual_days(root: Path, names: tuple[str,str], period: str,
                spatial_shape: tuple[int,int],
                fields: Mapping[str,Sequence[str]] = FIELD_PATHS) -> int:
    return min(_field_day_counts(Path(root),names,period,spatial_shape,fields).values())


def constants_from_bounds(a_lo: np.ndarray, a_hi: np.ndarray,
                          b_lo: np.ndarray, b_hi: np.ndarray,
                          epsilon: float) -> dict[str,float]:
    """Calculate empirical K1/K2 using observed coefficient ranges."""
    if not(math.isfinite(epsilon) and epsilon>0):
        raise ValueError("epsilon must be finite and positive")
    al,ah,bl,bh=[np.asarray(x,dtype=float).ravel() for x in (a_lo,a_hi,b_lo,b_hi)]
    if al.size!=ah.size or bl.size!=bh.size or al.size<1 or bl.size<1:
        raise ValueError("lower/upper shapes must match and be nonempty")
    if any(not np.isfinite(x).all() for x in (al,ah,bl,bh)) or (al>ah).any() or (bl>bh).any():
        raise ValueError("finite, ordered bounds required")
    k1=(np.linalg.norm(ah-al)+np.linalg.norm(bh-bl))/epsilon
    k2=(np.maximum(abs(al),abs(ah))**2).sum()+(np.maximum(abs(bl),abs(bh))**2).sum()
    return {"K1":float(k1),"K2":float(k2),"K":float(max(k1,k2))}


def _window_quantiles(root: Path, names: tuple[str,str], period: str,
                      start: int, stop: int, source: PairSource,
                      spatial_shape: tuple[int,int], mask: np.ndarray | None,
                      q_low: float, q_high: float, temp_dir: str | None,
                      collected: tuple[np.ndarray,np.ndarray] | None = None,
                      collected_first_label: int | None = None) -> dict[str,Any]:
    """Exact pooled quantiles with temporary disk streams, bounded RAM."""
    n=_pair_dimension(names,source)
    na,nb=n,n*n
    used_days=used_points=0
    with tempfile.TemporaryDirectory(dir=temp_dir) as td:
        with ExitStack() as stack:
            streams=[stack.enter_context(open(Path(td)/f"component_{i}.bin","wb"))
                     for i in range(na+nb)]
            for label in range(start,stop):
                if collected is None:
                    a,b=load_pair_maps(root,names,period,label,source,spatial_shape,mask)
                else:
                    offset=label-collected_first_label
                    a,b=_prepare_pair_maps(collected[0][offset],collected[1][offset],
                                           names,source,spatial_shape,mask)
                if not len(a):
                    continue
                for j in range(na):
                    np.ascontiguousarray(a[:,j]).tofile(streams[j])
                for j in range(nb):
                    np.ascontiguousarray(b[:,j]).tofile(streams[na+j])
                used_days+=1
                used_points+=len(a)
        if not used_points:
            raise ValueError(f"No finite coefficient values: {names} {period}, {start}..{stop-1}")
        low=[]; high=[]
        for j in range(na+nb):
            vals=np.memmap(Path(td)/f"component_{j}.bin",dtype=np.float64,
                           mode="r",shape=(used_points,))
            low.append(float(np.quantile(vals,q_low)))
            high.append(float(np.quantile(vals,q_high)))
            del vals
    return {
        "start_index":start,"end_index_inclusive":stop-1,
        "n_days":used_days,"n_points":used_points,
        "a_lower":low[:na],"a_upper":high[:na],
        "b_lower":low[na:],"b_upper":high[na:],
    }


def estimate_period(root: Path|str,names: tuple[str,str],start_year: int,end_year: int,
                    source: PairSource,*,window_days: int=1000,q_low: float=0.03,
                    q_high: float=0.97,epsilon: float=0.1,
                    spatial_shape: tuple[int,int]=(161,181),mask: np.ndarray|None=None,
                    temp_dir: str|None=None,
                    fields: Mapping[str,Sequence[str]]=FIELD_PATHS,
                    coefficient_index_shift: int=0,
                    skip_last_day_latest: bool=True) -> dict[str,Any]:
    """Estimate trimmed drift/diffusion bounds for one named period."""
    root=Path(root)
    pair_key(names)
    if type(window_days) is not int or window_days<1:
        raise ValueError("window_days must be positive")
    if type(coefficient_index_shift) is not int:
        raise ValueError("coefficient_index_shift must be integer")
    if not(0<=q_low<q_high<=1):
        raise ValueError("invalid quantiles")
    if (start_year,end_year) not in PERIODS:
        raise ValueError(f"unsupported period {start_year}-{end_year}")
    source=_upgrade_obsolete_source(source)
    period=f"{start_year}-{end_year}"
    field_day_counts=_field_day_counts(root,names,period,spatial_shape,fields)
    raw_n_days=min(field_day_counts.values())
    max_raw_n_days=max(field_day_counts.values())
    unequal=raw_n_days!=max_raw_n_days
    if unequal:
        print(f"{period} {pair_key(names)}: unequal field lengths {field_day_counts}; "
              f"using {raw_n_days} shared days",flush=True)
    first_day=date(start_year,1,1)
    raw_observed_end=first_day+timedelta(days=raw_n_days-1)
    latest=(start_year,end_year)==PERIODS[-1]
    period_end=date(end_year+1 if latest else end_year,1,1)
    if raw_observed_end>=period_end:
        raise ValueError(f"{period} field end {raw_observed_end} exceeds {period_end}")
    if type(skip_last_day_latest) is not bool:
        raise ValueError("skip_last_day_latest must be bool")
    skipped=bool(latest and skip_last_day_latest and not unequal)
    n_days=raw_n_days-int(skipped)
    if n_days<2:
        raise ValueError("not enough observations after optional final-day skip")
    observed_end=first_day+timedelta(days=n_days-1)
    first_index=(first_day-date(1979,1,1)).days+1+coefficient_index_shift
    stop_index=first_index+n_days-1 # exclusive; time axis has n_days-1 transitions
    collected=None
    collected_offset=None
    collected_paths=None
    if source.storage=="collected":
        a,b,first_in_file,ap,bp=_open_collected_maps(
            root,names,period,first_index,stop_index-1,source,spatial_shape
        )
        collected=(a,b)
        collected_offset=first_in_file
        collected_paths=[ap,bp]
        print(f"  Using collected A/B: {Path(ap).name}, {Path(bp).name}; "
              f"labels {first_index}..{stop_index-1}",flush=True)
    windows=[]
    for start in range(first_index,stop_index,window_days):
        stop=min(start+window_days,stop_index)
        windows.append(_window_quantiles(root,names,period,start,stop,source,
                                         spatial_shape,mask,q_low,q_high,temp_dir,
                                         collected=collected,
                                         collected_first_label=collected_offset))
    al=np.min([w["a_lower"] for w in windows],axis=0)
    ah=np.max([w["a_upper"] for w in windows],axis=0)
    bl=np.min([w["b_lower"] for w in windows],axis=0)
    bh=np.max([w["b_upper"] for w in windows],axis=0)
    bounds=constants_from_bounds(al,ah,bl,bh,epsilon)
    return {
        "status":"ok","pair":list(names),"pair_key":pair_key(names),"period":period,
        "observed_first_date":first_day.isoformat(),
        "observed_last_date":observed_end.isoformat(),
        "raw_observed_last_date":raw_observed_end.isoformat(),
        "field_observed_days":field_day_counts,
        "truncated_field_days":{n:days-raw_n_days for n,days in field_day_counts.items()},
        "aligned_to_shortest_field":unequal,
        "skipped_last_observation":skipped,
        "first_coefficient_index":first_index,
        "last_coefficient_index":stop_index-1,
        "n_raw_observed_days":raw_n_days,
        "n_observed_days":n_days,
        "n_coefficient_days":sum(w["n_days"] for w in windows),
        "n_valid_points":sum(w["n_points"] for w in windows),
        "epsilon":epsilon,"q_low":q_low,"q_high":q_high,
        "window_days":window_days,
        "bound_method":"exact pooled quantile per time window; envelope across windows",
        "source":{
            "a_pattern":source.a_pattern,"b_pattern":source.b_pattern,
            "layout":source.layout,"diffusion":source.diffusion,
            "storage":source.storage,
            "state_dimension":_pair_dimension(names,source),
            "coefficient_file_order":(
                list(reversed(names)) if source.reverse_components else list(names)
            ),
            "reverse_components":source.reverse_components,
            "files":collected_paths,
        },
        "a_lower":al.tolist(),"a_upper":ah.tolist(),
        "b_lower":bl.tolist(),"b_upper":bh.tolist(),
        **bounds,"windows":windows,
        "qualification":"empirical trimmed ranges only, not a global SDE theorem proof",
    }


def run_all(root: str|Path,*,
            variables: Sequence[str]=tuple(FIELD_PATHS),
            selected_pairs: Sequence[tuple[str,str]]|None=None,
            periods: Sequence[tuple[int,int]]=PERIODS,
            coefficient_sources: Mapping[str,PairSource|Mapping[str,str]]|None=None,
            field_paths: Mapping[str,Sequence[str]]|None=None,
            output: str|Path|None=None,
            window_days: int=1000,epsilon: float=0.1,
            q_low: float=0.03,q_high: float=0.97,
            spatial_shape: tuple[int,int]=(161,181),
            mask_policy: str="auto",temp_dir: str|None=None,
            fail_on_missing: bool=False,
            coefficient_index_shift: int=0,
            skip_last_day_latest: bool=True) -> list[dict[str,Any]]:
    """Compute selected pair/period SDE diagnostics and persist CSV + JSON.

    Compatibility: current GitHub main.py still passes obsolete ``index1``
    placeholders and omits Flux from its default variable tuple. For that
    *specific* default 4-variable tuple, include Flux automatically.
    Explicit selected_pairs are never broadened.
    """
    root=Path(root)
    variables=tuple(variables)
    if variables==( "sensible","latent","SST","Pressure"):
        variables=("sensible","latent","Flux","SST","Pressure")
        print("SDE: added Flux to the historical default variable set",flush=True)
    if not variables or len(variables)!=len(set(variables)):
        raise ValueError("variables must be unique and nonempty")
    for name in variables:
        pair_key((name,name))
    pairs=(list(combinations_with_replacement(variables,2)) if selected_pairs is None
           else [tuple(p) for p in selected_pairs])
    if not pairs:
        raise ValueError("No selected pairs")
    for pair in pairs:
        pair_key(pair)
        if not set(pair).issubset(variables):
            raise ValueError(f"Unknown variable in {pair}; selected: {variables}")
    all_periods=[tuple(p) for p in periods]
    if not all_periods or any(p not in PERIODS for p in all_periods):
        raise ValueError(f"periods must belong to {PERIODS}")
    sources=coefficient_sources or {}
    fields = {name: tuple(paths) for name, paths in FIELD_PATHS.items()}
    if field_paths is not None:
        for name, configured in field_paths.items():
            # Use explicit preferences first, keep modern/legacy *field* aliases
            # as fallbacks. This does not re-enable legacy coefficient paths.
            fields[name] = tuple(dict.fromkeys(
                tuple(configured) + tuple(fields.get(name, ()))
            ))
    mask=read_mask(root,spatial_shape,mask_policy)
    destination=Path(output) if output is not None else root/"Conditions"
    destination.mkdir(parents=True,exist_ok=True)
    results=[]
    for years in all_periods:
        for pair in pairs:
            key=pair_key(pair)
            opts=sources.get(key,sources.get("default"))
            if opts is None:
                source=DEFAULT_SOURCE
            elif isinstance(opts,PairSource):
                source=opts
            elif isinstance(opts,Mapping):
                overrides=dict(opts)
                # Preserve custom callers that supplied daily paths without
                # specifying storage or layout. No implicit Coeff_data fallback.
                if ("storage" not in overrides and
                    "{index}" in overrides.get("a_pattern","") and
                    "{index1}" not in overrides.get("a_pattern", "")):
                    overrides.setdefault("storage","daily")
                    overrides.setdefault("layout","last")
                    overrides.setdefault("state_dimension",None)
                source=PairSource(**{**DEFAULT_SOURCE.__dict__,**overrides})
            else:
                raise TypeError(f"Invalid coefficient source for {key}: {type(opts).__name__}")
            source=_upgrade_obsolete_source(source)
            print(f"Checking {key} / {years[0]}-{years[1]} ...",flush=True)
            try:
                source = _resolve_source_for_pair(root, pair, years, source)
                result=estimate_period(root,pair,*years,source,
                                       window_days=window_days,q_low=q_low,
                                       q_high=q_high,epsilon=epsilon,
                                       spatial_shape=spatial_shape,mask=mask,
                                       temp_dir=temp_dir,fields=fields,
                                       coefficient_index_shift=coefficient_index_shift,
                                       skip_last_day_latest=skip_last_day_latest)
            except FileNotFoundError as exc:
                if fail_on_missing:
                    raise
                error_text = str(exc)
                stage = ("raw_field" if "input field" in error_text.lower()
                         else "coefficient_files")
                result={"status":"missing_input","pair":list(pair),
                        "pair_key":key,"period":f"{years[0]}-{years[1]}",
                        "missing_stage":stage,
                        "error":error_text,"K1":None,"K2":None,"K":None}
                print(f"  Missing {stage}: {error_text}", flush=True)
            report=destination/"details"/key/f"{years[0]}-{years[1]}.json"
            report.parent.mkdir(parents=True,exist_ok=True)
            report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
            results.append(result)
            print(f"  {result['status']}: K1={result.get('K1')}, "
                  f"K2={result.get('K2')}",flush=True)
    with (destination/"sde_constants.csv").open("w",newline="",encoding="utf-8") as handle:
        columns=["pair_key","period","status","K1","K2","K","epsilon","q_low",
                 "q_high","n_coefficient_days","n_valid_points",
                 "observed_first_date","observed_last_date","error"]
        writer=csv.DictWriter(handle,fieldnames=columns,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    return results
