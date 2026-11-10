"""Experimental semiparametric stationary-PDF tails for Ocean_components.

The trusted central shape is read from the existing stationary NPZ. Beyond
short transition zones, the PDF follows Generalized Pareto exceedance tails
estimated from a reproducible, stratified sample of observed daily flux values.

This is a post-hoc HYBRID PDF, not the zero-current solution for the original
polynomial drift / fitted diffusion outside the trusted core. It does not
modify previously estimated A, B, original PDF or its JSON metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import genpareto

from Data_processing.io import field_candidates

SCHEMA_VERSION = 1


def _raw_source(root, name, start_year, end_year):
    paths = field_candidates(root, name, start_year, end_year)
    path = next((p for p in paths if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            f"No ERA5 flux observations found for {name}, {start_year}-{end_year}; "
            f"tried: {paths}"
        )
    return path


def _signature(path):
    s = path.stat()
    return {"path": str(path.resolve()), "size": int(s.st_size),
            "mtime_ns": int(s.st_mtime_ns)}


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def observed_sample(root, name, start_year, end_year, *,
                    sample_days=1600, pixels_per_day=160, seed=1979,
                    refresh=False):
    """Read bounded, temporally stratified raw observations, with persistent cache.

    Days are sampled without replacement across the entire period, then finite
    grid pixels are sampled uniformly within each selected day. No giant array
    is copied or loaded into RAM. This is not a complete empirical histogram.
    """
    root = Path(root)
    source = _raw_source(root, name, start_year, end_year)
    if sample_days < 10 or pixels_per_day < 10:
        raise ValueError("sample_days and pixels_per_day must both be >= 10")
    tag = f"{name}_{start_year}-{end_year}_{sample_days}days_{pixels_per_day}pixels"
    cache = root / "Functional" / "Stationary" / "Hybrid" / "Samples" / (tag + ".npz")
    meta_path = cache.with_suffix(".json")
    spec = {"schema": SCHEMA_VERSION, "source": _signature(source),
            "sample_days": sample_days, "pixels_per_day": pixels_per_day,
            "seed": seed, "sampling": "uniform_days_and_uniform_pixels_within_day"}
    if not refresh and cache.is_file() and meta_path.is_file():
        if _json(meta_path) == spec:
            with np.load(cache, allow_pickle=False) as data:
                values = np.asarray(data["values"], dtype=np.float64)
            print(f"Reusing cached ERA5 tail sample: {cache}", flush=True)
            return values, spec

    field = np.load(source, mmap_mode="r", allow_pickle=False)
    if field.ndim < 2:
        raise ValueError(f"Expected daily gridded ERA5 array, got shape {field.shape}")
    rng = np.random.default_rng(int(seed))
    n_days = min(sample_days, int(field.shape[0]))
    days = np.sort(rng.choice(int(field.shape[0]), size=n_days, replace=False))
    values = []
    for day in days:
        flattened = np.asarray(field[int(day)]).ravel()
        take = min(int(pixels_per_day), flattened.size)
        positions = rng.choice(flattened.size, size=take, replace=False)
        part = np.asarray(flattened[positions], dtype=np.float64)
        good = part[np.isfinite(part)]
        if good.size:
            values.append(good)
    if not values:
        raise ValueError(f"No finite ERA5 observations sampled from {source}")
    sample = np.concatenate(values)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, values=sample)
    _safe_write_json(meta_path, spec)
    print(f"Saved ERA5 tail sample: {cache} ({sample.size:,} values)", flush=True)
    return sample, spec


def _gpd_estimate(sample, side, quantile):
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    if side == "right":
        threshold = float(np.quantile(sample, quantile))
        excess = sample[sample > threshold] - threshold
    else:
        threshold = float(np.quantile(sample, 1 - quantile))
        excess = threshold - sample[sample < threshold]
    if excess.size < 100:
        raise ValueError(f"Not enough extreme observations in {side} tail: {excess.size}")
    shape, _, scale = genpareto.fit(excess, floc=0)
    if not (np.isfinite(shape) and np.isfinite(scale) and scale > 0):
        raise ValueError(f"Invalid {side} GPD fit")
    return {"threshold": threshold, "shape": float(shape), "scale": float(scale),
            "exceedance_probability": float(excess.size / sample.size),
            "n_excess": int(excess.size)}


def _fit_scale_for_shape(excess, shape, initial):
    """Refit local scale by likelihood after pooling GPD shape with full period."""
    zmax = float(excess.max())
    zmean = float(excess.mean())
    lower = 1e-5 * max(zmean, 1e-8)
    if shape < 0:
        lower = max(lower, -float(shape) * zmax * 1.00001)
    upper = max(initial * 20.0, lower * 30.0, zmean * 20.0)
    def objective(scale):
        pdf = genpareto.logpdf(excess, c=shape, loc=0, scale=scale)
        if not np.isfinite(pdf).all():
            return np.inf
        return -float(pdf.sum())
    result = minimize_scalar(objective, bounds=(lower, upper), method="bounded")
    if not result.success or not np.isfinite(result.fun):
        raise ValueError("Could not fit GPD scale for pooled shape")
    return float(result.x)


def _pooled_gpd(local_sample, reference_sample, side, quantile, reference_weight):
    local = _gpd_estimate(local_sample, side, quantile)
    ref = _gpd_estimate(reference_sample, side, quantile)
    shape = (1 - reference_weight) * local["shape"] + reference_weight * ref["shape"]
    threshold = local["threshold"]
    if side == "right":
        excess = local_sample[local_sample > threshold] - threshold
    else:
        excess = threshold - local_sample[local_sample < threshold]
    scale = _fit_scale_for_shape(excess, shape, local["scale"])
    return {**local, "shape": float(shape), "scale": scale,
            "local_shape": local["shape"], "reference_shape": ref["shape"],
            "reference_weight": float(reference_weight),
            "reference_threshold": ref["threshold"]}


def _smooth_monotone_log_bridge(distance, length, p0, outward_slope,
                                p1, terminal_slope):
    """C1 monotone bridge in log-PDF, matching BOTH endpoint values/slopes.

    In outward distance t, define -d(log p)/dt = exp(linear log k + c*s*(1-s)).
    A one-dimensional root chooses c so that integrating k gives precisely
    log(p0)-log(p1). The rate k is strictly positive, hence no new bubble.
    """
    if length <= 0 or p0 <= 0 or p1 <= 0:
        raise ValueError("Invalid bridge endpoints")
    if not (outward_slope < 0 and terminal_slope < 0):
        raise ValueError("Bridge cannot be monotone: the endpoint slopes point outward")
    drop = float(np.log(p0) - np.log(p1))
    if drop <= 0:
        raise ValueError("Tail PDF is not lower than central PDF at bridge boundaries")
    k0, k1 = -float(outward_slope), -float(terminal_slope)
    s_knots = np.linspace(0.0, 1.0, 1025)
    logk0, logk1 = np.log(k0), np.log(k1)
    base = (1 - s_knots) * logk0 + s_knots * logk1
    arch = s_knots * (1 - s_knots)
    # Stable log-integral through the root range; exp is bounded by clipping.
    def total_drop(c):
        lograte = np.clip(base + c * arch, -700.0, 400.0)
        return float(length * np.trapezoid(np.exp(lograte), s_knots))
    c = brentq(lambda a: total_drop(a) - drop, -2800.0, 1600.0, xtol=1e-10)
    rate = np.exp(np.clip(base + c * arch, -700.0, 400.0))
    integrated = cumulative_trapezoid(rate, s_knots, initial=0.0)
    s = np.asarray(distance, dtype=np.float64) / length
    # Hermite integration preserves the exact endpoint derivatives; a linear
    # interpolation of cumulative integrals would merely be C0, not C1.
    primitive = CubicHermiteSpline(s_knots, integrated, rate)
    integral = primitive(s) * length
    return np.log(p0) - integral


def _combine(grid, base_pdf, sample, reference, interval,
             *, tail_quantile, transition_quantile, reference_weight):
    """Return normalized hybrid PDF, color-coded section masks and provenance."""
    grid = np.asarray(grid, dtype=np.float64)
    base_pdf = np.asarray(base_pdf, dtype=np.float64)
    if grid.ndim != 1 or base_pdf.shape != grid.shape or grid.size < 50:
        raise ValueError("Expected aligned, dense 1D stationary grid and density")
    if np.any(np.diff(grid) <= 0) or np.any(base_pdf <= 0):
        raise ValueError("Stationary grid must be increasing and PDF positive")
    if not (0.5 < transition_quantile < tail_quantile < 1):
        raise ValueError("Require 0.5 < transition_quantile < tail_quantile < 1")
    if not 0 <= reference_weight <= 1:
        raise ValueError("reference_weight must be between 0 and 1")
    xmin, xmax = map(float, interval)
    if not (grid[0] < xmin < xmax < grid[-1]):
        raise ValueError("Fitted support lies outside density grid")

    dx = float(np.median(np.diff(grid)))
    mode = float(grid[np.argmax(base_pdf)])
    logbase = np.log(base_pdf)
    slope = np.gradient(logbase, grid)
    minimum_gap = max(4 * dx, 0.025 * (xmax - xmin))
    anchor_margin = max(3 * dx, 0.02 * (xmax - xmin))

    # Place transition start INSIDE supported regression interval, to avoid
    # the quartic drift's extrapolation artifact in the interpolation bridge.
    l_anchor = float(np.clip(np.quantile(sample, 1 - transition_quantile),
                             xmin + anchor_margin, mode - anchor_margin))
    r_anchor = float(np.clip(np.quantile(sample, transition_quantile),
                             mode + anchor_margin, xmax - anchor_margin))
    l_anchor = float(grid[np.argmin(abs(grid - l_anchor))])
    r_anchor = float(grid[np.argmin(abs(grid - r_anchor))])
    # If central fitted PDF is nonmonotone near a chosen anchor, find a nearby
    # local point where it decays away from the mode; never silently hide a bubble.
    for side, anchor in (("left", l_anchor), ("right", r_anchor)):
        k = int(np.argmin(abs(grid - anchor)))
        if side == "left" and not slope[k] > 0:
            candidates = np.where((grid > xmin) & (grid < mode - anchor_margin)
                                  & (slope > 1e-8))[0]
            if candidates.size == 0:
                raise ValueError("No increasing left core near supported fit")
            l_anchor = float(grid[candidates[np.argmin(abs(grid[candidates] - anchor))]])
        if side == "right" and not slope[k] < 0:
            candidates = np.where((grid < xmax) & (grid > mode + anchor_margin)
                                  & (slope < -1e-8))[0]
            if candidates.size == 0:
                raise ValueError("No decreasing right core near supported fit")
            r_anchor = float(grid[candidates[np.argmin(abs(grid[candidates] - anchor))]])

    loghyb = logbase.copy()
    section = np.zeros(grid.size, dtype=np.int8)  # 0 core, +/-1 bridge, +/-2 tail
    diagnostics = {}
    for side, sign, anchor in (("left", -1, l_anchor), ("right", 1, r_anchor)):
        fit = _pooled_gpd(sample, reference, side, tail_quantile, reference_weight)
        threshold = float(fit["threshold"])
        # If a quantile estimate falls too near the supported core, enlarge the
        # transition. Fit GPD *at the actual threshold* instead of misusing it.
        if sign * (threshold - anchor) < minimum_gap:
            threshold = anchor + sign * minimum_gap
            excess = ((sample - threshold) if sign == 1 else (threshold - sample))
            excess = excess[excess > 0]
            if excess.size < 100:
                raise ValueError(f"Not enough observations beyond adjusted {side} threshold")
            shape = float(fit["shape"])
            fit["scale"] = _fit_scale_for_shape(excess, shape, fit["scale"])
            fit["threshold"] = float(threshold)
            fit["n_excess"] = int(excess.size)
            fit["exceedance_probability"] = float(excess.size / sample.size)

        length = sign * (threshold - anchor)
        if not grid[0] < threshold < grid[-1]:
            raise ValueError(f"{side} GPD threshold {threshold:.3g} outside stationary grid")
        p0 = float(np.exp(np.interp(anchor, grid, logbase)))
        core_slope = float(np.interp(anchor, grid, slope))
        outward_slope = sign * core_slope
        shape, scale = fit["shape"], fit["scale"]
        if shape <= -1:
            raise ValueError(f"{side}: GPD shape={shape:.3g} yields nondecreasing density at threshold")
        empirical_p_at_threshold = fit["exceedance_probability"] / scale
        # Prevent an impossible rising bridge if fitted core and raw histogram
        # disagree. Make and report a conservative amplitude constraint.
        p1 = min(empirical_p_at_threshold, p0 * np.exp(-max(1e-5 * length, 0.002)))
        amplitude_ratio = p1 / empirical_p_at_threshold
        tail_slope = -(1 + shape) / scale  # d log p / d outward distance

        dist = sign * (grid - anchor)
        bridge_mask = (dist > 0) & (dist < length)
        tail_mask = dist >= length
        section[bridge_mask] = sign
        section[tail_mask] = 2 * sign
        loghyb[bridge_mask] = _smooth_monotone_log_bridge(
            dist[bridge_mask], length, p0, outward_slope, p1, tail_slope)
        if np.any(tail_mask):
            excess_on_grid = dist[tail_mask] - length
            logtail = genpareto.logpdf(excess_on_grid, c=shape, loc=0, scale=scale)
            loghyb[tail_mask] = np.log(p1 * scale) + logtail
        diagnostics[side] = {**fit, "anchor": anchor, "tail_start": threshold,
                             "core_outward_log_slope": outward_slope,
                             "empirical_tail_density_at_threshold": float(empirical_p_at_threshold),
                             "used_tail_density_at_threshold": float(p1),
                             "amplitude_ratio_to_empirical": float(amplitude_ratio)}

    # The original stationary PDF is normalized on a finite grid. Preserve
    # its shape inside the trusted region, then normalize the new total PDF.
    finite = np.isfinite(loghyb)
    if not np.all(finite):
        # A fitted bounded GP might end before the available grid; zero beyond.
        loghyb = np.where(finite, loghyb, -np.inf)
    scaled = np.exp(loghyb - float(np.max(loghyb)))
    z = float(np.trapezoid(scaled, grid))
    if not np.isfinite(z) or z <= 0:
        raise ValueError("Hybrid density cannot be normalized")
    pdf = scaled / z
    for side, sign in (("left", -1), ("right", 1)):
        info = diagnostics[side]
        tail_edge = float(grid[0] if sign == -1 else grid[-1])
        excess_from_threshold = sign * (tail_edge - info["threshold"])
        survival = float(genpareto.sf(max(0.0, excess_from_threshold),
                                          c=info["shape"], loc=0, scale=info["scale"]))
        tail_mass_total = (info["used_tail_density_at_threshold"] * info["scale"]
                           * np.exp(-float(np.max(loghyb))) / z)
        info["estimated_probability_beyond_grid"] = float(tail_mass_total * survival)
        info["predicted_total_tail_probability"] = float(tail_mass_total)
        info["predicted_to_observed_tail_probability_ratio"] = float(
            tail_mass_total / info["exceedance_probability"])
        info["finite_gpd_mean"] = bool(info["shape"] < 1)
        info["finite_gpd_variance"] = bool(info["shape"] < 0.5)
    return pdf, section, diagnostics


def build_hybrid_density(root, name, start_year, end_year, *,
                         names=("sensible", "latent"),
                         tail_quantile=0.975, transition_quantile=0.90,
                         reference_weight=0.70, reference_period=(1979, 2026),
                         sample_days=1600, pixels_per_day=160,
                         seed=1979, refresh_samples=False, refresh_hybrid=False):
    """Build or reuse separate, validated experimental density / metadata.

    Does not modify existing stationary PDF or grouped A/B regression cache.
    Returns (grid, hybrid_pdf, sections, details, cached_path).
    """
    root = Path(root)
    original = root / "Functional" / "Stationary" / f"{names[0]}-{names[1]}_{start_year}-{end_year}.npz"
    original_json = original.with_suffix(".json")
    if not original.is_file() or not original_json.is_file():
        raise FileNotFoundError(f"Run main.py task='stationary' first: missing {original} or JSON")
    original_meta = _json(original_json)
    if name not in original_meta.get("fits", {}):
        raise ValueError(f"{name} not found in stationary JSON")
    fit = original_meta["fits"][name]
    if fit.get("stationary_integration_method") != "numeric_cumulative_trapezoid_of_2a_over_b2":
        raise ValueError("Recompute stationary using the latest consistent numerical method")
    interval = fit["reliable_regression_interval"]
    local_sample, local_spec = observed_sample(
        root, name, start_year, end_year, sample_days=sample_days,
        pixels_per_day=pixels_per_day, seed=seed, refresh=refresh_samples)
    if (start_year, end_year) == tuple(reference_period):
        reference_sample, reference_spec = local_sample, local_spec
    else:
        reference_sample, reference_spec = observed_sample(
            root, name, *reference_period, sample_days=sample_days,
            pixels_per_day=pixels_per_day, seed=seed, refresh=refresh_samples)

    tag = f"{name}_{start_year}-{end_year}_gpd_reference_{reference_period[0]}-{reference_period[1]}"
    cache = root / "Functional" / "Stationary" / "Hybrid" / (tag + ".npz")
    cache_meta = cache.with_suffix(".json")
    spec = {
        "schema": SCHEMA_VERSION,
        "original_pdf": _signature(original),
        "original_json": _signature(original_json),
        "local_observations": local_spec,
        "reference_observations": reference_spec,
        "tail_quantile": tail_quantile,
        "transition_quantile": transition_quantile,
        "reference_weight": reference_weight,
        "reference_period": list(reference_period),
        "period": f"{start_year}-{end_year}",
        "variable": name,
    }
    if not refresh_hybrid and not refresh_samples and cache.is_file() and cache_meta.is_file():
        old = _json(cache_meta)
        if old.get("inputs") == spec:
            with np.load(cache, allow_pickle=False) as saved:
                grid = np.array(saved["grid"], copy=True)
                pdf = np.array(saved["pdf"], copy=True)
                sections = np.array(saved["section"], copy=True)
            print(f"Reusing hybrid tail PDF: {cache}", flush=True)
            return grid, pdf, sections, old["diagnostics"], cache

    with np.load(original, allow_pickle=False) as original_pdf:
        grid = np.array(original_pdf["grid"], dtype=np.float64)
        central = np.array(original_pdf[f"pdf_{name}"], dtype=np.float64)
    pdf, sections, diagnostics = _combine(
        grid, central, local_sample, reference_sample, interval,
        tail_quantile=tail_quantile, transition_quantile=transition_quantile,
        reference_weight=reference_weight)
    metadata = {
        "inputs": spec,
        "interpretation": (
            "Experimental semiparametric hybrid density, NOT the stationary solution "
            "of the original drift/diffusion outside the central fit"
        ),
        "coordinate_warning": (
            "GPD tails fitted to raw ERA5 heat-flux observations; full-period "
            "stationary core used observation_minus_A then spatial block averaging. "
            "Their state coordinates are not proven identical."
        ),
        "center_source": str(original),
        "diagnostics": diagnostics,
        "mean": float(np.trapezoid(grid * pdf, grid)),
        "standard_deviation": float(np.sqrt(np.trapezoid(
            (grid - np.trapezoid(grid * pdf, grid)) ** 2 * pdf, grid))),
        "integral_on_output_grid": float(np.trapezoid(pdf, grid)),
        "mass_in_original_fit_support": float(np.trapezoid(
            np.interp(np.linspace(*interval, 201), grid, pdf),
            np.linspace(*interval, 201))),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, grid=grid, pdf=pdf, section=sections)
    _safe_write_json(cache_meta, metadata)
    print(f"Saved hybrid tail PDF: {cache}", flush=True)
    print(f"  mean={metadata['mean']:.5g}; fit-support mass="
          f"{metadata['mass_in_original_fit_support']:.2%}", flush=True)
    for side, d in diagnostics.items():
        print(f"  {side}: core={d['anchor']:.3f}, GPD={d['tail_start']:.3f}, "
              f"shape={d['shape']:.3f}, scale={d['scale']:.3f}, "
              f"empirical scale ratio={d['amplitude_ratio_to_empirical']:.3f}, "
              f"outside grid~{d['estimated_probability_beyond_grid']:.3%}, "
              f"model/observed tail freq={d['predicted_to_observed_tail_probability_ratio']:.3f}",
              flush=True)
        if d["estimated_probability_beyond_grid"] > 0.005:
            print(f"  WARNING: {side} tail has >0.5% probability beyond "
                  "the finite density grid; moments may be unstable", flush=True)
        if abs(d["predicted_to_observed_tail_probability_ratio"] - 1) > 0.3:
            print(f"  WARNING: {side} GPD tail probability differs from "
                  "observed frequency by >30%; inspect the join and histogram", flush=True)
        if not d["finite_gpd_mean"] or not d["finite_gpd_variance"]:
            print(f"  WARNING: {side} estimated GPD shape={d['shape']:.3f}; "
                  "unbounded-domain moments may not exist", flush=True)
    return grid, pdf, sections, diagnostics, cache
