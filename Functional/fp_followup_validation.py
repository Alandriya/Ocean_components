"""Follow-up validation of Ocean_components hybrid tails and wide-stencil 2D FP.

Two independent diagnostics, with no modification to production results:

  hybrid: re-read *stored* hybrid density and original FP core, reproduce the
          implemented bridge, and independently integrate its decay rate by
          adaptive quadrature.  Reports endpoint and grid-replay errors.
  wide_ou: stationary correlated OU benchmark deliberately requiring lattice
           diagonals beyond the 3x3 nearest-neighbour stencil, for both signs
           of D12. Compares CTMC with an independent centered FV method.

The optional mode 'all' runs both.  Designed for the actual public
Functional/hybrid_tails.py, Functional/fokker_planck_2d_solver.py and the
previously installed Functional/fp2d_validation.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import genpareto

from Functional.fokker_planck_2d_solver import _build_generator, _solve_stationary_mass
from Functional.fp2d_validation import (
    _exact_ou_bin_mass, _mass_metrics, _moments,
    centered_full_tensor_fv_generator, solve_fv_stationary,
)
from Functional.hybrid_tails import _smooth_monotone_log_bridge


DEFAULT_PERIODS = ((1979, 1989), (1989, 1999), (1999, 2009),
                   (2009, 2019), (2019, 2026), (1979, 2026))


def _plain(z):
    if isinstance(z, dict):
        return {str(k): _plain(v) for k, v in z.items()}
    if isinstance(z, (list, tuple)):
        return [_plain(v) for v in z]
    if isinstance(z, np.ndarray):
        return _plain(z.tolist())
    if isinstance(z, (np.floating,)):
        z = float(z)
    if isinstance(z, (np.integer,)):
        z = int(z)
    if isinstance(z, float) and not math.isfinite(z):
        return None
    return z


def _save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _save_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows([{k: _plain(r.get(k)) for k in keys} for r in rows])


def _tex_float(z):
    return "--" if z is None or not math.isfinite(float(z)) else f"{float(z):.3g}"


def _bridge_replay_parameters(length, p0, p1, outward_slope, terminal_slope):
    """Reproduce current 1025-point quadrature and solve, then check independently.

    This function returns the numerical c *used by the implementation*, not an
    analytic c.  The separate quad call below is a genuine verification of
    the root equation at a much stricter quadrature tolerance.
    """
    if not (length > 0 and p0 > p1 > 0 and outward_slope < 0 and terminal_slope < 0):
        raise ValueError("Invalid monotone bridge endpoints or slopes")
    logk0 = math.log(-outward_slope)
    logk1 = math.log(-terminal_slope)
    s = np.linspace(0.0, 1.0, 1025)
    base = (1.0 - s) * logk0 + s * logk1
    arch = s * (1.0 - s)
    drop = math.log(p0 / p1)

    def quadrature_on_implementation_nodes(c):
        return float(length * np.trapezoid(np.exp(np.clip(base + c * arch, -700.0, 400.0)), s))

    c = float(brentq(lambda z: quadrature_on_implementation_nodes(z) - drop,
                     -2800.0, 1600.0, xtol=1e-10))

    def exact_integrand(s_):
        # No clipping of ordinary values in independent verification.
        lograte = (1.0 - s_) * logk0 + s_ * logk1 + c * s_ * (1.0 - s_)
        if lograte < -745:
            return 0.0
        if lograte > 700:
            raise FloatingPointError("Unclipped bridge rate overflows")
        return math.exp(lograte)

    # Split near endpoints to control narrow layers for large |c|.
    sections = (0., .01, .05, .15, .35, .65, .85, .95, .99, 1.)
    adaptive_integral = 0.0
    abs_err_bound = 0.0
    for a, b in zip(sections[:-1], sections[1:]):
        val, err = quad(exact_integrand, a, b,
                        epsabs=1e-12, epsrel=1e-11, limit=250)
        adaptive_integral += val
        abs_err_bound += err
    adaptive_integral *= length
    abs_err_bound *= length
    implementation_integral = quadrature_on_implementation_nodes(c)
    log_gap = drop - adaptive_integral
    return {
        "c": c,
        "target_log_drop": drop,
        "implementation_log_drop": implementation_integral,
        "independent_log_drop": adaptive_integral,
        "root_equation_residual": abs(implementation_integral - drop),
        "independent_integral_residual": abs(adaptive_integral - drop),
        "adaptive_quad_reported_error": abs_err_bound,
        "relative_terminal_amplitude_error_bound": float(math.expm1(min(700., abs(log_gap)))),
        "relative_terminal_amplitude_error_actual": float(math.expm1(log_gap)),
    }


def diagnose_hybrid_cache(root, name, start_year, end_year, *,
                          names=("sensible", "latent"),
                          reference_period=(1979, 2026),
                          output_dir=None):
    """Numerically audit one existing saved hybrid reconstruction.

    Both source files are required.  No ERA5 arrays are loaded and no model is
    refitted.  Works with the actual cache naming of build_hybrid_density().
    """
    root = Path(root)
    tag = f"{name}_{start_year}-{end_year}_gpd_reference_{reference_period[0]}-{reference_period[1]}"
    hybrid_npz = root / "Functional" / "Stationary" / "Hybrid" / f"{tag}.npz"
    hybrid_json = hybrid_npz.with_suffix(".json")
    core_npz = root / "Functional" / "Stationary" / f"{names[0]}-{names[1]}_{start_year}-{end_year}.npz"
    for path in (hybrid_npz, hybrid_json, core_npz):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required saved source: {path}")

    meta = json.loads(hybrid_json.read_text(encoding="utf-8"))
    with np.load(hybrid_npz, allow_pickle=False) as f:
        grid = np.asarray(f["grid"], dtype=float).copy()
        observed_pdf = np.asarray(f["pdf"], dtype=float).copy()
        sections = np.asarray(f["section"], dtype=int).copy()
    with np.load(core_npz, allow_pickle=False) as f:
        core_grid = np.asarray(f["grid"], dtype=float).copy()
        core_pdf = np.asarray(f[f"pdf_{name}"], dtype=float).copy()
    if grid.ndim != 1 or not np.array_equal(grid, core_grid):
        raise ValueError("Hybrid and original core grids are not identical")
    if (np.any(np.diff(grid) <= 0) or np.any(core_pdf <= 0)
            or np.any(~np.isfinite(core_pdf)) or np.any(observed_pdf < 0)):
        raise ValueError("Invalid saved density arrays")
    if observed_pdf.shape != grid.shape or sections.shape != grid.shape:
        raise ValueError("Saved hybrid arrays have inconsistent shapes")
    diagnostics = meta["diagnostics"]
    lograw = np.log(core_pdf).copy()
    side_rows = []
    maxgrid = float(np.max(observed_pdf))
    for side, sign in (("left", -1), ("right", 1)):
        info = diagnostics[side]
        anchor = float(info["anchor"])
        threshold = float(info["tail_start"])
        length = sign * (threshold - anchor)
        p0 = float(np.exp(np.interp(anchor, grid, np.log(core_pdf))))
        p1 = float(info["used_tail_density_at_threshold"])
        outward = float(info["core_outward_log_slope"])
        terminal = -(1.0 + float(info["shape"])) / float(info["scale"])
        r = _bridge_replay_parameters(length, p0, p1, outward, terminal)
        dist = sign * (grid - anchor)
        bridge_mask = (dist > 0) & (dist < length)
        tail_mask = dist >= length
        # Strictly check that saved section labels match the actual code paths.
        if np.any(sections[bridge_mask] != sign) or np.any(sections[tail_mask] != 2 * sign):
            raise ValueError(f"Saved section labels disagree with {side}-side transition")
        if np.any(bridge_mask):
            lograw[bridge_mask] = _smooth_monotone_log_bridge(
                dist[bridge_mask], length, p0, outward, p1, terminal
            )
        if np.any(tail_mask):
            ext = dist[tail_mask] - length
            with np.errstate(divide="ignore", invalid="ignore"):
                lp = genpareto.logpdf(ext, c=float(info["shape"]),
                                     loc=0, scale=float(info["scale"]))
            lograw[tail_mask] = np.log(p1 * float(info["scale"])) + lp

        # A finite-spacing one-sided difference is a diagnostic of the sampled
        # *grid*, not a mismatch in the continuously differentiable construction.
        start_idx = int(np.argmin(abs(grid - anchor)))
        k = np.flatnonzero(bridge_mask)
        discrete_slope_error = None
        if k.size:
            j = int(k[np.argmin(dist[k])])
            dxout = dist[j]
            if dxout > 0 and observed_pdf[j] > 0 and observed_pdf[start_idx] > 0:
                fd = (math.log(observed_pdf[j]) - math.log(observed_pdf[start_idx])) / dxout
                discrete_slope_error = abs(fd - outward)
        # Analytical endpoint derivatives of replayed Hermite interpolation
        # are identical by construction, independently of the saved grid.
        row = {
            "period": f"{start_year}-{end_year}", "variable": name, "side": side,
            "L": length, "c": r["c"],
            "integral_residual": r["independent_integral_residual"],
            "implementation_root_residual": r["root_equation_residual"],
            "quad_error_estimate": r["adaptive_quad_reported_error"],
            "relative_terminal_error": abs(r["relative_terminal_amplitude_error_actual"]),
            "relative_terminal_error_bound": r["relative_terminal_amplitude_error_bound"],
            "finite_grid_start_secant_slope_error": discrete_slope_error,
            "observed_amplitude_ratio": float(info.get("amplitude_ratio_to_empirical", float("nan"))),
        }
        side_rows.append(row)

    # Replay the *entire* normalized hybrid construction and compare against
    # the actually saved final grid. This detects changes in implementation or
    # stale caches. The same current bridge function is used for this replay,
    # while the integral residual above is independently evaluated.
    if np.any(np.isnan(lograw)):
        raise ValueError("NaN in reconstructed log density")
    finite = np.isfinite(lograw)
    if not np.any(finite):
        raise ValueError("No finite reconstructed density values")
    scaled = np.exp(np.where(finite, lograw, -np.inf) - float(np.max(lograw[finite])))
    replay = scaled / np.trapezoid(scaled, grid)
    maximum_grid_difference = float(np.max(np.abs(replay - observed_pdf)))
    active = observed_pdf > 1e-10 * maxgrid
    maximum_relative_difference = float(np.max(np.abs(replay[active] - observed_pdf[active]) /
                                                observed_pdf[active])) if np.any(active) else None
    monotonicity_violations = 0
    for side, sign in (("left", -1), ("right", 1)):
        points = np.flatnonzero((sections == sign) | (sections == 2 * sign))
        points = sorted(points, key=lambda k: sign * (grid[k] - float(diagnostics[side]["anchor"])))
        if len(points) > 1:
            vals = observed_pdf[points]
            monotonicity_violations += int(np.count_nonzero(np.diff(vals) > 1e-11 * maxgrid))
    summary = {
        "period": f"{start_year}-{end_year}", "variable": name,
        "source_hybrid_npz": str(hybrid_npz), "source_core_npz": str(core_npz),
        "integral_normalization_error": abs(float(np.trapezoid(observed_pdf, grid)) - 1.0),
        "maximum_absolute_saved_vs_replay_density": maximum_grid_difference,
        "maximum_relative_saved_vs_replay_density": maximum_relative_difference,
        "monotonicity_violation_pairs": monotonicity_violations,
        "maximum_independent_integral_residual": max(row["integral_residual"] for row in side_rows),
        "maximum_relative_terminal_error": max(row["relative_terminal_error"] for row in side_rows),
        "interpretation": (
            "Integral check uses independent adaptive quadrature. Derivative equality at the "
            "continuous bridge endpoints is imposed by construction. The one-sided secant "
            "slope reflects finite output-grid spacing, not a failure of continuous C1 matching. "
            "Checks concern algorithmic joining, not statistical GPD fit error or coordinate equivalence."
        ),
    }
    out_dir = Path(output_dir) if output_dir is not None else root / "Functional" / "Stationary" / "Hybrid" / "Validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_json(out_dir / f"{tag}_audit.json", {"summary": summary, "sides": side_rows})
    print(f"HYBRID {name} {start_year}-{end_year}: integral residual max "
          f"{summary['maximum_independent_integral_residual']:.3g}, "
          f"saved/replay max={maximum_grid_difference:.3g}", flush=True)
    return summary, side_rows


def run_hybrid(root, output_dir, periods=DEFAULT_PERIODS,
               names=("sensible", "latent"), reference_period=(1979, 2026),
               fail_on_missing=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries, sides, failures, invalid = [], [], [], []
    for period in periods:
        for name in names:
            try:
                summary, per_side = diagnose_hybrid_cache(
                    root, name, period[0], period[1], names=names,
                    reference_period=reference_period, output_dir=output_dir
                )
                summaries.append(summary)
                sides.extend(per_side)
            except FileNotFoundError as exc:
                if fail_on_missing:
                    raise
                failures.append({"period": f"{period[0]}-{period[1]}",
                                 "variable": name, "error": str(exc)})
                print(f"SKIPPED missing hybrid cache for {name} {period}: {exc}", flush=True)
            except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
                if fail_on_missing:
                    raise
                invalid.append({"period": f"{period[0]}-{period[1]}",
                                "variable": name, "error": f"{type(exc).__name__}: {exc}"})
                print(f"FAILED hybrid validation for {name} {period}: {exc}", flush=True)
    _save_csv(output_dir / "hybrid_join_summary.csv", summaries)
    _save_csv(output_dir / "hybrid_join_by_side.csv", sides)
    _save_json(output_dir / "hybrid_join_manifest.json", {
        "audited": len(summaries), "sides": len(sides), "skipped": failures, "failed": invalid,
        "periods_requested": [list(p) for p in periods], "names": list(names),
    })
    if not summaries:
        raise FileNotFoundError("No cached hybrid reconstruction available. Run the repository's hybrid-tail builder first.")
    _write_hybrid_latex(output_dir / "hybrid_join_table.tex", sides, summaries)
    return {"summary": summaries, "sides": sides, "skipped": failures, "failed": invalid}


def _write_hybrid_latex(path, side_rows, summary_rows):
    by = {(x["period"], x["variable"]): x for x in summary_rows}
    lines = [r"% Generated numeric audit. Does not include GPD parameter-estimation errors.",
             r"\begin{table}[H]", r"\centering",
             r"\caption{Numerical validation of FP--GPD transition intervals.}",
             r"\label{tab:hybrid_join_numerical_audit}",
             r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Period & Flux/side & $\varepsilon_{\mathrm{int}}$ & Relative endpoint error & Saved/replay & $|Z-1|$ \\",
             r"\midrule"]
    for r in side_rows:
        su = by[(r["period"], r["variable"])]
        lines.append(f"{r['period']} & {r['variable']} {r['side']} & "
                     f"{_tex_float(r['integral_residual'])} & "
                     f"{_tex_float(r['relative_terminal_error'])} & "
                     f"{_tex_float(su['maximum_absolute_saved_vs_replay_density'])} & "
                     f"{_tex_float(su['integral_normalization_error'])}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_wide_ou(output_dir, sizes=(40, 60, 80, 100),
                signs=(-1, 1), max_stencil_radius=9):
    """Exact OU validation forcing a diagonal radius >=2 for every experiment.

    D = [[1, +/-2], [+/-2, 9]]; Sigma = 4 I; domain [-3.5,3.5]^2.
    Since h1=h2, the nearest diagonal has physical slope 1. However,
    positivity of the axial residuals requires slope >= |D12|/D00 = 2
    and slope <= D11/|D12| = 4.5. The drift constraints select
    directions such as (m,n)=(1,3) rather than the nearest diagonal.
    Hence every admissible diagonal requires slope >=2 and, because of drift
    constraints, the selected radius is typically 3. Actual radius counts
    are logged and checked. Positive and negative D12 are both tested.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sigma = 4.0 * np.eye(2)
    mu = np.zeros(2)
    rows = []
    for sign in signs:
        if sign not in (-1, 1):
            raise ValueError("Each sign must be +1 or -1")
        D = np.array([[1., 2. * sign], [2. * sign, 9.]], dtype=float)
        drift_mat = -0.5 * D @ np.linalg.inv(sigma)
        for n in sizes:
            n = int(n)
            if n < 16:
                raise ValueError("Use n >= 16")
            x_edges = np.linspace(-3.5, 3.5, n + 1)
            y_edges = np.linspace(-3.5, 3.5, n + 1)
            xc = (x_edges[:-1] + x_edges[1:]) / 2.
            yc = (y_edges[:-1] + y_edges[1:]) / 2.
            hx, hy = float(x_edges[1] - x_edges[0]), float(y_edges[1] - y_edges[0])
            xx, yy = np.meshgrid(xc, yc, indexing="ij")
            states = np.stack((xx, yy), axis=0)
            drift = np.einsum("ab,bij->aij", drift_mat, states)
            diffusion = np.empty((4, n, n), dtype=float)
            for channel, value in enumerate((D[0, 0], D[1, 1], D[0, 1], D[1, 0])):
                diffusion[channel] = value
            true_mass = _exact_ou_bin_mass(x_edges, y_edges, mu, sigma, quad_order=8)
            domain = np.ones((n, n), dtype=bool)
            start = perf_counter()
            Q, _, radii, _, diag = _build_generator(
                drift, diffusion, domain, true_mass, hx, hy,
                int(max_stencil_radius), 1e-10
            )
            t_ctmc_assembly = perf_counter() - start
            maxradius = diag["maximum_stencil_radius_used"]
            if maxradius < 2 or sum(int(v) for k, v in diag["stencil_radius_counts"].items() if int(k) >= 2) == 0:
                raise AssertionError("This is not a genuine wide-stencil test")
            if diag["strongly_connected_components"] != 1:
                raise RuntimeError("The tested CTMC generator is reducible")
            start = perf_counter()
            pi, solve_diag = _solve_stationary_mass(Q, 1e-11)
            t_ctmc_solve = perf_counter() - start
            ctmc = pi.reshape(n, n)
            ct_err = _mass_metrics(ctmc, true_mass)
            ct_mom = _moments(ctmc, xc, yc)
            ex_mom = _moments(true_mass, xc, yc)
            fv_err = {"total_variation": None, "hellinger": None}
            fv_mass = np.full((n, n), np.nan)
            fv_diag = {"raw_minimum_mass": None, "positivity_valid": False}
            fv_error = None
            t_fv_assembly, t_fv_solve = None, None
            start = perf_counter()
            try:
                op = centered_full_tensor_fv_generator(
                    drift[0], drift[1], diffusion[0], diffusion[1], diffusion[2], hx, hy
                )
                t_fv_assembly = perf_counter() - start
                start = perf_counter()
                fv_vec, fv_diag = solve_fv_stationary(op)
                t_fv_solve = perf_counter() - start
                fv_mass = fv_vec.reshape((n, n))
                if fv_diag["positivity_valid"]:
                    fv_err = _mass_metrics(fv_mass, true_mass)
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                fv_error = str(exc)
            positive_wide = int(np.count_nonzero((radii >= 2) & domain))
            row = {
                "grid": n,
                "D12_sign": int(sign), "D12": float(D[0, 1]),
                "max_stencil_radius": maxradius,
                "fraction_radius_ge_2": positive_wide / float(n * n),
                "radius_counts": json.dumps(diag["stencil_radius_counts"]),
                "ctmc_TV_vs_exact": ct_err["total_variation"],
                "ctmc_Hellinger_vs_exact": ct_err["hellinger"],
                "fv_TV_vs_exact": fv_err["total_variation"],
                "fv_Hellinger_vs_exact": fv_err["hellinger"],
                "fv_raw_minimum_mass": fv_diag.get("raw_minimum_mass"),
                "fv_positivity_valid": bool(fv_diag.get("positivity_valid", False)),
                "fv_error": fv_error,
                "ctmc_max_stationary_residual": solve_diag["maximum_absolute_stationary_residual"],
                "max_pre_censor_drift_error": diag["maximum_first_moment_reconstruction_error"],
                "max_pre_censor_diffusion_error": diag["maximum_second_moment_reconstruction_error"],
                "censored_rate_fraction": diag["total_censored_rate_fraction"],
                "error_mean_x": ct_mom["mean_x"] - ex_mom["mean_x"],
                "error_mean_y": ct_mom["mean_y"] - ex_mom["mean_y"],
                "error_covariance": ct_mom["cov_xy"] - ex_mom["cov_xy"],
                "ctmc_assembly_seconds": t_ctmc_assembly,
                "ctmc_solve_seconds": t_ctmc_solve,
                "fv_assembly_seconds": t_fv_assembly,
                "fv_solve_seconds": t_fv_solve,
            }
            rows.append(row)
            print(f"WIDE-OU D12={D[0,1]:+g} {n}x{n}: "
                  f"radius={maxradius}, wide_cells={positive_wide}/{n*n}, "
                  f"CTMC TV={ct_err['total_variation']:.5g}, "
                  f"FV TV={row['fv_TV_vs_exact']}", flush=True)
            np.savez_compressed(
                out_dir / f"wide_ou_sign_{'pos' if sign>0 else 'neg'}_{n}x{n}.npz",
                x_edges=x_edges, y_edges=y_edges, D=D, Sigma=sigma,
                exact_mass=true_mass, ctmc_mass=ctmc, fv_mass=fv_mass,
                stencil_radii=radii, drift=drift,
            )
    _save_csv(out_dir / "wide_ou_comparison.csv", rows)
    _save_json(out_dir / "wide_ou_comparison.json", {"setup": {
        "Sigma": sigma, "D_for_positive_cross": [[1., 2.], [2., 9.]],
        "D_for_negative_cross": [[1., -2.], [-2., 9.]],
        "domain": [-3.5, 3.5, -3.5, 3.5],
        "explanation": "A unit 3x3 diagonal slope is infeasible since |D12|/D00=2; positive locally consistent jump rates require genuinely wide lattice diagonals.",
        "discretization_warning": "CTMC censored boundary jumps and centered FV zero-face-flux treatment differ at finite h.",
    }, "results": rows})
    _write_wide_latex(out_dir / "wide_ou_table.tex", rows)
    return rows


def _write_wide_latex(path, rows):
    lines = [r"% Synthetic wide-stencil benchmark. Negative FV masses are excluded.",
             r"\begin{table}[H]", r"\centering",
             r"\caption{Stationary OU benchmark requiring wide diagonal jumps.}",
             r"\label{tab:wide_ou_validation}",
             r"\begin{tabular}{rrrrrrr}", r"\toprule",
             r"Grid & $D_{12}$ & Max radius & CTMC TV & FV TV & CTMC $D_H$ & FV $D_H$ \\",
             r"\midrule"]
    for r in rows:
        lines.append(f"${r['grid']}\\times{r['grid']}$ & {r['D12']:+.0f} & {r['max_stencil_radius']} & "
                     f"{_tex_float(r['ctmc_TV_vs_exact'])} & "
                     f"{_tex_float(r['fv_TV_vs_exact'])} & "
                     f"{_tex_float(r['ctmc_Hellinger_vs_exact'])} & "
                     f"{_tex_float(r['fv_Hellinger_vs_exact'])}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_followup(mode="all", root=None, output_dir=None,
                 hybrid_periods=DEFAULT_PERIODS,
                 names=("sensible", "latent"),
                 wide_sizes=(40, 60, 80, 100),
                 max_stencil_radius=9,
                 require_all_hybrids=False):
    if mode not in ("hybrid", "wide_ou", "all"):
        raise ValueError("mode must be hybrid, wide_ou, or all")
    root = Path(root) if root is not None else Path(".")
    output_dir = Path(output_dir) if output_dir is not None else (
        root / "Functional" / "FokkerPlanck2D" / "Validation" / "Followup"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"mode": mode, "output_dir": str(output_dir.resolve())}
    if mode in ("wide_ou", "all"):
        result["wide_ou"] = run_wide_ou(
            output_dir / "wide_ou", sizes=wide_sizes,
            max_stencil_radius=max_stencil_radius,
        )
    if mode in ("hybrid", "all"):
        result["hybrid"] = run_hybrid(
            root, output_dir / "hybrid", periods=hybrid_periods,
            names=names, fail_on_missing=require_all_hybrids
        )
    _save_json(output_dir / "followup_manifest.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("wide_ou", "hybrid", "all"), default="wide_ou")
    p.add_argument("--root", default=".")
    p.add_argument("--output", default=None)
    p.add_argument("--sizes", type=int, nargs="+", default=[40, 60, 80, 100])
    p.add_argument("--radius", type=int, default=9)
    p.add_argument("--period", type=int, nargs=2, action="append", default=None,
                   metavar=("START", "END"))
    p.add_argument("--strict-hybrid", action="store_true")
    args = p.parse_args()
    run_followup(mode=args.mode, root=args.root, output_dir=args.output,
                 wide_sizes=args.sizes, max_stencil_radius=args.radius,
                 hybrid_periods=tuple(map(tuple, args.period)) if args.period else DEFAULT_PERIODS,
                 require_all_hybrids=args.strict_hybrid)
