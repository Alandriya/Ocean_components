"""Reproducible 2-D Fokker--Planck validation for Ocean_components.

Designed for the public module Functional/fokker_planck_2d_solver.py:
 _build_generator, _local_jump_rates, _candidate_directions, _jump_moments,
 _solve_stationary_mass.

The validation intentionally does not change the production solver or ERA5 data.

Modes:
  synthetic: correlated 2-D OU exact stationary distribution, Markov vs
             independent conservative centered finite-volume discretization.
  era5:       recompute pre-/post-censor jump moments on saved solver NPZs.
  all:        run both; ERA5 requires saved files.

Note: the finite-volume comparison is an independently coded *centered* method,
not an implementation of Loy--Zanella and not guaranteed positivity preserving.
"""

from __future__ import annotations

import csv
import json
import math
from time import perf_counter
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, spsolve
import warnings

from Functional.fokker_planck_2d_solver import (
    _build_generator,
    _candidate_directions,
    _jump_moments,
    _local_jump_rates,
    _solve_stationary_mass,
)


def _jsonable(value):
    """Compute jsonable for 2D Fokker-Planck validation from ``value``."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, obj):
    """Write JSON for the 2D Fokker-Planck validation workflow."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]):
    """Write the supplied records to a CSV file."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _safe_weights(mass, domain):
    """Compute safe weights for 2D Fokker-Planck validation from ``mass``, ``domain``."""
    weights = np.asarray(mass, dtype=float)[domain]
    if not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Invalid stationary weights")
    return weights / weights.sum()


def _mass_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    """Compute mass metrics for 2D Fokker-Planck validation from ``a``, ``b``."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if np.any(a < -1e-11) or np.any(b < -1e-11):
        raise ValueError("A compared distribution has negative masses")
    a = np.clip(a, 0, None)
    b = np.clip(b, 0, None)
    a /= a.sum()
    b /= b.sum()
    return {
        "total_variation": float(0.5 * np.sum(np.abs(a - b))),
        "hellinger": float(np.sqrt(0.5 * np.sum((np.sqrt(a) - np.sqrt(b)) ** 2))),
        "l1": float(np.sum(np.abs(a - b))),
    }


def _moments(mass, xx, yy):
    """Compute the moments used in the current diagnostic."""
    p = np.asarray(mass, dtype=float)
    p = p / p.sum()
    x, y = np.meshgrid(xx, yy, indexing="ij")
    mx, my = float(np.sum(p * x)), float(np.sum(p * y))
    cov_xx = float(np.sum(p * (x - mx) ** 2))
    cov_yy = float(np.sum(p * (y - my) ** 2))
    cov_xy = float(np.sum(p * (x - mx) * (y - my)))
    return {"mean_x": mx, "mean_y": my,
            "var_x": cov_xx, "var_y": cov_yy, "cov_xy": cov_xy}


def _exact_ou_bin_mass(x_edges, y_edges, mu, sigma, quad_order=5):
    """Tensor Gauss--Legendre integration of unnormalized 2-D Gaussian in bins."""
    wpos, w = leggauss(quad_order)
    xmid = 0.5 * (x_edges[1:] + x_edges[:-1])
    ymid = 0.5 * (y_edges[1:] + y_edges[:-1])
    hx = np.diff(x_edges)
    hy = np.diff(y_edges)
    xx = xmid[:, None] + 0.5 * hx[:, None] * wpos[None, :]
    yy = ymid[:, None] + 0.5 * hy[:, None] * wpos[None, :]
    inv = np.linalg.inv(sigma)
    norm = 2 * math.pi * math.sqrt(np.linalg.det(sigma))
    masses = np.zeros((len(xmid), len(ymid)))
    for k, wx in enumerate(w):
        for l, wy in enumerate(w):
            dx = xx[:, k][:, None] - mu[0]
            dy = yy[:, l][None, :] - mu[1]
            exponent = -0.5 * (inv[0, 0] * dx * dx + 2 * inv[0, 1] * dx * dy + inv[1, 1] * dy * dy)
            masses += wx * wy * np.exp(exponent) / norm
    masses *= 0.25 * hx[:, None] * hy[None, :]
    return masses / masses.sum()


def _sparse_map(rows, cols, data, shape):
    """Compute sparse map for 2D Fokker-Planck validation from ``rows``, ``cols``, ``data``, ``shape``."""
    return sparse.csr_matrix((np.asarray(data, dtype=float), (rows, cols)), shape=shape)


def centered_full_tensor_fv_generator(a0, a1, d00, d11, d01, hx, hy):
    """Independent zero-normal-flux central finite-volume forward operator.

    For p cell-centred and J_i = A_i p - (1/2) sum_j d_j(D_ij p),
    each internal face flux uses centered normal differences and a symmetric
    average of center-to-center transverse derivatives. Boundary face fluxes
    are *zero*. D_ij may vary. This is conservative by construction, but NOT
    monotone or guaranteed to produce a nonnegative stationary solution.

    Returns K such that dp/dt = K @ p (stationary K @ p = 0).
    """
    a0, a1, d00, d11, d01 = (np.asarray(z, dtype=float) for z in (a0, a1, d00, d11, d01))
    nx, ny = a0.shape
    for z in (a1, d00, d11, d01):
        if z.shape != (nx, ny):
            raise ValueError("All coefficient arrays must have the same shape")
    if nx < 3 or ny < 3 or hx <= 0 or hy <= 0:
        raise ValueError("Need >=3 cells per direction and positive widths")
    N = nx * ny
    def idx(i, j):
        """Compute idx for centered full tensor fv generator from ``i``, ``j``."""
        return i * ny + j

    # Cell-centre gradients including one-sided derivatives at rectangle edges.
    gx_r, gx_c, gx_v, gy_r, gy_c, gy_v = [], [], [], [], [], []
    for i in range(nx):
        for j in range(ny):
            k = idx(i, j)
            if i == 0:
                parts = [(i, -1.0 / hx), (i + 1, 1.0 / hx)]
            elif i == nx - 1:
                parts = [(i - 1, -1.0 / hx), (i, 1.0 / hx)]
            else:
                parts = [(i - 1, -0.5 / hx), (i + 1, 0.5 / hx)]
            for ii, v in parts:
                gx_r.append(k); gx_c.append(idx(ii, j)); gx_v.append(v)
            if j == 0:
                parts = [(j, -1.0 / hy), (j + 1, 1.0 / hy)]
            elif j == ny - 1:
                parts = [(j - 1, -1.0 / hy), (j, 1.0 / hy)]
            else:
                parts = [(j - 1, -0.5 / hy), (j + 1, 0.5 / hy)]
            for jj, v in parts:
                gy_r.append(k); gy_c.append(idx(i, jj)); gy_v.append(v)
    gx = _sparse_map(gx_r, gx_c, gx_v, (N, N))
    gy = _sparse_map(gy_r, gy_c, gy_v, (N, N))

    # Vertical faces: average, normal gradient, conservative incidence.
    va_r, va_c, va_v, vg_r, vg_c, vg_v, vi_r, vi_c, vi_v = ([] for _ in range(9))
    for i in range(nx - 1):
        for j in range(ny):
            f = i * ny + j
            left, right = idx(i, j), idx(i + 1, j)
            for k in (left, right):
                va_r.append(f); va_c.append(k); va_v.append(0.5)
            for k, v in ((left, -1.0 / hx), (right, 1.0 / hx)):
                vg_r.append(f); vg_c.append(k); vg_v.append(v)
            for k, v in ((left, -1.0 / hx), (right, 1.0 / hx)):
                vi_r.append(k); vi_c.append(f); vi_v.append(v)
    nf_v = (nx - 1) * ny
    va = _sparse_map(va_r, va_c, va_v, (nf_v, N))
    vg = _sparse_map(vg_r, vg_c, vg_v, (nf_v, N))
    vi = _sparse_map(vi_r, vi_c, vi_v, (N, nf_v))

    # Horizontal faces.
    ha_r, ha_c, ha_v, hg_r, hg_c, hg_v, hi_r, hi_c, hi_v = ([] for _ in range(9))
    for i in range(nx):
        for j in range(ny - 1):
            f = i * (ny - 1) + j
            bottom, top = idx(i, j), idx(i, j + 1)
            for k in (bottom, top):
                ha_r.append(f); ha_c.append(k); ha_v.append(0.5)
            for k, v in ((bottom, -1.0 / hy), (top, 1.0 / hy)):
                hg_r.append(f); hg_c.append(k); hg_v.append(v)
                hi_r.append(k); hi_c.append(f); hi_v.append(v)
    nf_h = nx * (ny - 1)
    ha = _sparse_map(ha_r, ha_c, ha_v, (nf_h, N))
    hg = _sparse_map(hg_r, hg_c, hg_v, (nf_h, N))
    hi = _sparse_map(hi_r, hi_c, hi_v, (N, nf_h))

    fx = va @ sparse.diags(a0.ravel()) - 0.5 * (
        vg @ sparse.diags(d00.ravel()) + va @ gy @ sparse.diags(d01.ravel())
    )
    fy = ha @ sparse.diags(a1.ravel()) - 0.5 * (
        hg @ sparse.diags(d11.ravel()) + ha @ gx @ sparse.diags(d01.ravel())
    )
    # vi maps an internal face to -F_right/h on its left cell and
    # +F_left/h on its right cell, hence minus the flux divergence.
    op = vi @ fx + hi @ fy
    op = op.tocsr()
    op.sum_duplicates()
    return op


def solve_fv_stationary(op, positivity_tol=1e-10):
    """Solve Kp=0 with sum p=1 (all rectangular cell areas identical)."""
    n = op.shape[0]
    K = op.tolil(copy=True)
    K[0, :] = np.ones(n)
    K = K.tocsr()
    rhs = np.zeros(n)
    rhs[0] = 1.0
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=MatrixRankWarning)
        p = np.asarray(spsolve(K, rhs), dtype=float)
    if not np.all(np.isfinite(p)) or abs(p.sum() - 1.0) > 1e-6:
        raise ValueError("FV solve returned invalid probabilities")
    residual = np.asarray(op @ p).ravel()
    info = {"raw_minimum_mass": float(p.min()),
            "raw_negative_mass_total": float(np.sum(-p[p < 0])),
            "stationary_residual_linf": float(np.max(np.abs(residual))),
            "positivity_valid": bool(float(p.min()) >= -positivity_tol),
            "conservation_max_column_sum": float(np.max(np.abs(np.asarray(op.sum(axis=0)).ravel())))}
    if info["positivity_valid"]:
        p = np.maximum(p, 0.0)
        p /= p.sum()
    return p, info


def run_synthetic(out_dir, sizes=(40, 48, 56), radius=9, rho_cross=0.36):
    """Compute run synthetic for 2D Fokker-Planck validation from ``out_dir``, ``sizes``, ``radius``, ``rho_cross``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (-0.95 < rho_cross < 0.95):
        raise ValueError("rho_cross must be inside (-0.95,0.95)")
    sigma = np.array([[0.85, 0.45], [0.45, 1.2]], dtype=float)
    d = np.array([[0.8, rho_cross * math.sqrt(0.8 * 0.72)],
                  [rho_cross * math.sqrt(0.8 * 0.72), 0.72]], dtype=float)
    mu = np.array([0.0, 0.0])
    drift_matrix = -0.5 * (d @ np.linalg.inv(sigma))
    rows = []
    for n in sizes:
        n = int(n)
        if n < 8:
            raise ValueError("Synthetic grid size must be >=8")
        x_edges = np.linspace(-3.0, 3.0, n + 1)
        y_edges = np.linspace(-3.0, 3.0, n + 1)
        xx = 0.5 * (x_edges[:-1] + x_edges[1:])
        yy = 0.5 * (y_edges[:-1] + y_edges[1:])
        hx, hy = float(x_edges[1] - x_edges[0]), float(y_edges[1] - y_edges[0])
        x, y = np.meshgrid(xx, yy, indexing="ij")
        shift = np.stack([x - mu[0], y - mu[1]], axis=0)
        drift = np.einsum("ab,bij->aij", drift_matrix, shift)
        diffusion = np.empty((4, n, n))
        for k, v in enumerate((d[0, 0], d[1, 1], d[0, 1], d[1, 0])):
            diffusion[k] = v
        true_mass = _exact_ou_bin_mass(x_edges, y_edges, mu, sigma)
        mask = np.ones((n, n), dtype=bool)

        markov_start = perf_counter()
        q, coords, radii, censored, diag = _build_generator(
            drift, diffusion, mask, true_mass, hx, hy, radius, 1e-10
        )
        markov_assembly_s = perf_counter() - markov_start
        if diag["strongly_connected_components"] != 1:
            raise RuntimeError(f"Markov chain not irreducible for N={n}")
        markov_solve_start = perf_counter()
        pi, solver_diag = _solve_stationary_mass(q, 1e-11)
        markov_solve_s = perf_counter() - markov_solve_start
        markov_mass = np.zeros((n, n))
        markov_mass[mask] = pi
        moment_error = {k: float(_moments(markov_mass, xx, yy)[k] - _moments(true_mass, xx, yy)[k])
                        for k in _moments(true_mass, xx, yy)}
        markov_err = _mass_metrics(markov_mass, true_mass)

        fv_start = perf_counter()
        op = centered_full_tensor_fv_generator(
            drift[0], drift[1], diffusion[0], diffusion[1], diffusion[2], hx, hy
        )
        fv_assembly_s = perf_counter() - fv_start
        fv_solve_start = perf_counter()
        fv_vector, fv_diag = solve_fv_stationary(op)
        fv_solve_s = perf_counter() - fv_solve_start
        fv_mass = fv_vector.reshape(n, n)
        if fv_diag["positivity_valid"]:
            fv_err = _mass_metrics(fv_mass, true_mass)
            between = _mass_metrics(fv_mass, markov_mass)
        else:
            fv_err = {"total_variation": None, "hellinger": None, "l1": None}
            between = {"total_variation": None, "hellinger": None, "l1": None}
        row = {
            "grid": n, "h": hx, "cross_D12": d[0, 1],
            "markov_generator_nonzeros": int(q.nnz), "fv_operator_nonzeros": int(op.nnz),
            "markov_assembly_seconds": markov_assembly_s, "markov_solve_seconds": markov_solve_s,
            "fv_assembly_seconds": fv_assembly_s, "fv_solve_seconds": fv_solve_s,
            "markov_TV_vs_exact": markov_err["total_variation"],
            "markov_Hellinger_vs_exact": markov_err["hellinger"],
            "fv_TV_vs_exact": fv_err["total_variation"],
            "fv_Hellinger_vs_exact": fv_err["hellinger"],
            "markov_vs_fv_TV": between["total_variation"],
            "markov_stationary_residual": solver_diag["maximum_absolute_stationary_residual"],
            "fv_stationary_residual": fv_diag["stationary_residual_linf"],
            "fv_minimum_raw_mass": fv_diag["raw_minimum_mass"],
            "fv_positivity_valid": fv_diag["positivity_valid"],
            "markov_pre_censor_drift_error": diag["maximum_first_moment_reconstruction_error"],
            "markov_pre_censor_diffusion_error": diag["maximum_second_moment_reconstruction_error"],
            "max_stencil_radius": diag["maximum_stencil_radius_used"],
            "censored_rate_fraction": diag["total_censored_rate_fraction"],
            "error_mean_x": moment_error["mean_x"],
            "error_mean_y": moment_error["mean_y"],
            "error_cov_xy": moment_error["cov_xy"],
        }
        rows.append(row)
        print(f"OU {n}x{n}  CTMC TV={row['markov_TV_vs_exact']:.6g}  "
              f"FV TV={str(row['fv_TV_vs_exact'])}  "
              f"FV min={row['fv_minimum_raw_mass']:.3g}", flush=True)
        np.savez_compressed(out_dir / f"ou_{n}x{n}_solutions.npz",
                            x_edges=x_edges, y_edges=y_edges, exact_mass=true_mass,
                            markov_mass=markov_mass, finite_volume_raw_mass=fv_mass,
                            drift=drift, diffusion=diffusion)
    _write_csv(out_dir / "synthetic_comparison.csv", rows)
    _write_json(out_dir / "synthetic_comparison.json", {"setup": {
        "model": "reversible correlated OU, pointwise zero probability current",
        "Sigma": sigma.tolist(), "D": d.tolist(), "mu": mu.tolist(),
        "domain": [-3.0, 3.0, -3.0, 3.0],
        "boundary": "censored CTMC jumps vs zero FV face flux",
        "warning": "FV comparator is centered and not positivity preserving. Different finite-grid reflecting boundaries are not strictly identical."
    }, "results": rows})
    _write_latex_synthetic(out_dir / "synthetic_comparison_table.tex", rows)
    return rows


def _write_latex_synthetic(path, rows):
    """Write latex synthetic for the 2D Fokker-Planck validation workflow."""
    out = [r"% Automatically generated. Synthetic OU results, not ERA5.",
           r"\begin{table}[H]", r"\centering", r"\caption{Synthetic correlated Ornstein--Uhlenbeck benchmark on a fixed rectangle.}",
           r"\label{tab:fp2d_ou_validation}", r"\begin{tabular}{rrrrr}",
           r"\toprule", r"Grid & CTMC TV & FV TV & CTMC $D_{\rm H}$ & FV $D_{\rm H}$ \\", r"\midrule"]
    for z in rows:
        def fmt(v):
            """Format the supplied value for diagnostic output."""
            return f"{v:.4g}" if v is not None else "--"
        out.append(f"${z['grid']}\\times{z['grid']}$ & {fmt(z['markov_TV_vs_exact'])} & {fmt(z['fv_TV_vs_exact'])} & {fmt(z['markov_Hellinger_vs_exact'])} & {fmt(z['fv_Hellinger_vs_exact'])}" + r" \\")
    out.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def diagnose_saved_era5(stationary_npz, out_dir, radius=9, tol=1e-10):
    """Audit actual stored solver coefficients and retained domain, no ERA5 reload.

    Uses precisely the domain_mask, drift_solver, diffusion_solver and stationary
    mass saved by the production solver. Rebuilds *proposed* rates and separates
    local moment residuals from boundary-censored residuals.
    """
    stationary_npz = Path(stationary_npz)
    if not stationary_npz.is_file():
        raise FileNotFoundError(f"Missing solved NPZ: {stationary_npz}")
    with np.load(stationary_npz, allow_pickle=False) as f:
        required = {"x_edges", "y_edges", "domain_mask", "drift_solver", "diffusion_solver", "stationary_mass"}
        missing = required.difference(f.files)
        if missing:
            raise ValueError(f"Saved stationary NPZ missing required keys: {sorted(missing)}")
        dat = {k: np.asarray(f[k]).copy() for k in required}
        if "empirical_mass_conditioned" in f.files:
            empirical = np.asarray(f["empirical_mass_conditioned"]).copy()
        else:
            empirical = None
    domain = dat["domain_mask"].astype(bool)
    drift = dat["drift_solver"]
    diffusion = dat["diffusion_solver"]
    x_edges, y_edges = dat["x_edges"], dat["y_edges"]
    hx = float(x_edges[1] - x_edges[0])
    hy = float(y_edges[1] - y_edges[0])
    ncell = int(np.sum(domain))
    if ncell < 2:
        raise ValueError("Need >=2 retained states")
    if (not np.allclose(np.diff(x_edges), hx, rtol=1e-12)
            or not np.allclose(np.diff(y_edges), hy, rtol=1e-12)):
        raise ValueError("Nonuniform grid cannot use production wide stencil")
    coords = np.argwhere(domain)
    stat_w = _safe_weights(dat["stationary_mass"], domain)
    if empirical is not None:
        emp_w = _safe_weights(empirical, domain)
    else:
        emp_w = np.full(ncell, 1.0 / ncell)

    candidates = _candidate_directions(hx, hy, int(radius))
    rows = []
    for k, (i, j) in enumerate(coords):
        a0, a1 = float(drift[0, i, j]), float(drift[1, i, j])
        d00 = float(diffusion[0, i, j]); d11 = float(diffusion[1, i, j])
        d01 = 0.5 * float(diffusion[2, i, j] + diffusion[3, i, j])
        jumps, choice = _local_jump_rates(
            a0, a1, d00, d11, d01, hx, hy, candidates, float(tol)
        )
        kept, rejected = [], []
        for di, dj, rate in jumps:
            ni, nj = i + di, j + dj
            if (0 <= ni < domain.shape[0] and 0 <= nj < domain.shape[1]
                    and domain[ni, nj]):
                kept.append((di, dj, rate))
            else:
                rejected.append((di, dj, rate))
        pre_a, pre_d = _jump_moments(jumps, hx, hy)
        post_a, post_d = _jump_moments(kept, hx, hy)
        a = np.array([a0, a1])
        d = np.array([[d00, d01], [d01, d11]])
        proposed_rate = sum(q for _, _, q in jumps)
        rejected_rate = sum(q for _, _, q in rejected)
        rmax = max((math.hypot(di * hx, dj * hy)
                    for di, dj, rate in jumps if rate > 0), default=0.0)
        third = sum(rate * math.hypot(di * hx, dj * hy) ** 3 / 6.0
                    for di, dj, rate in jumps)
        rows.append({
            "i": int(i), "j": int(j), "x": float(0.5 * (x_edges[i]+x_edges[i+1])),
            "y": float(0.5 * (y_edges[j]+y_edges[j+1])),
            "stationary_weight": float(stat_w[k]), "empirical_weight": float(emp_w[k]),
            "pre_drift_error": float(np.max(np.abs(pre_a - a))),
            "pre_diffusion_error": float(np.max(np.abs(pre_d - d))),
            "post_drift_error": float(np.max(np.abs(post_a - a))),
            "post_diffusion_error": float(np.max(np.abs(post_d - d))),
            "censored_rate_fraction": float(rejected_rate / proposed_rate) if proposed_rate else 0.0,
            "censored_rate": float(rejected_rate),
            "proposed_rate": float(proposed_rate),
            "censored_any": bool(rejected_rate > 0),
            "max_jump_length": float(rmax),
            "third_order_remainder_coefficient": float(third),
            "stencil_radius": int(max(choice["m"], choice["n"])),
        })
    def maxof(key):
        """Compute maxof for diagnose saved ERA5 from ``key``."""
        return max(r[key] for r in rows)
    def meanof(key, w):
        """Compute meanof for diagnose saved ERA5 from ``key``, ``w``."""
        return float(np.dot([r[key] for r in rows], w))
    summary = {
        "source": str(stationary_npz.resolve()), "grid_shape": list(domain.shape),
        "retained_cells": ncell, "h1": hx, "h2": hy,
        "search_radius": int(radius),
        "maximum_stencil_radius_used": int(maxof("stencil_radius")),
        "max_jump_length": maxof("max_jump_length"),
        "max_pre_drift_error": maxof("pre_drift_error"),
        "max_pre_diffusion_error": maxof("pre_diffusion_error"),
        "max_post_drift_error": maxof("post_drift_error"),
        "max_post_diffusion_error": maxof("post_diffusion_error"),
        "stationary_weighted_post_drift_error": meanof("post_drift_error", stat_w),
        "stationary_weighted_post_diffusion_error": meanof("post_diffusion_error", stat_w),
        "empirical_weighted_post_drift_error": meanof("post_drift_error", emp_w),
        "empirical_weighted_post_diffusion_error": meanof("post_diffusion_error", emp_w),
        "cells_with_positive_censored_rate": int(sum(z["censored_any"] for z in rows)),
        "fraction_cells_with_positive_censored_rate": float(np.mean([r["censored_any"] for r in rows])),
        "stationary_mass_on_censored_cells": float(sum(r["stationary_weight"] for r in rows if r["censored_any"])),
        "stationary_weighted_censored_fraction": meanof("censored_rate_fraction", stat_w),
        "empirical_weighted_censored_fraction": meanof("censored_rate_fraction", emp_w),
        "global_censored_rate_fraction": float(sum(z["censored_rate"] for z in rows) / sum(z["proposed_rate"] for z in rows)),
        "max_third_order_remainder_coefficient": maxof("third_order_remainder_coefficient"),
        "stationary_weighted_third_order_remainder_coefficient": meanof("third_order_remainder_coefficient", stat_w),
        "note": "Pre-censor moments are algebraically exact where rates are feasible. Post-censor moments generally are not. These metrics do not prove continuum invariant-measure convergence.",
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / f"{stationary_npz.stem}_cell_audit.csv", rows)
    _write_json(out_dir / f"{stationary_npz.stem}_summary.json", summary)
    print(f"ERA5 {stationary_npz.name}: max pre drift error {summary['max_pre_drift_error']:.3g}, "
          f"max post drift error {summary['max_post_drift_error']:.3g}, "
          f"censored cells {summary['cells_with_positive_censored_rate']}/{ncell}", flush=True)
    return summary



def _largest_fully_retained_square(mask):
    """Index bounds of the largest all-True square in a Boolean solver mask."""
    valid = np.asarray(mask, dtype=bool)
    nx, ny = valid.shape
    dp = np.zeros((nx + 1, ny + 1), dtype=np.int32)
    size = 0
    end = (0, 0)
    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            if valid[i - 1, j - 1]:
                dp[i, j] = min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]) + 1
                if dp[i, j] > size:
                    size = int(dp[i, j])
                    end = (i, j)
    return (end[0] - size, end[0], end[1] - size, end[1])


def compare_era5_on_retained_rectangle(stationary_npz, out_dir,
                                       radius=9, tol=1e-10, minimum_side=12):
    """CTMC/FV comparison using identical *real* coefficients on a full square.

    This is a secondary benchmark: BOTH methods are recalculated on the
    largest fully retained square. Their results are NOT compared as if they
    were stationary solutions on the original irregular retained domain.
    """
    stationary_npz = Path(stationary_npz)
    with np.load(stationary_npz, allow_pickle=False) as f:
        mask = np.asarray(f["domain_mask"]).astype(bool)
        i0, i1, j0, j1 = _largest_fully_retained_square(mask)
        side = i1 - i0
        if side < minimum_side:
            raise ValueError(f"No fully retained {minimum_side}x{minimum_side} square (largest side={side})")
        drift = np.asarray(f["drift_solver"])[:, i0:i1, j0:j1].copy()
        D = np.asarray(f["diffusion_solver"])[:, i0:i1, j0:j1].copy()
        empirical = np.asarray(f["empirical_mass_conditioned"])[i0:i1, j0:j1].copy()
        xe = np.asarray(f["x_edges"])[i0:i1 + 1].copy()
        ye = np.asarray(f["y_edges"])[j0:j1 + 1].copy()
    hx, hy = float(xe[1] - xe[0]), float(ye[1] - ye[0])
    ones = np.ones((side, side), dtype=bool)
    if np.any(~np.isfinite(drift)) or np.any(~np.isfinite(D)):
        raise ValueError("Nonfinite real coefficients in fully retained rectangle")
    start = perf_counter()
    q, _, _, _, gd = _build_generator(drift, D, ones, empirical, hx, hy, radius, tol)
    t_build_ctmc = perf_counter() - start
    if gd["strongly_connected_components"] != 1:
        raise ValueError("Selected rectangle gives a reducible Markov chain")
    start = perf_counter()
    ctmc_mass, stationary_diag = _solve_stationary_mass(q, 1e-11)
    t_solve_ctmc = perf_counter() - start

    start = perf_counter()
    K = centered_full_tensor_fv_generator(drift[0], drift[1], D[0], D[1],
                                          0.5 * (D[2] + D[3]), hx, hy)
    t_build_fv = perf_counter() - start
    start = perf_counter()
    fv_mass, fv_diag = solve_fv_stationary(K)
    t_solve_fv = perf_counter() - start
    if fv_diag["positivity_valid"]:
        comparisons = _mass_metrics(ctmc_mass, fv_mass)
    else:
        comparisons = {"total_variation": None, "hellinger": None,
                       "l1": None}
    results = {
        "source": str(stationary_npz.resolve()),
        "rectangle_index_bounds_i0_i1_j0_j1": [i0, i1, j0, j1],
        "side_cells": side, "h1": hx, "h2": hy,
        "markov": gd, "markov_linear_solve": stationary_diag,
        "fv": fv_diag, "ctmc_vs_fv_on_same_rectangle": comparisons,
        "ctmc_build_seconds": t_build_ctmc, "ctmc_solve_seconds": t_solve_ctmc,
        "fv_build_seconds": t_build_fv, "fv_solve_seconds": t_solve_fv,
        "ctmc_nonzeros": int(q.nnz), "fv_nonzeros": int(K.nnz),
        "interpretation": "Same reconstructed coefficients and same rectangular domain, but different finite-grid boundary treatments. Does not assess accuracy on the original irregular ERA5 domain.",
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stationary_npz.stem
    _write_json(out_dir / f"{stem}_rectangle_comparison.json", results)
    np.savez_compressed(out_dir / f"{stem}_rectangle_solutions.npz",
                        x_edges=xe, y_edges=ye,
                        ctmc_mass=ctmc_mass.reshape((side, side)),
                        fv_mass=fv_mass.reshape((side, side)),
                        drift=drift, diffusion=D)
    print(f"ERA5 full retained rectangle {side}x{side}: CTMC/FV TV={comparisons['total_variation']}, "
          f"FV positivity valid={fv_diag['positivity_valid']}", flush=True)
    return results


def _write_era5_latex(path, rows):
    """Write ERA5 latex for the 2D Fokker-Planck validation workflow."""
    out = [r"% Automatically generated from the saved ERA5 stationary NPZ files.",
           r"% Audit of local moments before and after censoring. Not a convergence theorem.",
           r"\begin{table}[H]", r"\centering", r"\caption{Local moment reconstruction errors before and after boundary censoring.}",
           r"\label{tab:fp2d_boundary_consistency_audit}",
           r"\begin{tabular}{rrrrrr}", r"\toprule",
           r"Grid & Pre drift & Post drift & Pre diffusion & Post diffusion & Censored cells \\",
           r"\midrule"]
    for z in rows:
        shape = z["grid_shape"]
        out.append(f"${shape[0]}\\times{shape[1]}$ & {z['max_pre_drift_error']:.2e} & {z['max_post_drift_error']:.2e} & {z['max_pre_diffusion_error']:.2e} & {z['max_post_diffusion_error']:.2e} & {z['cells_with_positive_censored_rate']}" + r" \\")
    out.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def run_validation(mode="synthetic", output_dir=None,
                   synthetic_sizes=(40, 48, 56),
                   stationary_npzs=(), max_stencil_radius=9,
                   rate_tolerance=1e-10, correlation=0.36,
                   compare_era5_rectangles=False):
    """Callable from main.py's run_selected_task or Python CLI."""
    if mode not in ("synthetic", "era5", "all"):
        raise ValueError("mode must be 'synthetic', 'era5' or 'all'")
    output_dir = Path(output_dir) if output_dir is not None else Path("Functional/FokkerPlanck2D/Validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"mode": mode, "output_dir": str(output_dir.resolve()), "synthetic": [], "era5": [], "era5_rectangle_comparisons": []}
    if mode in ("synthetic", "all"):
        result["synthetic"] = run_synthetic(
            output_dir / "synthetic", sizes=synthetic_sizes,
            radius=max_stencil_radius, rho_cross=correlation
        )
    if mode in ("era5", "all"):
        if not stationary_npzs:
            raise ValueError("ERA5 mode needs existing stationary_npzs: run fokker_planck_2d_solve first")
        out = output_dir / "era5"
        for path in stationary_npzs:
            result["era5"].append(diagnose_saved_era5(
                path, out, radius=max_stencil_radius, tol=rate_tolerance
            ))
        if compare_era5_rectangles:
            for path in stationary_npzs:
                try:
                    result["era5_rectangle_comparisons"].append(
                        compare_era5_on_retained_rectangle(
                            path, out / "rectangular_comparisons",
                            radius=max_stencil_radius, tol=rate_tolerance
                        )
                    )
                except (ValueError, RuntimeError, MatrixRankWarning) as exc:
                    warning = {"source": str(path), "status": "not computed",
                               "reason": str(exc)}
                    result["era5_rectangle_comparisons"].append(warning)
                    print(f"ERA5 rectangle comparison skipped: {warning}", flush=True)
        _write_era5_latex(out / "era5_boundary_audit_table.tex", result["era5"])
        _write_csv(out / "era5_boundary_audit_summary.csv", result["era5"])
    _write_json(output_dir / "validation_manifest.json", result)
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["synthetic", "era5", "all"], default="synthetic")
    p.add_argument("--output", default="Functional/FokkerPlanck2D/Validation")
    p.add_argument("--sizes", nargs="+", type=int, default=[40, 48, 56])
    p.add_argument("--stationary-npz", nargs="*", default=[])
    p.add_argument("--radius", type=int, default=9)
    p.add_argument("--correlation", type=float, default=0.36)
    p.add_argument("--compare-era5-rectangles", action="store_true",
                   help="Also compare CTMC/FV on each saved grid's largest fully retained square")
    args = p.parse_args()
    run_validation(mode=args.mode, output_dir=args.output, synthetic_sizes=args.sizes,
                   stationary_npzs=args.stationary_npz,
                   max_stencil_radius=args.radius, correlation=args.correlation,
                   compare_era5_rectangles=args.compare_era5_rectangles)
