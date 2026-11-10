"""Fast audit of saved stationary PDFs without reading the large ERA5/A/B arrays.

Usage:
    python audit_saved_stationary.py --root "D:/Nastya/Data/OceanFull/"

Reads only existing Functional/Stationary/*.json and *.npz outputs. It checks
probability normalization, reliability bounds, probability mass and the signed
contribution to the fitted mean from outside those bounds. This helps identify
artificial far-tail modes and incompatible fit histories before recomputation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PERIODS = ((1979, 1989), (1989, 1999), (1999, 2009),
           (2009, 2019), (2019, 2026), (1979, 2026))


def integrate_between(x, y, low, high):
    """Integrate piecewise-linear y(x) between bounds, including end points."""
    low = max(float(low), float(x[0]))
    high = min(float(high), float(x[-1]))
    if low >= high:
        return 0.0
    interior = x[(x > low) & (x < high)]
    points = np.r_[low, interior, high]
    return float(np.trapezoid(np.interp(points, x, y), points))


def audit(root, names=("sensible", "latent")):
    folder = Path(root) / "Functional" / "Stationary"
    for start, end in PERIODS:
        stem = f"{'-'.join(names)}_{start}-{end}"
        jpath = folder / f"{stem}.json"
        npath = folder / f"{stem}.npz"
        print(f"\n{'=' * 72}\n{start}-{end}")
        if not jpath.is_file() or not npath.is_file():
            print(f"MISSING: {jpath if not jpath.is_file() else npath}")
            continue
        meta = json.loads(jpath.read_text(encoding="utf-8"))
        prov = meta.get("provenance", {})
        print("Method:", meta.get("method", "not recorded"))
        print("Source:", prov.get("input_type", "not recorded"))
        print("State coordinate:", prov.get("state_construction", "not recorded"))
        with np.load(npath, allow_pickle=False) as archive:
            grid = np.asarray(archive["grid"], dtype=float)
            if (grid.ndim != 1 or grid.size < 2 or not np.isfinite(grid).all()
                    or not np.all(np.diff(grid) > 0)):
                raise ValueError(f"Invalid grid in {npath}")
            for name in names:
                fit = meta.get("fits", {}).get(name, {})
                key = f"pdf_{name}"
                if key not in archive:
                    print(f"{name}: MISSING {key}")
                    continue
                p = np.asarray(archive[key], dtype=float)
                if p.shape != grid.shape or not np.isfinite(p).all() or np.any(p < 0):
                    raise ValueError(f"Invalid PDF for {name}: {npath}")
                normalization = float(np.trapezoid(p, grid))
                mean_calculated = float(np.trapezoid(grid * p, grid))
                rmse = fit.get("drift_rmse")
                fit_range = fit.get("reliable_regression_interval")
                print(f"\n{name}: PDF mean={mean_calculated:+.6f}; "
                      f"JSON mean={fit.get('mean', 'N/A')}; "
                      f"integral={normalization:.9f}; "
                      f"A RMSE={rmse if rmse is not None else 'N/A'}")
                print("  Model:", fit.get("diffusion_model", "NOT RECORDED (legacy PDF)"))
                print("  Integration:", fit.get("stationary_integration_method", "NOT RECORDED"))
                print(f"  Mode x={grid[np.argmax(p)]:+.4f}; "
                      f"PDF peak={np.max(p):.6g}")
                if fit_range is not None and len(fit_range) == 2:
                    xmin, xmax = map(float, fit_range)
                    mass_inside = integrate_between(grid, p, xmin, xmax)
                    moment_inside = integrate_between(grid, grid * p, xmin, xmax)
                    mass_outside = normalization - mass_inside
                    moment_outside = mean_calculated - moment_inside
                    conditional_mean = (moment_inside / mass_inside if mass_inside > 0
                                        else float("nan"))
                    print(f"  Fit support [{xmin:+.3f}, {xmax:+.3f}]")
                    print(f"  Mass OUTSIDE fit support: {mass_outside:.3%}")
                    print(f"  Mean contribution OUTSIDE: {moment_outside:+.6f}")
                    print(f"  Mean conditional INSIDE: {conditional_mean:+.6f}")
                    if mass_outside > 0.05:
                        print("  WARNING: >5% of PDF is outside fitted-data support")
                if rmse is None or fit.get("diffusion_model") is None:
                    print("  WARNING: saved output predates current fit/PDF metadata; "
                          "the numerical model may be stale")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="D:/Nastya/Data/OceanFull/")
    args = parser.parse_args()
    audit(args.root)


if __name__ == "__main__":
    main()
