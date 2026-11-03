"""Calculate empirical and parametric copulas independently from rendering."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from Data_processing.io import load_field


def estimate_copula_characteristics(files_path_prefix, start_year, end_year,
                                     *, names=("sensible", "latent"),
                                     empirical_bins=60, checkerboard_bins=50,
                                     sample_size=300000, random_seed=12345,
                                     joint_grid_points=400):
    """Estimate and store all supported copulas and dependence metrics.

    Parameters
    ----------
    files_path_prefix : str or Path
        Root directory containing original fields and stationary density fits.
    start_year, end_year : int
        Requested data period.
    names : tuple of str
        Variables of the joint distribution.
    empirical_bins : int
        Binning for the empirical copula and Gaussian/Student comparisons.
    checkerboard_bins : int
        Binning for checkerboard and Bernstein constructions.
    sample_size : int
        Deterministic valid paired sample size.
    random_seed : int
        Random generator seed.
    joint_grid_points : int
        Resolution for each coordinate of joint stationary density grids.

    Returns
    -------
    pathlib.Path
        NPZ file containing copula and stationary joint density grids.
    """
    from Functional.copula_calculations import (
        estimate_kendall_dependence, build_empirical_copula_density,
        build_gaussian_copula_density_grid, fit_student_t_copula,
        build_student_t_copula_density_grid, compare_copula_densities,
        calculate_student_t_tail_dependence, build_checkerboard_copula,
        build_bernstein_copula_grid, build_gaussian_copula_stationary_density,
        build_checkerboard_stationary_density, build_bernstein_stationary_density,
        calculate_joint_density_moments, calculate_marginal_errors,
    )

    root = Path(files_path_prefix)
    fields = [load_field(root, name, start_year, end_year) for name in names]
    stationary = root / "Functional" / "Stationary" / f"{names[0]}-{names[1]}_{start_year}-{end_year}.npz"
    if not stationary.is_file():
        raise FileNotFoundError(f"Calculate stationary densities first: {stationary}")
    with np.load(stationary) as data:
        source_grid = data["grid"]
        grids = [np.linspace(source_grid[0], source_grid[-1], joint_grid_points)] * 2
        densities = [np.interp(grids[i], source_grid, data[f"pdf_{name}"])
                     for i, name in enumerate(names)]
    dependence = estimate_kendall_dependence(*fields, sample_size=sample_size, random_seed=random_seed)
    empirical = build_empirical_copula_density(dependence["sampled_x1"],
                                               dependence["sampled_x2"], bins=empirical_bins)
    centers = empirical["centers"]
    gaussian = build_gaussian_copula_density_grid(dependence["rho"], centers)
    student_fit = fit_student_t_copula(empirical["u"], empirical["v"], dependence["rho"])
    student = build_student_t_copula_density_grid(student_fit["rho"], student_fit["df"], centers)
    checkerboard = build_checkerboard_copula(empirical["u"], empirical["v"], bins=checkerboard_bins)
    bernstein = build_bernstein_copula_grid(checkerboard, grid_size=200)
    gaussian_joint = build_gaussian_copula_stationary_density(*(
        grids[0], densities[0], grids[1], densities[1], dependence["rho"]))
    checkerboard_joint = build_checkerboard_stationary_density(
        grids[0], densities[0], grids[1], densities[1], checkerboard)
    bernstein_joint = build_bernstein_stationary_density(
        grids[0], densities[0], grids[1], densities[1], checkerboard)
    payload = {"centers": centers, "empirical": empirical["density"],
               "gaussian": gaussian, "student_t": student,
               "checkerboard_centers": checkerboard["centers"],
               "checkerboard": checkerboard["density"],
               "bernstein_grid": bernstein["grid"], "bernstein": bernstein["density"],
               "x1": grids[0], "x2": grids[1],
               "joint_gaussian": gaussian_joint["joint_density"],
               "joint_checkerboard": checkerboard_joint["joint_density"],
               "joint_bernstein": bernstein_joint["joint_density"]}
    summary = {"pair": list(names), "period": f"{start_year}-{end_year}",
               "kendall_tau": float(dependence["tau"]), "gaussian_rho": float(dependence["rho"]),
               "student_t_rho": float(student_fit["rho"]),
               "student_t_df": float(student_fit["df"]),
               "tail_dependence": calculate_student_t_tail_dependence(student_fit["rho"], student_fit["df"]),
               "gaussian_distance": compare_copula_densities(empirical["density"], gaussian, centers),
               "student_t_distance": compare_copula_densities(empirical["density"], student, centers),
               "moments": {"gaussian": calculate_joint_density_moments(gaussian_joint["joint_density"], *grids),
                           "checkerboard": checkerboard_joint["moments"],
                           "bernstein": bernstein_joint["moments"]}}
    output = root / "Functional" / "Copulas"
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"{names[0]}-{names[1]}_{start_year}-{end_year}.npz"
    np.savez_compressed(target, **payload)
    target.with_suffix(".json").write_text(json.dumps(summary, indent=2,
                                                      default=lambda x: float(x) if isinstance(x, np.number) else str(x)),
                                           encoding="utf-8")
    return target
