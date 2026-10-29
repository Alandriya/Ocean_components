"""Actively maintained calculation entry points.

Historical one-off recipes are preserved verbatim in main_parts_legacy.py;
that file is reference-only and is not imported/executed by main.py.
"""
from __future__ import annotations

from datetime import date, timedelta
from itertools import combinations_with_replacement
from pathlib import Path
from struct import unpack

import numpy as np

from Eigenvalues.eigenvalues import count_eigenvalues_all

SPATIAL_SHAPE = (161, 181)
FIELD_PATHS = {
    "sensible": ("DATA/Fluxes/sensible_grouped_{period}.npy", "Fluxes/sensible_grouped_{period}.npy"),
    "latent": ("DATA/Fluxes/latent_grouped_{period}.npy", "Fluxes/latent_grouped_{period}.npy"),
    "SST": ("DATA/SST/SST_{period}_grouped.npy", "SST/SST_{period}_grouped.npy"),
    "Pressure": ("DATA/Pressure/PRESS_{period}_grouped.npy", "Pressure/PRESS_{period}_grouped.npy"),
}


def read_ocean_mask(files_path_prefix, spatial_shape=SPATIAL_SHAPE):
    """Read the original binary mask; accept historical root and DATA locations."""
    root = Path(files_path_prefix)
    mask_file = next((p for p in (root / "DATA" / "mask", root / "mask") if p.is_file()), None)
    if mask_file is None:
        raise FileNotFoundError(f"Ocean mask missing under {root}")
    raw = mask_file.read_bytes()
    size = int(np.prod(spatial_shape))
    if len(raw) < size:
        raise ValueError(f"Mask file too short: {mask_file}")
    return np.frombuffer(raw[:size], dtype=np.bool_).copy().reshape(spatial_shape)


def _load_field(root, name, period, spatial_shape):
    options = [root / pattern.format(period=period) for pattern in FIELD_PATHS[name]]
    path = next((p for p in options if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Missing {name} for {period}. Checked: {options}")
    field = np.load(path, mmap_mode="r")
    n_space = int(np.prod(spatial_shape))
    if field.ndim != 2:
        raise ValueError(f"{path}: expected two dimensions, got {field.shape}")
    if field.shape[0] == n_space:
        return field
    if field.shape[1] == n_space:
        return field.T
    raise ValueError(f"{path}: shape {field.shape} cannot match spatial grid {spatial_shape}")


def run_eigenvalues_for_period(files_path_prefix, start_year=2019, end_year=None,
                               n_bins=100, n_components=3, dt=1.0,
                               overwrite=True, save_counts=False,
                               spatial_shape=SPATIAL_SHAPE):
    """Recompute all ten unordered field pairs for a complete input file period.

    Default period is 2019-2026, i.e. the latest data-file *label* in the
    original project; the actual number of timesteps is read from the files.
    For earlier blocks call (1979,1989), (1989,1999), (1999,2009),
    (2009,2019). Never infer coverage of the final calendar year from name.
    """
    if end_year is None:
        end_year = 2026 if start_year == 2019 else start_year + 10
    period = f"{start_year}-{end_year}"
    root = Path(files_path_prefix)
    arrays = {name: _load_field(root, name, period, spatial_shape) for name in FIELD_PATHS}
    lengths = {name: arr.shape[1] for name, arr in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Inconsistent timeline lengths for {period}: {lengths}. Align dates first.")
    days = next(iter(lengths.values()))
    if days < 2:
        raise ValueError("At least two consecutive daily observations are needed")
    # offset is one-based for compatibility with legacy eigenvalues_{day}.npy.
    offset = (date(start_year, 1, 1) - date(1979, 1, 1)).days + 1
    last_date = date(start_year, 1, 1) + timedelta(days=days - 1)
    print(f"Calculating {period}: {days} days, {date(start_year, 1, 1)}..{last_date}, "
          f"10 pairs, output indices {offset}..{offset + days - 2}", flush=True)
    count_eigenvalues_all(str(root), arrays, spatial_shape, offset,
                          n_bins=n_bins, n_components=n_components, dt=dt,
                          overwrite=overwrite, save_counts=save_counts)
    return {"period": period, "days": days, "pairs": list(combinations_with_replacement(arrays, 2)),
            "first_index": offset, "last_index": offset + days - 2}


def run_historical_eigenvalues(files_path_prefix, **kwargs):
    """Explicit opt-in: run all four earlier ten-year files."""
    return [run_eigenvalues_for_period(files_path_prefix, year, year + 10, **kwargs)
            for year in (1979, 1989, 1999, 2009)]


def run_distribution_workflow(files_path_prefix, start_year=2019, end_year=2026,
                              b_type="simple", include_isolines=False,
                              include_copulas=False, copula_bins=(50,)):
    """Estimate stationary distributions; optionally run original isoline/copula plots.

    This is the live part moved from the old main.py, with the unconditional
    debugging raise removed and correctly normalized integration quadrature.
    Original commented experiments are archived under legacy_scripts/.
    """
    from Functional.func_estimation import I, log_b2, trapezoid, count_constants, get_isolines
    from Plotting.plot_func_estimations import plot_ab_functional_2d, plot_areas, plot_areas_map
    from Functional.copula_calculations import (
        estimate_kendall_dependence, build_gaussian_copula_stationary_density,
        calculate_copula_marginals, calculate_joint_density_moments,
        calculate_marginal_errors, build_empirical_2d_density,
        build_empirical_copula_density, build_gaussian_copula_density_grid,
        compare_copula_densities, fit_student_t_copula,
        build_student_t_copula_density_grid, calculate_student_t_tail_dependence,
        build_checkerboard_copula, build_checkerboard_stationary_density,
        build_bernstein_copula_grid, build_bernstein_stationary_density,
    )
    from Plotting.copula_plotting import (
        plot_copula_marginals, plot_joint_density, plot_joint_contours,
        plot_empirical_copula_comparison, plot_empirical_copula_contours,
        plot_copula_space_comparison, plot_copula_space_contours,
        plot_copula_density, plot_copula_models_contours,
        plot_checkerboard_copula, plot_checkerboard_stationary_density,
        plot_checkerboard_marginals, plot_empirical_checkerboard_contours,
        plot_bernstein_copula, plot_bernstein_stationary_density,
        plot_bernstein_marginals, plot_checkerboard_bernstein_comparison,
    )
    height, width = 161, 181
    data1_name, data2_name = "sensible", "latent"
    str_types = f"{data1_name}-{data2_name}"
    coef_start = (date(start_year, 1, 1) - date(1979, 1, 1)).days + 1
    known_ends = {(1979, 1989): 3653, (1989, 1999): 7305,
                  (1999, 2009): 10958, (2009, 2019): 14610,
                  (2019, 2026): 17106}
    if (start_year, end_year) not in known_ends:
        raise ValueError("Coefficient file indexes for this period need configuration")
    coef_end = known_ends[start_year, end_year]
    mask = read_ocean_mask(files_path_prefix)
    files_path_prefix = str(Path(files_path_prefix)) + "/"

    data1_array = np.load(files_path_prefix + f'DATA/Fluxes/sensible_grouped_{start_year}-{end_year}.npy')
    data1_array = data1_array.transpose().reshape((-1, height, width))
    data2_array = np.load(files_path_prefix + f'DATA/Fluxes/latent_grouped_{start_year}-{end_year}.npy')
    data2_array = data2_array.transpose().reshape((-1, height, width))
    quantiles1 = np.load(files_path_prefix + f'Functional/{str_types}/{coef_start}-{coef_end}_quantiles1.npy')
    quantiles2 = np.load(files_path_prefix + f'Functional/{str_types}/{coef_start}-{coef_end}_quantiles2.npy')
    a_grouped = np.load(files_path_prefix + f'Functional/{str_types}/{coef_start}-{coef_end}_a_grouped.npy')
    b_grouped = np.load(files_path_prefix + f'Functional/{str_types}/{coef_start}-{coef_end}_b_grouped_log_{b_type}.npy')
    data = [quantiles1, quantiles2, a_grouped, b_grouped]
    coefs = plot_ab_functional_2d(files_path_prefix, data, data1_name, data2_name, b_type, start_year)
    sensible_params, latent_params = coefs
    def prob_stationary(x):
        if x > x_max:
            return np.exp(I(x_max, args) - log_b2(x, args)) / z
        if x < x_min:
            return np.exp(I(x_min, args) - log_b2(x, args)) / z
        return np.exp(I(x, args) - log_b2(x, args)) / z
    x0, x1, k, x_min, x_max, c1, c2, c3, c4, z = sensible_params
    dL, dC, dR, prefL, prefC, prefR, const1, const2, const3, const4 = count_constants(sensible_params)
    args = [c1, c2, c3, c4, x0, x1, k, const1, const2, const3, const4]
    z = 1.0
    z = trapezoid([prob_stationary(x) for x in np.linspace(-2000, 1500, 2500)], x=np.linspace(-2000, 1500, 2500))
    y1 = [prob_stationary(x) for x in np.linspace(-500, 500, 2500)]
    p1_stationary = [prob_stationary(x) for x in np.linspace(-2000, 1500, 2500)]
    x = np.linspace(-2000, 1500, 2500)
    x0, x1, k, x_min, x_max, c1, c2, c3, c4, z = latent_params
    if b_type == 'eigen' and start_year == 1999:
        x_max = -1
    dL, dC, dR, prefL, prefC, prefR, const1, const2, const3, const4 = count_constants(latent_params)
    args = [c1, c2, c3, c4, x0, x1, k, const1, const2, const3, const4]
    z = 1.0
    z = trapezoid([prob_stationary(x) for x in np.linspace(-2000, 1500, 2500)], x=np.linspace(-2000, 1500, 2500))
    y1 = [prob_stationary(x) for x in np.linspace(-800, 300, 2500)]
    p2_stationary = [prob_stationary(x) for x in np.linspace(-2000, 1500, 2500)]
    if not include_isolines and (not include_copulas):
        return {'grid': np.linspace(-2000, 1500, 2500), 'sensible_pdf': np.asarray(p1_stationary), 'latent_pdf': np.asarray(p2_stationary), 'sensible_params': sensible_params, 'latent_params': latent_params}
    amount = 5
    colors_list = ['yellow', 'orange', 'red', 'violet', 'blue']
    x0, x1, k, x_min, x_max, c1, c2, c3, c4, z = sensible_params
    dL, dC, dR, prefL, prefC, prefR, const1, const2, const3, const4 = count_constants(sensible_params)
    args = [c1, c2, c3, c4, x0, x1, k, const1, const2, const3, const4]
    z = 1.0
    z = trapezoid([prob_stationary(x) for x in np.linspace(-2000, 1500, 2500)], x=np.linspace(-2000, 1500, 2500))
    isolines = get_isolines(prob_stationary, np.linspace(-500, 500, 2500), amount)
    print(isolines)
    isolines_list = [(isolines[i], isolines[i + 1]) for i in range(amount)]
    plot_areas(files_path_prefix, 'sensible', prob_stationary, np.linspace(-500, 500, 2500), isolines_list, colors_list)
    plot_areas_map(files_path_prefix, data1_name, isolines_list, data1_array, 0, 10, mask, 0, [1, 2, 3, 4, 5], colors_list)
    x0, x1, k, x_min, x_max, c1, c2, c3, c4, z = latent_params
    if b_type == 'eigen' and start_year == 1999:
        x_max = -1
    dL, dC, dR, prefL, prefC, prefR, const1, const2, const3, const4 = count_constants(latent_params)
    args = [c1, c2, c3, c4, x0, x1, k, const1, const2, const3, const4]
    z = 1.0
    z = trapezoid([prob_stationary(x) for x in np.linspace(-2000, 1500, 2500)], x=np.linspace(-2000, 1500, 2500))
    isolines = get_isolines(prob_stationary, np.linspace(-800, 300, 2500), amount)
    print(isolines)
    isolines_list = [(isolines[i], isolines[i + 1]) for i in range(amount)]
    plot_areas(files_path_prefix, 'latent', prob_stationary, np.linspace(-800, 300, 2500), isolines_list, colors_list)
    plot_areas_map(files_path_prefix, data2_name, isolines_list, data2_array, 0, 10, mask, 0, [1, 2, 3, 4, 5], colors_list)
    if not include_copulas:
        return {'grid': np.linspace(-2000, 1500, 2500), 'sensible_pdf': np.asarray(p1_stationary), 'latent_pdf': np.asarray(p2_stationary), 'sensible_params': sensible_params, 'latent_params': latent_params}
    x1_stationary = np.linspace(-2000, 1500, 2500)
    x2_stationary = np.linspace(-2000, 1500, 2500)
    bernstein_results_by_m = {}
    for m in copula_bins:
        bernstein_results_by_m[m] = {}
        print(f'm = {m}')
        dependence = estimate_kendall_dependence(data1_array, data2_array, sample_size=300000, random_seed=12345)
        rho = dependence['rho']
        copula_result = build_gaussian_copula_stationary_density(x1_stationary, p1_stationary, x2_stationary, p2_stationary, rho)
        joint_density = copula_result['joint_density']
        marginal_x1, marginal_x2 = calculate_copula_marginals(joint_density, x1_stationary, x2_stationary)
        moments = calculate_joint_density_moments(joint_density, x1_stationary, x2_stationary)
        error_x1, error_x2 = calculate_marginal_errors(joint_density, x1_stationary, p1_stationary, x2_stationary, p2_stationary)
        print()
        print()
        print(f"Mean X1: {moments['mean_x1']}")
        print(f"Mean X2: {moments['mean_x2']}")
        print(f"SD X1: {moments['sd_x1']}")
        print(f"SD X2: {moments['sd_x2']}")
        print(f"Covariance: {moments['covariance']}")
        print(f"Pearson correlation: {moments['correlation']}")
        plot_copula_marginals(files_path_prefix, x1_stationary, copula_result['p1'], marginal_x1, x2_stationary, copula_result['p2'], marginal_x2)
        plot_joint_density(files_path_prefix, x1_stationary, x2_stationary, joint_density, title='Gaussian copula stationary density', xlim=(-60, 150), ylim=(-250, 50))
        plot_joint_contours(files_path_prefix, x1_stationary, x2_stationary, joint_density, levels=15, title='Gaussian copula stationary density', xlim=(-60, 150), ylim=(-250, 50))
        empirical_2d_density = build_empirical_2d_density(data1_array, data2_array, x1_stationary, x2_stationary)
        plot_empirical_copula_comparison(files_path_prefix, x1_stationary, x2_stationary, empirical_2d_density, joint_density, xlim=(-60, 150), ylim=(-250, 50), same_scale=True)
        plot_empirical_copula_contours(files_path_prefix, x1_stationary, x2_stationary, empirical_2d_density, joint_density, levels=12, xlim=(-60, 150), ylim=(-250, 50))
        empirical_copula = build_empirical_copula_density(dependence['sampled_x1'], dependence['sampled_x2'], bins=60)
        gaussian_copula = build_gaussian_copula_density_grid(dependence['rho'], empirical_copula['centers'])
        copula_distances = compare_copula_densities(empirical_copula['density'], gaussian_copula, empirical_copula['centers'])
        plot_copula_space_comparison(files_path_prefix, empirical_copula['centers'], empirical_copula['density'], gaussian_copula)
        plot_copula_space_contours(files_path_prefix, empirical_copula['centers'], empirical_copula['density'], gaussian_copula, levels=12)
        student_fit = fit_student_t_copula(empirical_copula['u'], empirical_copula['v'], dependence['rho'])
        student_copula = build_student_t_copula_density_grid(student_fit['rho'], student_fit['df'], empirical_copula['centers'])
        student_distances = compare_copula_densities(empirical_copula['density'], student_copula, empirical_copula['centers'])
        tail_dependence = calculate_student_t_tail_dependence(student_fit['rho'], student_fit['df'])
        plot_copula_density(files_path_prefix, empirical_copula['centers'], empirical_copula['density'], 'copula_empirical', 'Empirical copula')
        plot_copula_density(files_path_prefix, empirical_copula['centers'], gaussian_copula, 'copula_gaussian', 'Gaussian copula')
        plot_copula_density(files_path_prefix, empirical_copula['centers'], student_copula, 'copula_student_t', 'Student t copula')
        plot_copula_models_contours(files_path_prefix, empirical_copula['centers'], empirical_copula['density'], gaussian_copula, student_copula)
        checkerboard = build_checkerboard_copula(empirical_copula['u'], empirical_copula['v'], bins=m)
        checkerboard_result = build_checkerboard_stationary_density(x1_stationary, p1_stationary, x2_stationary, p2_stationary, checkerboard)
        checkerboard_density = checkerboard_result['joint_density']
        checkerboard_moments = checkerboard_result['moments']
        print()
        print(f"Checkerboard mean X1: {checkerboard_moments['mean_x1']}")
        print(f"Checkerboard mean X2: {checkerboard_moments['mean_x2']}")
        print(f"Checkerboard SD X1: {checkerboard_moments['sd_x1']}")
        print(f"Checkerboard SD X2: {checkerboard_moments['sd_x2']}")
        print(f"Checkerboard covariance: {checkerboard_moments['covariance']}")
        print(f"Checkerboard Pearson correlation: {checkerboard_moments['correlation']}")
        print(f"Checkerboard L1 marginal error X1: {checkerboard_result['error_x1']}")
        print(f"Checkerboard L1 marginal error X2: {checkerboard_result['error_x2']}")
        bernstein_results_by_m[m]['pearson_correlation'] = checkerboard_moments['correlation']
        bernstein_results_by_m[m]['covariance'] = checkerboard_moments['covariance']
        bernstein_results_by_m[m]['l1_x1'] = checkerboard_result['error_x1']
        bernstein_results_by_m[m]['l1_x2'] = checkerboard_result['error_x2']
        plot_checkerboard_copula(files_path_prefix, checkerboard['centers'], checkerboard['density'])
        plot_checkerboard_stationary_density(files_path_prefix, x1_stationary, x2_stationary, checkerboard_density, xlim=(-50, 150), ylim=(-250, 50))
        plot_checkerboard_marginals(files_path_prefix, x1_stationary, checkerboard_result['p1'], checkerboard_result['marginal_x1'], x2_stationary, checkerboard_result['p2'], checkerboard_result['marginal_x2'])
        empirical_2d_density = build_empirical_2d_density(data1_array, data2_array, x1_stationary, x2_stationary)
        plot_empirical_checkerboard_contours(files_path_prefix, x1_stationary, x2_stationary, empirical_2d_density, checkerboard_density, levels=10, xlim=(-50, 150), ylim=(-250, 50))
        bernstein = build_bernstein_copula_grid(checkerboard, grid_size=200)
        bernstein_result = build_bernstein_stationary_density(x1_stationary, p1_stationary, x2_stationary, p2_stationary, checkerboard)
        bernstein_density = bernstein_result['joint_density']
        bernstein_moments = bernstein_result['moments']
        print()
        print(f"Bernstein mean X1: {bernstein_moments['mean_x1']}")
        print(f"Bernstein mean X2: {bernstein_moments['mean_x2']}")
        print(f"Bernstein SD X1: {bernstein_moments['sd_x1']}")
        print(f"Bernstein SD X2: {bernstein_moments['sd_x2']}")
        print(f"Bernstein covariance: {bernstein_moments['covariance']}")
        print(f"Bernstein Pearson correlation: {bernstein_moments['correlation']}")
        print(f"Bernstein L1 marginal error X1: {bernstein_result['error_x1']}")
        print(f"Bernstein L1 marginal error X2: {bernstein_result['error_x2']}")
        plot_bernstein_copula(files_path_prefix, bernstein['grid'], bernstein['density'])
        plot_bernstein_stationary_density(files_path_prefix, x1_stationary, x2_stationary, bernstein_density, xlim=(-50, 150), ylim=(-250, 50))
        plot_bernstein_marginals(files_path_prefix, x1_stationary, bernstein_result['p1'], bernstein_result['marginal_x1'], x2_stationary, bernstein_result['p2'], bernstein_result['marginal_x2'])
        plot_checkerboard_bernstein_comparison(files_path_prefix, checkerboard, bernstein)
        print('--------------------------------------------------------------')
    m_values = sorted(bernstein_results_by_m.keys())
    corr_values = [bernstein_results_by_m[m]['pearson_correlation'] for m in m_values]
    cov_values = [bernstein_results_by_m[m]['covariance'] for m in m_values]
    l1_x1_values = [bernstein_results_by_m[m]['l1_x1'] for m in m_values]
    l1_x2_values = [bernstein_results_by_m[m]['l1_x2'] for m in m_values]
    print(f'm values: {m_values}')
    print(f'correlations: {corr_values}')
    print(f'covariances: {cov_values}')
    print(f'L1 X1: {l1_x1_values}')
    print(f'L1 X2: {l1_x2_values}')
