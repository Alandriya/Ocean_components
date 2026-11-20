"""Ocean numerical workflows; based on the current GitHub task catalog.

Includes stationary spatial diagnostics and retains numerical workflows.
"""

from __future__ import annotations

from project_paths import files_path_prefix as SHARED_DATA_PREFIX

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from Coefficients.common import coerce_series, load_water_mask
from Coefficients.runner import diagnose_joint_groups
from Coefficients.parallel import run_parallel_period
from Coefficients.collection import normalize_methods
from Coefficients.bridges import run_boundary_workflow
from Coefficients.derived_workflows import run_continuous_cf
from Data_processing.io import (load_field, field_candidates, read_ocean_mask, start_label,
                                dates_from_labels)
from Data_processing.mean_year import save_mean_year, load_scalar_mean_year_source
from Data_processing.collect_variables import collect_variable_periods
from Eigenvalues.workflows import run_eigenvalues_for_period, collect_spectral_values
from Functional.stationary import estimate_stationary_characteristics
from Functional.copula_workflows import estimate_copula_characteristics

PERIODS = ((1979, 1989), (1989, 1999), (1999, 2009),
           (2009, 2019), (2019, 2026))


def run_selected_task(task, **settings):
    """Execute a numeric task without creating plotted figures.

    Parameters
    ----------
    task : str
        One of coefficients, bridge_coefficients, bbt, derived_cf, eigenvalues,
        collect_eigenvalues, sde_conditions, mean_year, trends, components, stationary, stationary_spatial, copulas, copula_quintiles.
    **settings : dict
        Settings matching the corresponding keyword variables in main().

    Returns
    -------
    object
        Task-specific paths, manifest, or numerical summary.
    """
    root = settings["files_path_prefix"]
    names = tuple(settings["names"])
    start_year, end_year = settings["start_year"], settings["end_year"]
    mask = settings.get("mask")
    n_bins_2d = settings["n_bins_2d"]
    overwrite = settings["overwrite"]
    periods = settings.get("periods", PERIODS)

    if task == "quantile_hourly_resampling":
        from Coefficients.quantile_resampling import run_quantile_hourly_resampling
        return run_quantile_hourly_resampling(
            files_path_prefix=root,
            hourly_files=settings.get("resample_hourly_files", settings["hourly_files"]),
            mask=mask,
            start_year=settings.get("resample_start_year", 1979),
            end_year=settings.get("resample_end_year", 1989),
            first_date=settings.get("resample_first_date"),
            last_date=settings.get("resample_last_date"),
            day_stride=settings.get("resample_day_stride", 7),
            max_days=settings.get("resample_max_days"),
            hours=settings.get("resample_hours", (4, 8, 12)),
            replicates=settings.get("resample_replicates", 3),
            spatial_steps=settings.get("resample_spatial_steps", (1, 2, 4)),
            bins_1d=settings.get("resample_bins_1d", (144,)),
            bins_2d=settings.get("resample_bins_2d", (15, 20)),
            random_seed=settings.get("resample_random_seed", 1979),
            compare_min_group_size=settings.get("resample_min_group_size", 30),
            hourly_spatial_stride=settings.get("resample_source_spatial_stride", "auto"),
            hourly_npy_start=settings.get("resample_npy_start"),
            hourly_npy_variable_names=settings.get("resample_npy_variable_names"),
            grib_cache_dir=settings.get("resample_grib_cache_dir"),
            resume=settings.get("resample_resume", True),
            progress_every=settings.get("resample_progress_every", 10),
            output_dir=Path(root) / "Functional" / "Diagnostics" /
                       "quantile_resampling" /
                       f"{settings.get('resample_start_year', 1979)}-"
                       f"{settings.get('resample_end_year', 1989)}",
        )
    elif task == "quantile_diagnostics":
        from Coefficients.quantile_diagnostics import run_quantile_diagnostics
        chosen_periods = (periods if settings.get("quantile_all_periods", True)
                          else ((start_year, end_year),))
        return run_quantile_diagnostics(
            files_path_prefix=root, names=names, periods=chosen_periods,
            mask=mask, bins=settings.get("quantile_bins", (8, 12, 15, 20)),
            thresholds=settings.get("quantile_thresholds", (2, 10, 20, 30, 50)),
            first_transition=settings.get("quantile_start", 1),
            stop_transition=settings.get("quantile_stop"),
            every=settings.get("quantile_every", 1),
            corr_every=settings.get("quantile_corr_every", 30),
            corr_bins=settings.get("quantile_corr_bins", 15),
            corr_lags=settings.get("quantile_corr_lags", (1, 4, 8)),
            out_dir=Path(root) / "Functional" / "Diagnostics" / "quantile_cells",
        )
    elif task == "quantile_diagnostics_1d":
        from Coefficients.quantile_diagnostics import run_quantile_diagnostics_1d
        chosen_periods = (periods if settings.get("quantile_1d_all_periods", True)
                          else ((start_year, end_year),))
        variable = settings.get("quantile_1d_variable", "sensible")
        return run_quantile_diagnostics_1d(
            files_path_prefix=root, variable=variable, periods=chosen_periods,
            mask=mask,
            bins=settings.get("quantile_1d_bins", (20, 50, 100, 144, 200, 250)),
            thresholds=settings.get("quantile_1d_thresholds", (2, 10, 20, 30, 50)),
            first_transition=settings.get("quantile_1d_start", 1),
            stop_transition=settings.get("quantile_1d_stop"),
            every=settings.get("quantile_1d_every", 1),
            corr_every=settings.get("quantile_1d_corr_every", 30),
            corr_bins=settings.get("quantile_1d_corr_bins", 144),
            corr_lags=settings.get("quantile_1d_corr_lags", (1, 4, 8)),
            out_dir=Path(root) / "Functional" / "Diagnostics" / "quantile_cells_1d" / variable,
        )
    elif task == "quantile_hourly":
        from Coefficients.quantile_hourly import run_hourly_quantile_diagnostics
        hourly_period_end = settings.get("hourly_end_year") or end_year
        return run_hourly_quantile_diagnostics(
            files_path_prefix=root, hourly_files=settings["hourly_files"],
            mask=mask, start_year=start_year, end_year=hourly_period_end,
            first_date=settings.get("hourly_first_date"),
            last_date=settings.get("hourly_last_date"),
            day_stride=settings.get("hourly_day_stride", 1),
            max_days=settings.get("hourly_max_days"),
            modes=settings.get("hourly_modes", (
                "daily_mean4", "hourly_4", "hourly_8", "hourly_12", "hourly_24")),
            bins_1d=settings.get("hourly_bins_1d", (20, 50, 100, 144, 200, 250)),
            bins_2d=settings.get("hourly_bins_2d", (8, 12, 15, 20)),
            thresholds=settings.get("hourly_thresholds", (2, 10, 20, 30, 50)),
            corr_every=settings.get("hourly_corr_every", 30),
            corr_bin_1d=settings.get("hourly_corr_bin_1d", 144),
            corr_bin_2d=settings.get("hourly_corr_bin_2d", 15),
            spatial_lags=settings.get("hourly_spatial_lags", (1, 4, 8)),
            temporal_lags=settings.get("hourly_temporal_lags", (1, 2, 3, 6)),
            hourly_spatial_stride=settings.get("hourly_spatial_stride", "auto"),
            hourly_npy_start=settings.get("hourly_npy_start"),
            hourly_npy_variable_names=settings.get("hourly_npy_variable_names"),
            daily_reference=settings.get("hourly_compare_daily", False),
            compare_min_group_size=settings.get("hourly_compare_min_group_size", 30),
            progress_every=settings.get("hourly_progress_every", 30),
            grib_cache_dir=settings.get("hourly_grib_cache_dir"),
            resume=settings.get("hourly_resume", True),
            output_dir=Path(root) / "Functional" / "Diagnostics" / "quantile_hourly" /
                       f"{start_year}-{hourly_period_end}",
        )
    elif task == "stationary_spatial":
        from Functional.stationary_diagnostics import prepare_stationary_diagnostics
        return prepare_stationary_diagnostics(
            root, names, start_year, end_year,
            regions_period=settings.get("spatial_regions_period", (1979, 2026)),
            map_date=settings.get("spatial_map_date", "1979-01-01"),
            density_source=settings.get("spatial_density_source", "auto"),
            hybrid_reference=settings.get("spatial_hybrid_reference", (1979, 2026)),
            chunk_days=settings.get("spatial_chunk_days", 32),
            compute_frequencies=settings.get("spatial_compute_frequencies", False),
            overwrite=settings.get("spatial_refresh", False),
        )
    elif task == "bbt":
        from Coefficients.bbt import BBT

        # Process the assembled, time-first B array in bounded disk chunks.
        period = settings.get("bbt_period", "1979-2026")
        pair = f"{names[0]}_{names[1]}"
        folder = Path(root) / "Coefficients" / period / pair
        results = {}
        for method in normalize_methods(settings.get("coefficient_methods", "both")):
            results[method] = BBT(
                folder / f"B_{period}_{method}.npy",
                chunk_days=settings.get("bbt_chunk_days", 32),
                overwrite=overwrite,
            )
        return results
    elif task == "validate_bbt":
        from Functional.bbt_diagnostics import validate_bbt_files
        period = settings.get("bbt_period", "1979-2026")
        pair = f"{names[0]}_{names[1]}"
        folder = Path(root) / "Coefficients" / period / pair
        results = {}
        for method in normalize_methods(settings.get("coefficient_methods", "both")):
            results[method] = validate_bbt_files(
                folder / f"B_{period}_{method}.npy",
                folder / f"BBT_{period}_{method}.npy",
                chunk_days=settings.get("bbt_validation_chunk_days", 32),
                comparison_rtol=settings.get("bbt_comparison_rtol", 1e-5),
                comparison_atol=settings.get("bbt_comparison_atol", 1e-8),
                psd_rtol=settings.get("bbt_psd_rtol", 1e-7),
                psd_atol=settings.get("bbt_psd_atol", 1e-10),
                raise_on_failure=settings.get("bbt_raise_on_failure", False),
            )
        return results
    elif task == "fokker_planck_2d_prepare":
        from Functional.fokker_planck_2d import prepare_phase_space_coefficients
        return prepare_phase_space_coefficients(
            root, start_year, end_year, names=names,
            method=settings.get("fp2d_method", "sp"),
            bins=settings.get("fp2d_bins", (80, 80)),
            quantile_bounds=settings.get("fp2d_quantile_bounds", (0.005, 0.995)),
            min_count=settings.get("fp2d_min_count", 100),
            chunk_days=settings.get("fp2d_chunk_days", 32),
            spatial_block_size=settings.get("fp2d_spatial_block_size", 5),
            sample_size=settings.get("fp2d_sample_size", 500000),
            random_seed=settings.get("fp2d_random_seed", 1979),
            psd_rtol=settings.get("fp2d_psd_rtol", 1e-7),
            psd_atol=settings.get("fp2d_psd_atol", 1e-10),
            overwrite=overwrite,
        )
    elif task == "fokker_planck_2d_solve":
        from Functional.fokker_planck_2d_solver import solve_stationary_fokker_planck_2d
        return solve_stationary_fokker_planck_2d(
            root, start_year, end_year, names=names,
            method=settings.get("fp2d_method", "sp"),
            bins=settings.get("fp2d_bins", (80, 80)),
            max_stencil_radius=settings.get("fp2d_solver_max_stencil_radius", 7),
            max_hole_cells=settings.get("fp2d_solver_max_hole_cells", 16),
            hole_neighbors=settings.get("fp2d_solver_hole_neighbors", 4),
            rate_tolerance=settings.get("fp2d_solver_rate_tolerance", 1e-10),
            negative_tolerance=settings.get("fp2d_solver_negative_tolerance", 1e-12),
            overwrite=overwrite,
        )
    elif task == "components":
        from Coefficients.spatial_components import run_components
        return run_components(
            root, start_year, end_year, variable=settings.get("components_variable", "sensible"),
            n_bins=settings.get("components_bins", 10), window_days=settings.get("components_window_days", 30),
            step_days=settings.get("components_step_days", 7), first_date=settings.get("components_first_date", "2019-01-01"),
            last_date=settings.get("components_last_date", "2019-03-31"), spatial_stride=settings.get("components_spatial_stride", 1),
            dt=settings.get("dt", 1.0), n_init=settings.get("components_n_init", 10),
            reg_covar=settings.get("components_reg_covar", 1e-6), min_group_size=settings.get("components_min_group_size", 100),
            min_weight=settings.get("components_min_weight", .05), min_effective_count=settings.get("components_min_effective_count", 20),
            bic_threshold=settings.get("components_bic_threshold", 10.), random_state=settings.get("components_random_state", 42),
            mask=mask, overwrite=settings.get("components_overwrite", False), progress=settings.get("components_progress", True),
        )
    elif task == "coefficients":
        coefficient_methods = normalize_methods(settings.get("coefficient_methods", "both"))
        offset = start_label(start_year)
        if settings["diagnostic_only"]:
            array1 = load_field(root, names[0], start_year, end_year)
            array2 = load_field(root, names[1], start_year, end_year)
            return diagnose_joint_groups(
                array1=array1, array2=array2, mask=mask,
                out_dir=str(Path(root) / "Functional" / "Diagnostics" /
                            f"{start_year}-{end_year}" / "group_diagnostics"),
                bins_list=(10, 12, 15, 18, 20),
                n_transitions=settings["diagnostic_days"],
                time_start=0, index_offset=offset,
                min_group_size=2, thresholds=(2, 10, 20, 30),
                plot_bins=15, make_plots=False,
            )
        source_files = []
        for name in names:
            file = next((candidate for candidate in
                         field_candidates(root, name, start_year, end_year)
                         if candidate.is_file()), None)
            if file is None:
                raise FileNotFoundError(f"Missing input NPY for variable {name}")
            source_files.append(str(file.resolve()))
        return run_parallel_period(
            files_path_prefix=root,
            source_files=source_files,
            mask=mask,
            names=names,
            start_year=start_year,
            end_year=end_year,
            start_index=offset,
            parallel_workers=settings.get("parallel_workers", 4),
            methods=coefficient_methods,
            n_bins=settings["n_bins"],
            n_bins_2d=n_bins_2d,
            n_components=settings["n_components"],
            n_init=settings["n_init"],
            chunk_transitions=settings["chunk_transitions"],
            dt=settings["dt"],
            run_1d=settings["run_1d"],
            run_2d=settings["run_2d"],
            overwrite=overwrite,
            delete_daily_after_collect=settings.get("delete_daily_after_collect", True),
        )
    elif task == "bridge_coefficients":
        return run_boundary_workflow(root, periods, names, n_bins_2d=n_bins_2d,
                                     n_components=settings["n_components"],
                                     n_init=settings["n_init"], dt=settings["dt"],
                                     mask=mask,
                                     methods=normalize_methods(settings.get("coefficient_methods", "both")),
                                     overwrite=overwrite)
    elif task == "derived_cf":
        return run_continuous_cf(root, names, n_bins_2d=n_bins_2d,
                                 method=settings["method"], mask=mask,
                                 correlation_window=settings["correlation_window"],
                                 mean_width=settings["mean_width"], overwrite=overwrite)
    elif task == "eigenvalues":
        return run_eigenvalues_for_period(
            files_path_prefix=str(root), start_year=start_year, end_year=end_year,
            n_bins=settings["n_bins"], n_components=settings["n_components"],
            dt=settings["dt"], overwrite=overwrite,
            save_counts=settings["save_counts"], variables=settings["spectral_variables"])
    elif task == "collect_eigenvalues":
        results = {}
        from itertools import combinations_with_replacement
        for pair in combinations_with_replacement(settings["spectral_variables"], 2):
            results[pair] = collect_spectral_values(root, pair, periods,
                                                     overwrite=overwrite)
        return results
    elif task == "sde_conditions":
        from Data_processing.sde_conditions import run_all

        source = {
            "a_pattern": f"Coefficients/{{period}}/2d_bins_{n_bins_2d}/"
                         "{left}_{right}/Semiparametric_2d/A_{index1}-{}.npy",
            "b_pattern": f"Coefficients/{{period}}/2d_bins_{n_bins_2d}/"
                         "{left}_{right}/Semiparametric_2d/B_{index}.npy",
            "layout": "first", "diffusion": "matrix",
        }
        field_paths = {
            "sensible": ("DATA/Fluxes/sensible_{period}_grouped.npy",
                         "DATA/Fluxes/sensible_grouped_{period}.npy"),
            "latent": ("DATA/Fluxes/latent_{period}_grouped.npy",
                       "DATA/Fluxes/latent_grouped_{period}.npy"),
            "SST": ("DATA/SST/SST_{period}_grouped.npy",
                    "DATA/SST/SST_grouped_{period}.npy"),
            "Pressure": ("DATA/Pressure/Pressure_{period}_grouped.npy",
                         "DATA/Pressure/PRESS_{period}_grouped.npy"),
        }
        return run_all(root, variables=settings["sde_variables"],
                       selected_pairs=settings["sde_pairs"], periods=periods,
                       coefficient_sources={"default": source}, field_paths=field_paths,
                       window_days=settings["window_days"], q_low=settings["q_low"],
                       q_high=settings["q_high"], epsilon=settings["epsilon"],
                       mask_policy="auto", fail_on_missing=False,
                       coefficient_index_shift=1,
                       output=Path(root) / "Conditions")
    elif task == "trends":
        from Data_processing.trends import run_trends
        return run_trends(root, start_year, end_year,
                          sources=settings["trend_sources"], variables=settings["trend_variables"],
                          coefficient_pairs=settings["trend_coefficient_pairs"], coefficients=settings["trend_coefficients"],
                          derived_pair=settings["trend_derived_pair"], derived=settings["trend_derived"],
                          spectral_pairs=settings["trend_spectral_pairs"], method=settings["trend_method"],
                          n_bins_2d=n_bins_2d, spectral_bins=settings["trend_spectral_bins"],
                          spectral_periods=settings["trend_spectral_periods"], mode_extrema=settings["trend_mode_extrema"],
                          windows=settings["trend_windows"], min_coverage=settings["trend_min_coverage"],
                          last_date=settings["trend_last_date"], mask=mask, chunk_days=settings["trend_chunk_days"],
                          overwrite=settings["trend_overwrite"], periods=periods)
    elif task == "mean_year":
        source = settings["mean_year_source"]
        name = settings["mean_year_name"]
        identity = {}
        if source == "Variables":
            values = load_field(root, name, start_year, end_year)
            timestamps = [date(start_year, 1, 1) + timedelta(days=i)
                          for i in range(len(values))]
        elif source == "Coefficients_1d":
            variable = settings["mean_year_variable"]
            method = settings["method"]
            bins = settings.get("mean_year_bins_1d", settings.get("n_bins"))
            identity = dict(variable=variable, method=method, n_bins_1d=bins)
            values, timestamps = load_scalar_mean_year_source(
                root, variable, name, start_year, end_year,
                method=method, n_bins_1d=bins,
                spatial_shape=mask.shape if mask is not None else (161, 181),
            )
        elif source in ("Coefficients", "Derived"):
            method = settings["method"]
            identity = dict(pair=names, method=method, n_bins_2d=n_bins_2d)
            # Prefer the canonical full-period A/B arrays from the manifest.
            manifest_path = (Path(root) / "Coefficients" / "continuous" /
                             f"manifest_{names[0]}_{names[1]}_{n_bins_2d}.json")
            coefficient_names = ("A", "B", "B_with_reg")
            if source == "Coefficients" and name in coefficient_names and manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                filename = Path(manifest["methods"][method][name])
                first_label = int(manifest["first_label"])
            else:
                folder = (Path(root) / "Coefficients" / "continuous" /
                          f"2d_bins_{n_bins_2d}" / f"{names[0]}_{names[1]}" /
                          ("Nonparametric_2d" if method == "np" else "Semiparametric_2d"))
                source_files = list(folder.glob(f"{name}_*-*.npy"))
                if len(source_files) != 1:
                    raise FileNotFoundError(
                        f"Expected one {name} continuous array: {source_files}"
                    )
                filename = source_files[0]
                first_label = int(filename.stem.removeprefix(f"{name}_").split("-")[0])
            values = np.load(filename, mmap_mode="r")
            timestamps = dates_from_labels(first_label, len(values), convention="end")
        elif source == "Eigenvalues":
            pair = settings["spectral_pair"]
            identity = dict(pair=pair)
            kind = "eigenvalues" if pair[0] == pair[1] else "singular_values"
            folder = Path(root) / "Eigenvalues" / f"{pair[0]}-{pair[1]}"
            paths = list(folder.glob(f"{kind}_collected_*-*.npy"))
            if len(paths) != 1:
                raise FileNotFoundError(f"Expected one collected {kind} array")
            values = np.load(paths[0], mmap_mode="r")
            metadata = json.loads(paths[0].with_suffix(".json").read_text(encoding="utf-8"))
            timestamps = [date(1979, 1, 1) + timedelta(days=label - 1)
                          for label in metadata["labels"]]
        else:
            raise ValueError(f"Unknown mean-year source: {source}")
        return save_mean_year(root, source, name, values, timestamps,
                              mask=mask if values.ndim >= 3 else None,
                              years=(start_year, end_year),
                              progress=settings.get("mean_year_progress", True), **identity)
    elif task == "stationary":
        return estimate_stationary_characteristics(
            root, start_year, end_year,
            names=names, b_type="simple",
            method=settings.get("stationary_method", "sp"),
            quantile_bins=settings.get("stationary_quantile_bins", 260),
            trim_bins=settings.get("stationary_trim_bins", 5),
            chunk_days=settings.get("stationary_chunk_days", 16),
            spatial_block_size=settings.get("stationary_spatial_block_size", 5),
            sample_size=settings.get("stationary_sample_size", 400000),
            refresh_grouped=settings.get("stationary_refresh_grouped", False),
        )
    elif task == "copulas":
        return estimate_copula_characteristics(root, start_year, end_year,
                                               names=names, sample_size=settings["sample_size"],
                                               empirical_bins=settings["empirical_bins"],
                                               checkerboard_bins=settings["checkerboard_bins"])
    elif task == "grib":
        from Data_processing.grib import convert_era5_grib
        grib_files = settings.get(
            "grib_files", (Path(root) / "DATA" / "GRIB" / "data.grib",)
        )
        ocean_mask = Path(root) / "DATA" / "mask"
        if not ocean_mask.is_file():
            ocean_mask = None
        return convert_era5_grib(
            grib_files=grib_files,
            output_root=Path(root),
            variables=settings.get("grib_variables", ("avg_ishf", "avg_slhtf")),
            period_label=f"{start_year}-{end_year}",
            hours=settings.get("grib_hours", (0, 6, 12, 18)),
            spatial_stride=settings.get("grib_spatial_stride", "auto"),
            mask=ocean_mask,
            flux_units=settings.get("grib_flux_units", "native"),
            output_dtype=settings.get("grib_output_dtype", "float32"),
            sum_dtype=settings.get("grib_sum_dtype", "float64"),
            derive_flux=settings.get("grib_derive_flux", True),
            overwrite=settings.get("grib_overwrite", False),
        )
    elif task == "grib-check":
        # Paths are relative to files_path_prefix unless explicitly absolute.
        def resolve_input(value):
            candidate = Path(value)
            return candidate if candidate.is_absolute() else Path(root) / candidate

        old = np.load(resolve_input(settings.get(
            "grib_check_old", "DATA/Fluxes/Sensible_1979-1989_grouped_old.npy"
        )), mmap_mode="r")
        new = np.load(resolve_input(settings.get(
            "grib_check_new", "DATA/Fluxes/sensible_1979-1989_grouped.npy"
        )), mmap_mode="r")
        dates = np.load(resolve_input(settings.get(
            "grib_check_dates", "DATA/ERA5_metadata/ERA5_1979-1989_avg_ishf_dates.npy"
        ))).astype("datetime64[D]")
        old_start = np.datetime64(settings.get("grib_check_old_start", "1979-01-01"))
        old_indices = (dates - old_start).astype("timedelta64[D]").astype(int)
        new_indices = np.flatnonzero((old_indices >= 0) & (old_indices < old.shape[1]))
        if old.ndim != 2 or new.ndim != 2 or old.shape[0] != new.shape[0] or not len(new_indices):
            raise ValueError("GRIB arrays are incompatible or have no overlapping dates")
        mismatch = 0
        maximum_error = 0.0
        for first in range(0, old.shape[0], 256):
            a = old[first:first + 256, old_indices[new_indices]]
            b = new[first:first + 256, new_indices]
            mismatch += np.count_nonzero(~np.isclose(
                a, b,
                rtol=settings.get("grib_check_rtol", 1e-5),
                atol=settings.get("grib_check_atol", 1e-4),
                equal_nan=True))
            valid = np.isfinite(a) & np.isfinite(b)
            if valid.any():
                maximum_error = max(maximum_error, float(np.max(np.abs(a[valid] - b[valid]))))
        return {"mismatching_cells": int(mismatch),
                "maximum_absolute_error": maximum_error}
    elif task == 'collect':
        collect_variable_periods(root, name="latent", chunk_days=32,
                                 overwrite=False)
        return None
    elif task == "extract_1d_series":
        from Forecast_1d.extract_points import DEFAULT_SUBDIR, extract

        data_root = Path(settings["files_path_prefix"])
        points_file = settings.get("extract_1d_points_file")
        output_dir = settings.get("extract_1d_output_dir")
        return extract(
            data_root=data_root,
            points_file=Path(points_file) if points_file else
            Path(__file__).resolve().parent / "Forecast_1d" / "points.json",
            destination=Path(output_dir) if output_dir else
            data_root / "DATA" / "Forecast_1d" / DEFAULT_SUBDIR,
            chunk_days=int(settings.get("extract_1d_chunk_days", 64)),
            overwrite=bool(settings.get("extract_1d_overwrite", False)),
            dry_run=bool(settings.get("extract_1d_dry_run", False)),
        )
    elif task == "copula_quintiles":
        from Functional.copula_quintiles import estimate_joint_quantile_classes
        chosen_periods = ((*periods, (start_year, end_year))
                          if settings.get("copula_quintile_all_periods", False)
                          else ((start_year, end_year),))
        outputs = [estimate_joint_quantile_classes(
            root, first, last, names=names,
            classes=settings.get("copula_quintile_classes", 5),
            hybrid_reference=settings.get("copula_quintile_hybrid_reference", (1979, 2026)),
        ) for first, last in dict.fromkeys(chosen_periods)]
        return outputs if len(outputs) > 1 else outputs[0]
    else:
        raise ValueError(f"Unsupported numeric task: {task}")


def main():
    """Configure numerical workflows and run the selected task."""
    files_path_prefix = SHARED_DATA_PREFIX
    start_year = 19792026  # default: full period; 1979 for the first decade

    # --------------------------------------------------------------------------
    # STATIONARY SPATIAL DIAGNOSTICS: anomalies, quintiles, daily classes.
    # task = "stationary_spatial"
    spatial_regions_period = (1979, 2026)
    spatial_map_date = "1979-01-01"
    spatial_density_source = "auto"  # hybrid preferred; 'hybrid' / 'fp'
    spatial_hybrid_reference = (1979, 2026)
    spatial_chunk_days = 32
    spatial_compute_frequencies = False
    spatial_refresh = False  # explicit spatial cache invalidation

    # --------------------------------------------------------------------------
    # HOURLY QUANTILE DIAGNOSTICS: no time shifts, fixed h=24 hours.
    # task = "quantile_hourly"
    hourly_files = (
        "DATA/GRIB/sensible_1979-1989_hourly.grib",
        "DATA/GRIB/latent_1979-1989_hourly.grib",
    )
    hourly_end_year = None
    hourly_first_date = None
    hourly_last_date = None
    hourly_modes = ("daily_mean4", "hourly_4", "hourly_8", "hourly_12", "hourly_24")
    hourly_bins_1d = (20, 50, 100, 144, 200, 250)
    hourly_bins_2d = (8, 12, 15, 20)
    hourly_thresholds = (2, 10, 20, 30, 50)
    hourly_day_stride = 1
    hourly_max_days = 3
    hourly_corr_every = 30
    hourly_corr_bin_1d = 144
    hourly_corr_bin_2d = 15
    hourly_spatial_lags = (1, 4, 8)
    hourly_temporal_lags = (1, 2, 3, 6)
    hourly_spatial_stride = "auto"
    hourly_npy_start = None
    hourly_npy_variable_names = None
    hourly_compare_daily = True
    hourly_compare_min_group_size = 30
    hourly_progress_every = 10
    hourly_grib_cache_dir = None
    hourly_resume = True

    # --------------------------------------------------------------------------
    # RANDOMIZED-HOUR / SPATIAL-THINNING DIAGNOSTIC (current GitHub feature).
    # task = "quantile_hourly_resampling"
    resample_hourly_files = hourly_files
    resample_start_year = 1979
    resample_end_year = 1989
    resample_first_date = None
    resample_last_date = None
    resample_day_stride = 7
    resample_max_days = None
    resample_hours = (4, 8, 12)
    resample_replicates = 3
    resample_spatial_steps = (1, 2, 4)
    resample_bins_1d = (144,)
    resample_bins_2d = (15, 20)
    resample_min_group_size = 30
    resample_random_seed = 1979
    resample_source_spatial_stride = "auto"
    resample_npy_start = None
    resample_npy_variable_names = None
    resample_grib_cache_dir = None
    resample_resume = True
    resample_progress_every = 10

    # --------------------------------------------------------------------------
    # DAILY QUANTILE DIAGNOSTICS: current 1D/2D routes retained.
    # task = "quantile_diagnostics"
    quantile_all_periods = True
    quantile_bins = (8, 12, 15, 20)
    quantile_thresholds = (2, 10, 20, 30, 50)
    quantile_start = 1
    quantile_stop = None
    quantile_every = 1
    quantile_corr_every = 30
    quantile_corr_bins = 15
    quantile_corr_lags = (1, 4, 8)

    # task = "quantile_diagnostics_1d"
    quantile_1d_variable = "sensible"
    quantile_1d_all_periods = True
    quantile_1d_bins = (20, 50, 100, 144, 200, 250)
    quantile_1d_thresholds = (2, 10, 20, 30, 50)
    quantile_1d_start = 1
    quantile_1d_stop = None
    quantile_1d_every = 1
    quantile_1d_corr_every = 30
    quantile_1d_corr_bins = 144
    quantile_1d_corr_lags = (1, 4, 8)

    # --------------------------------------------------------------------------
    # COMPONENTS: experimental scalar spatial mixtures, two components per rank.
    task = "components"
    components_variable = "sensible"
    components_bins = 10
    components_window_days = 30  # number of daily increments, not observations
    components_step_days = 7
    components_first_date = "2019-01-01"  # first observation, inclusive
    components_last_date = "2019-03-31"  # last observation, inclusive
    components_spatial_stride = 1  # 2 or 3 for spatial-thinning sensitivity
    components_n_init = 10
    components_reg_covar = 1e-6
    components_min_group_size = 100
    components_min_weight = 0.05
    components_min_effective_count = 20
    components_bic_threshold = 10.0
    components_random_state = 42
    components_overwrite = False  # resume, do not overwrite normal coefficients
    components_progress = True

    # --------------------------------------------------------------------------
    # COEFFICIENTS: estimate daily NP/SP drift and diffusion coefficients.
    # task = "coefficients"
    run_1d = False
    run_2d = True
    diagnostic_only = False
    diagnostic_days = 7
    chunk_transitions = 7
    parallel_workers = 8
    delete_daily_after_collect = False
    # --------------------------------------------------------------------------
    # task = "bridge_coefficients"
    # task = "bbt"
    bbt_period = "1979-2026"
    bbt_chunk_days = 32
    # --------------------------------------------------------------------------
    # task = "validate_bbt"
    bbt_validation_chunk_days = 32
    bbt_comparison_rtol = 1e-5
    bbt_comparison_atol = 1e-8
    bbt_psd_rtol = 1e-7
    bbt_psd_atol = 1e-10
    bbt_raise_on_failure = False
    # --------------------------------------------------------------------------
    # task = "fokker_planck_2d_prepare"
    fp2d_method = "sp"
    fp2d_bins = (80, 80)
    fp2d_min_count = 100
    fp2d_quantile_bounds = (0.005, 0.995)
    fp2d_chunk_days = 32
    fp2d_spatial_block_size = 5
    fp2d_sample_size = 500000
    fp2d_random_seed = 1979
    fp2d_psd_rtol = 1e-7
    fp2d_psd_atol = 1e-10

    # task = "fokker_planck_2d_solve"
    fp2d_solver_max_stencil_radius = 9
    fp2d_solver_max_hole_cells = 16
    fp2d_solver_hole_neighbors = 4
    fp2d_solver_rate_tolerance = 1e-10
    fp2d_solver_negative_tolerance = 1e-12
    # --------------------------------------------------------------------------
    # task = "derived_cf"
    correlation_window = 14
    mean_width = 7
    # --------------------------------------------------------------------------
    # task = "eigenvalues"
    save_counts = False
    # task = "collect_eigenvalues"
    # --------------------------------------------------------------------------
    # task = "sde_conditions"
    sde_variables = ("sensible", "latent")
    sde_pairs = None
    window_days = 1000
    q_low, q_high = 0.03, 0.97
    epsilon = 0.1
    # --------------------------------------------------------------------------
    # task = "mean_year"
    # mean_year_source = "Variables"
    # mean_year_name = "sensible"  # A/B/B_with_reg or C/F/FS for coefficient sources
    # mean_year_source = "Coefficients_1d"
    mean_year_source = "Coefficients"
    mean_year_name = "B" #A, B, B_with_reg
    # mean_year_source = "Derived"
    # mean_year_name = "C" #C, F, CF
    # Coefficients/Derived use shared names, method, n_bins_2d below.
    # mean_year_source = "Eigenvalues"
    spectral_pair = ("Flux", "Flux")  # only for mean_year_source="Eigenvalues"

    mean_year_variable = "sensible"  # only for Coefficients_1d
    mean_year_bins_1d = 144
    mean_year_progress = True
    # --------------------------------------------------------------------------
    # TRENDS: daily spatial extrema / linear regression over complete day blocks.
    # task = "trends"
    trend_sources = ("Variables", "Coefficients", "Eigenvalues")  # optional "Derived"
    trend_variables = ("Flux", "SST", "Pressure", "sensible", "latent")
    trend_coefficient_pairs = (("Flux", "SST"), ("SST", "Pressure"),
                               ("Flux", "Pressure"), ("sensible", "latent"))
    trend_coefficients = ("A", "B")
    trend_derived_pair = ("sensible", "latent")
    trend_derived = ("C", "F")
    trend_spectral_pairs = (("Flux", "Flux"), ("Flux", "SST"), ("Flux", "Pressure"),
                            ("SST", "SST"), ("SST", "Pressure"), ("Pressure", "Pressure"))
    trend_method = "sp"
    trend_windows = (30, 365)
    trend_mode_extrema = True  # False: only leading eigenvalues / singular values
    trend_spectral_bins = 144  # None to infer per spectrum; do not mix quantile schemes
    trend_spectral_periods = PERIODS  # MUST match the original spectral quantile periods
    # trend_spectral_periods = ((1979, 2026),)  # if calculated using full-period quantiles
    trend_min_coverage = 0.9  # fraction of valid days required per complete 30/365-day block
    trend_last_date = None  # e.g. "2026-01-10"; use actual available dates by default
    trend_chunk_days = 16
    trend_overwrite = True
    # --------------------------------------------------------------------------
    # task = "stationary"
    stationary_method = "sp"
    stationary_quantile_bins = 260
    stationary_trim_bins = 5
    stationary_chunk_days = 16
    stationary_spatial_block_size = 5
    stationary_sample_size = 400000
    stationary_refresh_grouped = False
    # --------------------------------------------------------------------------
    # task = "copulas"
    sample_size = 300000
    empirical_bins = 60
    checkerboard_bins = 50

    # task = "copula_quintiles"  # use cached copula NPZ/JSON, no refitting
    copula_quintile_classes = 5
    copula_quintile_all_periods = False
    copula_quintile_hybrid_reference = (1979, 2026)

    # --------------------------------------------------------------------------
    # task = "grib"
    grib_files = (Path(files_path_prefix) / "DATA" / "GRIB" / "data.grib",)
    grib_variables = ("avg_ishf", "avg_slhtf")
    grib_hours = (0, 6, 12, 18)
    grib_spatial_stride = "auto"
    grib_flux_units = "native"
    grib_output_dtype = "float32"
    grib_sum_dtype = "float64"
    grib_derive_flux = True
    grib_overwrite = False
    # --------------------------------------------------------------------------
    # task = "grib-check"
    grib_check_old = "DATA/Fluxes/Sensible_1979-1989_grouped_old.npy"
    grib_check_new = "DATA/Fluxes/sensible_1979-1989_grouped.npy"
    grib_check_dates = "DATA/ERA5_metadata/ERA5_1979-1989_avg_ishf_dates.npy"
    grib_check_old_start = "1979-01-01"
    grib_check_rtol = 1e-5
    grib_check_atol = 1e-4
    # --------------------------------------------------------------------------
    # LOCAL 1D SERIES EXTRACTION: save offshore flux time series and patches.
    # task = "extract_1d_series"
    extract_1d_points_file = None  # Use Forecast_1d/points.json by default.
    extract_1d_output_dir = None  # Use DATA_ROOT/DATA/Forecast_1d/ by default.
    extract_1d_chunk_days = 64
    extract_1d_overwrite = False  # Never replace an existing dataset by default.
    extract_1d_dry_run = False  # Set True to validate inputs without writing.

    # ==========================================================================
    # SHARED SETTINGS: preserved from the current GitHub entrypoint.
    end_year = start_year + 10
    if start_year == 2019:
        end_year = 2026
    elif start_year == 19792026:
        start_year = 1979
        end_year = 2026
    # names = ("sensible", "latent")
    names = ("Flux", "SST")
    periods = PERIODS
    coefficient_methods = "sp"
    method = "sp"
    n_bins = 144
    n_bins_2d = 12
    n_components = 3
    n_init = 10
    dt = 1.0
    overwrite = True  # the original numerical-task default; spatial_refresh is separate
    spectral_variables = ("Flux", "SST", "Pressure")

    mask = (load_water_mask(read_ocean_mask(files_path_prefix))
            if task in ("coefficients", "bridge_coefficients", "derived_cf", "mean_year", "trends", "components",
                        "quantile_diagnostics", "quantile_diagnostics_1d", "quantile_hourly",
                        "quantile_hourly_resampling")
            else None)
    settings = locals().copy()
    settings.pop("task", None)
    return run_selected_task(task, **settings)


if __name__ == "__main__":
    main()
