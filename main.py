"""Ocean numerical workflows; based on the current GitHub task catalog.

Adds reproducible article maps/regions and retains numerical workflows.
"""

from __future__ import annotations

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
from Data_processing.mean_year import save_mean_year
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
        collect_eigenvalues, sde_conditions, mean_year, stationary, copulas.
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

    # Current scalar and bivariate quantile diagnostics (retained).
    if task == "quantile_diagnostics":
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
    if task == "quantile_diagnostics_1d":
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
    if task == "quantile_hourly":
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
            output_dir=Path(root) / "Functional" / "Diagnostics" / "quantile_hourly" /
                       f"{start_year}-{hourly_period_end}",
        )

    # Article diagnostics: the first three geographical/probability figures.
    if task in ("article_prepare", "article_figures", "article_all",
                "article_isolines", "article_regions", "article_classes"):
        opts = dict(
            regions_period=settings.get("article_regions_period", (1979, 1989)),
            map_date=settings.get("article_map_date", "1979-01-01"),
            density_source=settings.get("article_density_source", "auto"),
            hybrid_reference=settings.get("article_hybrid_reference", (1979, 2026)),
            chunk_days=settings.get("article_chunk_days", 32),
            compute_frequencies=settings.get("article_compute_frequencies", False),
        )
        if task == "article_prepare":
            from Functional.article_diagnostics import prepare_article_diagnostics
            return prepare_article_diagnostics(
                root, names, start_year, end_year,
                overwrite=settings.get("article_refresh", False), **opts)
        from Plotting.article_figures import plot_article_figures
        figures = {
            "article_isolines": ("isolines",),
            "article_regions": ("regions",),
            "article_classes": ("classes",),
        }.get(task, ("isolines", "regions", "classes"))
        return plot_article_figures(
            root, names, start_year, end_year,
            refresh=settings.get("article_refresh", False),
            language=settings.get("article_language", "en"),
            dpi=settings.get("article_dpi", 300),
            include_copulas=(task == "article_all"),
            isoline_levels=settings.get("article_isoline_levels"),
            second_anomaly_files=settings.get("article_second_anomaly_files"),
            figures=figures, **opts)

    if task == "bbt":
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

    if task == "validate_bbt":
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
    if task == "fokker_planck_2d_prepare":
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
    if task == "fokker_planck_2d_solve":
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

    if task == "coefficients":
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
    if task == "bridge_coefficients":
        return run_boundary_workflow(root, periods, names, n_bins_2d=n_bins_2d,
                                     n_components=settings["n_components"],
                                     n_init=settings["n_init"], dt=settings["dt"],
                                     mask=mask,
                                     methods=normalize_methods(settings.get("coefficient_methods", "both")),
                                     overwrite=overwrite)
    if task == "derived_cf":
        return run_continuous_cf(root, names, n_bins_2d=n_bins_2d,
                                 method=settings["method"], mask=mask,
                                 correlation_window=settings["correlation_window"],
                                 mean_width=settings["mean_width"], overwrite=overwrite)
    if task == "eigenvalues":
        return run_eigenvalues_for_period(
            files_path_prefix=str(root), start_year=start_year, end_year=end_year,
            n_bins=settings["n_bins"], n_components=settings["n_components"],
            dt=settings["dt"], overwrite=overwrite,
            save_counts=settings["save_counts"], variables=settings["spectral_variables"])
    if task == "collect_eigenvalues":
        results = {}
        from itertools import combinations_with_replacement
        for pair in combinations_with_replacement(settings["spectral_variables"], 2):
            results[pair] = collect_spectral_values(root, pair, periods,
                                                     overwrite=overwrite)
        return results
    if task == "sde_conditions":
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
    if task == "mean_year":
        source = settings["mean_year_source"]
        name = settings["mean_year_name"]
        if source == "Variables":
            values = load_field(root, name, start_year, end_year)
            timestamps = [date(start_year, 1, 1) + timedelta(days=i)
                          for i in range(len(values))]
        elif source in ("Coefficients", "Derived"):
            method = settings["method"]
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
                              years=(start_year, end_year))
    if task == "stationary":
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
    if task == "copulas":
        return estimate_copula_characteristics(root, start_year, end_year,
                                               names=names, sample_size=settings["sample_size"],
                                               empirical_bins=settings["empirical_bins"],
                                               checkerboard_bins=settings["checkerboard_bins"])
    if task == "grib":
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
    if task == "grib-check":
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
    raise ValueError(f"Unsupported numeric task: {task}")


def main():
    """Configure numerical workflows and run the selected task."""
    files_path_prefix = "D:/Nastya/Data/OceanFull/"
    # files_path_prefix = "/home/aosipova/OceanFull/"
    start_year = 19792026  # default: full period; 1979 for the first decade

    # --------------------------------------------------------------------------
    # ARTICLE: maps for Fig 1 (isoline anomalies), Fig 2 (5 regions), Fig 3
    # (geographic class map) and optional copula Figs 4--8.
    task = "article_figures"  # or article_prepare, article_all,
    # task = "article_isolines"  # only Fig 1
    # task = "article_regions"  # only Fig 2
    # task = "article_classes"  # only Fig 3
    article_regions_period = (1979, 2026)
    article_map_date = "1979-01-01"
    article_density_source = "auto"  # hybrid preferred; 'hybrid' / 'fp'
    article_hybrid_reference = (1979, 2026)
    article_chunk_days = 32
    article_compute_frequencies = False
    article_language = "ru"  # switch to 'ru' for the Russian manuscript
    article_dpi = 300
    article_refresh = False  # explicit article cache invalidation
    article_isoline_levels = None
    article_second_anomaly_files = None  # optional {'sensible': 'file.npy'}

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
    hourly_day_stride = 7
    hourly_max_days = None
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
    # COEFFICIENTS: estimate daily NP/SP drift and diffusion coefficients.
    # task = "coefficients"
    run_1d = False
    run_2d = True
    diagnostic_only = False
    diagnostic_days = 7
    chunk_transitions = 7
    parallel_workers = 8
    delete_daily_after_collect = False

    # task = "bridge_coefficients"
    # task = "bbt"
    bbt_period = "1979-2026"
    bbt_chunk_days = 32

    # task = "validate_bbt"
    bbt_validation_chunk_days = 32
    bbt_comparison_rtol = 1e-5
    bbt_comparison_atol = 1e-8
    bbt_psd_rtol = 1e-7
    bbt_psd_atol = 1e-10
    bbt_raise_on_failure = False

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

    # task = "derived_cf"
    correlation_window = 14
    mean_width = 7

    # task = "eigenvalues"
    save_counts = False
    # task = "collect_eigenvalues"

    # task = "sde_conditions"
    sde_variables = ("sensible", "latent", "SST", "Pressure")
    sde_pairs = None
    window_days = 1000
    q_low, q_high = 0.03, 0.97
    epsilon = 0.1

    # task = "mean_year"
    mean_year_source = "Variables"
    mean_year_name = "sensible"
    spectral_pair = ("Flux", "Flux")

    # task = "stationary"
    stationary_method = "sp"
    stationary_quantile_bins = 260
    stationary_trim_bins = 5
    stationary_chunk_days = 16
    stationary_spatial_block_size = 5
    stationary_sample_size = 400000
    stationary_refresh_grouped = False

    # task = "copulas"
    sample_size = 300000
    empirical_bins = 60
    checkerboard_bins = 50

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

    # task = "grib-check"
    grib_check_old = "DATA/Fluxes/Sensible_1979-1989_grouped_old.npy"
    grib_check_new = "DATA/Fluxes/sensible_1979-1989_grouped.npy"
    grib_check_dates = "DATA/ERA5_metadata/ERA5_1979-1989_avg_ishf_dates.npy"
    grib_check_old_start = "1979-01-01"
    grib_check_rtol = 1e-5
    grib_check_atol = 1e-4

    # ==========================================================================
    # SHARED SETTINGS: preserved from the current GitHub entrypoint.
    end_year = start_year + 10
    if start_year == 2019:
        end_year = 2026
    elif start_year == 19792026:
        start_year = 1979
        end_year = 2026
    names = ("sensible", "latent")
    periods = PERIODS
    coefficient_methods = "sp"
    method = "sp"
    n_bins = 144
    n_bins_2d = 12
    n_components = 3
    n_init = 10
    dt = 1.0
    overwrite = True  # the original numerical-task default; article_refresh is separate
    spectral_variables = ("Flux", "SST", "Pressure")

    mask = (load_water_mask(read_ocean_mask(files_path_prefix))
            if task in ("coefficients", "bridge_coefficients", "derived_cf", "mean_year",
                        "quantile_diagnostics", "quantile_diagnostics_1d", "quantile_hourly")
            else None)
    settings = locals().copy()
    settings.pop("task", None)
    return run_selected_task(task, **settings)


if __name__ == "__main__":
    main()
