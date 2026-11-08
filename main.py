"""Ocean numerical workflows; edit task and settings in the entry-point block."""

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

    if task == "coefficients":
        coefficient_methods = normalize_methods(settings.get("coefficient_methods", "both"))
        offset = start_label(start_year)
        if settings["diagnostic_only"]:
            array1 = load_field(root, names[0], start_year, end_year)
            array2 = load_field(root, names[1], start_year, end_year)
            return diagnose_joint_groups(
                array1=array1, array2=array2, mask=mask,
                out_dir=str(Path(root) / "Plots" / "Diagnostics" /
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
            names=names, b_type=settings["b_type"],
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
        collect_variable_periods(
            root,
            name="latent",
            chunk_days=32,
            overwrite=False,
        )
    else:
        raise ValueError(f"Unsupported numeric task: {task}")


def main():
    """Configure numerical workflows and run the selected task."""
    # Root directory containing input data and calculated numerical outputs.
    files_path_prefix = "D:/Nastya/Data/OceanFull/"
    # files_path_prefix = "/home/aosipova/OceanFull/"
    start_year = 19792026
    # task = 'collect'

    # --------------------------------------------------------------------------
    # COEFFICIENTS: estimate daily NP/SP drift and diffusion coefficients.
    # task = "coefficients"
    run_1d = False
    run_2d = True
    diagnostic_only = False
    diagnostic_days = 7
    chunk_transitions = 7
    parallel_workers = 8  # Use 1 to disable multiprocessing.
    delete_daily_after_collect = False

    # --------------------------------------------------------------------------
    # BRIDGE COEFFICIENTS: calculate cross-period transitions and assemble them.
    # task = "bridge_coefficients"
    # Uses periods, names, coefficient_methods, n_bins_2d, n_components,
    # n_init, dt and overwrite from the shared settings section below.

    # --------------------------------------------------------------------------
    # BBT: multiply each assembled two-dimensional diffusion matrix by its transpose.
    # task = "bbt"
    bbt_period = "1979-2026"
    bbt_chunk_days = 32

    # --------------------------------------------------------------------------
    # DERIVED CF: calculate C, F and FS from the continuous coefficient series.
    # task = "derived_cf"
    correlation_window = 14
    mean_width = 7

    # --------------------------------------------------------------------------
    # EIGENVALUES: calculate period-level eigenvalues and singular values.
    # task = "eigenvalues"
    save_counts = False

    # --------------------------------------------------------------------------
    # COLLECT EIGENVALUES: assemble spectral series across multiple periods.
    # task = "collect_eigenvalues"
    # Uses spectral_variables, periods and overwrite from shared settings.

    # --------------------------------------------------------------------------
    # SDE CONDITIONS: check conditions for the estimated stochastic model.
    # task = "sde_conditions"
    sde_variables = ("sensible", "latent", "SST", "Pressure")
    sde_pairs = None
    window_days = 1000
    q_low, q_high = 0.03, 0.97
    epsilon = 0.1

    # --------------------------------------------------------------------------
    # MEAN YEAR: compute a mean annual profile from a selected data source.
    # task = "mean_year"
    mean_year_source = "Variables"  # Variables, Coefficients, Derived, Eigenvalues
    mean_year_name = "sensible"
    spectral_pair = ("Flux", "Flux")  # Used when mean_year_source is Eigenvalues.

    # --------------------------------------------------------------------------
    # STATIONARY: fit one-dimensional stationary densities from full-period arrays.
    task = "stationary"
    stationary_method = "sp"  # "sp" or "np"; controls assembled A/B or BBT.
    stationary_quantile_bins = 260
    stationary_trim_bins = 5
    stationary_chunk_days = 16
    stationary_spatial_block_size = 5  # Use 1 to disable block averaging.
    stationary_sample_size = 400000
    stationary_refresh_grouped = False  # True forces recalculation of grouped fits.

    # --------------------------------------------------------------------------
    # COPULAS: estimate dependence and joint-distribution characteristics.
    # task = "copulas"
    sample_size = 300000
    empirical_bins = 60
    checkerboard_bins = 50

    # --------------------------------------------------------------------------
    # GRIB: convert ERA5 GRIB messages into daily arrays.
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
    # GRIB CHECK: compare the reprocessed daily array with the historical one.
    # task = "grib-check"
    grib_check_old = "DATA/Fluxes/Sensible_1979-1989_grouped_old.npy"
    grib_check_new = "DATA/Fluxes/sensible_1979-1989_grouped.npy"
    grib_check_dates = "DATA/ERA5_metadata/ERA5_1979-1989_avg_ishf_dates.npy"
    grib_check_old_start = "1979-01-01"
    grib_check_rtol = 1e-5
    grib_check_atol = 1e-4

    # ==========================================================================
    # SHARED SETTINGS: used by several numerical tasks.
    # Keep the input root and start_year at the beginning of this function.
    # Set end_year manually: 1989 for 1979-1989, or 2026 for 1979-2026.
    # The selected period is NEVER changed automatically, including stationary.
    end_year = start_year + 10
    if start_year == 2019:
        end_year = 2026
    elif start_year == 19792026:
        start_year = 1979
        end_year = 2026
    names = ("sensible", "latent")
    periods = PERIODS
    coefficient_methods = "sp"  # Effective value in the supplied script; np/sp/both.
    method = "sp"  # Used by derived_cf and mean_year, not by NP/SP estimation.
    n_bins = 144  # Coefficients and eigenvalues.
    n_bins_2d = 12  # Coefficients, bridges, CF, SDE and mean-year workflows.
    n_components = 3  # Coefficients, bridges and eigenvalues.
    n_init = 10  # Coefficients and bridges.
    dt = 1.0  # Time increment for coefficient and spectral estimators.
    overwrite = False
    spectral_variables = ("Flux", "SST", "Pressure")  # Spectral tasks.

    # Load a spatial mask only for workflows that require it.
    mask = (load_water_mask(read_ocean_mask(files_path_prefix))
            if task in ("coefficients", "bridge_coefficients", "derived_cf", "mean_year")
            else None)

    settings = locals().copy()
    settings.pop("task", None)
    return run_selected_task(task, **settings)


if __name__ == "__main__":
    main()
