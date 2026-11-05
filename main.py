"""Ocean numerical workflows; edit task and settings in the entry-point block."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from Coefficients.common import coerce_series, load_water_mask
from Coefficients.runner import diagnose_joint_groups, run_decade_coefficients, collect_period_coefficients
from Coefficients.bridges import run_boundary_workflow
from Coefficients.derived_workflows import run_continuous_cf
from Data_processing.io import (load_field, read_ocean_mask, start_label,
                                dates_from_labels)
from Data_processing.mean_year import save_mean_year
from Eigenvalues.workflows import run_eigenvalues_for_period, collect_spectral_values
from Functional.stationary import estimate_stationary_characteristics
from Functional.copula_workflows import estimate_copula_characteristics
from Data_processing.grib import convert_era5_grib

PERIODS = ((1979, 1989), (1989, 1999), (1999, 2009),
           (2009, 2019), (2019, 2026))


def run_selected_task(task, **settings):
    """Execute a numeric task without creating plotted figures.

    Parameters
    ----------
    task : str
        One of coefficients, bridge_coefficients, derived_cf, eigenvalues,
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

    if task == "coefficients":
        array1 = load_field(root, names[0], start_year, end_year)
        array2 = load_field(root, names[1], start_year, end_year)
        offset = start_label(start_year)
        if settings["diagnostic_only"]:
            return diagnose_joint_groups(
                array1=array1, array2=array2, mask=mask,
                out_dir=str(Path(root) / "Coefficients" /
                            f"aligned_{start_year}-{end_year}" / "group_diagnostics"),
                bins_list=(10, 12, 15, 18, 20),
                n_transitions=settings["diagnostic_days"],
                time_start=0, index_offset=offset,
                min_group_size=2, thresholds=(2, 10, 20, 30),
                plot_bins=15, make_plots=False,
            )
        # Preserve the existing decade estimator call and its numerical arguments.
        run_decade_coefficients(
            files_path_prefix=str(root), array1=array1, array2=array2, mask=mask,
            names=names, start_year=start_year, end_year=end_year,
            start_index=offset, n_bins=settings["n_bins"],
            n_components=settings["n_components"], n_init=settings["n_init"],
            dt=settings["dt"], overwrite=overwrite, resume=True,
            chunk_transitions=settings["chunk_transitions"],
            run_1d=settings["run_1d"], run_2d=settings["run_2d"],
            n_bins_2d=n_bins_2d,
        )
        return collect_period_coefficients(
            files_path_prefix=str(root), names=names,
            start_year=start_year, end_year=end_year,
            start_label=offset + 1,
            end_label=offset + coerce_series(array1).shape[0] - 1,
            n_bins_2d=n_bins_2d,
            run_1d=settings["run_1d"], run_2d=settings["run_2d"],
            methods=("np", "sp"), overwrite=overwrite,
        )
    if task == "bridge_coefficients":
        return run_boundary_workflow(root, periods, names, n_bins_2d=n_bins_2d,
                                     n_components=settings["n_components"],
                                     n_init=settings["n_init"], dt=settings["dt"],
                                     mask=mask, overwrite=overwrite)
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
            "a_pattern": f"Coefficients/aligned_{{period}}/2d_bins_{n_bins_2d}/"
                         "{left}_{right}/Nonparametric_2d/daily/A_{index}.npy",
            "b_pattern": f"Coefficients/aligned_{{period}}/2d_bins_{n_bins_2d}/"
                         "{left}_{right}/Nonparametric_2d/daily/B_{index}.npy",
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
            folder = (Path(root) / "Coefficients" / "continuous" /
                      f"2d_bins_{n_bins_2d}" / f"{names[0]}_{names[1]}" /
                      ("Nonparametric_2d" if settings["method"] == "np" else "Semiparametric_2d"))
            source_files = list(folder.glob(f"{name}_*-*.npy"))
            if len(source_files) != 1:
                raise FileNotFoundError(f"Expected one {name} continuous array: {source_files}")
            values = np.load(source_files[0], mmap_mode="r")
            first_label = int(source_files[0].stem.removeprefix(f"{name}_").split("-")[0])
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
        return estimate_stationary_characteristics(root, start_year, end_year,
                                                   names=names, b_type=settings["b_type"])
    if task == "copulas":
        return estimate_copula_characteristics(root, start_year, end_year,
                                               names=names, sample_size=settings["sample_size"],
                                               empirical_bins=settings["empirical_bins"],
                                               checkerboard_bins=settings["checkerboard_bins"])
    elif task == 'grib':
        grib_files = [Path(root) / "DATA" / "GRIB" / "data.grib",]

        ocean_mask = Path(root) / "DATA" / "mask"
        if not ocean_mask.is_file():
            ocean_mask = None

        # variables = ("sst", "sp", "avg_ishf", "avg_ilhf")
        report = convert_era5_grib(
            grib_files=grib_files,
            output_root=Path(root),
            variable= "avg_ishf",
            period_label="2019-2026",
            hours=(0, 6, 12, 18),
            spatial_stride="auto",
            mask=ocean_mask,
            flux_units="native",  # Preserve J m-2 for historic comparisons.
            output_dtype="float32",
            sum_dtype="float64",
            derive_flux=True,
            overwrite=False,
        )
    elif task == 'grib-check':
        old = np.load(Path(root) / "DATA" / "Fluxes" / "Sensible_1979-1989_grouped_old.npy")
        new = np.load(Path(root) / "DATA" / "Fluxes" / "sensible_1979-1989_grouped.npy")
        dates = np.load(Path(root) / "DATA/ERA5_metadata/ERA5_1979-1989_avg_ishf_dates.npy").astype("datetime64[D]")
        old_start = np.datetime64("2019-01-01")
        old_indices = (dates - old_start).astype("timedelta64[D]").astype(int)
        new_indices = np.flatnonzero((old_indices >= 0) & (old_indices < old.shape[1]))

        assert old.ndim == new.ndim == 2
        assert old.shape[0] == new.shape[0]
        assert len(new_indices) > 0, "No overlapping dates"

        mismatches = 0
        max_error = 0.0

        for start in range(0, old.shape[0], 256):
            stop = start + 256

            a = old[start:stop, old_indices[new_indices]]
            b = new[start:stop, new_indices]

            equal = np.isclose(a, b, rtol=1e-5, atol=1e-4, equal_nan=True)
            mismatches += np.count_nonzero(~equal)

            valid = np.isfinite(a) & np.isfinite(b)
            if valid.any():
                max_error = max(max_error, float(np.max(np.abs(a[valid] - b[valid]))))

        print("Overlap:", dates[new_indices[0]], dates[new_indices[-1]])
        print("Mismatching cells:", mismatches)
        print("Maximum absolute error:", max_error)
        print("Arrays match:", mismatches == 0)

    else:
        raise ValueError(f"Unsupported numeric task: {task}")


def main():
    """Provide editable parameters and run the explicitly selected numeric task."""
    # Root directory containing input data and calculated numerical outputs.
    files_path_prefix = "D:/Nastya/Data/OceanFull/"
    # files_path_prefix = "/home/aosipova/OceanFull/"

    task = 'grib'
    # task = "coefficients"
    start_year = 1979
    end_year = start_year + 10
    if start_year == 2019:
        end_year = 2026
    n_bins = 144
    n_components = 3
    n_init = 10
    run_1d = False
    run_2d = True
    n_bins_2d = 12
    diagnostic_only = False
    diagnostic_days = 7
    chunk_transitions = 7
    dt = 1.0
    overwrite = True
    names = ("sensible", "latent")
    method = "sp"
    periods = PERIODS
    save_counts = False
    spectral_variables = ("Flux", "SST", "Pressure")
    spectral_pair = ("Flux", "Flux")
    correlation_window = 14
    mean_width = 7
    sde_variables = ("sensible", "latent", "SST", "Pressure")
    sde_pairs = None
    window_days = 1000
    q_low, q_high = 0.03, 0.97
    epsilon = 0.1
    mean_year_source = "Variables"
    mean_year_name = "sensible"
    b_type = "simple"
    sample_size = 300000
    empirical_bins = 60
    checkerboard_bins = 50

    # Only numerical tasks that require a spatial mask load it.
    mask = (load_water_mask(read_ocean_mask(files_path_prefix))
            if task in ("coefficients", "bridge_coefficients", "derived_cf", "mean_year") else None)
    settings = locals().copy()
    settings.pop("task", None)
    return run_selected_task(task, **settings)


if __name__ == "__main__":
    main()
