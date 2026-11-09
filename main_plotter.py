"""Render cached ocean-analysis results without repeating numerical estimation."""

from __future__ import annotations

from Plotting.workflows import (
    plot_spectral_modes, coefficient_video, plot_copula_results,
)
from Plotting.map_workflows import (
    plot_variables, plot_mean_year, coefficient_frames,
)
from Plotting.stationary_plots import plot_stationary_distributions
from Plotting.functional_fit_plots import plot_functional_fits


def main():
    """Configure visualization workflows and run the selected plotting task."""
    files_path_prefix = "D:/Nastya/Data/OceanFull/"
    # files_path_prefix = "/home/aosipova/OceanFull/"
    start_year = 19792026

    # --------------------------------------------------------------------------
    # VARIABLES: maps of fluxes, sea-surface temperature and pressure.
    # task = "variables"

    # --------------------------------------------------------------------------
    # EIGENVALUES: visualize saved eigenvalue and singular-value modes.
    # task = "eigenvalues"
    spectral_names = ("Flux", "SST")
    n_bins = 250
    n_modes = 3

    # --------------------------------------------------------------------------
    # MEAN YEAR: visualize the annual climatology of a selected variable.
    # task = "mean_year"
    mean_year_source = "Variables"
    mean_year_name = "sensible"

    # --------------------------------------------------------------------------
    # COEFFICIENT FRAMES: render A/B/C/F/FS maps for a selected time interval.
    # task = "coefficient_frames"
    n_bins_2d = 12

    # --------------------------------------------------------------------------
    # VIDEOS: combine previously generated coefficient frames into a video.
    # task = "videos"
    fps = 5

    # --------------------------------------------------------------------------
    # STATIONARY: draw saved stationary densities and observed histograms.
    task = "stationary"
    histogram_bins = 100
    histogram_chunk_days = 128
    refresh_histograms = False

    # --------------------------------------------------------------------------
    # COPULAS: visualize saved copula and joint-density results.
    # task = "copulas"

    # --------------------------------------------------------------------------
    # FUNCTIONAL FITS: read exactly the grouped inputs used by stationary.
    # task = "functional_fits"
    functional_method = "sp"
    functional_b_type = "simple"
    functional_quantile_bins = 260
    functional_trim_bins = 5
    functional_chunk_days = 16
    functional_spatial_block_size = 5
    functional_sample_size = 400000
    functional_refresh_grouped = False  # Normal plotting NEVER launches 2-pass regrouping.
    functional_verify_stationary = True  # Compare fitted params with saved density JSON.

    # ==========================================================================
    # SHARED SETTINGS: select both years explicitly, without implicit overrides.
    end_year = start_year + 10
    if start_year == 2019:
        end_year = 2026
    elif start_year == 19792026:
        start_year = 1979
        end_year = 2026
    names = ("sensible", "latent")
    first_day = 0
    last_day = 5
    method = "np"
    coefficient = "A"

    if task == "variables":
        return plot_variables(
            files_path_prefix, start_year, end_year,
            start=first_day, stop=last_day,
        )
    if task == "eigenvalues":
        return plot_spectral_modes(
            files_path_prefix, start_year, end_year,
            names=spectral_names, first_day=first_day,
            last_day=last_day, n_bins=n_bins, n_modes=n_modes,
        )
    if task == "mean_year":
        return plot_mean_year(
            files_path_prefix, mean_year_source, mean_year_name,
            start_year, end_year,
        )
    if task == "coefficient_frames":
        return coefficient_frames(
            files_path_prefix, names, n_bins_2d, method,
            coefficient, start=first_day, stop=last_day,
        )
    if task == "videos":
        return coefficient_video(
            files_path_prefix, names, method, coefficient, fps=fps,
        )
    if task == "stationary":
        return plot_stationary_distributions(
            files_path_prefix, names, start_year, end_year,
            histogram_bins=histogram_bins,
            histogram_chunk_days=histogram_chunk_days,
            refresh_histograms=refresh_histograms,
        )
    if task == "copulas":
        return plot_copula_results(
            files_path_prefix, names, start_year, end_year,
        )
    if task == "functional_fits":
        return plot_functional_fits(
            files_path_prefix,
            start_year,
            end_year,
            names=names,
            method=functional_method,
            b_type=functional_b_type,
            quantile_bins=functional_quantile_bins,
            trim_bins=functional_trim_bins,
            chunk_days=functional_chunk_days,
            spatial_block_size=functional_spatial_block_size,
            sample_size=functional_sample_size,
            refresh_grouped=functional_refresh_grouped,
        )
    raise ValueError(f"Unsupported plotting task: {task}")


if __name__ == "__main__":
    main()
