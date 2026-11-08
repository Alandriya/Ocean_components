"""Render cached ocean-analysis results without repeating numerical estimation."""

from __future__ import annotations

from Plotting.workflows import (
    plot_spectral_modes, coefficient_video, plot_copula_results,
)
from Plotting.map_workflows import (
    plot_variables, plot_mean_year, coefficient_frames,
)
from Plotting.stationary_plots import plot_stationary_distributions


def main():
    """Configure visualization workflows and run the selected plotting task."""
    files_path_prefix = "D:/Nastya/Data/OceanFull/"
    # files_path_prefix = "/home/aosipova/OceanFull/"
    start_year = 1989

    # --------------------------------------------------------------------------
    # VARIABLES: maps of fluxes, sea-surface temperature and pressure.
    task = "variables"
    # Uses first_day and last_day from the shared section below.

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
    # STATIONARY: draw individual stationary distributions for the fluxes.
    # task = "stationary"
    # Uses names and the selected period from shared settings.

    # --------------------------------------------------------------------------
    # COPULAS: visualize saved copula and joint-density results.
    # task = "copulas"
    # Uses names and the selected period from shared settings.

    # ==========================================================================
    # SHARED SETTINGS: used by multiple plotting tasks.
    # Set end_year manually: 1989 for 1979-1989, or 2026 for 1979-2026.
    # This period is used unchanged for stationary plots and other tasks.
    end_year = 1999
    names = ("sensible", "latent")
    first_day = 0  # First frame or observation (inclusive).
    last_day = 5  # Last frame or observation (exclusive).
    method = "np"  # Used by coefficient_frames and videos.
    coefficient = "A"  # Used by coefficient_frames and videos.

    if task == "variables":
        return plot_variables(files_path_prefix, start_year, end_year,
                              start=first_day, stop=last_day)
    if task == "eigenvalues":
        return plot_spectral_modes(files_path_prefix, start_year, end_year,
                                   names=spectral_names, first_day=first_day,
                                   last_day=last_day, n_bins=n_bins, n_modes=n_modes)
    if task == "mean_year":
        return plot_mean_year(files_path_prefix, mean_year_source, mean_year_name,
                              start_year, end_year)
    if task == "coefficient_frames":
        return coefficient_frames(files_path_prefix, names, n_bins_2d, method,
                                  coefficient, start=first_day, stop=last_day)
    if task == "videos":
        return coefficient_video(files_path_prefix, names, method, coefficient, fps=fps)
    if task == "stationary":
        return plot_stationary_distributions(files_path_prefix, names, start_year, end_year)
    if task == "copulas":
        return plot_copula_results(files_path_prefix, names, start_year, end_year)
    raise ValueError(f"Unsupported plotting task: {task}")


if __name__ == "__main__":
    main()
