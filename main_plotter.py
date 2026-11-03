"""Render cached ocean-analysis results without repeating numerical estimation."""

from __future__ import annotations

from Plotting.workflows import (
    plot_variables, plot_spectral_modes, plot_mean_year, coefficient_frames,
    coefficient_video, plot_stationary_distributions, plot_copula_results,
)


def main():
    """Select and run a visualization task using editable local parameters."""
    files_path_prefix = "D:/Nastya/Data/OceanFull/"
    # files_path_prefix = "/home/aosipova/OceanFull/"
    task = "variables"
    start_year = 1989
    end_year = 2019 if start_year == 2009 else (2026 if start_year == 2019 else start_year + 10)
    names = ("sensible", "latent")
    spectral_names = ("Flux", "SST")
    n_bins = 250
    n_modes = 3
    n_bins_2d = 12
    method = "np"
    coefficient = "A"
    first_day = 0
    last_day = 5
    day = 0
    mean_year_source = "Variables"
    mean_year_name = "sensible"
    fps = 5

    if task == "variables":
        return plot_variables(files_path_prefix, start_year, end_year, day=day)
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
