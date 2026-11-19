"""Render cached ocean-analysis results and stationary spatial diagnostics.

Based on the latest GitHub main_plotter.py (including FP2D/hybrid tasks).
Visualization tasks use cached numerical results. No refitting is performed.
"""

from __future__ import annotations

import datetime

from project_paths import files_path_prefix as SHARED_DATA_PREFIX
from Plotting.language import set_plot_language

from Plotting.workflows import (
    plot_spectral_modes, coefficient_video,
)
from Plotting.map_workflows import (
    plot_variables, plot_mean_year, coefficient_frames,
)
from Plotting.stationary_plots import plot_stationary_distributions
from Plotting.hybrid_stationary_plots import plot_hybrid_stationary_distributions
from Plotting.hybrid_fp2d_marginal_overlay import plot_hybrid_fp2d_marginal_overlay
from Plotting.functional_fit_plots import plot_functional_fits
from Plotting.fokker_planck_2d_plots import (
    plot_fp2d_stationary_comparison,
    plot_fp2d_grid_convergence,
)
from Plotting.stationary_spatial_plots import plot_stationary_spatial
from Plotting.copula_plotting import plot_copula_results


def main():
    """Configure visualization workflows and run the selected plotting task."""
    # GLOBAL FIGURE LANGUAGE — choose one: "en" or "ru".
    # plot_language = "en"
    plot_language = "ru"
    set_plot_language(plot_language)

    # Monochrome for stationary maps/regions and individual copula contours.
    monochrome = False
    # monochrome = True  # Black-and-white versions for printing.

    files_path_prefix = SHARED_DATA_PREFIX
    start_year = 2019

    # --------------------------------------------------------------------------
    # STATIONARY SPATIAL: render cached anomalies, PDF quintiles, classes.
    # task = "stationary_spatial"   # all three plot types
    # task = "stationary_isolines"
    # task = "stationary_regions"
    # task = "stationary_classes"
    # Match the numerical settings in main.py (task="stationary_spatial").
    spatial_regions_period = (1979, 2026)
    spatial_map_date = "1979-01-01"
    spatial_density_source = "auto"
    spatial_hybrid_reference = (1979, 2026)
    spatial_chunk_days = 32
    spatial_compute_frequencies = False
    spatial_dpi = 300
    spatial_isoline_levels = None
    spatial_second_anomaly_files = None  # optional {'sensible': 'file.npy'}

    # --------------------------------------------------------------------------
    # VARIABLES: maps of fluxes, sea-surface temperature and pressure.
    # task = "variables"

    # --------------------------------------------------------------------------
    # EIGENVALUES: visualize saved eigenvalue and singular-value modes.
    task = "eigenvalues"
    spectral_names = ("Flux", "Pressure")
    n_bins = None  # Auto-detect saved spectrum size; or set the original n_bins explicitly.
    # start_year/end_year must still match the spectral calculation period.
    n_modes = 3

    # --------------------------------------------------------------------------
    # MEAN YEAR: visualize the annual climatology of a selected variable.
    # task = "mean_year"
    mean_year_source = "Variables"
    mean_year_name = "sensible"  # A/B/B_with_reg or C/F/FS for coefficient sources
    # Coefficients/Derived use shared names, method, n_bins_2d below.
    spectral_pair = ("Flux", "Flux")  # only for mean_year_source="Eigenvalues"
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
    trend_dpi = 300

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
    # task = "stationary"
    histogram_bins = 100
    histogram_chunk_days = 128
    refresh_histograms = False

    # --------------------------------------------------------------------------
    # STATIONARY HYBRID: central FP density + smoothed empirical-GPD tails.
    # New PNG/NPZ outputs; never overwrites the old stationary density.
    # task = "stationary_hybrid"
    hybrid_tail_quantile = 0.975
    hybrid_transition_quantile = 0.90
    hybrid_reference_weight = 0.70
    hybrid_reference_period = (1979, 2026)
    hybrid_sample_days = 1600
    hybrid_pixels_per_day = 160
    hybrid_refresh_samples = False
    hybrid_refresh_hybrid = False
    hybrid_show_original = True

    # --------------------------------------------------------------------------
    # COPULAS: saved rank densities, individual/combined contours, joint densities and marginals.
    # task = "copulas"
    copula_plot_dpi = 300

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
    functional_refresh_grouped = False
    functional_verify_stationary = True

    # --------------------------------------------------------------------------
    # FOKKER-PLANCK 2D COMPARISON:
    # empirical joint density, stationary FP density, differences and marginals.
    # task = "fokker_planck_2d_compare"
    fp2d_plot_method = "sp"
    fp2d_plot_bins = (80, 80)
    fp2d_relative_floor = 0.01
    fp2d_plot_dpi = 300

    # --------------------------------------------------------------------------
    # FOKKER-PLANCK 2D GRID CONVERGENCE:
    # compare already solved 60x60, 80x80 and 100x100 stationary results.
    # task = "fokker_planck_2d_convergence"
    fp2d_convergence_resolutions = (60, 80, 100)
    fp2d_convergence_common_resolution = 800

    # --------------------------------------------------------------------------
    # HYBRID vs 2D FP MARGINALS (new upstream task).
    # task = "hybrid_fp2d_marginal_overlay"
    overlay_stationary_file = None
    overlay_dpi = 300

    # ==========================================================================
    # SHARED SETTINGS: select both years explicitly, without implicit overrides.
    end_year = start_year + 10
    if start_year == 2019:
        end_year = 2026
    elif start_year == 19792026:
        start_year = 1979
        end_year = 2026

    names = ("sensible", "latent")
    # first_day = 0
    # last_day = 365
    first_day = (datetime.datetime(2025, 1, 1)-datetime.datetime(2019, 1, 1)).days
    last_day = (datetime.datetime(2026, 1, 1)-datetime.datetime(2019, 1, 1)).days
    method = "np"
    coefficient = "A"

    if task in ("stationary_spatial", "stationary_isolines",
                "stationary_regions", "stationary_classes"):
        chosen = {"stationary_isolines": ("isolines",),
                  "stationary_regions": ("regions",),
                  "stationary_classes": ("classes",)}.get(
            task, ("isolines", "regions", "classes"))
        return plot_stationary_spatial(
            files_path_prefix, names, start_year, end_year,
            regions_period=spatial_regions_period, map_date=spatial_map_date,
            density_source=spatial_density_source,
            hybrid_reference=spatial_hybrid_reference,
            chunk_days=spatial_chunk_days,
            language=plot_language, dpi=spatial_dpi,
            compute_frequencies=spatial_compute_frequencies,
            isoline_levels=spatial_isoline_levels,
            second_anomaly_files=spatial_second_anomaly_files,
            figures=chosen, monochrome=monochrome,
        )
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
    if task == "trends":
        from Plotting.trend_plots import plot_trends
        return plot_trends(files_path_prefix, start_year, end_year,
                           sources=trend_sources, variables=trend_variables,
                           coefficient_pairs=trend_coefficient_pairs, coefficients=trend_coefficients,
                           derived_pair=trend_derived_pair, derived=trend_derived,
                           spectral_pairs=trend_spectral_pairs, method=trend_method,
                           n_bins_2d=n_bins_2d, mode_extrema=trend_mode_extrema,
                           windows=trend_windows, language=plot_language, dpi=trend_dpi)
    if task == "mean_year":
        identity = {}
        if mean_year_source in ("Coefficients", "Derived"):
            identity = dict(pair=names, method=method, n_bins_2d=n_bins_2d)
        elif mean_year_source == "Eigenvalues":
            identity = dict(pair=spectral_pair)
        return plot_mean_year(
            files_path_prefix, mean_year_source, mean_year_name,
            start_year, end_year, **identity,
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
    if task == "stationary_hybrid":
        return plot_hybrid_stationary_distributions(
            files_path_prefix, names, start_year, end_year,
            histogram_bins=histogram_bins,
            histogram_chunk_days=histogram_chunk_days,
            refresh_histograms=refresh_histograms,
            tail_quantile=hybrid_tail_quantile,
            transition_quantile=hybrid_transition_quantile,
            reference_weight=hybrid_reference_weight,
            reference_period=hybrid_reference_period,
            sample_days=hybrid_sample_days,
            pixels_per_day=hybrid_pixels_per_day,
            refresh_samples=hybrid_refresh_samples,
            refresh_hybrid=hybrid_refresh_hybrid,
            show_original=hybrid_show_original,
        )
    if task == "copulas":
        return plot_copula_results(
            files_path_prefix, names, start_year, end_year,
            language=plot_language, dpi=copula_plot_dpi, monochrome=monochrome,
        )
    if task == "functional_fits":
        return plot_functional_fits(
            files_path_prefix, start_year, end_year,
            names=names, method=functional_method, b_type=functional_b_type,
            quantile_bins=functional_quantile_bins,
            trim_bins=functional_trim_bins,
            chunk_days=functional_chunk_days,
            spatial_block_size=functional_spatial_block_size,
            sample_size=functional_sample_size,
            refresh_grouped=functional_refresh_grouped,
            verify_stationary=functional_verify_stationary,
        )
    if task == "fokker_planck_2d_compare":
        return plot_fp2d_stationary_comparison(
            files_path_prefix, start_year, end_year,
            names=names, method=fp2d_plot_method, bins=fp2d_plot_bins,
            relative_floor=fp2d_relative_floor, dpi=fp2d_plot_dpi,
        )
    if task == "fokker_planck_2d_convergence":
        return plot_fp2d_grid_convergence(
            files_path_prefix, start_year, end_year,
            names=names, method=fp2d_plot_method,
            resolutions=fp2d_convergence_resolutions,
            common_resolution=fp2d_convergence_common_resolution,
            dpi=fp2d_plot_dpi,
        )
    if task == "hybrid_fp2d_marginal_overlay":
        return plot_hybrid_fp2d_marginal_overlay(
            files_path_prefix, start_year, end_year,
            names=names, method=fp2d_plot_method, bins=fp2d_plot_bins,
            stationary_file=overlay_stationary_file,
            histogram_bins=histogram_bins,
            histogram_chunk_days=histogram_chunk_days,
            refresh_histograms=refresh_histograms,
            tail_quantile=hybrid_tail_quantile,
            transition_quantile=hybrid_transition_quantile,
            reference_weight=hybrid_reference_weight,
            reference_period=hybrid_reference_period,
            sample_days=hybrid_sample_days,
            pixels_per_day=hybrid_pixels_per_day,
            refresh_samples=hybrid_refresh_samples,
            refresh_hybrid=hybrid_refresh_hybrid,
            dpi=overlay_dpi,
        )
    raise ValueError(f"Unsupported plotting task: {task}")


if __name__ == "__main__":
    main()
