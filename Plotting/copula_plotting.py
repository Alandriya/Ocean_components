"""Copula density, contour and marginal plots from cached numerical results."""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from Plotting.language import with_plot_language

NAMES_EN = {"sensible": "Sensible heat flux", "latent": "Latent heat flux",
            "Flux": "Total heat flux", "SST": "Sea surface temperature",
            "Pressure": "Atmospheric pressure", "A": "Coefficient A",
            "B": "Coefficient B", "C": "Coefficient C",
            "F": "Coefficient F", "FS": "Coefficient FS"}


@with_plot_language
def plot_copula_results(files_path_prefix, names, start_year, end_year, *, language="en", dpi=300,
                        monochrome=False):
    """Save every copula figure independently, using cached results.

    Existing copula/FP NPZ files are read-only; all statistical calculations
    have already been performed. The legacy YlOrRd palette is retained; all
    joint densities use the same physical display ranges. Russian alternatives
    are commented at each label.

    Returns a list of independent image paths under
    Plots/Copulas/Figures/<pair>_<period>/, with language-specific filenames.
    This intentionally does not create composite / subplot-grid figures.
    monochrome=True applies to the four individual contours only; density,
    combined-contour and marginal plots retain their existing styles.
    """
    root = Path(files_path_prefix)
    pair = f"{names[0]}-{names[1]}"
    period = f"{start_year}-{end_year}"
    source = root / "Functional" / "Copulas" / f"{pair}_{period}.npz"
    if not source.is_file():
        raise FileNotFoundError(
            f"Saved copula models not found: {source}; run task='copulas' first")

    # Filenames follow the individual scientific-figure examples in Figures(1).zip.
    folder = root / "Plots" / "Copulas" / "Figures" / f"{pair}_{period}"
    folder.mkdir(parents=True, exist_ok=True)
    outputs = []
    suffix = "eng" if language == "en" else "ru"
    cmap = "YlOrRd"  # Keep the original yellow/orange/red palette.
    display_ranges = {"sensible": (-100.0, 100.0), "latent": (-250.0, 50.0)}
    rank_models = (
        ("empirical", "centers", "Empirical copula", "copula_empirical_eng.png"),
        # Label: empirical copula.
        ("gaussian", "centers", "Gaussian copula", "copula_gaussian_eng.png"),
        # Label: Gaussian copula.
        ("student_t", "centers", "Student t copula", "copula_student_t_eng.png"),
        # Label: Student-t copula.
        ("checkerboard", "checkerboard_centers", "Checkerboard copula", "copula_checkerboard_eng.png"),
        # Label: checkerboard copula.
        ("bernstein", "bernstein_grid", "Bernstein copula", "copula_bernstein_eng.png"),
        # Label: Bernstein copula.
    )
    joint_models = (
        ("joint_gaussian", "Gaussian copula stationary density", "gaussian_stationary_density_eng.png"),
        # Label: Gaussian-copula joint stationary density.
        ("joint_checkerboard", "Checkerboard copula stationary density", "checkerboard_stationary_density_eng.png"),
        # Label: checkerboard-copula joint stationary density.
        ("joint_bernstein", "Bernstein copula stationary density", "bernstein_stationary_density_eng.png"),
        # Label: Bernstein-copula joint stationary density.
    )

    def _cap(data_arrays, percentile):
        """Compute cap for plot copula results from ``data_arrays``, ``percentile``."""
        finite_caps = []
        for values in data_arrays:
            a = np.asarray(values, dtype=float)
            finite = a[np.isfinite(a) & (a >= 0)]
            if finite.size:
                finite_caps.append(float(np.percentile(finite, percentile)))
        cap = max(finite_caps, default=1.0)
        return cap if np.isfinite(cap) and cap > 0 else 1.0

    def _write_one(fig, name):
        """Write one for the copula plotting workflow."""
        destination = folder / name.replace("_eng.png", f"_{suffix}.png")
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=dpi)
        outputs.append(destination)
        print(f"[copula plot] Saved: {destination}", flush=True)

    def _rank_labels(ax):
        """Compute rank labels for plot copula results from ``ax``."""
        ax.set_xlabel("U = F1(X1)")
        # U = F1(X1): rank of the first variable.
        ax.set_ylabel("V = F2(X2)")
        # V = F2(X2): rank of the second variable.
        ax.set_xlim(0.01, 0.99)
        ax.set_ylim(0.01, 0.99)
        ax.set_aspect("equal", adjustable="box")

    def _bounds(coordinate, density, variable):
        """Determine the relevant interval bounds."""
        if variable in display_ranges:
            # Keep the requested physical limits even for a narrower/coarser grid.
            return display_ranges[variable]
        coordinate = np.asarray(coordinate, dtype=float)
        # Use the central 99% of the saved marginal for other variables.
        density = np.maximum(np.asarray(density, dtype=float), 0.0)
        weights = 0.5 * (density[:-1] + density[1:]) * np.diff(coordinate)
        cdf = np.concatenate(([0.0], np.cumsum(weights)))
        if cdf[-1] > 0:
            cdf /= cdf[-1]
            lo, hi = np.interp((0.005, 0.995), cdf, coordinate)
        else:
            lo, hi = coordinate[0], coordinate[-1]
        lo = max(float(lo), float(coordinate[0]))
        hi = min(float(hi), float(coordinate[-1]))
        visible = int(np.count_nonzero((coordinate >= lo) & (coordinate <= hi)))
        if lo >= hi or visible < 4:
            lo, hi = float(coordinate[0]), float(coordinate[-1])
        return lo, hi

    # Shared colour limits let the reader compare models on the same scale.
    # The 99th / 99.5th percentiles reduce distortion from singular corner cells.
    with np.load(source, allow_pickle=False) as data, plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "savefig.facecolor": "white",
    }):
        rank_arrays = {}
        for key, grid_key, title, filename in rank_models:
            xy = np.asarray(data[grid_key], dtype=float)
            pdf = np.asarray(data[key], dtype=float)
            if xy.ndim != 1 or len(xy) < 2 or np.any(np.diff(xy) <= 0):
                raise ValueError(f"{grid_key}: rank grid must be increasing")
            if pdf.shape != (len(xy), len(xy)):
                raise ValueError(f"{key}: copula density {pdf.shape} does not match rank grid {len(xy)}")
            rank_arrays[key] = (xy, pdf, title, filename)
        # Apply the common vmax only to visible plotting region: singularities
        # at nearly 0 or 1 should not wash out the meaningful central colour map.
        rank_vmax = _cap([
            pdf[np.ix_((xy >= 0.02) & (xy <= 0.98),
                        (xy >= 0.02) & (xy <= 0.98))]
            for xy, pdf, _, _ in rank_arrays.values()
        ], percentile=99.0)
        for key, (xy, pdf, title, filename) in rank_arrays.items():
            fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
            try:
                image = ax.pcolormesh(xy, xy, np.ma.masked_invalid(pdf),
                                      shading="auto", cmap=cmap,
                                      vmin=0, vmax=rank_vmax)
                ax.set_title(title)
                # Compare empirical, Gaussian, Student-t, checkerboard and Bernstein copulas.
                _rank_labels(ax)
                if key == "checkerboard":
                    ax.set_xlim(0.02, 0.98)
                    ax.set_ylim(0.02, 0.98)
                fig.colorbar(image, ax=ax, label="Copula density")
                # Color scale: copula density.
                _write_one(fig, filename)
            finally:
                plt.close(fig)

        # Old-style normalized-level overlays: the models need not have
        # identical absolute peaks for their shapes to be comparable.
        def _plot_contours(models, title, filename):
            """Plot contours for the copula plotting workflow."""
            fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
            try:
                for key, colour, linestyle, legend in models:
                    xy, pdf, _, _ = rank_arrays[key]
                    vmax = float(np.nanmax(pdf))
                    if np.isfinite(vmax) and vmax > 0:
                        ax.contour(xy, xy, pdf / vmax,
                                   levels=np.linspace(0.05, 0.90, 10),
                                   colors=colour, linestyles=linestyle,
                                   linewidths=1.3)
                    ax.plot([], [], color=colour, linestyle=linestyle,
                            linewidth=1.7, label=legend)
                _rank_labels(ax)
                ax.set_title(title)
                # Compare copula density contours.
                ax.legend(loc="upper left", fontsize=10)
                _write_one(fig, filename)
            finally:
                plt.close(fig)

        _plot_contours(
            (("empirical", "black", "-", "Empirical"),
             ("gaussian", "darkorange", "--", "Gaussian"),
             ("student_t", "forestgreen", ":", "Student t")),
            # Legend: empirical, Gaussian and Student-t.
            "Empirical vs Gaussian vs Student t copula",
            "copula_models_contours_eng.png",
        )
        _plot_contours(
            (("checkerboard", "black", "-", "Checkerboard"),
             ("bernstein", "red", "--", "Bernstein")),
            # Legend: checkerboard and Bernstein.
            "Checkerboard vs Bernstein copula",
            "checkerboard_bernstein_contours_eng.png",
        )

        x1 = np.asarray(data["x1"], dtype=float)
        x2 = np.asarray(data["x2"], dtype=float)
        if x1.ndim != 1 or x2.ndim != 1:
            raise ValueError("Joint-density coordinates must be 1D")
        joint = {}
        for key, title, filename in joint_models:
            pdf = np.asarray(data[key], dtype=float)
            if pdf.shape != (len(x2), len(x1)):
                raise ValueError(f"{key}: expected {(len(x2), len(x1))}, got {pdf.shape}")
            joint[key] = (pdf, title, filename)
        ref = joint["joint_gaussian"][0]
        bounds_x = _bounds(x1, np.trapezoid(ref, x2, axis=0), names[0])
        bounds_y = _bounds(x2, np.trapezoid(ref, x1, axis=1), names[1])
        # All models use identical physical bounds and the same colour scale.
        visible_x = (x1 >= bounds_x[0]) & (x1 <= bounds_x[1])
        visible_y = (x2 >= bounds_y[0]) & (x2 <= bounds_y[1])
        vmax = _cap([
            pdf[np.ix_(visible_y, visible_x)]
            for pdf, _, _ in joint.values()
        ], percentile=99.5)
        for key, (pdf, title, filename) in joint.items():
            fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
            try:
                image = ax.pcolormesh(x1, x2, np.ma.masked_invalid(pdf),
                                      shading="auto", cmap=cmap,
                                      vmin=0.0, vmax=vmax)
                ax.set_xlim(*bounds_x)
                ax.set_ylim(*bounds_y)
                ax.set_title(title)
                # Joint stationary copula density.
                ax.set_xlabel(NAMES_EN.get(names[0], names[0]))
                # ax.set_xlabel(NAMES_RU.get(names[0], names[0]))
                ax.set_ylabel(NAMES_EN.get(names[1], names[1]))
                # ax.set_ylabel(NAMES_RU.get(names[1], names[1]))
                fig.colorbar(image, ax=ax, label="Stationary joint density")
                # Color scale: joint stationary density.
                _write_one(fig, filename)
            finally:
                plt.close(fig)

        # Two additional figures reproduce the sample Bernstein marginals.
        # They require the classical 1D stationary PDFs already used to build
        # the saved copula models. If not present, simply omit the two figures.
        stationary = (root / "Functional" / "Stationary" /
                      f"{pair}_{period}.npz")
        if stationary.is_file():
            with np.load(stationary, allow_pickle=False) as marginal_source:
                old_grid = np.asarray(marginal_source["grid"], dtype=float)
                marginal_reference = [
                    np.interp(x, old_grid,
                              np.asarray(marginal_source[f"pdf_{name}"], dtype=float))
                    for x, name in ((x1, names[0]), (x2, names[1]))
                ]
            bernstein = joint["joint_bernstein"][0]
            derived = [np.trapezoid(bernstein, x2, axis=0),
                       np.trapezoid(bernstein, x1, axis=1)]
            for index, (x, reference, reconstructed) in enumerate(
                    zip((x1, x2), marginal_reference, derived), start=1):
                fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
                try:
                    ax.plot(x, reference, color="tab:blue", lw=2,
                            label="1D stationary")
                    # Label: one-dimensional stationary density.
                    ax.plot(x, reconstructed, color="tab:orange", ls="--", lw=2,
                            label="Bernstein 2D marginal")
                    # Label: marginal of the two-dimensional Bernstein copula.
                    ax.set_xlim(*(bounds_x if index == 1 else bounds_y))
                    ax.set_xlabel(NAMES_EN.get(names[index - 1], names[index - 1]))
                    # ax.set_xlabel(NAMES_RU.get(names[index - 1], names[index - 1]))
                    ax.set_ylabel("Probability density")
                    # Probability density.
                    ax.set_title(f"Bernstein copula: marginal of {names[index - 1]}")
                    # Bernstein copula: marginal of the selected variable.
                    ax.legend()
                    _write_one(fig, f"bernstein_marginal_x{index}_eng.png")
                finally:
                    plt.close(fig)
        else:
            print(f"[copula plot] Stationary marginals not found: {stationary}"
                  " (skipping two marginal figures)", flush=True)
    outputs.extend(plot_copula_contours(root, names, start_year, end_year,
                                        language=language, dpi=dpi, output_dir=folder,
                                        monochrome=monochrome))
    print(f"[copula plot] Saved {len(outputs)} individual figures from {source}", flush=True)
    return outputs


@with_plot_language
def plot_copula_contours(root, pair, start_year, end_year, *, language="en", dpi=300,
                         output_dir=None, monochrome=False):
    """Draw four individual model contours from cached rank PDFs."""
    period = (int(start_year), int(end_year))
    if output_dir is None:
        output_dir = Path(root) / "Plots" / "Copulas" / "Figures" / f"{pair[0]}-{pair[1]}_{period[0]}-{period[1]}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    suffix = "eng" if language == "en" else "ru"
    source = (Path(root) / "Functional" / "Copulas" /
              f"{pair[0]}-{pair[1]}_{period[0]}-{period[1]}.npz")
    with np.load(source, allow_pickle=False) as data:
        models = ("gaussian", "student_t", "checkerboard", "bernstein")
        outputs = []
        titles_en = {"gaussian": "Gaussian copula", "student_t": "Student t copula",
                     "checkerboard": "Checkerboard copula", "bernstein": "Bernstein copula"}
        titles_ru = {"gaussian": "Гауссова копула", "student_t": "Копула Стьюдента",
                     "checkerboard": "Шахматная копула", "bernstein": "Копула Бернштейна"}
        for name in models:
            coord_key = ("bernstein_grid" if name == "bernstein" else
                         "checkerboard_centers" if name == "checkerboard" else "centers")
            grid = np.asarray(data[coord_key], dtype=float)
            pdf = np.asarray(data[name], dtype=float)
            if pdf.shape != (len(grid), len(grid)):
                raise ValueError(f"{name}: unexpected shape {pdf.shape}")
            positive = pdf[np.isfinite(pdf)]
            peak = float(positive.max()) if positive.size else 0.
            if peak <= 0:
                raise ValueError(f"{name}: no positive copula density")
            fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
            try:
                if monochrome:
                    levels = np.linspace(0.05, 0.90, 10)
                    line_styles = [":", "--", "-.", "-", ":", "--", "-.", "-", "--", "-"]
                    line_widths = [1.0, 1.0, 1.1, 1.2, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45]
                    contours = ax.contour(grid, grid, pdf / peak,
                                          levels=levels,
                                          colors="black", linewidths=line_widths,
                                          linestyles=line_styles)
                    ax.clabel(contours, inline=True, fontsize=7, fmt="%.2f", colors="black")
                else:
                    contours = ax.contour(grid, grid, pdf / peak,
                                          levels=np.linspace(0.05, 0.90, 10),
                                          cmap="YlOrRd", linewidths=1.5)
                    ax.clabel(contours, inline=True, fontsize=7, fmt="%.2f")
                ax.set_xlim(.02, .98)
                ax.set_ylim(.02, .98)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlabel(r"$U=F_1(X_1)$")
                ax.set_ylabel(r"$V=F_2(X_2)$")
                # ax.set_title((titles_en if language == "en" else titles_ru)[name])
                path = Path(output_dir) / f"copula_{name}_contours_{suffix}{'_bw' if monochrome else ''}.png"
                fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
                outputs.append(path)
            finally:
                plt.close(fig)
    return outputs



# Additional diagnostics with distinct inputs/outputs, retained from the old module.
# These are not generated by the standard cached copula workflow.
def plot_copula_marginals(files_path_prefix, x1, p1, marginal_x1, x2, p2, marginal_x2):
    """Plot copula marginals for the copula plotting workflow."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x1, p1, label="1D stationary")
    ax.plot(x1, marginal_x1, "--", label="2D copula marginal")
    ax.set_xlabel("Sensible heat flux")
    ax.set_ylabel("Density")
    ax.legend()
    fig.savefig(files_path_prefix + f'videos/Functional/2d_copula_sensible_marginal.png', dpi=300,)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x2, p2, label="1D stationary")
    ax.plot(x2, marginal_x2, "--", label="2D copula marginal")
    ax.set_xlabel("Latent heat flux")
    ax.set_ylabel("Density")
    ax.legend()
    fig.savefig(files_path_prefix + f'videos/Functional/2d_copula_latent_marginal.png', dpi=300,)
    plt.close(fig)


def plot_joint_contours(files_path_prefix, x1, x2, joint_density, levels=15, title="Gaussian copula stationary density", xlim=None, ylim=None):
    """Plot joint contours for the copula plotting workflow."""
    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    contour = ax.contour(x1, x2, joint_density, levels=levels)
    ax.clabel(contour, inline=True, fontsize=8)
    ax.set_xlabel("Sensible heat flux")
    ax.set_ylabel("Latent heat flux")
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.savefig(files_path_prefix + f'videos/Functional/joint_contours.png', dpi=300,)
    plt.close(fig)


def plot_empirical_copula_comparison(files_path_prefix, x1, x2, empirical_density, copula_density, xlim=None, ylim=None, same_scale=True):
    """Plot empirical copula comparison for the copula plotting workflow."""
    vmax = max(np.nanmax(empirical_density), np.nanmax(copula_density)) if same_scale else None

    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    mesh = ax.pcolormesh(x1, x2, empirical_density, shading="auto", vmin=0, vmax=vmax)
    fig.colorbar(mesh, ax=ax, label="Density")
    ax.set_xlabel("Sensible heat flux")
    ax.set_ylabel("Latent heat flux")
    ax.set_title("Empirical joint density")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.savefig(files_path_prefix + f'videos/Functional/empirical_joint_density.png', dpi=300,)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    mesh = ax.pcolormesh(x1, x2, copula_density, shading="auto", vmin=0, vmax=vmax)
    fig.colorbar(mesh, ax=ax, label="Density")
    ax.set_xlabel("Sensible heat flux")
    ax.set_ylabel("Latent heat flux")
    ax.set_title("Gaussian copula stationary density")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.savefig(files_path_prefix + f'videos/Functional/copula_density_2d.png', dpi=300,)
    plt.close(fig)


def plot_empirical_copula_contours(files_path_prefix, x1, x2, empirical_density, copula_density, levels=12, xlim=None, ylim=None):
    """Plot empirical copula contours for the copula plotting workflow."""
    empirical_scaled = empirical_density / np.nanmax(empirical_density)
    copula_scaled = copula_density / np.nanmax(copula_density)
    contour_levels = np.linspace(0.05, 0.95, levels)

    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    ax.contour(x1, x2, empirical_scaled, levels=contour_levels, linewidths=1.5)
    ax.contour(x1, x2, copula_scaled, levels=contour_levels, linestyles="--", linewidths=1.5)
    ax.set_xlabel("Sensible heat flux")
    ax.set_ylabel("Latent heat flux")
    ax.set_title("Empirical (solid) vs Gaussian copula (dashed)")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.savefig(files_path_prefix + f'videos/Functional/joint_contours_empirical.png', dpi=300,)
    plt.close(fig)


def plot_copula_space_contours(files_path_prefix, centers, empirical_density, gaussian_density, levels=12):
    """Plot copula space contours for the copula plotting workflow."""
    empirical_scaled = empirical_density / np.nanmax(empirical_density)
    gaussian_scaled = gaussian_density / np.nanmax(gaussian_density)
    contour_levels = np.linspace(0.05, 0.95, levels)

    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    ax.contour(centers, centers, empirical_scaled, levels=contour_levels, linewidths=1.5)
    ax.contour(centers, centers, gaussian_scaled, levels=contour_levels, linestyles="--", linewidths=1.5)
    ax.plot([0, 1], [0, 1], ":", linewidth=1)
    ax.set_xlabel("U = F1(X1)")
    ax.set_ylabel("V = F2(X2)")
    ax.set_title("Empirical copula (solid) vs Gaussian copula (dashed)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(files_path_prefix + f'videos/Functional/copula_space_contours.png', dpi=300,)
    plt.close(fig)


def save_figure(files_path_prefix, fig, filename):
    """Save figure for the copula plotting workflow."""
    os.makedirs(files_path_prefix + "videos/Functional/", exist_ok=True)
    fig.savefig(files_path_prefix + f"videos/Functional/{filename}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_checkerboard_marginals(files_path_prefix, x1, p1, marginal_x1, x2, p2, marginal_x2):
    """Plot checkerboard marginals for the copula plotting workflow."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x1, p1, label="1D stationary")
    ax.plot(x1, marginal_x1, "--", label="Checkerboard 2D marginal")
    ax.set_xlabel("Sensible heat flux")
    ax.set_ylabel("Density")
    ax.legend()
    save_figure(files_path_prefix, fig, "checkerboard_marginal_x1.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x2, p2, label="1D stationary")
    ax.plot(x2, marginal_x2, "--", label="Checkerboard 2D marginal")
    ax.set_xlabel("Latent heat flux")
    ax.set_ylabel("Density")
    ax.legend()
    save_figure(files_path_prefix, fig, "checkerboard_marginal_x2.png")


def plot_empirical_checkerboard_contours(files_path_prefix, x1, x2, empirical_density, checkerboard_density, levels=10, xlim=None, ylim=None):
    """Plot empirical checkerboard contours for the copula plotting workflow."""
    empirical_scaled = empirical_density / np.nanmax(empirical_density)
    checkerboard_scaled = checkerboard_density / np.nanmax(checkerboard_density)
    contour_levels = np.linspace(0.05, 0.9, levels)

    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    ax.contour(x1, x2, empirical_scaled, levels=contour_levels, linewidths=1.5)
    ax.contour(x1, x2, checkerboard_scaled, levels=contour_levels, linestyles="--", linewidths=1.5)

    ax.plot([], [], "-", label="Empirical")
    ax.plot([], [], "--", label="Checkerboard stationary")
    ax.set_xlabel("Sensible heat flux")
    ax.set_ylabel("Latent heat flux")
    ax.set_title("Empirical vs checkerboard stationary density")
    ax.legend()

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    save_figure(files_path_prefix, fig, "empirical_checkerboard_contours.png")


def plot_bernstein_m_sensitivity(files_path_prefix):
    """Plot bernstein m sensitivity for the copula plotting workflow."""
    m_values = np.array([30, 40, 50, 60, 70])

    correlation = np.array([
        0.48003290516952485, 0.4930341404032566, 0.5011049214969485,
        0.507138952874245, 0.5111178229066088
    ])

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(m_values, correlation, marker='o', linewidth=2.5)
    ax.axvline(50, linestyle='--', linewidth=1.5, label='Selected m = 50')

    ax.set_xlabel('Bernstein degree m')
    ax.set_ylabel('Pearson correlation')
    ax.set_xticks(m_values)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(files_path_prefix + 'videos/Functional/copula_m_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
