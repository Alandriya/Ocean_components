"""Plots of stationary quantile regions, anomaly isolines and geographic classes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D
import numpy as np

from Functional.stationary_diagnostics import prepare_stationary_diagnostics
from Plotting.language import with_plot_language

# Original yellow/orange/pink/violet classes, light-green land.
# The ordering is lower to higher heat-flux value, not chronological order.
AREA_COLORS = ("#ffff86", "#ffd27d", "#ff8083", "#f6a9dc", "#7774fb")
# Monochrome: identical class order and hatches in both figures.
AREA_HATCHES = ("..", "||", "//", r"\\", "xx")
AREA_GRAYS = ("#ffffff", "#f5f5f5", "#ebebeb", "#e1e1e1", "#d7d7d7")
CLASS_LAND_GRAY = "#a6a6a6"
LAND_COLOR = "#baf0c0"
MAP_EXTENT = (-90., 0., 0., 80.)  # lon 90 W..0 E; lat 0..80 N
VARIABLES = {"sensible": "Sensible heat flux", "latent": "Latent heat flux"}

LABELS = {
    "en": {
        "regions": "Five equal-probability regions",
        "isolines": "Anomaly isolines",
        "classes": "Stationary quantile classes",
        "density": "Probability density",
        "heat": "Heat flux (W m$^{-2}$)",
        "ocean": "ERA5 ocean cells",
        "land": "Land",
        "area": "Area",
        "method1": "1D stationary-reference anomaly",
        "method2": "2D alternative estimate",
        "sensible": "Sensible heat flux",
        "latent": "Latent heat flux",
    },
    "ru": {
        "regions": "Пять равновероятных областей",
        "isolines": "Изолинии аномалий",
        "classes": "Классы стационарных квантилей",
        "density": "Плотность вероятности",
        "heat": "Тепловой поток (Вт м$^{-2}$)",
        "ocean": "Океанические ячейки ERA5",
        "land": "Суша",
        "area": "Область",
        "method1": "Аномалия относительно 1D стационарного уровня",
        "method2": "Двумерная оценка",
        "sensible": "явный поток",
        "latent": "скрытый поток",
    },
}

ISOLINE_LEVELS = {
    "sensible": (-60, -45, -30, -15, 0, 15, 30),
    "latent": (-160, -120, -80, -40, 0, 40, 80),
}


def _save(fig, path, dpi):
    """Save with the shared, scoped figure language."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"[stationary spatial plots] Saved {path}", flush=True)
    return path


def _figure_directory(root, period):
    target = Path(root) / "Plots" / "Stationary" / "Spatial" / f"{period[0]}-{period[1]}"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _map_style(ax, lon, lat):
    ax.set_xlim(lon[0], lon[-1])
    ax.set_ylim(lat[-1], lat[0])
    ticks_lon = np.linspace(lon[0], lon[-1], 4)
    ticks_lat = (0, 30, 60, 80)
    ax.set_xticks(ticks_lon)
    ax.set_xticklabels([f"{round(abs(x))}°W" if x < 0 else "0°"
                        for x in ticks_lon])
    ax.set_yticks(ticks_lat)
    ax.set_yticklabels(["0°", "30°N", "60°N", "80°N"])
    ax.grid(color="#d6d6d6", lw=0.5, zorder=1)
    ax.set_aspect("auto")


def _lon_lat(shape):
    height, width = shape
    return (np.linspace(-90, 0, width, dtype=np.float64),
            np.linspace(80, 0, height, dtype=np.float64))


def _map_background(ax, ocean, *, monochrome=False):
    bg = np.ma.array(np.ones(ocean.shape), mask=ocean)
    cmap = ListedColormap([CLASS_LAND_GRAY if monochrome else LAND_COLOR])
    cmap.set_bad("white")
    ax.imshow(bg, cmap=cmap, origin="upper", extent=MAP_EXTENT,
              interpolation="nearest", zorder=0, aspect="auto")


def _class_grid_edges(shape):
    """Pixel edges matching the original imshow extent, with row 0 at north."""
    height, width = shape
    west, east, south, north = MAP_EXTENT
    return (np.linspace(west, east, width + 1),
            np.linspace(north, south, height + 1))


def _class_mask_path(mask, x_edges, y_edges):
    """Combine contiguous row runs into one hatch path without cell borders."""
    transitions = np.diff(np.pad(mask.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    rows, starts = np.nonzero(transitions == 1)
    end_rows, ends = np.nonzero(transitions == -1)
    if not rows.size:
        return None
    assert np.array_equal(rows, end_rows)
    l, r = x_edges[starts], x_edges[ends]
    b, t = y_edges[rows + 1], y_edges[rows]
    vertices = np.stack((np.stack((l, b), axis=-1),
                         np.stack((r, b), axis=-1),
                         np.stack((r, t), axis=-1),
                         np.stack((l, t), axis=-1),
                         np.stack((l, b), axis=-1)), axis=1).reshape(-1, 2)
    codes = np.tile([MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO,
                     MplPath.LINETO, MplPath.CLOSEPOLY], len(rows))
    return MplPath(vertices, codes)


def _grid_boundary_segments(vertical, horizontal, x_edges, y_edges):
    """Draw only shared cell edges at which the selected state changes."""
    rows, cols = np.nonzero(vertical)
    x = x_edges[cols + 1]
    vertical_segments = np.stack((np.column_stack((x, y_edges[rows])),
                                  np.column_stack((x, y_edges[rows + 1]))), axis=1)
    rows, cols = np.nonzero(horizontal)
    y = y_edges[rows + 1]
    horizontal_segments = np.stack((np.column_stack((x_edges[cols], y)),
                                    np.column_stack((x_edges[cols + 1], y))), axis=1)
    return np.concatenate((vertical_segments, horizontal_segments), axis=0)


def _draw_hatched_classes(ax, ocean, classes):
    """Plot categorical pixels, not interpolated contours of class numbers."""
    x_edges, y_edges = _class_grid_edges(classes.shape)
    valid = ocean & (classes >= 1) & (classes <= 5)
    # Only land is gray. Missing ocean cells have no hatch (white).
    land = np.ma.masked_where(ocean, np.ones(ocean.shape))
    cmap = ListedColormap([CLASS_LAND_GRAY])
    cmap.set_bad("white")
    ax.imshow(land, cmap=cmap, origin="upper", extent=MAP_EXTENT,
              interpolation="nearest", zorder=0, aspect="auto")
    for k, (gray, hatch) in enumerate(zip(AREA_GRAYS, AREA_HATCHES), start=1):
        path = _class_mask_path(valid & (classes == k), x_edges, y_edges)
        if path is not None:
            ax.add_patch(PathPatch(path, facecolor=gray, edgecolor="black",
                                   linewidth=0, hatch=hatch, zorder=2))

    # Keep both class and coastline boundaries exactly on original cell edges.
    # No contours between cells of the same class; no interpolation/smoothing.
    state = np.where(valid, classes, 0)
    vertical = (ocean[:, :-1] & ocean[:, 1:] &
                (state[:, :-1] != state[:, 1:]))
    horizontal = (ocean[:-1, :] & ocean[1:, :] &
                  (state[:-1, :] != state[1:, :]))
    class_edges = _grid_boundary_segments(vertical, horizontal, x_edges, y_edges)
    coast_edges = _grid_boundary_segments(ocean[:, :-1] != ocean[:, 1:],
                                          ocean[:-1, :] != ocean[1:, :],
                                          x_edges, y_edges)
    ax.add_collection(LineCollection(class_edges, colors="black", linewidths=0.55,
                                     zorder=3))
    ax.add_collection(LineCollection(coast_edges, colors="black", linewidths=0.8,
                                     zorder=4))


@with_plot_language
def plot_area_density(cache, variable, period, output_dir, *, language="en", dpi=300,
                      display_xlim=None, monochrome=False):
    """Exactly five shaded quintiles of an existing reference PDF."""
    t = LABELS[language]
    with np.load(cache, allow_pickle=False) as data:
        grid, pdf = np.asarray(data["grid"]), np.asarray(data["pdf"])
        qs = np.asarray(data["thresholds"])
        masses = np.asarray(data["region_masses"])
    bounds = np.r_[grid[0], qs, grid[-1]]
    fig, ax = plt.subplots(figsize=(10, 5.4), constrained_layout=True)
    try:
        if monochrome:
            for i, (left, right) in enumerate(zip(bounds[:-1], bounds[1:])):
                inside = (grid > left) & (grid < right)
                xx = np.r_[left, grid[inside], right]
                yy = np.interp(xx, grid, pdf)
                # ax.fill_between(xx, yy, color=AREA_COLORS[i], alpha=0.8,
                #                 label=f"{t['area']} {i+1} ({masses[i]*100:.1f}%)")
                ax.fill_between(xx, yy, facecolor=AREA_GRAYS[i], edgecolor="black",
                                hatch=AREA_HATCHES[i], linewidth=0.6,
                                label=f"{t['area']} {i+1} ({masses[i]*100:.1f}%)")
            # ax.plot(grid, pdf, color="#aa182c", lw=1.6, label=r"$p_{ref}(x)$")
            ax.plot(grid, pdf, color="black", lw=1.6, label=r"$p_{ref}(x)$")
            for q in qs:
                ax.axvline(q, color="black", lw=0.75, ls="--")
        else:
            for i, (left, right) in enumerate(zip(bounds[:-1], bounds[1:])):
                inside = (grid > left) & (grid < right)
                xx = np.r_[left, grid[inside], right]
                yy = np.interp(xx, grid, pdf)
                ax.fill_between(xx, yy, color=AREA_COLORS[i], alpha=0.8,
                                label=f"{t['area']} {i+1} ({masses[i]*100:.1f}%)")
            ax.plot(grid, pdf, color="#aa182c", lw=1.6, label=r"$p_{ref}(x)$")
            for q in qs:
                ax.axvline(q, color="#666666", lw=1, ls="--", alpha=0.65)
        if display_xlim is None:
            if variable == "sensible":
                display_xlim = (-200, 100)
            elif variable == "latent":
                display_xlim = (-500, 150)
            else:
                display_xlim = (float(grid[0]), float(grid[-1]))
        ax.set_xlim(*display_xlim)
        ax.set_ylim(bottom=0)
        # ax.set_title(f"{t['regions']}: {t.get(variable, variable)} ({period[0]}–{period[1]})")
        ax.set_xlabel(t["heat"])
        ax.set_ylabel(t["density"])
        ax.grid(True, color="#dddddd", alpha=0.4, zorder=0)
        if monochrome:
            ax.legend(loc="upper left", fontsize=9, handlelength=3, handleheight=1.5)
        else:
            ax.legend(loc="upper left", fontsize=9)
        path = Path(output_dir) / f"{variable}_regions_{language}{'_bw' if monochrome else ''}.png"
        return _save(fig, path, dpi)
    finally:
        plt.close(fig)


@with_plot_language
def plot_anomaly_isolines(cache, variable, period, output_dir, *, language="en",
                          dpi=300, levels=None, second_map=None, second_label=None,
                          monochrome=False):
    """Delta X = ERA5 time mean minus a *model* reference mean.

    Optional second_map is an explicitly supplied 2D anomaly (H,W) NPY. It is
    never inferred or fabricated from the 1D field.
    """
    t = LABELS[language]
    with np.load(cache, allow_pickle=False) as data:
        ocean = np.asarray(data["ocean_mask"], dtype=bool)
        anomaly = np.asarray(data["anomaly"], dtype=float)
        mu = float(data["model_mean"])
    lon, lat = _lon_lat(ocean.shape)
    left = np.where(ocean, anomaly, np.nan)
    if levels is None:
        levels = ISOLINE_LEVELS.get(variable)
    if levels is None:
        values = left[np.isfinite(left)]
        if not values.size:
            raise ValueError(f"No valid ocean anomaly for {variable}")
        levels = np.linspace(*np.percentile(values, (5, 95)), 9)
    levels = np.asarray(levels, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 8.5), constrained_layout=True)
    try:
        _map_background(ax, ocean, monochrome=monochrome)
        if monochrome:
            valid_levels = levels[(levels > np.nanmin(left)) & (levels < np.nanmax(left))]
            if valid_levels.size:
                neg = valid_levels[valid_levels < 0]
                zero = valid_levels[np.isclose(valid_levels, 0)]
                pos = valid_levels[valid_levels > 0]
                if neg.size:
                    cs_neg = ax.contour(lon, lat, left, levels=neg, colors="black",
                                        linewidths=1.2, linestyles="--", zorder=3)
                    ax.clabel(cs_neg, fmt="%g", fontsize=8, inline=True, inline_spacing=2,
                              colors="black")
                if zero.size:
                    cs_zero = ax.contour(lon, lat, left, levels=zero, colors="black",
                                         linewidths=2.0, linestyles="-", zorder=4)
                    ax.clabel(cs_zero, fmt="%g", fontsize=8, inline=True, inline_spacing=2,
                              colors="black")
                if pos.size:
                    cs_pos = ax.contour(lon, lat, left, levels=pos, colors="black",
                                        linewidths=1.4, linestyles="-", zorder=3)
                    ax.clabel(cs_pos, fmt="%g", fontsize=8, inline=True, inline_spacing=2,
                              colors="black")
            legend = [
                Line2D([0], [0], color="black", lw=1.2, ls="--",
                       label=(t["method1"] + ": отрицательные" if language == "ru"
                              else t["method1"] + ": negative")),
                Line2D([0], [0], color="black", lw=2.0, ls="-",
                       label=("Нулевая изолиния" if language == "ru" else "Zero isoline")),
                Line2D([0], [0], color="black", lw=1.4, ls="-",
                       label=(t["method1"] + ": положительные" if language == "ru"
                              else t["method1"] + ": positive")),
            ]
            if second_map is not None:
                source = Path(second_map)
                if not source.is_file():
                    raise FileNotFoundError(f"Optional 2D anomaly not found: {source}")
                other = np.load(source, mmap_mode="r", allow_pickle=False)
                if other.shape != ocean.shape:
                    raise ValueError(f"2D map shape {other.shape} != {ocean.shape}")
                other = np.where(ocean, np.asarray(other, float), np.nan)
                valid2 = levels[(levels > np.nanmin(other)) & (levels < np.nanmax(other))]
                if valid2.size:
                    neg2 = valid2[valid2 < 0]
                    zero2 = valid2[np.isclose(valid2, 0)]
                    pos2 = valid2[valid2 > 0]
                    if neg2.size:
                        cs2_neg = ax.contour(lon, lat, other, levels=neg2,
                                             colors="black", linewidths=1.0,
                                             linestyles=":", zorder=4)
                        ax.clabel(cs2_neg, fmt="%g", fontsize=7, inline=True, colors="black")
                    if zero2.size:
                        cs2_zero = ax.contour(lon, lat, other, levels=zero2,
                                              colors="black", linewidths=1.8,
                                              linestyles="-.", zorder=5)
                        ax.clabel(cs2_zero, fmt="%g", fontsize=7, inline=True, colors="black")
                    if pos2.size:
                        cs2_pos = ax.contour(lon, lat, other, levels=pos2,
                                             colors="black", linewidths=1.0,
                                             linestyles='-.', zorder=4)
                        ax.clabel(cs2_pos, fmt="%g", fontsize=7, inline=True, colors="black")
                legend.extend([
                    Line2D([0], [0], color="black", ls=":", lw=1.4,
                           label=((second_label or t["method2"]) + (": отрицательные" if language == "ru" else ": negative"))),
                    Line2D([0], [0], color="black", ls='-.', lw=1.8,
                           label=((second_label or t["method2"]) + (": нулевая/положительные" if language == "ru" else ": zero/positive")))
                ])
        else:
            valid_levels = levels[(levels > np.nanmin(left)) & (levels < np.nanmax(left))]
            if valid_levels.size:
                cs = ax.contour(lon, lat, left, levels=valid_levels,
                                cmap="autumn", linewidths=1.5, zorder=3)
                ax.clabel(cs, fmt="%g", fontsize=8, inline=True, inline_spacing=2)
            legend = [Line2D([0], [0], color="#f39800", lw=2,
                             label=t["method1"])]
            if second_map is not None:
                source = Path(second_map)
                if not source.is_file():
                    raise FileNotFoundError(f"Optional 2D anomaly not found: {source}")
                other = np.load(source, mmap_mode="r", allow_pickle=False)
                if other.shape != ocean.shape:
                    raise ValueError(f"2D map shape {other.shape} != {ocean.shape}")
                other = np.where(ocean, np.asarray(other, float), np.nan)
                valid2 = levels[(levels > np.nanmin(other)) & (levels < np.nanmax(other))]
                if valid2.size:
                    cs2 = ax.contour(lon, lat, other, levels=valid2,
                                     cmap="cool", linewidths=1.15, linestyles="--", zorder=4)
                    ax.clabel(cs2, fmt="%g", fontsize=7, inline=True)
                legend.append(Line2D([0], [0], color="#5453b0", ls="--", lw=2,
                                     label=second_label or t["method2"]))
        _map_style(ax, lon, lat)
        # ax.set_title(f"{t['isolines']}: {t.get(variable, variable)} ({period[0]}–{period[1]})")
        # RU: Изолинии ΔX = среднее ERA5 − стационарное модельное среднее
        ax.text(0.015, 0.975, fr"$\Delta X=\overline{{X}}-\mu_{{ref}}$, $\mu_{{ref}}={mu:.2f}$ W m$^{{-2}}$",
                fontsize=9, transform=ax.transAxes,
                va="top", bbox=dict(fc="white", ec="none", alpha=0.8, pad=3))
        ax.legend(handles=legend, loc="lower right", fontsize=9)
        path = Path(output_dir) / f"{variable}_isolines_{language}{'_bw' if monochrome else ''}.png"
        return _save(fig, path, dpi)
    finally:
        plt.close(fig)


@with_plot_language
def plot_class_map(cache, variable, map_date, output_dir, *, language="en", dpi=300,
                   monochrome=False):
    """ERA5 map at map_date, classified by stationary quantile thresholds."""
    t = LABELS[language]
    with np.load(cache, allow_pickle=False) as data:
        ocean = np.asarray(data["ocean_mask"], dtype=bool)
        classes = np.asarray(data["class_map"], dtype=np.uint8)
        qs = np.asarray(data["thresholds"])
    lon, lat = _lon_lat(ocean.shape)
    fig, ax = plt.subplots(figsize=(10, 8.5), constrained_layout=True)
    try:
        if monochrome:
            # Original categorical color map (disabled for monochrome printing):
            # img = np.ma.masked_where(~ocean | (classes == 0), classes)
            # cmap = ListedColormap(AREA_COLORS)
            # cmap.set_bad(LAND_COLOR)
            # artist = ax.imshow(img, origin="upper", extent=MAP_EXTENT,
            #                    cmap=cmap, norm=BoundaryNorm(np.arange(.5, 6.5, 1), 5),
            #                    interpolation="nearest", zorder=2, aspect="auto")
            _draw_hatched_classes(ax, ocean, classes)
            _map_style(ax, lon, lat)
            # cb = fig.colorbar(artist, ax=ax, ticks=np.arange(1, 6), fraction=.04, pad=.018)
            # cb.ax.set_yticklabels([f"{t['area']} {k}" for k in range(1, 6)])
            # cb.set_label(t["area"])
            handles = [Patch(facecolor=gray, edgecolor="black", linewidth=0.6,
                             hatch=hatch, label=f"{t['area']} {k}")
                       for k, (gray, hatch) in enumerate(zip(AREA_GRAYS, AREA_HATCHES), 1)]
            ax.legend(handles=handles, loc="center left",
                      bbox_to_anchor=(1.01, 0.5), fontsize=9, handlelength=3,
                      handleheight=1.8, labelspacing=0.8, borderpad=0.8,
                      framealpha=1, facecolor="white", edgecolor="black")
        else:
            img = np.ma.masked_where(~ocean | (classes == 0), classes)
            cmap = ListedColormap(AREA_COLORS)
            cmap.set_bad(LAND_COLOR)
            artist = ax.imshow(img, origin="upper", extent=MAP_EXTENT,
                               cmap=cmap, norm=BoundaryNorm(np.arange(.5, 6.5, 1), 5),
                               interpolation="nearest", zorder=2, aspect="auto")
            _map_style(ax, lon, lat)
            cb = fig.colorbar(artist, ax=ax, ticks=np.arange(1, 6), fraction=.04, pad=.018)
            cb.ax.set_yticklabels([f"{t['area']} {k}" for k in range(1, 6)])
        # ax.set_title(f"{t['classes']}: {t.get(variable, variable)} — {map_date}")
        # RU: Карта областей по квантильным порогам стационарной плотности
        # Retain numerical thresholds in JSON/NPZ rather than obscuring map labels.
        path = Path(output_dir) / f"{variable}_classes_{map_date}_{language}{'_bw' if monochrome else ''}.png"
        return _save(fig, path, dpi)
    finally:
        plt.close(fig)


@with_plot_language
def plot_tail_frequencies(cache, variable, period, output_dir, *, language="en", dpi=300):
    """Optional supplementary map: local frequency of regions 1/5 over the period."""
    with np.load(cache, allow_pickle=False) as data:
        if "frequencies" not in data.files:
            raise ValueError("No cached frequencies; rerun with compute_frequencies=True")
        freq = np.asarray(data["frequencies"], dtype=float)
        ocean = np.asarray(data["ocean_mask"], dtype=bool)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), constrained_layout=True)
    try:
        for ax, k in zip(axes, (0, 4)):
            img = ax.imshow(np.ma.masked_where(~ocean, freq[k]), cmap="YlOrRd",
                            vmin=0, vmax=max(0.05, min(1, float(np.nanpercentile(freq[k], 99)))),
                            origin="upper", extent=MAP_EXTENT, aspect="auto")
            _map_style(ax, *_lon_lat(ocean.shape))
            # ax.set_title(f"{variable}: region {k+1} frequency, {period[0]}–{period[1]}")
            fig.colorbar(img, ax=ax, shrink=.78, label="Fraction of valid days")
        output = Path(output_dir) / f"{variable}_tail_frequency_{language}.png"
        return _save(fig, output, dpi)
    finally:
        plt.close(fig)


@with_plot_language
def plot_stationary_spatial(files_path_prefix, names=("sensible", "latent"),
                         start_year=1979, end_year=2026, *,
                         regions_period=(1979, 1989), map_date="1979-01-01",
                         density_source="auto", hybrid_reference=(1979, 2026),
                         chunk_days=32, language="en", dpi=300,
                         compute_frequencies=False,
                         isoline_levels=None, second_anomaly_files=None,
                         figures=("isolines", "regions", "classes"), monochrome=False):
    """Render spatial diagnostics from valid precomputed caches only."""
    if language not in LABELS:
        raise ValueError("language must be 'en' or 'ru'")
    root = Path(files_path_prefix)
    period = (int(start_year), int(end_year))
    output_dir = _figure_directory(root, period)
    caches = prepare_stationary_diagnostics(root, tuple(names), *period,
                                         regions_period=regions_period, map_date=map_date,
                                         density_source=density_source,
                                         hybrid_reference=hybrid_reference,
                                         chunk_days=chunk_days,
                                         compute_frequencies=compute_frequencies,
                                         cache_only=True)
    allowed = {"isolines", "regions", "classes"}
    figures = tuple(figures)
    if not set(figures).issubset(allowed):
        raise ValueError(f"Unknown stationary spatial figure selector: {figures}")
    outputs = []
    for name, cache in caches.items():
        levels = isoline_levels.get(name) if isinstance(isoline_levels, dict) else isoline_levels
        other = second_anomaly_files.get(name) if second_anomaly_files else None
        if other is not None and not Path(other).is_absolute():
            other = root / other
        if "isolines" in figures:
            outputs.append(plot_anomaly_isolines(cache, name, period, output_dir,
                                                 language=language, dpi=dpi,
                                                 levels=levels, second_map=other,
                                                 monochrome=monochrome))
        if "regions" in figures:
            outputs.append(plot_area_density(cache, name, regions_period, output_dir,
                                             language=language, dpi=dpi,
                                             monochrome=monochrome))
        if "classes" in figures:
            outputs.append(plot_class_map(cache, name, map_date, output_dir,
                                          language=language, dpi=dpi,
                                          monochrome=monochrome))
        if compute_frequencies:
            outputs.append(plot_tail_frequencies(cache, name, period, output_dir,
                                                  language=language, dpi=dpi))
    manifest = {
        "name": "Stationary spatial diagnostics", "period": list(period), "figure_language": language,
        "regions_period": list(regions_period), "map_date": map_date,
        "density_source_setting": density_source,
        "figures": [str(p) for p in outputs],
        "data_cache": {k: str(v) for k, v in caches.items()},
        "monochrome": bool(monochrome),
        "important_note": "Reference-relative anomalies; hybrid PDF is not the original FP solution.",
    }
    selector = "-".join(figures) + ("-frequencies" if compute_frequencies else "")
    (output_dir / f"figure_manifest_{selector}_{language}{'_bw' if monochrome else ''}.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[stationary spatial plots] Completed: {len(outputs)} figures in {output_dir}", flush=True)
    return outputs

