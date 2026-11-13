"""Publication-ready maps and probability regions for the Ocean_components paper.

Revives plot_isolines_map / plot_areas / plot_areas_map from the pre-refactor
Plotting/plot_func_estimations.py. All output is a separate, 300-dpi PNG.
English captions are the default; the corresponding Russian texts are in
LABELS['ru'] for a future Russian-language edition.

Figure numbering follows main-ru.pdf, pp. 6--15:
  Fig. 1: geographic anomalies (sensible/latent)
  Fig. 2: five 20%-probability stationary regions (sensible/latent)
  Fig. 3: geographic classes for 1979-01-01 (sensible/latent)
  Figs. 4--8: existing stored copula models, per-model contours and marginals.
"""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
import numpy as np

from Functional.article_diagnostics import prepare_article_diagnostics

# Original yellow/orange/pink/violet classes, light-green land.
# The ordering is lower to higher heat-flux value, not chronological order.
AREA_COLORS = ("#ffff86", "#ffd27d", "#ff8083", "#f6a9dc", "#7774fb")
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
        "sensible": "Поток явного тепла",
        "latent": "Поток скрытого тепла",
    },
}

ISOLINE_LEVELS = {
    "sensible": (-60, -45, -30, -15, 0, 15, 30),
    "latent": (-160, -120, -80, -40, 0, 40, 80),
}


def _save(fig, path, dpi):
    """Save directly: Plotting.paths.save_figure otherwise translates EN to RU."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"[article figures] Saved {path}", flush=True)
    return path


def _figure_directory(root, period):
    target = Path(root) / "Plots" / "Article" / f"{period[0]}-{period[1]}"
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
    ax.set_yticklabels(["EQ", "30°N", "60°N", "80°N"])
    ax.grid(color="#d6d6d6", lw=0.5, zorder=1)
    ax.set_aspect("auto")


def _lon_lat(shape):
    height, width = shape
    return (np.linspace(-90, 0, width, dtype=np.float64),
            np.linspace(80, 0, height, dtype=np.float64))


def _map_background(ax, ocean):
    bg = np.ma.array(np.ones(ocean.shape), mask=ocean)
    cmap = ListedColormap([LAND_COLOR])
    cmap.set_bad("white")
    ax.imshow(bg, cmap=cmap, origin="upper", extent=MAP_EXTENT,
              interpolation="nearest", zorder=0, aspect="auto")


def plot_area_density(cache, variable, period, output_dir, *, language="en", dpi=300,
                      display_xlim=None):
    """Figure 2: exactly five shaded quintiles of an existing reference PDF."""
    t = LABELS[language]
    with np.load(cache, allow_pickle=False) as data:
        grid, pdf = np.asarray(data["grid"]), np.asarray(data["pdf"])
        qs = np.asarray(data["thresholds"])
        masses = np.asarray(data["region_masses"])
    bounds = np.r_[grid[0], qs, grid[-1]]
    fig, ax = plt.subplots(figsize=(10, 5.4), constrained_layout=True)
    try:
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
                display_xlim = (-450, 450)
            elif variable == "latent":
                display_xlim = (-750, 250)
            else:
                display_xlim = (float(grid[0]), float(grid[-1]))
        ax.set_xlim(*display_xlim)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{t['regions']}: {t.get(variable, variable)} ({period[0]}–{period[1]})")
        ax.set_xlabel(t["heat"])
        ax.set_ylabel(t["density"])
        ax.grid(True, color="#dddddd", alpha=0.4, zorder=0)
        ax.legend(loc="upper left", fontsize=9)
        path = Path(output_dir) / f"fig02_{variable}_regions_{language}.png"
        return _save(fig, path, dpi)
    finally:
        plt.close(fig)


def plot_anomaly_isolines(cache, variable, period, output_dir, *, language="en",
                          dpi=300, levels=None, second_map=None, second_label=None):
    """Figure 1: correct Delta X = ERA5 time mean minus a *model* reference mean.

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
        _map_background(ax, ocean)
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
        ax.set_title(f"{t['isolines']}: {t.get(variable, variable)} ({period[0]}–{period[1]})")
        # RU: Изолинии ΔX = среднее ERA5 − стационарное модельное среднее
        ax.text(0.015, 0.975, fr"$\Delta X=\overline{{X}}-\mu_{{ref}}$, $\mu_{{ref}}={mu:.2f}$ W m$^{{-2}}$",
                fontsize=9, transform=ax.transAxes,
                va="top", bbox=dict(fc="white", ec="none", alpha=0.8, pad=3))
        ax.legend(handles=legend, loc="lower right", fontsize=9)
        path = Path(output_dir) / f"fig01_{variable}_isolines_{language}.png"
        return _save(fig, path, dpi)
    finally:
        plt.close(fig)


def plot_class_map(cache, variable, map_date, output_dir, *, language="en", dpi=300):
    """Figure 3: 1979-01-01 ERA5 map, quantile-classified by Figure 2 thresholds."""
    t = LABELS[language]
    with np.load(cache, allow_pickle=False) as data:
        ocean = np.asarray(data["ocean_mask"], dtype=bool)
        classes = np.asarray(data["class_map"], dtype=np.uint8)
        qs = np.asarray(data["thresholds"])
    lon, lat = _lon_lat(ocean.shape)
    fig, ax = plt.subplots(figsize=(10, 8.5), constrained_layout=True)
    try:
        img = np.ma.masked_where(~ocean | (classes == 0), classes)
        cmap = ListedColormap(AREA_COLORS)
        cmap.set_bad(LAND_COLOR)
        artist = ax.imshow(img, origin="upper", extent=MAP_EXTENT,
                           cmap=cmap, norm=BoundaryNorm(np.arange(.5, 6.5, 1), 5),
                           interpolation="nearest", zorder=2, aspect="auto")
        _map_style(ax, lon, lat)
        cb = fig.colorbar(artist, ax=ax, ticks=np.arange(1, 6), fraction=.04, pad=.018)
        cb.ax.set_yticklabels([f"{t['area']} {k}" for k in range(1, 6)])
        cb.set_label(t["area"])
        ax.set_title(f"{t['classes']}: {t.get(variable, variable)} — {map_date}")
        # RU: Карта областей по квантильным порогам стационарной плотности
        # Retain numerical thresholds in JSON/NPZ rather than obscuring map labels.
        path = Path(output_dir) / f"fig03_{variable}_classes_{map_date}_{language}.png"
        return _save(fig, path, dpi)
    finally:
        plt.close(fig)


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
            ax.set_title(f"{variable}: region {k+1} frequency, {period[0]}–{period[1]}")
            fig.colorbar(img, ax=ax, shrink=.78, label="Fraction of valid days")
        output = Path(output_dir) / f"supp_{variable}_tail_frequency_{language}.png"
        return _save(fig, output, dpi)
    finally:
        plt.close(fig)


def _copula_contours(root, pair, period, output_dir, language, dpi):
    """Recreate Fig. 6 four individual model contours from cached rank PDFs."""
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
                contours = ax.contour(grid, grid, pdf / peak,
                                      levels=np.linspace(0.05, 0.90, 10),
                                      cmap="YlOrRd", linewidths=1.5)
                ax.clabel(contours, inline=True, fontsize=7, fmt="%.2f")
                ax.set_xlim(.02, .98)
                ax.set_ylim(.02, .98)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlabel(r"$U=F_1(X_1)$")
                ax.set_ylabel(r"$V=F_2(X_2)$")
                ax.set_title((titles_en if language == "en" else titles_ru)[name])
                path = Path(output_dir) / f"fig06_{name}_contours_{language}.png"
                outputs.append(_save(fig, path, dpi))
            finally:
                plt.close(fig)
    return outputs


def plot_article_copulas(files_path_prefix, names=("sensible", "latent"),
                         start_year=1979, end_year=2026, *, language="en", dpi=300):
    """Restore manuscript Figures 4--8 from the stored copula results.

    This does not refit a copula; the new individual contours of Fig. 6 are
    drawn separately from the same cached density arrays.
    """
    from Plotting.workflows import plot_copula_results
    root = Path(files_path_prefix)
    period = (int(start_year), int(end_year))
    dest = _figure_directory(root, period)
    existing = plot_copula_results(root, tuple(names), *period)
    mapped = {
        "copula_empirical_eng.png": "fig04_empirical_copula_eng.png",
        "copula_gaussian_eng.png": "fig04_gaussian_copula_eng.png",
        "copula_student_t_eng.png": "fig04_student_t_copula_eng.png",
        "copula_checkerboard_eng.png": "fig05_checkerboard_copula_eng.png",
        "copula_bernstein_eng.png": "fig05_bernstein_copula_eng.png",
        "gaussian_stationary_density_eng.png": "fig07_gaussian_joint_eng.png",
        "checkerboard_stationary_density_eng.png": "fig07_checkerboard_joint_eng.png",
        "bernstein_stationary_density_eng.png": "fig07_bernstein_joint_eng.png",
        "bernstein_marginal_x1_eng.png": "fig08_bernstein_marginal_sensible_eng.png",
        "bernstein_marginal_x2_eng.png": "fig08_bernstein_marginal_latent_eng.png",
    }
    outputs = []
    for item in existing:
        item = Path(item)
        if item.name in mapped:
            target = dest / mapped[item.name]
            if item.resolve() != target.resolve():
                shutil.copy2(item, target)
            outputs.append(target)
    outputs += _copula_contours(root, tuple(names), period, dest, language, dpi)
    return outputs


def plot_article_figures(files_path_prefix, names=("sensible", "latent"),
                         start_year=1979, end_year=2026, *,
                         regions_period=(1979, 1989), map_date="1979-01-01",
                         density_source="auto", hybrid_reference=(1979, 2026),
                         chunk_days=32, refresh=False, language="en", dpi=300,
                         include_copulas=False, compute_frequencies=False,
                         isoline_levels=None, second_anomaly_files=None,
                         figures=("isolines", "regions", "classes")):
    """Build article figures 1--3 (optional 4--8), all as independent PNGs."""
    if language not in LABELS:
        raise ValueError("language must be 'en' or 'ru'")
    root = Path(files_path_prefix)
    period = (int(start_year), int(end_year))
    output_dir = _figure_directory(root, period)
    caches = prepare_article_diagnostics(root, tuple(names), *period,
                                         regions_period=regions_period, map_date=map_date,
                                         density_source=density_source,
                                         hybrid_reference=hybrid_reference,
                                         chunk_days=chunk_days,
                                         compute_frequencies=compute_frequencies,
                                         overwrite=refresh)
    allowed = {"isolines", "regions", "classes"}
    figures = tuple(figures)
    if not set(figures).issubset(allowed):
        raise ValueError(f"Unknown article figure selector: {figures}")
    outputs = []
    for name, cache in caches.items():
        levels = isoline_levels.get(name) if isinstance(isoline_levels, dict) else isoline_levels
        other = second_anomaly_files.get(name) if second_anomaly_files else None
        if other is not None and not Path(other).is_absolute():
            other = root / other
        if "isolines" in figures:
            outputs.append(plot_anomaly_isolines(cache, name, period, output_dir,
                                                 language=language, dpi=dpi,
                                                 levels=levels, second_map=other))
        if "regions" in figures:
            outputs.append(plot_area_density(cache, name, regions_period, output_dir,
                                             language=language, dpi=dpi))
        if "classes" in figures:
            outputs.append(plot_class_map(cache, name, map_date, output_dir,
                                          language=language, dpi=dpi))
        if compute_frequencies:
            outputs.append(plot_tail_frequencies(cache, name, period, output_dir,
                                                  language=language, dpi=dpi))
    if include_copulas:
        outputs.extend(plot_article_copulas(root, tuple(names), *period,
                                            language=language, dpi=dpi))
    manifest = {
        "name": "Article figures", "period": list(period), "figure_language": language,
        "regions_period": list(regions_period), "map_date": map_date,
        "density_source_setting": density_source,
        "figures": [str(p) for p in outputs],
        "data_cache": {k: str(v) for k, v in caches.items()},
        "important_note": "Reference-relative anomalies; hybrid PDF is not the original FP solution.",
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[article figures] Completed: {len(outputs)} figures in {output_dir}", flush=True)
    return outputs
