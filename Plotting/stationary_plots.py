"""Standalone plotting helpers for one-dimensional stationary distributions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from Data_processing.io import field_candidates, load_field

NAMES_RU = {
    "sensible": "Явный тепловой поток",
    "latent": "Скрытый тепловой поток",
    "Flux": "Полный тепловой поток",
    "SST": "Температура поверхности океана",
    "Pressure": "Атмосферное давление",
}
NAMES = {
    "sensible": "Sensible Heat Flux",
    "latent": "Latent Heat Flux",
    "Flux": "Cumulative Heat Flux",
    "SST": "Sea Surface Temperature",
    "Pressure": "Surface Pressure",
}


def plot_path(root: Path, *parts: str) -> Path:
    """Build a figure path under ``Plots`` and create parent folders."""
    path = root / "Plots"
    for part in parts[:-1]:
        path /= part
    path.mkdir(parents=True, exist_ok=True)
    return path / parts[-1]


def save_figure(fig: plt.Figure, path: Path) -> Path:
    """Save a Matplotlib figure and return its output path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return path


def _stationary_plot_config(name: str) -> dict:
    """Return legacy plotting ranges for known heat-flux variables."""
    legacy = {
        "sensible": {
            "xlim": (-300.0, 300.0),
            "hist_range": (-300.0, 300.0),
            "xlabel_ru": "Значения явного теплового потока",
            "xlabel_en": "Sensible flux values",
        },
        "latent": {
            "xlim": (-800.0, 300.0),
            "hist_range": (-800.0, 300.0),
            "xlabel_ru": "Значения скрытого теплового потока",
            "xlabel_en": "Latent flux values",
        },
    }
    if name in legacy:
        return legacy[name]
    return {
        "xlim": None,
        "hist_range": None,
        "xlabel_ru": "Значение переменной",
        "xlabel_en": "Variable value",
    }


def _histogram_cache_path(root, name, start_year, end_year, bins):
    """Return the cached observed-histogram filename."""
    folder = root / "Functional" / "Stationary" / "Histograms"
    return folder / f"{name}_{start_year}-{end_year}_{bins}bins.npz"


def _source_signature(root, name, start_year, end_year):
    """Return the path, size and modification time of the raw field source."""
    path = next(
        (
            candidate
            for candidate in field_candidates(root, name, start_year, end_year)
            if candidate.is_file()
        ),
        None,
    )
    if path is None:
        raise FileNotFoundError(
            f"Cannot locate source field for {name}, {start_year}-{end_year}"
        )
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _observed_histogram(files_path_prefix, name, start_year, end_year,
                        *, bins=100, value_range=None, chunk_days=128,
                        refresh=False):
    """Calculate or reuse a chunked observed histogram.

    The raw multidecadal field is memory-mapped and read in bounded time chunks.
    This avoids converting the complete field to one float64 array merely to
    draw a diagnostic histogram. The resulting small histogram is cached, so
    subsequent plotting runs do not rescan the source data.
    """
    root = Path(files_path_prefix)
    if bins < 1 or chunk_days < 1:
        raise ValueError("bins and chunk_days must be positive")
    if value_range is None:
        raise ValueError("A finite histogram range is required for cached plotting")

    low, high = map(float, value_range)
    if not np.isfinite([low, high]).all() or not low < high:
        raise ValueError("Invalid histogram range")

    source = _source_signature(root, name, start_year, end_year)
    cache = _histogram_cache_path(root, name, start_year, end_year, bins)
    meta_path = cache.with_suffix(".json")
    expected = {
        "name": name,
        "period": f"{start_year}-{end_year}",
        "bins": int(bins),
        "range": [low, high],
        "source": source,
    }

    if not refresh and cache.is_file() and meta_path.is_file():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata == expected:
            with np.load(cache, allow_pickle=False) as saved:
                return (
                    np.array(saved["density"], copy=True),
                    np.array(saved["edges"], copy=True),
                    cache,
                )

    field = load_field(root, name, start_year, end_year)
    edges = np.linspace(low, high, bins + 1)
    counts = np.zeros(bins, dtype=np.int64)

    for start in range(0, len(field), chunk_days):
        stop = min(len(field), start + chunk_days)
        block = np.asarray(field[start:stop])
        finite = block[np.isfinite(block)]
        if finite.size:
            counts += np.histogram(finite, bins=edges)[0]

    widths = np.diff(edges)
    total = int(counts.sum())
    if total == 0:
        raise ValueError(
            f"No finite {name} observations inside histogram range {value_range}"
        )
    density = counts.astype(np.float64) / (total * widths)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, density=density, edges=edges, counts=counts)
    meta_path.write_text(
        json.dumps(expected, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved observed histogram cache: {cache}", flush=True)
    return density, edges, cache


def plot_stationary_distributions(files_path_prefix, names, start_year, end_year,
                                  *, histogram_bins=100,
                                  histogram_chunk_days=128,
                                  refresh_histograms=False):
    """Draw stationary densities together with cached observed histograms.

    The red stationary density is loaded from the numerical stationary result.
    The observed histogram is diagnostic only and is never used to fit the
    density.
    """
    root = Path(files_path_prefix)
    source = (
        root
        / "Functional"
        / "Stationary"
        / f"{names[0]}-{names[1]}_{start_year}-{end_year}.npz"
    )
    outputs = []

    sns.set_style("whitegrid")

    with np.load(source, allow_pickle=False) as densities:
        grid = np.asarray(densities["grid"], dtype=float)

        for name in names:
            config = _stationary_plot_config(name)
            hist_density, hist_edges, hist_cache = _observed_histogram(
                root,
                name,
                start_year,
                end_year,
                bins=histogram_bins,
                value_range=config["hist_range"],
                chunk_days=histogram_chunk_days,
                refresh=refresh_histograms,
            )
            print(f"Using observed histogram cache: {hist_cache}", flush=True)

            fig, ax = plt.subplots(1, 1, figsize=(12, 7))
            ax.plot(
                grid,
                densities[f"pdf_{name}"],
                c="r",
                linewidth=2.0,
                label="Stationary density",
            )
            ax.stairs(
                hist_density,
                hist_edges,
                fill=True,
                alpha=0.5,
                label="Observed histogram",
            )

            if config["xlim"] is not None:
                ax.set_xlim(*config["xlim"])

            ax.set_title(NAMES.get(name, name), fontsize=16)
            ax.set_xlabel(config["xlabel_en"], fontsize=14)
            ax.set_ylabel("Probability density", fontsize=14)
            ax.legend()
            fig.tight_layout()

            output = save_figure(
                fig,
                plot_path(
                    root,
                    "Distributions",
                    "Stationary",
                    f"{name}_{start_year}-{end_year}_eng.png",
                ),
            )
            plt.close(fig)
            outputs.append(output)

    return outputs
