"""Regression tests for stationary 2D Fokker-Planck plotting."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "fp2d_plots_under_test",
        ROOT / "Plotting" / "fokker_planck_2d_plots.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_result(path, n, shift=0.0):
    edges = np.linspace(-5.0, 5.0, n + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    xx, yy = np.meshgrid(centers, centers, indexing="ij")
    empirical = np.exp(-0.5 * (xx**2 + yy**2))
    empirical /= empirical.sum()
    stationary = np.exp(
        -0.5 * ((xx - shift) ** 2 + (yy + 0.5 * shift) ** 2)
    )
    stationary /= stationary.sum()
    area = (edges[1] - edges[0]) ** 2
    domain = np.ones((n, n), dtype=bool)

    np.savez_compressed(
        path,
        x_edges=edges,
        y_edges=edges,
        x_centers=centers,
        y_centers=centers,
        domain_mask=domain,
        stationary_mass=stationary,
        stationary_density=stationary / area,
        empirical_mass_conditioned=empirical,
        empirical_density_conditioned=empirical / area,
    )


def test_marginals_are_normalized():
    module = _load_module()
    mass = np.ones((5, 7), dtype=float)
    x_edges = np.linspace(0.0, 5.0, 6)
    y_edges = np.linspace(0.0, 14.0, 8)
    px, py = module._marginals(mass, x_edges, y_edges)
    assert np.isclose(np.sum(px * np.diff(x_edges)), 1.0)
    assert np.isclose(np.sum(py * np.diff(y_edges)), 1.0)


def test_comparison_plot_writes_both_figures(tmp_path):
    module = _load_module()
    source = tmp_path / "stationary.npz"
    _synthetic_result(source, 20, shift=0.3)

    comparison, marginals = module.plot_fp2d_stationary_comparison(
        tmp_path,
        1979,
        2026,
        stationary_file=source,
        bins=(20, 20),
        dpi=80,
    )

    assert comparison.is_file() and comparison.stat().st_size > 1000
    assert marginals.is_file() and marginals.stat().st_size > 1000


def test_grid_convergence_writes_summary(tmp_path):
    module = _load_module()
    files = {}
    for resolution, shift in ((20, 0.5), (30, 0.3), (40, 0.1)):
        source = tmp_path / f"{resolution}.npz"
        _synthetic_result(source, resolution, shift=shift)
        files[resolution] = source

    figure, summary = module.plot_fp2d_grid_convergence(
        tmp_path,
        1979,
        2026,
        resolutions=(20, 30, 40),
        stationary_files=files,
        common_resolution=200,
        dpi=80,
    )

    assert figure.is_file() and figure.stat().st_size > 1000
    data = json.loads(summary.read_text())
    assert set(data["moments"]) == {"20", "30", "40"}
    assert "20x20_vs_30x30" in data["pairwise_stationary_distances"]
    assert "30x30_vs_40x40" in data["pairwise_stationary_distances"]


def test_relative_difference_masks_low_empirical_density(tmp_path):
    module = _load_module()
    source = tmp_path / "stationary.npz"
    _synthetic_result(source, 20, shift=0.2)
    result = module._load_stationary_result(source)
    _empirical, _fp, _difference, relative, threshold = (
        module._comparison_fields(result, 0.5)
    )
    assert threshold > 0.0
    assert np.isnan(relative).any()
    assert np.isfinite(relative).any()
