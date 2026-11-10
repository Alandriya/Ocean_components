"""Synthetic regression tests for the stationary 2D Fokker-Planck solver."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_solver():
    spec = importlib.util.spec_from_file_location(
        "fp2d_solver_under_test",
        ROOT / "Functional" / "fokker_planck_2d_solver.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wide_stencil_exactly_matches_local_moments():
    solver = _load_solver()
    dx = 3.0
    dy = 5.0
    a0, a1 = 2.5, -3.0
    d00, d11, d01 = 160.0, 360.0, 130.0
    candidates = solver._candidate_directions(dx, dy, 7)
    jumps, _choice = solver._local_jump_rates(
        a0, a1, d00, d11, d01,
        dx, dy, candidates, 1e-12,
    )
    first, second = solver._jump_moments(jumps, dx, dy)
    np.testing.assert_allclose(first, [a0, a1], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        second,
        [[d00, d01], [d01, d11]],
        atol=1e-12,
        rtol=1e-12,
    )
    assert all(rate >= 0 for _di, _dj, rate in jumps)


def test_generator_is_conservative_and_nonnegative_off_diagonal():
    solver = _load_solver()
    n = 13
    x = np.arange(n, dtype=float)
    y = np.arange(n, dtype=float)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    domain = np.ones((n, n), dtype=bool)
    drift = np.stack((-0.1 * (xx - 6.0), -0.1 * (yy - 6.0)))
    diffusion = np.empty((4, n, n), dtype=float)
    diffusion[0] = 8.0
    diffusion[1] = 10.0
    diffusion[2] = 2.0
    diffusion[3] = 2.0
    empirical = np.ones((n, n), dtype=float)
    empirical /= empirical.sum()

    generator, _coords, _radius, _censored, diag = solver._build_generator(
        drift,
        diffusion,
        domain,
        empirical,
        1.0,
        1.0,
        7,
        1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(generator.sum(axis=1)).ravel(),
        0.0,
        atol=1e-12,
    )
    assert diag["minimum_off_diagonal_rate"] >= 0.0
    assert diag["strongly_connected_components"] == 1
    assert diag["maximum_first_moment_reconstruction_error"] < 1e-11
    assert diag["maximum_second_moment_reconstruction_error"] < 1e-11


def test_end_to_end_stationary_solution_is_normalized_and_positive():
    solver = _load_solver()

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        n = 21
        edges = np.linspace(-10.5, 10.5, n + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        xx, yy = np.meshgrid(centers, centers, indexing="ij")

        drift = np.stack((-0.12 * xx, -0.08 * yy))
        diffusion = np.empty((4, n, n), dtype=float)
        diffusion[0] = 4.0
        diffusion[1] = 6.0
        diffusion[2] = 1.0
        diffusion[3] = 1.0

        reliable = np.ones((n, n), dtype=bool)
        occupancy = np.ones((n, n), dtype=np.int64) * 100
        empirical = np.exp(-0.03 * xx**2 - 0.02 * yy**2)
        empirical /= empirical.sum()

        phase = root / "synthetic_80x80.npz"
        np.savez_compressed(
            phase,
            x_edges=edges,
            y_edges=edges,
            x_centers=centers,
            y_centers=centers,
            drift=drift,
            diffusion=diffusion,
            occupancy=occupancy,
            reliable_mask=reliable,
            empirical_mass=empirical,
        )

        output = solver.solve_stationary_fokker_planck_2d(
            root,
            1979,
            2026,
            names=("sensible", "latent"),
            method="sp",
            bins=(n, n),
            phase_file=phase,
            max_stencil_radius=7,
            overwrite=True,
        )

        with np.load(output, allow_pickle=False) as saved:
            stationary = saved["stationary_mass"]
            domain = saved["domain_mask"]
            assert np.isclose(stationary.sum(), 1.0, atol=1e-12)
            assert np.all(stationary[domain] > 0.0)
            assert np.all(stationary[~domain] == 0.0)

        report = json.loads(output.with_suffix(".json").read_text())
        assert report["generator"]["strongly_connected_components"] == 1
        assert report["linear_solve"]["maximum_absolute_stationary_residual"] < 1e-10
