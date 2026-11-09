"""Regression tests for the stationary dependency and B-branch workflow."""

from __future__ import annotations

from pathlib import Path
import importlib.util

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_stationary():
    spec = importlib.util.spec_from_file_location(
        "stationary_under_test",
        ROOT / "Functional" / "stationary.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sensible_right_branch_has_minimum_at_x0():
    stationary = _load_stationary()
    x = np.linspace(-125.0, 30.0, 251)
    expected = {
        "x0": -6.0,
        "x1": -40.0,
        "c1": 0.014,
        "c2": 0.31,
        "c3": 4.8,
        "c4": 0.36,
    }
    y = stationary._evaluate_log_diffusion(x, expected)
    fit = stationary._fit_log_diffusion(x, y, "sensible")

    x0 = fit["x0"]
    at_join = stationary._evaluate_log_diffusion(np.array([x0]), fit)[0]
    at_zero = stationary._evaluate_log_diffusion(np.array([0.0]), fit)[0]
    at_positive = stationary._evaluate_log_diffusion(np.array([20.0]), fit)[0]

    assert np.isclose(at_join, fit["observed_minimum_value"], atol=1e-12)
    assert at_zero > at_join
    assert at_positive > at_join


def test_stationary_density_is_normalized_from_fitted_dependencies():
    stationary = _load_stationary()
    x = np.linspace(-125.0, 30.0, 251)
    diffusion = {
        "x0": -6.0,
        "x1": -40.0,
        "c1": 0.014,
        "c2": 0.31,
        "c3": 4.8,
        "c4": 0.36,
    }
    drift_coeffs = np.array([-2.0, -0.23, 0.0, 0.0, 0.0])
    grid = np.linspace(-500.0, 300.0, 3000)

    density = stationary._stationary_density(
        grid,
        drift_coeffs,
        diffusion,
        "sensible",
        float(x.min()),
        float(x.max()),
    )

    assert np.all(np.isfinite(density))
    assert np.all(density >= 0.0)
    assert np.isclose(np.trapezoid(density, grid), 1.0, atol=1e-10)


def test_saved_dependencies_roundtrip(tmp_path):
    stationary = _load_stationary()
    names = ("sensible", "latent")
    inputs = {
        "sensible": (
            np.array([-2.0, -1.0, 0.0]),
            np.array([1.0, 0.0, -1.0]),
            np.array([4.0, 3.0, 4.0]),
        ),
        "latent": (
            np.array([-3.0, -1.0, 1.0]),
            np.array([2.0, 0.0, -2.0]),
            np.array([5.0, 4.0, 5.0]),
        ),
    }

    path = stationary._save_stationary_dependencies(
        tmp_path,
        names,
        1979,
        2026,
        "sp",
        inputs,
        {"test": True},
    )
    loaded, loaded_path = stationary.load_stationary_dependencies(
        tmp_path,
        names,
        1979,
        2026,
        method="sp",
    )

    assert loaded_path == path
    for name in names:
        for actual, expected in zip(loaded[name], inputs[name]):
            np.testing.assert_array_equal(actual, expected)
