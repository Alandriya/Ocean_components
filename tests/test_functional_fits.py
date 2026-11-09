"""Offline synthetic tests of continuous diffusion fits and restored plotting."""
import importlib.util
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def model_logb2(x, x1, x0, c1, c2, c3, c4):
    d_l = c3 + c2 * np.sqrt(abs(x1)) - c1 * abs(x1)
    d_r = c3 + (c2 - c4) * np.sqrt(abs(x0))
    if x < x1:
        return c1 * abs(x) + d_l
    if x < x0:
        return c2 * np.sqrt(abs(x)) + c3
    return c4 * np.sqrt(abs(x)) + d_r


@pytest.fixture
def modules(monkeypatch):
    functional = types.ModuleType("Functional")
    functional.__path__ = []
    func_estimation = types.ModuleType("Functional.func_estimation")
    func_estimation.model_logb2 = model_logb2
    monkeypatch.setitem(sys.modules, "Functional", functional)
    monkeypatch.setitem(sys.modules, "Functional.func_estimation", func_estimation)

    spec = importlib.util.spec_from_file_location(
        "Functional.stationary", ROOT / "Functional" / "stationary.py")
    stationary = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "Functional.stationary", stationary)
    spec.loader.exec_module(stationary)
    plotting_spec = importlib.util.spec_from_file_location(
        "functional_fit_plots", ROOT / "Plotting" / "functional_fit_plots.py")
    plots = importlib.util.module_from_spec(plotting_spec)
    plotting_spec.loader.exec_module(plots)
    return stationary, plots


def synthetic_data(variable, *, noisy=False):
    if variable == "sensible":
        x = np.r_[np.linspace(-120, -0.5, 240), [0.0], np.linspace(0.5, 30, 60)]
        c1, c2, c3, c4 = 0.014, 0.255, 5.7, 0.27
    else:
        x = np.r_[np.linspace(-300, -0.5, 300), [0.0], np.linspace(0.5, 30, 60)]
        c1, c2, c3, c4 = 0.0052, 0.22, 6.05, 0.34
    y = np.array([model_logb2(v, -40, 0., c1, c2, c3, c4) for v in x])
    if noisy:
        y = y + 0.005 * np.sin(x * 0.2)
    drift = -0.25 * x - 2.0 + 0.0001 * x**2
    return x, drift, y, (c1, c2, c3, c4)


@pytest.mark.parametrize('variable', ('sensible','latent'))
def test_fit_recovers_continuous_model(modules, variable):
    stationary, plots = modules
    x, drift, y, expected = synthetic_data(variable)
    fit = stationary._fit_log_diffusion(x, y, variable)
    np.testing.assert_allclose(
        [fit['c1'], fit['c2'], fit['c3'], fit['c4']], expected,
        atol=1e-10, rtol=1e-10)
    assert fit['rmse_log_b'] < 1e-10
    branches = plots._diffusion_branches(x, fit)
    assert abs(branches[0][1][-1] - branches[1][1][0]) < 1e-12
    assert abs(branches[1][1][-1] - branches[2][1][0]) < 1e-12
    for xs, ys, _color, _label in branches:
        expected_curve = [model_logb2(float(z), fit["x1"], fit["x0"],
                                     fit["c1"], fit["c2"], fit["c3"], fit["c4"])
                          for z in xs]
        np.testing.assert_allclose(ys, expected_curve, rtol=1e-12, atol=1e-12)


def test_noisy_latent_join_is_still_exact(modules):
    stationary, plots = modules
    x, drift, y, _ = synthetic_data('latent', noisy=True)
    fit = stationary._fit_log_diffusion(x, y, 'latent')
    branch = plots._diffusion_branches(x, fit)
    assert np.isclose(branch[0][1][-1], branch[1][1][0], atol=1e-12)
    assert np.isclose(branch[1][1][-1], branch[2][1][0], atol=1e-12)


def test_historical_plot_uses_white_background(tmp_path, monkeypatch, modules):
    stationary, plots = modules
    data = {}
    for name in ('sensible','latent'):
        x, drift, y, _ = synthetic_data(name, noisy=True)
        data[name] = (x, drift, y)
    monkeypatch.setattr(plots, '_historical_grouped_inputs', lambda *a, **k: data)
    result = plots.plot_functional_fits(tmp_path, 1979, 1989)
    assert result.is_file() and result.stat().st_size > 10_000
    assert result.name == 'fit_sensible_latent_1979-1989.png'
    # The figure is closed after saving, avoiding Matplotlib global state leaks.
    assert plt.get_fignums() == []
