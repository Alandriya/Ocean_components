"""Self-contained numerical checks for the stationary fit/density contract."""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np

MODULE = Path(__file__).resolve().parents[1] / 'Functional' / 'stationary.py'
spec = importlib.util.spec_from_file_location('stationary_under_test', MODULE)
stationary = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(stationary)


def _smooth_input(variable):
    x0, x1 = (-8.0, -50.0) if variable == 'sensible' else (-2.0, -40.0)
    limits = (-130, 40) if variable == 'sensible' else (-310, 35)
    x = np.sort(np.r_[np.linspace(*limits, 200), x0])
    truth = dict(x0=x0, x1=x1, c1=0.009, c2=0.25, c3=5.9, c4=0.29)
    y = stationary._shifted_log_variance(x, **truth)
    return x, y, truth


def test_shifted_fit_reconstructs_same_formula_as_pdf():
    for variable in ('sensible', 'latent'):
        x, y, expected = _smooth_input(variable)
        params = stationary._fit_log_diffusion(x, y, variable)
        for key, value in expected.items():
            np.testing.assert_allclose(params[key], value, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(params['rmse_log_b'], 0.0, atol=1e-10)
        predicted = stationary._shifted_log_variance(
            x, **{k: params[k] for k in expected})
        np.testing.assert_allclose(y, predicted, atol=1e-10)


def test_numeric_pdf_matches_gaussian_for_constant_diffusion():
    grid = np.linspace(-90, 90, 2500)
    # Drift -0.1*(x-5), b²=1: Gaussian mean 5 and variance 5.
    params = [-8.0, -50.0, [0.5, -0.1, 0.0, 0.0, 0.0],
              -90.0, 90.0, 0.0, 0.0, 0.0, 0.0, 1]
    pdf = stationary._stationary_density(grid, params, 'sensible')
    mean = np.trapezoid(grid * pdf, grid)
    variance = np.trapezoid((grid - mean) ** 2 * pdf, grid)
    np.testing.assert_allclose(np.trapezoid(pdf, grid), 1.0, atol=1e-10)
    np.testing.assert_allclose(mean, 5.0, atol=0.05)
    np.testing.assert_allclose(variance, 5.0, atol=0.05)


def test_numeric_pdf_uses_shifted_model_not_old_unshifted():
    grid = np.linspace(-300, 100, 4000)
    params = [-8.0, -50.0, [0.0, -0.004, 0, 0, 0],
              -250.0, 95.0, 0.01, 0.27, 5.9, 0.29, 1]
    pdf = stationary._stationary_density(grid, params, 'sensible')
    g = stationary._shifted_log_variance(
        grid, x0=-8.0, x1=-50.0, c1=0.01, c2=0.27, c3=5.9, c4=0.29)
    # Inside the trusted fit interval, the zero-current identity requires
    # d/dx[log(pdf) + log(b²)] = 2a/b².
    good = (grid > -200) & (grid < 90) & (np.abs(grid + 50) > 1) & (np.abs(grid + 8) > 1)
    derivative = np.gradient(np.log(pdf) + g, grid)
    rhs = 2 * (-0.004 * grid) * np.exp(-g)
    np.testing.assert_allclose(derivative[good], rhs[good], atol=4e-6, rtol=3e-3)


def test_json_has_drift_rmse_and_explicit_mean_kind(tmp_path, monkeypatch):
    fake_pkg = types.ModuleType('Data_processing')
    fake_pkg.__path__ = []
    fake_io = types.ModuleType('Data_processing.io')
    fake_io.start_label = lambda year: 1
    monkeypatch.setitem(sys.modules, 'Data_processing', fake_pkg)
    monkeypatch.setitem(sys.modules, 'Data_processing.io', fake_io)

    folder = tmp_path / 'Functional' / 'sensible-latent'
    folder.mkdir(parents=True)
    x_s, y_s, _ = _smooth_input('sensible')
    x_l, y_l, _ = _smooth_input('latent')
    # Expected old files are rectangular arrays, so equal number of x bins.
    assert x_s.size == x_l.size
    arrays = {
        'quantiles1': x_s,
        'quantiles2': x_l,
        'a_grouped': np.vstack((-.1 * (x_s - 5), -.12 * (x_l + 10))),
        'b_grouped_log_simple': np.vstack((y_s, y_l)),
    }
    for key, arr in arrays.items():
        np.save(folder / f'1-3653_{key}.npy', arr)
    result = stationary.estimate_stationary_characteristics(
        tmp_path, 1979, 1989, grid_points=900)
    meta = json.loads(result.with_suffix('.json').read_text(encoding='utf-8'))
    for name in ('sensible', 'latent'):
        entry = meta['fits'][name]
        assert entry['drift_rmse'] < 1e-8
        assert entry['mean_definition'].startswith('Mean under the normalized stationary PDF')
        assert entry['diffusion_model'] == 'shifted_x0_x1_piecewise_log_variance'
        assert entry['stationary_integration_method'].startswith('numeric_')
        assert 0.99999 < entry['density_integral'] < 1.00001


def test_model_id_distinguishes_old_incompatible_density_json(tmp_path, monkeypatch):
    # Load plot module while injecting the stationary module from this package.
    package = types.ModuleType('Functional')
    package.__path__ = []
    monkeypatch.setitem(sys.modules, 'Functional', package)
    monkeypatch.setitem(sys.modules, 'Functional.stationary', stationary)
    module_path = MODULE.parents[1] / 'Plotting' / 'functional_fit_plots.py'
    plot_spec = importlib.util.spec_from_file_location('functional_plot_under_test', module_path)
    plot_mod = importlib.util.module_from_spec(plot_spec)
    assert plot_spec.loader is not None
    plot_spec.loader.exec_module(plot_mod)

    folder = tmp_path / 'Functional' / 'Stationary'
    folder.mkdir(parents=True)
    meta_path = folder / 'sensible-latent_1979-2026.json'
    fit = {
        key: 1.0 for key in ('x0', 'x1', 'c1', 'c2', 'c3', 'c4')
    }
    fit['drift_polynomial_ascending'] = [0.0] * 5
    fit['diffusion_model'] = 'shifted_x0_x1_piecewise_log_variance'
    fit['stationary_integration_method'] = 'numeric_cumulative_trapezoid_of_2a_over_b2'
    meta_path.write_text(json.dumps({
        'method': 'sp', 'provenance': {'input_type': 'assembled_coefficients'},
        'fits': {'sensible': fit, 'latent': fit},
    }))
    actual = {'sensible': fit, 'latent': fit}
    assert plot_mod._check_against_saved_stationary(
        tmp_path, ('sensible', 'latent'), 1979, 2026, 'sp', actual)

    stale = json.loads(meta_path.read_text())
    stale['fits']['sensible'].pop('stationary_integration_method')
    meta_path.write_text(json.dumps(stale))
    import pytest
    with pytest.warns(RuntimeWarning, match='DOES NOT MATCH'):
        assert not plot_mod._check_against_saved_stationary(
            tmp_path, ('sensible', 'latent'), 1979, 2026, 'sp', actual)
