"""Synthetic tests for two-dimensional Fokker-Planck phase-grid preparation."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "fp2d_under_test",
        ROOT / "Functional" / "fokker_planck_2d.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase_eigenvalues_for_psd_matrix():
    module = _load_module()
    d = np.empty((4, 2, 3), dtype=float)
    d[0] = 4.0
    d[1] = 9.0
    d[2] = 1.0
    d[3] = 1.0
    eig = module._phase_eigenvalues(d)
    expected = np.linalg.eigvalsh(np.array([[4.0, 1.0], [1.0, 9.0]]))
    np.testing.assert_allclose(eig[0], expected[0])
    np.testing.assert_allclose(eig[1], expected[1])


def test_block_mean_uses_shared_valid_mask():
    module = _load_module()
    values = np.arange(16.0).reshape(1, 4, 4)
    valid = np.ones_like(values, dtype=bool)
    valid[:, 0, 0] = False
    result = module._block_mean(values, valid, 2)
    assert result.shape == (1, 2, 2)
    assert np.isclose(result[0, 0, 0], np.mean([1.0, 4.0, 5.0]))


def test_end_to_end_phase_grid_and_time_alignment(monkeypatch):
    module = _load_module()

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        period = "1979-1989"
        pair = "sensible_latent"
        coeff = root / "Coefficients" / period / pair
        coeff.mkdir(parents=True)

        t = 6
        h = w = 2
        transitions = t - 1

        sensible = np.arange(t * h * w, dtype=float).reshape(t, h, w)
        latent = 2.0 * sensible - 10.0

        a = np.zeros((transitions, 2, h, w), dtype=float)
        bbt = np.zeros((transitions, 4, h, w), dtype=float)

        # Encode the previous state into drift to catch a one-step alignment shift.
        a[:, 0] = sensible[:-1]
        a[:, 1] = latent[:-1]
        bbt[:, 0] = 4.0
        bbt[:, 1] = 9.0
        bbt[:, 2] = 1.0
        bbt[:, 3] = 1.0

        sensible_path = root / "sensible.npy"
        latent_path = root / "latent.npy"
        mask_path = root / "mask.npy"
        np.save(sensible_path, sensible)
        np.save(latent_path, latent)
        np.save(mask_path, np.ones((h, w), dtype=np.uint8))
        np.save(coeff / f"A_{period}_sp.npy", a)
        np.save(coeff / f"BBT_{period}_sp.npy", bbt)

        dp = types.ModuleType("Data_processing")
        io = types.ModuleType("Data_processing.io")
        io.field_candidates = lambda _root, name, _sy, _ey: [
            sensible_path if name == "sensible" else latent_path
        ]
        io.load_field = lambda _root, name, _sy, _ey: (
            np.load(sensible_path, mmap_mode="r")
            if name == "sensible"
            else np.load(latent_path, mmap_mode="r")
        )
        io.read_ocean_mask = lambda _root: mask_path

        coeff_pkg = types.ModuleType("Coefficients")
        common = types.ModuleType("Coefficients.common")
        common.load_water_mask = lambda source, shape: np.load(source).reshape(shape).astype(bool)

        monkeypatch.setitem(sys.modules, "Data_processing", dp)
        monkeypatch.setitem(sys.modules, "Data_processing.io", io)
        monkeypatch.setitem(sys.modules, "Coefficients", coeff_pkg)
        monkeypatch.setitem(sys.modules, "Coefficients.common", common)

        result = module.prepare_phase_space_coefficients(
            root,
            1979,
            1989,
            names=("sensible", "latent"),
            method="sp",
            bins=(4, 4),
            quantile_bounds=(0.0, 1.0),
            min_count=1,
            chunk_days=2,
            spatial_block_size=1,
            sample_size=1000,
            overwrite=True,
        )

        with np.load(result, allow_pickle=False) as saved:
            diffusion = saved["diffusion"]
            reliable = saved["reliable_mask"]
            assert reliable.any()
            np.testing.assert_allclose(diffusion[0][reliable], 4.0)
            np.testing.assert_allclose(diffusion[1][reliable], 9.0)
            np.testing.assert_allclose(diffusion[2][reliable], 1.0)
            np.testing.assert_allclose(diffusion[3][reliable], 1.0)

        metadata = json.loads(result.with_suffix(".json").read_text())
        assert metadata["diagnostics"]["psd_violations"] == 0
        assert metadata["time_alignment"]["first_state_date"] == "1979-01-01"
        assert metadata["time_alignment"]["last_observation_date"] == "1979-01-06"
