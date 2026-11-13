"""Minimal offline integration checks for canonical 1979--2026 paths.

Run with PYTHONPATH pointing to the repository root. Synthetic, not ERA5.
"""
from datetime import date, timedelta
from pathlib import Path
import tempfile
import os
import subprocess
import sys
import types
import numpy as np

from Forecast_nn.source_paths import (data_paths, grouped_length,
    last_available_date, scalar_sde_source, joint_sde_source)


def _write_array(path, shape, fill=0.):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=shape)
    array[:] = fill
    array.flush()
    del array


def test_data_layout_and_actual_last_date(tmp_path):
    paths = data_paths(tmp_path)
    assert paths['sensible'].name == 'sensible_1979-2026_grouped.npy'
    assert paths['latent'].name == 'latent_1979-2026_grouped.npy'
    assert paths['Flux'].parent.name == 'Fluxes'
    assert paths['SST'].parent.name == 'SST'
    assert paths['Pressure'].name == 'Pressure_1979-2026_grouped.npy'
    for name in ('sensible', 'latent'):
        _write_array(paths[name], (161*181, 8))
    assert grouped_length(paths['sensible']) == 8
    assert last_available_date(paths) == '1979-01-08'


def test_scalar_assembled_files(tmp_path):
    parent = tmp_path / 'Coefficients' / '1979-2026' / '1d_bins_144' / 'sensible' / 'Semiparametric'
    _write_array(parent / 'A_2-8.npy', (7, 161, 181), 1.)
    _write_array(parent / 'B_2-8.npy', (7, 161, 181), 2.)
    src = scalar_sde_source(tmp_path, 'sensible', field_length=8)
    assert src['a_file'].name == 'A_2-8.npy'
    assert src['b_file'].name == 'B_2-8.npy'
    assert src['time_offset'] == 1


def test_joint_canonical_layout_and_causality(tmp_path, monkeypatch):
    # The project's full Forecast_nn.data.py supplies these two constants.
    # A small stub makes this test self-contained and independent of its other deps.
    original = sys.modules.get('Forecast_nn.data')
    stub = types.ModuleType('Forecast_nn.data')
    stub.NATIVE_HEIGHT, stub.NATIVE_WIDTH = 161, 181
    monkeypatch.setitem(sys.modules, 'Forecast_nn.data', stub)
    from Forecast_nn.joint_sde import JointSDEReader, covariance_from_b
    joint = joint_sde_source(tmp_path)
    a_path, b_path = joint['a_file'], joint['b_file']
    _write_array(a_path, (7, 2, 161, 181), 0.)
    _write_array(b_path, (7, 4, 161, 181), 0.)
    b = np.lib.format.open_memmap(b_path, mode='r+')
    b[:, 0] = 2.0  # B00
    b[:, 1] = 3.0  # B11
    b[:, 2] = 0.4  # B01
    b[:, 3] = 0.5  # B10
    b.flush()
    del b
    # Simulate unchanged old CLI default. Canonical B must be auto-detected.
    reader = JointSDEReader(a_path, b_path, downsample=1, hw=(161,181),
                            state_std=[1.0,1.0], b_layout='row_major')
    assert reader.b_layout == 'diagonal_first'
    import torch
    _, bt = reader.get_normalized([1])
    assert bt.shape == (1,2,2,161,181)
    assert torch.allclose(bt[0,:,:,0,0], torch.tensor([[2.0,.4],[.5,3.0]]))
    assert np.isclose(covariance_from_b(bt)[0,0,0,0,0].item(), 4 + .4**2)
    try:
        reader.get_normalized([0])
    except IndexError:
        pass
    else:
        raise AssertionError('future-leakage: X[0] has no observed A/B')
    if original is not None:
        monkeypatch.setitem(sys.modules, 'Forecast_nn.data', original)


def test_config_uses_actual_headers_and_assembled_files(tmp_path):
    paths = data_paths(tmp_path)
    for name in ('sensible','latent'):
        _write_array(paths[name], (161*181, 8))
        folder = tmp_path / 'Coefficients' / '1979-2026' / '1d_bins_144' / name / 'Semiparametric'
        _write_array(folder / 'A_2-8.npy', (7,161,181))
        _write_array(folder / 'B_2-8.npy', (7,161,181))
    script = ('from Forecast_nn.config import cfg; '
              'assert cfg.data_last_date == "1979-01-08"; '
              'assert cfg.sde_sources["latent"]["a_file"].name == "A_2-8.npy"; '
              'assert cfg.data_files["Pressure"].name == "Pressure_1979-2026_grouped.npy"; '
              'print("Config header check OK")')
    env = dict(os.environ, OCEAN_DATA_ROOT=str(tmp_path))
    subprocess.run([sys.executable, '-c', script], env=env, check=True)


def run():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        test_data_layout_and_actual_last_date(root)
    with tempfile.TemporaryDirectory() as tmp:
        test_scalar_assembled_files(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        class Monkey:
            def setitem(self, obj, key, value): obj[key] = value
        test_joint_canonical_layout_and_causality(Path(tmp), Monkey())
    with tempfile.TemporaryDirectory() as tmp:
        test_config_uses_actual_headers_and_assembled_files(Path(tmp))
    print('4 / 4 forecast path checks passed (synthetic NPY data)')


if __name__ == '__main__':
    run()
