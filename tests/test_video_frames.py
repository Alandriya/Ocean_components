"""Regression tests for optional geographic map captions and observation dates."""
from contextlib import ExitStack
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from Data_processing.io import start_label
from Plotting import map_workflows as maps
from Plotting.language import plot_language_context
from Plotting.plot_eigenvalues import _FrameFigure
from Plotting import stationary_plots as stationary
from Plotting import workflows


def _capture(fig, path, records):
    """Record map captions, actual plotted values, and figure-level date."""
    axes = [ax for ax in fig.axes if ax.images]
    records.append((fig._suptitle.get_text() if fig._suptitle else None,
                    [ax.get_title() for ax in axes],
                    [np.ma.array(ax.images[0].get_array()).copy() for ax in axes],
                    Path(path)))
    return Path(path)


@pytest.mark.parametrize('style', ['auto', 'original'])
@pytest.mark.parametrize('language', ['ru', 'en'])
def test_coefficient_dates_and_titles(tmp_path, monkeypatch, style, language):
    """Verify endpoint dates, skipped days, localization, and identical map data."""
    first = start_label(2020) + 58
    parent = tmp_path / 'Coefficients/continuous/2d_bins_12/sensible_latent/Nonparametric_2d'
    parent.mkdir(parents=True)
    data = np.arange(4*2*3*4, dtype=float).reshape(4, 2, 3, 4) - 40
    np.save(parent / f'A_{first}-{first+3}.npy', data)
    records = []
    monkeypatch.setattr(maps, 'save_figure', lambda fig, path, **kw: _capture(fig, path, records))
    with plot_language_context(language):
        for video in (False, True):
            maps.coefficient_frames(tmp_path, ('sensible', 'latent'), 12, 'np', 'A',
                                    start=1, stop=4, step=2, style=style, video_frames=video)
    expected = (['Коэффициент A — явный поток', 'Коэффициент A — скрытый поток'] if language == 'ru'
                else ['Coefficient A — Sensible heat flux', 'Coefficient A — Latent heat flux'])
    for plain, labeled, day in zip(records[:2], records[2:], (1, 3)):
        assert plain[0] is None and plain[1] == ['', '']
        assert labeled[0] == (date(2020, 1, 1) + timedelta(days=58+day)).strftime('%d.%m.%Y')
        assert labeled[1] == expected
        assert labeled[3].parent.name == 'video_frames'
        assert labeled[3].name == plain[3].name
        for a, b in zip(plain[2], labeled[2]):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('coefficient,channels', [('B', 4), ('C', 2), ('F', 1), ('FS', 2)])
def test_other_coefficient_titles(tmp_path, monkeypatch, coefficient, channels):
    """Check that all coefficient types have optional captions and valid dates."""
    parent = tmp_path / 'Coefficients/continuous/2d_bins_12/sensible_latent/Nonparametric_2d'
    parent.mkdir(parents=True)
    first = start_label(2024)
    np.save(parent / f'{coefficient}_{first}-{first}.npy', np.ones((1, channels, 3, 4)))
    records = []
    monkeypatch.setattr(maps, 'save_figure', lambda fig, path, **kw: _capture(fig, path, records))
    with plot_language_context('en'):
        maps.coefficient_frames(tmp_path, ('sensible', 'latent'), 12, 'np', coefficient, video_frames=True)
        maps.coefficient_frames(tmp_path, ('sensible', 'latent'), 12, 'np', coefficient)
    assert records[0][0] == '01.01.2024'
    assert all(records[0][1]) and records[1][1] == [''] * channels
    if coefficient == 'B':
        assert records[0][1] == [f'Coefficient B — {a} / {b}' for a, b in (
            ('Sensible heat flux', 'Sensible heat flux'), ('Latent heat flux', 'Latent heat flux'),
            ('Sensible heat flux', 'Latent heat flux'), ('Latent heat flux', 'Sensible heat flux'))]


def test_variable_titles(tmp_path, monkeypatch):
    """Check actual observation dates and both variable map groups."""
    records = []
    monkeypatch.setattr(maps, 'load_field', lambda *args: np.ones((3, 3, 4)))
    monkeypatch.setattr(maps, 'save_figure', lambda fig, path, **kw: _capture(fig, path, records))
    for video in (False, True):
        maps.plot_variables(tmp_path, 2024, 2025, start=1, stop=2, video_frames=video)
    assert all(rec[0] is None and not any(rec[1]) for rec in records[:2])
    assert all(rec[0] == '02.01.2024' and all(rec[1]) for rec in records[2:])


def test_spectral_frame_titles(tmp_path):
    """Check titles on the reused spectral frame and stable PNG dimensions."""
    with ExitStack() as stack:
        for video in (False, True):
            frame = _FrameFigure(stack, 1, 1, (2, 2), (2, 2), video_frames=video)
            frame.update(0, 0, np.ones((2, 2)), 'Mode 1')
            assert frame.images[0, 0].axes.get_title() == ('Mode 1' if video else '')
            sizes = []
            for day in ('01.01.2024', '02.01.2024'):
                path = tmp_path / f'{video}-{day}.png'
                frame.save(path, day)
                assert (frame.title.get_text() if frame.title else None) == (day if video else None)
                with Image.open(path) as image:
                    sizes.append(image.size)
            if video:
                assert sizes[0] == sizes[1]


def test_stationary_class_date(tmp_path, monkeypatch):
    """Use the actual cached map date for the optional stationary-map heading."""
    cache = tmp_path / 'map.npz'
    np.savez(cache, ocean_mask=np.ones((3, 4), bool), class_map=np.ones((3, 4), np.uint8),
             thresholds=np.arange(4))
    titles = []
    monkeypatch.setattr(stationary, '_save', lambda fig, path, dpi: _capture(fig, path, titles))
    for mode in (False, True):
        stationary.plot_class_map(cache, 'sensible', '2024-02-29', tmp_path, video_frames=mode)
    assert titles[0][0] is None and not any(titles[0][1])
    assert titles[1][0] == '2024-02-29' and all(titles[1][1])


def test_encoder_uses_video_directory(tmp_path):
    """Select only the labeled sequence, never unlabeled publication images."""
    base = tmp_path / 'Plots/Coefficients/A/sensible-latent/np'
    base.mkdir(parents=True)
    Image.new('RGB', (16, 16)).save(base / 'frame_00000.png')
    with pytest.raises(FileNotFoundError, match='video_frames'):
        workflows.coefficient_video(tmp_path, ('sensible', 'latent'), 'np', 'A', video_frames=True)
