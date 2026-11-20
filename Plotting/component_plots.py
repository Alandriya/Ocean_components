"""Geographic views and temporal tracks of cached spatial mixture components."""
from __future__ import annotations

from datetime import date
from pathlib import Path
import json

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from tqdm.auto import tqdm

from Coefficients.spatial_components import component_run_parts


DRIFT_CMAP = LinearSegmentedColormap.from_list('component_drift', ['#1f3a93', '#ffffff', '#c1121f'], N=257)
LAND_CMAP = ListedColormap(['lightgreen'])
LAND_CMAP.set_bad((0, 0, 0, 0))
NAMES = {'sensible': ('Sensible heat flux', 'Явный тепловой поток'),
         'latent': ('Latent heat flux', 'Скрытый тепловой поток'),
         'Flux': ('Total heat flux', 'Суммарный тепловой поток'),
         'SST': ('Sea surface temperature', 'Температура поверхности моря'),
         'Pressure': ('Atmospheric pressure', 'Атмосферное давление')}


def _map(ax, water, data, title, cmap, norm, *, ru, show_y=True):
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad('#d7d7d7')  # Missing ocean data is not land.
    image = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, norm=norm,
                      extent=(-90., 0., 0., 80.), origin='upper',
                      interpolation='none', aspect='equal')
    land = np.ma.masked_where(water, np.ones(water.shape))
    ax.imshow(land, cmap=LAND_CMAP, vmin=0, vmax=1, origin='upper',
              extent=(-90., 0., 0., 80.), interpolation='none', aspect='equal')
    ax.set_xticks([-90,-60,-30,0], ['90° з. д.', '60° з. д.', '30° з. д.', '0'] if ru else ['90W','60W','30W','0'])
    ax.set_yticks([0,30,60,80], ['0','30° с. ш.','60° с. ш.','80° с. ш.'] if ru else ['0','30N','60N','80N'])
    ax.tick_params(axis='both', labelsize=9, pad=2)
    ax.tick_params(axis='y', labelleft=show_y)
    ax.grid(True, color='#bcbcbc', linewidth=.65, alpha=.5)
    ax.set_title(title, fontsize=12)
    return image


def _norm_limits(paths, drift_limit, diffusion_limit, corrected):
    amax, bmax = 0., 0.
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            for key, signed in (('a', True), ('b_corrected' if corrected else 'b', False)):
                x = data[key]
                x = x[np.isfinite(x)]
                if x.size:
                    if signed:
                        amax = max(amax, float(np.max(np.abs(x))))
                    else:
                        bmax = max(bmax, float(np.max(x)))
    amax = (amax or 1.) if drift_limit is None else float(drift_limit)
    bmax = (bmax or 1.) if diffusion_limit is None else float(diffusion_limit)
    if not np.isfinite(amax) or amax <= 0 or not np.isfinite(bmax) or bmax <= 0:
        raise ValueError('Color limits must be finite and positive')
    return TwoSlopeNorm(vmin=-amax, vcenter=0., vmax=amax), Normalize(vmin=0., vmax=bmax)


def _masked_components(data, key, supported_only):
    maps = np.asarray(data[key], dtype=float).copy()
    if supported_only:
        groups = data['bin_map']
        valid = groups >= 0
        keep = np.zeros(groups.shape, dtype=bool)
        keep[valid] = data['supported'][groups[valid]]
        maps[:, ~keep] = np.nan
    return maps


def _panel_figure(path, water, top, bottom, top_name, bottom_name, top_norm, *,
                  top_cmap, caption, ru, dpi):
    fig, axes = plt.subplots(2, 2, figsize=(12.4,10.8), constrained_layout=True)
    try:
        for row, values, norm, cmap, prefix in ((0,top,top_norm,top_cmap,top_name),
                                               (1,bottom,Normalize(0,1),'plasma',bottom_name)):
            for k, ax in enumerate(axes[row]):
                im = _map(ax, water, values[k], f'{prefix} {k+1}', cmap, norm, ru=ru, show_y=(k==0))
                cax = make_axes_locatable(ax).append_axes('right', size='4.2%', pad=.10)
                cb = fig.colorbar(im, cax=cax)
                cb.ax.tick_params(labelsize=9, pad=2)
        fig.suptitle(caption, fontsize=14)
        fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=.12)
    finally:
        plt.close(fig)
    return path


def _plot_tracks(paths, path, group, *, ru, dpi, corrected):
    # Compact arrays only; no loading raw data or full map sequence.
    days, arrays, flags, status = [], [], [], []
    for file in paths:
        with np.load(file, allow_pickle=False) as d:
            days.append(date.fromisoformat(str(d['endpoint_date'].item())))
            arrays.append(np.stack((d['a'][group], d['b_corrected' if corrected else 'b'][group], d['weight'][group])))
            flags.append(bool(d['supported'][group]))
            status.append(int(d['tracking'][group]))
    values = np.asarray(arrays)
    flags, status = np.asarray(flags), np.asarray(status)
    fig, axes = plt.subplots(3, 1, figsize=(12,8), sharex=True, constrained_layout=True)
    try:
        for i, ax in enumerate(axes):
            for k, color in enumerate(('#3056a0','#c1121f')):
                x = values[:, i, k]
                label = ('Компонента' if ru else 'Component') + f' {k+1}'
                ax.scatter(np.array(days)[flags], x[flags], color=color, s=22, label=label)
                ax.scatter(np.array(days)[~flags], x[~flags], color=color, s=25, marker='x', alpha=.5)
                for j in range(1,len(days)):
                    if flags[j-1] and flags[j] and status[j] == 2:
                        ax.plot(days[j-1:j+1], x[j-1:j+1], color=color, lw=1.2)
            ax.set_ylabel((['Дрейф A','Диффузия B','Вес p'] if ru else ['Drift A','Diffusion B','Weight p'])[i])
            ax.grid(alpha=.3)
        axes[0].legend(loc='best')
        axes[-1].set_ylim(-.03,1.03)
        locator = mdates.AutoDateLocator()
        axes[-1].xaxis.set_major_locator(locator)
        axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        title = (f'Квантильная группа {group+1}: эволюция компонент (× — слабая поддержка)' if ru
                 else f'Quantile group {group+1}: component tracks (× — weak support)')
        fig.suptitle(title)
        fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=.12)
    finally:
        plt.close(fig)
    return path


def plot_components(files_path_prefix, start_year, end_year, *, variable='sensible',
                    n_bins=10, window_days=30, step_days=7, first_date='2019-01-01',
                    last_date='2019-03-31', spatial_stride=1, plot_first_date=None,
                    plot_last_date=None, plot_diffusion=True, corrected_diffusion=False,
                    supported_only=False, drift_limit=None, diffusion_limit=None,
                    track_group=5, language='ru', dpi=200, progress=True):
    """Render paired drift/posterior maps, optional diffusion/weight maps and tracks.

    One shared scale per physical quantity across panels AND windows.
    The time track is per quantile rank, not a trajectory of a fixed location.
    """
    root = Path(files_path_prefix)
    parts = component_run_parts(start_year,end_year,variable,n_bins,window_days,step_days,
                                first_date,last_date,spatial_stride)
    source = root/'Coefficients'/'Components'/Path(*parts)
    meta = source/'run.json'
    if not meta.is_file():
        raise FileNotFoundError(f'No component run: {meta}. Run task="components" in main.py first.')
    cfg = json.loads(meta.read_text(encoding='utf-8'))
    paths = [source/name for name in cfg['windows']]
    if not paths or any(not p.is_file() for p in paths):
        raise FileNotFoundError('Component run is incomplete; finish the numerical task first.')
    lo = date.fromisoformat(str(plot_first_date or first_date))
    hi = date.fromisoformat(str(plot_last_date or last_date))
    selected = [p for p in paths if lo <= date.fromisoformat(p.stem.removeprefix('window_')) <= hi]
    if not selected:
        raise ValueError('No saved component windows in the requested plot dates')
    if track_group is not None and not 1 <= track_group <= n_bins:
        raise ValueError(f'track_group must lie in 1..{n_bins}, or be None')
    ru = language in ('ru','rus')
    if not ru and language not in ('en','eng'):
        raise ValueError('language must be ru or en')
    anorm,bnorm = _norm_limits(paths,drift_limit,diffusion_limit,corrected_diffusion)
    dest = root/'Plots'/'Components'/Path(*parts)/('ru' if ru else 'en')
    if supported_only:
        dest = dest/'supported_only'
    dest.mkdir(parents=True,exist_ok=True)
    results = []
    for path in tqdm(selected,desc='Component maps',unit='window',disable=not progress):
        with np.load(path,allow_pickle=False) as data:
            day = str(data['endpoint_date'].item())
            since = str(data['window_first_date'].item())
            water = data['water_mask'].astype(bool)
            n_supported = int(data['supported'].sum())
            name = NAMES[variable][int(ru)]
            caption = (f'{name} — {since} → {day}\nГруппы с поддержкой двух компонент: {n_supported}/{n_bins}' if ru else
                       f'{name} — {since} → {day}\nGroups supporting two components: {n_supported}/{n_bins}')
            results.append(_panel_figure(dest/f'Drift_{day}.png', water,
                _masked_components(data,'A',supported_only), _masked_components(data,'Posterior',supported_only),
                'Дрейф A' if ru else 'Drift A', 'Вероятность r' if ru else 'Posterior r', anorm,
                top_cmap=DRIFT_CMAP, caption=caption, ru=ru,dpi=dpi))
            if plot_diffusion:
                key = 'B_corrected' if corrected_diffusion else 'B'
                suffix = '_corrected' if corrected_diffusion else ''
                results.append(_panel_figure(dest/f'Diffusion{suffix}_{day}.png', water,
                    _masked_components(data,key,supported_only), _masked_components(data,'P',supported_only),
                    'Диффузия B' if ru else 'Diffusion B', 'Вес p' if ru else 'Mixture weight p', bnorm,
                    top_cmap='plasma',caption=caption,ru=ru,dpi=dpi))
    if track_group is not None:
        suffix = '_corrected' if corrected_diffusion else ''
        results.append(_plot_tracks(selected,dest/f'Tracks_group{track_group}{suffix}_{lo}_{hi}.png',track_group-1,
                                    ru=ru,dpi=dpi,corrected=corrected_diffusion))
    print(f'[components] Saved {len(results)} plots: {dest}',flush=True)
    return results
