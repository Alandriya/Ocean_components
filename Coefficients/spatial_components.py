"""Experimental scalar, spatially grouped moving Gaussian mixtures.

Daily groups are exactly those of prepare_transition (previous state ranks).
Pool raw increments of the same rank over a trailing window of transitions.
Track two components independently for each rank across adjacent windows.
This is not a replacement for estimate_sp or a fitted switching SDE.
"""
from __future__ import annotations

from collections import deque
from datetime import date, timedelta
from pathlib import Path
import csv
import hashlib
import json
import os
import tempfile

import numpy as np
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

from Coefficients.common import prepare_transition, load_water_mask
from Data_processing.io import field_candidates, load_field, read_ocean_mask, start_label


def component_run_parts(start_year, end_year, variable, n_bins, window_days, step_days,
                        first_date, last_date, spatial_stride=1):
    if variable not in ('sensible', 'latent', 'Flux', 'SST', 'Pressure'):
        raise ValueError(f'Unknown scalar variable: {variable}')
    for value in (n_bins, window_days, step_days, spatial_stride):
        if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < 1:
            raise ValueError('Bins, window, step and stride must be positive integers')
    if step_days > window_days:
        raise ValueError('step_days must be <= window_days for adjacent tracking')
    first, last = date.fromisoformat(str(first_date)), date.fromisoformat(str(last_date))
    if first > last or (last - first).days < window_days:
        raise ValueError('Date interval must contain at least window_days transitions')
    return (f'{start_year}-{end_year}', variable, f'1d_bins_{n_bins}',
            f'k2_window{window_days}_step{step_days}_stride{spatial_stride}',
            f'{first.isoformat()}_{last.isoformat()}')


def match_components(means, stds, weights, previous=None):
    """Minimum-cost bijection for K=2 using normalized (mu,sigma) distance.

    Weights are not used as a distance coordinate, following the clustering
    rationale in the article. A weak or missing previous fit resets labels.
    Returns permutation, status (1=initial/reset, 2=matched), cost, ambiguity.
    """
    if previous is None:
        return np.lexsort((stds, means)), 1, np.nan, np.nan
    old = np.asarray(previous, dtype=float)
    new = np.column_stack((means, stds))
    scale = max(float(np.max(stds)), float(np.max(old[:, 1])),
                float(np.ptp(np.r_[means, old[:, 0]])), 1e-12)
    distances = np.sum(((old[:, None, :] - new[None, :, :]) / scale) ** 2, axis=2)
    costs = np.array([distances[0, 0] + distances[1, 1],
                      distances[0, 1] + distances[1, 0]])
    choice = int(np.argmin(costs))
    return np.array([0, 1] if choice == 0 else [1, 0]), 2, float(np.sqrt(costs[choice])), float(abs(costs[0]-costs[1]))


def _fit(y, k, n_init, reg_covar, seed):
    # EM as in semiparametric.py; k-means starts help avoid symmetric collapse.
    return GaussianMixture(n_components=k, covariance_type='full', tol=1e-4,
                           max_iter=1000, init_params='kmeans', n_init=n_init,
                           random_state=seed, reg_covar=reg_covar).fit(y)


def fit_component_window(grouped_days, water, *, n_bins=10, spatial_stride=1,
                         dt=1., n_init=10, reg_covar=1e-6, min_group_size=100,
                         min_weight=0.05, min_effective_count=20, bic_threshold=10.,
                         seed=42, previous=None):
    """Fit a two-component mixture per rank across a window.

    Maps are evaluated at the final transition, and posterior probabilities
    condition on the observed final increment (retrospective, not a forecast).
    All fitted groups are saved; support is a diagnostic, not a proof.
    B is the regularized within-component diffusion sqrt(var/dt); the corrected
    B is also stored. A_mix/B_mix reproduce the fitted mixture moments.
    """
    if not grouped_days:
        raise ValueError('Empty window')
    shape = tuple(grouped_days[-1].shape)
    water = load_water_mask(water, shape)
    if any(g.shape != shape or g.dimension != 1 or g.n_requested_bins != n_bins for g in grouped_days):
        raise ValueError('Inconsistent scalar daily groups')
    if (not np.isfinite(dt) or dt <= 0 or not np.isfinite(reg_covar) or reg_covar <= 0
            or n_init < 1 or min_group_size < 2 or not 0 < min_weight <= .5
            or min_effective_count < 1 or not np.isfinite(bic_threshold) or spatial_stride < 1):
        raise ValueError('Invalid mixture settings')
    if previous is not None and np.asarray(previous).shape != (n_bins, 2, 2):
        raise ValueError('Wrong previous component shape')
    # Retain the full daily rank boundaries even for a thinned fitting sample.
    chunks = [[] for _ in range(n_bins)]
    all_counts = np.zeros(n_bins, dtype=np.int64)
    for daily in grouped_days:
        for g in daily.groups:
            idx = g.flat_indices
            inc = np.asarray(g.increments).reshape(-1)
            all_counts[g.bin_id] += len(inc)
            if spatial_stride > 1:
                selected = ((idx // shape[1]) % spatial_stride == 0) & ((idx % shape[1]) % spatial_stride == 0)
                inc = inc[selected]
            if len(inc):
                chunks[g.bin_id].append(inc)
    end_groups = {g.bin_id: g for g in grouped_days[-1].groups}
    params = {key: np.full((n_bins, 2), np.nan) for key in ('mu', 'sigma', 'a', 'b', 'b_corrected', 'weight', 'effective_count')}
    diag = {key: np.full(n_bins, np.nan) for key in ('bic1', 'bic2', 'delta_bic', 'separation', 'match_cost', 'match_margin')}
    n_fit = np.zeros(n_bins, dtype=np.int64)
    converged = np.zeros(n_bins, dtype=bool)
    supported = np.zeros(n_bins, dtype=bool)
    tracking = np.zeros(n_bins, dtype=np.int8)
    maps = {key: np.full((2, *shape), np.nan) for key in ('A', 'B', 'B_corrected', 'P', 'Posterior')}
    mix_a, mix_b, mix_b_reg = (np.full(shape, np.nan) for _ in range(3))
    bin_map = np.full(shape, -1, dtype=np.int32)
    for gid, g in end_groups.items():
        bin_map.ravel()[g.flat_indices] = gid
    next_previous = np.full((n_bins, 2, 2), np.nan)
    for gid, sample in enumerate(chunks):
        if not sample:
            continue
        y = np.concatenate(sample).reshape(-1, 1)
        n_fit[gid] = len(y)
        if len(y) < min_group_size or np.ptp(y) == 0:
            continue
        gm1 = _fit(y, 1, 1, reg_covar, (seed+gid) % (2**32-1))
        gm = _fit(y, 2, n_init, reg_covar, (seed+gid) % (2**32-1))
        mu, var, w = gm.means_[:, 0], gm.covariances_[:, 0, 0], gm.weights_
        std = np.sqrt(var)
        old = None if previous is None or not np.isfinite(previous[gid]).all() else previous[gid]
        order, status, cost, margin = match_components(mu, std, w, old)
        mu, var, std, w = mu[order], var[order], std[order], w[order]
        diag['bic1'][gid], diag['bic2'][gid] = gm1.bic(y), gm.bic(y)
        diag['delta_bic'][gid] = diag['bic1'][gid]-diag['bic2'][gid]
        diag['separation'][gid] = abs(mu[0]-mu[1])/np.sqrt(np.mean(var))
        diag['match_cost'][gid], diag['match_margin'][gid] = cost, margin
        converged[gid] = gm.converged_ and gm1.converged_
        supported[gid] = (converged[gid] and np.min(w) >= min_weight and
                          np.min(w)*len(y) >= min_effective_count and
                          diag['delta_bic'][gid] >= bic_threshold)
        tracking[gid] = status
        params['mu'][gid], params['sigma'][gid] = mu, std
        params['a'][gid], params['b'][gid] = mu/dt, std/np.sqrt(dt)
        params['b_corrected'][gid] = np.sqrt(np.maximum(var-reg_covar, 0.)/dt)
        params['weight'][gid], params['effective_count'][gid] = w, w*len(y)
        if supported[gid]:
            next_previous[gid] = np.column_stack((mu, std))
        group = end_groups.get(gid)
        if group is None:
            continue
        idx = group.flat_indices
        for dest, key in (('A', 'a'), ('B', 'b'), ('B_corrected', 'b_corrected'), ('P', 'weight')):
            maps[dest].reshape(2,-1)[:, idx] = params[key][gid,:,None]
        maps['Posterior'].reshape(2,-1)[:, idx] = gm.predict_proba(group.increments.reshape(-1,1))[:,order].T
        mean = np.dot(w, mu)
        variance = np.dot(w, var+(mu-mean)**2)
        mix_a.ravel()[idx] = mean/dt
        mix_b.ravel()[idx] = np.sqrt(max(0., variance-reg_covar)/dt)
        mix_b_reg.ravel()[idx] = np.sqrt(variance/dt)
    edges = np.full((len(grouped_days), n_bins+1), np.nan)
    for i, day in enumerate(grouped_days):
        if len(day.edges) == n_bins+1:
            edges[i] = day.edges
    return {**maps, **params, **diag, 'A_mix': mix_a, 'B_mix': mix_b, 'B_with_reg': mix_b_reg,
            'water_mask': water, 'bin_map': bin_map, 'n_fit': n_fit, 'n_all': all_counts,
            'converged': converged, 'supported': supported, 'tracking': tracking,
            'next_previous': next_previous, 'daily_edges': edges}


def _atomic_npz(path, **data):
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as f:
        temp = Path(f.name)
    try:
        np.savez_compressed(temp, **data)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _atomic_text(path, content):
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False, mode='w', encoding='utf-8', newline='') as f:
        temp = Path(f.name)
        f.write(content)
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def run_components(files_path_prefix, start_year, end_year, *, variable='sensible',
                   n_bins=10, window_days=30, step_days=7, first_date='2019-01-01',
                   last_date='2019-03-31', spatial_stride=1, dt=1., n_init=10,
                   reg_covar=1e-6, min_group_size=100, min_weight=.05,
                   min_effective_count=20, bic_threshold=10., random_state=42,
                   mask=None, overwrite=False, progress=True):
    """Run sliding windows consecutively; output per-window NPZ and compact CSV."""
    root = Path(files_path_prefix)
    parts = component_run_parts(start_year, end_year, variable, n_bins, window_days,
                                step_days, first_date, last_date, spatial_stride)
    source = next((p for p in field_candidates(root, variable, start_year, end_year) if p.is_file()), None)
    if source is None:
        raise FileNotFoundError(f'Missing {variable} field for {start_year}-{end_year}')
    print(f'[components] Loading {source}', flush=True)
    fields = load_field(root, variable, start_year, end_year)
    water = load_water_mask(read_ocean_mask(root, fields.shape[1:]) if mask is None else mask, fields.shape[1:])
    if not water.any():
        raise ValueError('No ocean cells')
    first, last = date.fromisoformat(str(first_date)), date.fromisoformat(str(last_date))
    origin = date(start_year, 1, 1)
    begin, finish = (first-origin).days, (last-origin).days
    if begin < 0 or finish >= len(fields):
        raise ValueError(f'Requested {first}..{last}; available {origin}..{origin+timedelta(days=len(fields)-1)}')
    if (not np.isfinite(dt) or dt <= 0 or not np.isfinite(reg_covar) or reg_covar <= 0
            or n_init < 1 or min_group_size < 2 or not 0 < min_weight <= .5
            or min_effective_count < 1 or not np.isfinite(bic_threshold) or not 0 <= random_state < 2**32-1):
        raise ValueError('Invalid EM settings')
    dest = root / 'Coefficients' / 'Components' / Path(*parts)
    dest.mkdir(parents=True, exist_ok=True)
    stat = source.stat()
    metadata = dict(version=1, variable=variable, period=[start_year,end_year],
                    first_date=str(first), last_date=str(last), n_bins=n_bins,
                    window_days=window_days, step_days=step_days, spatial_stride=spatial_stride,
                    dt=dt, n_init=n_init, reg_covar=reg_covar, min_group_size=min_group_size,
                    min_weight=min_weight, min_effective_count=min_effective_count,
                    bic_threshold=bic_threshold, random_state=random_state,
                    source_size=stat.st_size, source_mtime_ns=stat.st_mtime_ns,
                    mask_sha256=hashlib.sha256(water.tobytes()).hexdigest(),
                    shape=list(fields.shape), dtype=str(fields.dtype),
                    grouping='daily previous-state quantile rank; trailing complete transitions',
                    matching='minimum squared distance (mu,sigma), previous supported rank only',
                    init='kmeans', label_convention='endpoint', n_components=2)
    ends = list(range(begin+window_days, finish+1, step_days))
    metadata['windows'] = [f'window_{origin+timedelta(days=t)}.npz' for t in ends]
    meta_path = dest/'run.json'
    if meta_path.exists() and not overwrite and json.loads(meta_path.read_text(encoding='utf-8')) != metadata:
        raise ValueError(f'Changed input/configuration: {meta_path}. Set components_overwrite=True to replace this run.')
    if not meta_path.exists() and not overwrite and any(dest.glob('window_*.npz')):
        raise ValueError(f'Unverified existing windows: {dest}; set overwrite=True')
    if overwrite:
        for p in dest.glob('window_*.npz'):
            p.unlink()
        (dest/'parameters.csv').unlink(missing_ok=True)
    _atomic_text(meta_path, json.dumps(metadata, ensure_ascii=False, indent=2))
    previous = None
    rolling = deque(maxlen=window_days)
    last_loaded = begin
    rows = []
    dirty = overwrite
    print(f'[components] {variable}: {len(ends)} windows; {n_bins} groups, K=2; {first}..{last}', flush=True)
    with threadpool_limits(limits=1):
        for end, filename in tqdm(zip(ends, metadata['windows']), total=len(ends), desc='Spatial components', unit='window', disable=not progress):
            path = dest/filename
            data = None
            if not dirty and path.is_file():
                try:
                    with np.load(path, allow_pickle=False) as stored:
                        data = {k: stored[k] for k in ('next_previous','a','b','b_corrected','weight','mu','sigma','effective_count','bic1','bic2','delta_bic','separation','match_cost','match_margin','n_fit','n_all','supported','converged','tracking')}
                    if data['next_previous'].shape != (n_bins,2,2) or data['a'].shape != (n_bins,2):
                        raise ValueError('Invalid component shape')
                except (OSError, ValueError, KeyError, EOFError):
                    data = None
            if data is None:
                dirty = True  # subsequent identities depend on this fit; never reuse stale later fits
                needed = end-window_days+1
                if last_loaded < needed-1:
                    rolling.clear()
                    last_loaded = needed-1
                for t in range(last_loaded+1, end+1):
                    rolling.append(prepare_transition(fields[t-1], fields[t], n_bins, water))
                last_loaded = end
                if len(rolling) != window_days:
                    raise RuntimeError('Incomplete rolling window')
                data = fit_component_window(list(rolling), water, n_bins=n_bins,
                      spatial_stride=spatial_stride, dt=dt, n_init=n_init, reg_covar=reg_covar,
                      min_group_size=min_group_size, min_weight=min_weight,
                      min_effective_count=min_effective_count, bic_threshold=bic_threshold,
                      seed=(random_state+1000*(start_label(start_year)+end))%(2**32-1), previous=previous)
                _atomic_npz(path, **data, endpoint_label=start_label(start_year)+end,
                            window_first_date=str(origin+timedelta(days=end-window_days)),
                            endpoint_date=str(origin+timedelta(days=end)))
            previous = data['next_previous']
            for g in range(n_bins):
                for k in range(2):
                    row = dict(window_first_date=str(origin+timedelta(days=end-window_days)),
                               endpoint_date=str(origin+timedelta(days=end)),
                               group=g+1, component=k+1)
                    for key in ('a','b','b_corrected','weight','mu','sigma','effective_count'):
                        row[key] = float(data[key][g,k])
                    for key in ('bic1','bic2','delta_bic','separation','match_cost','match_margin','n_fit','n_all','supported','converged','tracking'):
                        row[key] = data[key][g].item()
                    rows.append(row)
            if progress:
                tqdm.write(f"{origin+timedelta(days=end)}: {int(data['supported'].sum())}/{n_bins} groups supported")
    import io
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    _atomic_text(dest/'parameters.csv', stream.getvalue())
    print(f'[components] Saved {dest}', flush=True)
    return dest
