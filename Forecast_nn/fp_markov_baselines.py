"""Matched baseline comparison for the stationary Fokker--Planck Markov probe.

Uses the SAME 5x5 aggregation, valid-block masking, dates, and previously
saved conditional Markov means as fp_markov_probe.py.  No FP recomputation.

CAUTION: the supplied 1979--2026 FP lookup incorporates evaluation-period
statistics.  Comparisons involving Markov are exploratory, NOT a clean test.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from Forecast_nn.fp_markov_probe import (
    GRID,
    _field_length,
    _native_time_array,
    block_means_joint,
    current_state_indices,
)


ROOT_DEFAULT = Path('/home/aosipova/EM_ocean')


def load_lookup(path: Path) -> dict:
    needed = {'coordinates', 'pi', 'x_edges', 'y_edges', 'domain_mask', 'mean'}
    with np.load(path, allow_pickle=False) as source:
        missing = needed - set(source.files)
        if missing:
            raise ValueError(f'Markov lookup missing {sorted(missing)}: {path}')
        obj = {key: np.asarray(source[key]) for key in needed}
    domain = obj['domain_mask'].astype(bool)
    coordinates = obj['coordinates']
    nstates = int(domain.sum())
    if (coordinates.shape != (nstates, 2)
            or not np.array_equal(coordinates, np.argwhere(domain))):
        raise ValueError('Lookup state order does not match domain_mask; do not use.')
    if obj['mean'].shape != (4, nstates, 2):
        raise ValueError('Lookup must have 4 horizons (0..3), n_states, 2 variables.')
    if obj['pi'].shape != (nstates,) or not np.isclose(obj['pi'].sum(), 1, atol=1e-8):
        raise ValueError('Invalid stationary probability vector')
    if (len(obj['x_edges']) != domain.shape[0]+1 or
            len(obj['y_edges']) != domain.shape[1]+1):
        raise ValueError('Phase domain and edge arrays disagree')
    if not np.isfinite(obj['mean']).all():
        raise ValueError('Markov conditional means must be finite')
    obj['domain_mask'] = domain
    obj['x_centers'] = (obj['x_edges'][:-1] + obj['x_edges'][1:]) / 2
    obj['y_centers'] = (obj['y_edges'][:-1] + obj['y_edges'][1:]) / 2
    centers = np.column_stack((obj['x_centers'][coordinates[:, 0]],
                               obj['y_centers'][coordinates[:, 1]]))
    obj['stationary_mean'] = obj['pi'] @ centers
    return obj


def finite_mean(x: np.ndarray, axis: int):
    good = np.isfinite(x)
    count = np.sum(good, axis=axis)
    total = np.sum(np.where(good, x, 0.), axis=axis, dtype=np.float64)
    result = np.full_like(total, np.nan, dtype=np.float64)
    np.divide(total, count, out=result, where=count > 0)
    return result


def fit_ar1(blocks: np.ndarray, min_pairs=60, clip=0.995) -> dict:
    """OLS x(t+1)=a+phi*x(t), separately by variable, global and per block.

    Only the pre-evaluation training window can be supplied here.
    """
    if blocks.ndim != 4 or blocks.shape[-1] != 2 or len(blocks) < 3:
        raise ValueError('AR1 fit needs at least three block frames [T,by,bx,2]')
    x, y = blocks[:-1], blocks[1:]
    good = np.isfinite(x) & np.isfinite(y)
    n = good.sum(axis=0)
    sx = np.where(good, x, 0.).sum(axis=0, dtype=np.float64)
    sy = np.where(good, y, 0.).sum(axis=0, dtype=np.float64)
    mx = sx / np.maximum(n, 1)
    my = sy / np.maximum(n, 1)
    xx = np.where(good, x-mx, 0.)
    yy = np.where(good, y-my, 0.)
    varx = (xx * xx).sum(axis=0, dtype=np.float64)
    covxy = (xx * yy).sum(axis=0, dtype=np.float64)

    # Global fit uses all valid block-day transitions, per variable.
    ng = good.sum(axis=(0, 1, 2))
    mgx = sx.sum(axis=(0, 1)) / np.maximum(ng, 1)
    mgy = sy.sum(axis=(0, 1)) / np.maximum(ng, 1)
    xg = np.where(good, x-mgx, 0.)
    yg = np.where(good, y-mgy, 0.)
    vg = np.square(xg).sum(axis=(0, 1, 2), dtype=np.float64)
    cg = (xg * yg).sum(axis=(0, 1, 2), dtype=np.float64)
    if np.any(ng < 2) or np.any(vg < 1e-12):
        raise ValueError('Insufficient finite variance for global AR1 fit')
    pg = np.clip(cg/vg, -clip, clip)
    ag = mgy-pg*mgx

    # Avoid invalid division warnings, then fall back to global coefficients.
    pl = np.full(n.shape, np.nan, dtype=np.float64)
    ok = (n >= min_pairs) & (varx > 1e-12)
    np.divide(covxy, varx, out=pl, where=ok)
    pl = np.where(ok, np.clip(pl, -clip, clip), pg)
    al = np.where(ok, my-pl*mx, ag)

    local_climatology = finite_mean(blocks, axis=0)
    local_climatology = np.where(np.isfinite(local_climatology), local_climatology, mgx)
    return {
        'global_intercept': ag, 'global_phi': pg,
        'local_intercept': al, 'local_phi': pl,
        'global_training_mean': mgx,
        'local_training_mean': local_climatology,
        'n_local_ar1_fit': int(ok.sum()),
        'n_local_ar1_total': int(ok.size),
        'local_pairs_min': int(n.min()),
        'local_pairs_max': int(n.max()),
    }


def ar1_forecast(current: np.ndarray, intercept, phi, horizon: int) -> np.ndarray:
    pred = np.asarray(current, dtype=np.float64).copy()
    for _ in range(horizon):
        pred = intercept + phi * pred
    return pred


def trailing_mean(blocks: np.ndarray, origins: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean at each origin, including today; no future observations."""
    if np.any(origins < window-1):
        raise ValueError('Not enough history to calculate trailing mean')
    current = blocks[origins]
    sums = np.zeros_like(current, dtype=np.float64)
    count = np.zeros_like(current, dtype=np.int16)
    for lag in range(window):
        value = blocks[origins-lag]
        ok = np.isfinite(value)
        sums += np.where(ok, value, 0.)
        count += ok
    avg = np.full_like(sums, np.nan)
    np.divide(sums, count, out=avg, where=count > 0)
    return np.where(count >= 2, avg, current)


def _scores(pred: np.ndarray, truth: np.ndarray, valid: np.ndarray):
    """Return channel-wise SSE aggregated over blocks per forecast origin."""
    error = np.where(valid[..., None], pred-truth, 0.)
    daily_sse = np.sum(error**2, axis=(1, 2), dtype=np.float64)
    return daily_sse


def moving_block_bootstrap(daily_sse, day_counts, *, reps=300, block_days=14,
                           seed=2025):
    """95% confidence interval for skill vs persistence; resample dates, not pixels."""
    if reps <= 0:
        return {}
    rng = np.random.default_rng(seed)
    n = len(day_counts)
    names = list(daily_sse)
    values = np.zeros((reps, len(names)), dtype=np.float64)
    block_len = min(block_days, n)
    n_blocks = (n + block_len-1)//block_len
    offsets = np.arange(block_len, dtype=np.int64)
    for i in range(reps):
        starts = rng.integers(0, n, size=n_blocks)
        sampled = ((starts[:, None]+offsets[None, :]) % n).ravel()[:n]
        reference = daily_sse['persistence'][sampled].sum()
        for j, name in enumerate(names):
            values[i, j] = 1.0 - daily_sse[name][sampled].sum()/reference
    return {name: {
        'mse_skill_ci95_low': float(np.quantile(values[:, j], .025)),
        'mse_skill_ci95_high': float(np.quantile(values[:, j], .975)),
        'prob_skill_positive_bootstrap': float(np.mean(values[:, j] > 0)),
    } for j, name in enumerate(names)}


def evaluate(blocks: np.ndarray, eval_origins: np.ndarray,
             lookup: dict, fitted: dict, *, bootstrap_reps=300,
             bootstrap_block_days=14, seed=2025):
    """All methods use identical forecast origins and valid joint-space pixels."""
    if max(eval_origins)+3 >= len(blocks):
        raise ValueError('Origins must have at least three future target days')
    current = blocks[eval_origins]
    moving7 = trailing_mean(blocks, eval_origins, 7)
    moving30 = trailing_mean(blocks, eval_origins, 30)
    sid = current_state_indices(current, lookup)
    covered = sid >= 0
    if not np.any(covered):
        raise ValueError('No current state lies inside stationary phase domain')
    stationary_mean = lookup['stationary_mean']
    local_mean = fitted['local_training_mean']
    summaries = {}
    for tau in (1, 2, 3):
        truth = blocks[eval_origins + tau]
        valid = np.isfinite(current).all(axis=-1) & np.isfinite(truth).all(axis=-1)
        counts = valid.sum(axis=(1, 2))
        if not counts.sum():
            raise ValueError(f'No valid current/target blocks for tau={tau}')
        prior = lookup['mean'][tau, np.maximum(sid, 0)]
        markov = np.where(covered[..., None], prior, current)
        reversion = markov-current
        predictions = {
            'persistence': current,
            'mean_7d': moving7,
            'mean_30d': moving30,
            'stationary_mean_constant': np.broadcast_to(stationary_mean, current.shape),
            'local_climatology_constant': np.broadcast_to(local_mean, current.shape),
            'markov_25pct': current + .25*reversion,
            'markov_50pct': current + .50*reversion,
            'markov_100pct': markov,
            'ar1_global': ar1_forecast(current, fitted['global_intercept'],
                                      fitted['global_phi'], tau),
            'ar1_local': ar1_forecast(current, fitted['local_intercept'],
                                     fitted['local_phi'], tau),
        }
        for timescale in (3, 7, 14):
            alpha = -np.expm1(-tau/timescale)
            predictions[f'revert_FP_tau{timescale}d'] = (
                current + alpha*(stationary_mean-current))
            predictions[f'revert_local_tau{timescale}d'] = (
                current + alpha*(local_mean-current))
        daily_sse = {name: _scores(p, truth, valid) for name, p in predictions.items()}
        n = int(counts.sum())
        base_sse = daily_sse['persistence'].sum(axis=0)
        base_joint = base_sse.sum()
        ci = moving_block_bootstrap(daily_sse, counts, reps=bootstrap_reps,
                                    block_days=bootstrap_block_days, seed=seed+tau)
        metrics = {}
        for name, err in daily_sse.items():
            sse = err.sum(axis=0)
            mse = sse / n
            metrics[name] = {
                'rmse_sensible': float(np.sqrt(mse[0])),
                'rmse_latent': float(np.sqrt(mse[1])),
                'rmse_joint': float(np.sqrt(mse.mean())),
                'mse_skill_vs_persistence': float(1 - sse.sum()/base_joint),
                'mse_skill_sensible': float(1 - sse[0]/base_sse[0]),
                'mse_skill_latent': float(1 - sse[1]/base_sse[1]),
                **ci.get(name, {}),
            }
        summaries[str(tau)] = {
            'n_initial_times': int(len(eval_origins)),
            'n_valid_block_forecasts': n,
            'fraction_valid_blocks_with_supported_phase_state': float(
                np.sum(covered & valid)/n),
            'models': metrics,
        }
    return summaries


def build_report(args, root, eval_start, n_dates, fit_start, lookup,
                 fitted, results):
    start_date = date.fromisoformat(args.base_date)
    def day(idx):
        return (start_date + timedelta(days=int(idx))).isoformat()
    return {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'warning': 'Exploratory: stationary FP lookup was calculated on the full 1979--2026 interval, including evaluation dates; NOT independent out-of-sample Markov validation.',
        'root': str(root), 'lookup_file': str(args.lookup),
        'grid_native': list(GRID), 'block_native': args.block,
        'dates_total': n_dates,
        'evaluation_first_origin_index': int(eval_start),
        'evaluation_first_origin_date': day(eval_start),
        'evaluation_last_origin_date': day(n_dates-4),
        'evaluation_last_target_date': day(n_dates-1),
        'evaluation_step_days': args.step,
        'ar1_fit_first_date': day(fit_start),
        'ar1_fit_last_date': day(eval_start-1),
        'ar1_fit_days': int(eval_start-fit_start),
        'ar1_local_fitted_channel_blocks': fitted['n_local_ar1_fit'],
        'ar1_local_channel_blocks_total': fitted['n_local_ar1_total'],
        'ar1_global_coefficients': {
            'intercept': fitted['global_intercept'].tolist(),
            'phi': fitted['global_phi'].tolist(),
        },
        'stationary_mean': lookup['stationary_mean'].tolist(),
        'bootstrap': {'reps': args.bootstrap_reps,
                      'block_days': args.bootstrap_block_days,
                      'unit': 'forecast origin (day), paired across all models'},
        'scores_by_horizon': results,
    }


def save_csv(report, path):
    rows = []
    for horizon, record in report['scores_by_horizon'].items():
        for model, score in record['models'].items():
            rows.append({'horizon_days': horizon, 'model': model,
                         'n_forecasts': record['n_valid_block_forecasts'], **score})
    rows.sort(key=lambda r: (int(r['horizon_days']), r['rmse_joint']))
    columns = list(rows[0])
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=ROOT_DEFAULT)
    ap.add_argument('--lookup', type=Path, default=None,
                    help='Existing markov_lookup.npz from fp_markov_probe (no Q rebuild)')
    ap.add_argument('--eval-days', type=int, default=365)
    ap.add_argument('--fit-days', type=int, default=730,
                    help='Historical days before evaluation to fit AR1 and local climatology')
    ap.add_argument('--step', type=int, default=1)
    ap.add_argument('--block', type=int, default=5)
    ap.add_argument('--base-date', default='1979-01-01')
    ap.add_argument('--bootstrap-reps', type=int, default=300)
    ap.add_argument('--bootstrap-block-days', type=int, default=14)
    ap.add_argument('--min-local-ar1-pairs', type=int, default=60)
    args = ap.parse_args()
    root = args.root.resolve()
    if args.eval_days < 1 or args.fit_days < 3 or args.step < 1:
        ap.error('Require --eval-days >= 1, --fit-days >= 3, --step >= 1')
    if args.block != 5:
        ap.error('Use --block 5 to compare with the existing 5x5 Markov probe')
    if args.bootstrap_reps < 0 or args.bootstrap_block_days < 1:
        ap.error('Require bootstrap reps >= 0, block days >= 1')
    if args.min_local_ar1_pairs < 2:
        ap.error('Require --min-local-ar1-pairs >= 2')
    args.lookup = args.lookup or (root/'Forecast'/'Results'/'fp_markov_probe'/'markov_lookup.npz')
    flux = [root/'DATA'/'Fluxes'/f'{v}_1979-2026_grouped.npy'
            for v in ('sensible', 'latent')]
    mask_path = root/'DATA'/'mask'
    for p in (*flux, mask_path, args.lookup):
        if not p.is_file():
            raise FileNotFoundError(f'Required file not found: {p}')
    lookup = load_lookup(args.lookup)
    lengths = [_field_length(p) for p in flux]
    if lengths[0] != lengths[1]:
        raise ValueError(f'Flux time lengths differ: {lengths}')
    n_dates = lengths[0]
    # Exactly mirrors the original probe: origins start at n-(eval_days+3),
    # with the next three days available.  Include 29 preceding days for means.
    eval_start = max(29, n_dates-(args.eval_days+3))
    if n_dates-3 <= eval_start:
        raise ValueError('Too few dates after the evaluation start')
    origins_absolute = np.arange(eval_start, n_dates-3, args.step)
    fit_start = max(0, eval_start-args.fit_days)
    if eval_start-fit_start < 3:
        raise ValueError('Too few pre-evaluation dates to fit AR1')
    mask = np.fromfile(mask_path, dtype=np.uint8)
    if mask.size != GRID[0]*GRID[1]:
        raise ValueError(f'Mask size {mask.size} != native grid size')
    mask = mask.reshape(GRID).astype(bool)
    print(f'[inputs] root={root} lookup={args.lookup} states={len(lookup["pi"])}', flush=True)
    print(f'[dates] n={n_dates} eval_start_index={eval_start} '
          f'origins={len(origins_absolute)} fit_start_index={fit_start}', flush=True)
    print('[fit] reading historical blocks before evaluation only', flush=True)
    fit_fields = [_native_time_array(p, fit_start, eval_start)[0] for p in flux]
    fit_blocks = block_means_joint(fit_fields[0], fit_fields[1], mask, args.block)
    fitted = fit_ar1(fit_blocks, min_pairs=args.min_local_ar1_pairs)
    del fit_fields, fit_blocks
    print(f'[fit] global phi={fitted["global_phi"].tolist()} '
          f'local fitted={fitted["n_local_ar1_fit"]}/'
          f'{fitted["n_local_ar1_total"]}', flush=True)
    print(f'[eval] reading {len(origins_absolute)} forecast origins (same alignment as original probe)', flush=True)
    eval_load_start = eval_start-29
    eval_fields = [_native_time_array(p, eval_load_start, n_dates)[0] for p in flux]
    blocks = block_means_joint(eval_fields[0], eval_fields[1], mask, args.block)
    origins_relative = origins_absolute-eval_load_start
    result = evaluate(blocks, origins_relative, lookup, fitted,
                      bootstrap_reps=args.bootstrap_reps,
                      bootstrap_block_days=args.bootstrap_block_days)
    report = build_report(args, root, eval_start, n_dates, fit_start, lookup, fitted, result)
    original_report_path = root/'Forecast'/'Results'/'fp_markov_probe'/'markov_probe_metrics.json'
    if original_report_path.is_file():
        original = json.loads(original_report_path.read_text(encoding='utf-8'))
        if (original.get('num_dates') == n_dates and
                original.get('eval_start_index_zero_based') == eval_start and
                original.get('spatial_aggregation_native') == '5x5' and
                original.get('scores_by_horizon', {}).get('1', {}).get('n_initial_times') == len(origins_absolute)):
            differences = {}
            for horizon in ('1', '2', '3'):
                older = original['scores_by_horizon'][horizon]
                newer = result[horizon]
                if older['n_valid_block_forecasts'] != newer['n_valid_block_forecasts']:
                    raise RuntimeError(f'Different valid block count at horizon {horizon}')
                for name in ('persistence', 'stationary_mean_constant',
                             'markov_25pct', 'markov_50pct', 'markov_100pct'):
                    delta = (newer['models'][name]['rmse_joint'] -
                             older['models'][name]['rmse_joint'])
                    differences[f'{horizon}d_{name}'] = float(delta)
                    if abs(delta) > 1e-5:
                        raise RuntimeError('Baseline/previous probe RMSE mismatch: '
                                           f'{horizon}d {name}: {delta:.6g}. '
                                           'Check input fluxes and lookup provenance.')
            report['previous_probe_consistency'] = {
                'status': 'passed', 'file': str(original_report_path),
                'max_abs_joint_rmse_difference': max(abs(v) for v in differences.values()),
            }
            print('[check] Matches previous Markov-probe metrics (tolerance 1e-5)', flush=True)
        else:
            report['previous_probe_consistency'] = {
                'status': 'not_comparable_metadata', 'file': str(original_report_path)}
            print('[check] Original Markov-probe metadata differ; automatic parity check skipped', flush=True)
    outdir = root/'Forecast'/'Results'/'fp_markov_baselines'
    outdir.mkdir(parents=True, exist_ok=True)
    out_json = outdir/'fp_baseline_metrics.json'
    out_csv = outdir/'fp_baseline_ranking.csv'
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    save_csv(report, out_csv)
    for horizon, item in result.items():
        print(f'\n[HORIZON {horizon} DAYS] valid={item["n_valid_block_forecasts"]} '
              f'markov_coverage={item["fraction_valid_blocks_with_supported_phase_state"]:.4f}')
        for name, score in sorted(item['models'].items(), key=lambda x:x[1]['rmse_joint']):
            print(f' {name:30s} RMSE={score["rmse_joint"]:9.4f} '
                  f'skill={score["mse_skill_vs_persistence"]:+8.3%}', flush=True)
    print(f'\n[outputs] {out_json}\n[outputs] {out_csv}', flush=True)
    print('[warning] FP lookup fitted on full date range; Markov comparison is exploratory.', flush=True)


if __name__ == '__main__':
    main()
