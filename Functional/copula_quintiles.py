"""Joint quantile-class probabilities from cached copula fits (no ERA5 reload).

The repository stores densities with rows v (latent), columns u (sensible).
This module exports matrices with rows u, columns v, as in the manuscript.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.stats import beta, norm, t


def _edges(k):
    if not isinstance(k, (int, np.integer)) or k < 2:
        raise ValueError('classes must be an integer >= 2')
    return np.linspace(0., 1., k + 1)


def _check_matrix(p, k, *, uniform=True, atol=2e-8):
    p = np.asarray(p, dtype=float)
    if p.shape != (k, k) or not np.isfinite(p).all() or p.min() < -atol:
        raise ValueError('Invalid rectangle probabilities')
    p = np.maximum(p, 0.)  # round-off only; never rebalance a fitted model
    if not np.isclose(p.sum(), 1., atol=atol, rtol=0.):
        raise ValueError(f'Probability mass {p.sum()} != 1')
    if uniform and (not np.allclose(p.sum(0), 1/k, atol=atol, rtol=0.) or
                    not np.allclose(p.sum(1), 1/k, atol=atol, rtol=0.)):
        raise ValueError('Copula does not have uniform class marginals')
    return p


def _cell_masses(density, centers, *, atol=1e-9):
    """Read the uniform histogram/checkerboard convention in copula_workflows."""
    density = np.asarray(density, dtype=float)
    centers = np.asarray(centers, dtype=float)
    m = centers.size
    if centers.ndim != 1 or m < 1 or density.shape != (m, m):
        raise ValueError('Unexpected density/centers shapes')
    if not np.allclose(centers, (np.arange(m) + .5)/m, atol=atol, rtol=0.):
        raise ValueError('Expected midpoint grid on [0,1]')
    p = density.T / m**2  # transpose: repository v,u -> manuscript u,v
    if not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1., atol=atol, rtol=0.):
        raise ValueError('Invalid saved cell masses')
    return p


def histogram_classes(cell_mass, k=5):
    """Exact integration of a piecewise constant density, even for m % k != 0."""
    p = np.asarray(cell_mass, dtype=float)
    m = p.shape[0]
    if p.shape != (m, m):
        raise ValueError('Expected square mass matrix')
    edges, fine = _edges(k), np.linspace(0., 1., m+1)
    overlap = np.maximum(0., np.minimum(edges[1:, None], fine[None, 1:]) -
                         np.maximum(edges[:-1, None], fine[None, :-1])) * m
    return overlap @ p @ overlap.T


def bernstein_classes(cell_mass, k=5):
    """Integrate normalized beta bases exactly via beta CDF differences."""
    p = np.asarray(cell_mass, dtype=float)
    m = p.shape[0]
    if p.shape != (m, m):
        raise ValueError('Expected square mass matrix')
    i = np.arange(1, m+1)
    b = np.diff(beta.cdf(_edges(k)[:, None], i[None, :], (m+1-i)[None, :]), axis=0)
    return b @ p @ b.T


def elliptical_classes(rho, k=5, *, df=None):
    """Integrate Gaussian or Student-t rectangles via 1D conditional CDF.

    With df=None, conditional normal; otherwise conditional t_(df+1).
    Integration over probability u avoids arbitrary cutoffs near 0 and 1.
    """
    rho = float(rho)
    if not np.isfinite(rho) or abs(rho) >= 1:
        raise ValueError('Expected |rho| < 1')
    if df is not None and (not np.isfinite(df) or df <= 0):
        raise ValueError('Student degrees of freedom must be positive')
    e = _edges(k)
    z = norm.ppf(e) if df is None else t.ppf(e, df)
    scale = np.sqrt(1-rho*rho)
    p = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            def conditional(u):
                x = norm.ppf(u) if df is None else t.ppf(u, df)
                s = scale if df is None else scale*np.sqrt((df+x*x)/(df+1))
                lo, hi = (z[j]-rho*x)/s, (z[j+1]-rho*x)/s
                dist = norm if df is None else t(df+1)
                # Avoid subtracting two values close to one in the right tail.
                return float(dist.sf(lo)-dist.sf(hi) if lo > 0 else dist.cdf(hi)-dist.cdf(lo))
            p[i,j] = quad(conditional, e[i], e[i+1], epsabs=1e-11,
                          epsrel=1e-9, limit=300)[0]
    return _check_matrix(p, k)


def _metrics(p, empirical):
    k = len(p)
    return dict(p11=float(p[0,0]), pkk=float(p[-1,-1]), p1k=float(p[0,-1]),
                pk1=float(p[-1,0]), same_class=float(np.trace(p)),
                tv_independence=float(.5*np.abs(p-1/k**2).sum()),
                tv_empirical=float(.5*np.abs(p-empirical).sum()),
                row_sums=p.sum(axis=1).tolist(), column_sums=p.sum(axis=0).tolist())


def _trapz(y, x, axis=-1):
    # NumPy 1.x and 2.x compatibility.
    fn = getattr(np, 'trapezoid', None)
    return (fn or np.trapz)(y, x, axis=axis)


def _moments(x, p):
    x, p = np.asarray(x, float), np.asarray(p, float)
    if (x.ndim != 1 or p.shape != x.shape or len(x) < 2 or
            not np.isfinite(x).all() or not np.isfinite(p).all() or
            np.any(np.diff(x) <= 0) or np.any(p < 0)):
        raise ValueError('Invalid marginal grid/density')
    mass = float(_trapz(p, x))
    if mass <= 0:
        raise ValueError('Zero marginal mass')
    p = p/mass
    mu = float(_trapz(x*p, x))
    var = float(_trapz((x-mu)**2*p, x))
    return dict(mass=mass, mean=mu, variance=var, sd=float(np.sqrt(var)),
                grid_min=float(x[0]), grid_max=float(x[-1]), nodes=int(len(x)))


def audit_marginal_moments(root, names, period, data, *, hybrid_reference=(1979, 2026)):
    """Compare FP, GPD hybrid, interpolated FP and integrated cached joint margins."""
    root = Path(root)
    first, last = period
    base = root/'Functional'/'Stationary'
    fp = base/f'{names[0]}-{names[1]}_{first}-{last}.npz'
    out = {'fp_source': str(fp), 'fp': {}, 'fp_on_joint_grid': {},
           'hybrid': {}, 'joint': {}, 'missing': []}
    if fp.is_file():
        with np.load(fp, allow_pickle=False) as f:
            for i, name in enumerate(names, 1):
                x, pdf = f['grid'], f[f'pdf_{name}']
                out['fp'][name] = _moments(x, pdf)
                gx = data[f'x{i}']
                out['fp_on_joint_grid'][name] = _moments(gx, np.interp(gx, x, pdf))
    else:
        out['missing'].append(str(fp))
    for name in names:
        h = base/'Hybrid'/f'{name}_{first}-{last}_gpd_reference_{hybrid_reference[0]}-{hybrid_reference[1]}.npz'
        if h.is_file():
            with np.load(h, allow_pickle=False) as f:
                out['hybrid'][name] = dict(source=str(h), **_moments(f['grid'], f['pdf']))
        else:
            out['missing'].append(str(h))
    for model in ('gaussian', 'checkerboard', 'bernstein'):
        key = f'joint_{model}'
        if key not in data.files:
            continue
        x, y, pdf = data['x1'], data['x2'], data[key]
        if pdf.shape != (len(y), len(x)):
            raise ValueError(f'Invalid joint grid: {key}')
        px, py = _trapz(pdf, y, axis=0), _trapz(pdf, x, axis=1)
        out['joint'][model] = {names[0]: _moments(x, px), names[1]: _moments(y, py)}
    return out


_NAMES_RU = {'empirical': 'Эмпирическая', 'gaussian': 'Гауссова',
             'student_t': 'Стьюдента', 'checkerboard': 'Шахматная', 'bernstein': 'Бернштейна'}


def _tex_results(path, matrices, summaries, period, k):
    """A directly insertable article fragment; all displayed values in probabilities."""
    def fmt(x):
        return '$'+f'{x:.4f}'.replace('.', '{,}')+'$'
    lines = [r'\begin{table}[htbp]', r'\centering', r'\small',
             rf'\caption{{Матрицы совместных вероятностей за {period[0]}--{period[1]} гг. Строки соответствуют классам явного, столбцы --- скрытого потока.}}',
             r'\label{tab:joint_quintile_matrix}', r'\setlength{\tabcolsep}{4pt}']
    # All matrices rather than just the preferred one. Keep floats modest for k=5.
    for model in ('empirical', 'checkerboard', 'bernstein'):
        lines += [r'\par\smallskip\textit{'+_NAMES_RU[model]+r'}\par\smallskip',
                  r'\begin{tabular}{@{}l'+'r'*k+r'@{}}', r'\toprule',
                  '$i/j$ & '+' & '.join(map(str,range(1,k+1)))+r' \\', r'\midrule']
        for i, row in enumerate(matrices[model], 1):
            lines += [str(i)+' & '+' & '.join(fmt(x) for x in row)+r' \\']
        lines += [r'\bottomrule', r'\end{tabular}']
    lines += [r'\end{table}', r'\begin{table}[htbp]', r'\centering', r'\small',
              rf'\caption{{Характеристики совместных классов за {period[0]}--{period[1]} гг. $D_{{\mathrm{{TV}}}}^{{({k})}}$ --- расстояние до эмпирической матрицы.}}',
              r'\label{tab:joint_quintile_summary}', r'\setlength{\tabcolsep}{4pt}',
              r'\begin{tabular}{@{}lrrrrrr@{}}', r'\toprule',
              rf'Модель & $\pi_{{11}}$ & $\pi_{{{k}{k}}}$ & $\pi_{{1{k}}}$ & $\pi_{{{k}1}}$ & $\sum_i\pi_{{ii}}$ & $D_{{\mathrm{{TV}}}}^{{({k})}}$ \\', r'\midrule']
    for model, s in summaries.items():
        lines += [_NAMES_RU[model]+' & '+' & '.join(fmt(s[z]) for z in ('p11','pkk','p1k','pk1','same_class','tv_empirical'))+r' \\']
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def estimate_joint_quantile_classes(files_path_prefix, start_year, end_year, *,
                                    names=('sensible','latent'), classes=5,
                                    hybrid_reference=(1979,2026), out_dir=None):
    """Read the NPZ/JSON created by task='copulas'; export CSV, NPZ, JSON and TeX."""
    names = tuple(names)
    if len(names) != 2:
        raise ValueError('Expected two variables')
    _edges(classes)
    period = (int(start_year), int(end_year))
    root = Path(files_path_prefix)
    tag = f'{names[0]}-{names[1]}_{period[0]}-{period[1]}'
    source = root/'Functional'/'Copulas'/f'{tag}.npz'
    meta = source.with_suffix('.json')
    if not source.is_file() or not meta.is_file():
        raise FileNotFoundError(f"Run main.py task='copulas' for {period} first: {source}, {meta}")
    fit = json.loads(meta.read_text(encoding='utf-8'))
    if tuple(fit['pair']) != names or fit['period'] != f'{period[0]}-{period[1]}':
        raise ValueError('Saved copula identity does not match requested pair/period')
    with np.load(source, allow_pickle=False) as data:
        empirical_mass = _cell_masses(data['empirical'], data['centers'])
        checker_mass = _cell_masses(data['checkerboard'], data['checkerboard_centers'])
        # Checkerboard balancing is the basis of BOTH nonparametric models.
        _check_matrix(checker_mass, len(checker_mass))
        matrices = {'empirical': _check_matrix(histogram_classes(empirical_mass, classes), classes, uniform=False),
                    'gaussian': elliptical_classes(fit['gaussian_rho'], classes),
                    'student_t': elliptical_classes(fit['student_t_rho'], classes, df=fit['student_t_df']),
                    'checkerboard': _check_matrix(histogram_classes(checker_mass, classes), classes),
                    'bernstein': _check_matrix(bernstein_classes(checker_mass, classes), classes)}
        audit = audit_marginal_moments(root, names, period, data, hybrid_reference=hybrid_reference)
    summaries = {model: _metrics(p, matrices['empirical']) for model,p in matrices.items()}
    target = Path(out_dir) if out_dir is not None else root/'Functional'/'Copulas'/'Quintiles'/f'{tag}_k{classes}'
    target.mkdir(parents=True, exist_ok=True)
    for model, p in matrices.items():
        with (target/f'{model}.csv').open('w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow(['row_class']+list(range(1,classes+1)))
            for i, row in enumerate(p,1):
                w.writerow([i]+list(row))
    np.savez_compressed(target/'matrices.npz', edges=_edges(classes), **matrices)
    payload = dict(pair=list(names), period=list(period), classes=classes, source=str(source),
                   source_json=str(meta), axes={'rows':names[0], 'columns':names[1]},
                   empirical='integrated saved rank histogram (not direct physical-class counts)',
                   empirical_bins=len(empirical_mass), checkerboard_bins=len(checker_mass),
                   gaussian_rho=fit['gaussian_rho'], student_t_rho=fit['student_t_rho'],
                   student_t_df=fit['student_t_df'], independence_probability=1/classes**2,
                   matrices={m:p.tolist() for m,p in matrices.items()}, metrics=summaries,
                   cached_joint_moments=fit.get('moments', {}), marginal_audit=audit)
    (target/'summary.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    if names == ('sensible','latent') and classes == 5:
        _tex_results(target/'joint_quintiles_results.tex', matrices, summaries, period, classes)
    for model, s in summaries.items():
        print(f"{model:12s} P11={s['p11']:.6f} Pkk={s['pkk']:.6f} "
              f"same={s['same_class']:.6f} TV_emp={s['tv_empirical']:.6f}")
    print(f'Joint quantile results: {target}')
    return target
