"""Daily ocean-cell statistics and blockwise linear trends (no rendering).

Restores the analysis of extract_extreme/get_trends using current time-first
arrays, one-based transition labels, and real eigenvalue/SVD spectra.
"""
from __future__ import annotations

import csv
import json
from datetime import date, timedelta
from pathlib import Path
from itertools import combinations_with_replacement

import numpy as np
from tqdm.auto import tqdm
from Data_processing.io import BASE_DATE, field_candidates, load_field, read_ocean_mask

VARIABLES = ('Flux', 'SST', 'Pressure', 'sensible', 'latent')
COEFFICIENT_PAIRS = (('Flux', 'SST'), ('SST', 'Pressure'), ('Flux', 'Pressure'), ('sensible', 'latent'))
SPECTRAL_PAIRS = tuple(combinations_with_replacement(VARIABLES[:3], 2))
PERIODS = ((1979, 1989), (1989, 1999), (1999, 2009), (2009, 2019), (2019, 2026))
STAT_NAMES = ('max', 'min', 'mean', 'range')


def day_label(d):
    """Convert the date to the project day-index convention."""
    return (d - BASE_DATE).days + 1


def _bounds(start_year, end_year, last_date=None):
    # The project's 2026 full-period filename may include real days in 2026.
    """Determine the relevant interval bounds."""
    first = day_label(date(start_year, 1, 1))
    stop = day_label(date(end_year + (end_year == 2026), 1, 1))
    if last_date is not None:
        stop = min(stop, day_label(date.fromisoformat(str(last_date))) + 1)
    if stop <= first:
        raise ValueError('Empty trends period')
    return first, stop


def _npy(path):
    """Load or resolve the requested NPY array."""
    try:
        return np.load(path, mmap_mode='r', allow_pickle=False)
    except (ValueError, OSError, EOFError) as exc:
        raise ValueError(f'Cannot read numeric NPY {path}: {exc}') from exc


def spatial_statistics(field, ocean):
    """Unweighted sea-cell extrema, mean and range, like the historical code."""
    x = np.asarray(field)
    if x.ndim == 2:
        x = x[None, ...]
    if x.ndim != 3 or x.shape[-2:] != ocean.shape:
        raise ValueError(f'Expected channels x {ocean.shape}; got {x.shape}')
    out = np.full((x.shape[0], 4), np.nan)
    for i, channel in enumerate(x):
        valid = channel[ocean & np.isfinite(channel)]
        if valid.size:
            hi, lo = float(valid.max()), float(valid.min())
            out[i] = hi, lo, np.mean(valid, dtype=np.float64), hi - lo
    return out


def daily_stats(values, ocean, desc, *, first=0, stop=None, chunk_days=16):
    """Bound memory by reading short time blocks, not the entire memmap."""
    stop = len(values) if stop is None else stop
    if values.ndim not in (3, 4) or values.shape[-2:] != ocean.shape:
        raise ValueError(f'{desc}: invalid input shape {values.shape}')
    n_channels = values.shape[1] if values.ndim == 4 else 1
    out = np.full((stop-first, n_channels, 4), np.nan)
    for begin in tqdm(range(first, stop, chunk_days), desc=desc, unit='chunk'):
        for j, field in enumerate(values[begin:min(begin+chunk_days, stop)]):
            out[begin-first+j] = spatial_statistics(field, ocean)
    return out


def block_statistics(labels, daily, width, *, anchor=None, min_coverage=.9, spectral_min=False):
    """Complete non-overlapping calendar-day blocks, preserving missing dates.

    Historical spectral-vector minima use min of daily minima; other columns
    use means of daily statistics. Empty days are not treated as zeros.
    """
    labels = np.asarray(labels, dtype=np.int64)
    daily = np.asarray(daily, dtype=float)
    if len(labels) != len(daily) or not len(labels) or np.any(np.diff(labels) <= 0):
        raise ValueError('Expected increasing distinct daily labels')
    if width < 1 or not 0 < min_coverage <= 1:
        raise ValueError('Invalid block width or coverage')
    anchor = int(labels[0] if anchor is None else anchor)
    if anchor > labels[0]:
        raise ValueError('Block anchor cannot follow first observation')
    n_blocks = (int(labels[-1])-anchor+1)//width
    shape = (n_blocks, *daily.shape[1:])
    out = np.full(shape, np.nan)
    coverage = np.zeros(shape, dtype=np.int32)
    starts = anchor + np.arange(n_blocks)*width
    for i, start in enumerate(starts):
        left, right = np.searchsorted(labels, (start, start+width))
        block = daily[left:right]
        finite = np.isfinite(block)
        count = finite.sum(axis=0)
        coverage[i] = count
        np.divide(np.where(finite, block, 0).sum(axis=0), count,
                  out=out[i], where=count >= np.ceil(width*min_coverage))
        if spectral_min and len(block):
            minima = np.min(np.where(finite[:, :, 1], block[:, :, 1], np.inf), axis=0)
            out[i, :, 1] = np.where(count[:, 1] >= np.ceil(width*min_coverage), minima, np.nan)
    return starts, out, coverage


def linear_fit(labels, values):
    """OLS slope in units per day; intercept at 1979-01-01, no p-value claim."""
    x = np.asarray(labels, dtype=float)-1
    y = np.asarray(values, dtype=float)
    good = np.isfinite(y)
    if good.sum() < 2:
        return (np.nan,)*4 + (int(good.sum()),)
    x, y = x[good], y[good]
    dx, dy = x-x.mean(), y-y.mean()
    slope = float(np.dot(dx, dy)/np.dot(dx, dx))
    intercept = float(y.mean()-slope*x.mean())
    r2 = 1-float(np.sum((y-(slope*x+intercept))**2)/np.dot(dy,dy)) if np.dot(dy,dy)>0 else np.nan
    return slope, intercept, slope*float(x[-1]-x[0]), r2, len(x)


def result_path(root, category, identifier, start_year, end_year, *, method='sp', n_bins=12):
    """Construct the output path for the requested result."""
    base = Path(root)/'Trends'/category
    if category in ('Coefficients', 'Derived'):
        base = base/f'{n_bins}_bins'/method
    return base/f'{identifier}_{start_year}-{end_year}.npz'


def _publish(path, labels, daily, channels, windows, anchor, min_coverage, metadata, *, spectral_min=False):
    """Publish the completed output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels, dtype=np.int64)
    if not len(labels):
        raise ValueError(f'No observations for {path}')
    if daily.shape != (len(labels), len(channels), 4):
        raise ValueError(f'Unexpected statistics shape {daily.shape}')
    payload = dict(labels=labels, daily=daily, channels=np.asarray(channels),
                   statistics=np.asarray(STAT_NAMES), windows=np.asarray(windows),
                   metadata=json.dumps(dict(metadata, min_coverage=min_coverage,
                       first_date=str(BASE_DATE+timedelta(days=int(labels[0])-1)),
                       last_date=str(BASE_DATE+timedelta(days=int(labels[-1])-1)),
                       spectral_min=spectral_min, time_units='days since 1979-01-01'), ensure_ascii=False))
    rows=[]
    for w in windows:
        starts, block, coverage = block_statistics(labels, daily, w, anchor=anchor,
                                                   min_coverage=min_coverage, spectral_min=spectral_min)
        payload.update({f'labels_{w}': starts, f'values_{w}':block, f'coverage_{w}':coverage})
        fits=np.empty((len(channels),4,5), dtype=float)
        for c, channel in enumerate(channels):
            for j, stat in enumerate(STAT_NAMES):
                fit=linear_fit(starts, block[:,c,j]); fits[c,j]=fit
                rows.append((channel, stat, w, fit[0], fit[0]*365.2425, *fit[1:]))
        payload[f'fits_{w}']=fits
    tmp=path.with_name('.'+path.name+'.tmp.npz')
    np.savez_compressed(tmp, **payload); tmp.replace(path)
    with path.with_suffix('.csv').open('w', newline='', encoding='utf-8-sig') as f:
        out=csv.writer(f); out.writerow(('channel','statistic','window_days','slope_per_day',
             'slope_per_year','intercept','fitted_change_between_first_last_blocks','r2','n_blocks'))
        out.writerows(rows)
    print(f'[trends] Saved {path}', flush=True)
    return path


def _field_blocks(root, name, sy, ey, periods, ocean):
    # Prefer the exact requested collected array. Fall back only if absent,
    # not if a candidate exists but is corrupt (that must be repaired).
    """Yield blocks of input observations for memory-efficient processing."""
    def read(a,b):
        """Read the requested trends data using the supplied inputs."""
        path=next((p for p in field_candidates(root,name,a,b) if p.is_file()),None)
        try:
            return load_field(root,name,a,b,ocean.shape)
        except (ValueError, OSError, EOFError) as exc:
            raise ValueError(f'Cannot load {name} {a}-{b} from {path}: {exc}') from exc
    if any(p.is_file() for p in field_candidates(root,name,sy,ey)):
        return [(sy,ey,read(sy,ey))]
    chosen=[(a,b) for a,b in periods if a < ey and b > sy]
    if not chosen:
        raise FileNotFoundError(f'No sources for {name} {sy}-{ey}')
    return [(a,b,read(a,b)) for a,b in chosen]


def _coefficient_file(root,pair,key,method,n_bins):
    """Resolve the file containing the requested coefficient."""
    base=Path(root)/'Coefficients'; pair_id='_'.join(pair)
    parent=base/'continuous'/f'2d_bins_{n_bins}'/pair_id/('Nonparametric_2d' if method=='np' else 'Semiparametric_2d')
    manifest=base/'continuous'/f'manifest_{pair_id}_{n_bins}.json'
    if key in ('A','B','B_with_reg') and manifest.is_file():
        m=json.loads(manifest.read_text(encoding='utf-8'))
        source=m['methods'].get(method,{}).get(key)
        if source:
            import re
            # An old absolute path may refer to another laptop. Only use its
            # local canonical counterpart, never silently another method.
            basename=re.split(r'[\\/]', str(source))[-1]
            match=re.fullmatch(rf'{key}_(\d{{4}}-\d{{4}})_{method}\.npy',basename)
            options=[]
            if match:
                options.append(base/match.group(1)/pair_id/basename)
            options.append(parent/f"{key}_{m['first_label']}-{m['last_label']}.npy")
            p=next((p for p in options if p.is_file()),None)
            if p is not None:
                return p,int(m['first_label']),int(m['last_label'])
            raise FileNotFoundError(f'{key}: local manifest sources not found: {options}; manifest={manifest}')
    sources=sorted(parent.glob(f'{key}_*-*.npy'))
    if len(sources)!=1:
        raise FileNotFoundError(f'Expected one {key} source under {parent}; found {sources}')
    first,last=map(int,sources[0].stem.removeprefix(key+'_').split('-'))
    return sources[0],first,last


def _channels(pair,key):
    """Identify the relevant field channels."""
    if key=='A': return tuple(f'A({name})' for name in pair)
    if key in ('B','B_with_reg'): return tuple(f'{key}{suffix}' for suffix in ('00','11','01','10'))
    if key in ('C','FS'): return (f'{key}1',f'{key}2')
    if key=='F': return ('F',)
    raise ValueError(f'Unknown coefficient {key}')


def _spectral_daily(root,pair,first,stop, *, ocean, mode_extrema, spectral_periods, n_bins):
    """Load the daily spectral result."""
    folder=Path(root)/'Eigenvalues'/'-'.join(pair)
    mixed=pair[0]!=pair[1]
    kind='singular_values' if mixed else 'eigenvalues'
    sources={}
    for path in folder.glob(f'{kind}_*.npy'):
        token=path.stem.removeprefix(kind+'_')
        if token.isdecimal() and first<=int(token)<stop:
            sources[int(token)]=path
    if not sources:
        raise FileNotFoundError(f'No daily {kind} in {folder} for requested period')
    labels=np.arange(first,max(sources)+1)
    scalar=np.full((len(labels),1,4),np.nan)
    # Read spectra only once. Rank 1, not the sum or squared singular values.
    detected_bins=n_bins
    for l,p in tqdm(sorted(sources.items()), desc=f'{folder.name} spectrum',unit='day'):
        vals=np.asarray(_npy(p))
        if vals.ndim!=1 or not len(vals) or not np.isfinite(vals).all():
            raise ValueError(f'Invalid spectrum: {p}')
        if detected_bins is None: detected_bins=len(vals)
        if len(vals)!=detected_bins:
            raise ValueError(f'{p}: {len(vals)} bins, expected {detected_bins}; do not mix quantile schemes')
        scalar[l-first,0,:3]=float(vals[0]); scalar[l-first,0,3]=0.
    if not mode_extrema:
        return labels,scalar,None,None
    from Data_processing.io import load_spectral_field
    from Eigenvalues.eigenvalues import get_bins
    # Reuse the exact, existing quantile cache (not a new estimator).
    from Plotting.workflows import _cached_spectral_quantiles
    from Plotting.plot_eigenvalues import _fix_signs, _fix_svd_signs
    names=('left vector 1','right vector 1','mode contribution 1') if mixed else ('vector 1','mode contribution 1')
    modes=np.full((len(labels),len(names),4),np.nan)
    consumed=set()
    for a,b in spectral_periods:
        begin=day_label(date(a,1,1)); nominal_end=day_label(date(b+(b==2026),1,1))
        selected=[l for l in sorted(sources) if max(first,begin)<=l<min(stop,nominal_end)]
        if not selected: continue
        fields={v:load_spectral_field(root,v,a,b,ocean.shape) for v in dict.fromkeys(pair)}
        lengths={x.shape[1] for x in fields.values()}
        if len(lengths)!=1: raise ValueError(f'Unequal spectral input lengths, {pair}, {a}-{b}')
        period_stop=begin+next(iter(lengths))-1  # last day has no increment
        selected=[l for l in selected if l<period_stop]
        if not selected: continue
        bins=len(_npy(sources[selected[0]]))
        edges={v:_cached_spectral_quantiles(root,v,a,b,bins,series) for v,series in fields.items()}
        prev=None; previous_label=None
        for l in tqdm(selected,desc=f'{folder.name} vectors {a}-{b}',unit='day'):
            if l in consumed: raise ValueError(f'Overlapping spectral periods at {l}')
            if previous_label is None or l!=previous_label+1: prev=None
            if mixed:
                u=np.asarray(_npy(folder/f'left_singularvectors_{l}.npy'))[:,:1]
                v=np.asarray(_npy(folder/f'right_singularvectors_{l}.npy'))[:,:1]
                u,v=_fix_svd_signs(u,v,prev); prev=(u.copy(),v.copy())
            else:
                u=np.asarray(_npy(folder/f'eigenvectors_{l}.npy'))[:,:1]
                u=_fix_signs(u,prev); v=u; prev=u.copy()
            if len(u)!=bins or len(v)!=bins: raise ValueError(f'Inconsistent vector bins {folder}, label {l}')
            i=get_bins(fields[pair[0]][:,l-begin],edges[pair[0]])
            j=get_bins(fields[pair[1]][:,l-begin],edges[pair[1]]) if mixed else i
            water=ocean.ravel(); good=water&(i>=0)&(j>=0)
            left=np.full(i.shape,np.nan); right=np.full(i.shape,np.nan); contribution=np.full(i.shape,np.nan)
            # Vector maps use the same joint-valid mask as current plotting.
            left[good]=u[i[good],0]; right[good]=v[j[good],0]
            contribution[good]=scalar[l-first,0,0]*left[good]*right[good]
            maps=[left,right,contribution] if mixed else [left,contribution]
            modes[l-first]=spatial_statistics(np.asarray(maps).reshape(-1,*ocean.shape),ocean)
            consumed.add(l); previous_label=l
    if consumed!=set(sources):
        missing=sorted(set(sources)-consumed)
        raise ValueError(f'{folder.name}: {len(missing)} spectra outside configured spectral periods; first {missing[:5]}. Match trend_spectral_periods to original quantile periods.')
    return labels,scalar,modes,names


def run_trends(root,start_year,end_year,*, sources=('Variables','Coefficients','Eigenvalues'),
               variables=VARIABLES, coefficient_pairs=COEFFICIENT_PAIRS,
               coefficients=('A','B'), derived_pair=('sensible','latent'), derived=('C','F'),
               spectral_pairs=SPECTRAL_PAIRS, method='sp', n_bins_2d=12,
               spectral_bins=144, spectral_periods=PERIODS, mode_extrema=True,
               windows=(30,365), min_coverage=.9, last_date=None, mask=None,
               chunk_days=16, overwrite=True, periods=PERIODS):
    """Recalculate trends from existing numerical arrays; never change inputs."""
    if method not in ('np','sp') or chunk_days<1 or not windows or any(w<1 for w in windows):
        raise ValueError('Invalid trends settings')
    if not 0<min_coverage<=1 or not set(sources)<= {'Variables','Coefficients','Derived','Eigenvalues'}:
        raise ValueError('Invalid sources or coverage')
    ocean=np.asarray(read_ocean_mask(root) if mask is None else mask,dtype=bool)
    first,stop=_bounds(start_year,end_year,last_date)
    results=[]
    def target(cat,ident):
        """Target the requested trends data using the supplied inputs."""
        return result_path(root,cat,ident,start_year,end_year,method=method,n_bins=n_bins_2d)
    def publish(path,labels,daily,ch,meta, spectral_min=False):
        """Publish the requested trends data using the supplied inputs."""
        results.append(_publish(path,labels,daily,ch,windows,first,min_coverage,meta,spectral_min=spectral_min))
    def reusable(path):
        """Check whether the existing saved result can be reused."""
        if path.exists() and not overwrite:
            results.append(path); return True
        return False
    if 'Variables' in sources:
        for name in variables:
            path=target('Variables',name)
            if reusable(path): continue
            chunks=[]; all_labels=[]
            for a,b,arr in _field_blocks(root,name,start_year,end_year,periods,ocean):
                begin=day_label(date(a,1,1)); end=begin+len(arr)
                if b!=end_year and end!=day_label(date(b,1,1)):
                    raise ValueError(f'{name} {a}-{b}: incomplete nonfinal variable period')
                lo,hi=max(first,begin),min(stop,end)
                if hi<=lo: continue
                chunks.append(daily_stats(arr,ocean,name,first=lo-begin,stop=hi-begin,chunk_days=chunk_days))
                all_labels.append(np.arange(lo,hi))
            if not chunks: raise ValueError(f'No observations for {name}')
            labels=np.concatenate(all_labels)
            if np.any(np.diff(labels)!=1): raise ValueError(f'{name}: gap/overlap in source dates')
            publish(path,labels,np.concatenate(chunks),(name,),dict(category='Variables',variable=name))
    for cat,pairs,keys in (('Coefficients',coefficient_pairs,coefficients),('Derived',(derived_pair,),derived)):
        if cat not in sources: continue
        for pair in pairs:
            for key in keys:
                path=target(cat,'-'.join(pair)+'_'+key)
                if reusable(path): continue
                file,begin,end=_coefficient_file(root,pair,key,method,n_bins_2d)
                arr=_npy(file)
                if len(arr)!=end-begin+1: raise ValueError(f'{file}: labels and length disagree')
                lo,hi=max(first,begin),min(stop,end+1)
                if hi<=lo: raise ValueError(f'{file}: no dates within request')
                daily=daily_stats(arr,ocean,f'{pair} {key}',first=lo-begin,stop=hi-begin,chunk_days=chunk_days)
                publish(path,np.arange(lo,hi),daily,_channels(pair,key),dict(category=cat,pair=list(pair),coefficient=key,method=method,n_bins=n_bins_2d,source=str(file)))
    if 'Eigenvalues' in sources:
        for pair in spectral_pairs:
            ident='-'.join(pair)
            p=target('Eigenvalues',ident+'_spectrum'); q=target('Eigenvalues',ident+'_mode1')
            if p.exists() and (not mode_extrema or (q.exists() and target('Eigenvalues',ident+'_contribution1').exists())) and not overwrite:
                results.append(p); results.extend([q,target('Eigenvalues',ident+'_contribution1')] if mode_extrema else []); continue
            labels,scalar,modes,ch=_spectral_daily(root,pair,first,stop,ocean=ocean,mode_extrema=mode_extrema,spectral_periods=spectral_periods,n_bins=spectral_bins)
            metadata=dict(category='Eigenvalues',pair=list(pair),spectral_kind='eigenvalue' if pair[0]==pair[1] else 'singular value',spectral_bins=spectral_bins, spectral_periods=list(spectral_periods))
            publish(p,labels,scalar,('leading value',),metadata)
            if modes is not None:
                # Historical min aggregation applies to vectors only; keep
                # contributions in a separate file with ordinary block means.
                publish(q,labels,modes[:,:-1],ch[:-1],dict(metadata,measurement='signed vector maps'),spectral_min=True)
                publish(target('Eigenvalues',ident+'_contribution1'),labels,modes[:,-1:],ch[-1:],dict(metadata,measurement='sign-invariant contribution'))
    return results
