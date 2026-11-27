"""Publication-style trends, rendered exclusively from cached trend statistics."""
from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from Data_processing.io import BASE_DATE
from Data_processing.trends import (VARIABLES, COEFFICIENT_PAIRS, SPECTRAL_PAIRS,
                                     result_path)

UNIT = {'Flux': r'W/m$^2$', 'sensible': r'W/m$^2$', 'latent': r'W/m$^2$',
        'SST': 'K', 'Pressure': 'Pa'}
RAW_COLORS = {'Flux': 'b', 'SST': 'r', 'Pressure': 'orange',
              'sensible': 'b', 'latent': 'orange'}
RU = {'max': 'макс.', 'min': 'мин.', 'mean': 'среднее', 'max - min': 'макс. − мин.',
      'max regression fit': 'тренд максимума', 'min regression fit': 'тренд минимума',
      'fitted trend': 'линейный тренд', 'leading value': 'ведущее значение',
      'left vector 1':'левый вектор 1', 'right vector 1':'правый вектор 1',
      'vector 1':'вектор 1', 'mode contribution 1':'вклад моды 1',
      'Flux':'Тепловой поток', 'SST':'ТПМ', 'Pressure':'Давление',
      'sensible':'Явный поток','latent':'Скрытый поток'}


def _load(path, window):
    """Compute load for trend plots from ``path``, ``window``."""
    if not path.is_file():
        raise FileNotFoundError(f'Missing calculated trends {path}; run main.py with task="trends" first')
    with np.load(path, allow_pickle=False) as z:
        if f'values_{window}' not in z:
            raise ValueError(f'{path}: window {window} not calculated')
        return dict(labels=z[f'labels_{window}'], values=z[f'values_{window}'],
                    fits=z[f'fits_{window}'], channels=z['channels'].tolist(),
                    metadata=json.loads(z['metadata'].item()))


def _plot_series(ax, record, channel, title, units, plot_kind, language, window,
                 *, color='b', equations=True):
    """Plot series for the trend plots workflow."""
    tr=lambda s: RU.get(s,s) if language=='ru' else s
    days=[BASE_DATE+timedelta(days=int(l)-1) for l in record['labels']]
    x=record['labels'].astype(float)-1
    values=record['values'][:,channel]; fits=record['fits'][channel]
    if not len(days):
        ax.text(.5,.5,'No complete windows' if language=='en' else 'Нет полных интервалов',
                ha='center',va='center',transform=ax.transAxes)
    def line(j, label, c, *, fit=None, fit_label=None):
        """Line the requested trend plots data using the supplied inputs."""
        ax.plot(days,values[:,j],color=c,alpha=.75 if j!=2 else 1,
                lw=1,label=tr(label))
        if fit is not None and np.isfinite(fits[j,:2]).all():
            k,b=fits[j,:2]
            y=np.where(np.isfinite(values[:,j]),k*x+b,np.nan)
            # Keep the fitted equation in the historical legend where requested.
            label=f'{k:.2e} * x {b:+.3g}' if equations else tr(fit_label or 'fitted trend')
            ax.plot(days,y,'--',color=fit,lw=1.2,label=label)
    if plot_kind=='range':
        line(3,'max - min',color,fit='darkviolet')
    elif plot_kind=='spectrum':
        line(2,title,'b',fit='darkviolet',fit_label='fitted trend')
    else:
        line(0,'max','r',fit='darkviolet',fit_label='max regression fit')
        line(1,'min','b',fit='orange',fit_label='min regression fit')
        line(2,'mean','g')
    if title and plot_kind!='spectrum':
        ax.set_title(tr(title),fontsize=20,fontweight='bold')
    if units: ax.set_ylabel(units,rotation=0,ha='right',va='center',labelpad=12)
    locator=mdates.AutoDateLocator(minticks=5,maxticks=12)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    if window==365: ax.xaxis.set_minor_locator(mdates.MonthLocator())
    if ax.lines: ax.legend(loc='upper left',bbox_to_anchor=(1.015,1),borderaxespad=0,fontsize=12)
    ax.grid(True,color='#dddddd',linewidth=.7)


def _render(rows, path, kind, language, window, *, dpi, equations=True, title=None):
    """Compute render for trend plots from ``rows``, ``path``, ``kind``, ``language``, ``window``."""
    if not rows: return None
    try:
        # seaborn whitegrid look retained from plot_extreme.py without changing globals.
        with plt.style.context('seaborn-v0_8-whitegrid'), plt.rc_context({'font.size':14}):
            fig, axes=plt.subplots(len(rows),1,figsize=(20,10 if len(rows)<=3 else 3.0*len(rows)),
                                   squeeze=False)
            try:
                for ax,(record,j,label,units,color) in zip(axes[:,0],rows):
                    _plot_series(ax,record,j,label,units,kind,language,window,color=color,equations=equations)
                if title: fig.suptitle(title,fontsize=20,fontweight='bold')
                fig.tight_layout()
                path.parent.mkdir(parents=True,exist_ok=True)
                fig.savefig(path,dpi=dpi,bbox_inches='tight',facecolor='white')
                print(f'[trends] Saved {path}',flush=True)
            finally:
                plt.close(fig)
    except OSError:
        raise
    return path


def _units(key,pair,j):
    """Compute units for trend plots from ``key``, ``pair``, ``j``."""
    if key=='A': return UNIT.get(pair[j],'')+r' / day'
    # B is the stored estimator, not silently squared or re-scaled.
    # Off-diagonal diffusion factors may have method-dependent meanings.
    if key in ('B','B_with_reg'): return ''
    return ''


def plot_trends(root,start_year,end_year,*, sources=('Variables','Coefficients','Eigenvalues'),
                variables=VARIABLES, coefficient_pairs=COEFFICIENT_PAIRS,
                coefficients=('A','B'), derived_pair=('sensible','latent'), derived=('C','F'),
                spectral_pairs=SPECTRAL_PAIRS, method='sp', n_bins_2d=12,
                mode_extrema=True, windows=(30,365), language='en', dpi=300):
    """Plot trends for the trend plots workflow."""
    if language not in ('en','ru'): raise ValueError('language must be en or ru')
    root=Path(root); period=f'{start_year}-{end_year}'; outputs=[]
    base=root/'Plots'/'Trends'/period
    def load(cat,ident,w):
        """Load the requested trend plots data using the supplied inputs."""
        return _load(result_path(root,cat,ident,start_year,end_year,method=method,n_bins=n_bins_2d),w)
    def render(rows,cat,name,w,kind,**kw):
        """Render the requested trend plots data using the supplied inputs."""
        dest=base/cat
        if cat in ('Coefficients','Derived'): dest=dest/f'{n_bins_2d}_bins'/method
        p=_render(rows,dest/f'{name}_{w}d_{language}.png',kind,language,w,dpi=dpi,**kw)
        if p: outputs.append(p)
    for w in windows:
        if 'Variables' in sources:
            for group in (tuple(v for v in ('Flux','SST','Pressure') if v in variables),
                          tuple(v for v in variables if v not in ('Flux','SST','Pressure'))):
                if not group: continue
                rows=[(load('Variables',v,w),0,v,UNIT.get(v,''),RAW_COLORS.get(v,'b')) for v in group]
                label='-'.join(group)
                render(rows,'Variables',label+'_range',w,'range')
                render(rows,'Variables',label+'_extrema',w,'extrema')
        for cat,pairs,keys in (('Coefficients',coefficient_pairs,coefficients),('Derived',(derived_pair,),derived)):
            if cat not in sources: continue
            cache={}
            for pair in pairs:
                for key in keys:
                    rec=load(cat,'-'.join(pair)+'_'+key,w); cache[(pair,key)]=rec
                    rows=[(rec,j,ch,_units(key,pair,j),'b') for j,ch in enumerate(rec['channels'])]
                    render(rows,cat,'-'.join(pair)+'_'+key,w,'extrema')
            # Fig. 5: old plot_extreme_3d selected Flux/SST from Flux-SST,
            # Pressure from SST-Pressure. Do not average across different pairs.
            if cat=='Coefficients' and 'A' in keys and all((p,'A') in cache for p in (('Flux','SST'),('SST','Pressure'))):
                rows=[]
                for p,j in ((('Flux','SST'),0),(('Flux','SST'),1),(('SST','Pressure'),1)):
                    rows.append((cache[(p,'A')],j,p[j],_units('A',p,j),'b'))
                render(rows,cat,'A_Flux-SST-Pressure',w,'extrema')
        if 'Eigenvalues' in sources:
            rows=[]
            for pair in spectral_pairs:
                ident='-'.join(pair); rec=load('Eigenvalues',ident+'_spectrum',w)
                unit=UNIT[pair[0]]+r' × '+UNIT[pair[1]]+r' / day'
                label=ident+(' '+r'$\lambda_1$' if pair[0]==pair[1] else ' '+r'$\sigma_1$')
                rows.append((rec,0,label,unit,'b'))
                if mode_extrema:
                    rec=load('Eigenvalues',ident+'_mode1',w)
                    mode_rows=[(rec,j,ch,'','b') for j,ch in enumerate(rec['channels'])]
                    render(mode_rows,'Eigenvalues',ident+'_vector1',w,'extrema',equations=False)
                    rec=load('Eigenvalues',ident+'_contribution1',w)
                    render([(rec,0,'mode contribution 1',unit,'b')], 'Eigenvalues',ident+'_contribution1',w,'extrema',equations=False)
            # Original six-row panel order from Figs. 8–9.
            order=['Flux-Flux','Flux-SST','SST-SST','SST-Pressure','Pressure-Pressure','Flux-Pressure']
            rows.sort(key=lambda x: next((i for i,s in enumerate(order) if x[2].startswith(s+' ')),len(order)))
            title=(f'Eigenvalues and singular values trends, mean of every {w} days' if language=='en'
                   else f'Тренды собственных и сингулярных значений, усреднение {w} дней')
            render(rows,'Eigenvalues','leading_spectrum',w,'spectrum',equations=False,title=title)
    return outputs
