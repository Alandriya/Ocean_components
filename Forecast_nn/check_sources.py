"""Show exactly which 1979--2026 data/SDE files Forecast_nn will use.

Run from the project root:
    python -m Forecast_nn.check_sources

This performs lightweight header validation only; it never trains models or
loads multi-year arrays into RAM.
"""
from pathlib import Path
import numpy as np

from Forecast_nn.config import cfg
from Forecast_nn.source_paths import grouped_length


def main():
    print(f'Forecast dataset: {cfg.period}, base={cfg.base_date}, '
          f'last observed day={cfg.data_last_date}', flush=True)
    print(f'Root: {cfg.root_path}', flush=True)
    failures = []
    lengths = {}
    for name, path in cfg.data_files.items():
        p = Path(path)
        if not p.is_file():
            print(f'[DATA] {name:9s} MISSING  {p}', flush=True)
            if name in ('sensible', 'latent'):
                failures.append(f'{name} field missing')
            continue
        try:
            length = grouped_length(p)
            lengths[name] = length
            print(f'[DATA] {name:9s} OK       days={length}, {p}', flush=True)
        except (ValueError, OSError) as e:
            failures.append(f'{name}: {e}')
            print(f'[DATA] {name:9s} INVALID  {e}', flush=True)
    if 'sensible' in lengths and 'latent' in lengths and lengths['sensible'] != lengths['latent']:
        failures.append(f'Unequal sensible/latent lengths: {lengths}')

    for name in ('sensible', 'latent', 'Flux', 'SST', 'Pressure'):
        source = cfg.sde_sources[name]
        pa, pb = Path(source['a_file']), Path(source['b_file'])
        if not (pa.is_file() and pb.is_file()):
            print(f'[1D SDE] {name:9s} NOT ASSEMBLED  A={pa} B={pb}', flush=True)
            # Missing coefficients are acceptable for models without SDE.
            continue
        try:
            a = np.load(pa, mmap_mode='r', allow_pickle=False)
            b = np.load(pb, mmap_mode='r', allow_pickle=False)
            shape_a, shape_b = a.shape, b.shape
            del a, b
            if shape_a != shape_b or len(shape_a) != 3 or shape_a[1:] != (161, 181):
                raise ValueError(f'expected A/B as (time,161,181), got {shape_a}, {shape_b}')
            if name in lengths and shape_a[0] != lengths[name]-1:
                raise ValueError(f'field length {lengths[name]} vs SDE transitions {shape_a[0]}')
            print(f'[1D SDE] {name:9s} OK       transitions={shape_a[0]}, offset={source["time_offset"]}', flush=True)
        except (OSError, ValueError) as e:
            failures.append(f'{name} SDE: {e}')
            print(f'[1D SDE] {name:9s} INVALID  {e}', flush=True)

    source = cfg.joint_sde_sources
    pa, pb = Path(source['a_file']), Path(source['b_file'])
    if not (pa.is_file() and pb.is_file()):
        print(f'[2D SDE] NOT ASSEMBLED  A={pa} B={pb}', flush=True)
    else:
        try:
            a = np.load(pa, mmap_mode='r', allow_pickle=False)
            b = np.load(pb, mmap_mode='r', allow_pickle=False)
            shape_a, shape_b = a.shape, b.shape
            del a, b
            if len(shape_a) != 4 or shape_a[1:] != (2, 161, 181):
                raise ValueError(f'expected A (time,2,161,181), got {shape_a}')
            if shape_b != (shape_a[0], 4, 161, 181):
                raise ValueError(f'expected B (time,4,161,181), got {shape_b}')
            if 'sensible' in lengths and shape_a[0] != lengths['sensible']-1:
                raise ValueError('2D coefficient length does not match flux history')
            print(f'[2D SDE] OK       transitions={shape_a[0]}, '
                  f'B layout={source["b_layout"]}', flush=True)
        except (OSError, ValueError) as e:
            failures.append(f'Joint SDE: {e}')
            print(f'[2D SDE] INVALID  {e}', flush=True)
    if failures:
        raise SystemExit('Input validation failed: ' + '; '.join(failures))
    print('Input check completed. Missing optional SDE files are reported above.', flush=True)


if __name__ == '__main__':
    main()
