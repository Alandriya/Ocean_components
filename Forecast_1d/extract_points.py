"""Extract aligned daily sensible/latent series and ocean-only neighborhoods.

Run from the repository root:
    python -m Forecast_1d.extract_points --dry-run
    python -m Forecast_1d.extract_points

Only numpy and the root project_paths.py are needed. Sources are memory-mapped,
not copied into RAM. Output is built in a temporary directory and published
only after all arrays and the manifest have been saved successfully.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
import uuid
from datetime import date, timedelta
from pathlib import Path

import numpy as np

GRID = (161, 181)
PERIOD = '1979-2026'
BASE_DATE = date(1979, 1, 1)
CHANNEL_NAMES = ('sensible', 'latent')
PATCH_WIDTH = 15
CHECK_WIDTH = 21
STAT_WIDTHS = (3, 7, 15)
from .experiment import DEFAULT_SUBDIR


def _read_project_data_root() -> Path:
    from project_paths import DATA_ROOT
    return Path(DATA_ROOT).expanduser()


def _source_path(root: Path, channel: str) -> Path:
    return root / 'DATA' / 'Fluxes' / f'{channel}_{PERIOD}_grouped.npy'


def _inspect_source(path: Path) -> tuple[np.ndarray, str, int]:
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}')
    source = np.load(path, mmap_mode='r', allow_pickle=False)
    if source.dtype.kind not in 'biuf':
        raise TypeError(f'Expected numeric NPY in {path}, received {source.dtype}')
    height, width = GRID
    size = height * width
    if source.ndim == 3 and source.shape[1:] == GRID:
        layout, length = 'time_hw', int(source.shape[0])
    elif source.ndim == 2 and source.shape[0] == size and source.shape[1] != size:
        layout, length = 'space_time', int(source.shape[1])
    elif source.ndim == 2 and source.shape[1] == size and source.shape[0] != size:
        layout, length = 'time_space', int(source.shape[0])
    else:
        raise ValueError(
            f'Unsupported {path.name} shape {source.shape}; expected '
            f'[T,161,181], [29141,T] or [T,29141]')
    if length < 33:
        raise ValueError(f'Input has only {length} days: {path}')
    return source, layout, length


def _mask_from_bytes(path: Path) -> tuple[np.ndarray, str]:
    if not path.is_file():
        raise FileNotFoundError(f'Missing ocean mask: {path}')
    content = path.read_bytes()
    if len(content) != GRID[0] * GRID[1]:
        raise ValueError(f'Expected exactly {GRID[0] * GRID[1]} mask bytes, got {len(content)}: {path}')
    values = np.frombuffer(content, dtype=np.uint8)
    if np.any((values != 0) & (values != 1)):
        raise ValueError('Ocean mask must contain 0 for land and 1 for water')
    return values.reshape(GRID).astype(bool, copy=True), hashlib.sha256(content).hexdigest()


def _validate_points(path: Path, water: np.ndarray):
    payload = json.loads(path.read_text(encoding='utf-8'))
    if tuple(payload.get('grid_shape', ())) != GRID:
        raise ValueError(f'Point manifest expected native GRID={GRID}: {path}')
    entries = payload['points']
    if not entries or [p['id'] for p in entries] != [f'P{i}' for i in range(1, len(entries)+1)]:
        raise ValueError('Expected ordered, unique point IDs P1..Pn')
    if len({(p['grid_y'],p['grid_x']) for p in entries}) != len(entries):
        raise ValueError('Point coordinates must be unique')
    points = []
    for item in entries:
        y, x = int(item['grid_y']), int(item['grid_x'])
        if not (0 <= y < GRID[0] and 0 <= x < GRID[1]):
            raise ValueError(f'Point {item["id"]} outside map: {(y,x)}')
        lat = 80.0 - 0.5 * y
        lon = -90.0 + 0.5 * x
        if not (abs(lat - float(item['lat_deg'])) < 1e-7 and
                abs(lon - float(item['lon_deg'])) < 1e-7):
            raise ValueError(f'Point {item["id"]} geographical coordinates inconsistent with native grid')
        half = CHECK_WIDTH // 2
        if y - half < 0 or x - half < 0 or y + half >= GRID[0] or x + half >= GRID[1]:
            raise ValueError(f'Point {item["id"]} too close to domain edge for {CHECK_WIDTH}x{CHECK_WIDTH}')
        patch = water[y-half:y+half+1, x-half:x+half+1]
        if patch.shape != (CHECK_WIDTH, CHECK_WIDTH) or not patch.all():
            raise ValueError(f'Point {item["id"]} is not surrounded by all-water {CHECK_WIDTH}x{CHECK_WIDTH} patch')
        points.append(item)
    return points


def _unique_pixels(points):
    radius = PATCH_WIDTH // 2
    rows, cols = np.mgrid[-radius:radius+1, -radius:radius+1]
    flat = np.stack([
        (int(p['grid_y']) + rows) * GRID[1] + (int(p['grid_x']) + cols)
        for p in points
    ], axis=0)
    unique, inverse = np.unique(flat.reshape(-1), return_inverse=True)
    return unique, inverse


def _read_unique_pixels(source, layout, begin: int, end: int, unique: np.ndarray):
    """Return [days, distinct selected grid cells] without loading full history."""
    if layout == 'space_time':
        values = source[unique, begin:end].T
    elif layout == 'time_space':
        values = source[begin:end, unique]
    elif layout == 'time_hw':
        values = source[begin:end].reshape(end-begin, GRID[0]*GRID[1])[:, unique]
    else:
        raise AssertionError(f'Unknown layout: {layout}')
    return np.asarray(values, dtype=np.float32)


def _nanmean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[T,P,K,K] -> [T,P] means and population SD; no invalid-value imputation."""
    valid = np.isfinite(values)
    count = valid.sum(axis=(-1, -2))
    v = np.where(valid, values, 0).astype(np.float64)
    sums = v.sum(axis=(-1, -2))
    means = np.full(count.shape, np.nan, dtype=np.float64)
    np.divide(sums, count, out=means, where=count > 0)
    second = np.full(count.shape, np.nan, dtype=np.float64)
    np.divide((v*v).sum(axis=(-1,-2)), count, out=second, where=count > 0)
    var = np.maximum(0.0, second-means*means)
    return means.astype(np.float32), np.sqrt(var).astype(np.float32)


def _write_points_csv(path: Path, points):
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['id','region','grid_y','grid_x','lat_deg','lon_deg','water_cells_21x21'])
        for item in points:
            writer.writerow([item['id'], item['region'], item['grid_y'], item['grid_x'],
                             item['lat_deg'], item['lon_deg'], CHECK_WIDTH**2])


def _close_output_memmaps(memmap_arrays):
    """Flush and release every output mapping before Windows directory operations.

    Deleting Python references alone is not sufficient if a loop variable still
    holds the final memmap. Explicitly closing the underlying mappings avoids
    WinError 5 / WinError 32 when renaming or cleaning up staging directories.
    This helper is only used after all writes to these arrays have finished.
    """
    first_error = None
    for array in memmap_arrays:
        try:
            array.flush()
            mapping = getattr(array, '_mmap', None)
            if mapping is not None and not mapping.closed:
                mapping.close()
        except Exception as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


def _rename_with_retry(source: Path, destination: Path, *, attempts: int = 6):
    """Allow transient Windows indexing/antivirus locks to disappear."""
    for attempt in range(attempts):
        try:
            source.rename(destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(min(0.2 * (attempt + 1), 1.0))


def extract(data_root: Path, points_file: Path, destination: Path, *,
            chunk_days: int = 64, overwrite: bool = False, dry_run: bool = False):
    root = Path(data_root).expanduser()
    destination = Path(destination).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f'DATA_ROOT is not a directory: {root}. Check project_paths.py')
    if chunk_days < 1:
        raise ValueError('chunk_days must be >=1')
    mask_path = root / 'DATA' / 'mask'
    water, mask_sha256 = _mask_from_bytes(mask_path)
    points = _validate_points(points_file, water)
    inputs = {}
    lengths = {}
    for channel in CHANNEL_NAMES:
        path = _source_path(root, channel)
        data, layout, length = _inspect_source(path)
        inputs[channel] = (path, data, layout)
        lengths[channel] = length
        print(f'[source] {channel}: {path} shape={data.shape}, layout={layout}, dtype={data.dtype}', flush=True)
    if lengths['sensible'] != lengths['latent']:
        raise ValueError(f'Sensible/latent dates do not align: {lengths}')
    n_time = lengths['sensible']
    last_date = BASE_DATE + timedelta(days=n_time-1)
    if last_date.year > 2026:
        raise ValueError(f'Period {PERIOD} input includes days after 2026: last={last_date}')
    unique, inverse = _unique_pixels(points)
    bytes_est = n_time * len(points) * len(CHANNEL_NAMES) * PATCH_WIDTH**2 * 4
    print(f'[preflight] {n_time} days, {BASE_DATE}..{last_date}; points={len(points)}, '
          f'water_21x21=OK; patches={PATCH_WIDTH}x{PATCH_WIDTH}; '
          f'~{bytes_est / 2**20:.1f} MiB main array; output={destination}', flush=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f'Output already exists: {destination}. Use --overwrite explicitly to rebuild.')
    if dry_run:
        print('[dry-run] Inputs and offshore masks validated; no output files created.', flush=True)
        return {'n_days': n_time, 'last_date': last_date.isoformat(), 'destination': str(destination)}

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f'.{destination.name}.building-{uuid.uuid4().hex[:12]}'
    staging.mkdir()
    backup = None
    publishing = False
    published = False
    try:
        shape = (n_time, len(points), len(CHANNEL_NAMES), PATCH_WIDTH, PATCH_WIDTH)
        writers = []

        def create_writer(filename, array_shape):
            array = np.lib.format.open_memmap(staging / filename, mode='w+',
                                               dtype=np.float32, shape=array_shape)
            writers.append(array)
            return array

        try:
            patches = create_writer('patches_15x15.npy', shape)
            centers = create_writer('centers.npy', (n_time, len(points), len(CHANNEL_NAMES)))
            means = {n: create_writer(f'mean_{n}x{n}.npy', centers.shape)
                     for n in STAT_WIDTHS}
            stds = {n: create_writer(f'std_{n}x{n}.npy', centers.shape)
                    for n in STAT_WIDTHS}
            valid_centers = np.zeros((len(points),len(CHANNEL_NAMES)),dtype=np.int64)
            valid_patch = np.zeros((len(points),len(CHANNEL_NAMES)),dtype=np.int64)
            for begin in range(0, n_time, chunk_days):
                end = min(begin + chunk_days, n_time)
                for c, channel in enumerate(CHANNEL_NAMES):
                    _, data, layout = inputs[channel]
                    selected = _read_unique_pixels(data, layout, begin, end, unique)
                    local = selected[:, inverse].reshape(end-begin, len(points), PATCH_WIDTH, PATCH_WIDTH)
                    patches[begin:end, :, c] = local
                    central = local[..., PATCH_WIDTH//2, PATCH_WIDTH//2]
                    centers[begin:end, :, c] = central
                    valid_centers[:,c] += np.isfinite(central).sum(axis=0)
                    valid_patch[:,c] += np.isfinite(local).sum(axis=(0,2,3))
                    for w in STAT_WIDTHS:
                        offset = (PATCH_WIDTH-w)//2
                        selected_patch = local[..., offset:offset+w, offset:offset+w]
                        avg, sd = _nanmean_std(selected_patch)
                        means[w][begin:end, :, c] = avg
                        stds[w][begin:end, :, c] = sd
                if begin == 0 or end == n_time or end % (chunk_days*20) == 0:
                    print(f'[extract] days {end}/{n_time} ({100*end/n_time:.1f}%)', flush=True)
        finally:
            # Even if the loop fails, release every writable mapping before
            # Windows attempts to delete or rename the temporary directory.
            _close_output_memmaps(writers)
            writers.clear()

        dates = np.datetime64(BASE_DATE.isoformat(), 'D') + np.arange(n_time, dtype='timedelta64[D]')
        np.save(staging/'dates.npy', dates, allow_pickle=False)
        np.save(staging/'patch_mask_15x15.npy',
                np.ones((len(points), PATCH_WIDTH, PATCH_WIDTH), dtype=np.uint8), allow_pickle=False)
        _write_points_csv(staging/'points.csv', points)
        shutil.copyfile(points_file, staging/'points.json')
        metadata = {
            'description': f'Unnormalized raw daily flux series; {len(points)} offshore locations; no train/val/test split',
            'base_date': BASE_DATE.isoformat(),
            'last_date': last_date.isoformat(),
            'n_dates': n_time,
            'channels': list(CHANNEL_NAMES),
            'grid_shape': list(GRID),
            'grid_axes': 'row 0 = 80N, column 0 = 90W; 0.5-degree spacing',
            'points': [{'id': p['id'], 'region': p['region'], 'grid_y':p['grid_y'], 'grid_x':p['grid_x'],
                        'lat_deg':p['lat_deg'], 'lon_deg':p['lon_deg']} for p in points],
            'shapes': {'patches_15x15.npy':list(shape), 'centers.npy':[n_time, len(points), 2],
                       'dates.npy':[n_time], 'patch_mask_15x15.npy':[len(points),15,15]},
            'stats_windows': list(STAT_WIDTHS),
            'mask': {'path':str(mask_path), 'sha256':mask_sha256, 'semantics':'1=water, 0=land',
                     'water_21x21_verified':True},
            'sources': {channel:{'path':str(inputs[channel][0]), 'layout':inputs[channel][2],
                        'dtype':str(inputs[channel][1].dtype), 'size_bytes':inputs[channel][0].stat().st_size}
                        for channel in CHANNEL_NAMES},
            'finite_fraction_centers': (valid_centers/n_time).tolist(),
            'finite_fraction_patches_15x15':(valid_patch/(n_time*PATCH_WIDTH**2)).tolist(),
            'dtype': 'float32',
            'value_processing': 'original physical values cast to float32; NaN/inf preserved; no normalization/filling',
            'point_order': f'P1 through P{len(points)}',
            'files': ['patches_15x15.npy','centers.npy', 'dates.npy', 'patch_mask_15x15.npy',
                      'points.csv','points.json', 'manifest.json'] +
                     [f'{kind}_{w}x{w}.npy' for kind in ('mean','std') for w in STAT_WIDTHS],
        }
        (staging/'manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
        publishing = True
        if destination.exists():
            if not overwrite:
                raise FileExistsError(f'Output already exists: {destination}')
            backup = destination.parent / f'.{destination.name}.previous-{uuid.uuid4().hex[:12]}'
            _rename_with_retry(destination, backup)
        try:
            _rename_with_retry(staging, destination)
        except BaseException:
            if backup is not None and not destination.exists() and backup.exists():
                _rename_with_retry(backup, destination)
                backup = None
            raise
        published = True
        if backup is not None:
            try:
                shutil.rmtree(backup)
            except OSError as exc:
                print(f'[warning] Previous output could not be removed: {backup}: {exc}', flush=True)
        print(f'[DONE] Saved {destination}', flush=True)
        return metadata
    finally:
        # If publication fails, keep the fully written staging directory for
        # later validation/recovery rather than deleting the user's work.
        if staging.exists():
            if publishing and not published:
                print(f'[recovery] Output is available in staging: {staging}', flush=True)
            else:
                try:
                    shutil.rmtree(staging)
                except OSError as exc:
                    print(f'[warning] Could not clean temporary directory {staging}: {exc}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=None,
                        help='Optional explicit data root (default: project_paths.DATA_ROOT)')
    parser.add_argument('--expected-data-root', type=Path, default=None,
                        help='Fail fast if configured DATA_ROOT is not this directory')
    parser.add_argument('--points', type=Path, default=Path(__file__).with_name('points.json'))
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--chunk-days', type=int, default=64)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    data_root = args.data_root if args.data_root is not None else _read_project_data_root()
    if args.expected_data_root is not None:
        actual = os.path.normcase(os.path.abspath(os.fspath(data_root)))
        desired = os.path.normcase(os.path.abspath(os.fspath(args.expected_data_root)))
        if actual != desired:
            parser.error(f'DATA_ROOT mismatch: project_paths.py points to {data_root}; '
                         f'expected {args.expected_data_root}. Set active DATA_ROOT in project_paths.py.')
    if args.output is None and args.points.resolve() != Path(__file__).with_name('points.json').resolve():
        parser.error('A custom --points file requires an explicit --output directory')
    destination = args.output or data_root / 'DATA' / 'Forecast_1d' / DEFAULT_SUBDIR
    extract(data_root, args.points, destination, chunk_days=args.chunk_days,
            overwrite=args.overwrite, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
