"""Inspect and optionally publish a completed Windows extraction staging folder.

Use from the repository root on the computer that ran extraction:
    python -m Forecast_1d.recover_staging
    python -m Forecast_1d.recover_staging --publish

This tool never overwrites a finished output and never deletes staging data.
It only publishes a directory when every expected file and selected array
values have passed validation.
"""
from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .extract_points import (
    DEFAULT_SUBDIR, PATCH_WIDTH, STAT_WIDTHS,
    _read_project_data_root, _rename_with_retry,
)


@contextmanager
def _open_array(path: Path):
    """Open array for the recover staging workflow."""
    data = np.load(path, mmap_mode='r', allow_pickle=False)
    try:
        yield data
    finally:
        mapping = getattr(data, '_mmap', None)
        if mapping is not None and not mapping.closed:
            mapping.close()


def validate_staging(staging: Path) -> tuple[bool, str]:
    """Check expected files, dates, array dimensions, and sample contents."""
    manifest_path = staging / 'manifest.json'
    if not manifest_path.is_file():
        return False, 'manifest.json is missing'
    try:
        metadata = json.loads(manifest_path.read_text(encoding='utf-8'))
        n = int(metadata['n_dates'])
        points = json.loads((staging / 'points.json').read_text(encoding='utf-8'))['points']
        p = len(points)
        if not p or [v['id'] for v in points] != [f'P{i}' for i in range(1,p+1)]:
            return False, 'invalid point identities'
        if [v['id'] for v in metadata['points']] != [v['id'] for v in points]:
            return False, 'point manifest mismatch'
        if n <= 0 or metadata['channels'] != ['sensible', 'latent']:
            return False, 'invalid manifest dates/channels'
        expected = set(metadata['files'])
        expected.update({'manifest.json', 'centers.npy', 'patches_15x15.npy', 'dates.npy'})
        missing = sorted(name for name in expected if not (staging / name).is_file())
        if missing:
            return False, f'missing files: {", ".join(missing)}'
        with _open_array(staging / 'dates.npy') as dates:
            if dates.shape != (n,) or dates.dtype.kind != 'M':
                return False, 'invalid dates shape/dtype'
            if (str(dates[0]) != metadata['base_date'] or
                    str(dates[-1]) != metadata['last_date']):
                return False, 'dates do not match manifest'
        with _open_array(staging / 'patches_15x15.npy') as patches:
            if patches.shape != (n, p, 2, PATCH_WIDTH, PATCH_WIDTH):
                return False, 'invalid patches array shape'
            if patches.dtype != np.float32:
                return False, 'invalid patches array dtype'
            samples = sorted(set((0, n // 2, n - 1)))
            sampled = np.array(patches[samples], copy=True)
        with _open_array(staging / 'centers.npy') as centers:
            if centers.shape != (n, p, 2):
                return False, 'invalid centers array shape'
            center_samples = np.asarray(centers[samples])
            if not np.array_equal(sampled[..., 7, 7], center_samples, equal_nan=True):
                return False, 'center values do not match patch centers'
        for width in STAT_WIDTHS:
            i = (PATCH_WIDTH - width) // 2
            patch = sampled[..., i:i+width, i:i+width]
            finite = np.isfinite(patch)
            counts = finite.sum(axis=(-2,-1))
            numeric = np.where(finite, patch.astype(np.float64), 0.0)
            sums = numeric.sum(axis=(-2,-1))
            means = np.full(sums.shape, np.nan)
            np.divide(sums, counts, out=means, where=counts > 0)
            variances = np.full(sums.shape, np.nan)
            np.divide((numeric * numeric).sum(axis=(-2,-1)), counts,
                      out=variances, where=counts > 0)
            sds = np.sqrt(np.maximum(0.0, variances - means * means))
            for kind, check in (('mean', means), ('std', sds)):
                with _open_array(staging / f'{kind}_{width}x{width}.npy') as arr:
                    if arr.shape != (n, p, 2):
                        return False, f'invalid {kind}_{width}x{width}.npy shape'
                    if not np.allclose(arr[samples], check, atol=1e-4, rtol=1e-4,
                                       equal_nan=True):
                        return False, f'{kind}_{width}x{width}.npy samples disagree with patches'
        with _open_array(staging / 'patch_mask_15x15.npy') as mask:
            if mask.shape != (p, PATCH_WIDTH, PATCH_WIDTH) or not np.all(mask == 1):
                return False, 'invalid patch mask'
    except (OSError, ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError) as exc:
        return False, f'validation failed: {exc}'
    return True, f'complete dataset with {n} daily observations'


def recover(*, destination: Path, staging: Path | None = None, publish: bool = False):
    """Recover the requested recover staging data using the supplied inputs."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f'Finished destination already exists: {destination}')
    candidates = ([Path(staging)] if staging is not None else
                  sorted(destination.parent.glob(f'.{destination.name}.building-*')))
    if not candidates:
        print('[inspect] No staging directories found; rerun extraction with python main.py.')
        return False
    valid = []
    for folder in candidates:
        if not folder.is_dir():
            print(f'[inspect] Missing staging directory: {folder}')
            continue
        good, message = validate_staging(folder)
        print(f'[inspect] {"COMPLETE" if good else "INCOMPLETE"}: {folder} ({message})')
        if good:
            valid.append(folder)
    if not publish:
        print('[inspect] No files modified. Pass --publish to finalize a complete staging directory.')
        return bool(valid)
    if len(valid) != 1:
        raise RuntimeError(f'Expected exactly one complete staging directory, found {len(valid)}. '
                           'Pass --staging PATH to select one, or rerun extraction.')
    _rename_with_retry(valid[0], destination)
    print(f'[DONE] Recovered completed data in {destination}')
    return True


def main(argv=None):
    """Run the recover staging workflow using its configured options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=None,
                        help='Final output folder (default: DATA_ROOT/DATA/Forecast_1d/...)')
    parser.add_argument('--staging', type=Path, default=None,
                        help='Inspect only this staging folder')
    parser.add_argument('--publish', action='store_true',
                        help='Publish one complete staged dataset; never overwrite an output')
    args = parser.parse_args(argv)
    root = _read_project_data_root()
    destination = (args.output if args.output is not None else
                   root / 'DATA' / 'Forecast_1d' / DEFAULT_SUBDIR)
    ok = recover(destination=destination, staging=args.staging, publish=args.publish)
    if args.publish and not ok:
        parser.exit(2)


if __name__ == '__main__':
    main()
