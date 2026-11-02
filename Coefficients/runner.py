"""Function-only orchestration for aligned NP/SP experiments and group diagnostics.

Import from project-root main.py; no CLI, argument parsing or automatic execution.
All existing daily coefficient output layouts are preserved.
"""
from pathlib import Path
import csv
import json
import os
import tempfile
import time
from typing import Iterable, Tuple

import numpy as np
from Coefficients.coefficients_cf import count_cf_from_daily
from Coefficients.common import coerce_series, load_water_mask, load_pair_series, prepare_pair_transition
from Coefficients.nonparametric import count_1d_nonparam, count_2d_nonparam
from Coefficients.semiparametric import count_semi_1d_AB, count_semi_2d_AB


def _validate_methods(methods):
    """Accept np/sp/both, or an iterable of np and sp identifiers."""
    if isinstance(methods, str):
        chosen = ("np", "sp") if methods == "both" else (methods,)
    else:
        chosen = tuple(dict.fromkeys(methods))
    if not chosen or any(m not in ("np", "sp") for m in chosen):
        raise ValueError("methods must be 'np', 'sp', 'both' or a nonempty subset of ('np','sp')")
    return chosen


def _daily_result_specs(prefix, base, names, n_bins, n_bins_2d,
                        methods, run_1d, run_2d, spatial_shape):
    """Describe every required output for each enabled estimator and method."""
    specs = []
    if run_1d:
        for variable in names:
            branch = prefix / base / f"1d_bins_{n_bins}" / variable
            if "np" in methods:
                specs.append((
                    branch / "Nonparametric" / "daily", ("A", "B"),
                    tuple(spatial_shape),
                ))
            if "sp" in methods:
                specs.append((
                    branch / "Semiparametric" / "daily",
                    ("A", "B", "B_with_reg"), tuple(spatial_shape),
                ))
    if run_2d:
        branch = prefix / base / f"2d_bins_{n_bins_2d}" / f"{names[0]}_{names[1]}"
        if "np" in methods:
            specs.append((
                branch / "Nonparametric_2d" / "daily", ("A", "B"),
                ((2, *spatial_shape), (4, *spatial_shape)),
            ))
        if "sp" in methods:
            specs.append((
                branch / "Semiparametric_2d" / "daily",
                ("A", "B", "B_with_reg"),
                ((2, *spatial_shape), (4, *spatial_shape), (4, *spatial_shape)),
            ))
    return specs


def _daily_file_complete(file_path: Path, expected_shape) -> bool:
    """Reject absent, truncated, or shape-incompatible daily .npy files."""
    if not file_path.is_file():
        return False
    try:
        array = np.load(file_path, mmap_mode="r", allow_pickle=False)
        ready = array.shape == tuple(expected_shape) and np.issubdtype(array.dtype, np.number)
        del array
        return bool(ready)
    except (OSError, ValueError, EOFError, TypeError):
        return False


def _missing_daily_outputs(spec, first_transition, end_transition, start_index,
                           remove_invalid=False):
    """Detect incomplete files; optionally remove every malformed file in a chunk."""
    folder, fields, shapes = spec
    incomplete = False
    for t in range(first_transition, end_transition):
        label = t + start_index
        for index, field in enumerate(fields):
            expected = shapes[index] if isinstance(shapes[0], tuple) else shapes
            filename = folder / f"{field}_{label}.npy"
            if not _daily_file_complete(filename, expected):
                incomplete = True
                if remove_invalid and filename.exists():
                    filename.unlink()
                if not remove_invalid:
                    return True
    return incomplete


def _first_unfinished_transition(specs, n_observations, start_index):
    """Find the earliest transition missing any requested NP/SP output."""
    for t in range(1, n_observations):
        if any(_missing_daily_outputs(spec, t, t + 1, start_index) for spec in specs):
            return t
    return None


def run_decade_coefficients(
    files_path_prefix: str,
    array1: np.ndarray,
    array2: np.ndarray,
    mask: np.ndarray,
    names: Tuple[str, str],
    start_year: int,
    end_year: int,
    start_index: int,
    n_bins: int = 250,
    n_components: int = 3,
    dt: float = 1.0,
    overwrite: bool = False,
    chunk_transitions: int = 7,
    min_group_size: int = 2,
    n_init: int = 10,
    run_1d: bool = True,
    run_2d: bool = False,
    n_bins_2d: int = 12,
    methods: Tuple[str, ...] = ("np", "sp"),
    compute_cf: bool = False,
    correlation_window: int = 14,
    mean_width: int = 7,
    resume: bool = True,
) -> None:
    """Run aligned estimators and resume at the first unfinished transition.

    With overwrite=False and resume=True, an existing daily output counts as
    complete only if every expected .npy file is loadable and has the expected
    shape. A partially written method/day is recomputed by that method only.
    Later complete days are still skipped by the underlying daily estimators.
    With overwrite=True, all requested days and methods are recalculated.

    The output label for transition X[t-1] -> X[t] is t + start_index.
    """
    methods = _validate_methods(methods)
    x0, x1 = coerce_series(array1), coerce_series(array2)
    if x0.shape != x1.shape:
        raise ValueError(f'Input series must have identical shapes, got {x0.shape} and {x1.shape}')
    if x0.shape[0] < 2:
        raise ValueError('At least two observations are required')
    if len(names) != 2 or not all(isinstance(n, str) and n for n in names):
        raise ValueError('names must contain two nonempty variable names')
    if names[0] == names[1]:
        raise ValueError('Variable names must differ to avoid overwriting 1D output paths')
    if not isinstance(chunk_transitions, int) or chunk_transitions < 1:
        raise ValueError('chunk_transitions must be a positive integer')
    if not run_1d and not run_2d:
        raise ValueError('At least one of run_1d or run_2d must be True')
    if compute_cf and not run_2d:
        raise ValueError('compute_cf=True requires run_2d=True')
    water = load_water_mask(mask, x0.shape[1:])
    if not water.any():
        raise ValueError('The mask contains no water cells (mask=1)')

    prefix = Path(files_path_prefix)
    base = Path('Coefficients') / f'aligned_{start_year}-{end_year}'
    print(f'Input: {x0.shape}; water cells: {int(water.sum())}/{water.size}', flush=True)
    print(f'Period: {start_year}-{end_year}; transitions: {x0.shape[0]-1}; '
          f'GMM components: {n_components}', flush=True)
    if run_1d:
        print(f'1D quantile groups: {n_bins}', flush=True)
    if run_2d:
        print(f'2D quantile groups: {n_bins_2d} per axis (up to {n_bins_2d**2} joint groups)', flush=True)

    specs = _daily_result_specs(
        prefix, base, names, n_bins, n_bins_2d, methods,
        run_1d, run_2d, x0.shape[1:],
    )
    if overwrite or not resume:
        first_transition = 1
    else:
        first_transition = _first_unfinished_transition(specs, len(x0), start_index)

    if first_transition is None:
        print('All requested daily NP/SP outputs already exist; nothing to recalculate.', flush=True)
    else:
        first_label = first_transition + start_index
        print(f'Resuming coefficient calculations from transition {first_transition} '
              f'(output label {first_label}).', flush=True)
        total_transitions = x0.shape[0] - 1
        for chunk_start in range(first_transition - 1, total_transitions, chunk_transitions):
            chunk_end = min(x0.shape[0], chunk_start + chunk_transitions + 1)
            first_t, end_t = chunk_start + 1, chunk_end
            first_label = first_t + start_index
            last_label = chunk_end - 1 + start_index
            print(f'Checking transitions {first_label}..{last_label} '
                  f'({last_label-start_index}/{total_transitions})', flush=True)

            if run_1d:
                for variable, series in ((names[0], x0), (names[1], x1)):
                    relative_output = base / f'1d_bins_{n_bins}' / variable
                    if "np" in methods:
                        spec = (prefix / relative_output / 'Nonparametric' / 'daily',
                                ('A', 'B'), tuple(x0.shape[1:]))
                        if overwrite or _missing_daily_outputs(spec, first_t, end_t, start_index, remove_invalid=True):
                            count_1d_nonparam(
                                str(prefix), series, chunk_start, chunk_end,
                                path=str(relative_output), start_index=start_index,
                                quantiles_amount=n_bins, mask=water,
                                min_group_size=min_group_size, overwrite=overwrite, dt=dt,
                            )
                    if "sp" in methods:
                        spec = (prefix / relative_output / 'Semiparametric' / 'daily',
                                ('A', 'B', 'B_with_reg'), tuple(x0.shape[1:]))
                        if overwrite or _missing_daily_outputs(spec, first_t, end_t, start_index, remove_invalid=True):
                            count_semi_1d_AB(
                                str(prefix), series, chunk_start, chunk_end,
                                path=str(relative_output), start_index=start_index,
                                quantiles_amount=n_bins, n_components=n_components,
                                mask=water, min_group_size=min_group_size,
                                n_init=n_init, remove_reg_covar=True,
                                overwrite=overwrite, dt=dt,
                            )

            if run_2d:
                pair_dir = base / f'2d_bins_{n_bins_2d}' / f'{names[0]}_{names[1]}'
                if "np" in methods:
                    spec = (prefix / pair_dir / 'Nonparametric_2d' / 'daily',
                            ('A', 'B'), ((2, *x0.shape[1:]), (4, *x0.shape[1:])))
                    if overwrite or _missing_daily_outputs(spec, first_t, end_t, start_index, remove_invalid=True):
                        count_2d_nonparam(
                            str(prefix), x0, x1, chunk_start, chunk_end,
                            path=str(pair_dir), start_index=start_index,
                            quantiles_amount=n_bins_2d, mask=water,
                            min_group_size=min_group_size, overwrite=overwrite, dt=dt,
                        )
                if "sp" in methods:
                    spec = (prefix / pair_dir / 'Semiparametric_2d' / 'daily',
                            ('A', 'B', 'B_with_reg'),
                            ((2, *x0.shape[1:]), (4, *x0.shape[1:]), (4, *x0.shape[1:])))
                    if overwrite or _missing_daily_outputs(spec, first_t, end_t, start_index, remove_invalid=True):
                        count_semi_2d_AB(
                            str(prefix), x0, x1, chunk_start, chunk_end,
                            path=str(pair_dir), start_index=start_index,
                            quantiles_amount=n_bins_2d, n_components=n_components,
                            mask=water, min_group_size=min_group_size,
                            n_init=n_init, remove_reg_covar=True,
                            overwrite=overwrite, dt=dt,
                        )

    if compute_cf:
        pair_dir = base / f'2d_bins_{n_bins_2d}' / f'{names[0]}_{names[1]}'
        for method_folder, method_key in (("Nonparametric_2d", "np"),
                                           ("Semiparametric_2d", "sp")):
            if method_key in methods:
                count_cf_from_daily(
                    files_path_prefix, method_folder, 0, x0.shape[0],
                    path=str(pair_dir), start_index=start_index,
                    pair_name=f"{names[0]}_{names[1]}/{method_folder}",
                    correlation_window=correlation_window, mean_width=mean_width,
                    mask=water,
                )
    print(f'All coefficient calculations finished. Output: {prefix/base}', flush=True)


def _format_collection_time(seconds: float) -> str:
    """Format elapsed/estimated time as HH:MM:SS."""
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{remaining_seconds:02d}'


def _collection_progress(stage: str, completed: int, total: int, started_at: float) -> None:
    """Print an ETA based on the observed speed of the current operation."""
    elapsed = max(0.0, time.perf_counter() - started_at)
    eta = elapsed * (total - completed) / completed if completed else float('nan')
    eta_text = _format_collection_time(eta) if completed else 'estimating...'
    print(
        f'  {stage}: {completed}/{total} ({completed / total:.0%}); '
        f'elapsed {_format_collection_time(elapsed)}; ETA {eta_text}',
        flush=True,
    )


def collect_daily_coefficients(
    files_path_prefix: str,
    *,
    path: str,
    method: str,
    start_label: int,
    end_label: int,
    coefficients=None,
    overwrite: bool = False,
    show_progress: bool = True,
) -> dict[str, Path]:
    """Stack daily coefficient maps over an inclusive range of saved labels.

    Read:  <root>/<path>/<method>/daily/<A|B|B_with_reg>_<label>.npy
    Write: <root>/<path>/<method>/<A|B|B_with_reg>_<start>-<end>.npy

    The first output axis is time, in ascending saved-label order. For 2D,
    A has shape (days,2,H,W) and B has shape (days,4,H,W) in channel order
    [00,11,01,10]. Large outputs are written via memory mapping and then
    atomically moved into place, so they do not need to fit in RAM.

    With show_progress=True, report input validation and writing progress,
    including an approximate ETA based on observed throughput. The ETA for
    validation and writing is reported separately because their speeds differ.
    """
    allowed = ('Nonparametric', 'Semiparametric',
               'Nonparametric_2d', 'Semiparametric_2d')
    if method not in allowed:
        raise ValueError(f'method must be one of {allowed}')
    if not isinstance(start_label, int) or not isinstance(end_label, int) or end_label < start_label:
        raise ValueError('start_label and end_label must be integers, inclusive, start <= end')
    if coefficients is None:
        coefficients = ('A', 'B', 'B_with_reg') if method.startswith('Semiparametric') else ('A', 'B')
    coefficients = tuple(dict.fromkeys(coefficients))
    allowed_fields = ('A', 'B', 'B_with_reg') if method.startswith('Semiparametric') else ('A', 'B')
    if not coefficients or any(field not in allowed_fields for field in coefficients):
        raise ValueError(f'coefficients must be a nonempty subset of {allowed_fields}')

    parent = Path(files_path_prefix) / path / method
    daily = parent / 'daily'
    if not daily.is_dir():
        raise FileNotFoundError(f'Daily output directory does not exist: {daily}')

    # Check the complete input range before writing any output files.
    definitions = []
    n_days = end_label - start_label + 1
    total_checks = n_days * len(coefficients)
    check_count = 0
    validation_started = time.perf_counter()
    last_validation_update = validation_started
    validation_interval = max(1, (total_checks + 4) // 5)
    if show_progress:
        print(f'[{method}] Checking {n_days} daily maps for each of '
              f'{len(coefficients)} coefficient(s)...', flush=True)
    for field in coefficients:
        filenames = [daily / f'{field}_{label}.npy'
                     for label in range(start_label, end_label + 1)]
        absent = [p for p in filenames if not p.is_file()]
        if absent:
            raise FileNotFoundError(
                f'{len(absent)} missing daily files for {field}; first: {absent[0]}'
            )
        shape = dtype = None
        for file in filenames:
            try:
                arr = np.load(file, mmap_mode='r', allow_pickle=False)
            except (OSError, ValueError, EOFError, TypeError) as exc:
                raise ValueError(f'Cannot read daily array {file}: {exc}') from exc
            if shape is None:
                shape, dtype = arr.shape, arr.dtype
                expected_channels = 2 if (method.endswith('_2d') and field == 'A') else \
                                    4 if method.endswith('_2d') else None
                if expected_channels is not None and (len(shape) != 3 or shape[0] != expected_channels):
                    raise ValueError(f'Unexpected shape {shape} in {file}; expected first axis {expected_channels}')
                if expected_channels is None and len(shape) != 2:
                    raise ValueError(f'Unexpected 1D coefficient map shape {shape} in {file}')
                if not np.issubdtype(dtype, np.number):
                    raise ValueError(f'Expected numeric data in {file}')
            elif arr.shape != shape or arr.dtype != dtype:
                raise ValueError(
                    f'Incompatible shape or dtype in {file}: {arr.shape}, {arr.dtype}; '
                    f'expected {shape}, {dtype}'
                )
            del arr
            check_count += 1
            now = time.perf_counter()
            if show_progress and (
                check_count == total_checks
                or check_count % validation_interval == 0
                or now - last_validation_update >= 30
            ):
                _collection_progress('Checking inputs', check_count, total_checks, validation_started)
                last_validation_update = now
        definitions.append((field, filenames, shape, dtype))

    outputs = {}
    for field, filenames, shape, dtype in definitions:
        destination = parent / f'{field}_{start_label}-{end_label}.npy'
        if destination.exists() and not overwrite:
            outputs[field] = destination
            if show_progress:
                print(f'[{method}] {field}: merged file already exists; skipping: '
                      f'{destination}', flush=True)
            continue
        temp_name = None
        write_started = time.perf_counter()
        last_write_update = write_started
        write_interval = max(1, (n_days + 9) // 10)
        if show_progress:
            print(f'[{method}] Collecting {field}: {n_days} days -> {destination.name}', flush=True)
        try:
            with tempfile.NamedTemporaryFile(
                dir=parent, prefix=f'.{field}_{start_label}-{end_label}_',
                suffix='.npy', delete=False,
            ) as temporary:
                temp_name = temporary.name
            combined = np.lib.format.open_memmap(
                temp_name, mode='w+', dtype=dtype, shape=(n_days, *shape)
            )
            for day_index, filename in enumerate(filenames):
                current = np.load(filename, mmap_mode='r', allow_pickle=False)
                combined[day_index] = current
                del current
                now = time.perf_counter()
                processed = day_index + 1
                if show_progress and (
                    processed == n_days
                    or processed % write_interval == 0
                    or now - last_write_update >= 30
                ):
                    _collection_progress(f'Writing {field}', processed, n_days, write_started)
                    last_write_update = now
            combined.flush()
            del combined
            os.replace(temp_name, destination)
            temp_name = None
        finally:
            if temp_name is not None:
                Path(temp_name).unlink(missing_ok=True)
        outputs[field] = destination
        if show_progress:
            print(f'[{method}] Saved merged {field}: {destination} '
                  f'({n_days} days; {_format_collection_time(time.perf_counter() - write_started)})',
                  flush=True)
    return outputs


def collect_period_coefficients(
    files_path_prefix: str,
    *,
    names: Tuple[str, str],
    start_year: int,
    end_year: int,
    start_label: int,
    end_label: int,
    n_bins: int = 250,
    n_bins_2d: int = 15,
    run_1d: bool = False,
    run_2d: bool = True,
    methods: Tuple[str, ...] = ('np', 'sp'),
    overwrite: bool = False,
    show_progress: bool = True,
) -> dict[str, dict[str, Path]]:
    """Merge every enabled estimator/method for an inclusive label range.

    Report individual coefficient ETAs and overall method-level progress.
    """
    methods = _validate_methods(methods)
    if not run_1d and not run_2d:
        raise ValueError('At least one of run_1d or run_2d must be True')
    root = Path('Coefficients') / f'aligned_{start_year}-{end_year}'
    targets = []
    if run_1d:
        for variable in names:
            branch = root / f'1d_bins_{n_bins}' / variable
            if 'np' in methods:
                targets.append((f'1d/{variable}/np', branch, 'Nonparametric'))
            if 'sp' in methods:
                targets.append((f'1d/{variable}/sp', branch, 'Semiparametric'))
    if run_2d:
        branch = root / f'2d_bins_{n_bins_2d}' / f'{names[0]}_{names[1]}'
        if 'np' in methods:
            targets.append(('2d/np', branch, 'Nonparametric_2d'))
        if 'sp' in methods:
            targets.append(('2d/sp', branch, 'Semiparametric_2d'))

    result = {}
    started_at = time.perf_counter()
    if show_progress:
        print(f'Starting merged collection: {len(targets)} method(s), '
              f'{end_label - start_label + 1} days each.', flush=True)
    for index, (key, branch, method) in enumerate(targets, start=1):
        if show_progress:
            print(f'Method {index}/{len(targets)}: {key} ({method})', flush=True)
        result[key] = collect_daily_coefficients(
            files_path_prefix, path=str(branch), method=method,
            start_label=start_label, end_label=end_label,
            overwrite=overwrite, show_progress=show_progress,
        )
        if show_progress:
            elapsed = time.perf_counter() - started_at
            left = len(targets) - index
            estimated = elapsed * left / index
            print(f'Methods completed: {index}/{len(targets)}; '
                  f'elapsed {_format_collection_time(elapsed)}; '
                  f'approximate remaining {_format_collection_time(estimated)} '
                  f'(based on average method time).', flush=True)
    return result


def run_2d_coefficients(
    array1, array2, mask, files_path_prefix,
    *, quantiles_amount: int = 15,
    time_start: int = 0, time_end: int = None,
    path: str = "Coefficients/", start_index: int = 0,
    methods=("np", "sp"), dt: float = 1.0,
    n_components: int = 3, n_init: int = 10,
    min_group_size: int = 30, reg_covar: float = 1e-6,
    random_state: int = 42, overwrite: bool = False,
    remove_reg_covar: bool = True,
    compute_cf: bool = False, correlation_window: int = 14,
    mean_width: int = 7, pair_name: str = "", resume: bool = True,
):
    """Run coupled 2D estimators, resuming at the first incomplete day.

    Inputs accept in-memory arrays and supported .npy layouts. Daily output
    format and optional C/F/FS paths remain the same as before.
    """
    methods = _validate_methods(methods)
    x0, x1 = load_pair_series(array1, array2)
    water = load_water_mask(mask, x0.shape[1:])
    if not water.any():
        raise ValueError("The water mask contains no water cells")
    end = len(x0) if time_end is None else time_end
    if not isinstance(time_start, int) or not isinstance(end, int):
        raise TypeError("time_start and time_end must be integer observation indices")
    if not 0 <= time_start < end <= len(x0) or end - time_start < 2:
        raise ValueError("The time range must include at least one full transition")

    root = Path(files_path_prefix) / path
    np_spec = (
        root / 'Nonparametric_2d' / 'daily', ('A', 'B'),
        ((2, *x0.shape[1:]), (4, *x0.shape[1:])),
    )
    sp_spec = (
        root / 'Semiparametric_2d' / 'daily',
        ('A', 'B', 'B_with_reg'),
        ((2, *x0.shape[1:]), (4, *x0.shape[1:]), (4, *x0.shape[1:])),
    )
    specs = []
    if 'np' in methods:
        specs.append(np_spec)
    if 'sp' in methods:
        specs.append(sp_spec)
    first = time_start + 1
    if not overwrite and resume:
        first = next((t for t in range(time_start + 1, end)
                      if any(_missing_daily_outputs(spec, t, t + 1, start_index)
                             for spec in specs)), None)
    if first is not None:
        previous = first - 1
        if 'np' in methods and (overwrite or _missing_daily_outputs(
                np_spec, first, end, start_index, remove_invalid=True)):
            count_2d_nonparam(files_path_prefix, x0, x1, previous, end,
                              path=path, start_index=start_index,
                              quantiles_amount=quantiles_amount, mask=water,
                              min_group_size=min_group_size, overwrite=overwrite, dt=dt)
        if 'sp' in methods and (overwrite or _missing_daily_outputs(
                sp_spec, first, end, start_index, remove_invalid=True)):
            count_semi_2d_AB(files_path_prefix, x0, x1, previous, end,
                             path=path, start_index=start_index,
                             quantiles_amount=quantiles_amount, mask=water,
                             min_group_size=min_group_size, overwrite=overwrite, dt=dt,
                             n_components=n_components, n_init=n_init,
                             reg_covar=reg_covar, random_state=random_state,
                             remove_reg_covar=remove_reg_covar)
    if compute_cf:
        for method_folder, method_key in (("Nonparametric_2d", "np"),
                                           ("Semiparametric_2d", "sp")):
            if method_key in methods:
                cf_name = str(Path(pair_name) / method_folder) if pair_name else method_folder
                count_cf_from_daily(
                    files_path_prefix, method_folder, time_start, end,
                    path=path, start_index=start_index, pair_name=cf_name,
                    correlation_window=correlation_window, mean_width=mean_width,
                    mask=water,
                )
    return {
        "methods": methods,
        "n_transitions": end - time_start - 1,
        "quantiles_per_axis": quantiles_amount,
        "output_path": str(root),
    }


def _write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        raise ValueError(f"No results to write to {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _plot_occupancy(count_matrix: np.ndarray, output: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 6))
    plot = ax.imshow(np.log1p(count_matrix.T), origin="lower", aspect="equal")
    ax.set_xlabel("Quantile index: first variable")
    ax.set_ylabel("Quantile index: second variable")
    ax.set_title(title)
    bar = fig.colorbar(plot, ax=ax)
    bar.set_label("log(1 + observations in group)")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def diagnose_joint_groups(
    array1,
    array2,
    mask,
    out_dir,
    bins_list: Iterable[int] = (10, 12, 15, 18, 20),
    n_transitions: int = 7,
    time_start: int = 0,
    index_offset: int = 1,
    min_group_size: int = 2,
    thresholds: tuple[int, ...] = (2, 10, 20, 30),
    plot_bins: int = 15,
    make_plots: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Inspect the FIRST n transitions from time_start for each bin count.

    Both inputs can be ndarray in shape (T,H,W), (H*W,T), (T,H*W),
    or paths to .npy files. A mask value of 1 denotes water; 0 denotes land.
    """
    x0, x1 = coerce_series(array1), coerce_series(array2)
    if x0.shape != x1.shape:
        raise ValueError(f"Input series have different shapes: {x0.shape} and {x1.shape}")
    water = load_water_mask(mask, x0.shape[1:])
    n_water = int(water.sum())
    if not n_water:
        raise ValueError("Water mask contains no water cells (mask=1)")
    if n_transitions < 1:
        raise ValueError("n_transitions must be positive")
    if not 0 <= time_start < x0.shape[0] - 1:
        raise ValueError("time_start must leave at least one subsequent observation")
    if time_start + n_transitions >= x0.shape[0]:
        raise ValueError(
            f"Need at least {time_start + n_transitions + 1} time points "
            f"to inspect {n_transitions} transitions; got {x0.shape[0]}"
        )
    n_bins_all = tuple(dict.fromkeys(int(n) for n in bins_list))
    if not n_bins_all or any(n < 1 for n in n_bins_all):
        raise ValueError("bins_list must contain positive integers")
    if min_group_size < 2:
        raise ValueError("min_group_size must be at least 2 for aligned NP and SP")
    limits = tuple(sorted(set(int(n) for n in (*thresholds, min_group_size))))
    if any(n < 1 for n in limits):
        raise ValueError("Group size thresholds must be positive")

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    daily: list[dict] = []
    group_records: list[dict] = []
    running_occupancy = np.zeros((plot_bins, plot_bins), dtype=np.float64)
    plot_days = 0
    for n in n_bins_all:
        for t in range(time_start + 1, time_start + n_transitions + 1):
            grouped = prepare_pair_transition(
                x0[t - 1], x0[t], x1[t - 1], x1[t], n, water
            )
            sizes = np.zeros(n * n, dtype=np.int64)
            for group in grouped.groups:
                sizes[group.bin_id] = group.increments.shape[0]
            valid = int(grouped.n_valid_pairs)
            if sizes.sum() != valid:
                raise AssertionError("Group size accounting does not match valid pairs")
            occupied = sizes[sizes > 0]
            record = {
                "bins_per_axis": n,
                "possible_groups": n * n,
                "transition_t": t,
                "output_day_index": t + index_offset,
                "water_cells": n_water,
                "valid_transition_cells": valid,
                "valid_pct_of_water": round(100 * valid / n_water, 4),
                "occupied_groups": int(occupied.size),
                "empty_groups": int((sizes == 0).sum()),
                "singleton_groups": int((sizes == 1).sum()),
                "group_size_min_occupied": int(occupied.min()) if occupied.size else 0,
                "group_size_p10_occupied": round(float(np.percentile(occupied, 10)), 2) if occupied.size else 0.0,
                "group_size_median_occupied": float(np.median(occupied)) if occupied.size else 0.0,
                "group_size_p90_occupied": round(float(np.percentile(occupied, 90)), 2) if occupied.size else 0.0,
                "group_size_max_occupied": int(occupied.max()) if occupied.size else 0,
            }
            for threshold in limits:
                eligible = sizes >= threshold
                count_eligible = int(np.count_nonzero(eligible))
                kept = int(sizes[eligible].sum())
                record[f"groups_ge_{threshold}"] = count_eligible
                record[f"cells_in_groups_ge_{threshold}"] = kept
                record[f"coverage_valid_pct_ge_{threshold}"] = round(
                    100 * kept / valid, 4) if valid else 0.0
                record[f"coverage_water_pct_ge_{threshold}"] = round(
                    100 * kept / n_water, 4)
            daily.append(record)
            for bin_id, count in enumerate(sizes):
                group_records.append({
                    "bins_per_axis": n,
                    "transition_t": t,
                    "output_day_index": t + index_offset,
                    "bin_first": bin_id // n,
                    "bin_second": bin_id % n,
                    "group_size": int(count),
                })
            print(
                f"bins={n:2d} (up to {n*n:3d} groups), day={t + index_offset:5d}: "
                f"occupied={occupied.size:3d}, singletons={(sizes==1).sum():3d}, "
                f"coverage(n>={min_group_size})="
                f"{record[f'coverage_valid_pct_ge_{min_group_size}']:.2f}%, "
                f"coverage(n>=30)={record.get('coverage_valid_pct_ge_30', 0.0):.2f}%",
                flush=True,
            )
            if n == plot_bins:
                counts = sizes.reshape(n, n)
                running_occupancy += counts
                plot_days += 1
                if make_plots and t == time_start + 1:
                    _plot_occupancy(
                        counts, target / f"occupancy_bins_{plot_bins}_first_day.png",
                        f"Joint groups: {plot_bins} x {plot_bins}, day {t + index_offset}",
                    )
    if make_plots and plot_days:
        _plot_occupancy(
            running_occupancy / plot_days,
            target / f"occupancy_bins_{plot_bins}_mean.png",
            f"Mean occupancy over {plot_days} transitions: {plot_bins} x {plot_bins}",
        )

    by_bin: list[dict] = []
    for n in n_bins_all:
        subset = [item for item in daily if item["bins_per_axis"] == n]
        row = {
            "bins_per_axis": n,
            "possible_groups": n * n,
            "days_tested": len(subset),
            "mean_occupied_groups": round(float(np.mean([r["occupied_groups"] for r in subset])), 2),
            "mean_empty_groups": round(float(np.mean([r["empty_groups"] for r in subset])), 2),
            "mean_singleton_groups": round(float(np.mean([r["singleton_groups"] for r in subset])), 2),
            "min_group_size_seen": min(r["group_size_min_occupied"] for r in subset),
            "mean_group_size_median": round(float(np.mean([r["group_size_median_occupied"] for r in subset])), 2),
        }
        for threshold in limits:
            key = f"coverage_valid_pct_ge_{threshold}"
            row[f"mean_{key}"] = round(float(np.mean([r[key] for r in subset])), 4)
            row[f"min_{key}"] = round(float(np.min([r[key] for r in subset])), 4)
            row[f"mean_groups_ge_{threshold}"] = round(
                float(np.mean([r[f'groups_ge_{threshold}'] for r in subset])), 2
            )
        by_bin.append(row)

    _write_csv(target / "daily_summary.csv", daily)
    _write_csv(target / "by_bin_summary.csv", by_bin)
    _write_csv(target / "group_counts.csv", group_records)
    settings = {
        "bins_list": list(n_bins_all), "n_transitions": n_transitions,
        "time_start": time_start, "index_offset": index_offset,
        "thresholds": list(limits), "plot_bins": plot_bins,
        "min_group_size": min_group_size, "mask_water_value": 1,
        "water_cells": n_water, "source_shape": list(x0.shape),
        "estimator_grouping_function": "Coefficients.common.prepare_pair_transition",
    }
    (target / "config.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nSummary over inspected days:")
    for row in by_bin:
        print(
            f"  {row['bins_per_axis']:2d}x{row['bins_per_axis']:<2d}: "
            f"occupied={row['mean_occupied_groups']:.1f}, "
            f"coverage(n>=2)={row.get('mean_coverage_valid_pct_ge_2', 0.0):.2f}%, "
            f"coverage(n>=30)={row.get('mean_coverage_valid_pct_ge_30', 0.0):.2f}%"
        )
    print(f"\nSaved diagnostics to: {target}", flush=True)
    return daily, by_bin
