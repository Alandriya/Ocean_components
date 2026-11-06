"""Process-parallel estimation using the repository's unchanged NP/SP estimators.

This module can be copied into an existing Ocean_components checkout. It is
intentionally separate from the serial numerical core and delegates all
statistics to ``Coefficients.runner.run_decade_coefficients``.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path
import multiprocessing

import numpy as np

from Coefficients.collection import (
    collect_period_coefficients,
    period_dir,
    normalize_methods,
    METHOD_FOLDERS,
)
from Coefficients.common import coerce_series


def split_transition_ranges(n_observations: int, workers: int) -> tuple[tuple[int, int], ...]:
    """Partition all endpoint transition indices without duplicates.

    Parameters
    ----------
    n_observations : int
        Number of observations in both input series.
    workers : int
        Requested number of independent processes.

    Returns
    -------
    tuple of tuple of int
        Half-open ranges ``(first, stop)`` of transition indices. Each
        transition ``t`` uses input observations ``t - 1`` and ``t``.
    """
    if type(n_observations) is not int or n_observations < 2:
        raise ValueError("n_observations must be >= 2")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    count = n_observations - 1
    used = min(workers, count)
    per_worker, extra = divmod(count, used)
    ranges = []
    first = 1
    for worker in range(used):
        stop = first + per_worker + int(worker < extra)
        ranges.append((first, stop))
        first = stop
    assert first == n_observations
    return tuple(ranges)


def _worker(job: dict) -> tuple[int, int]:
    """Estimate one disjoint block after mapping both original input files.

    Parameters
    ----------
    job : dict
        Input paths, relative transition bounds, and estimator settings.

    Returns
    -------
    tuple of int
        Completed transition range.
    """
    from Coefficients.runner import run_decade_coefficients

    a = coerce_series(np.load(job["files"][0], mmap_mode="r", allow_pickle=False))
    b = coerce_series(np.load(job["files"][1], mmap_mode="r", allow_pickle=False))
    first, stop = job["transitions"]
    if a.shape != b.shape or len(a) != job["observations"]:
        raise ValueError("The two input arrays do not have matching dimensions")

    settings = dict(job["settings"])
    settings["array1"] = a[first - 1:stop]
    settings["array2"] = b[first - 1:stop]
    settings["start_index"] += first - 1
    settings["compute_cf"] = False
    # Control BLAS threading inside each process to avoid CPU oversubscription.
    try:
        from threadpoolctl import threadpool_limits
        limiter = threadpool_limits(limits=1)
    except ImportError:
        limiter = nullcontext()
    with limiter:
        run_decade_coefficients(**settings)
    return first, stop


def _has_complete_collections(root: Path, names, start_year, end_year,
                              first_label, last_label, n_bins, n_bins_2d,
                              run_1d, run_2d, methods) -> bool:
    """Verify only the selected NP/SP outputs before skipping calculation."""
    from Coefficients.collection import _aggregate_valid

    base = root / "Coefficients" / f"{start_year}-{end_year}"
    count = last_label - first_label + 1
    paths = []
    if run_1d:
        for name in names:
            for selected in methods:
                folder = METHOD_FOLDERS[selected][0]
                paths.append((base / f"1d_bins_{n_bins}" / name / folder, folder))
    if run_2d:
        for selected in methods:
            folder = METHOD_FOLDERS[selected][1]
            paths.append((base / f"2d_bins_{n_bins_2d}" / f"{names[0]}_{names[1]}" / folder, folder))
    if not paths:
        return False
    return all(
        _aggregate_valid(folder / f"{field}_{first_label}-{last_label}.npy",
                         method, field, count)
        for folder, method in paths
        for field in (("A", "B", "B_with_reg") if method.startswith("Semi") else ("A", "B"))
    )


def run_parallel_period(
    files_path_prefix,
    source_files,
    mask,
    names,
    start_year,
    end_year,
    start_index,
    *,
    parallel_workers=4,
    methods=("np", "sp"),
    n_bins=250,
    n_bins_2d=12,
    n_components=3,
    n_init=10,
    chunk_transitions=7,
    dt=1.0,
    run_1d=False,
    run_2d=True,
    overwrite=False,
    delete_daily_after_collect=True,
    min_group_size=2,
):
    """Estimate and merge only the requested NP/SP methods in parallel.

    Parameters
    ----------
    files_path_prefix : str or Path
        Root for source arrays and coefficient outputs.
    source_files : sequence of two paths
        Existing NPY input files corresponding to ``names``.
    mask : ndarray
        Ocean mask broadcast over the spatial grid.
    names : tuple of str
        Names of two simultaneously estimated variables.
    start_year, end_year : int
        Period folder identifiers.
    start_index : int
        One-based label for the first input observation.
    parallel_workers : int
        Worker process count, or 1 for serial execution.
    methods : str or sequence of str
        Requested estimators: "np", "sp", "both", or ("np", "sp").
    n_bins, n_bins_2d, n_components, n_init, chunk_transitions : int
        Original numerical configuration.
    dt : float
        Model timestep.
    run_1d, run_2d : bool
        Enable scalar and coupled estimators.
    overwrite : bool
        Recompute existing daily and combined outputs.
    delete_daily_after_collect : bool
        Remove intermediate files only after all selected estimators are aggregated.
    min_group_size : int
        Minimum observations per group.

    Returns
    -------
    dict
        Aggregated output paths for requested methods only.

    Notes
    -----
    The old runner writes into ``aligned_YYYY-YYYY``. This workflow supports
    both that historic layout and the newer compact layout. A compact folder
    is temporarily moved back to the historic location only if the installed
    runner still uses the historic layout. On worker failure, original daily
    outputs are retained so a subsequent run can resume.
    """
    from Coefficients import runner

    methods = normalize_methods(methods)
    root = Path(files_path_prefix).resolve()
    files = tuple(str(Path(path).resolve()) for path in source_files)
    if len(files) != 2 or not all(Path(f).is_file() for f in files):
        raise FileNotFoundError("Two existing input .npy files are required")
    a, b = (coerce_series(np.load(path, mmap_mode="r", allow_pickle=False)) for path in files)
    if a.shape != b.shape or len(a) < 2:
        raise ValueError("Both time series must have the same shape and >=2 observations")
    if not run_1d and not run_2d:
        raise ValueError("At least one of run_1d / run_2d must be enabled")
    if type(parallel_workers) is not int or parallel_workers < 1:
        raise ValueError("parallel_workers must be a positive integer")
    first_label = start_index + 1
    last_label = start_index + len(a) - 1
    modern = root / "Coefficients" / f"{start_year}-{end_year}"
    old = root / "Coefficients" / f"aligned_{start_year}-{end_year}"
    if old.exists() and modern.exists():
        raise FileExistsError(f"Both period folders exist: {old} and {modern}")

    if not overwrite and _has_complete_collections(
        root, names, start_year, end_year, first_label, last_label,
        n_bins, n_bins_2d, run_1d, run_2d, methods,
    ):
        print(f"All {methods} aggregate outputs exist; skipping daily estimation.", flush=True)
    else:
        # Existing runner modules can use either compact or historic folders.
        uses_compact = "period_dir" in runner.run_decade_coefficients.__code__.co_names
        if not uses_compact and modern.exists():
            modern.rename(old)
        if uses_compact and old.exists():
            period_dir(root, start_year, end_year)

        settings = dict(
            files_path_prefix=str(root), mask=np.asarray(mask), names=tuple(names),
            start_year=start_year, end_year=end_year, start_index=start_index,
            n_bins=n_bins, n_bins_2d=n_bins_2d,
            n_components=n_components, n_init=n_init,
            chunk_transitions=chunk_transitions, dt=dt,
            min_group_size=min_group_size, run_1d=run_1d, run_2d=run_2d,
            methods=methods, overwrite=overwrite, resume=True,
        )
        ranges = split_transition_ranges(len(a), parallel_workers)
        jobs = [dict(files=files, observations=len(a), transitions=part, settings=settings)
                for part in ranges]
        for idx, (first, stop) in enumerate(ranges, start=1):
            print(f"Process {idx}/{len(ranges)}: coefficient labels "
                  f"{start_index + first}..{start_index + stop - 1}", flush=True)

        if len(jobs) == 1:
            _worker(jobs[0])
        else:
            with ProcessPoolExecutor(
                max_workers=len(jobs), mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                pending = {pool.submit(_worker, job): job["transitions"] for job in jobs}
                for future in as_completed(pending):
                    expected = pending[future]
                    if future.result() != expected:
                        raise RuntimeError(f"Unexpected process output for {expected}")
        # This move is performed after all processes finish successfully.
        period_dir(root, start_year, end_year)

    merged = collect_period_coefficients(
        str(root), names=tuple(names), start_year=start_year, end_year=end_year,
        start_label=first_label, end_label=last_label,
        n_bins=n_bins, n_bins_2d=n_bins_2d, run_1d=run_1d, run_2d=run_2d,
        methods=methods, overwrite=overwrite,
        delete_daily=delete_daily_after_collect,
    )
    expected = set()
    if run_1d:
        expected.update(f"1d/{name}/{method}" for name in names for method in methods)
    if run_2d:
        expected.update(f"2d/{method}" for method in methods)
    if set(merged) != expected:
        raise RuntimeError(f"Not all requested {methods} outputs were collected: {set(merged)}")
    return merged
