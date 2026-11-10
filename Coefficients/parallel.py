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
import os
import time

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




def _unique_scalar_inputs(names, files):
    """Map one or two caller-supplied names to independent scalar sources.

    The existing main.py always supplies two names. Repeating one name is a
    supported way of selecting a single scalar variable without modifying main.
    """
    if len(names) != len(files) or len(names) not in (1, 2):
        raise ValueError("1D mode needs one or two variable names and matching source files")
    unique = {}
    for name, filename in zip(names, files):
        if not isinstance(name, str) or not name or any(ch in name for ch in "/\\"):
            raise ValueError(f"Invalid scalar variable name: {name!r}")
        path = Path(filename).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing scalar input {name!r}: {path}")
        if name in unique and unique[name] != path:
            raise ValueError(f"Variable {name!r} supplied with different source files")
        unique[name] = path
    return unique


def _scalar_worker_impl(job):
    """Write one independent 1D variable over a disjoint transition interval.

    Each worker opens the input only once and processes bounded transition
    chunks. Already valid daily maps are left untouched when overwrite=False.
    """
    from Coefficients.nonparametric import count_1d_nonparam
    from Coefficients.semiparametric import count_semi_1d_AB
    from Coefficients.runner import _daily_file_complete

    filename = Path(job["file"])
    values = coerce_series(np.load(filename, mmap_mode="r", allow_pickle=False))
    first, stop = job["transitions"]  # coefficient transitions in [first, stop)
    if not 1 <= first < stop <= len(values):
        raise ValueError(f"Invalid scalar transition range {first}:{stop} for {filename}")
    root = Path(job["root"])
    relative = Path("Coefficients") / job["period"] / (
        f"1d_bins_{job['n_bins']}") / job["name"]
    mask = job["mask"]
    total = stop - first
    chunks = (total + job["chunk_transitions"] - 1) // job["chunk_transitions"]
    interval = max(1, int(job.get("log_every_chunks", 3)))
    started = time.perf_counter()
    timings = {m: 0.0 for m in job["methods"]}
    pid = os.getpid()
    tag = job.get("job_tag", job["name"])
    prefix = f"[1D][pid={pid}][{tag}]"
    print(f"{prefix} INPUT {filename.name}; shape={tuple(values.shape)}; "
          f"bins={job['n_bins']}; methods={','.join(job['methods'])}; "
          f"labels={job['start_index'] + first}..{job['start_index'] + stop - 1}; "
          f"chunks={chunks}", flush=True)
    for chunk_index, first_day in enumerate(range(first, stop, job["chunk_transitions"]), 1):
        stop_day = min(stop, first_day + job["chunk_transitions"])
        block = values[first_day - 1:stop_day]  # previous + endpoints
        base_label = job["start_index"] + first_day - 1
        report = chunk_index == 1 or chunk_index == chunks or chunk_index % interval == 0
        if report:
            print(f"{prefix} RUN chunk {chunk_index}/{chunks}, "
                  f"labels {base_label + 1}..{base_label + len(block) - 1}, "
                  f"methods={','.join(job['methods'])}", flush=True)
        for method in job["methods"]:
            method_started = time.perf_counter()
            method_folder = METHOD_FOLDERS[method][0]
            fields = ("A", "B", "B_with_reg") if method == "sp" else ("A", "B")
            daily = root / relative / method_folder / "daily"
            if not job["overwrite"]:
                all_ready = True
                # Validate header and shape, not merely file existence.
                for label in range(base_label + 1, base_label + len(block)):
                    for field in fields:
                        candidate = daily / f"{field}_{label}.npy"
                        if not _daily_file_complete(candidate, values.shape[1:]):
                            all_ready = False
                            if candidate.is_file():
                                candidate.unlink()  # invalid output, will be regenerated
                if all_ready:
                    timings[method] += time.perf_counter() - method_started
                    continue
            common = dict(
                files_path_prefix=str(root), data_array=block,
                time_start=0, time_end=len(block), path=str(relative),
                start_index=base_label, quantiles_amount=job["n_bins"],
                mask=mask, min_group_size=job["min_group_size"],
                overwrite=job["overwrite"], dt=job["dt"],
            )
            try:
                if method == "sp":
                    count_semi_1d_AB(
                        **common, n_components=job["n_components"],
                        n_init=job["n_init"], remove_reg_covar=True,
                    )
                else:
                    # The NP function calls its input 'flux', not 'data_array'.
                    common["flux"] = common.pop("data_array")
                    count_1d_nonparam(**common)
            except Exception as exc:
                raise RuntimeError(
                    f"1D {job['name']}/{method}: failed for transition labels "
                    f"{base_label + 1}..{base_label + len(block) - 1} "
                    f"(n_bins={job['n_bins']}): {exc}"
                ) from exc
            timings[method] += time.perf_counter() - method_started
        if report:
            done = stop_day - first
            elapsed = max(time.perf_counter() - started, 1.e-6)
            remaining = elapsed * (total - done) / max(done, 1)
            method_times = ', '.join(f"{m}={timings[m]:.1f}s" for m in job["methods"])
            print(f"{prefix} PROGRESS {done}/{total} transitions "
                  f"({100 * done / total:.1f}%); "
                  f"elapsed={elapsed:.1f}s; ETA~{remaining:.1f}s; "
                  f"method elapsed: {method_times}", flush=True)
    return job["name"], first, stop


def _scalar_worker(job):
    """Execute scalar work with limited BLAS threads and per-process logging.

    Large numbers of small 1D Gaussian-mixture fits can become slower if
    many independent processes each spawn multi-threaded BLAS kernels.
    Limiting each worker to one BLAS thread mirrors the working 2D route.
    """
    tag = job.get("job_tag", job["name"])
    pid = os.getpid()
    started = time.perf_counter()
    print(f"[1D][pid={pid}][{tag}] START worker; "
          f"variable={job['name']}; methods={job['methods']}", flush=True)
    try:
        from threadpoolctl import threadpool_limits
        limiter = threadpool_limits(limits=1)
    except ImportError:
        limiter = nullcontext()
        print(f"[1D][pid={pid}][{tag}] WARNING: threadpoolctl not installed; "
              "BLAS thread count is not limited", flush=True)
    try:
        with limiter:
            result = _scalar_worker_impl(job)
    except Exception:
        print(f"[1D][pid={pid}][{tag}] FAILED "
              f"after {time.perf_counter()-started:.1f}s", flush=True)
        raise
    print(f"[1D][pid={pid}][{tag}] DONE "
          f"elapsed={time.perf_counter()-started:.1f}s", flush=True)
    return result


def _run_scalar_period(
    files_path_prefix, source_files, mask, names, start_year, end_year,
    start_index, *, parallel_workers, methods, n_bins, n_components, n_init,
    chunk_transitions, dt, overwrite, delete_daily_after_collect, min_group_size,
):
    """Independent scalar NP/SP workflow, preserving the project's output tree.

    Unlike bivariate mode, unrelated variables are not forced to have equal
    numbers of observations or to have different names. Scalar files have
    shape (T,H,W); methods and bin counts are kept exactly as requested.
    """
    from Coefficients.collection import (
        _aggregate_valid, _cleanup_daily, collect_daily_coefficients,
    )

    root = Path(files_path_prefix).resolve()
    inputs = _unique_scalar_inputs(tuple(names), tuple(source_files))
    period = f"{start_year}-{end_year}"
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be positive for the scalar estimator")
    if not isinstance(chunk_transitions, int) or chunk_transitions < 1:
        raise ValueError("chunk_transitions must be positive")
    if type(parallel_workers) is not int or parallel_workers < 1:
        raise ValueError("parallel_workers must be a positive integer")
    if min_group_size < 2 or dt <= 0:
        raise ValueError("min_group_size must be >=2 and dt must be positive")
    # Retain existing coefficients, in particular the working 2D results.
    period_dir(root, start_year, end_year)
    results = {}
    jobs = []
    pending = []
    overall_started = time.perf_counter()
    print(f"[1D][main] period={period}; variables={tuple(inputs)}; "
          f"methods={methods}; quantile bins={n_bins}; "
          f"GMM components={n_components}, n_init={n_init}; "
          f"requested workers={parallel_workers}; "
          f"chunk transitions={chunk_transitions}", flush=True)
    for name, filename in inputs.items():
        series = coerce_series(np.load(filename, mmap_mode="r", allow_pickle=False))
        n_obs = int(len(series))
        first_label = start_index + 1
        last_label = start_index + n_obs - 1
        spatial_shape = series.shape[1:]
        if mask is not None and np.asarray(mask).shape != spatial_shape:
            raise ValueError(
                f"Ocean mask {np.asarray(mask).shape} does not match {name} {spatial_shape}"
            )
        needed = []
        for method in methods:
            folder = METHOD_FOLDERS[method][0]
            parent = root / "Coefficients" / period / f"1d_bins_{n_bins}" / name / folder
            fields = ("A", "B", "B_with_reg") if method == "sp" else ("A", "B")
            complete = not overwrite and all(
                _aggregate_valid(parent / f"{field}_{first_label}-{last_label}.npy",
                                 folder, field, n_obs - 1, spatial_shape)
                for field in fields
            )
            if complete:
                print(f"1D {name}/{method}: collected files exist; reusing.", flush=True)
            else:
                needed.append(method)
            pending.append((name, method, first_label, last_label, parent, fields))
        if needed:
            ranges = split_transition_ranges(n_obs, parallel_workers)
            for job_index, (first, stop) in enumerate(ranges, 1):
                tag = f"{name} {job_index}/{len(ranges)}"
                print(f"[1D][main] QUEUED {tag}; "
                      f"labels={start_index + first}..{start_index + stop - 1}; "
                      f"methods={','.join(needed)}", flush=True)
                jobs.append(dict(
                    job_tag=tag, log_every_chunks=3,
                    name=name, file=str(filename), root=str(root), period=period,
                    transitions=(first, stop), methods=tuple(needed),
                    start_index=start_index, n_bins=n_bins, mask=np.asarray(mask)
                    if mask is not None else None,
                    n_components=n_components, n_init=n_init,
                    min_group_size=min_group_size, chunk_transitions=chunk_transitions,
                    dt=dt, overwrite=overwrite,
                ))
        del series

    if jobs:
        print(f"[1D][main] START {len(jobs)} jobs; "
              f"up to {min(parallel_workers, len(jobs))} worker processes", flush=True)
        completed = 0
        if parallel_workers == 1 or len(jobs) == 1:
            for job in jobs:
                _scalar_worker(job)
                completed += 1
                print(f"[1D][main] COMPLETED {completed}/{len(jobs)} jobs "
                      f"({100*completed/len(jobs):.1f}%); "
                      f"elapsed={time.perf_counter()-overall_started:.1f}s", flush=True)
        else:
            with ProcessPoolExecutor(
                max_workers=parallel_workers,
                mp_context=multiprocessing.get_context("spawn"),
            ) as pool:
                futures = {pool.submit(_scalar_worker, job): job for job in jobs}
                for future in as_completed(futures):
                    result = future.result()  # propagate estimator exceptions
                    completed += 1
                    job = futures[future]
                    print(f"[1D][main] COMPLETED {completed}/{len(jobs)} jobs "
                          f"({100*completed/len(jobs):.1f}%); "
                          f"last={job['job_tag']}; output={result[0]}; "
                          f"elapsed={time.perf_counter()-overall_started:.1f}s", flush=True)

    # Merge/verify all requested methods *before* optional cleanup.
    print(f"[1D][main] MERGE starting for {len(pending)} variable/method "
          "collections (progress from the collector follows)", flush=True)
    for name, method, first_label, last_label, parent, fields in pending:
        key = f"1d/{name}/{method}"
        folder = METHOD_FOLDERS[method][0]
        rel = Path("Coefficients") / period / f"1d_bins_{n_bins}" / name
        files = collect_daily_coefficients(
            str(root), path=str(rel), method=folder,
            start_label=first_label, end_label=last_label,
            overwrite=overwrite, delete_daily=False,
        )
        n_days = last_label - first_label + 1
        for field, destination in files.items():
            if not _aggregate_valid(destination, folder, field, n_days,
                                    daily_shape=tuple(np.asarray(mask).shape)
                                    if mask is not None else None):
                raise RuntimeError(f"Invalid collected scalar result: {destination}")
        results[key] = files
        print(f"[1D][main] MERGED {key}: "
              f"{n_days} transitions, {len(files)} arrays", flush=True)

    if delete_daily_after_collect:
        for name, method, first_label, last_label, parent, fields in pending:
            _cleanup_daily(parent / "daily", fields, first_label, last_label)
    print(f"[1D][main] COMPLETE: "
          f"{len(results)} collections; total elapsed "
          f"{time.perf_counter()-overall_started:.1f}s", flush=True)
    return results


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
    if run_1d and run_2d and (len(names) != 2 or names[0] == names[1]):
        raise ValueError("Simultaneous 1D+2D mode requires two distinct variables")
    if run_1d:
        scalar = _run_scalar_period(
            files_path_prefix, source_files, mask, names, start_year, end_year,
            start_index, parallel_workers=parallel_workers, methods=methods,
            n_bins=n_bins, n_components=n_components, n_init=n_init,
            chunk_transitions=chunk_transitions, dt=dt,
            overwrite=overwrite,
            delete_daily_after_collect=delete_daily_after_collect,
            min_group_size=min_group_size,
        )
        if not run_2d:
            return scalar
        bivariate = run_parallel_period(
            files_path_prefix, source_files, mask, names,
            start_year, end_year, start_index,
            parallel_workers=parallel_workers, methods=methods,
            n_bins=n_bins, n_bins_2d=n_bins_2d,
            n_components=n_components, n_init=n_init,
            chunk_transitions=chunk_transitions, dt=dt,
            run_1d=False, run_2d=True, overwrite=overwrite,
            delete_daily_after_collect=delete_daily_after_collect,
            min_group_size=min_group_size,
        )
        return {**scalar, **bivariate}
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
