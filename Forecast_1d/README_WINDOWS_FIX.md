# Forecast_1d: Windows memory-map publication fix

This update fixes `PermissionError: [WinError 5]` during the final staging-directory rename and `WinError 32` during cleanup. The original extraction loop left a reference to the last writable `np.memmap`, `std_15x15.npy`, open. Windows does not allow the directory operation while that file is mapped.

## Installation

Copy this archive into the Git repository directory and extract it with directory structure preserved. Replace only `Forecast_1d/extract_points.py` and add `Forecast_1d/recover_staging.py`. Do not change `main.py`, `project_paths.py`, or your already-selected point definitions.

## Check the interrupted extraction first

The previous run completed all time steps, but the original exception handler may have deleted some of the staged files. Inspect the existing staging directory before rerunning:

```shell
python -m Forecast_1d.recover_staging
```

If the output says `COMPLETE`, publish it without recalculation:

```shell
python -m Forecast_1d.recover_staging --publish
```

If it says `INCOMPLETE` or `No staging directories found`, rerun extraction:

```shell
python main.py
```

The new extractor preserves a **complete** staging directory if a transient Windows filesystem lock still prevents publication. The recovery tool checks files and samples before publishing and never overwrites an existing final directory. A partially deleted original staging folder is not repaired or deleted automatically.

A successful run ends with:

```text
[DONE] Saved <DATA_ROOT>/DATA/Forecast_1d/atlantic10_native_1979-2026
```

## Validation

```shell
python -m unittest discover -s Forecast_1d/tests -p "test_windows_memmap.py" -v
```

The tests include an assertion that every writable mapping is closed before rename, a simulated publication failure and recovery, an incomplete-staging rejection, and an early processing error. The original extraction unit tests also pass with this version.

The code still requires NumPy and the project's regular `project_paths.py`; no GPU, Slurm, Docker, or `statsmodels` is involved.
