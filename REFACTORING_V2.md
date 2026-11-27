# Refactoring 2.0 — review and migration

Source: the supplied archive, based on Git commit `a31081cc98cc5e720eaae1191e62c323ee891106` in [Ocean_components](https://github.com/Alandriya/Ocean_components). No remote branch, commit or repository setting was modified.

## Scope

- Original folder structure retained. 30 modules consolidated into 11; two obsolete GRIB snapshots removed. Python-file count: 224 → 203 (21 fewer, including tests).
- Imports and references updated, with no remaining static Python imports pointing to the removed modules. The existing scientific algorithms, task names, configured paths and array formats were not intentionally changed.
- English module docstrings and English function/class/method docstrings provided throughout the Python files, including tests and nested helpers. Existing meaningful documentation retained. New documentation is predominantly concise one-line descriptions; scientific parameters already documented in detail remain intact. Russian plot labels, localization strings and generated Russian LaTeX content are intentionally preserved as runtime content, not comments.
- Removed unused `plotter`, `pymoo`, `pyswarms` requirements (no direct imports in the supplied source); added `eccodes`, `rasterio`, `cdsapi`, `Pillow`, `threadpoolctl`, `imageio-ffmpeg`, and a NumPy 2 floor. PyTorch stays a separate CPU/CUDA-specific installation. This is a dependency manifest, not a fully locked environment; an installation from scratch was not tested.
- Both root READMEs updated; relevant subproject README references migrated. No change to the configured numerical or plotting task defaults.

## Migration map

| Removed module(s) | Canonical module |
|---|---|
| `Eigenvalues/workflows.py` | `Eigenvalues/eigenvalues.py` |
| `Forecast_1d/models/seasonal.py` | `Forecast_1d/models/baselines.py` |
| `Forecast_1d/models/nbeats.py`, `nhits.py`, `patchtst.py`, `dlinear.py`, `tide.py` | `Forecast_1d/models/advanced.py` |
| `Forecast_1d/audit_future_weather.py`, `audit_oracle_drivers.py`, `audit_predictability.py`, `audit_sst_vs_atmosphere.py`, `audit_training.py` | `Forecast_1d/audit.py` |
| `Forecast_nn/logging_utils.py` | `Forecast_nn/runtime.py` |
| `Forecast_probs/graph_runner.py` | `Forecast_probs/graph.py` |
| `Functional/copula_calculations.py`, `copula_quintiles.py`, `copula_workflows.py` | `Functional/copula.py` |
| `Functional/stationary_diagnostics.py` | `Functional/stationary.py` |
| `Plotting/component_diagnostics.py` | `Plotting/component_plots.py` |
| `Plotting/stationary_spatial_plots.py` | `Plotting/stationary_plots.py` |
| `Plotting/hybrid_fp2d_marginal_overlay.py` | `Plotting/hybrid_stationary_plots.py` |
| `Data_processing/grib_before_duplicates_fix.py`, `grib_before_level_fix.py` | Deleted; current `Data_processing/grib.py` retained |

The old module paths are not retained as shim files: this is intentional to actually reduce the file count. Update external scripts/notebooks if they import old modules. Existing state-dict checkpoints are expected to be unaffected by module moves; serialized whole Python model objects that reference old class paths may require migration. No real checkpoints were available for verification.

### Command changes

| Previous command | New command |
|---|---|
| `python -m Forecast_1d.audit_future_weather` | `python -m Forecast_1d.audit future-weather` |
| `python -m Forecast_1d.audit_oracle_drivers` | `python -m Forecast_1d.audit oracle-drivers` |
| `python -m Forecast_1d.audit_predictability` | `python -m Forecast_1d.audit predictability` |
| `python -m Forecast_1d.audit_sst_vs_atmosphere` | `python -m Forecast_1d.audit sst-vs-atmosphere` |
| `python -m Forecast_1d.audit_training` | `python -m Forecast_1d.audit training` |
| `python -m Forecast_probs.graph_runner` | `python -m Forecast_probs.graph` |
| `python -m Plotting.component_diagnostics` | `python -m Plotting.component_plots` |
| `python -m Plotting.hybrid_fp2d_marginal_overlay` | `python -m Plotting.hybrid_stationary_plots` |

Each audit keeps its own arguments and scenario outputs. Conflicting names `audit`, `evaluate`, `main`, `SCENARIOS` and `GROUPS` were renamed in the combined module; the new dispatcher forwards remaining CLI arguments unchanged. The graph launcher and logging-specific imports are kept lazy to avoid adding dependencies when only shared algorithms/runtime helpers are imported. The component plotting launcher selects the Agg backend at CLI execution rather than import time, so importing the combined plotting module does not globally switch the caller's backend.

## Validation

Environment: Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, PyTorch 2.10.0 CPU, Matplotlib 3.10.8. No production ERA5 arrays, GPU or full native GRIB environment were available.

- All 203 retained Python files parsed and compiled. Every module and every function/class/method has a docstring; no Cyrillic in Python comments or docstrings. Runtime strings are excluded from this check.
- All 34 shell/Slurm launchers passed `bash -n`. Static import scan: zero references to deleted Python modules.
- Structural AST comparison: all 192 unmerged retained files had equivalent executable syntax after ignoring docstrings and import statements, prior to the explicit test updates and whitespace/header cleanup. All 174 original top-level definitions in merged modules were retained; 168 were equivalent after normalizing import statements, docstrings and audit identifiers; six exceptions were CLI entry points (five audit handlers and the component plotting launcher). The graph launcher was moved into a function. This is a structural comparison, not a numerical proof.
- Merged-module imports and main entry-point imports succeeded. Five audit CLI help invocations and hybrid plotting CLI help succeeded.
- Focused spectral, copula, level-2 models and console logging suite: **18 passed**, 5 subtests passed. Added audit-dispatch regression: **1 passed**, 5 subtests passed.
- Separate stationary/plotting/functional test subset: **22 passed, 4 failed**. The failures were already present in the original full suite: three obsolete stationary-regression API assumptions and one sensible-branch fit tolerance failure. They were not rewritten to hide failures.

The existing tests in `tests/test_eigen_binning.py` were migrated from the missing `main_parts` module to the canonical spectral module and updated for the existing explicit variable list, mixed-mode inputs and current figure path. One test uses the real IO reader locally because other legacy tests replace it in `sys.modules`. A new audit CLI dispatch test was added to the existing level-2 test file. No failing scientific assertion was removed.

### Full test-suite comparison

Command (from project root):

```bash
python -m pytest tests Forecast_nn/tests Forecast_1d/tests -q --disable-warnings --continue-on-collection-errors
```

Both runs used `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `PYTHONPATH=.`.

| Result | Original archive | Refactoring 2.0 |
|---|---:|---:|
| Passed | 226 | 234 |
| Failed | 24 | 24 |
| Errors | 5 | 4 |
| Passed subtests | 8 | 13 |

The missing `main_parts` import was resolved and all seven spectral tests plus the new audit-dispatch test pass. The final full suite had no newly failing or erroring test identifiers relative to the original full suite. The remaining suite failures/errors are not treated as a passing release. See the exact test identifiers in `TEST_RESULTS.txt` supplied alongside this project. Existing failures include stale test API expectations, incomplete fixture paths, legacy global `sys.modules` stubs and missing source files (`apply_refactor`, `Forecast_nn.fp_markov_probe`, `Forecast_1d.install_main_task`). The full real-data and GPU regressions remain outstanding.

## Applying and reviewing the archive

The ZIP contains the source tree, including the existing local-only `Cluster/` launchers, but excludes `.git`, IDE settings, bytecode, generated test outputs and caches. The supplied archive already omitted tracked media under `video examples/`; the patch does **not** delete them. Do not delete existing media or external `DATA_ROOT` files when applying this update.

For a Git-based review, the supplied `REVIEW.patch` is relative to the original commit and excludes local-only ignored files. From a clean working tree at the matching revision (the whitespace option accommodates Windows CRLF checkouts):

```bash
git apply --check --ignore-space-change /path/to/REVIEW.patch
git apply --ignore-space-change /path/to/REVIEW.patch
git status --short
git diff
```

Alternatively, copy the source files into your working tree and delete exactly the obsolete module paths listed in the migration table; ordinary extraction will not remove old files. Review the result before committing. No push has been performed.

Documentation standards: [PEP 257](https://peps.python.org/pep-0257/), [NumPy docstring guide](https://numpydoc.readthedocs.io/en/latest/format.html), [NumPy trapezoid](https://numpy.org/doc/stable/reference/generated/numpy.trapezoid.html), [Git apply](https://git-scm.com/docs/git-apply).
