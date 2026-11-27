"""Idempotently add an fp2d_validate task to Ocean_components/main.py.

Usage from repository root:
    python scripts/install_fp2d_validation_task.py
    python scripts/install_fp2d_validation_task.py --dry-run

Creates main.py.fp2d_prevalidation.bak before the first modification.
Does not change the selected task and never overwrites a backup.
"""
from __future__ import annotations

import argparse
from pathlib import Path

BRANCH = '''    elif task == "fp2d_validate":
        from Functional.fp2d_validation import run_validation
        mode = settings.get("fp2d_validation_mode", "synthetic")
        explicit_files = settings.get("fp2d_validation_npzs")
        if explicit_files is None and mode in ("era5", "all"):
            # The existing solver saves these NPZ files after its own task runs.
            from Functional.fokker_planck_2d_solver import _phase_file
            directory = Path(root) / "Functional" / "FokkerPlanck2D" / "Stationary"
            explicit_files = [
                directory / (
                    _phase_file(root, names, start_year, end_year,
                                settings.get("fp2d_method", "sp"), bins).stem
                    + "_stationary_wide_stencil.npz"
                )
                for bins in settings.get("fp2d_validation_era5_bins",
                                         ((60, 60), (80, 80), (100, 100)))
            ]
        return run_validation(
            mode=mode,
            output_dir=Path(root) / "Functional" / "FokkerPlanck2D" / "Validation",
            synthetic_sizes=settings.get("fp2d_validation_sizes", (40, 48, 56)),
            stationary_npzs=explicit_files or (),
            max_stencil_radius=settings.get("fp2d_solver_max_stencil_radius", 9),
            rate_tolerance=settings.get("fp2d_solver_rate_tolerance", 1e-10),
            correlation=settings.get("fp2d_validation_correlation", 0.36),
            compare_era5_rectangles=settings.get("fp2d_validation_compare_era5_rectangles", False),
        )
'''

CONFIG = '''    # task = "fp2d_validate"    # Select below in main(): task = "fp2d_validate"
    # First run "synthetic"; after all FP2D solves, use "all".
    fp2d_validation_mode = "synthetic"  # "synthetic", "era5", "all"
    fp2d_validation_sizes = (40, 48, 56)
    fp2d_validation_correlation = 0.36
    fp2d_validation_era5_bins = ((60, 60), (80, 80), (100, 100))
    fp2d_validation_compare_era5_rectangles = False  # Optional extra real-coefficient CTMC/FV comparison
    fp2d_validation_npzs = None  # Optional explicit list of existing *_stationary_wide_stencil.npz paths
'''


def install(main_file: Path, dry_run=False):
    """Install the requested install 2D Fokker-Planck validation task data using the supplied inputs."""
    main_file = Path(main_file)
    if not main_file.is_file():
        raise FileNotFoundError(f"No main.py found: {main_file}")
    original = main_file.read_text(encoding="utf-8")
    body = original
    if 'elif task == "fp2d_validate":' not in body:
        anchor = '    elif task == "components":'
        if body.count(anchor) != 1:
            raise RuntimeError("Cannot find unique elif task == components anchor")
        body = body.replace(anchor, BRANCH + anchor, 1)
    if "fp2d_validation_sizes =" not in body:
        anchor = '    # task = "fokker_planck_2d_solve"'
        if body.count(anchor) != 1:
            raise RuntimeError("Cannot find unique fp2d_solve configuration anchor")
        body = body.replace(anchor, CONFIG + '\n' + anchor, 1)
    if body == original:
        print("Already installed. main.py left untouched.")
        return
    compile(body, str(main_file), "exec")
    if dry_run:
        print("Patch applies cleanly. No changes written (--dry-run).")
        return
    backup = main_file.with_name(main_file.name + ".fp2d_prevalidation.bak")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    main_file.write_text(body, encoding="utf-8")
    print(f"Added task fp2d_validate to {main_file}. Backup: {backup}")
    print('Choose task = "fp2d_validate" near the bottom of main() and run python main.py.')


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--main", type=Path, default=Path("main.py"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    install(args.main, args.dry_run)
