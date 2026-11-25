"""Add an idempotent fp_followup_validate task to Ocean_components/main.py.

Run from repository root:
    python scripts/install_fp_followup_task.py --dry-run
    python scripts/install_fp_followup_task.py

The script preserves the existing task selection and all prior configuration.
It makes a backup only if a change is needed. It never overwrites a backup.
"""
from __future__ import annotations

import argparse
from pathlib import Path

BRANCH = '''    elif task == "fp_followup_validate":
        from Functional.fp_followup_validation import run_followup
        return run_followup(
            mode=settings.get("fp_followup_mode", "wide_ou"),
            root=root,
            output_dir=(Path(root) / "Functional" / "FokkerPlanck2D" /
                        "Validation" / "Followup"),
            hybrid_periods=settings.get("fp_followup_hybrid_periods", (
                (1979, 1989), (1989, 1999), (1999, 2009),
                (2009, 2019), (2019, 2026), (1979, 2026))),
            names=names,
            wide_sizes=settings.get("fp_followup_wide_sizes", (40, 60, 80, 100)),
            max_stencil_radius=settings.get("fp_followup_radius", 9),
            require_all_hybrids=settings.get("fp_followup_require_all_hybrids", False),
        )
'''

CONFIG = '''    # NEW FOLLOW-UP VALIDATION (read-only for existing results).
    # task = "fp_followup_validate"
    fp_followup_mode = "wide_ou"  # "wide_ou", "hybrid", "all"
    fp_followup_wide_sizes = (40, 60, 80, 100)
    fp_followup_radius = 9
    fp_followup_hybrid_periods = (
        (1979, 1989), (1989, 1999), (1999, 2009),
        (2009, 2019), (2019, 2026), (1979, 2026),
    )
    fp_followup_require_all_hybrids = False
'''


def install(path: Path, dry_run=False):
    path = Path(path)
    old = path.read_text(encoding="utf-8")
    new = old
    branch_key = '    elif task == "fp_followup_validate":'
    if branch_key not in new:
        anchor = '    elif task == "fokker_planck_2d_solve":'
        if anchor not in new:
            raise ValueError(f"Cannot find solver task anchor in {path}")
        new = new.replace(anchor, BRANCH + anchor, 1)
    if '    fp_followup_mode = ' not in new:
        anchor = '    # task = "fokker_planck_2d_solve"'
        if anchor not in new:
            raise ValueError(f"Cannot find configuration anchor in {path}")
        new = new.replace(anchor, CONFIG + '\n' + anchor, 1)
    if new == old:
        print("Already installed. No changes to main.py.")
        return
    compile(new, str(path), "exec")
    if dry_run:
        print("Installation is applicable. --dry-run left main.py unchanged.")
        return
    backup = path.with_suffix(path.suffix + ".before_fp_followup.bak")
    if not backup.exists():
        backup.write_text(old, encoding="utf-8")
    path.write_text(new, encoding="utf-8")
    print(f"New task installed in {path}. Backup: {backup}")
    print('Set task = "fp_followup_validate" near the bottom of main() to run.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, default=Path("main.py"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    install(args.main, args.dry_run)
