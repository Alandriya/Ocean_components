"""Single source of truth for Ocean_components DATA locations.

Choose ONE active DATA_ROOT below; comment the others. This is a directory for
arrays, ERA5, fitted coefficients, diagnostics, plots, NN checkpoints and logs.
It is not the Git repository root. All project entry points import this file.

The Git repository root is inferred from this file automatically.
"""
from __future__ import annotations

import argparse
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent

# --- Choose ONE, and comment the other two ---
DATA_ROOT = Path(r"D:/OceanFull/")    # Work laptop (Windows)
# DATA_ROOT = Path("/home/aosipova/OceanFull/")       # GPU cluster (Linux)
# DATA_ROOT = Path(r"C:/OceanFull/")               # Second laptop (Windows)

# Legacy routines concatenate a string prefix directly with "DATA/...".
files_path_prefix = str(DATA_ROOT).rstrip("/\\") + "/"


def main(argv=None):
    """Run the project paths workflow using its configured options."""
    p = argparse.ArgumentParser(description="Print or validate the configured paths")
    p.add_argument("--data-root", action="store_true", help="Print only the data directory")
    p.add_argument("--code-root", action="store_true", help="Print only the Git code directory")
    p.add_argument("--check", action="store_true", help="Check essential folders")
    args = p.parse_args(argv)
    if args.data_root:
        print(DATA_ROOT)
    elif args.code_root:
        print(CODE_ROOT)
    else:
        print(f"CODE_ROOT: {CODE_ROOT}")
        print(f"DATA_ROOT: {DATA_ROOT}")
    if args.check:
        for pth in (CODE_ROOT / "Forecast_nn", DATA_ROOT, DATA_ROOT / "DATA"):
            status = "OK" if pth.is_dir() else "MISSING"
            print(f"[{status}] {pth}")
        if not DATA_ROOT.is_dir():
            raise SystemExit(2)


if __name__ == "__main__":
    main()
