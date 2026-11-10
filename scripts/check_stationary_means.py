"""Compare stationary-PDF means with optional empirical ERA5 heat-flux means.

By default, reads only small saved JSON files (fast). With ``--read-raw``,
streams the original daily arrays to calculate the empirical mean, allowing
comparison of the assembled 1979-2026 array to the weighted decade means.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow: python scripts/check_stationary_means.py --root ...
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PERIODS = ((1979, 1989), (1989, 1999), (1999, 2009),
           (2009, 2019), (2019, 2026), (1979, 2026))


def raw_field_summary(root, name, start_year, end_year, chunk_days=64):
    """Return the sum and count of finite water-cell observations, by chunks."""
    from Data_processing.io import load_field, read_ocean_mask

    field = load_field(root, name, start_year, end_year)
    mask = read_ocean_mask(root, field.shape[1:])
    total = 0.0
    count = 0
    for start in range(0, len(field), chunk_days):
        block = np.asarray(field[start:start + chunk_days])
        valid = np.isfinite(block) & mask[None, :, :]
        total += float(np.sum(block, where=valid, initial=0.0, dtype=np.float64))
        count += int(np.count_nonzero(valid))
    if count == 0:
        raise ValueError(f"No finite water observations: {name}, {start_year}-{end_year}")
    return total, count


def report(root, names=("sensible", "latent"), *, read_raw=False, chunk_days=64):
    """Print PDF and, optionally, observed means for all known periods."""
    root = Path(root)
    read_raw = True
    for name in names:
        print(f"\n{name} — MEAN DIAGNOSTICS")
        print("Period       stationary_PDF_mean    drift_RMSE       ERA5_raw_mean")
        raw_parts = {}
        for start_year, end_year in PERIODS:
            period = f"{start_year}-{end_year}"
            file = (root / "Functional" / "Stationary" /
                    f"{'-'.join(names)}_{period}.json")
            pdf_mean, drift_rmse = None, None
            if file.is_file():
                fit = json.loads(file.read_text(encoding="utf-8")).get("fits", {}).get(name, {})
                pdf_mean = fit.get("mean")
                drift_rmse = fit.get("drift_rmse")
            empirical = None
            if read_raw:
                try:
                    total, count = raw_field_summary(root, name, start_year, end_year,
                                                     chunk_days=chunk_days)
                    raw_parts[(start_year, end_year)] = (total, count)
                    empirical = total / count
                except FileNotFoundError as err:
                    print(f"NOTE: data not available for {period}: {err}")
            fmt = lambda x: f"{x:>18.5f}" if x is not None else "               N/A"
            print(f"{period:<13}{fmt(pdf_mean)} {fmt(drift_rmse)} {fmt(empirical)}")
        if read_raw and all(p in raw_parts for p in PERIODS):
            segments = [raw_parts[p] for p in PERIODS[:-1]]
            combined = raw_parts[PERIODS[-1]]
            mean_decades = sum(v[0] for v in segments) / sum(v[1] for v in segments)
            mean_full = combined[0] / combined[1]
            difference = mean_full - mean_decades
            print(f"ERA5 weighted mean of decades: {mean_decades:.8g}")
            print(f"ERA5 mean of assembled array: {mean_full:.8g}")
            print(f"Difference: {difference:+.8g}")
            if not np.isclose(mean_decades, mean_full, atol=1e-5, rtol=1e-6):
                print("WARNING: assembled raw data does NOT match the decade-weighted mean. "
                      "Check field concatenation, masks, units and day ordering.")
            else:
                print("OK: assembled raw mean matches the weighted decade means.")
        print("Note: stationary PDF mean is NOT the mean of ERA5 observations; "
              "it is also not required to equal the mean of decadal PDF means.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="D:/Nastya/Data/OceanFull/",
                        help="OceanFull data root")
    parser.add_argument("--read-raw", action="store_true",
                        help="Also stream ERA5 raw fields to compare empirical means")
    parser.add_argument("--chunk-days", type=int, default=64)
    args = parser.parse_args()
    if args.chunk_days <= 0:
        parser.error("--chunk-days must be positive")
    report(args.root, read_raw=args.read_raw, chunk_days=args.chunk_days)


if __name__ == "__main__":
    main()
