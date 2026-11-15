"""Fit independent SARIMA models at ten ocean locations.

Examples:
    python -m Forecast_1d.train --model sarima
    python -m Forecast_1d.train --model sarima --order 2,0,1 --seasonal-order 1,0,0,7
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import PointDataset, dataset_dir, results_dir
from .metrics import save_json
from .models.sarima import (fast_ar1_supported, fit_ar1, fit_sarimax, parse_order)


def build_parser():
    p = argparse.ArgumentParser(description="Fit independent per-location SARIMA models")
    p.add_argument("--model", choices=["sarima"], default="sarima")
    p.add_argument("--dataset-dir", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--order", default="1,0,0")
    p.add_argument("--seasonal-order", default="0,0,0,0")
    p.add_argument("--trend", default="c", choices=["n", "c", "t", "ct"])
    p.add_argument("--backend", default="auto", choices=["auto", "ar1_ols", "statsmodels"])
    p.add_argument("--in-len", type=int, default=30)
    p.add_argument("--out-len", type=int, default=3)
    p.add_argument("--val-days", type=int, default=730)
    p.add_argument("--test-days", type=int, default=365)
    p.add_argument("--fit-days", type=int, default=0,
                   help="Fit on this many last training days (0: full training period)")
    p.add_argument("--maxiter", type=int, default=200)
    p.add_argument("--overwrite", action="store_true")
    return p


def run(args):
    data = PointDataset(dataset_dir(args.dataset_dir))
    split = data.get_splits(args.val_days, args.test_days, args.in_len, args.out_len)
    order, seasonal = parse_order(args.order, 3), parse_order(args.seasonal_order, 4)
    can_fast = fast_ar1_supported(order, seasonal, args.trend)
    if args.backend == "ar1_ols" and not can_fast:
        raise ValueError("ar1_ols backend supports only SARIMA(1,0,0) with trend='c'")
    backend = ("ar1_ols" if args.backend == "auto" and can_fast else
               "statsmodels" if args.backend == "auto" else args.backend)
    if args.fit_days < 0:
        raise ValueError("fit-days must be nonnegative")
    fit_start = max(0, split.train_end - args.fit_days) if args.fit_days else 0
    fit_end = split.train_end
    if fit_end - fit_start < 45:
        raise ValueError("Fitting interval must contain at least 45 days")
    out_dir = results_dir(args.results_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    params_path = out_dir / "params.npz"
    meta_path = out_dir / "train_metadata.json"
    if (params_path.exists() or meta_path.exists()) and not args.overwrite:
        raise FileExistsError(f"Checkpoint exists in {out_dir}; pass --overwrite to refit")
    print(f"[setup] dataset={data.path} dates={data.n_dates} fit=[{fit_start},{fit_end}) "
          f"train_end={split.train_end} backend={backend} order={order} seasonal={seasonal}", flush=True)
    if backend == "ar1_ols":
        params = fit_ar1(data.values, fit_start, fit_end)
    else:
        params = fit_sarimax(data.values, fit_start, fit_end, order,
                            seasonal, args.trend, maxiter=args.maxiter)
    metadata = {
        "model": args.model, "backend": backend,
        "order": list(order), "seasonal_order": list(seasonal), "trend": args.trend,
        "fit_start": fit_start, "fit_end": fit_end, "train_end": split.train_end,
        "val_days": args.val_days, "test_days": args.test_days,
        "in_len": args.in_len, "out_len": args.out_len,
        "first_date": str(data.dates[0]), "last_date": str(data.dates[-1]),
        "train_last_date": str(data.dates[split.train_end-1]),
        "dataset_fingerprint_sha256": data.fingerprint(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parameter_interpretation": ("[intercept, phi, residual_variance, fit_pairs] per point/channel"
                                     if backend == "ar1_ols" else "statsmodels SARIMAX params"),
    }
    # No pickle deserialization is needed to reload the fitted parameters.
    np.savez_compressed(params_path, params=params)
    save_json(meta_path, metadata)
    print(f"[saved] {params_path}", flush=True)
    print(f"[saved] {meta_path}", flush=True)
    return out_dir


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    main()
