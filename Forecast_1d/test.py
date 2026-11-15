"""Evaluate train-free baselines and SARIMA using identical rolling forecast origins.

At origin t, the prediction uses data observed no later than t. Parameters
for SARIMA are never refitted on validation or test observations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import PointDataset, dataset_dir, results_dir
from .metrics import save_json, score_predictions
from .models.baselines import input_mean, persistence
from .models.sarima import forecast_ar1, forecast_sarimax

BASELINES = ("persistence", "input_mean")


def build_parser():
    p = argparse.ArgumentParser(description="Evaluate 1D forecast models and baselines")
    p.add_argument("--model", choices=["sarima", "persistence", "input_mean", "all"],
                   default="all")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--dataset-dir", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--in-len", type=int, default=30)
    p.add_argument("--out-len", type=int, default=3)
    p.add_argument("--val-days", type=int, default=730)
    p.add_argument("--test-days", type=int, default=365)
    p.add_argument("--overwrite", action="store_true")
    return p


def _load_trained_sarima(model_root, data, split):
    meta_path = model_root / "sarima" / "train_metadata.json"
    param_path = model_root / "sarima" / "params.npz"
    if not meta_path.is_file() or not param_path.is_file():
        raise FileNotFoundError("SARIMA checkpoint missing; run python -m Forecast_1d.train first")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (meta["train_end"] != split.train_end or
        meta["val_days"] != split.val_end-split.val_start or
        meta["test_days"] != split.test_end-split.test_start or
        meta["in_len"] != split.in_len or meta["out_len"] != split.out_len):
        raise ValueError("Evaluation split differs from training checkpoint")
    if meta["dataset_fingerprint_sha256"] != data.fingerprint():
        raise ValueError("Dataset fingerprint differs from training checkpoint")
    with np.load(param_path, allow_pickle=False) as source:
        params = np.asarray(source["params"], dtype=np.float64)
    return meta, params


def run(args):
    data = PointDataset(dataset_dir(args.dataset_dir))
    split = data.get_splits(args.val_days, args.test_days, args.in_len, args.out_len)
    origins = split.origins(args.split)
    if origins.size == 0:
        raise ValueError("No valid origins in evaluation interval")
    model_root = results_dir(args.results_dir)
    dest = model_root / "evaluations" / args.split
    names = list(BASELINES) + (["sarima"] if args.model in ("sarima", "all") else [])
    if args.model in BASELINES:
        names = list(BASELINES)  # Always include both training-free baselines.
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        if (dest / f"{name}_predictions.npz").exists() and not args.overwrite:
            raise FileExistsError(f"Prediction files already exist in {dest}; pass --overwrite")
    if (dest / "metrics.json").exists() and not args.overwrite:
        raise FileExistsError(f"Existing metrics in {dest}; pass --overwrite")
    target = np.stack([np.asarray(data.values[origins + h], dtype=np.float64)
                       for h in range(1, split.out_len+1)], axis=-1)
    forecasts = {
        "persistence": persistence(data.values, origins, split.out_len),
        "input_mean": input_mean(data.values, origins, split.out_len, split.in_len),
    }
    if "sarima" in names:
        meta, params = _load_trained_sarima(model_root, data, split)
        if meta["backend"] == "ar1_ols":
            forecasts["sarima"] = forecast_ar1(data.values, origins, split.out_len, params)
        elif meta["backend"] == "statsmodels":
            forecasts["sarima"] = forecast_sarimax(
                data.values, origins, split.out_len, params,
                meta["fit_start"], meta["fit_end"], meta["order"],
                meta["seasonal_order"], meta["trend"])
        else:
            raise ValueError(f"Unknown SARIMA backend in checkpoint: {meta['backend']}")
    shared_mask = np.isfinite(target)
    for name in names:
        shared_mask &= np.isfinite(forecasts[name])
    summary = {
        "split": args.split, "dataset": str(data.path),
        "first_forecast_date": str(data.dates[int(origins[0])+1]),
        "last_forecast_date": str(data.dates[int(origins[-1])+split.out_len]),
        "train_end": split.train_end,
        "eval_origin_count": int(len(origins)),
        "common_finite_target_cells": int(shared_mask.sum()),
        "in_len": split.in_len, "out_len": split.out_len,
        "target_domain": "all ten offshore point centers, sensible and latent",
        "rolling_origin": "observations up to origin only, fixed fitted parameters",
        "models": {},
    }
    for name in names:
        pred = forecasts[name]
        scores = score_predictions(pred, target, baseline=forecasts["persistence"],
                                   shared_mask=shared_mask)
        summary["models"][name] = scores
        filename = dest / f"{name}_predictions.npz"
        np.savez_compressed(filename,
                            prediction=pred.astype(np.float32),
                            target=target.astype(np.float32),
                            origins=origins,
                            dates=data.dates[origins].astype("datetime64[D]"))
        print(f"[saved] {filename}", flush=True)
        for h, result in scores.items():
            print(f"[score] {name} h{h}: rmse={result['joint']['rmse']:.5f} "
                  f"skill={result['persistence_mse_skill']}", flush=True)
    summary_path = dest / "metrics.json"
    save_json(summary_path, summary)
    print(f"[saved] {summary_path}", flush=True)
    return dest


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    main()
