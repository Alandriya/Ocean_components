"""Train level-0 SARIMA or independent level-1 neural models per location."""
from __future__ import annotations

import argparse
import copy
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import PointDataset, dataset_dir, results_dir
from .metrics import save_json
from .experiment import DEFAULT_HORIZON, horizon_root
from .models.sarima import fast_ar1_supported, fit_ar1, fit_sarimax, parse_order

ALL_POINTS = tuple(f"P{i}" for i in range(1, 11))
DEFAULT_POINTS = ("P3", "P5", "P7", "P9")
NEURAL_MODELS = ("mlp", "tcn", "gru")


def build_parser():
    p = argparse.ArgumentParser(description="Train Forecast_1d statistical or neural models")
    p.add_argument("--model", choices=["sarima", *NEURAL_MODELS], default="sarima")
    p.add_argument("--points", nargs="+", choices=ALL_POINTS, default=None,
                   help="Neural models default to P3 P5 P7 P9; SARIMA fits all ten points.")
    p.add_argument("--dataset-dir", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--order", default="1,0,0")
    p.add_argument("--seasonal-order", default="0,0,0,0")
    p.add_argument("--trend", default="c", choices=["n", "c", "t", "ct"])
    p.add_argument("--backend", default="auto", choices=["auto", "ar1_ols", "statsmodels"])
    p.add_argument("--in-len", type=int, default=None,
                   help="Neural default: 60; SARIMA metadata default: 30.")
    p.add_argument("--out-len", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--val-days", type=int, default=730)
    p.add_argument("--test-days", type=int, default=365)
    p.add_argument("--fit-days", type=int, default=0)
    p.add_argument("--maxiter", type=int, default=200)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--min-delta", type=float, default=1e-5)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--overwrite", action="store_true")
    return p


def _run_sarima(args):
    if getattr(args, "points", None):
        raise ValueError("The SARIMA level-0 model is fitted independently at all ten points; omit --points.")
    in_len = getattr(args, "in_len", None) or 30
    data = PointDataset(dataset_dir(args.dataset_dir))
    split = data.get_splits(args.val_days, args.test_days, in_len, args.out_len)
    order, seasonal = parse_order(args.order, 3), parse_order(args.seasonal_order, 4)
    can_fast = fast_ar1_supported(order, seasonal, args.trend)
    if args.backend == "ar1_ols" and not can_fast:
        raise ValueError("ar1_ols supports only SARIMA(1,0,0), trend='c'")
    backend = ("ar1_ols" if args.backend == "auto" and can_fast else
               "statsmodels" if args.backend == "auto" else args.backend)
    if args.fit_days < 0:
        raise ValueError("fit-days must be nonnegative")
    fit_start = max(0, split.train_end - args.fit_days) if args.fit_days else 0
    fit_end = split.train_end
    if fit_end - fit_start < 45:
        raise ValueError("Fitting interval must contain at least 45 days")
    out_dir = horizon_root(results_dir(args.results_dir), args.out_len) / "sarima"
    out_dir.mkdir(parents=True, exist_ok=True)
    params_path, meta_path = out_dir / "params.npz", out_dir / "train_metadata.json"
    if (params_path.exists() or meta_path.exists()) and not args.overwrite:
        raise FileExistsError(f"Checkpoint exists in {out_dir}; pass --overwrite to refit")
    print(
        f"[setup] model=sarima dates={data.n_dates} train_end={split.train_end} "
        f"backend={backend} order={order} seasonal={seasonal}", flush=True
    )
    if backend == "ar1_ols":
        params = fit_ar1(data.values, fit_start, fit_end)
    else:
        params = fit_sarimax(
            data.values, fit_start, fit_end, order, seasonal,
            args.trend, maxiter=args.maxiter
        )
    metadata = {
        "model": "sarima", "level": 0, "backend": backend,
        "order": list(order), "seasonal_order": list(seasonal), "trend": args.trend,
        "fit_start": fit_start, "fit_end": fit_end, "train_end": split.train_end,
        "val_days": args.val_days, "test_days": args.test_days,
        "in_len": in_len, "out_len": args.out_len,
        "first_date": str(data.dates[0]), "last_date": str(data.dates[-1]),
        "train_last_date": str(data.dates[split.train_end - 1]),
        "dataset_fingerprint_sha256": data.fingerprint(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parameter_interpretation": (
            "[intercept, phi, residual_variance, fit_pairs] per point/channel"
            if backend == "ar1_ols" else "statsmodels SARIMAX params"
        ),
    }
    np.savez_compressed(params_path, params=params)
    save_json(meta_path, metadata)
    print(f"[saved] {params_path}", flush=True)
    print(f"[saved] {meta_path}", flush=True)
    return out_dir


def _make_windows(values, origins, point_index, in_len, out_len, mean):
    xs, ys = [], []
    for origin in origins:
        x = np.asarray(
            values[origin - in_len + 1:origin + 1, point_index, :],
            dtype=np.float32
        )
        y = np.asarray(
            values[origin + 1:origin + out_len + 1, point_index, :],
            dtype=np.float32
        )
        if x.shape != (in_len, 2) or y.shape != (out_len, 2):
            continue
        if not np.isfinite(y).all():
            continue
        x = np.where(np.isfinite(x), x, mean[None, :])
        xs.append(x)
        ys.append(y)
    if not xs:
        raise ValueError(f"No complete training windows for point P{point_index + 1}")
    return np.stack(xs), np.stack(ys)


def _resolve_device(torch, requested):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _train_one_point(args, data, split, point_id, fingerprint):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from .models.neural import build_model, default_config

    point_index = int(point_id[1:]) - 1
    in_len = args.in_len
    train_origins = np.arange(
        in_len - 1, split.train_end - args.out_len, dtype=np.int64
    )
    val_origins = split.origins("val")
    train_raw = np.asarray(data.values[:split.train_end, point_index, :], dtype=np.float64)
    means = np.array([
        np.nanmean(train_raw[:, c]) if np.isfinite(train_raw[:, c]).any() else np.nan
        for c in range(2)
    ], dtype=np.float64)
    scales = np.array([
        np.nanstd(train_raw[:, c]) if np.isfinite(train_raw[:, c]).any() else np.nan
        for c in range(2)
    ], dtype=np.float64)
    if not np.isfinite(means).all() or not np.isfinite(scales).all():
        raise ValueError(f"Point {point_id} has a channel with no finite training values")
    scales = np.where(scales < 1e-8, 1.0, scales)
    x_train, y_train = _make_windows(
        data.values, train_origins, point_index, in_len, args.out_len, means
    )
    x_val, y_val = _make_windows(
        data.values, val_origins, point_index, in_len, args.out_len, means
    )
    x_train = ((x_train - means) / scales).astype(np.float32)
    y_train = ((y_train - means) / scales).astype(np.float32)
    x_val = ((x_val - means) / scales).astype(np.float32)
    y_val = ((y_val - means) / scales).astype(np.float32)

    torch.manual_seed(args.seed + point_index)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + point_index)
    device = _resolve_device(torch, args.device)
    config = default_config(
        args.model, hidden_size=args.hidden_size, dropout=args.dropout
    )
    model = build_model(
        args.model, in_len, args.out_len, config, input_channels=2
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_fn = torch.nn.MSELoss()
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    x_val_t = torch.from_numpy(x_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    out_dir = horizon_root(results_dir(args.results_dir), args.out_len) / args.model / point_id
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "best.pt"
    metadata_path = out_dir / "train_metadata.json"
    history_path = out_dir / "history.json"
    if any(p.exists() for p in (checkpoint_path, metadata_path, history_path)) and not args.overwrite:
        raise FileExistsError(f"Existing {args.model}/{point_id} model; pass --overwrite")

    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []
    print(
        f"[fit] model={args.model} point={point_id} device={device} "
        f"train_windows={len(train_ds)} val_windows={len(x_val)} "
        f"input={in_len}x2 output={args.out_len}x2 config={config}", flush=True
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum, train_count = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            loss = loss_fn(prediction, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_sum += float(loss.detach().cpu()) * len(xb)
            train_count += len(xb)
        train_loss = train_sum / max(train_count, 1)

        model.eval()
        with torch.no_grad():
            val_prediction = model(x_val_t)
            val_loss = float(loss_fn(val_prediction, y_val_t).cpu())

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss
        })
        print(
            f"[epoch] model={args.model} point={point_id} epoch={epoch}/{args.epochs} "
            f"train_loss={train_loss:.7g} val_loss={val_loss:.7g}", flush=True
        )
        if val_loss < best_loss - args.min_delta:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"[early-stop] model={args.model} point={point_id} epoch={epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError(f"No valid checkpoint was produced for {args.model}/{point_id}")
    temp_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
    torch.save({"state_dict": best_state}, temp_checkpoint)
    temp_checkpoint.replace(checkpoint_path)

    metadata = {
        "model": args.model,
        "model_level": 1,
        "point_id": point_id,
        "channels": ["sensible", "latent"],
        "input_channels": 2,
        "feature_names": ["sensible", "latent"],
        "input_length": in_len,
        "horizon": args.out_len,
        "train_end": split.train_end,
        "val_days": args.val_days,
        "test_days": args.test_days,
        "normalization": {
            "mean": means.tolist(),
            "scale": scales.tolist(),
            "fit_interval": [0, split.train_end],
        },
        "model_config": config,
        "best_epoch": best_epoch,
        "best_validation_loss_normalized": best_loss,
        "epochs_completed": len(history),
        "seed": args.seed,
        "dataset_fingerprint_sha256": fingerprint,
        "first_date": str(data.dates[0]),
        "train_last_date": str(data.dates[split.train_end - 1]),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(metadata_path, metadata)
    save_json(history_path, history)
    print(f"[saved] {checkpoint_path}", flush=True)
    print(f"[saved] {metadata_path}", flush=True)
    print(
        f"[best] model={args.model} point={point_id} epoch={best_epoch} "
        f"val_loss={best_loss:.7g}", flush=True
    )
    del model, optimizer, train_loader, x_val_t, y_val_t
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out_dir


def _run_neural(args):
    if args.in_len is None:
        args.in_len = 60
    if args.in_len < 2 or args.out_len < 1:
        raise ValueError("Neural input length must be >=2 and horizon >=1")
    point_ids = tuple(args.points or DEFAULT_POINTS)
    if len(set(point_ids)) != len(point_ids):
        raise ValueError("Point IDs must be unique")
    data = PointDataset(dataset_dir(args.dataset_dir))
    split = data.get_splits(args.val_days, args.test_days, args.in_len, args.out_len)
    fingerprint = data.fingerprint()
    output = []
    for point_id in point_ids:
        output.append(_train_one_point(args, data, split, point_id, fingerprint))
    return output


def run(args):
    if args.model == "sarima":
        return _run_sarima(args)
    if args.model in NEURAL_MODELS:
        return _run_neural(args)
    raise ValueError(f"Unsupported model: {args.model}")


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
