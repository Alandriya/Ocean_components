"""Train level-0 SARIMA or independent level-1/2 neural models per location."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import PointDataset, dataset_dir, results_dir
from .metrics import save_json
from .experiment import (DEFAULT_HORIZON, DEFAULT_POINTS, DEFAULT_TARGET_MODE,
                         TARGET_MODES, horizon_root, target_root, select_points)
from .models.sarima import fast_ar1_supported, fit_ar1, fit_sarimax, parse_order

ALL_POINTS = DEFAULT_POINTS
from .registry import NEURAL_MODELS, model_level


def build_parser():
    """Construct the command-line argument parser."""
    p = argparse.ArgumentParser(description="Train Forecast_1d statistical or neural models")
    p.add_argument("--model", choices=["sarima", *NEURAL_MODELS], default="sarima")
    p.add_argument("--points", nargs="+", default=None,
                   help="Neural models default to all loaded locations; SARIMA fits all locations.")
    p.add_argument("--dataset-dir", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--target-mode", choices=TARGET_MODES, default=DEFAULT_TARGET_MODE,
                   help="Neural target: raw, train-only anomalies, or OOF Seasonal AR(1) corrections.")
    p.add_argument("--climatology-radius", type=int, default=15,
                   help="Calendar smoothing radius in days, fitted on train only.")
    p.add_argument("--residual-oof-folds", type=int, default=4,
                   help="Expanding time folds for SAR residual train targets (default: 4).")
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
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--min-delta", type=float, default=1e-5)
    p.add_argument("--scheduler", choices=["none", "plateau"], default="plateau")
    p.add_argument("--lr-patience", type=int, default=3)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--monitor", choices=["normalized", "sar"], default="normalized",
                   help="Checkpoint and scheduler criterion: normalized MSE or 1 - macro SAR skill on validation.")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--overwrite", action="store_true")
    return p


def _run_sarima(args):
    """Compute run SARIMA for train from ``args``."""
    if getattr(args, "points", None):
        raise ValueError("The SARIMA level-0 model is fitted independently at all loaded points; omit --points.")
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
    """Construct windows for the train workflow."""
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
    """Resolve the requested compute device."""
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _train_one_point(args, data, split, point_id, fingerprint):
    """Train one point for the train workflow."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from .models.neural import build_model, default_config

    point_index = data.point_ids.index(point_id)
    in_len = args.in_len
    train_origins = np.arange(
        in_len - 1, split.train_end - args.out_len, dtype=np.int64
    )
    val_origins = split.origins("val")
    target_mode = getattr(args, "target_mode", "raw")
    radius = getattr(args, "climatology_radius", 15)
    if target_mode not in TARGET_MODES:
        raise ValueError(f"Unsupported target mode: {target_mode}")
    if target_mode == "sar_residual":
        if args.model not in ("mlp", "tcn", "gru"):
            raise ValueError("SAR residual currently supports Level-1 MLP, TCN and GRU only")
        from .sar_residual import prepare_residual_training
        residual = prepare_residual_training(
            data.values[:, point_index, :], data.dates, split.train_end,
            val_origins, in_len, args.out_len, radius,
            getattr(args, "residual_oof_folds", 4)
        )
        x_train, y_train = residual.x_train, residual.y_train
        x_val, y_val = residual.x_val, residual.y_val
        means, scales = residual.input_mean, residual.input_scale
    else:
        climatology = None
        if target_mode == "anomaly":
            from Forecast_1d.models.baselines import calendar_index, fit_seasonal
            if not 0 <= radius < 183:
                raise ValueError("climatology-radius must be in 0..182")
            # The fitted seasonal cycle sees only dates and observations in train.
            fitted, _ = fit_seasonal(
                data.values[:, point_index:point_index+1, :],
                data.dates, split.train_end, radius=radius
            )
            climatology = fitted[:, 0, :].copy()  # [366,2]
            seasonal_values = climatology[calendar_index(data.dates)]
            point_values = np.asarray(data.values[:, point_index, :], dtype=np.float64) - seasonal_values
            work_values, work_index = point_values[:, None, :], 0
        else:
            work_values, work_index = data.values, point_index
        train_raw = np.asarray(work_values[:split.train_end, work_index, :], dtype=np.float64)
        train_raw = np.where(np.isfinite(train_raw), train_raw, np.nan)
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
            work_values, train_origins, work_index, in_len, args.out_len, means
        )
        x_val, y_val = _make_windows(
            work_values, val_origins, work_index, in_len, args.out_len, means
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
    if target_mode == "sar_residual":
        from .sar_residual import zero_output_head
        zero_output_head(model, args.model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    monitor = getattr(args, "monitor", "normalized")
    scheduler_name = getattr(args, "scheduler", "none")
    grad_clip = getattr(args, "grad_clip", 1.0)
    lr_patience = getattr(args, "lr_patience", 3)
    lr_factor = getattr(args, "lr_factor", 0.5)
    min_lr = getattr(args, "min_lr", 1e-6)
    if grad_clip <= 0 or args.patience < 1 or args.epochs < 1 or args.min_delta < 0:
        raise ValueError("Invalid clipping, epoch or early stopping settings")
    if scheduler_name == "plateau" and not (0 < min_lr <= args.learning_rate and 0 < lr_factor < 1 and lr_patience >= 0 and args.patience > lr_patience + 1):
        raise ValueError("Invalid scheduler settings; early-stop patience must exceed LR patience + 1")
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience,
        threshold=args.min_delta, threshold_mode="abs", min_lr=min_lr)
        if scheduler_name == "plateau" else None)
    sar_denominator = None
    if monitor == "sar":
        if target_mode == "sar_residual":
            # In correction space, the Seasonal AR(1) predictor is exactly zero.
            sar_denominator = np.mean(y_val.astype(np.float64) ** 2, axis=0)
            if not np.isfinite(sar_denominator).all() or np.any(sar_denominator <= 1e-12):
                raise ValueError("Nonfinite/zero SAR residual baseline MSE in a channel/lead")
        else:
            sar_denominator = None
        if target_mode != "sar_residual":
            # The denominator is used ONLY for validation/checkpoint selection.
            # It is not propagated into the training objective.
            from Forecast_1d.models.baselines import fit_seasonal, forecast_seasonal
            values = data.values[:, point_index:point_index+1, :]
            clim, phi = fit_seasonal(values, data.dates, split.train_end, radius=15)
            sar = forecast_seasonal(values, data.dates, val_origins, args.out_len, clim, phi)[:, 0].transpose(0, 2, 1)
            # Fail rather than silently compare different supports. All targets and
            # last observations must be finite in this optional monitor.
            if len(y_val) != len(val_origins) or not np.isfinite(sar).all():
                raise ValueError("SAR monitor requires complete validation windows and finite last observations; use normalized monitor otherwise")
            if target_mode == "anomaly":
                # Convert the raw SAR forecast onto the anomaly training scale.
                from Forecast_1d.models.baselines import calendar_index
                lead_dates = data.dates[val_origins[:, None] + np.arange(1, args.out_len+1)]
                sar = sar - climatology[calendar_index(lead_dates)]
            normalized_sar = (sar - means[None, None, :]) / scales[None, None, :]
            sar_denominator = np.mean((normalized_sar - y_val.astype(np.float64)) ** 2, axis=0)
            if not np.isfinite(sar_denominator).all() or np.any(sar_denominator <= 1e-12):
                raise ValueError("Invalid/zero SAR validation MSE in a channel/lead")
    loss_fn = torch.nn.MSELoss()
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    x_val_t = torch.from_numpy(x_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    out_dir = target_root(results_dir(args.results_dir), args.out_len, target_mode) / args.model / point_id
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "best.pt"
    metadata_path = out_dir / "train_metadata.json"
    history_path = out_dir / "history.json"
    if any(p.exists() for p in (checkpoint_path, metadata_path, history_path)) and not args.overwrite:
        raise FileExistsError(f"Existing {args.model}/{point_id} model; pass --overwrite")

    best_loss = float("inf")
    best_epoch = 0
    best_normalized = float("inf")
    best_sar_skill = None
    patience_anchor = float("inf")
    stop_reason = "max_epochs"
    best_state = None
    if target_mode == "sar_residual":
        # Keep the exact zero-correction SAR baseline as epoch 0.
        # This guarantees no worse *validation selection score* than SAR.
        zero_cell_mse = np.mean(y_val.astype(np.float64) ** 2, axis=0)
        best_normalized = float(zero_cell_mse.mean())
        best_loss = (float((zero_cell_mse / sar_denominator).mean())
                     if sar_denominator is not None else best_normalized)
        best_sar_skill = 0.0 if monitor == "sar" else None
        patience_anchor = best_loss
        best_state = {key: value.detach().cpu().clone()
                      for key, value in model.state_dict().items()}
    epochs_without_improvement = 0
    history = []
    print(
        f"[fit] model={args.model} point={point_id} device={device} "
        f"target_mode={target_mode} train_windows={len(train_ds)} val_windows={len(x_val)} "
        f"input={in_len}x2 output={args.out_len}x2 config={config}", flush=True
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum, train_count = 0.0, 0
        clipped, n_batches, grad_norm_sum = 0, 0, 0.0
        lr_used = optimizer.param_groups[0]["lr"]
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            loss = loss_fn(prediction, yb)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip, error_if_nonfinite=True).detach().cpu())
            n_batches += 1
            clipped += int(grad_norm > grad_clip)
            grad_norm_sum += grad_norm
            optimizer.step()
            train_sum += float(loss.detach().cpu()) * len(xb)
            train_count += len(xb)
        train_loss = train_sum / max(train_count, 1)

        model.eval()
        squared_sum = torch.zeros((args.out_len, 2), dtype=torch.float64, device=device)
        with torch.no_grad():
            for i in range(0, len(x_val_t), args.batch_size):
                delta = (model(x_val_t[i:i+args.batch_size]) - y_val_t[i:i+args.batch_size]).double()
                squared_sum += delta.square().sum(dim=0)
        cell_mse = squared_sum.cpu().numpy() / len(x_val_t)
        val_loss = float(cell_mse.mean())
        selection_loss = (float((cell_mse / sar_denominator).mean())
                          if sar_denominator is not None else val_loss)
        if not np.isfinite(train_loss) or not np.isfinite(selection_loss):
            raise RuntimeError(f"Nonfinite training or validation loss: {args.model}/{point_id}")
        sar_skill = 1.0 - selection_loss if monitor == "sar" else None
        if scheduler is not None:
            scheduler.step(selection_loss)
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "selection_loss": selection_loss, "val_sar_skill": sar_skill,
            "learning_rate": lr_used, "next_learning_rate": optimizer.param_groups[0]["lr"],
            "mean_preclip_grad_norm": grad_norm_sum / n_batches,
            "clip_fraction": clipped / n_batches,
        })
        print(
            f"[epoch] model={args.model} point={point_id} epoch={epoch}/{args.epochs} "
            f"train_loss={train_loss:.7g} val_loss={val_loss:.7g} "
            f"selection_loss={selection_loss:.7g} lr={lr_used:.3g} clip_fraction={clipped/n_batches:.3f}", flush=True
        )
        # Exact minimum controls checkpoint, min_delta controls patience only.
        if selection_loss < best_loss:
            best_loss = selection_loss
            best_normalized = val_loss
            best_sar_skill = sar_skill
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if selection_loss < patience_anchor - args.min_delta:
            patience_anchor = selection_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                stop_reason = "early_stopping"
                print(f"[early-stop] model={args.model} point={point_id} epoch={epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError(f"No valid checkpoint was produced for {args.model}/{point_id}")
    temp_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
    torch.save({"state_dict": best_state}, temp_checkpoint)
    temp_checkpoint.replace(checkpoint_path)

    metadata = {
        "model": args.model,
        "model_level": model_level(args.model),
        "point_id": point_id,
        "target_mode": target_mode,
        "anomaly_climatology": ({"radius_days": radius, "fit_end_exclusive": split.train_end,
                                "calendar": "366 month-day slots"}
                               if target_mode == "anomaly" else None),
        "sar_residual": (
            {"reference": "seasonal_ar1", "reference_fit_end_exclusive": split.train_end,
             "climatology_radius_days": radius,
             "oof_folds": len(residual.fold_edges) - 1,
             "oof_fold_edges": list(residual.fold_edges),
             "oof_train_origin_count": int(len(residual.train_origins)),
             "input_feature": "seasonal_anomaly", "residual_scale": residual.residual_scale.tolist(),
             "residual_center": [0.0, 0.0], "zero_output_initialization": True,
             "baseline_epoch_zero_candidate": True,
             "oof_burn_fraction": 0.33}
            if target_mode == "sar_residual" else None),
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
        "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "best_epoch": best_epoch,
        "best_validation_loss_normalized": best_normalized,
        "selection_metric": monitor,
        "best_selection_loss": best_loss,
        "best_validation_sar_skill": best_sar_skill,
        "stop_reason": stop_reason,
        "training_config": {"optimizer": "AdamW", "learning_rate": args.learning_rate,
                            "weight_decay": args.weight_decay, "batch_size": args.batch_size,
                            "epochs": args.epochs, "patience": args.patience,
                            "min_delta": args.min_delta, "grad_clip": grad_clip,
                            "scheduler": scheduler_name, "lr_patience": lr_patience,
                            "lr_factor": lr_factor, "min_lr": min_lr},
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
        f"selection_loss={best_loss:.7g}", flush=True
    )
    del model, optimizer, train_loader, x_val_t, y_val_t
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out_dir


def _run_neural(args):
    """Compute run neural for train from ``args``."""
    if args.in_len is None:
        args.in_len = 60
    if args.in_len < 2 or args.out_len < 1:
        raise ValueError("Neural input length must be >=2 and horizon >=1")
    data = PointDataset(dataset_dir(args.dataset_dir))
    point_ids = select_points(data.point_ids, args.points)
    split = data.get_splits(args.val_days, args.test_days, args.in_len, args.out_len)
    fingerprint = data.fingerprint()
    output = []
    for point_id in point_ids:
        output.append(_train_one_point(args, data, split, point_id, fingerprint))
    return output


def run(args):
    """Run the train workflow using its configured options."""
    if args.model == "sarima":
        if getattr(args, "target_mode", "raw") == "sar_residual":
            raise ValueError("SAR-residual uses Seasonal AR(1) fitted internally; no sarima checkpoint is needed")
        return _run_sarima(args)
    if args.model in NEURAL_MODELS:
        return _run_neural(args)
    raise ValueError(f"Unsupported model: {args.model}")


def main(argv=None):
    """Run the train workflow using its configured options."""
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
