"""Train multivariate-history, two-target 60->14 point forecasters.

This is deliberately separate from legacy train.py to preserve prior flux-only
models, results, checkpoint semantics and SAR-residual experiments.

Example:
 python -m Forecast_1d.train_features --model gru --feature-set all --target-mode anomaly --device cuda --amp
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .feature_experiment import (FEATURE_SETS, MODEL_NAMES, TARGET_MODES,
                                  FeatureExperiment, PointTransform, base_folder,
                                  save_json, set_seed, torch_device)
from .experiment import select_points


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=MODEL_NAMES, default="gru")
    p.add_argument("--points", nargs="+", default=None)
    p.add_argument("--feature-set", choices=FEATURE_SETS, default="all",
                   help="flux=2, era5=8, all=20 historical channels")
    p.add_argument("--target-mode", choices=TARGET_MODES, default="anomaly")
    p.add_argument("--features-dir", type=Path, default=None)
    p.add_argument("--dataset-dir", type=Path, default=None,
                   help="Original atlantic15 dataset, for unchanged split dates")
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--run-id", default=None, help="Defaults to seed<N>; change for alternate hyperparameters")
    p.add_argument("--in-len", type=int, default=60)
    p.add_argument("--out-len", type=int, default=14)
    p.add_argument("--val-days", type=int, default=730)
    p.add_argument("--test-days", type=int, default=365)
    p.add_argument("--climatology-radius", type=int, default=15)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--min-delta", type=float, default=1e-5)
    p.add_argument("--scheduler", choices=("none", "plateau"), default="plateau")
    p.add_argument("--lr-patience", type=int, default=3)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    p.add_argument("--amp", action="store_true", help="Enable CUDA BF16 autocast (A100)")
    p.add_argument("--overwrite", action="store_true")
    return p


def _batched_score(model, x, y, device, size, amp):
    import torch
    model.eval()
    errors = 0.0
    n = 0
    physical_sq_error = np.zeros(2, dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, len(x), size):
            xb = torch.from_numpy(x[start:start+size]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=amp and device.type == "cuda"):
                pred = model(xb).float()
            yb = torch.from_numpy(y[start:start+size]).to(device)
            err = pred - yb
            errors += err.square().sum().item()
            n += err.numel()
    return float(errors / n)


def train_one(exp, args, point_id, root):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from .models.neural import build_model, default_config

    point_idx = exp.point_ids.index(point_id)
    tr = PointTransform(exp, point_idx, args.feature_set, args.target_mode,
                        args.climatology_radius)
    split = exp.split
    train_origins, val_origins = split.origins("train"), split.origins("val")
    x_train, y_train = tr.windows(train_origins, args.in_len, args.out_len)
    x_val, y_val = tr.windows(val_origins, args.in_len, args.out_len)
    if (not np.isfinite(x_train).all() or not np.isfinite(y_train).all() or
            not np.isfinite(x_val).all() or not np.isfinite(y_val).all()):
        raise ValueError(f"Nonfinite train/validation windows in {point_id}")
    set_seed(args.seed + point_idx)
    device = torch_device(args.device)
    config = default_config(args.model, hidden_size=args.hidden_size, dropout=args.dropout)
    model = build_model(args.model, args.in_len, args.out_len,
                        config, input_channels=x_train.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=args.lr_patience,
        factor=args.lr_factor, min_lr=args.min_lr)
        if args.scheduler == "plateau" else None)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                        batch_size=args.batch_size, shuffle=True, num_workers=0)
    dest = root / args.model / point_id
    dest.mkdir(parents=True, exist_ok=True)
    existing = [name for name in ("best.pt", "last.pt", "train_metadata.json") if (dest/name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"{dest} has existing {existing}; specify another --run-id or --overwrite")
    best_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []
    saved = {
        "format_version": 1, "model": args.model, "point_id": point_id,
        "feature_set": args.feature_set, "target_mode": args.target_mode,
        "features_dir": str(exp.features.path), "flux_dataset_dir": str(exp.flux.path),
        "features_sha256": exp.hash,
        "feature_names": list(tr.feature_names),
        "n_input_channels": int(x_train.shape[2]), "input_length": args.in_len,
        "horizon": args.out_len, "train_end": split.train_end,
        "val_start": split.val_start, "val_end": split.val_end,
        "test_start": split.test_start, "test_end": split.test_end,
        "train_last_date": str(exp.features.dates[split.train_end-1]),
        "val_first_date": str(exp.features.dates[split.val_start]),
        "test_first_date": str(exp.features.dates[split.test_start]),
        "val_days": args.val_days, "test_days": args.test_days,
        "model_config": config, "transform": tr.metadata(),
        "optimizer": {"name": "AdamW", "learning_rate": args.learning_rate,
                       "weight_decay": args.weight_decay,
                       "batch_size": args.batch_size, "scheduler": args.scheduler,
                       "patience": args.patience, "min_delta": args.min_delta,
                       "grad_clip": args.grad_clip, "seed": args.seed + point_idx,
                       "amp_bf16": bool(args.amp)},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_metric": "validation_normalized_mse",
        "point_order": list(exp.point_ids),
    }
    for epoch in range(1, args.epochs+1):
        model.train()
        epoch_error, epoch_count = 0., 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=args.amp and device.type == "cuda"):
                pred = model(xb)
                loss = (pred.float() - yb).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss: point={point_id}, epoch={epoch}")
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            epoch_error += float(loss.detach().item()) * len(xb)
            epoch_count += len(xb)
        train_mse = epoch_error / epoch_count
        val_mse = _batched_score(model, x_val, y_val, device, args.batch_size, args.amp)
        improved = (val_mse < best_loss)
        meaningful = val_mse < (best_loss - args.min_delta)
        if improved:
            best_loss, best_epoch = val_mse, epoch
            temp = dest / "best.pt.tmp"
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "validation_mse": val_mse}, temp)
            temp.replace(dest / "best.pt")
        if meaningful:
            bad_epochs = 0
        else:
            bad_epochs += 1
        if scheduler:
            scheduler.step(val_mse)
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_mse": train_mse,
                        "val_mse": val_mse, "lr": lr})
        print(f"[train] {point_id} {args.model} [{args.feature_set}/{args.target_mode}] "
              f"epoch {epoch:03d} train={train_mse:.6f} val={val_mse:.6f} "
              f"best={best_loss:.6f} lr={lr:.2e}", flush=True)
        if bad_epochs >= args.patience:
            print(f"[early-stop] {point_id} epoch={epoch} best_epoch={best_epoch}", flush=True)
            break
    torch.save({"state_dict": model.state_dict(), "epoch": epoch}, dest / "last.pt")
    saved["best_epoch"] = best_epoch
    saved["best_validation_mse"] = best_loss
    saved["epochs_completed"] = epoch
    save_json(dest / "train_metadata.json", saved)
    with (dest / "history.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=("epoch", "train_mse", "val_mse", "lr"))
        writer.writeheader()
        writer.writerows(history)
    print(f"[saved] {dest}/best.pt, last.pt, history.csv, train_metadata.json", flush=True)


def run(args):
    if min(args.in_len, args.out_len, args.epochs, args.batch_size, args.patience) <= 0:
        raise ValueError("Window sizes, epochs, batch size and patience must be positive")
    if not 0 < args.lr_factor < 1:
        raise ValueError("--lr-factor must lie in (0,1)")
    exp = FeatureExperiment(features_directory=args.features_dir,
                            flux_directory=args.dataset_dir,
                            in_len=args.in_len, out_len=args.out_len,
                            val_days=args.val_days, test_days=args.test_days)
    run_id = args.run_id or f"seed{args.seed}"
    root = base_folder(results_override=args.results_dir, target_mode=args.target_mode,
                       feature_set=args.feature_set, run_id=run_id, out_len=args.out_len)
    points = select_points(exp.point_ids, args.points)
    print(f"[setup] features={exp.features.path} shape={exp.features.features.shape}"
          f" dates={exp.features.dates[0]}..{exp.features.dates[-1]}", flush=True)
    print(f"[setup] set={args.feature_set}, names={exp.selected_names(args.feature_set)}, "
          f"train_end={exp.split.train_end}, val={exp.split.val_start}:{exp.split.val_end}, "
          f"test={exp.split.test_start}:{exp.split.test_end}", flush=True)
    print(f"[setup] models={args.model}, points={points}, output={root}", flush=True)
    for point_id in points:
        train_one(exp, args, point_id, root)
    return root


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
