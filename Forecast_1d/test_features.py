"""Evaluate multivariate-history point forecasts on unchanged original calendar splits.

Only targets sensible/latent enter future scoring; future ERA5 data never enter
model inputs. Also evaluates persistence and train-only Seasonal AR(1) on EXACTLY
identical origins/targets (same sample mask) for meaningful forecast skill.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .experiment import select_points
from .feature_experiment import (FEATURE_SETS, MODEL_NAMES, TARGET_MODES,
                                  FeatureExperiment, PointTransform, base_folder,
                                  save_json, torch_device)
from .metrics import score_predictions
from Forecast_1d.models.baselines import fit_seasonal, forecast_seasonal


def build_parser():
    """Construct the command-line argument parser."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=MODEL_NAMES, default="gru")
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--points", nargs="+", default=None)
    p.add_argument("--feature-set", choices=FEATURE_SETS, default="all")
    p.add_argument("--target-mode", choices=TARGET_MODES, default="anomaly")
    p.add_argument("--features-dir", type=Path, default=None)
    p.add_argument("--dataset-dir", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--in-len", type=int, default=60)
    p.add_argument("--out-len", type=int, default=14)
    p.add_argument("--val-days", type=int, default=730)
    p.add_argument("--test-days", type=int, default=365)
    p.add_argument("--climatology-radius", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


def _load_forecast(exp, args, point_id, origins, model_folder, device):
    """Load forecast for the test features workflow."""
    import torch
    from .models.neural import build_model

    point_idx = exp.point_ids.index(point_id)
    folder = model_folder / point_id
    path = folder / "best.pt"
    meta_file = folder / "train_metadata.json"
    if not path.is_file() or not meta_file.is_file():
        raise FileNotFoundError(f"Missing best.pt or train_metadata.json in {folder}; train it first")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    expected = {
        "format_version": 1, "model": args.model, "point_id": point_id,
        "feature_set": args.feature_set, "target_mode": args.target_mode,
        "features_sha256": exp.hash, "input_length": args.in_len,
        "horizon": args.out_len, "train_end": exp.split.train_end,
        "val_start": exp.split.val_start, "val_end": exp.split.val_end,
        "test_start": exp.split.test_start, "test_end": exp.split.test_end,
        "val_days": args.val_days, "test_days": args.test_days,
        "point_order": list(exp.point_ids),
        "feature_names": list(exp.selected_names(args.feature_set)),
        "n_input_channels": len(exp.selected_names(args.feature_set)),
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"Checkpoint mismatch {folder}: {key}={meta.get(key)!r}, expected {value!r}")
    tr = PointTransform(exp, point_idx, args.feature_set,
                        args.target_mode, args.climatology_radius)
    tr.validate_metadata(meta["transform"])
    x, _ = tr.windows(origins, args.in_len, args.out_len)
    model = build_model(args.model, args.in_len, args.out_len,
                        meta["model_config"], input_channels=x.shape[2]).to(device)
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    pieces = []
    with torch.inference_mode():
        for begin in range(0, len(x), args.batch_size):
            batch = torch.from_numpy(x[begin:begin+args.batch_size]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=args.amp and device.type == "cuda"):
                y = model(batch)
            pieces.append(y.float().cpu().numpy())
    predicted = tr.unscale(np.concatenate(pieces, axis=0), origins)
    return predicted.transpose(0, 2, 1)  # [origins, 2, horizon]


def _aggregate_rmses(pred, target, persistence, sar):
    """Compute paired, non-averaged MSE skill on a common observation support."""
    good = (np.isfinite(pred) & np.isfinite(target) &
            np.isfinite(persistence) & np.isfinite(sar))
    if not good.all():
        raise ValueError("Nonfinite forecast or baseline; cannot compare unequal supports")
    mse = {name: float(np.mean((arr - target)**2))
           for name, arr in (("model", pred), ("persistence", persistence), ("sar", sar))}
    if mse["sar"] <= 0 or mse["persistence"] <= 0:
        raise ValueError("Zero MSE baseline prevents skill calculation")
    return {
        "rmse_model": float(np.sqrt(mse["model"])),
        "rmse_persistence": float(np.sqrt(mse["persistence"])),
        "rmse_sar": float(np.sqrt(mse["sar"])),
        "mse_skill_vs_persistence": 1.0 - mse["model"]/mse["persistence"],
        "mse_skill_vs_seasonal_ar1": 1.0 - mse["model"]/mse["sar"],
    }


def _rename_sar_skill(obj):
    """score_predictions calls its reference 'persistence' generically; relabel SAR scores."""
    if isinstance(obj, dict):
        return {("seasonal_ar1_mse_skill" if key == "persistence_mse_skill" else key):
                _rename_sar_skill(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_rename_sar_skill(value) for value in obj]
    return obj


def run(args):
    """Run the test features workflow using its configured options."""
    if args.batch_size < 1:
        raise ValueError("batch_size must be >=1")
    exp = FeatureExperiment(features_directory=args.features_dir,
                            flux_directory=args.dataset_dir,
                            in_len=args.in_len, out_len=args.out_len,
                            val_days=args.val_days, test_days=args.test_days)
    ids = select_points(exp.point_ids, args.points)
    point_indices = np.asarray([exp.point_ids.index(p) for p in ids])
    origins = exp.split.origins(args.split)
    if not len(origins):
        raise ValueError("Empty forecast-origin set")
    root = base_folder(results_override=args.results_dir, target_mode=args.target_mode,
                       feature_set=args.feature_set, run_id=args.run_id or f"seed{args.seed}",
                       out_len=args.out_len)
    model_folder = root / args.model
    dest = root / "evaluations" / args.model / args.split / "_".join(ids)
    if (dest / "metrics.json").exists() and not args.overwrite:
        raise FileExistsError(f"{dest}/metrics.json already exists; specify --overwrite")
    device = torch_device(args.device)
    forecasts = []
    print(f"[setup] data={exp.features.path}, split={args.split}, "
          f"origins={len(origins)}, dates {exp.features.dates[origins[0]+1]}"
          f"..{exp.features.dates[origins[-1]+args.out_len]}", flush=True)
    for point_id in ids:
        pred = _load_forecast(exp, args, point_id, origins, model_folder, device)
        forecasts.append(pred)
        print(f"[predict] {point_id} shape={pred.shape}", flush=True)
    model = np.stack(forecasts, axis=1)  # [origin, point, channel, horizon]
    observed = np.asarray(exp.features.targets, dtype=np.float64)
    target = np.stack([observed[origins+h][:, point_indices, :]
                       for h in range(1, args.out_len+1)], axis=-1)
    persistence = np.repeat(observed[origins][:, point_indices, :, None],
                            args.out_len, axis=-1)
    # This baseline sees ONLY historical data before the first validation target.
    climate, phi = fit_seasonal(observed, exp.features.dates,
                                exp.split.train_end, radius=args.climatology_radius)
    sar_full = forecast_seasonal(observed, exp.features.dates,
                                 origins, args.out_len, climate, phi)
    sar = sar_full[:, point_indices, :, :]
    if any(arr.shape != target.shape for arr in (model, persistence, sar)):
        raise AssertionError("Predictions/baselines use different target shapes")
    scores = _aggregate_rmses(model, target, persistence, sar)
    per_lead = {str(h+1): _aggregate_rmses(model[..., h], target[..., h],
                                          persistence[..., h], sar[..., h])
                for h in range(args.out_len)}
    detailed = score_predictions(model, target, baseline=persistence, point_ids=ids)
    detailed_sar = _rename_sar_skill(score_predictions(model, target, baseline=sar, point_ids=ids))
    result = {
        "model": args.model, "feature_set": args.feature_set,
        "feature_names": list(exp.selected_names(args.feature_set)),
        "target_mode": args.target_mode, "split": args.split,
        "points": list(ids), "n_origins": int(len(origins)),
        "n_features": len(exp.selected_names(args.feature_set)),
        "input_length": args.in_len, "horizon": args.out_len,
        "first_forecast_date": str(exp.features.dates[origins[0]+1]),
        "last_forecast_date": str(exp.features.dates[origins[-1]+args.out_len]),
        "feature_dataset_fingerprint": exp.hash,
        "train_last_date": str(exp.features.dates[exp.split.train_end-1]),
        "summary": scores, "per_lead": per_lead,
        "by_point_channel_horizon_vs_persistence": detailed,
        "by_point_channel_horizon_vs_seasonal_ar1": detailed_sar,
        "seasonal_ar1_climatology_radius": args.climatology_radius,
        "comparison_support": "Identical target dates/origins for model, SAR(1), persistence; full horizons only",
    }
    dest.mkdir(parents=True, exist_ok=True)
    save_json(dest / "metrics.json", result)
    np.savez_compressed(dest / "predictions.npz", prediction=model.astype(np.float32),
                        target=target.astype(np.float32),
                        persistence=persistence.astype(np.float32),
                        seasonal_ar1=sar.astype(np.float32), origins=origins,
                        dates=exp.features.dates[origins+1])
    print(f"[result] RMSE model={scores['rmse_model']:.4f}, "
          f"SAR={scores['rmse_sar']:.4f}, persistence={scores['rmse_persistence']:.4f}, "
          f"skill_vs_SAR={scores['mse_skill_vs_seasonal_ar1']:+.3%}", flush=True)
    print(f"[saved] {dest / 'metrics.json'}", flush=True)
    return result


def main(argv=None):
    """Run the test features workflow using its configured options."""
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
