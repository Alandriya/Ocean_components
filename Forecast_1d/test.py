"""Evaluate level-0 baselines/SARIMA and level-1 point-specific neural models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import PointDataset, dataset_dir, results_dir
from .metrics import save_json, score_predictions
from .experiment import DEFAULT_HORIZON, horizon_root, DEFAULT_POINTS, LEGACY_POINTS, select_points
from .models.baselines import input_mean, persistence
from .models.sarima import forecast_ar1, forecast_sarimax

ALL_POINTS = DEFAULT_POINTS
from .registry import BASELINES, NEW_BASELINES, LEVEL0, LEVEL1, LEVEL2, NEURAL_MODELS, ALL_MODELS
from .models.seasonal import fit_seasonal, forecast_seasonal
from .models.ridge_var import fit_ridge, forecast_ridge


def point_key(point_ids):
    return "_".join(point_ids)


def evaluation_dir(root, split, point_ids):
    """Keep the legacy flat directory for the original all-ten-point evaluation."""
    base = root / "evaluations" / split
    return base if tuple(point_ids) == LEGACY_POINTS else base / point_key(point_ids)


def build_parser():
    p = argparse.ArgumentParser(description="Evaluate Forecast_1d models")
    p.add_argument(
        "--model",
        choices=[*ALL_MODELS, "level0", "level0_new", "level1", "level2", "level01", "all"],
        default="all"
    )
    p.add_argument("--points", nargs="+", default=None)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--dataset-dir", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--in-len", type=int, default=60)
    p.add_argument("--mean-days", type=int, default=30,
                   help="Input-mean baseline window; independent of neural input length.")
    p.add_argument("--out-len", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--val-days", type=int, default=730)
    p.add_argument("--test-days", type=int, default=365)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--overwrite", action="store_true")
    return p


def _selected_models(model):
    if model == "level0":
        return list(LEVEL0)
    if model == "level1":
        return [*BASELINES, *LEVEL1]
    if model == "level0_new":
        return [*BASELINES, *NEW_BASELINES]
    if model == "level2":
        return [*BASELINES, *LEVEL2]
    if model == "level01":
        return [*BASELINES, "sarima", *LEVEL1]
    if model == "all":
        return list(ALL_MODELS)
    if model in BASELINES:
        return list(BASELINES)
    return [*BASELINES, model]


def _load_trained_sarima(model_root, data, split):
    meta_path = model_root / "sarima" / "train_metadata.json"
    param_path = model_root / "sarima" / "params.npz"
    if not meta_path.is_file() or not param_path.is_file():
        raise FileNotFoundError(
            "SARIMA checkpoint missing; run python -m Forecast_1d.train --model sarima"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (meta["train_end"] != split.train_end or
            meta["val_days"] != split.val_end - split.val_start or
            meta["test_days"] != split.test_end - split.test_start or
            meta["out_len"] != split.out_len):
        raise ValueError("Evaluation split differs from SARIMA training checkpoint")
    if meta["dataset_fingerprint_sha256"] != data.fingerprint():
        raise ValueError("Dataset fingerprint differs from SARIMA checkpoint")
    with np.load(param_path, allow_pickle=False) as source:
        params = np.asarray(source["params"], dtype=np.float64)
    return meta, params


def _resolve_device(torch, requested):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _forecast_neural(model_name, point_id, point_index, data, origins,
                     model_root, split, args, device):
    import torch
    from .models.neural import build_model

    model_dir = model_root / model_name / point_id
    meta_path = model_dir / "train_metadata.json"
    checkpoint_path = model_dir / "best.pt"
    if not meta_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing {model_name}/{point_id} checkpoint; train this point first."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = {
        "point_id": point_id,
        "train_end": split.train_end,
        "val_days": args.val_days,
        "test_days": args.test_days,
        "input_length": args.in_len,
        "horizon": args.out_len,
        "dataset_fingerprint_sha256": data.fingerprint(),
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(
                f"Checkpoint mismatch for {model_name}/{point_id}: "
                f"{key}={meta.get(key)!r}, expected {value!r}"
            )
    normalization = meta["normalization"]
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    scale = np.asarray(normalization["scale"], dtype=np.float32)
    in_len = args.in_len
    windows = []
    for origin in origins:
        x = np.asarray(
            data.values[origin - in_len + 1:origin + 1, point_index, :],
            dtype=np.float32
        )
        if x.shape != (in_len, 2):
            raise ValueError(f"Insufficient input history at origin {origin}")
        x = np.where(np.isfinite(x), x, mean[None, :])
        windows.append((x - mean) / scale)
    x_array = np.stack(windows).astype(np.float32)

    model = build_model(
        model_name, in_len, args.out_len, meta["model_config"],
        input_channels=int(meta.get("input_channels", 2))
    ).to(device)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    batches = []
    with torch.no_grad():
        for start in range(0, len(x_array), args.batch_size):
            xb = torch.from_numpy(x_array[start:start + args.batch_size]).to(device)
            batches.append(model(xb).cpu().numpy())
    normalized = np.concatenate(batches, axis=0)
    prediction = normalized * scale[None, None, :] + mean[None, None, :]
    return prediction.transpose(0, 2, 1).astype(np.float64)


def run(args):
    data = PointDataset(dataset_dir(args.dataset_dir))
    requested_points = getattr(args, "points", None)
    point_ids = select_points(data.point_ids, requested_points)
    point_indices = np.asarray([data.point_ids.index(p) for p in point_ids], dtype=np.int64)
    in_len = getattr(args, "in_len", 60)
    mean_days = getattr(args, "mean_days", getattr(args, "in_len", 30))
    split = data.get_splits(
        args.val_days, args.test_days, in_len, args.out_len
    )
    if mean_days < 1:
        raise ValueError("mean-days must be positive")
    origins = split.origins(args.split)
    origins = origins[origins >= mean_days - 1]
    if origins.size == 0:
        raise ValueError("No valid forecast origins")
    model_root = horizon_root(results_dir(args.results_dir), args.out_len)
    names = _selected_models(args.model)
    key = point_key(point_ids)
    dest = evaluation_dir(model_root, args.split, point_ids)
    dest.mkdir(parents=True, exist_ok=True)

    target = np.stack([
        np.asarray(data.values[origins + h][:, point_indices, :], dtype=np.float64)
        for h in range(1, split.out_len + 1)
    ], axis=-1)

    full_persistence = persistence(data.values, origins, split.out_len)
    full_mean = input_mean(data.values, origins, split.out_len, mean_days)
    forecasts = {
        "persistence": full_persistence[:, point_indices, :, :],
        "input_mean": full_mean[:, point_indices, :, :],
    }

    if "sarima" in names:
        meta, params = _load_trained_sarima(model_root, data, split)
        if meta["backend"] == "ar1_ols":
            full = forecast_ar1(data.values, origins, split.out_len, params)
        elif meta["backend"] == "statsmodels":
            full = forecast_sarimax(
                data.values, origins, split.out_len, params,
                meta["fit_start"], meta["fit_end"], meta["order"],
                meta["seasonal_order"], meta["trend"]
            )
        else:
            raise ValueError(f"Unknown SARIMA backend: {meta['backend']}")
        forecasts["sarima"] = full[:, point_indices, :, :]

    # Fixed baseline configurations; fit exclusively on the train interval.
    baseline_config = {}
    selected_values = data.values[:, point_indices, :]
    if any(name in names for name in ("seasonal_climatology", "seasonal_ar1")):
        clim, phi = fit_seasonal(selected_values, data.dates, split.train_end, radius=15)
        if "seasonal_climatology" in names:
            forecasts["seasonal_climatology"] = forecast_seasonal(selected_values, data.dates, origins, split.out_len, clim)
            baseline_config["seasonal_climatology"] = {"radius_days": 15, "calendar": "366 month-day slots", "fit_end_exclusive": split.train_end}
        if "seasonal_ar1" in names:
            forecasts["seasonal_ar1"] = forecast_seasonal(selected_values, data.dates, origins, split.out_len, clim, phi)
            baseline_config["seasonal_ar1"] = {"radius_days": 15, "phi": phi.tolist(), "fit_end_exclusive": split.train_end, "intercept": 0, "phi_clip": 0.999}
    if "ridge_var" in names:
        fitted = fit_ridge(selected_values, split.train_end, in_len, split.out_len, alpha=0.01)
        forecasts["ridge_var"] = forecast_ridge(selected_values, origins, in_len, split.out_len, fitted)
        baseline_config["ridge_var"] = {"alpha": 0.01, "input_length": in_len, "fit_end_exclusive": split.train_end, "direct_multioutput": True}
    neural_names = [name for name in names if name in NEURAL_MODELS]
    if neural_names:
        import torch
        device = _resolve_device(torch, getattr(args, "device", "auto"))
        for model_name in neural_names:
            point_predictions = []
            for point_id, point_index in zip(point_ids, point_indices):
                prediction = _forecast_neural(
                    model_name, point_id, int(point_index), data, origins,
                    model_root, split, args, device
                )
                point_predictions.append(prediction)
            forecasts[model_name] = np.stack(point_predictions, axis=1)
            print(
                f"[forecast] model={model_name} points={','.join(point_ids)} "
                f"device={device}", flush=True
            )

    # Validate the existing experiment before writing any artifacts.
    summary_path = dest / "metrics.json"
    protocol = {
        "split": args.split,
        "dataset_fingerprint_sha256": data.fingerprint(),
        "point_ids": list(point_ids),
        "first_forecast_date": str(data.dates[int(origins[0]) + 1]),
        "last_forecast_date": str(data.dates[int(origins[-1]) + split.out_len]),
        "train_end": split.train_end,
        "eval_origin_count": int(len(origins)),
        "in_len": in_len, "mean_days": mean_days, "out_len": split.out_len,
        "val_days": args.val_days, "test_days": args.test_days,
    }
    if summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        for field, value in protocol.items():
            if previous.get(field) != value:
                raise ValueError(f"Existing evaluation protocol mismatch: {field}. Use a fresh --results-dir.")
    # Fixed scoring support across separate model evaluations. Never reward a
    # model for failing to predict a difficult, otherwise scorable observation.
    common_mask = np.isfinite(target) & np.isfinite(forecasts["persistence"]) & np.isfinite(forecasts["input_mean"])
    if summary_path.exists() and previous.get("common_finite_target_cells", int(common_mask.sum())) != int(common_mask.sum()):
        raise ValueError("Existing evaluation support differs")
    for name in names:
        if not np.isfinite(forecasts[name][common_mask]).all():
            raise ValueError(f"Non-finite {name} forecast on common evaluation support")
        prediction_path = dest / f"{name}_predictions.npz"
        if prediction_path.exists() and name not in BASELINES and not args.overwrite:
            raise FileExistsError(f"Prediction file exists: {prediction_path}; pass --overwrite")

    for name in names:
        prediction_path = dest / f"{name}_predictions.npz"
        if prediction_path.exists() and not args.overwrite:
            if name in BASELINES:
                with np.load(prediction_path, allow_pickle=False) as archive:
                    old_origins = np.asarray(archive["origins"], dtype=np.int64)
                    old_points = tuple(str(x) for x in archive["point_ids"].tolist())
                    old_prediction = np.asarray(archive["prediction"], dtype=np.float64)
                    old_target = np.asarray(archive["target"], dtype=np.float64)
                if (old_points != point_ids or not np.array_equal(old_origins, origins)
                        or old_target.shape != target.shape
                        or not np.allclose(old_target, target, equal_nan=True, atol=1e-5)
                        or old_prediction.shape != forecasts[name].shape
                        or not np.allclose(old_prediction, forecasts[name], equal_nan=True, atol=1e-5, rtol=1e-6)):
                    raise ValueError(f"Existing baseline file is incompatible: {prediction_path}")
                # Keep full-precision freshly computed baseline for scoring.
            else:
                raise FileExistsError(
                    f"Prediction file exists: {prediction_path}; pass --overwrite"
                )
        else:
            np.savez_compressed(
                prediction_path,
                prediction=forecasts[name].astype(np.float32),
                target=target.astype(np.float32),
                origins=origins,
                dates=data.dates[origins].astype("datetime64[D]"),
                point_ids=np.asarray(point_ids, dtype=str),
            )
            print(f"[saved] {prediction_path}", flush=True)

    summary_path = dest / "metrics.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if tuple(summary.get("point_ids", [])) != point_ids:
            raise ValueError("Existing metrics file has a different point selection")
        if summary.get("dataset_fingerprint_sha256") != data.fingerprint():
            raise ValueError("Existing metrics file belongs to a different dataset")
    else:
        summary = {
            "split": args.split,
            "dataset": str(data.path),
            "dataset_fingerprint_sha256": data.fingerprint(),
            "point_ids": list(point_ids),
            "first_forecast_date": str(data.dates[int(origins[0]) + 1]),
            "last_forecast_date": str(data.dates[int(origins[-1]) + split.out_len]),
            "train_end": split.train_end,
            "eval_origin_count": int(len(origins)),
            "in_len": in_len,
            "mean_days": mean_days,
            "out_len": split.out_len,
            "val_days": args.val_days,
            "test_days": args.test_days,
            "target_domain": "selected offshore point centers, sensible and latent",
            "rolling_origin": "observations up to origin only, fixed fitted parameters",
            "models": {},
        }

    for name in names:
        scores = score_predictions(
            forecasts[name], target, baseline=forecasts["persistence"],
            point_ids=point_ids, shared_mask=common_mask
        )
        summary["models"][name] = scores
        for horizon, result in scores.items():
            print(
                f"[score] model={name} h{horizon} "
                f"joint_rmse={result['joint']['rmse']} "
                f"persistence_skill={result.get('persistence_mse_skill')}",
                flush=True
            )
    summary.setdefault("baseline_config", {}).update(baseline_config)
    summary["common_finite_target_cells"] = int(common_mask.sum())
    summary["updated_utc"] = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
    ).isoformat()
    save_json(summary_path, summary)
    print(f"[saved] {summary_path}", flush=True)
    return dest


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
