"""Collect Forecast_1d metrics into JSON and CSV using only Python's standard library.

This module intentionally does not import NumPy, PyTorch, pandas, or Matplotlib.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from .experiment import DEFAULT_HORIZON, horizon_root
from .registry import model_level, ALL_MODELS

DEFAULT_POINTS = ("P3", "P5", "P7", "P9")
ALL_POINTS = tuple(f"P{i}" for i in range(1, 11))
CSV_FIELDS = [
    "level", "model", "split", "point_id", "channel", "horizon_days",
    "n", "rmse", "mae", "bias", "mse", "persistence_mse_skill", "sarima_mse_skill", "seasonal_climatology_mse_skill", "seasonal_ar1_mse_skill", "ridge_var_mse_skill",
]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Collect Forecast_1d metrics without third-party dependencies"
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--points", nargs="+", choices=ALL_POINTS,
                        default=list(DEFAULT_POINTS))
    parser.add_argument("--out-len", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def _results_root(override):
    if override is not None:
        return Path(override).expanduser().resolve()
    from project_paths import DATA_ROOT
    return Path(DATA_ROOT).expanduser().resolve() / "Forecast" / "Results" / "Forecast_1d"


def _level_for(model):
    return model_level(model)

def _metric_row(model, split, point_id, channel, horizon, score):
    return {
        "level": _level_for(model),
        "model": model,
        "split": split,
        "point_id": point_id,
        "channel": channel,
        "horizon_days": int(horizon),
        "n": score.get("n"),
        "rmse": score.get("rmse"),
        "mae": score.get("mae"),
        "bias": score.get("bias"),
        "mse": score.get("mse"),
        "persistence_mse_skill": score.get("persistence_mse_skill"),
    }


def collect(args):
    points = tuple(args.points)
    point_key = "_".join(points)
    root = horizon_root(_results_root(args.results_dir), args.out_len)
    evaluation_root = root / "evaluations" / args.split
    metrics_path = (
        evaluation_root / "metrics.json"
        if points == ALL_POINTS else evaluation_root / point_key / "metrics.json"
    )
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"Metrics file not found: {metrics_path}. Run Forecast_1d.test first."
        )
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if tuple(payload.get("point_ids", ())) != points:
        raise ValueError("Requested point selection differs from metrics.json")

    if payload.get("out_len") != args.out_len:
        raise ValueError("Metrics forecast length does not match --out-len")
    rows = []
    for model, horizons in payload.get("models", {}).items():
        for horizon, result in horizons.items():
            rows.append(_metric_row(
                model, args.split, "ALL", "joint", horizon, result["joint"]
            ))
            for channel, score in result.get("channels", {}).items():
                rows.append(_metric_row(
                    model, args.split, "ALL", channel, horizon, score
                ))
            for point_id, channel_scores in result.get("points", {}).items():
                for channel, score in channel_scores.items():
                    rows.append(_metric_row(
                        model, args.split, point_id, channel, horizon, score
                    ))

    # References use identical evaluation cells, not just aggregated joint scores.
    for reference_name in ("sarima", "seasonal_climatology", "seasonal_ar1", "ridge_var"):
        reference = {(r["point_id"], r["channel"], r["horizon_days"]): r
                     for r in rows if r["model"] == reference_name}
        for row in rows:
            ref = reference.get((row["point_id"], row["channel"], row["horizon_days"]))
            row[reference_name + "_mse_skill"] = (1 - row["mse"] / ref["mse"]
                if ref and ref["n"] == row["n"] and ref["mse"] not in (None, 0) and row["mse"] is not None
                else None)
    ranking = []
    expected_cells = len(points)*2*args.out_len
    for model in payload.get("models", {}):
        joint = [r for r in rows if r["model"] == model and r["point_id"] == "ALL" and r["channel"] == "joint"]
        local = [r for r in rows if r["model"] == model and r["point_id"] in points]
        n = sum(r["n"] for r in joint)
        if not n or len(joint) != args.out_len or len(local) != expected_cells:
            raise ValueError(f"Incomplete metrics for {model}")
        item = {"level": model_level(model), "model": model, "split": args.split,
                "n": n, "rmse": math.sqrt(sum(r["n"]*r["mse"] for r in joint if r["n"])/n),
                "mae": sum(r["n"]*r["mae"] for r in joint if r["n"])/n,
                "bias": sum(r["n"]*r["bias"] for r in joint if r["n"])/n}
        for ref_name in ("persistence", "sarima", "seasonal_climatology", "seasonal_ar1", "ridge_var"):
            scores = [r.get(ref_name+"_mse_skill") for r in local]
            item["macro_"+ref_name+"_mse_skill"] = (sum(scores)/expected_cells if all(v is not None for v in scores) else None)
        ranking.append(item)
    ranking.sort(key=lambda r: r["rmse"])
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir else root / "summary"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"forecast_1d_{args.split}_{point_key}"
    if args.out_len != 3:
        stem += f"_h{args.out_len}"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    ranking_path = output_dir / f"{stem}_ranking.csv"
    result_payload = {
        "split": args.split,
        "out_len": args.out_len,
        "point_ids": list(points),
        "source_metrics": str(metrics_path),
        "models": sorted(payload.get("models", {}).keys()),
        "rows": rows,
        "ranking_by_rmse": ranking,
        "missing_models": sorted(set(ALL_MODELS) - set(payload.get("models", {}))),
    }
    json_path.write_text(
        json.dumps(result_payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8"
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    if ranking:
        with ranking_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ranking[0]))
            writer.writeheader()
            writer.writerows(ranking)
        print(f"[saved] {ranking_path}", flush=True)
    print(f"[saved] {json_path}", flush=True)
    print(f"[saved] {csv_path}", flush=True)
    print(f"[summary] models={','.join(result_payload['models'])} rows={len(rows)}", flush=True)
    return json_path, csv_path


def main(argv=None):
    return collect(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
