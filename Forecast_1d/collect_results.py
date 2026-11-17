"""Collect Forecast_1d metrics into JSON and CSV using only Python's standard library.

This module intentionally does not import NumPy, PyTorch, pandas, or Matplotlib.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DEFAULT_POINTS = ("P3", "P5", "P7", "P9")
ALL_POINTS = tuple(f"P{i}" for i in range(1, 11))
CSV_FIELDS = [
    "level", "model", "split", "point_id", "channel", "horizon_days",
    "n", "rmse", "mae", "bias", "mse", "persistence_mse_skill",
]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Collect Forecast_1d metrics without third-party dependencies"
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--points", nargs="+", choices=ALL_POINTS,
                        default=list(DEFAULT_POINTS))
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def _results_root(override):
    if override is not None:
        return Path(override).expanduser().resolve()
    from project_paths import DATA_ROOT
    return Path(DATA_ROOT).expanduser().resolve() / "Forecast" / "Results" / "Forecast_1d"


def _level_for(model):
    if model in ("persistence", "input_mean", "sarima"):
        return 0
    if model in ("mlp", "tcn", "gru"):
        return 1
    return None


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
    root = _results_root(args.results_dir)
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

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir else root / "summary"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"forecast_1d_{args.split}_{point_key}"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    result_payload = {
        "split": args.split,
        "point_ids": list(points),
        "source_metrics": str(metrics_path),
        "models": sorted(payload.get("models", {}).keys()),
        "rows": rows,
    }
    json_path.write_text(
        json.dumps(result_payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8"
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {json_path}", flush=True)
    print(f"[saved] {csv_path}", flush=True)
    print(f"[summary] models={','.join(result_payload['models'])} rows={len(rows)}", flush=True)
    return json_path, csv_path


def main(argv=None):
    return collect(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
