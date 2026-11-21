"""Plot observed series and rolling forecasts for selected points and models."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..data import CHANNELS, PointDataset, dataset_dir, results_dir
from ..experiment import DEFAULT_HORIZON, DEFAULT_TARGET_MODE, TARGET_MODES, target_root, LEGACY_POINTS, select_points
from ..test import DEFAULT_POINTS, ALL_POINTS, point_key, evaluation_dir
from ..registry import ALL_MODELS, LEVEL0


def build_parser():
    parser = argparse.ArgumentParser(description="Plot 1D forecast results")
    parser.add_argument(
        "--model",
        choices=ALL_MODELS,
        required=True
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--points", nargs="+", default=None)
    parser.add_argument("--point", default="all")
    parser.add_argument("--out-len", type=int, default=DEFAULT_HORIZON,
                        help="Trained forecast length used to locate artifacts")
    parser.add_argument("--horizon", type=int, nargs="+", default=None,
                        help="Leads to plot; defaults to 1, 3, 7 and final lead (when available)")
    parser.add_argument("--all-horizons", action="store_true", help="Plot every lead")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--target-mode", choices=TARGET_MODES, default=DEFAULT_TARGET_MODE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--show-baselines", action="store_true")
    parser.add_argument("--compare", nargs="+", choices=ALL_MODELS, default=None,
                        help="Explicit models to overlay instead of default baselines")
    parser.add_argument("--dpi", type=int, default=160)
    return parser


def _load_predictions(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run Forecast_1d.test first.")
    with np.load(path, allow_pickle=False) as archive:
        point_ids = (
            tuple(str(x) for x in archive["point_ids"].tolist())
            if "point_ids" in archive.files else LEGACY_POINTS
        )
        return (
            np.asarray(archive["prediction"]),
            np.asarray(archive["target"]),
            np.asarray(archive["origins"], dtype=np.int64),
            point_ids,
        )


def run(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    data = PointDataset(dataset_dir(args.dataset_dir))
    root = target_root(results_dir(args.results_dir), getattr(args, "out_len", 3),
                       getattr(args, "target_mode", "raw"))
    requested_points = getattr(args, "points", None)
    requested_points = select_points(data.point_ids, requested_points)
    key = point_key(requested_points)
    source = evaluation_dir(root, args.split, requested_points)
    prediction, target, origins, saved_points = _load_predictions(
        source / f"{args.model}_predictions.npz"
    )
    if requested_points != saved_points:
        raise ValueError(
            f"Requested points {requested_points} differ from saved points {saved_points}"
        )
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("Predictions and targets must share [N,points,2,horizon]")
    if prediction.shape[0] != len(origins) or prediction.shape[1:3] != (len(saved_points), 2):
        raise ValueError("Invalid point count or origin metadata")
    point_indices = np.asarray([data.point_ids.index(p) for p in saved_points])
    expected = np.stack([
        np.asarray(data.values[origins + h][:, point_indices, :])
        for h in range(1, prediction.shape[-1] + 1)
    ], axis=-1)
    if not np.allclose(target, expected, equal_nan=True, atol=1e-4):
        raise ValueError("Saved evaluation targets differ from the source dataset")

    baselines = {}
    if args.show_baselines or getattr(args, "compare", None):
        for name in (getattr(args, "compare", None) or ("persistence", "input_mean", "sarima", "seasonal_climatology", "seasonal_ar1", "ridge_var")):
            if not (source / f"{name}_predictions.npz").is_file():
                if getattr(args, "compare", None):
                    raise FileNotFoundError(f"Missing comparison model: {name}")
                continue
            if name != args.model:
                baseline, _, baseline_origins, baseline_points = _load_predictions(
                    source / f"{name}_predictions.npz"
                )
                if (not np.array_equal(origins, baseline_origins) or
                        baseline_points != saved_points):
                    raise ValueError(f"Baseline metadata differs for {name}")
                baselines[name] = baseline

    if args.point == "all":
        selected_ids = saved_points
    else:
        if args.point not in saved_points:
            raise ValueError(f"{args.point} is not in saved point selection {saved_points}")
        selected_ids = (args.point,)

    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir else root / "plots" / args.split / args.model / key
    )
    output.mkdir(parents=True, exist_ok=True)
    image_paths = []
    target_days = data.dates[origins]
    if prediction.shape[-1] != getattr(args, "out_len", 3):
        raise ValueError("Saved forecast length differs from --out-len")
    if getattr(args, "all_horizons", False):
        horizons = list(range(1, prediction.shape[-1] + 1))
    else:
        horizons = args.horizon or (list(range(1, prediction.shape[-1] + 1)) if prediction.shape[-1] <= 3 else sorted({h for h in (1, 3, 7, prediction.shape[-1]) if h <= prediction.shape[-1]}))
    for horizon in horizons:
        if not 1 <= horizon <= prediction.shape[-1]:
            raise ValueError(f"Horizon {horizon} unavailable (1..{prediction.shape[-1]})")
        plot_days = (
            data.dates[origins + horizon]
            .astype("datetime64[ms]").astype(object)
        )
        for point_id in selected_ids:
            p = saved_points.index(point_id)
            info = data.points[int(point_id[1:]) - 1]
            fig, axes = plt.subplots(
                2, 1, figsize=(13, 7.5), sharex=True, constrained_layout=True
            )
            for channel, axis in enumerate(axes):
                valid = np.isfinite(target[:, p, channel, horizon - 1])
                axis.plot(
                    np.asarray(plot_days)[valid],
                    target[valid, p, channel, horizon - 1],
                    label="Observed", linewidth=1.25, color="#222222"
                )
                axis.plot(
                    plot_days, prediction[:, p, channel, horizon - 1],
                    label=f"{args.model} forecast", linewidth=1.0,
                    color="#d1495b", alpha=0.87
                )
                for name, baseline in baselines.items():
                    axis.plot(
                        plot_days, baseline[:, p, channel, horizon - 1],
                        label=name.replace("_", " "), linewidth=0.8,
                        linestyle="--", alpha=0.6
                    )
                axis.set_ylabel(f"{CHANNELS[channel]} flux")
                axis.grid(alpha=0.25)
                axis.legend(loc="upper right", ncol=2, fontsize=9)
            axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
            axes[-1].xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(axes[-1].xaxis.get_major_locator())
            )
            axes[-1].set_xlabel("Forecast target date")
            fig.suptitle(
                f"{point_id} | {info['region']} | {args.model} | "
                f"lead {horizon} day{'s' if horizon > 1 else ''} | {args.split}",
                fontsize=13
            )
            filename = output / f"{point_id}_{args.model}_h{horizon}_{args.split}.png"
            fig.savefig(filename, dpi=args.dpi)
            plt.close(fig)
            image_paths.append(filename)
            print(f"[plot] {filename}", flush=True)
    return image_paths


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
