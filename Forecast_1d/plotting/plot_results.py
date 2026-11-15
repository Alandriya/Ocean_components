"""Plot observed test/validation series and a named model's rolling fits.

Each figure contains two vertically stacked series (sensible and latent).
A separate figure is saved for each selected location and lead time.

Examples:
    python -m Forecast_1d.plotting.plot_results --model sarima
    python -m Forecast_1d.plotting.plot_results --model sarima --point P2 --horizon 1 3
    python -m Forecast_1d.plotting.plot_results --model input_mean --show-baselines
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..data import CHANNELS, PointDataset, dataset_dir, results_dir


def build_parser():
    parser = argparse.ArgumentParser(description="Plot observed time series and model forecasts")
    parser.add_argument("--model", choices=["sarima", "persistence", "input_mean"], required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--point", choices=["all"] + [f"P{i}" for i in range(1, 11)], default="all")
    parser.add_argument("--horizon", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--show-baselines", action="store_true")
    parser.add_argument("--dpi", type=int, default=160)
    return parser


def _load_predictions(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run python -m Forecast_1d.test first")
    with np.load(path, allow_pickle=False) as archive:
        return (np.asarray(archive["prediction"]),
                np.asarray(archive["target"]),
                np.asarray(archive["origins"], dtype=np.int64))


def run(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    data = PointDataset(dataset_dir(args.dataset_dir))
    root = results_dir(args.results_dir)
    source = root / "evaluations" / args.split
    prediction, target, origins = _load_predictions(source / f"{args.model}_predictions.npz")
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("Predictions and targets must have shape [N,10,2,horizon]")
    if prediction.shape[0] != len(origins) or prediction.shape[1:3] != (10, 2):
        raise ValueError("Invalid point count or origin metadata")
    if not np.allclose(target, np.stack([data.values[origins + h]
                          for h in range(1, prediction.shape[-1]+1)], axis=-1),
                          equal_nan=True, atol=1e-4):
        raise ValueError("Saved evaluation target differs from the source dataset")
    baselines = {}
    if args.show_baselines:
        for name in ("persistence", "input_mean"):
            if name != args.model:
                baseline, _, baseline_origins = _load_predictions(source / f"{name}_predictions.npz")
                if not np.array_equal(origins, baseline_origins):
                    raise ValueError(f"Baseline origins differ from {args.model}: {name}")
                baselines[name] = baseline
    points = (range(10) if args.point == "all" else [int(args.point[1:]) - 1])
    output = (Path(args.output_dir).expanduser().resolve() if args.output_dir
              else root / "plots" / args.split / args.model)
    output.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for h in args.horizon:
        if not 1 <= h <= prediction.shape[-1]:
            raise ValueError(f"Horizon {h} unavailable (1..{prediction.shape[-1]})")
        target_days = (data.dates[origins + h].astype("datetime64[ms]")
                       .astype("datetime64[ms]").astype(object))
        for p in points:
            point_id = f"P{p+1}"
            info = data.points[p]
            fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True, constrained_layout=True)
            for channel, axis in enumerate(axes):
                valid = np.isfinite(target[:, p, channel, h-1])
                axis.plot(np.asarray(target_days)[valid], target[valid, p, channel, h-1],
                          label="Observed", linewidth=1.25, color="#222222")
                axis.plot(target_days, prediction[:, p, channel, h-1],
                          label=f"{args.model} forecast", linewidth=1.0, color="#d1495b", alpha=0.87)
                for name, baseline in baselines.items():
                    axis.plot(target_days, baseline[:, p, channel, h-1],
                              label=name.replace("_", " "), linewidth=0.8, linestyle="--", alpha=0.6)
                axis.set_ylabel(f"{CHANNELS[channel]} flux")
                axis.grid(alpha=0.25)
                axis.legend(loc="upper right", ncol=2, fontsize=9)
            axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
            axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(
                axes[-1].xaxis.get_major_locator()))
            axes[-1].set_xlabel("Forecast target date")
            fig.suptitle(f"{point_id} | {info['region']} | {args.model} | "
                         f"lead {h} day{'s' if h > 1 else ''} | {args.split}", fontsize=13)
            filename = output / f"{point_id}_{args.model}_h{h}_{args.split}.png"
            fig.savefig(filename, dpi=args.dpi)
            plt.close(fig)
            image_paths.append(filename)
            print(f"[plot] {filename}", flush=True)
    return image_paths


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
