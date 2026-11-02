"""Diagnostics for the joint quantile groups used in aligned 2D NP/SP.

No Gaussian mixtures are fitted and no coefficient maps are calculated.
This module calls prepare_pair_transition from aligned_common, so the
quantile grouping is identical to the coefficient estimators.

CSV outputs:
- daily_summary.csv: group occupancy and retained-point coverage by day/bins.
- by_bin_summary.csv: mean and minimum coverage over the inspected days.
- group_counts.csv: size of each possible joint bin (including empty bins).
Plots are optional and require matplotlib.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from Coefficients.aligned_common import coerce_series, load_water_mask, prepare_pair_transition


def _write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        raise ValueError(f"No results to write to {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _plot_occupancy(count_matrix: np.ndarray, output: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 6))
    plot = ax.imshow(np.log1p(count_matrix.T), origin="lower", aspect="equal")
    ax.set_xlabel("Quantile index: first variable")
    ax.set_ylabel("Quantile index: second variable")
    ax.set_title(title)
    bar = fig.colorbar(plot, ax=ax)
    bar.set_label("log(1 + observations in group)")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def diagnose_joint_groups(
    array1,
    array2,
    mask,
    out_dir,
    bins_list: Iterable[int] = (10, 12, 15, 18, 20),
    n_transitions: int = 7,
    time_start: int = 0,
    index_offset: int = 1,
    min_group_size: int = 2,
    thresholds: tuple[int, ...] = (2, 10, 20, 30),
    plot_bins: int = 15,
    make_plots: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Inspect the FIRST n transitions from time_start for each bin count.

    Both inputs can be ndarray in shape (T,H,W), (H*W,T), (T,H*W),
    or paths to .npy files. A mask value of 1 denotes water; 0 denotes land.
    """
    x0, x1 = coerce_series(array1), coerce_series(array2)
    if x0.shape != x1.shape:
        raise ValueError(f"Input series have different shapes: {x0.shape} and {x1.shape}")
    water = load_water_mask(mask, x0.shape[1:])
    n_water = int(water.sum())
    if not n_water:
        raise ValueError("Water mask contains no water cells (mask=1)")
    if n_transitions < 1:
        raise ValueError("n_transitions must be positive")
    if not 0 <= time_start < x0.shape[0] - 1:
        raise ValueError("time_start must leave at least one subsequent observation")
    if time_start + n_transitions >= x0.shape[0]:
        raise ValueError(
            f"Need at least {time_start + n_transitions + 1} time points "
            f"to inspect {n_transitions} transitions; got {x0.shape[0]}"
        )
    n_bins_all = tuple(dict.fromkeys(int(n) for n in bins_list))
    if not n_bins_all or any(n < 1 for n in n_bins_all):
        raise ValueError("bins_list must contain positive integers")
    if min_group_size < 2:
        raise ValueError("min_group_size must be at least 2 for aligned NP and SP")
    limits = tuple(sorted(set(int(n) for n in (*thresholds, min_group_size))))
    if any(n < 1 for n in limits):
        raise ValueError("Group size thresholds must be positive")

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    daily: list[dict] = []
    group_records: list[dict] = []
    running_occupancy = np.zeros((plot_bins, plot_bins), dtype=np.float64)
    plot_days = 0
    for n in n_bins_all:
        for t in range(time_start + 1, time_start + n_transitions + 1):
            grouped = prepare_pair_transition(
                x0[t - 1], x0[t], x1[t - 1], x1[t], n, water
            )
            sizes = np.zeros(n * n, dtype=np.int64)
            for group in grouped.groups:
                sizes[group.bin_id] = group.increments.shape[0]
            valid = int(grouped.n_valid_pairs)
            if sizes.sum() != valid:
                raise AssertionError("Group size accounting does not match valid pairs")
            occupied = sizes[sizes > 0]
            record = {
                "bins_per_axis": n,
                "possible_groups": n * n,
                "transition_t": t,
                "output_day_index": t + index_offset,
                "water_cells": n_water,
                "valid_transition_cells": valid,
                "valid_pct_of_water": round(100 * valid / n_water, 4),
                "occupied_groups": int(occupied.size),
                "empty_groups": int((sizes == 0).sum()),
                "singleton_groups": int((sizes == 1).sum()),
                "group_size_min_occupied": int(occupied.min()) if occupied.size else 0,
                "group_size_p10_occupied": round(float(np.percentile(occupied, 10)), 2) if occupied.size else 0.0,
                "group_size_median_occupied": float(np.median(occupied)) if occupied.size else 0.0,
                "group_size_p90_occupied": round(float(np.percentile(occupied, 90)), 2) if occupied.size else 0.0,
                "group_size_max_occupied": int(occupied.max()) if occupied.size else 0,
            }
            for threshold in limits:
                eligible = sizes >= threshold
                count_eligible = int(np.count_nonzero(eligible))
                kept = int(sizes[eligible].sum())
                record[f"groups_ge_{threshold}"] = count_eligible
                record[f"cells_in_groups_ge_{threshold}"] = kept
                record[f"coverage_valid_pct_ge_{threshold}"] = round(
                    100 * kept / valid, 4) if valid else 0.0
                record[f"coverage_water_pct_ge_{threshold}"] = round(
                    100 * kept / n_water, 4)
            daily.append(record)
            for bin_id, count in enumerate(sizes):
                group_records.append({
                    "bins_per_axis": n,
                    "transition_t": t,
                    "output_day_index": t + index_offset,
                    "bin_first": bin_id // n,
                    "bin_second": bin_id % n,
                    "group_size": int(count),
                })
            print(
                f"bins={n:2d} (up to {n*n:3d} groups), day={t + index_offset:5d}: "
                f"occupied={occupied.size:3d}, singletons={(sizes==1).sum():3d}, "
                f"coverage(n>={min_group_size})="
                f"{record[f'coverage_valid_pct_ge_{min_group_size}']:.2f}%, "
                f"coverage(n>=30)={record.get('coverage_valid_pct_ge_30', 0.0):.2f}%",
                flush=True,
            )
            if n == plot_bins:
                counts = sizes.reshape(n, n)
                running_occupancy += counts
                plot_days += 1
                if make_plots and t == time_start + 1:
                    _plot_occupancy(
                        counts, target / f"occupancy_bins_{plot_bins}_first_day.png",
                        f"Joint groups: {plot_bins} x {plot_bins}, day {t + index_offset}",
                    )
    if make_plots and plot_days:
        _plot_occupancy(
            running_occupancy / plot_days,
            target / f"occupancy_bins_{plot_bins}_mean.png",
            f"Mean occupancy over {plot_days} transitions: {plot_bins} x {plot_bins}",
        )

    by_bin: list[dict] = []
    for n in n_bins_all:
        subset = [item for item in daily if item["bins_per_axis"] == n]
        row = {
            "bins_per_axis": n,
            "possible_groups": n * n,
            "days_tested": len(subset),
            "mean_occupied_groups": round(float(np.mean([r["occupied_groups"] for r in subset])), 2),
            "mean_empty_groups": round(float(np.mean([r["empty_groups"] for r in subset])), 2),
            "mean_singleton_groups": round(float(np.mean([r["singleton_groups"] for r in subset])), 2),
            "min_group_size_seen": min(r["group_size_min_occupied"] for r in subset),
            "mean_group_size_median": round(float(np.mean([r["group_size_median_occupied"] for r in subset])), 2),
        }
        for threshold in limits:
            key = f"coverage_valid_pct_ge_{threshold}"
            row[f"mean_{key}"] = round(float(np.mean([r[key] for r in subset])), 4)
            row[f"min_{key}"] = round(float(np.min([r[key] for r in subset])), 4)
            row[f"mean_groups_ge_{threshold}"] = round(
                float(np.mean([r[f'groups_ge_{threshold}'] for r in subset])), 2
            )
        by_bin.append(row)

    _write_csv(target / "daily_summary.csv", daily)
    _write_csv(target / "by_bin_summary.csv", by_bin)
    _write_csv(target / "group_counts.csv", group_records)
    settings = {
        "bins_list": list(n_bins_all), "n_transitions": n_transitions,
        "time_start": time_start, "index_offset": index_offset,
        "thresholds": list(limits), "plot_bins": plot_bins,
        "min_group_size": min_group_size, "mask_water_value": 1,
        "water_cells": n_water, "source_shape": list(x0.shape),
        "estimator_grouping_function": "aligned_common.prepare_pair_transition",
    }
    (target / "config.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nSummary over inspected days:")
    for row in by_bin:
        print(
            f"  {row['bins_per_axis']:2d}x{row['bins_per_axis']:<2d}: "
            f"occupied={row['mean_occupied_groups']:.1f}, "
            f"coverage(n>=2)={row.get('mean_coverage_valid_pct_ge_2', 0.0):.2f}%, "
            f"coverage(n>=30)={row.get('mean_coverage_valid_pct_ge_30', 0.0):.2f}%"
        )
    print(f"\nSaved diagnostics to: {target}", flush=True)
    return daily, by_bin
