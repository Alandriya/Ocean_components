"""Rolling C/F/FS computation from temporally continuous A/B fields."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from Coefficients.coefficients_cf import compute_c, compute_f, compute_fs


def run_continuous_cf(files_path_prefix, names=("sensible", "latent"), *,
                      n_bins_2d=12, method="np", correlation_window=14,
                      mean_width=7, mask=None, overwrite=False):
    """Calculate derived coefficients from all available daily transitions.

    Parameters
    ----------
    files_path_prefix : str or Path
        Root directory containing Coefficients/continuous arrays.
    names : tuple of str
        Ordered variable pair.
    n_bins_2d : int
        Quantile bin count in the parent directory.
    method : {'np', 'sp'}
        Existing estimator identifier.
    correlation_window : int
        Rolling number of transitions for C Pearson correlations.
    mean_width : int
        Rolling number of transitions for F and FS mean fields.
    mask : array_like of bool, optional
        Ocean grid mask.
    overwrite : bool
        Whether existing aggregated arrays are replaced.

    Returns
    -------
    dict
        Output paths and inclusive first/last transition labels.
    """
    if correlation_window < 3 or mean_width < 1:
        raise ValueError("Invalid window sizes")
    if method not in ("np", "sp"):
        raise ValueError("method must be 'np' or 'sp'")
    folder = "Nonparametric_2d" if method == "np" else "Semiparametric_2d"
    parent = (Path(files_path_prefix) / "Coefficients" / "continuous" /
              f"2d_bins_{n_bins_2d}" / f"{names[0]}_{names[1]}" / folder)
    manifest_path = Path(files_path_prefix) / "Coefficients" / "continuous" / (
        f"manifest_{names[0]}_{names[1]}_{n_bins_2d}.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if method not in manifest["methods"]:
        raise ValueError(f"Method {method} not present in the continuous coefficient series")
    a = np.load(manifest["methods"][method]["A"], mmap_mode="r")
    b = np.load(manifest["methods"][method]["B"], mmap_mode="r")
    if a.ndim != 4 or b.ndim != 4 or a.shape[1] != 2 or b.shape[1] != 4 or len(a) != len(b):
        raise ValueError("Expected aligned A=(T,2,H,W) and B=(T,4,H,W)")
    if len(a) < max(correlation_window, mean_width):
        raise ValueError("Insufficient number of days for the requested windows")
    first_label = manifest["first_label"]
    result = {}
    for key, width in (("C", correlation_window), ("F", mean_width), ("FS", mean_width)):
        first = first_label + width - 1
        last = first_label + len(a) - 1
        output = parent / f"{key}_{first}-{last}.npy"
        if not output.exists() or overwrite:
            output_shape = ((2, *a.shape[2:]) if key in ("C", "FS") else a.shape[2:])
            target = np.lib.format.open_memmap(output, mode="w+", dtype=float,
                                               shape=(len(a) - width + 1, *output_shape))
            for step in range(len(target)):
                aw = np.asarray(a[step:step + width], dtype=float)
                bw = np.asarray(b[step:step + width], dtype=float)
                if key == "C":
                    target[step] = compute_c(aw, bw, mask)
                else:
                    valid = (np.all(np.isfinite(aw), axis=(0, 1)) &
                             np.all(np.isfinite(bw), axis=(0, 1)))
                    average_a, average_b = aw.mean(axis=0), bw.mean(axis=0)
                    average_a[:, ~valid] = np.nan
                    average_b[:, ~valid] = np.nan
                    target[step] = (compute_f(average_a, average_b, mask) if key == "F"
                                    else compute_fs(average_a, average_b, mask))
            target.flush()
            del target
        result[key] = {"path": str(output), "first_label": first, "last_label": last}
    return result
