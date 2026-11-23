"""Build a dated 15-point, multivariate daily history from existing 1D flux data.

The original Forecast_1d extraction is left unchanged. ERA5 scalar inputs are
sampled at the SAME grid cells, and the existing flux neighborhood statistics
are appended without recomputation. Absolutely no future observations are
inserted into an input window; causal slicing is done by feature_data.py.

Run from an IDE via main.py's 'extract_1d_features' task (see installer), or:
  python -m Forecast_1d.build_history_features --dry-run
  python -m Forecast_1d.build_history_features
Requires numpy only, and an existing Forecast_1d/atlantic15_* extraction.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import uuid
from datetime import date
from pathlib import Path

import numpy as np

from .data import dataset_dir as default_dataset_dir

VARIABLES = {
    "sst": ("SST", "SST"),
    "sp": ("Pressure", "Pressure"),
    "u10": ("Atmosphere", "u10"),
    "v10": ("Atmosphere", "v10"),
    "t2m": ("Atmosphere", "t2m"),
    "d2m": ("Atmosphere", "d2m"),
}
GRIB_NAMES = {"sst": "sst", "sp": "sp", "u10": "10u", "v10": "10v",
              "t2m": "2t", "d2m": "2d"}
GRID = (161, 181)
GRID_SIZE = GRID[0] * GRID[1]
BASE_DATE = np.datetime64("1979-01-01", "D")
STAT_WIDTHS = (3, 7, 15)
FLUX_NAMES = ("sensible", "latent")
EXOG_NAMES = tuple(VARIABLES)
STAT_NAMES = tuple(f"{kind}_{flux}_{width}x{width}"
                   for width in STAT_WIDTHS
                   for kind in ("mean", "std") for flux in FLUX_NAMES)
FEATURE_NAMES = FLUX_NAMES + EXOG_NAMES + STAT_NAMES
assert len(FEATURE_NAMES) == 20


def _daily_dates(raw, label: str) -> np.ndarray:
    a = np.asarray(raw)
    if a.dtype.kind != "M" or a.ndim != 1:
        raise ValueError(f"{label}: expected 1-D datetime64 array, got {a.dtype}, shape {a.shape}")
    a = a.astype("datetime64[D]")
    if np.isnat(a).any() or (a.size > 1 and not np.all(np.diff(a) == np.timedelta64(1, "D"))):
        raise ValueError(f"{label}: dates must be complete, consecutive and unique")
    if a.size == 0:
        raise ValueError(f"{label}: empty date array")
    return a


def _source_layout(path: Path) -> tuple[np.ndarray, str, int]:
    """Accept exactly the three gridded layouts supported by extract_points.py."""
    src = np.load(path, mmap_mode="r", allow_pickle=False)
    if src.dtype.kind not in "fiub" or src.ndim not in (2, 3):
        raise ValueError(f"{path}: expected numeric array with 2 or 3 axes, got {src.shape} {src.dtype}")
    if src.ndim == 3 and tuple(src.shape[1:]) == GRID:
        return src, "time_hw", int(src.shape[0])
    if src.ndim == 2 and src.shape[0] == GRID_SIZE:
        return src, "space_time", int(src.shape[1])
    if src.ndim == 2 and src.shape[1] == GRID_SIZE:
        return src, "time_space", int(src.shape[0])
    raise ValueError(f"{path}: incompatible grid shape {src.shape}; expected 161x181")


def _center_values(src: np.ndarray, layout: str, pixel_indices: np.ndarray) -> np.ndarray:
    if layout == "space_time":
        a = src[pixel_indices, :].T
    elif layout == "time_space":
        a = src[:, pixel_indices]
    elif layout == "time_hw":
        # indexing only the selected flattened grid cells avoids copying 3D data
        ys, xs = np.divmod(pixel_indices, GRID[1])
        a = src[:, ys, xs]
    else:
        raise AssertionError(layout)
    return np.asarray(a, dtype=np.float32)


def _manifest_for_array(metadata_root: Path, path: Path, variable: str):
    matches = []
    for mdfile in sorted(metadata_root.glob("ERA5_*_metadata.json")):
        md = json.loads(mdfile.read_text(encoding="utf-8"))
        entry = md.get("variables", {}).get(GRIB_NAMES[variable])
        if not entry:
            continue
        # GRIB paths embedded in JSON may refer to a Windows workstation;
        # match only the filename and use the current DATA_ROOT for data.
        declared = str(entry.get("path", "")).replace("\\", "/").split("/")[-1]
        if declared != path.name:
            continue
        dates_name = mdfile.name.removeprefix("ERA5_").removesuffix("_metadata.json")
        dates_path = metadata_root / f"ERA5_{dates_name}_dates.npy"
        if not dates_path.is_file():
            raise FileNotFoundError(f"Metadata exists but dates file missing: {dates_path}")
        days = _daily_dates(np.load(dates_path, allow_pickle=False), str(dates_path))
        if md.get("actual_first_date") != str(days[0]) or md.get("actual_last_date") != str(days[-1]):
            raise ValueError(f"{mdfile}: first/last dates disagree with {dates_path}")
        for axis_name, expected in (("latitude", "lat_deg"), ("longitude", "lon_deg")):
            axes_path = metadata_root / f"ERA5_{dates_name}_{axis_name}s.npy"
            # 'latitude' -> 'latitudes', 'longitude' -> 'longitudes'
            if not axes_path.is_file():
                raise FileNotFoundError(f"Missing coordinate file for {path.name}: {axes_path}")
        matches.append((days, mdfile, dates_name, entry))
    if len(matches) > 1:
        raise ValueError(f"{path}: matched multiple ERA5 metadata manifests: {[str(v[1]) for v in matches]}")
    return matches[0] if matches else None


def _check_coordinates(metadata_root: Path, label: str, points: list[dict]):
    lat = np.load(metadata_root / f"ERA5_{label}_latitudes.npy", mmap_mode="r", allow_pickle=False)
    lon = np.load(metadata_root / f"ERA5_{label}_longitudes.npy", mmap_mode="r", allow_pickle=False)
    if lat.shape != GRID or lon.shape != GRID:
        raise ValueError(f"ERA5 {label}: coordinates {lat.shape}/{lon.shape} differ from {GRID}")
    for p in points:
        y, x = int(p["grid_y"]), int(p["grid_x"])
        if not np.isclose(float(lat[y,x]), float(p["lat_deg"]), atol=1e-4):
            raise ValueError(f"ERA5 {label}: latitude mismatch at {p['id']}: {lat[y,x]} vs {p['lat_deg']}")
        # Longitude may be -70 or 290 depending on ECMWF longitude convention.
        if not np.isclose(((float(lon[y,x]) - float(p["lon_deg"]) + 180) % 360) - 180, 0, atol=1e-4):
            raise ValueError(f"ERA5 {label}: longitude mismatch at {p['id']}: {lon[y,x]} vs {p['lon_deg']}")


def _legacy_dates(path: Path, n: int, variable: str) -> np.ndarray:
    """Fallback only for original SST/Pressure arrays without ERA5 sidecars.

    Original full-period gridded arrays start 1979-01-01 and have no explicit
    dates sidecar. The final 2019-2026 block includes an incomplete calendar
    year 2026, unlike the earlier half-open 10-year blocks. With no sidecars,
    the file start date is inferred from its name and days are assumed consecutive.
    """
    match = re.fullmatch(r"[A-Za-z0-9]+_(\d{4})-(\d{4})_grouped\.npy", path.name)
    if not match:
        raise ValueError(f"{variable}: no ERA5 metadata and cannot infer start from {path.name}")
    first, last = (int(x) for x in match.groups())
    if last <= first:
        raise ValueError(f"Invalid dates in legacy filename: {path.name}")
    start = np.datetime64(f"{first:04d}-01-01", "D")
    expected = int((np.datetime64(f"{last:04d}-01-01", "D") - start) / np.timedelta64(1, "D"))
    # Legacy naming uses half-open [first, last) for the earlier decades,
    # but the final and combined 1979-2026 products can include observations
    # during 2026 (through at most 2026-12-31). Do not infer missing *interior*
    # days here; they cannot be detected without explicit date sidecars.
    through_2026 = (first, last) in ((1979, 2026), (2019, 2026))
    final_limit = int((np.datetime64("2027-01-01", "D") - start) / np.timedelta64(1, "D"))
    if not (n == expected or (through_2026 and 0 < n <= final_limit)):
        raise ValueError(
            f"{path}: ambiguous no-metadata date range ({n} days, nominal {first}-{last}); "
            "provide ERA5_metadata/ERA5_*_dates.npy and matching JSON"
        )
    print(f"[legacy-date] {variable}: assuming {start} + {n} days for {path.name}", flush=True)
    return start + np.arange(n).astype("timedelta64[D]")


# Five canonical non-overlapping periods used by the Ocean_components archive.
# Do not scan every *_grouped.npy, because that also selects partial backups
# such as SST_2019-2025_grouped.npy and overlapping whole-period arrays.
CANONICAL_PERIODS = (
    "1979-1989", "1989-1999", "1999-2009", "2009-2019", "2019-2026",
)


def _choose_source_paths(directory: Path, prefix: str, variable: str) -> list[Path]:
    available = sorted(directory.glob(f"{prefix}_*_grouped.npy"))
    if not available:
        raise FileNotFoundError(f"No arrays for {variable} under {directory}")
    canonical = [directory / f"{prefix}_{period}_grouped.npy" for period in CANONICAL_PERIODS]
    missing = [p.name for p in canonical if not p.is_file()]
    if not missing:
        chosen = canonical
    else:
        whole = directory / f"{prefix}_1979-2026_grouped.npy"
        if whole.is_file() and not any(p.is_file() for p in canonical):
            chosen = [whole]
        else:
            raise FileNotFoundError(
                f"{variable}: incomplete canonical period set in {directory}. "
                f"Missing: {missing}. Existing: {[p.name for p in available]}. "
                "Use all five standard period files, or one 1979-2026 file; "
                "do not mix overlapping sources."
            )
    ignored = sorted(set(available) - set(chosen))
    if ignored:
        print(f"[ignore] {variable}: ignoring noncanonical/overlapping files: "
              + ", ".join(p.name for p in ignored), flush=True)
    return chosen


def _find_sources(root: Path, variable: str, points: list[dict]):
    folder, prefix = VARIABLES[variable]
    paths = _choose_source_paths(root / "DATA" / folder, prefix, variable)
    metadata_root = root / "DATA" / "ERA5_metadata"
    pixel_indices = np.asarray([int(p["grid_y"]) * GRID[1] + int(p["grid_x"]) for p in points])
    sources = []
    for path in paths:
        src, layout, n = _source_layout(path)
        matching = _manifest_for_array(metadata_root, path, variable)
        if matching is not None:
            days, mdfile, label, entry = matching
            if len(days) != n:
                raise ValueError(f"{path.name}: {n} values but {len(days)} dates in {mdfile}")
            _check_coordinates(metadata_root, label, points)
            provenance = str(mdfile)
            units = entry.get("units")
        elif variable in ("sst", "sp"):
            days = _legacy_dates(path, n, variable)
            provenance = "legacy filename/start-date assumption (no date sidecar)"
            units = None
        else:
            raise ValueError(
                f"{variable}: missing ERA5 metadata for {path.name}. "
                f"Copy ERA5_*_metadata.json and corresponding *_dates/latitudes/longitudes.npy "
                f"to {metadata_root} (metadata is required for decadal atmosphere arrays)."
            )
        sources.append((path, src, layout, days, pixel_indices,
                        {"path": str(path), "date_source": provenance,
                         "first_date": str(days[0]), "last_date": str(days[-1]),
                         "n_days": int(n), "units": units}))
        print(f"[source] {variable}: {path.name} [{days[0]}..{days[-1]}], {src.shape}", flush=True)
    return sources


def _combine_sources(sources, target_dates: np.ndarray, variable: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    ntime = len(target_dates)
    npoints = len(sources[0][4])
    values = np.full((ntime, npoints), np.nan, dtype=np.float32)
    covered = np.zeros(ntime, dtype=bool)
    duplicates = 0
    for path, src, layout, days, pixels, meta in sources:
        # All dates are daily and monotonically increasing: an integer shift
        # is enough and never attempts an implicit join by array position.
        offsets = (days - target_dates[0]).astype("timedelta64[D]").astype(np.int64)
        chosen = (offsets >= 0) & (offsets < ntime)
        if not np.any(chosen):
            continue
        dest_idx = offsets[chosen]
        src_values = _center_values(src, layout, pixels)[chosen]
        for index, vector in zip(dest_idx, src_values):
            index = int(index)
            if covered[index]:
                if not np.array_equal(values[index], vector, equal_nan=True):
                    raise ValueError(
                        f"Conflicting {variable} data for {target_dates[index]} "
                        f"(second file {path.name}); refusal to merge"
                    )
                duplicates += 1
            else:
                covered[index] = True
                values[index] = vector
    if duplicates:
        print(f"[deduplicate] {variable}: {duplicates} identical overlapping dates ignored", flush=True)
    return values, covered, [item[-1] for item in sources]


def _flux_data(dataset: Path):
    centers = np.load(dataset / "centers.npy", mmap_mode="r", allow_pickle=False)
    dates = _daily_dates(np.load(dataset / "dates.npy", allow_pickle=False), str(dataset / "dates.npy"))
    if centers.ndim != 3 or centers.shape[0] != len(dates) or centers.shape[2] != 2:
        raise ValueError(f"centers.npy must be [T,P,2], dates [T]; got {centers.shape}")
    payload = json.loads((dataset / "points.json").read_text(encoding="utf-8"))
    pts = payload["points"]
    if len(pts) != centers.shape[1] or [p["id"] for p in pts] != [f"P{i}" for i in range(1, len(pts) + 1)]:
        raise ValueError("points.json order does not match centers.npy")
    return centers, dates, pts


def _stats(dataset: Path, n: int, p: int):
    results = []
    for width in STAT_WIDTHS:
        for kind in ("mean", "std"):
            path = dataset / f"{kind}_{width}x{width}.npy"
            a = np.load(path, mmap_mode="r", allow_pickle=False)
            if a.shape != (n, p, 2):
                raise ValueError(f"{path}: expected {(n,p,2)}; got {a.shape}")
            results.append((path, a))
    return results


def build(data_root: Path, dataset: Path | None = None, output: Path | None = None,
          *, allow_trim: bool = True, overwrite: bool = False, dry_run: bool = False) -> dict:
    """Create one feature vector per time/point, preserving untouched originals.

    Output features.npy [T,P,20], exogenous.npy [T,P,6], targets.npy [T,P,2],
    dates.npy [T], manifest.json. Source ERA5 variables are physical, unscaled.
    """
    root = Path(data_root).expanduser().resolve()
    ds = Path(dataset).expanduser().resolve() if dataset is not None else default_dataset_dir(root_override=root)
    out = Path(output).expanduser().resolve() if output is not None else ds / "features_era5"
    if out == ds:
        raise ValueError("Use a dedicated features output directory, not the original dataset directory")
    if not ds.is_dir():
        raise FileNotFoundError(f"Existing flux dataset not found: {ds}. Run extract_1d_series first")
    centers, dates, points = _flux_data(ds)
    n, p, _ = centers.shape
    stats = _stats(ds, n, p)
    print(f"[flux] {ds}: {n} days, {p} points, {dates[0]}..{dates[-1]}", flush=True)
    all_aux = []
    all_coverage = []
    provenance = {}
    for variable in EXOG_NAMES:
        sources = _find_sources(root, variable, points)
        values, coverage, files = _combine_sources(sources, dates, variable)
        if not np.any(coverage) or not coverage[0]:
            raise ValueError(f"{variable}: history must start on {dates[0]} to align with flux series")
        # Missing intervals in the middle are never silently filled or truncated.
        last = int(np.flatnonzero(coverage)[-1])
        first_gap = np.flatnonzero(~coverage[:last+1])
        if len(first_gap):
            raise ValueError(f"{variable}: missing internal day {dates[first_gap[0]]}; check period files/metadata")
        all_aux.append(values)
        all_coverage.append(coverage)
        provenance[variable] = files
    kept = min(int(np.flatnonzero(coverage)[-1]) + 1 for coverage in all_coverage)
    if kept != n:
        missing = {name: str(dates[int(np.flatnonzero(cov)[-1])+1])
                   for name, cov in zip(EXOG_NAMES, all_coverage) if not np.all(cov)}
        if not allow_trim:
            raise ValueError(f"Not all variables reach {dates[-1]}; first missing days: {missing}")
        print(f"[trim] From {n} to {kept} days ending {dates[kept-1]}; truncated tails: {missing}", flush=True)
    if kept < 90:
        raise ValueError(f"Only {kept} common days; insufficient for a 60->14 experiment")
    report = {
        "description": "Causally slice this dated matrix; features are not future forecasts",
        "shape": [kept, p, len(FEATURE_NAMES)],
        "source_flux_dataset": str(ds),
        "first_date": str(dates[0]), "last_date": str(dates[kept-1]),
        "full_flux_days": n, "aligned_days": kept, "n_points": p,
        "feature_names": list(FEATURE_NAMES), "exogenous_names": list(EXOG_NAMES),
        "target_names": list(FLUX_NAMES),
        "features_layout": "[day, point, channel]",
        "targets_layout": "[day, point, sensible/latent]",
        "value_units": {"sensible": "same as original centers.npy", "latent": "same as original centers.npy",
                        "sst": "K", "sp": "Pa", "u10": "m/s", "v10": "m/s", "t2m": "K", "d2m": "K"},
        "source_metadata": provenance,
        "processing": "Physical daily values; no normalization, imputation, or future data",
        "date_alignment": "ERA5 dates from sidecars; original full SST/Pressure use explicit legacy fallback",
        "flux_spatial_stats": "Existing 3x3,7x7,15x15 ocean patches; no re-extraction",
    }
    if dry_run:
        print(f"[dry-run] OK: will create {out} with features {report['shape']}; nothing written", flush=True)
        return report
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} already exists. --overwrite replaces features_era5 only")
    features = np.empty((kept, p, len(FEATURE_NAMES)), dtype=np.float32)
    features[..., :2] = centers[:kept]
    for i, vals in enumerate(all_aux):
        features[..., 2+i] = vals[:kept]
    for i, (_path, values) in enumerate(stats):
        features[..., 8+2*i:10+2*i] = values[:kept]
    if not np.isfinite(features).all():
        where = np.argwhere(~np.isfinite(features))[0]
        di, pi, ci = map(int, where)
        raise ValueError(f"Nonfinite feature {FEATURE_NAMES[ci]} at {dates[di]}, {points[pi]['id']}: "
                         f"{features[di,pi,ci]}; no silent imputation")

    out.parent.mkdir(parents=True, exist_ok=True)
    stage = out.parent / f".{out.name}.building-{uuid.uuid4().hex[:12]}"
    stage.mkdir()
    backup = None
    published = False
    try:
        np.save(stage / "features.npy", features, allow_pickle=False)
        np.save(stage / "exogenous.npy", features[..., 2:8].copy(), allow_pickle=False)
        np.save(stage / "targets.npy", features[..., :2].copy(), allow_pickle=False)
        np.save(stage / "dates.npy", dates[:kept], allow_pickle=False)
        shutil.copyfile(ds / "points.json", stage / "points.json")
        (stage / "manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if out.exists():
            backup = out.parent / f".{out.name}.previous-{uuid.uuid4().hex[:12]}"
            out.rename(backup)
        try:
            stage.rename(out)
        except BaseException:
            if backup is not None and backup.exists() and not out.exists():
                backup.rename(out)
                backup = None
            raise
        published = True
        if backup is not None:
            shutil.rmtree(backup)
        print(f"[DONE] {out}: {report['shape']} features, dates {dates[0]}..{dates[kept-1]}", flush=True)
        return report
    finally:
        if stage.exists():
            if published:
                shutil.rmtree(stage)
            else:
                print(f"[recovery] staging directory remains at {stage}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--strict-full-period", action="store_true", help="Reject missing final ERA5 days instead of trimming tail")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.data_root is None:
        from project_paths import DATA_ROOT
        root = DATA_ROOT
    else:
        root = args.data_root
    return build(root, args.dataset_dir, args.output, allow_trim=not args.strict_full_period,
                 overwrite=args.overwrite, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
