"""Diagnostic plots for the one-day, two-component spatial mixture experiment.

The script does NOT refit the two-component model. It reconstructs the exact
one-day spatial quantile groups from the original variable array and overlays:
  * empirical density histogram;
  * Gaussian KDE;
  * one-Gaussian MLE reference;
  * weighted densities of the two saved EM components;
  * their total mixture density.

Run from the repository root:
    python -m Plotting.component_diagnostics

Edit the USER SETTINGS section at the bottom before running.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import csv
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, kurtosis, norm, skew
from tqdm.auto import tqdm

from Coefficients.common import prepare_transition
from Coefficients.spatial_components import component_run_parts
from Data_processing.io import field_candidates, load_field


MONTHS_RU = (
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
)


def _date_label(value: date) -> str:
    return f"{value.day} {MONTHS_RU[value.month - 1]} {value.year}"


def _default_groups(n_bins: int) -> tuple[int, ...]:
    """One-based lower, central and upper quantile ranks without duplicates."""
    candidates = (1, (n_bins + 1) // 2, n_bins)
    return tuple(dict.fromkeys(candidates))


def _density_histogram(values: np.ndarray, bins: int, limits: tuple[float, float]):
    edges = np.linspace(limits[0], limits[1], bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    widths = np.diff(edges)
    density = counts / (values.size * widths)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, density, widths


def _safe_kde(values: np.ndarray, x: np.ndarray):
    if values.size < 3 or not np.isfinite(np.std(values)) or np.std(values) <= 0:
        return None
    try:
        return gaussian_kde(values)(x)
    except (np.linalg.LinAlgError, ValueError):
        return None


def _plot_distribution(
    values: np.ndarray,
    *,
    mu: np.ndarray,
    sigma: np.ndarray,
    weights: np.ndarray,
    title: str,
    subtitle: str,
    hist_bins: int,
    zoom_quantiles: tuple[float, float],
    kde_points: int,
    dpi: int,
    output_png: Path,
):
    """Create full-range and central-quantile views of one group/date."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3:
        raise ValueError(f"Too few finite increments for diagnostics: n={values.size}")

    vmin, vmax = float(np.min(values)), float(np.max(values))
    spread = vmax - vmin
    if not np.isfinite(spread) or spread <= 0:
        spread = max(abs(vmin), 1.0) * 1e-6
    full_limits = (vmin - 0.025 * spread, vmax + 0.025 * spread)
    qlo, qhi = np.quantile(values, zoom_quantiles)
    zoom_spread = max(float(qhi - qlo), spread * 1e-6)
    zoom_limits = (float(qlo - 0.06 * zoom_spread), float(qhi + 0.06 * zoom_spread))

    sample_mu = float(np.mean(values))
    sample_sigma = float(np.std(values, ddof=0))
    if sample_sigma <= 0:
        sample_sigma = None

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), constrained_layout=True)
    try:
        for ax, limits, panel_title in (
            (axes[0], full_limits, "Полный диапазон"),
            (axes[1], zoom_limits, f"Центральная часть: {zoom_quantiles[0]:.0%}–{zoom_quantiles[1]:.0%} (нормировка на всю выборку)"),
        ):
            x = np.linspace(limits[0], limits[1], kde_points)
            centers, hist_density, widths = _density_histogram(values, hist_bins, limits)
            ax.bar(
                centers, hist_density, width=widths, align="center",
                color="#c7c7c7", edgecolor="#666666", linewidth=0.45,
                alpha=0.55, label="Эмпирическая гистограмма", zorder=1,
            )

            kde = _safe_kde(values, x)
            if kde is not None:
                ax.plot(x, kde, color="#111111", linewidth=1.8,
                        label="KDE (эмпирическая плотность)", zorder=5)

            if sample_sigma is not None:
                ax.plot(x, norm.pdf(x, sample_mu, sample_sigma), color="#666666",
                        linestyle=":", linewidth=1.9,
                        label="Одна нормаль (MLE по выборке)", zorder=3)

            component_curves = []
            if (np.asarray(mu).shape == (2,) and np.asarray(sigma).shape == (2,)
                    and np.asarray(weights).shape == (2,)
                    and np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))
                    and np.all(np.isfinite(weights)) and np.all(np.asarray(sigma) > 0)):
                colors = ("#2468a2", "#d14a36")
                for k in range(2):
                    contribution = float(weights[k]) * norm.pdf(x, float(mu[k]), float(sigma[k]))
                    component_curves.append(contribution)
                    ax.plot(
                        x, contribution, color=colors[k], linestyle="--", linewidth=1.8,
                        label=(f"Компонента {k + 1}: p={weights[k]:.2f}, "
                               f"μ={mu[k]:.2f}, σ={sigma[k]:.2f}"), zorder=4,
                    )
                ax.plot(x, component_curves[0] + component_curves[1],
                        color="#7a3e9d", linewidth=2.5,
                        label="Смесь EM: p₁φ₁ + p₂φ₂", zorder=6)

            ax.set_xlim(*limits)
            ax.set_ylim(bottom=0)
            ax.set_title(panel_title, fontsize=12)
            ax.set_xlabel("Однодневное приращение sensible")
            ax.set_ylabel("Плотность вероятности")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8, loc="best", framealpha=0.92)

        fig.suptitle(f"{title}\n{subtitle}", fontsize=13)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
    finally:
        plt.close(fig)
    return output_png


def _load_run(root: Path, start_year: int, end_year: int, variable: str,
              n_bins: int, first_date: str, last_date: str):
    parts = component_run_parts(
        start_year, end_year, variable, n_bins, 1, 1,
        first_date, last_date, spatial_stride=1,
    )
    run_dir = root / "Coefficients" / "Components" / Path(*parts)
    meta_path = run_dir / "run.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Не найден расчёт components: {meta_path}")
    import json
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("window_days") != 1 or metadata.get("step_days") != 1:
        raise ValueError(f"Ожидается однодневный расчёт: {meta_path}")
    return parts, run_dir, metadata


def _reconstruct_daily_groups(fields, endpoint: date, start_year: int,
                              n_bins: int, water_mask: np.ndarray):
    origin = date(start_year, 1, 1)
    index = (endpoint - origin).days
    if index < 1 or index >= len(fields):
        raise IndexError(
            f"Дата {endpoint} вне доступных переходов {origin + timedelta(days=1)}.."
            f"{origin + timedelta(days=len(fields) - 1)}"
        )
    return prepare_transition(fields[index - 1], fields[index], n_bins, water_mask)


def _summarize(values, mu, sigma, weights, stored, group_id, endpoint, n_bins,
               lower_edge, upper_edge):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    result = {
        "date": endpoint.isoformat(),
        "n_bins": n_bins,
        "group": group_id + 1,
        "quantile_lower_edge": float(lower_edge),
        "quantile_upper_edge": float(upper_edge),
        "n_observations": int(values.size),
        "sample_mean": float(np.mean(values)),
        "sample_std_mle": float(np.std(values, ddof=0)),
        "sample_skewness": float(skew(values, bias=False)) if values.size > 2 else np.nan,
        "sample_excess_kurtosis": float(kurtosis(values, fisher=True, bias=False)) if values.size > 3 else np.nan,
        "component_1_mu": float(mu[0]),
        "component_1_sigma": float(sigma[0]),
        "component_1_weight": float(weights[0]),
        "component_2_mu": float(mu[1]),
        "component_2_sigma": float(sigma[1]),
        "component_2_weight": float(weights[1]),
        "separation": float(stored["separation"]),
        "bic_1": float(stored["bic1"]),
        "bic_2": float(stored["bic2"]),
        "delta_bic": float(stored["delta_bic"]),
        "supported": bool(stored["supported"]),
        "converged": bool(stored["converged"]),
        "tracking": int(stored["tracking"]),
    }
    return result


def plot_component_diagnostics(
    files_path_prefix,
    start_year,
    end_year,
    *,
    variable="sensible",
    n_bins_values=(10,),
    first_date="2019-01-01",
    last_date="2019-03-31",
    dates=("2019-01-15", "2019-02-15", "2019-03-15"),
    groups=None,
    hist_bins=55,
    zoom_quantiles=(0.01, 0.99),
    kde_points=900,
    dpi=220,
    progress=True,
):
    """Plot empirical distributions against saved K=2 mixture fits.

    Parameters
    ----------
    files_path_prefix : path-like
        Project data root (same root used by ``main.py``).
    start_year, end_year : int
        Period identifying the original variable array, e.g. 1979, 2026.
    variable : str
        Scalar variable used by the components experiment; default ``sensible``.
    n_bins_values : iterable[int]
        Saved quantile-group counts to diagnose. For example (6, 10, 12, 15, 20).
    first_date, last_date : ISO dates
        Must match the saved components run directory.
    dates : iterable[str]
        Endpoint dates to inspect. These are transition end dates, not start dates.
    groups : iterable[int] or None
        One-based quantile ranks. None selects the lower, central and upper ranks
        separately for each n_bins value.
    hist_bins : int
        Number of bins in each histogram view.
    zoom_quantiles : tuple[float, float]
        Central range displayed in the right-hand panel; all observations remain
        in the fit and density normalization.

    Returns
    -------
    list[Path]
        PNG plots, one CSV summary and one multipage PDF per n_bins value.
    """
    root = Path(files_path_prefix)
    n_bins_values = tuple(int(x) for x in n_bins_values)
    dates = tuple(date.fromisoformat(str(x)) for x in dates)
    if not n_bins_values or any(x < 2 for x in n_bins_values):
        raise ValueError("n_bins_values must contain positive integers >= 2")
    if not dates:
        raise ValueError("At least one endpoint date is required")
    if not (0 < zoom_quantiles[0] < zoom_quantiles[1] < 1):
        raise ValueError("zoom_quantiles must satisfy 0 < low < high < 1")
    if hist_bins < 5 or kde_points < 100:
        raise ValueError("hist_bins must be >= 5 and kde_points >= 100")

    # All requested runs use the same source field. load_field uses mmap-backed input.
    source_path = next((p for p in field_candidates(root, variable, start_year, end_year)
                        if p.is_file()), None)
    if source_path is None:
        raise FileNotFoundError(
            f"Не найден исходный массив {variable} за {start_year}-{end_year} "
            f"в корне данных {root}"
        )
    print(f"[components diagnostics] Loading {source_path}", flush=True)
    fields = load_field(root, variable, start_year, end_year)
    origin = date(start_year, 1, 1)
    results: list[Path] = []

    for n_bins in n_bins_values:
        parts, run_dir, metadata = _load_run(
            root, start_year, end_year, variable, n_bins, first_date, last_date,
        )
        expected_size = metadata.get("source_size")
        if expected_size is not None and int(expected_size) != source_path.stat().st_size:
            raise ValueError(
                f"Размер исходного массива изменился после расчёта components: "
                f"{source_path} ({source_path.stat().st_size} bytes; в run.json {expected_size})."
            )
        expected_mtime = metadata.get("source_mtime_ns")
        if expected_mtime is not None and int(expected_mtime) != source_path.stat().st_mtime_ns:
            warnings.warn(
                f"Время изменения исходного массива отличается от run.json: {source_path}. "
                "Далее будет дополнительно проверено точное совпадение bin_map.",
                RuntimeWarning,
            )
        saved_windows = set(metadata.get("windows", ()))
        selected_groups = tuple(groups) if groups is not None else _default_groups(n_bins)
        if any(not 1 <= int(g) <= n_bins for g in selected_groups):
            raise ValueError(f"For {n_bins} groups, selected group ranks must be in 1..{n_bins}")
        selected_groups = tuple(int(g) for g in selected_groups)

        output_dir = root / "Plots" / "Components_Diagnostics" / Path(*parts)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_rows = []
        pdf_path = output_dir / "distributions.pdf"

        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(pdf_path) as pdf:
            jobs = [(endpoint, group_rank) for endpoint in dates for group_rank in selected_groups]
            for endpoint, group_rank in tqdm(
                jobs, desc=f"Diagnostics: {n_bins} groups", unit="plot", disable=not progress
            ):
                filename = f"window_{endpoint.isoformat()}.npz"
                if filename not in saved_windows:
                    raise FileNotFoundError(
                        f"No saved mixture for {endpoint} in {run_dir}. "
                        f"Available endpoints are defined in run.json."
                    )
                npz_path = run_dir / filename
                if not npz_path.is_file():
                    raise FileNotFoundError(npz_path)

                with np.load(npz_path, allow_pickle=False) as archive:
                    endpoint_saved = str(archive["endpoint_date"].item())
                    if endpoint_saved != endpoint.isoformat():
                        raise ValueError(f"Date mismatch in {npz_path}: {endpoint_saved}")
                    water = archive["water_mask"].astype(bool, copy=True)
                    saved_bin_map = archive["bin_map"].copy()
                    saved_mu = archive["mu"].copy()
                    saved_sigma = archive["sigma"].copy()
                    saved_weights = archive["weight"].copy()
                    saved_supported = archive["supported"].copy()
                    saved_converged = archive["converged"].copy()
                    saved_n_fit = archive["n_fit"].copy()
                    saved_separation = archive["separation"].copy()
                    saved_bic1 = archive["bic1"].copy()
                    saved_bic2 = archive["bic2"].copy()
                    saved_delta_bic = archive["delta_bic"].copy()
                    saved_tracking = archive["tracking"].copy()
                    saved_edges = archive["daily_edges"].copy()

                transition = _reconstruct_daily_groups(
                    fields, endpoint, start_year, n_bins, water,
                )
                rebuilt_map = np.full(transition.shape, -1, dtype=np.int32)
                groups_by_id = {}
                for item in transition.groups:
                    rebuilt_map.ravel()[item.flat_indices] = item.bin_id
                    groups_by_id[item.bin_id] = item
                if not np.array_equal(rebuilt_map, saved_bin_map):
                    mismatch = int(np.count_nonzero(rebuilt_map != saved_bin_map))
                    raise ValueError(
                        f"Reconstructed quantile groups differ from saved run for {endpoint}, "
                        f"{n_bins} bins ({mismatch} cells). Check that the original variable "
                        "array and mask have not changed."
                    )

                group_id = group_rank - 1
                group = groups_by_id.get(group_id)
                if group is None:
                    raise ValueError(f"Group {group_rank} is empty on {endpoint}")
                values = np.asarray(group.increments, dtype=float).reshape(-1)
                expected_n = int(saved_n_fit[group_id])
                if values.size != expected_n:
                    raise ValueError(
                        f"Sample size mismatch for {endpoint}, group {group_rank}: "
                        f"reconstructed {values.size}, saved model {expected_n}."
                    )

                mu = saved_mu[group_id]
                sigma = saved_sigma[group_id]
                weights = saved_weights[group_id]
                edge_row = saved_edges[-1] if saved_edges.ndim == 2 else saved_edges
                lower_edge, upper_edge = float(edge_row[group_id]), float(edge_row[group_id + 1])
                supported = bool(saved_supported[group_id])
                converged = bool(saved_converged[group_id])
                delta_bic = float(saved_delta_bic[group_id])
                support_text = "поддержка пройдена" if supported else "поддержка НЕ пройдена"
                title = (
                    f"sensible · {n_bins} квантильных групп · группа {group_rank}/{n_bins} · "
                    f"{_date_label(endpoint)}"
                )
                subtitle = (
                    f"X(t−1) ∈ [{lower_edge:.3g}; {upper_edge:.3g}] · n={values.size} · "
                    f"{support_text} · EM converged={converged} · ΔBIC={delta_bic:.1f}"
                )
                png_path = output_dir / f"group_{group_rank:02d}_{endpoint.isoformat()}.png"
                _plot_distribution(
                    values, mu=mu, sigma=sigma, weights=weights,
                    title=title, subtitle=subtitle, hist_bins=hist_bins,
                    zoom_quantiles=zoom_quantiles, kde_points=kde_points,
                    dpi=dpi, output_png=png_path,
                )
                # Add each finished PNG as a page in the combined PDF.
                image = plt.imread(png_path)
                fig, ax = plt.subplots(figsize=(15.5, 6.2))
                try:
                    ax.imshow(image)
                    ax.axis("off")
                    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.05)
                finally:
                    plt.close(fig)

                summary_rows.append(_summarize(
                    values, mu, sigma, weights,
                    {"separation": saved_separation[group_id],
                     "bic1": saved_bic1[group_id], "bic2": saved_bic2[group_id],
                     "delta_bic": saved_delta_bic[group_id], "supported": supported,
                     "converged": converged, "tracking": saved_tracking[group_id]},
                    group_id, endpoint, n_bins, lower_edge, upper_edge,
                ))
                results.append(png_path)

        csv_path = output_dir / "diagnostics_summary.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)
        results.extend((csv_path, pdf_path))
        print(f"[components diagnostics] Saved {len(jobs)} plots and summary: {output_dir}", flush=True)

    return results


# =============================================================================
# USER SETTINGS — edit these values and run `python -m Plotting.component_diagnostics`
# =============================================================================
def main():
    from project_paths import files_path_prefix

    start_year = 1979
    end_year = 2026
    variable = "sensible"

    # Start with 10 groups. To compare all tested resolutions, use (6, 10, 12, 15, 20).
    n_bins_values = (10,)
    first_date = "2019-01-01"
    last_date = "2019-03-31"

    # Three dates distributed over the available quarter.
    dates = ("2019-01-15", "2019-02-15", "2019-03-15")

    # None automatically selects lower, central and upper ranks for each bin count.
    # For 10 groups this means groups 1, 5 and 10.
    groups = None

    plot_component_diagnostics(
        files_path_prefix, start_year, end_year,
        variable=variable,
        n_bins_values=n_bins_values,
        first_date=first_date,
        last_date=last_date,
        dates=dates,
        groups=groups,
        hist_bins=55,
        zoom_quantiles=(0.01, 0.99),
        kde_points=900,
        dpi=220,
        progress=True,
    )


if __name__ == "__main__":
    main()
