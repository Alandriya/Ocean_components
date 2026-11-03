"""Centralized image/video destinations and compatibility for old plot paths."""

from __future__ import annotations

import re
from pathlib import Path

from matplotlib import rcParams

rcParams["font.family"] = "DejaVu Sans"

_TRANSLATIONS = (
    ("Gaussian copula", "Гауссовская копула"),
    ("Empirical copula", "Эмпирическая копула"),
    ("Student t copula", "Копула Стьюдента"),
    ("Stationary density", "Плотность стационарного распределения"),
    ("stationary distribution", "стационарное распределение"),
    ("eigenvectors", "собственные векторы"),
    ("eigenvalues", "собственные значения"),
    ("singular values", "сингулярные значения"),
    ("eigenvector", "собственный вектор"),
    ("sensible heat flux", "поток явного тепла"),
    ("latent heat flux", "поток скрытого тепла"),
    ("Sensible flux", "Явный тепловой поток"),
    ("Latent flux", "Скрытый тепловой поток"),
    ("pressure", "давление"),
    ("temperature", "температура"),
    ("longitude", "долгота"),
    ("latitude", "широта"),
    ("values", "значения"),
    ("density", "плотность"),
    ("frequency", "частота"),
    ("observations", "наблюдения"),
    ("correlation", "корреляция"),
    ("diffusion", "диффузия"),
    ("drift", "снос"),
    ("distribution", "распределение"),
    ("mean year", "средний год"),
    ("difference", "разность"),
    ("comparison", "сравнение"),
    ("quantile", "квантиль"),
    ("month", "месяц"),
    ("year", "год"),
    ("days", "дни"),
    ("time", "время"),
    ("Flux", "Поток"),
    ("Sensible", "Явный"),
    ("Latent", "Скрытый"),
    ("Mean", "Среднее"),
    ("min", "мин"),
    ("max", "макс"),
)


def translate_label(value):
    """Translate frequent scientific labels while keeping symbols and units.

    Parameters
    ----------
    value : str
        Visible text from a plot or legend.

    Returns
    -------
    str
        Russian text where an established translation is available.
    """
    if not isinstance(value, str):
        return value
    result = value
    for original, translated in _TRANSLATIONS:
        def replace(match):
            return translated[0].upper() + translated[1:] if match.group(0)[:1].isupper() else translated
        result = re.sub(re.escape(original), replace, result, flags=re.IGNORECASE)
    return result


def translate_figure(figure):
    """Translate existing axes titles and labels before saving a legacy figure.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure whose visible metadata should be translated in place.

    Returns
    -------
    matplotlib.figure.Figure
        The same figure for use by the caller.
    """
    if figure._suptitle is not None:
        figure._suptitle.set_text(translate_label(figure._suptitle.get_text()))
    for axis in figure.axes:
        axis.set_title(translate_label(axis.get_title()))
        axis.set_xlabel(translate_label(axis.get_xlabel()))
        axis.set_ylabel(translate_label(axis.get_ylabel()))
        for label in axis.get_xticklabels() + axis.get_yticklabels():
            label.set_text(translate_label(label.get_text()))
        legend = axis.get_legend()
        if legend is not None:
            for item in legend.get_texts():
                item.set_text(translate_label(item.get_text()))
    return figure


def plot_path(root, *parts):
    """Return a safe path beneath <root>/Plots, creating its parent directory.

    Parameters
    ----------
    root : str or Path
        Ocean data root directory.
    *parts : str
        Relative thematic folders and filename.

    Returns
    -------
    pathlib.Path
        Requested output path under Plots.
    """
    relative = Path(*parts)
    if not parts or relative.is_absolute() or any(part in ("..", "") for part in relative.parts):
        raise ValueError("Plot destination must be a safe relative path")
    result = Path(root) / "Plots" / relative
    result.parent.mkdir(parents=True, exist_ok=True)
    return result


def _legacy_path_to_plots(filename):
    """Translate old figure output locations into the single Plots directory."""
    original = Path(filename)
    parts = original.parts
    if "Plots" in parts:
        return original
    categories = {
        "videos": "Videos", "Components": "Coefficients",
        "Func_repr": "Distributions", "Distributions": "Distributions",
        "Synthetic": "Synthetic", "Extreme": "Diagnostics",
        "Coefficients": "Diagnostics", "Eigenvalues": "Eigenvalues",
    }
    for index, part in enumerate(parts):
        if part in categories:
            prefix = Path(*parts[:index]) if index > 0 else Path(".")
            rest = list(parts[index + 1:])
            if part == "videos" and rest and rest[0].lower() in ("mean_year", "mean year"):
                rest[0] = "Mean Year"
                category = "Mean Year"
            elif part == "videos" and rest and rest[0] == "Eigenvalues":
                rest.pop(0)
                category = "Eigenvalues"
            elif part == "videos" and rest and rest[0] == "Functional":
                rest.pop(0)
                category = "Distributions"
            elif part == "videos" and rest and rest[0] == "2D":
                rest.pop(0)
                category = "Coefficients"
            else:
                category = categories[part]
            if part == "Extreme" and rest and rest[0] == "plots":
                rest[0] = "Extremes"
            return plot_path(prefix, category, *rest)
    raise ValueError(f"Cannot map legacy image path into Plots/: {filename}")


def save_figure(figure, filename, *args, **kwargs):
    """Save a figure exclusively under Plots, preserving original save options.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure to save.
    filename : str or Path
        Current or historical image pathname.
    *args, **kwargs : object
        Original Figure.savefig arguments.

    Returns
    -------
    pathlib.Path
        Actual output path.
    """
    result = _legacy_path_to_plots(filename)
    result.parent.mkdir(parents=True, exist_ok=True)
    translate_figure(figure)
    figure.savefig(str(result), *args, **kwargs)
    return result


def save_current_figure(filename, *args, **kwargs):
    """Route pyplot.savefig through the same output policy as save_figure."""
    import matplotlib.pyplot as plt
    return save_figure(plt.gcf(), filename, *args, **kwargs)
