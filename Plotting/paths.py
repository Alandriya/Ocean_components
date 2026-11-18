"""Centralized image/video destinations and compatibility for old plot paths."""

from __future__ import annotations

from pathlib import Path

from matplotlib import rcParams

rcParams["font.family"] = "DejaVu Sans"

from Plotting.language import translate_label, get_plot_language


def translate_figure(figure, language=None):
    """Explicitly localize all currently created Text artists (legacy API).

    Normal saving does not call this function: rendering is localized by
    Plotting.language without permanently mutating the source strings.
    """
    from matplotlib.text import Text
    for artist in figure.findobj(match=Text):
        value = artist.get_text()
        translated = translate_label(value, language or get_plot_language())
        if translated != value:
            artist.set_text(translated)
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
    # The shared rendering adapter localizes the text when drawing the figure.
    figure.savefig(str(result), *args, **kwargs)
    return result


def save_current_figure(filename, *args, **kwargs):
    """Route pyplot.savefig through the same output policy as save_figure."""
    import matplotlib.pyplot as plt
    return save_figure(plt.gcf(), filename, *args, **kwargs)
