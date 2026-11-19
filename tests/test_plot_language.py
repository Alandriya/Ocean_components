"""Offline render tests for the global English/Russian switch."""
import io
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from Plotting.language import (get_plot_language, set_plot_language,
                               plot_language_context, translate_label,
                               install_matplotlib_localization)
from Plotting.paths import save_figure


def svg(figure):
    buffer = io.StringIO()
    with matplotlib.rc_context({"svg.fonttype": "none"}):
        figure.savefig(buffer, format="svg", bbox_inches="tight")
    return buffer.getvalue()


class LanguageTests(unittest.TestCase):
    def setUp(self):
        self.before = get_plot_language()
        set_plot_language("en")

    def tearDown(self):
        plt.close("all")
        set_plot_language(self.before)

    def test_en_ru_labels_and_exact_scientific_names(self):
        self.assertEqual(translate_label("Собственные векторы", "en"), "Eigenvectors")
        self.assertEqual(translate_label("Плотность вероятности", "en"), "Probability density")
        self.assertEqual(translate_label("Bernstein copula stationary density", "ru"),
                         "Стационарная плотность копулы Бернштейна")
        self.assertEqual(translate_label("Stationary FP core", "ru"),
                         "Центральная часть решения ФП")
        self.assertEqual(translate_label("90° з. д.", "en"), "90°W")
        self.assertEqual(translate_label("80°N", "ru"), "80° с. ш.")
        self.assertEqual(translate_label("15 февраля", "en"), "15 February")
        self.assertEqual(translate_label(r"$\lambda_1$", "ru"), r"$\lambda_1$")
        self.assertEqual(translate_label("log(B²)", "ru"), "log(B²)")
        self.assertEqual(translate_label("1e-6, RMSE, 2026-01-01", "ru"),
                         "1e-6, RMSE, 2026-01-01")
        with self.assertRaises(ValueError):
            set_plot_language("de")

    def test_direct_save_and_colorbar_text(self):
        fig, ax = plt.subplots(constrained_layout=True)
        ax.set_title("Поля океанографических переменных")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Широта")
        image = ax.imshow(np.arange(4).reshape(2, 2))
        fig.colorbar(image, ax=ax, label="Probability density")
        ax.plot([0, 1], [1, 0], label="Stationary FP core")
        ax.legend()
        fig.text(.1, .1, "Observed histogram")
        ax.annotate("Prediction", (.5, .5))
        english = svg(fig)
        for label in ("Oceanographic variables", "Longitude", "Latitude",
                      "Probability density", "Stationary FP core", "Observed histogram", "Prediction"):
            self.assertIn(label, english)
        self.assertNotIn("Поля океанографических переменных", english)
        set_plot_language("ru")
        russian = svg(fig)
        for label in ("Поля океанографических переменных", "Долгота", "Широта",
                      "Плотность вероятности", "Центральная часть решения ФП", "Гистограмма наблюдений", "Прогноз"):
            self.assertIn(label, russian)
        self.assertNotIn("Probability density", russian)
        self.assertEqual(ax.get_title(), "Поля океанографических переменных")
        self.assertEqual(ax.get_xlabel(), "Longitude")
        self.assertEqual(ax.lines[0].get_label(), "Stationary FP core")
        self.assertEqual(svg(fig).count("Долгота"), russian.count("Долгота"))

    def test_regenerated_tick_labels_and_subfigures(self):
        fig = plt.figure(constrained_layout=True)
        sub = fig.subfigures(1, 1)
        sub.suptitle("Training curve")
        ax = sub.subplots()
        ax.plot([0, 1], [0, 1])
        ax.set_xticks([0, 1])
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: "90° з. д." if x == 0 else "EQ"))
        en = svg(fig)
        self.assertIn("90°W", en)
        self.assertIn("EQ", en)
        self.assertIn("Training curve", en)
        set_plot_language("ru")
        ru = svg(fig)
        self.assertIn("90° з. д.", ru)
        self.assertIn("Экватор", ru)
        self.assertIn("Кривая обучения", ru)

    def test_scoped_override_and_legacy_save_path(self):
        fig, ax = plt.subplots()
        ax.set_xlabel("Longitude")
        with tempfile.TemporaryDirectory() as d:
            with matplotlib.rc_context({"svg.fonttype": "none"}):
                with plot_language_context("ru"):
                    result = save_figure(fig, Path(d) / "videos" / "x.svg")
                    self.assertIn("Долгота", result.read_text(encoding="utf-8"))
                self.assertEqual(result, Path(d) / "Plots" / "Videos" / "x.svg")
                self.assertEqual(ax.get_xlabel(), "Longitude")
                self.assertEqual(get_plot_language(), "en")
                result = save_figure(fig, Path(d) / "Plots" / "x.svg")
                self.assertIn("Longitude", result.read_text(encoding="utf-8"))

    def test_install_is_idempotent(self):
        from matplotlib.text import Text
        original = Text.draw
        install_matplotlib_localization()
        self.assertIs(Text.draw, original)

    def test_stationary_regions_isolines_and_classes(self):
        # Only the renderer is being tested here: use precomputed synthetic caches.
        from unittest.mock import patch
        from Plotting import stationary_spatial_plots as a
        images = {}
        def capture(fig, path, dpi):
            images[str(path.name)] = svg(fig)
            return path
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.npz"
            grid = np.linspace(-100, 100, 101)
            ocean = np.ones((11, 12), bool)
            anomaly = np.linspace(-60, 60, ocean.size).reshape(ocean.shape)
            classes = (np.arange(ocean.size).reshape(ocean.shape) % 5 + 1).astype(np.uint8)
            np.savez(path, grid=grid, pdf=np.full_like(grid, .005),
                     thresholds=[-60, -20, 20, 60], region_masses=np.full(5, .2),
                     ocean_mask=ocean, anomaly=anomaly, model_mean=-15., class_map=classes)
            with patch.object(a, "_save", side_effect=capture):
                for lang in ("en", "ru"):
                    a.plot_area_density(path, "sensible", (1979, 2026), Path(d), language=lang)
                    a.plot_anomaly_isolines(path, "sensible", (1979, 2026), Path(d), language=lang)
                    a.plot_class_map(path, "sensible", "1979-01-01", Path(d), language=lang)
        self.assertIn("Probability density", images["sensible_regions_en.png"])
        self.assertIn("Плотность вероятности", images["sensible_regions_ru.png"])
        self.assertIn("1D stationary-reference anomaly", images["sensible_isolines_en.png"])
        self.assertIn("Аномалия относительно 1D стационарного уровня", images["sensible_isolines_ru.png"])
        self.assertIn("Area 1", images["sensible_classes_1979-01-01_en.png"])
        self.assertIn("Область 1", images["sensible_classes_1979-01-01_ru.png"])
        self.assertEqual(get_plot_language(), "en")


if __name__ == "__main__":
    unittest.main()
