"""Project-wide bilingual Matplotlib labels (English by default).

The lexicon is applied only to visible Text artists, never to filenames, arrays,
legend lookup keys, or metadata. Mathtext is preserved. Both rendering and
measurement use the same localized string, including automatically generated ticks.
"""
from __future__ import annotations

import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache, wraps

# English | Russian. Phrases precede individual words by length at compile time.
# Keep scientific acronyms/model identifiers, numeric values and formulae intact.
_PAIRS = '''
Sensible heat flux|Явный тепловой поток
Latent heat flux|Скрытый тепловой поток
Total heat flux|Суммарный тепловой поток
Heat flux|Тепловой поток
Cumulative heat flux|Полный тепловой поток
Sea surface temperature|Температура поверхности моря
Surface pressure|Давление у поверхности
Atmospheric pressure|Атмосферное давление
Sensible flux values|Значения явного теплового потока
Latent flux values|Значения скрытого теплового потока
Sensible values|Значения явного потока
Latent values|Значения скрытого потока
Sensible flux|Явный поток
Latent flux|Скрытый поток
Oceanographic variables|Поля океанографических переменных
Day of climatological year|День среднего года
Annual climatology|Средний год
Mean value|Среднее значение
Variable value|Значение переменной
Observed ERA5 histogram (display range)|Гистограмма ERA5 (область отображения)
Observed histogram|Гистограмма наблюдений
Stationary FP core|Центральная часть решения ФП
Smooth C¹ transitions|Гладкие переходы C¹
Empirical GPD tails|Эмпирические хвосты GPD
Experimental hybrid tails|Экспериментальные гибридные хвосты
2D stationary FP marginal (retained domain)|Маргинальная плотность двумерного решения ФП (сохранённая область)
Different normalization domains: ERA5 histogram (display range), 1D hybrid (output grid), 2D marginal (retained solver domain).|Различные области нормировки: гистограмма ERA5 (область отображения), гибридная 1D-плотность (выходная сетка), двумерная маргинальная плотность (область решателя).
Stationary Fokker-Planck density|Стационарная плотность Фоккера–Планка
Stationary 2D Fokker-Planck comparison|Сравнение двумерного стационарного решения Фоккера–Планка
FP density - empirical density|Плотность ФП − эмпирическая плотность
(FP - empirical) / empirical|(ФП − эмпирическая) / эмпирическая
shown for empirical|для эмпирической плотности
of peak|от максимума
Empirical joint density|Эмпирическая совместная плотность
Stationary joint density|Стационарная совместная плотность
Stationary density|Плотность стационарного распределения
Stationary distribution|Стационарное распределение
Probability density|Плотность вероятности
Stationary FP|Стационарное решение ФП
1D stationary|Одномерная стационарная плотность
2D copula marginal|Маргинальная плотность двумерной копулы
Checkerboard 2D marginal|Маргинальная плотность шахматной копулы
Bernstein 2D marginal|Маргинальная плотность копулы Бернштейна
Sensible stationary marginal|Стационарная маргинальная плотность явного потока
Latent stationary marginal|Стационарная маргинальная плотность скрытого потока
Sensible marginal|Маргинальная плотность явного потока
Latent marginal|Маргинальная плотность скрытого потока
Bernstein copula: marginal of sensible|Копула Бернштейна: маргинальная плотность явного потока
Bernstein copula: marginal of latent|Копула Бернштейна: маргинальная плотность скрытого потока
Stationary means vs grid resolution|Стационарные средние в зависимости от разрешения сетки
Stationary correlation vs grid resolution|Стационарная корреляция в зависимости от разрешения сетки
Grid size per axis|Число узлов по каждой оси
Stationary correlation|Стационарная корреляция
Mean sensible|Среднее явного потока
Mean latent|Среднее скрытого потока
Gaussian copula stationary density|Стационарная плотность гауссовской копулы
Checkerboard copula stationary density|Стационарная плотность шахматной копулы
Bernstein copula stationary density|Стационарная плотность копулы Бернштейна
Student t copula stationary density|Стационарная плотность копулы Стьюдента
Empirical checkerboard copula|Эмпирическая шахматная копула
Empirical vs Gaussian vs Student t copula|Сравнение эмпирической, гауссовской копул и копулы Стьюдента
Checkerboard vs Bernstein copula|Сравнение шахматной копулы и копулы Бернштейна
Empirical copula (solid) vs Gaussian copula (dashed)|Эмпирическая копула (сплошная) и гауссовская копула (пунктир)
Empirical (solid) vs Gaussian copula (dashed)|Эмпирическая плотность (сплошная) и гауссовская копула (пунктир)
Empirical vs checkerboard stationary density|Эмпирическая и шахматная стационарные плотности
Checkerboard stationary|Шахматная стационарная плотность
Gaussian copula|Гауссовская копула
Empirical copula|Эмпирическая копула
Student t copula|Копула Стьюдента
Checkerboard copula|Шахматная копула
Bernstein copula|Копула Бернштейна
Copula density|Плотность копулы
Bernstein degree|Степень копулы Бернштейна
Pearson correlation|Корреляция Пирсона
Spearman correlation|Корреляция Спирмена
Selected m|Выбранное m
Validation negative log-likelihood|Отрицательное логарифмическое правдоподобие на валидации
Negative log-likelihood|Отрицательное логарифмическое правдоподобие
Five equal-probability regions|Пять равновероятных областей
Anomaly isolines|Изолинии аномалий
Stationary quantile classes|Классы стационарных квантилей
ERA5 ocean cells|Океанические ячейки ERA5
1D stationary-reference anomaly|Аномалия относительно 1D стационарного уровня
2D alternative estimate|Двумерная оценка
Fraction of valid days|Доля дней с допустимыми данными
Method 1|Первый метод
2D method|Двумерный метод
Isolines of sensible heat flux|Изолинии явного теплового потока
Isolines of latent heat flux|Изолинии скрытого теплового потока
Isolines of|Изолинии
Singular modes|Сингулярные моды
Eigenvectors|Собственные векторы
Eigenvalues|Собственные значения
Singular values|Сингулярные значения
Eigenvector|Собственный вектор
Left vector|Левый вектор
Right vector|Правый вектор
Spectral mode contributions|Вклады спектральных мод
Mode contribution|Вклад моды
Annual mean mode contribution|Среднегодовой вклад моды
Number of years|Число лет
Cross moment|Смешанный момент
Second moment|Второй момент
Real values|Реальные значения
Actual values|Фактические значения
Absolute difference|Абсолютная разность
Absolute differences|Абсолютные разности
Absolute error|Абсолютная ошибка
Test values|Тестовые значения
Sensible heat flux forecast|Прогноз явного потока
Latent heat flux forecast|Прогноз скрытого потока
over the interval|в интервале
Forecast horizon|Горизонт прогноза
Forecast comparison|Сравнение прогнозов
Validation set|Валидационная выборка
Test set|Тестовая выборка
Confidence interval bound at level|Граница дов. интервала уровня
Prediction of point|Прогноз для точки
in cluster|в кластере
training on|обучение на
Train total loss|Полная функция потерь на обучении
Validation loss|Функция потерь на валидации
Train loss|Функция потерь на обучении
Dynamic loss|Динамическая функция потерь
SDE energy|Энергетический штраф СДУ
SDE drift MSE|MSE сноса СДУ
Loss / diagnostic|Функция потерь / диагностика
Training curve|Кривая обучения
Train PSNR|PSNR на обучении
Validation PSNR|PSNR на валидации
Train / validation PSNR|PSNR на обучении и валидации
Forecast target date|Дата прогноза
Input mean|Среднее входного окна
Persistence|Инерционный прогноз
lead|заблаговременность
Observed|Наблюдения
Forecast|Прогноз
Prediction|Прогноз
Predictions|Прогнозы
Validation|Валидация
Nonparametric method|Непараметрический метод
Semiparametric method|Полупараметрический метод
Number of quantiles|Количество квантилей
Quantile index: first variable|Индекс квантиля первой переменной
Quantile index: second variable|Индекс квантиля второй переменной
Quantile index|Индекс квантиля
Fraction of coefficients|Доля коэффициентов
Fraction a/b for pair|Отношение a/b для пары
Fitted normal|Подогнанное нормальное распределение
Fitted t|Подогнанное распределение Стьюдента
Drift magnitude (number of bin)|Модуль сноса (номер группы)
Drift field|Поле сноса
Diffusion trace|След матрицы диффузии
Diffusion ellipses|Эллипсы диффузии
Probability-current velocity magnitude|Модуль скорости потока вероятности
Stationary probability-flow velocity|Стационарная скорость потока вероятности
Coefficient extreme residuals|Остатки экстремальных значений коэффициента
Coefficient extreme|Экстремальные значения коэффициента
Flux correlation|Корреляция потоков
Joint distribution of sensible and latent fluxes|Совместное распределение явного и скрытого потоков
Sensible and latent fluxes distribution|Распределение явного и скрытого потоков
Sensible - sum of squared residuals|Явный поток — сумма квадратов остатков
Latent - sum of squared residuals|Скрытый поток — сумма квадратов остатков
Residials for sensible|Остатки для явного потока
Residials for latent|Остатки для скрытого потока
Shapiro-Wiik test p-value|p-значение критерия Шапиро–Уилка
A-flux value dependence|Зависимость A от значения потока
Dependence on|Зависимость от
SS (polynomial)|СКО (полином)
Gulf Stream, downstream of Cape Hatteras|Гольфстрим ниже по течению от мыса Хаттерас
Gulf Stream, western open-ocean extension|Западное продолжение Гольфстрима в открытом океане
Gulf Stream, eastern extension|Восточное продолжение Гольфстрима
North Atlantic Current, central basin|Северо-Атлантическое течение, центральная часть бассейна
Sargasso Sea|Саргассово море
Western tropical Atlantic|Западная тропическая Атлантика
Eastern tropical Atlantic|Восточная тропическая Атлантика
Eastern subtropical Atlantic|Восточная субтропическая Атлантика
Subpolar Atlantic, Irminger region|Субполярная Атлантика, район моря Ирмингера
Norwegian Sea, offshore waters|Норвежское море, открытая акватория
Coefficient|Коэффициент
coeff|коэффициент
Component|Компонент
Longitude|Долгота
Latitude|Широта
Equator|Экватор
Density|Плотность
Frequency|Частота
Observations|Наблюдения
Correlation|Корреляция
Correlations|Корреляции
Diffusion|Диффузия
Drift|Снос
Distribution|Распределение
Difference|Разность
Comparison|Сравнение
Quantile|Квантиль
Quantiles|Квантили
Area|Область
Region|Область
Land|Суша
Ocean|Океан
Month|Месяц
Year|Год
Days|Дни
Day|День
Time|Время
Values|Значения
Value|Значение
Mean|Среднее
Weight|Вес
Sigma|Сигма
Min|Минимум
Max|Максимум
Iteration|Итерация
Epoch|Эпоха
Loss|Функция потерь
Empirical|Эмпирическая
Gaussian|Гауссовская
Student t|Стьюдента
Checkerboard|Шахматная
Bernstein|Бернштейна
Pointwise|Поточечный
Real|Реальные значения
Raw|Исходные данные
Color|Цвет
Marginal|Маргинальная плотность
Bins|Группы
Joins|Точки сшивки
Field|Поле
Levels|Уровни
Sensible|Явный поток
Latent|Скрытый поток
Flux|Тепловой поток
SST|ТПМ
Pressure|Давление
January|января
February|февраля
March|марта
April|апреля
May|мая
June|июня
July|июля
August|августа
September|сентября
October|октября
November|ноября
December|декабря
Jan|янв.
Feb|февр.
Mar|мар.
Apr|апр.
Jun|июнь
Jul|июль
Aug|авг.
Sep|сент.
Oct|окт.
Nov|нояб.
Dec|дек.
'''
# Historical labels and variants not used for the reverse direction.
_TO_EN_ALIASES = {
    "Гауссова копула": "Gaussian copula",
    "Средний год": "Annual climatology",
    "Поля океанографических переменных": "Oceanographic variables",
    "поток явного тепла": "Sensible heat flux",
    "поток скрытого тепла": "Latent heat flux",
    "Температура поверхности океана": "Sea surface temperature",
    "Значения явного потока": "Sensible flux values",
    "Значения скрытого потока": "Latent flux values",
    "плотность стационарного распределения": "Stationary density",
    "среднее": "Mean", "мин": "Min", "макс": "Max",
    "явного теплового потока": "sensible heat flux",
    "скрытого теплового потока": "latent heat flux",
    "явного потока": "sensible heat flux", "скрытого потока": "latent heat flux",
    "Вт м": "W m", "Вт/м²": "W/m²", "дБ": "dB",
    "ФП": "FP", "СДУ": "SDE", "Фоккера–Планка": "Fokker–Planck",
}
_TO_RU_ALIASES = {
    "mean year": "Средний год", "average": "среднее", "W m": "Вт м", "W/m²": "Вт/м²",
    "dB": "дБ", "EQ": "Экватор", "SDE": "СДУ", "FP": "ФП",
    "Fokker–Planck": "Фоккера–Планка", "Fokker-Planck": "Фоккера–Планка",
    "val": "валидация", "test": "тест", "train": "обучение",
}


def _validate(language):
    if language not in ("en", "ru"):
        raise ValueError(f"plot_language must be 'en' or 'ru', got {language!r}")
    return language


_DEFAULT = _validate(os.environ.get("OCEAN_PLOT_LANGUAGE", "en"))
_LOCAL = ContextVar("ocean_plot_language", default=None)


def get_plot_language():
    """Current rendering language (a temporary scope takes priority)."""
    return _LOCAL.get() or _DEFAULT


def set_plot_language(language="en"):
    """Set the process default; child processes inherit the same setting."""
    global _DEFAULT
    _DEFAULT = _validate(language)
    os.environ["OCEAN_PLOT_LANGUAGE"] = _DEFAULT
    return _DEFAULT


@contextmanager
def plot_language_context(language):
    """Temporary override, used by the existing explicit article-language API."""
    token = _LOCAL.set(_validate(language))
    try:
        yield
    finally:
        _LOCAL.reset(token)


def with_plot_language(function):
    """Honor a plotting function's explicit keyword-only language if supplied."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        if kwargs.get("language") is None:
            kwargs["language"] = get_plot_language()
        with plot_language_context(kwargs["language"]):
            return function(*args, **kwargs)
    return wrapped


def _dictionary(language):
    table = {}
    for line in _PAIRS.strip().splitlines():
        first, second = line.split("|", 1)
        source, dest = (second, first) if language == "en" else (first, second)
        table.setdefault(source.casefold(), dest)
    aliases = _TO_EN_ALIASES if language == "en" else _TO_RU_ALIASES
    table.update({source.casefold(): dest for source, dest in aliases.items()})
    # Exact token boundaries; never replace characters inside variable names,
    # identifiers (x1, rmse_EM), technical abbreviations or formulae.
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(
        re.escape(key) for key in sorted(table, key=len, reverse=True)
    ) + r")(?!\w)", flags=re.IGNORECASE)
    return table, pattern


_TABLES = {lang: _dictionary(lang) for lang in ("en", "ru")}
_MATH = re.compile(r"(\$\$.*?\$\$|(?<!\\)\$(?:\\.|[^$])*?(?<!\\)\$)", re.S)
# Geographic labels: check the complete tick label, not units such as '100 W'.
_RU_GEO = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*°?\s*([звсю])\.\s*([дш])\.\s*$", re.I)
_EN_GEO = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*°?([WENS])\s*$", re.I)


@lru_cache(maxsize=8192)
def _translate(value, language):
    if language == "en":
        if value.strip().casefold() in ("экватор", "экв."):
            return "EQ"
        geo = _RU_GEO.fullmatch(value)
        if geo:
            letter = {"з": "W", "в": "E", "с": "N", "ю": "S"}.get(geo[2].casefold())
            if letter:
                return f"{geo[1]}°{letter}"
        # English labels and formulae that are already localized need no work.
        if not re.search(r"[А-Яа-яЁё]", value):
            return value
    else:
        if value.strip() == "EQ":
            return "Экватор"
        geo = _EN_GEO.fullmatch(value)
        if geo:
            return f"{geo[1]}° " + {"W": "з. д.", "E": "в. д.", "N": "с. ш.", "S": "ю. ш."}[geo[2].upper()]
        if not re.search(r"[A-Za-z]", value):
            return value

    if language == "ru":
        value = re.sub(r"\blead (\d+) days?\b", r"заблаговременность \1 сут.", value, flags=re.I)
    table, regex = _TABLES[language]

    def replace(match):
        raw = match.group(0)
        result = table[raw.casefold()]
        # English month names are proper names even inside a date label.
        if language == "en" and result in ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December", "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"):
            return result
        # Preserve lower-case axis labels; do not lowercase acronyms (FP, SDE).
        if raw[0].islower() and not result.isupper():
            return result[0].lower() + result[1:]
        if raw[0].isupper() and result[0].islower():
            return result[0].upper() + result[1:]
        return result

    # Formula fragments, including LaTeX commands, are left verbatim.
    return "".join(part if i % 2 else regex.sub(replace, part)
                   for i, part in enumerate(_MATH.split(value)))


def translate_label(value, language=None):
    """Translate known scientific labels in either direction; leave other text."""
    if not isinstance(value, str) or not value:
        return value
    return _translate(value, _validate(language or get_plot_language()))


@contextmanager
def _localized_text(artist):
    # A temporary replacement allows both text drawing and layout measurement to
    # see the same string without changing get_text() after the operation. This
    # includes colorbars, subfigures, legends, annotations and dynamic tick labels.
    # Matplotlib Text stores the source string in _text; do not use set_text()
    # here, since it marks parent Axes/Figure stale and may schedule another draw.
    original = artist.get_text()
    translated = translate_label(original)
    if original == translated:
        yield
        return
    artist._text = translated
    try:
        yield
    finally:
        artist._text = original


def install_matplotlib_localization():
    """Install one idempotent rendering adapter for project Matplotlib figures.

    This deliberately adapts Text.draw and Text.get_window_extent at the process
    level so legacy modules with direct Figure.savefig calls and locators that
    regenerate ticks are also covered. It leaves numeric Artist data unchanged.
    """
    from matplotlib.text import Text
    if getattr(Text, "_ocean_language_installed", False):
        return

    def localize(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            with _localized_text(self):
                return method(self, *args, **kwargs)
        return wrapped

    Text.draw = localize(Text.draw)
    Text.get_window_extent = localize(Text.get_window_extent)
    Text._ocean_language_installed = True
