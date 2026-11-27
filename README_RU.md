# Ocean_components

Научный программный комплекс для обработки реанализа ERA5, оценки коэффициентов стохастических моделей, анализа стационарных распределений, копул и пространственных спектральных характеристик, а также прогнозирования переменных системы океан — атмосфера в Северной Атлантике.

Основная модель — стохастическое дифференциальное уравнение Ито: \(dX_t=a(t,X_t)dt+b(t,X_t)dW_t\), где \(a\) — снос, \(b\) — диффузия. Используются явный и скрытый тепловые потоки (`sensible`, `latent`), суммарный поток (`Flux`), температура поверхности моря (`SST`) и поверхностное давление (`Pressure`). Метка периода 1979–2026 не гарантирует наличие всех наблюдений за 2026 год.

Англоязычное руководство: [README.md](README.md). Описание изменений: [REFACTORING_V2.md](REFACTORING_V2.md).

## 1. Структура проекта

| Каталог или файл | Назначение |
|---|---|
| `project_paths.py` | Общие пути к данным и коду. |
| `main.py` | Запуск численных расчётов и обработки данных. |
| `main_plotter.py` | Визуализация готовых результатов. |
| `Coefficients/` | Непараметрическая и полупараметрическая оценка сноса и диффузии, межпериодные переходы, сборка массивов, BBT и производные C/F/FS. |
| `Data_processing/` | Чтение GRIB, подготовка и объединение данных, среднегодовые поля, тренды, условия СДУ. |
| `Eigenvalues/eigenvalues.py` | Спектральные расчёты, собственные значения и векторы, сингулярные моды и сборка результатов по периодам. |
| `Functional/` | Стационарные плотности, гибридные хвосты, копулы, двумерное уравнение Фоккера — Планка. |
| `Plotting/` | Карты, спектральные графики, распределения, диагностика, видео. |
| `Forecast_nn/` | Пространственные нейросетевые и генеративные модели, в том числе с условиями СДУ. |
| `Forecast_1d/` | Точечные прогнозы, базовые и нейросетевые модели, исторические признаки и аудит. |
| `Forecast_probs/graph.py` | Прежний графовый вероятностный прогноз. |
| `scripts/`, `tests/` | Диагностика и тестирование. |

В Refactoring 2.0 объединены тематически близкие модули. Структура каталогов сохранена; точные пути удалённых и перенесённых модулей приведены в отчёте. `Cluster/` — локальный каталог Slurm/Docker, исключённый из Git; `Forecast_1d/jobs/` — отслеживаемые Git скрипты запуска.

## 2. Данные и установка

В `project_paths.py` выберите один активный `DATA_ROOT`. Это внешний каталог с ERA5, коэффициентами, результатами расчётов, моделями и рисунками, а не каталог исходного кода. Существующие каталоги данных, включая `Coefficients/continuous/`, сохраняются; их не нужно удалять при рефакторинге.

```bash
python -m venv .venv
# Activate the virtual environment before installing dependencies.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m project_paths --check
```

В Linux/macOS виртуальное окружение активируется командой `source .venv/bin/activate`, в Windows PowerShell — `.venv\Scripts\Activate.ps1`.

NumPy 2 или новее необходим для `numpy.trapezoid`. PyTorch устанавливается отдельно для подходящей CPU/CUDA-среды по [официальной инструкции](https://pytorch.org/get-started/locally/); нужен для `Forecast_nn` и нейросетевых моделей `Forecast_1d`. Для GRIB/TIGGE могут потребоваться совместимые нативные библиотеки ecCodes и GDAL, а для загрузки TIGGE — учётные данные CDS/ECDS. Для тестирования установите `pytest`. На кластере используйте существующий совместимый NVIDIA-контейнер.

## 3. Численные расчёты

В `main.py` задаются `task`, период, переменные и настройки выбранного алгоритма. В поставляемой конфигурации активно `fp_followup_validate`; перед запуском проверьте нужную задачу.

| Задачи | Назначение |
|---|---|
| `coefficients`, `bridge_coefficients`, `bbt`, `validate_bbt`, `derived_cf` | Коэффициенты СДУ и производные характеристики. |
| `eigenvalues`, `collect_eigenvalues` | Спектральные моды. |
| `mean_year`, `trends`, `components`, `sde_conditions` | Среднегодовые поля, тренды, смеси и условия СДУ. |
| `stationary`, `stationary_spatial` | Стационарные плотности и пространственная диагностика. |
| `copulas`, `copula_quintiles` | Копулы и совместные вероятности квантильных классов (`Functional/copula.py`). |
| `fokker_planck_2d_prepare`, `fokker_planck_2d_solve`, `fp2d_validate`, `fp_followup_validate` | Подготовка, решение и валидация двумерной задачи Фоккера — Планка. |
| `quantile_diagnostics`, `quantile_diagnostics_1d`, `quantile_hourly`, `quantile_hourly_resampling` | Диагностика квантилей и временного/пространственного прореживания. |
| `grib`, `grib-check`, `collect`, `extract_1d_series`, `extract_1d_features` | Подготовка данных. |

`start_year = 19792026` обозначает полный период 1979–2026; для отдельного десятилетия выбирается начальный год. Последняя дата определяется фактически доступными наблюдениями.

```bash
python main.py
python -m Data_processing.grib --help
```

## 4. Визуализация и язык рисунков

`main_plotter.py` строит рисунки по готовым массивам. В поставляемой конфигурации выбрана задача `videos`. Установите нужный `task`, например `stationary`, `stationary_hybrid`, `stationary_spatial`, `stationary_isolines`, `stationary_regions`, `stationary_classes`, `copulas`, `components`, `components_diagnostics`, `trends`, `fokker_planck_2d_compare` или `hybrid_fp2d_marginal_overlay`.

В начале `main()` выберите язык:

```python
# plot_language = "en"
plot_language = "ru"
set_plot_language(plot_language)
```

`en` — английский, `ru` — русский. Для отдельных CLI язык задаётся переменной окружения `OCEAN_PLOT_LANGUAGE` (по умолчанию `en`). Например:

```bash
OCEAN_PLOT_LANGUAGE=ru python -m Forecast_1d.plotting.plot_results --model sarima --split test
```

В PowerShell:

```powershell
$env:OCEAN_PLOT_LANGUAGE = 'ru'
python -m Forecast_1d.plotting.plot_results --model sarima --split test
```

В Python используйте `set_plot_language` или локальный контекст `plot_language_context` из `Plotting.language`. Локализация Matplotlib применяется к известным заголовкам, легендам, подписям, шкалам и аннотациям при отрисовке. Формулы, данные, числовые значения и имена моделей не изменяются. Произвольные новые фразы вне словаря `_PAIRS` могут остаться без перевода; Plotly-графики этим адаптером не локализуются. Русские строки интерфейса сохранены, комментарии и docstrings исходников — английские.

Рисунки записываются в `DATA_ROOT/Plots/`. Смена языка может перезаписать файл с тем же именем. Для сохранения двух языковых вариантов копируйте первый результат перед вторым запуском. Готовые изображения и видео автоматически не переводятся.

## 5. Прогнозирование

Пространственные модели: [Forecast_nn/README.md](Forecast_nn/README.md), [генеративные модели](Forecast_nn/GENERATIVE_README.md).

```bash
python -m Forecast_nn.check_sources
python -m Forecast_nn.train --model simvp --variable sensible --amp
python -m Forecast_nn.test --model simvp --variable sensible --split val
```

Точечные модели: [Forecast_1d/README_FORECASTING.md](Forecast_1d/README_FORECASTING.md). По умолчанию `points.json` содержит 15 точек; `points_10.json` сохраняет конфигурацию из десяти точек. Модели персистентности, среднего и сезонного цикла находятся в `models/baselines.py`, продвинутые нейросетевые архитектуры — в `models/advanced.py`.

```bash
python -m Forecast_1d.extract_points --dry-run
python -m Forecast_1d.extract_points
python -m Forecast_1d.train --model sarima
python -m Forecast_1d.test --model all --split val
python -m Forecast_1d.audit --help
```

Единый модуль аудита поддерживает режимы `future-weather`, `oracle-drivers`, `predictability`, `sst-vs-atmosphere`, `training`. Команда `python -m Forecast_1d.audit <mode> --help` показывает прежние настройки соответствующего эксперимента. Сценарии с будущими наблюдениями ERA5 являются диагностикой с использованием известного будущего, а не оперативным прогнозом.

Старый графовый эксперимент запускается через `python -m Forecast_probs.graph` и требует отдельного массива `DATA/FLUX_mini_diff.npy` и маски `DATA/mask`.

## 6. Проверка

```bash
python -m pip install pytest
python -m pytest tests Forecast_nn/tests Forecast_1d/tests -q
```

В репозитории есть тесты на синтетических данных, не требующие полной ERA5. Известные ограничения и фактический результат проверок Refactoring 2.0 приведены в [отчёте](REFACTORING_V2.md). Полный расчёт ERA5 и обучение на GPU следует проверять на рабочей машине или кластере.

## 7. Публикации и документация

Список научных публикаций с DOI сохранён в [английском README](README.md#12-related-publications). Документация: [PEP 257](https://peps.python.org/pep-0257/), [NumPy](https://numpy.org/doc/stable/), [Matplotlib](https://matplotlib.org/stable/), [PyTorch](https://pytorch.org/docs/stable/), [ecCodes](https://confluence.ecmwf.int/display/ECC/ecCodes+Home).
