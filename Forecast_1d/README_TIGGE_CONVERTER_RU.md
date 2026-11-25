# ECMWF TIGGE — проверка и преобразование GRIB для проекта Forecast_1d

Проверено на реальном GRIB, скачанном с [ECMWF TIGGE](https://ecds.ecmwf.int/datasets/tigge-forecasts?tab=download): выпуск 2026-10-01 00 UTC, `Control forecast`, пять полей, 61 шаг по 6 ч, полная область 0..80°N, 90..0°W, сетка 0,25°. Код **не меняет** исторические ERA5-массивы и обученные модели.

## Установка

Распаковать `Forecast_1d/convert_tigge_control.py` в `C:\Python projects\Ocean_components\Forecast_1d\`. Нужно `numpy` и `rasterio`:

```powershell
& C:\Python\python.exe -m pip install rasterio numpy
```

## Преобразование

Открыть PowerShell из корня проекта:

```powershell
cd 'C:\Python projects\Ocean_components'

& C:\Python\python.exe -m Forecast_1d.convert_tigge_control `
  --grib 'D:\Nastya\Data\OceanFull\DATA\TIGGE\ecmwf_2026-10-01_00.grib' `
  --points-json 'D:\Nastya\Data\OceanFull\DATA\Forecast_1d\atlantic15_native_1979-2026\points.json' `
  --out-dir 'D:\Nastya\Data\OceanFull\DATA\Forecast_1d\tigge_control' `
  --full-grid
```

**Измени значение `--grib` на фактический путь к своему GRIB**. Если нужна только 15-точечная выборка, убери `--full-grid`.

## Выход

- `tigge_ecmwf_control_20261001_0000_points.npz`:
  - `forecast_6h`: `[61,15,5]`, полный интервал `0,6,...,360` ч;
  - `forecast_daily`: `[14,15,5]`, для UTC-дней **2–15 октября 2026**, соответствует forecast day 1..14;
  - `fields`: `['u10','v10','t2m','d2m','sp']`;
  - `units`: `['m/s','m/s','K','K','Pa']`;
  - `target_dates`, `valid_times_utc`, `forecast_hours`, координаты и ID точек.
- `tigge_ecmwf_control_20261001_0000_era5grid_daily.npz` (только с `--full-grid`):
  - `forecast_daily`: `[14,161,181,5]` на исходной ERA5-сетке 0,5°, без интерполяции (каждый второй узел сетки TIGGE);
  - `lat_deg`: убывают 80...0°, `lon_deg`: возрастают -90...0°.
- `tigge_ecmwf_control_20261001_0000_metadata.json`: происхождение, SHA256 GRIB, временной охват, перечень полей и схема усреднения.

**Время**: выпущенный 01.10.2026 00 UTC прогноз имеет `T+24,30,36,42 ч` для UTC-дня 02.10.2026 и `T+336,342,348,354 ч` для UTC-дня 15.10.2026. `T+360 ч` — это 16.10.2026 00 UTC и **не входит** в 14-й дневной средний.

**Внимание к температуре**: GDAL по умолчанию конвертирует GRIB K в °C при чтении. Скрипт возвращает t2m и d2m обратно в K. См. [GDAL GRIB documentation](https://gdal.org/en/stable/drivers/raster/grib.html).

**Физические потоки**: этот файл содержит только прогнозируемую атмосферу, без SST и без готовых sensible/latent. Для физически корректного bulk-подхода надо вычислять `U × ΔT` и `U × Δq` на каждом 6-часовом сроке, затем усреднять за сутки (это не то же самое, что перемножать суточные средние). SST на будущие сроки берётся из заранее выбранного прогноза, без подстановки реанализа будущих дат.

**Следующий реальный прогноз**: сравнивать TIGGE + исторически прогнозируемую SST с наблюдаемыми ERA5 потоками на тех же 14 целевых UTC-днях, включая Seasonal AR(1) и 8-канальную GRU. Данные TIGGE не являются ERA5 и могут иметь систематические смещения, поэтому заранее нужно выбрать метод калибровки на train/validation, не используя test в подгонке.
