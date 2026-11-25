# Автоматическая загрузка ECMWF TIGGE через ECDS (Windows)

Веб-форма ECDS ограничивает выбор дат в интерфейсе; `cdsapi` позволяет не повторять действия вручную.
Этот файл работает **только с ECMWF control forecast 00 UTC** и сохраняет один GRIB на дату выпуска.
Не использует истинные будущие поля, это архив реально выпущенных прогнозов.

## 1. Авторизация

Войти на https://ecds.ecmwf.int/datasets/tigge-forecasts?tab=download и принять TIGGE licence.
Получить Personal Access Token по официальной инструкции: https://ecds.ecmwf.int/how-to-api

Создать файл `%USERPROFILE%\.cdsapirc` (обычный текст, без `.txt`):

```text
url: https://ecds.ecmwf.int/api
key: ВАШ_ЛИЧНЫЙ_ТОКЕН
```

Если у тебя другой `.cdsapirc` для Copernicus CDS, не перезаписывай его: можно сохранить ключ
в переменной окружения `ECDS_API_KEY`, скрипт принудительно использует URL ECDS. Не загружай ключ в GitHub.

Установка:

```powershell
C:\Python\python.exe -m pip install -U "cdsapi>=0.7.7"
```

## 2. Разместить файл

Скопировать `Forecast_1d/download_tigge_ecds.py` в:

`C:\Python projects\Ocean_components\Forecast_1d\download_tigge_ecds.py`

Из корня проекта:

```powershell
cd 'C:\Python projects\Ocean_components'
```

## 3. Проверка без загрузки

```powershell
C:\Python\python.exe -m Forecast_1d.download_tigge_ecds --max-dates 2
```

Скрипт печатает JSON запросов, но ничего не скачивает без `--execute`.

## 4. Сначала пилот, 2 последовательные даты

```powershell
C:\Python\python.exe -m Forecast_1d.download_tigge_ecds --start 2026-10-01 --end 2026-10-02 --execute
```

## 5. Даты test набора (351 дата инициализации)

```powershell
C:\Python\python.exe -m Forecast_1d.download_tigge_ecds --start 2025-10-30 --end 2026-10-15 --execute
```

**Осторожно:** диапазон включает 351 прогноз. При размере пилотного выпуска около 43 МБ
потребуется ориентировочно 15 ГБ свободного места. ECDS извлекает архивы с лент,
поэтому загрузка может быть длительной, запросы могут ждать в очереди.

Если запрос прерван, запускай точно ту же команду: GRIB с 305 корректными сообщениями
пропускаются; ошибочные файлы сохраняются как `.invalid` и загружаются повторно.

Чтобы сначала взять семь дат с интервалом в неделю:

```powershell
C:\Python\python.exe -m Forecast_1d.download_tigge_ecds --start 2026-09-01 --end 2026-10-15 --every 7 --execute
```

## 6. Состав и выход

`origin=ecmwf`, `forecast_type=control_forecast`, `level_type=single_level`,
`u10`, `v10`, `t2m`, `d2m`, `sp`, шаги `0,6,...,360`, область `[80,-90,0,0]`
в порядке **North/West/South/East**, сетка `0.25/0.25` и формат `grib`.

Файлы (пример):

`D:\Nastya\Data\OceanFull\DATA\Forecast_1d\TIGGE\raw\2026\10\tigge_ecmwf_control_20261001_0000.grib`

Для каждой даты 5 x 61 = 305 GRIB2 сообщений; скрипт проверяет структуру и число сообщений.
Это **не заменяет** научную проверку кодов полей, ориентации сетки и дат действия из
`Forecast_1d/convert_tigge_control.py` — применяй его к первым файлам.

Для другого каталога:

```powershell
C:\Python\python.exe -m Forecast_1d.download_tigge_ecds --start 2026-10-01 --end 2026-10-02 --out 'D:\TIGGE' --execute
```

Скрипт делает запросы по одному дню последовательно, без параллельной нагрузки и без
изменения исходных ERA5 или чекпойнтов. Результаты и проблемы записывает в
`download_log.jsonl`. При ошибке он останавливается, если не указан `--continue-on-error`.

## Источники ECMWF

- API ECDS: https://ecds.ecmwf.int/how-to-api
- TIGGE Dataset: https://ecds.ecmwf.int/datasets/tigge-forecasts?tab=download
- Справочник кодов ECDS TIGGE: https://confluence.ecmwf.int/spaces/DAC/pages/647861203/ecds-tigge.yaml
- Migration guide и синтаксис: https://confluence.ecmwf.int/spaces/DAC/pages/462894687/User+guide+for+migration+of+S2S+and+TIGGE+to+ECDS
- Разъяснение ECMWF по загрузке daily/monthly и лентам: https://forum.ecmwf.int/t/tigge-downloads-through-ecds-api-taking-too-long/15128

**Примечание:** код проверен локально без сетевых запросов; реальная ECDS авторизация и
доступность всех 351 архивных выпусков проверяются при запуске пользователем.
