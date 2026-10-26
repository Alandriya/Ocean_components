# Прогноз sensible + latent: DDPM, GAN, EOF–VAR и SDE-loss для SimVP

Этот **патч накладывается на рабочий каталог `EM_ocean/`** и не содержит `config.py`.
Оставьте собственные Linux-пути в `Forecast_nn/config.py` без изменений.
Код использует PyTorch, NumPy и уже существующие зависимости Forecast_nn, **не требует `openpyxl`, `skimage` или `xarray`**.

## 1. Содержание

- `Forecast_nn/models/generative.py`: условная **DDPM** с U-Net-денойзером и **conditional GAN** с PatchCritic. Прогнозируется совместный остаток от последней карты на все 3 дня; условием служат 30 дней истории, маска и пространственные координаты. Оба метода генерируют многомерные ансамбли целых трехдневных траекторий, а не независимые карты.
- `Forecast_nn/models/eof_var.py`: **EOF/PCA + ridge Vector Autoregression VAR(p)**. EOF оцениваются только по train; VAR обучается на ежедневных коэффициентах только train; ансамбль генерируется из ковариации ошибок train.
- `Forecast_nn/train_generative.py`, `test_generative.py`, `generative_utils.py`: обучение, хронологическое разбиение, `best.pth`, `last.pth`, логи в `Forecast/logs`, оценка ансамбля на validation/test и экспорт JSON.
- `Forecast_nn/joint_sde.py`: загрузчик полной исторической двумерной SDE, согласование во времени, расчёт D=BBᵀ, совместный Mahalanobis-loss и экспериментальная моментная регуляризация ансамбля.
- `Forecast_nn/train.py`: **добавлена возможность joint SDE-loss на обычной SimVP без дополнительного SDE-энкодера**. Обычные модели и предыдущие одномерные SDE-модели сохранены.
- `cluster_docker_id/*.sbatch`, `*.docker`: задания Slurm под базовый `nvcr.io/nvidia/pytorch:25.02-py3` через `docker-id`.
- `Forecast_nn/tests/test_generative_pipeline.py`: CPU smoke-тесты с синтетическими массивами.

## 2. Установка патча

На компьютере распаковать ZIP с сохранением дерева. На кластере в `EM_ocean/`:

```bash
cd /home/aosipova/EM_ocean
unzip -o Forecast_nn_generative_SDE_patch.zip
mkdir -p Forecast/logs
```

НЕ заменяйте локальный `Forecast_nn/config.py`: новая версия патча его не содержит.

## 3. Первые запуски через Docker/Slurm

```bash
sbatch cluster_docker_id/train_generative.sbatch ddpm joint
sbatch cluster_docker_id/train_generative.sbatch gan joint
sbatch cluster_docker_id/train_generative.sbatch var joint
```

Чтобы отличать выходные каталоги или менять гиперпараметры, редактируйте `cluster_docker_id/run_train_generative.docker`. Или запускайте из рабочего контейнера непосредственно:

```bash
python -u -m Forecast_nn.train_generative --model ddpm --variable joint --epochs 60 --width 32 --diffusion-steps 50 --batch-size 4 --amp
python -u -m Forecast_nn.train_generative --model gan --variable joint --epochs 60 --width 32 --batch-size 4 --amp
python -u -m Forecast_nn.train_generative --model var --variable joint --var-rank 24 --var-lags 5 --var-snapshot-stride 7
```

**Сначала использовать `--split val`:**

```bash
sbatch cluster_docker_id/test_generative.sbatch ddpm joint val
sbatch cluster_docker_id/test_generative.sbatch gan joint val
sbatch cluster_docker_id/test_generative.sbatch var joint val
```

Или:

```bash
python -u -m Forecast_nn.test_generative --model ddpm --variable joint --split val --members 8 --batch-size 2
```

Параметры модели в тестовом CLI по умолчанию такие же, как для команд выше; для другой ширины/числа шагов/ранга/лагов необходимо указать те же параметры или `--checkpoint /путь/к/best.pth`.

Итоговый тест на 2023 году запускать **только после выбора гиперпараметров** на validation 2021–2022.

### Метрики

В `Forecast/Results/ddpm/joint/<short_tag>/val/metrics_ddpm_joint.json` (аналогично GAN/VAR) записываются:

- `overall_rmse`, `overall_mae` и `overall_psnr` для **ансамблевого среднего**, в исходных физических единицах;
- `crps` — эмпирический маргинальный CRPS, усреднённый по всем океанским пикселям, каналам и горизонтам;
- `bivariate_energy_score` — 2D Energy Score совместного поля sensible–latent **в каждом океанском пикселе**, учитывающий междуканальную зависимость (но не полные пространственные зависимости);
- `spread`, `spread_skill_ratio`, MSE skill относительно persistence;
- метрики отдельно по горизонтам 1,2,3 и поканально.

**ВАЖНО:** обучающий/валидационный критерий DDPM — MSE предсказания случайного шума, а не RMSE итогового прогноза. Критерий GAN для выбора эпохи — MSE среднего двух сэмплов, а не adversarial loss. Конечные прогнозные метрики сравнивать только после `test_generative --split val`.

## 4. SimVP: SDE не во входе, а в loss

### Для sensible или latent, как это уже поддерживалось

```bash
python -u -m Forecast_nn.train --model simvp --variable sensible --sde-weight 1e-4 --epochs 70 --run-id sdeloss --amp
```

Это сохраняет **ту же базовую архитектуру SimVP**, но добавляет одно-компонентный SDE штраф. Убедитесь, что пути `cfg.sde_sources` верные. Для честного сравнения обучите такую же `simvp` с `--sde-weight 0` и тем же seed/режимом.

### Для joint sensible + latent: новая реализация

Нужны именно полные карты коэффициентов: A с двумя каналами и B с четырьмя компонентами **одной общей двумерной SDE**. Файлы `A1/B1` и `A2/B2` предыдущих одноканальных моделей не заменяют такую матрицу.

```bash
python -u -m Forecast_nn.train \
  --model simvp --variable joint \
  --sde-weight 1e-4 --joint-cov-floor 0.1 \
  --joint-a-file /home/aosipova/EM_ocean/Components/sensible-latent/A_joint.npy \
  --joint-b-file /home/aosipova/EM_ocean/Components/sensible-latent/B_joint.npy \
  --joint-b-layout row_major \
  --epochs 70 --run-id sdeloss --amp
```

**Имена `A_joint.npy` и `B_joint.npy` — ПРИМЕРЫ, не проверенные существующие файлы**; используй настоящие пути к сохраненным полным оценкам.

Допустимые `.npy` формы для A: `[time,2,H,W]`, `[time,H,W,2]`, `[29141,time,2]`, `[time,29141,2]` (также для уже прореженной сетки `[H,W]`). Для B то же самое с четырьмя компонентами. **Порядок четырёх компонент при `--joint-b-layout row_major`: `b11,b12,b21,b22`**. Опция `column_major` предполагает `b11,b21,b12,b22`. Проверь, как именно сохранены массивы из `semiparametric.py`: при неизвестной перестановке нужен явный конвертер. Файл элемента 0 = поток t=1 (02.01.1979), т. е. используется коэффициент A_t/B_t последнего наблюдённого дня t, **никаких будущих A_{t+1}/B_{t+1}**.

Детерминированная регуляризация имеет вид

```
L = MSE(pred, true) + dynamic_weight * L_delta
    + sde_weight * (1/2) mean_ocean [ (delta_1 - A_t*dt)^T
                         (B_t B_t^T*dt + cov_floor*I)^(-1)
                         (delta_1 - A_t*dt) ]
```

Здесь `delta_1` — предсказанный первый будущий шаг относительно последнего наблюдения, в нормированных train-единицах. Это **мягкое априорное предположение о сохранении исторически оценённого локального дрейфа**, а не точная дискретизация будущей SDE. В частности, дисперсию **одному детерминированному прогнозу навязать нельзя**. Ковариация используется только как вес ошибки дрейфа.

Провести абляцию `sde_weight = 0, 1e-5, 1e-4, 1e-3` на validation; следить за `RMSE`, `delta_correlation`, skill. Рекомендую `--run-id` для предотвращения коллизий настроек, не закодированных в коротком имени.

## 5. GAN с информированием ансамблевых моментов (экспериментальная опция)

```bash
python -u -m Forecast_nn.train_generative --model gan --variable joint \
  --sde-moment-weight 1e-5 --sde-cov-weight 0.01 \
  --sde-joint-a-file /путь/A_joint.npy --sde-joint-b-file /путь/B_joint.npy \
  --run-id sdemom --epochs 60 --amp
```

Дополнительная loss-функция сравнивает среднее 4 генераций первого будущего приращения с `A_t dt`, а эмпирическую ковариацию между sensible/latent с `B_t B_tᵀ dt`. **Оценка ковариации из 4 членов шумная**; коэффициент должен быть маленьким и подбираться на validation. Для DDPM такая моментная регуляризация не включена: дифференцировать десятки шагов генерации в каждом batch дорого. Для DDPM следующим информированием лучше сделать отдельную ветку с каузальными признаками `(A_t,D_t)` или score-guidance, но это отдельный эксперимент.

## 6. Как информировать VAR

Текущий VAR намеренно **без SDE**, чтобы быть базовой контрольной моделью. Осмысленный следующий вариант:

- При обучении добавлять проекции **прошлых** `A_t` как экзогенные регрессоры в EOF-коэффициентном пространстве (VARX), никогда `A_{t+1}`.
- Условную ковариацию инноваций в EOF-пространстве корректировать проектируемой `D_t=B_t B_tᵀ` (с усадкой к ковариации ошибок VAR, оценённой по train).
- Проверить всё это на validation, чтобы не предполагать автоматический выигрыш.

Такого SDE-VARX в патче **нет**; есть полная работающая неинформированная EOF-VAR.

## 7. Почему сначала ансамбли, а потом SDE

Существующие результаты: SimVP sensible val RMSE ≈ 22.66 Вт/м², Swin-GRU sensible ≈ 22.83; обе модели сглаживают изменения. Генеративные модели могут увеличить реалистичность отдельных траекторий **не снижая RMSE отдельных членов ансамбля**. Поэтому сравнивать надо RMSE **среднего ансамбля**, CRPS, Energy Score, spread/skill и динамику, а не RMSE произвольного одного сэмпла.

В первую очередь: базовая DDPM joint, GAN joint и EOF-VAR joint на validation. Затем SDE-loss SimVP joint и GAN с моментным штрафом. При неудаче стоит переходить к SST с теми же строгими границами времени и baseline.

## 8. Проверка

```bash
python -m unittest Forecast_nn.tests.test_generative_pipeline -v
```

Тесты включают прямой/обратный проход DDPM/GAN, covariance BBᵀ, реальные циклы train/val/sample для DDPM/GAN/VAR, совместную SimVP с SDE-loss, сохранение JSON/checkpoint. Проверено локально на CPU/синтетике; **не проверено на реальном массиве полного SDE и на GPU кластера**.

## Источники архитектур

- Ho, Jain & Abbeel (2020), *Denoising Diffusion Probabilistic Models*, NeurIPS: https://arxiv.org/abs/2006.11239
- Nichol & Dhariwal (2021), *Improved Denoising Diffusion Probabilistic Models*: https://arxiv.org/abs/2102.09672
- Mirza & Osindero (2014), *Conditional Generative Adversarial Nets*: https://arxiv.org/abs/1411.1784
- Isola et al. (2017), *Image-to-Image Translation with Conditional Adversarial Networks*: https://openaccess.thecvf.com/content_cvpr_2017/html/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.html
- Gneiting & Raftery (2007), *Strictly Proper Scoring Rules, Prediction, and Estimation*: https://doi.org/10.1198/016214506000001437
