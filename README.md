# saltcav

Стохастическая модель скважинного выщелачивания соляных камер (NaCl): рост камеры
с учётом стратификации рассола и анизотропии растворения, Монте-Карло по
неопределённости геологии и режима, оценка устойчивости с учётом ползучести соли.

## Документы

| Документ | Что в нём |
| --- | --- |
| [Руководство пользователя](docs/user-guide.md) | Установка, первый расчёт, работа в окне, отчёт, консоль |
| [Формат данных скважины](docs/data-format.md) | CSV: режим, рассол, сонар, прослои |
| [Изменения](CHANGELOG.md) | Что вошло в версии |
| [Паспорт работы](docs/passport.md) | Цели, физика, модель, данные, дорожная карта, риски |
| [План разработки ПО](docs/software-plan.md) | Стек, архитектура, этапы с критериями приёмки |
| [Прогресс и заметки](docs/progress.md) | Текущее состояние, журнал работ, открытые вопросы |
| [Журнал решений](docs/decisions.md) | Принятые решения с обоснованием и альтернативами |
| [Похожие проекты и литература](docs/literature.md) | Аналоги (SANSMIC, WinUbro, SafeInCave…), данные для валидации, статьи со ссылками |

Статус и следующие шаги — в [docs/progress.md](docs/progress.md).

## Установка

**Готовое приложение (Windows):** скачайте `saltcav-<версия>-setup.exe` со страницы
релизов репозитория (или из артефактов сборки «Сборка приложения» в Actions) и
запустите. Python не нужен, интернет не нужен; права администратора — не обязательно.
Примеры появятся в «Документы\saltcav». Проверка после установки: «Пуск» → командная
строка → `"%LOCALAPPDATA%\Programs\saltcav\saltcav.exe" selftest` (или просто
`saltcav selftest`, если при установке отмечено «добавить в PATH»).

**Из исходников (для разработки):**

```bash
git clone https://github.com/kizred/saltcav.git
cd saltcav
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements/lock-1.0.txt   # проверенные версии зависимостей 1.0
pip install -e ".[gui,mechanics,surrogate,calibration]"
saltcav shortcut                   # ярлык со значком на рабочем столе и в меню «Пуск»
saltcav gui
```

Интернет нужен только для установки (скачать пакеты); сама программа работает без
сети — все расчёты идут на этом компьютере, данные никуда не отправляются.

Для разработки — ещё `pip install -e ".[dev]"` и `pre-commit install`. Руководство
пользователя — [docs/user-guide.md](docs/user-guide.md) (то же — «Справка» в окне).

Дополнительные группы: `.[fast]` (Numba), `.[stochastic]`, `.[io]`, `.[viz]`, `.[gui]`
(окно просмотра), `.[mechanics]` (Gmsh, PARDISO), `.[surrogate]` (scikit-learn).

## Окно программы

```bash
saltcav gui configs/demo_calibration.yaml     # или ярлык saltcav-gui
```

Вкладки: **Параметры** (формы по разделам, схема скважины, ошибки с подсветкой) →
**Выщелачивание** (просмотр по шагам) → **Ансамбль** (огибающие, таблица реализаций) →
**Геомеханика** (разрез массива по годам) → **Устойчивость** (вероятность по ансамблю) →
**Калибровка** (по данным скважины). Кнопки: Проверить (Ctrl+K), Рассчитать (F5),
Ансамбль (F6), Геомеханика (F7), Калибровка (F8), Отчёт (Ctrl+P), Консоль Python
(Ctrl+`). Файл — тот же YAML, что и для командной строки.

## Проверка

```bash
pytest                       # все быстрые тесты
saltcav check configs/example.yaml
saltcav run configs/example.yaml   # прогон одной камеры
saltcav run configs/example_3d.yaml --out hist.npz   # прогон с сохранением шагов
saltcav view hist.npz                  # окно просмотра по шагам (нужен .[gui])
saltcav view hist.npz --field rate --animation a.gif   # анимация без окна
saltcav view hist.npz --paraview pv/series.pvd          # серия для ParaView
saltcav ensemble configs/example_ensemble.yaml -n 200 --jobs 0 --plot env.png   # ансамбль
saltcav view configs/example_ensemble.yaml --realization 14 -n 200   # реализация ансамбля
saltcav mechanics configs/example_mechanics.yaml --out mech.npz --plot mech.png   # геомеханика
saltcav mechanics hist.npz --years 30 --plot mech.png   # по сохранённой форме
saltcav mechanics h3d.npz --model 3d --plot mech3d.png   # полная 3D-форма (минуты)
saltcav surrogate configs/example_ensemble.yaml -n 200 --plot stab.png   # P(потери устойчивости)
saltcav data data/demo/well-1                                       # проверить данные скважины
saltcav report configs/example_ensemble.yaml --out report.html --all  # отчёт (минуты)
saltcav calibrate configs/demo_calibration.yaml --data data/demo/well-1 --plot calib.png \
        --write-config calibrated.yaml                              # калибровка → ансамбль
python benchmarks/sansmic/compare.py   # сверка с SANSMIC
```

## Python API и консоль

Всё, что делают окно и командная строка, доступно из Python — для серий расчётов,
своих графиков, Jupyter:

```python
import saltcav.api as sc

cfg = sc.load("configs/example.yaml")
cfg = sc.set(cfg, {"leaching.k0_m_per_s": 2e-5, "schedule.stages.0.rate_m3_per_h": 60})
res = sc.run(cfg)  # выщелачивание
print(sc.table(res))  # объём, радиус, рассол по времени
sc.plot(res, "leach.png")
ens = sc.ensemble(cfg, n=200)  # Монте-Карло; mechanics, stability, calibrate — так же
```

В окне — «Вид → Консоль Python» (Ctrl+`) и «Файл → Выполнить скрипт…» (Ctrl+R): там же
`cfg` — параметры из формы, `show(obj)` — результат на вкладку, `apply(cfg)` — в форму.
Примеры — `examples/`.

## Конфигурация в редакторе

Расчёт задаётся YAML-файлом (примеры — в `configs/`). Для подсказок и проверки прямо в
редакторе в начале файла указана схема:

```yaml
# $schema: ../schema/saltcav-config.schema.json
```

PyCharm и VS Code (с расширением YAML) тогда дописывают имена полей (Ctrl+Space),
показывают описание с единицами (Ctrl+Q / наведение) и подчёркивают ошибки: опечатку в
имени, значение вне диапазона, неизвестный режим. В своём файле путь к схеме пишется
относительно файла. Схема строится из модели конфигурации: после её изменения —
`saltcav schema` (тест проверяет, что схема в репозитории актуальна).

## Структура

| Путь | Назначение | Этап |
| --- | --- | --- |
| `src/saltcav/core` | единицы, конфигурация, генераторы случайных чисел | 0–1 |
| `src/saltcav/geometry` | R(z, φ), объём, нормали, углы ψ | 1 |
| `src/saltcav/leaching`, `operations` | модель выщелачивания и режим | 2–4 |
| `src/saltcav/stochastic` | случайные поля, ансамбли | 5 |
| `src/saltcav/mesh`, `mechanics` | сетка Gmsh, осесимметричный МКЭ, ползучесть, критерии | 6 |
| `src/saltcav/surrogate` | признаки формы, гауссов процесс, вероятность потери устойчивости | 7 |
| `src/saltcav/io`, `calibration` | данные и калибровка | 8 |
| `src/saltcav/viz`, `cli.py` | визуализация, командная строка | 9 |
| `configs/` | YAML-конфигурации | — |
| `schema/` | JSON Schema конфигурации для редактора (`saltcav schema`) | — |
| `benchmarks/sansmic` | сверка с SANSMIC: сценарий, эталон, сравнение | 2.4 |
| `data/` | данные компании (вне git); `demo/` — синтетика, `open/` — SANSMIC ([формат](docs/data-format.md)) | 8 |
| `docs/` | паспорт, план ПО, прогресс, журнал решений | — |

## Правила

- Внутри кода только СИ; единицы промысла переводятся в `saltcav.core.units`.
- Изменение без теста не принимается.
- Данные компании в git не коммитятся.
- Важные решения записываются в `docs/decisions.md`.

## Лицензия

© Sergei Kizred. Программа бесплатна, в том числе для коммерческого использования.
Изменение и распространение — только с письменного разрешения автора. Условия — в
[LICENSE.txt](LICENSE.txt).
