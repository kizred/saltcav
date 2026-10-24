# saltcav

Стохастическая модель скважинного выщелачивания соляных камер (NaCl): рост камеры
с учётом стратификации рассола и анизотропии растворения, Монте-Карло по
неопределённости геологии и режима, оценка устойчивости с учётом ползучести соли.

## Документы

| Документ | Что в нём |
| --- | --- |
| [Паспорт работы](docs/passport.md) | Цели, физика, модель, данные, дорожная карта, риски |
| [План разработки ПО](docs/software-plan.md) | Стек, архитектура, этапы с критериями приёмки |
| [Прогресс и заметки](docs/progress.md) | Текущее состояние, журнал работ, открытые вопросы |
| [Журнал решений](docs/decisions.md) | Принятые решения с обоснованием и альтернативами |
| [Похожие проекты и литература](docs/literature.md) | Аналоги (SANSMIC, WinUbro, SafeInCave…), данные для валидации, статьи со ссылками |

Статус и следующие шаги — в [docs/progress.md](docs/progress.md).

## Установка

```bash
git clone https://github.com/kizred/saltcav.git
cd saltcav
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

Дополнительные группы: `.[fast]` (Numba), `.[stochastic]`, `.[io]`, `.[viz]`, `.[gui]`
(окно просмотра), `.[mechanics]`.
FEniCSx (этап 6) ставится через conda-forge:

```bash
conda install -c conda-forge fenics-dolfinx mpich
```

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
python benchmarks/sansmic/compare.py   # сверка с SANSMIC
```

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
| `src/saltcav/mesh`, `mechanics` | Gmsh, FEniCSx, ползучесть | 6 |
| `src/saltcav/surrogate` | суррогат устойчивости | 7 |
| `src/saltcav/io`, `calibration` | данные и калибровка | 8 |
| `src/saltcav/viz`, `cli.py` | визуализация, командная строка | 9 |
| `configs/` | YAML-конфигурации | — |
| `schema/` | JSON Schema конфигурации для редактора (`saltcav schema`) | — |
| `benchmarks/sansmic` | сверка с SANSMIC: сценарий, эталон, сравнение | 2.4 |
| `data/` | данные компании (вне git) | 8 |
| `docs/` | паспорт, план ПО, прогресс, журнал решений | — |

## Правила

- Внутри кода только СИ; единицы промысла переводятся в `saltcav.core.units`.
- Изменение без теста не принимается.
- Данные компании в git не коммитятся.
- Важные решения записываются в `docs/decisions.md`.
