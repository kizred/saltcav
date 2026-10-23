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

Дополнительные группы: `.[fast]` (Numba), `.[stochastic]`, `.[io]`, `.[viz]`, `.[mechanics]`.
FEniCSx (этап 6) ставится через conda-forge:

```bash
conda install -c conda-forge fenics-dolfinx mpich
```

## Проверка

```bash
pytest                       # все быстрые тесты
saltcav check configs/example.yaml
saltcav run configs/example.yaml   # прогон одной камеры
```

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
| `benchmarks/` | SANSMIC, профили SPR | 2+ |
| `data/` | данные компании (вне git) | 8 |
| `docs/` | паспорт, план ПО, прогресс, журнал решений | — |

## Правила

- Внутри кода только СИ; единицы промысла переводятся в `saltcav.core.units`.
- Изменение без теста не принимается.
- Данные компании в git не коммитятся.
- Важные решения записываются в `docs/decisions.md`.
