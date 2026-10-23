# Похожие проекты и литература

Что уже сделано в мире по моделированию выщелачивания соляных камер и что из этого берём.
Ссылки проверены на момент составления (октябрь 2026). Описания — краткий пересказ, не
цитаты; за деталями — к первоисточнику.

**Приоритет:** ★★★ — читать в первую очередь, напрямую влияет на код или калибровку;
★★ — полезно для конкретного этапа; ★ — фон.

## 1. Похожие программы и модели

| Проект | Что это | Чем полезен нам | Приоритет |
| --- | --- | --- | --- |
| **SANSMIC** (Sandia) — [GitHub](https://github.com/sandialabs/sansmic), [PyPI](https://pypi.org/project/sansmic/) | Открытый код моделирования выщелачивания камер; используется Sandia для камер нефтехранилища SPR. Режимы: отбор, прямоток, противоток, выщелачивание с заполнением | Бенчмарк этапа 2.4 (ворота Г2): осесимметричная постановка, плюмовая модель струи, стратификация. **Внимание:** в 1.0.7 метрический режим (M_CM_M3) даёт неверные объём и радиус — считать в ft/in/bbl | ★★★ |
| SANSMIC: [руководство пользователя](https://www.sandia.gov/research/publications/details/sansmic-user-manual-2016-09-01/) (Roberts, Weber, Rudeen, 2016) | Описание входных данных, форматов и пример задачи; обновляет документацию 1983 г. | Пример задачи для сверки; формат входа | ★★★ |
| SANSMIC: [design document](https://www.osti.gov/biblio/1222452) (SAND2015-6334) | Математическая и численная модель «как построено», включая подвижную границу нефть–рассол; физика не менялась с работ Russo 1981/1983 | Уравнения струи и перемешивания для этапа 2.4 | ★★★ |
| HORSMIC — [руководство](https://www.sandia.gov/research/publications/details/a-users-manual-for-the-computer-code-horsmic-1994-01-01/) (Sandia, 1994) | Вариант SANSMIC для горизонтальных камер | Если понадобятся многоскважинные/горизонтальные схемы | ★ |
| **WinUbro / UBRO** (Chemkop, Польша) — [Urbańczyk, 2018](https://yadda.icm.edu.pl/baztech/element/bwmeta1.element.baztech-b1de1bab-3e3f-4ebc-85d2-518d187897d2/c/Urbanczyk_Model_numeryczny_14_2018.pdf) | Численная модель выщелачивания: одномерный поток, гравитационное равновесие вместо турбулентности, явная схема с противопоточной аппроксимацией. Лежит в основе UBRO/WinUbro | Близкая к нашей постановка — сравнить допущения и схему; цикл статей в Przegląd Solny | ★★★ |
| WinUbro на практике — [Saygun, 2008](https://open.metu.edu.tr/bitstream/handle/11511/17641/index.pdf) (магистерская, METU) | Моделирование начальной камеры для троны: скорости растворения по керну (горизонтальная и вертикальная), влияние нерастворимых, мощности слоя и температуры | Как измерять скорость растворения на керне под углами — для физмодели | ★★ |
| **SCI-FIE** — [Li, Gracie, Dusseault, 2026](https://www.earthdoc.org/content/papers/10.3997/2214-4609.202681023) (EAGE) | Физическая модель течения рассола и растворения с переходом от адвекции к плотностной конвекции; асимметрия камер возникает даже в однородной соли из-за вихрей | Аргумент, почему 3D (этап 3) и стохастика нужны даже без неоднородности | ★★ |
| MESHFREE GFDM — [Michel et al., 2021](https://orbilu.uni.lu/bitstream/10993/53441/1/MichelSuchdeEtAl2021_SolutionMining.pdf) (Comput. Particle Mech.) | Бессеточный 3D-метод: течение, турбулентность k-ε, теплоперенос, перенос соли, подвижная стенка (ALE), двухскважинная схема | Эталон «тяжёлой» CFD-постановки; для оценки, где наша 1D-колонна грубее | ★★ |
| 3D-модель с неоднородностью — [Cen et al., 2024](https://www.mdpi.com/2227-9717/12/6/1124) (Processes) | Трёхмерная модель формирования камеры ПХГ с учётом неоднородности, прослоев, температуры и параметров закачки | Как учитывают прослои в 3D — для этапов 3–4 | ★★ |
| Камеры в соли с высоким содержанием нерастворимых — [Scientific Reports, 2018](https://dx.doi.org/10.1038/s41598-017-18546-w) | Математическая модель выщелачивания ПХГ в формациях с большим количеством нерастворимых | Сверить модель насыпи (этап 2.3): разрыхление, осаждение | ★★★ |
| **SafeInCave** (TU Delft) — [GitHub](https://github.com/ADMIRE-Public/SafeInCave), [о проекте](https://tudelft.nl/citg/over-faculteit/afdelingen/geoscience-engineering/research/research-themes/energy-transition/safeincave) | Открытый фреймворк для 3D-геомеханики соляных камер (хранение водорода), с учётом неопределённостей | Кандидат для этапа 6 вместо или вместе с собственной обвязкой FEniCSx — изучить до старта геомеханики | ★★★ |

## 2. Валидационные данные: камеры SPR (Sandia)

Ежегодные отчёты сравнивают расчёт SANSMIC с сонарными съёмками до и после выщелачивания.
Это открытые данные для проверки формы, пока нет данных компании.

| Отчёт | Что внутри | Приоритет |
| --- | --- | --- |
| [Cavern Leaching Monitoring CY18–CY19](https://sandia.gov/research/publications/details/cavern-leaching-monitoring-cy18-and-cy19-2021-03-01) | 6 камер с сонаром после продаж; расхождение у подошвы при её подъёме — процесс, которого нет в SANSMIC | ★★★ |
| [CY20](https://www.sandia.gov/research/publications/details/strategic-petroleum-reserve-cavern-leaching-monitoring-cy20-2021-06-01/) | 10 камер с сонаром; хорошо воспроизводятся полки и расширения; та же проблема подъёма подошвы | ★★ |
| [CY21](https://www.sandia.gov/research/publications/details/strategic-petroleum-reserve-cavern-leaching-monitoring-cy21-2022-09-01/) | 9 камер с сонаром до и после | ★★ |
| [CY22](https://www.sandia.gov/research/publications/details/strategic-petroleum-reserve-cavern-leaching-monitoring-cy22-2023-11-01/) | 22 камеры с сонаром до и после — самый большой набор | ★★★ |
| [CY23](https://www.sandia.gov/research/publications/details/strategic-petroleum-reserve-cavern-leaching-monitoring-cy23-2024-09-01/) | Переоценка формы там, где колонну перерезали посреди отбора | ★★ |
| [Bryan Mound 113, remedial leach](https://www.osti.gov/biblio/1096489) | Верх зоны выщелачивания рос быстрее проекта; расчёт SANSMIC против сонара, корректировка плана | ★★★ |
| [SPR remedial leach program 2013](https://www.osti.gov/biblio/1096260) | Сонар против SANSMIC при обрыве колонн | ★★ |
| [Parametric study of SANSMIC input](https://www.sandia.gov/research/publications/details/parametric-study-of-sansmic-input-and-resulting-impact-on-predicted-cavern-2021-11-01/) (Zeitler, Ross, 2021) | Чувствительность формы к входам; рекомендуют меньшие вертикальные ячейки для совпадения с сонаром | ★★★ — прямо про наши численные параметры |

**Для нас:** подъём подошвы нерастворимыми, который SANSMIC не учитывает, у нас уже есть
(этап 2.3) — это возможный вклад работы. Ограничение: SPR — соляные купола побережья
Мексиканского залива, не пластовая соль с прослоями.

## 3. Растворение соли и анизотропия k(ψ)

| Источник | О чём | Для чего | Приоритет |
| --- | --- | --- | --- |
| Durie, Jessen, 1964 — Mechanism of the Dissolution of Salt in the Formation of Underground Salt Cavities (SPE Journal) | Классика механизма растворения при создании камер | Основа закона растворения | ★★ |
| [Kazemi, Jessen, 1965](https://onemine.org/documents/reservoir-engineering-laboratory-research-mechanism-of-flow-and-controlled-dissolution-of-salt-in-solution-mining) | Лабораторные опыты: картины течения и распределение концентрации в сферической и радиальной моделях | Постановка физмодели, режимы колонн | ★★ |
| [Sears, Jessen, 1967](https://onemine.org/documents/reservoir-engineering-general-controlled-solution-mining-in-massive-salt) | Управляемое выщелачивание, программа размыва, роль нерастворителя | Режимы и ступени для примеров конфигурации | ★ |
| [Jessen, Total Solution Mechanism](https://onemine.org/documents/total-solution-mechanism-48d9fbbc-4a3b-4310-844f-45d0e41947f6-) | Скорость растворения при ламинарном и турбулентном течении, одно- и многоскважинные схемы | Обзор | ★ |
| **Cohen et al., 2020** — [arXiv](https://arxiv.org/abs/2004.09898), [дополнительные материалы](https://labo.msc.u-paris.fr/~berhanu//papers/Cohen2020_DissolutionSuppCorr.pdf) | Опыты: растворение наклонных блоков соли за счёт плавучести, скорость отступания от угла наклона и плотности раствора | Данные для k(ψ) и показателя n; образец постановки физмодели | ★★★ |
| [Energy, 2023 — угол поверхности](https://ideas.repec.org/a/eee/energy/v263y2023ipbs0360544222026780.html) | Модель конвекции и массопереноса (COMSOL) для 9 углов; отношение «вверх/вбок» 2 в отдельных расчётах и 1,3 при одновременном растворении | Почему roof_ratio — неопределённость (диапазон 1,3–2) | ★★★ |
| [Китаева, 2014](https://engjournal.bmstu.ru/articles/1250/1250.pdf) (Инженерный журнал) | Обзор упрощённых моделей: пограничный слой, модель ступени, коническая камера, идеальное вытеснение | Русскоязычный обзор допущений | ★★ |
| [РУДН, 2012](https://repository.rudn.ru/ru/records/article/record/37600) | Растворение каменной соли с выбросами газа, коэффициент массоотдачи | Если в соли есть газ | ★ |
| [Мозер, 2003](https://pmi.spmi.ru/pmi/article/jats/9073) (Записки Горного института) | Интенсификация растворения: заглублённая подача, насыщение растворителя CO₂ | Фон по технологии | ★ |

**Соглашение об угле:** в литературе 180° часто означает растворение вверх; у нас ψ = 0° —
кровля. При переносе коэффициентов угол пересчитывать (см. `decisions.md`).

## 4. Геомеханика и ползучесть соли

| Источник | О чём | Для чего | Приоритет |
| --- | --- | --- | --- |
| [Reedlunn, 2018 — Enhancements to the Munson-Dawson Model](https://sandia.gov/research/publications/details/enhancements-to-the-munson-dawson-model-for-rock-salt-2018-11-01) (SAND2018-12601) | Ползучесть при низких напряжениях, эквивалентное напряжение Хосфорда, быстрый неявный решатель; исходный код в приложении | Реализация закона ползучести в FEniCSx (этап 6) | ★★★ |
| [Munson-Dawson в FLAC3D (Itasca)](https://docs.itascasoftware.com/itasca970/common/models/munsondawson/doc/modelmunsondawson.html) | Документация реализации модели в коммерческом коде | Сверка параметров и с коммерческим пакетом | ★★ |
| [Joint Project III — сравнение моделей](https://www.sandia.gov/research/publications/details/joint-project-iii-on-the-comparison-of-constitutive-models-for-the-mechanic-2018-08-01/) (Sandia, 2018) | Немецко-американское сравнение определяющих моделей соли на общих задачах | Бенчмарки для проверки МКЭ | ★★★ |
| [WEIMOS: два геомеханических сценария](https://www.taylorfrancis.com/chapters/oa-edit/10.1201/9781003295808-39/weimos-simulations-two-geomechanical-scenarios-rock-salt-resembling-structures-wipp-hampel-l%C3%BCdeling-g%C3%BCnther-sun-kurczinski-wolters-d%C3%BCsterloh-lux-yildirim-zapf-wacker-epkenhans-stahlmann-reedlunn) (открытый доступ) | Продолжение сравнения моделей на задачах, похожих на WIPP | Бенчмарки | ★★ |
| [Honório et al., 2026](https://repository.tudelft.nl/record/uuid:5e637fdb-d7ce-4f40-97e3-76ab6093d4d1) (CMAME) | Стабилизированная смешанная формулировка МКЭ для почти несжимаемой ползучести на линейных тетраэдрах | Если на тетраэдрах будут осцилляции — готовое решение | ★★ |
| [Горное эхо, 2020, № 2](https://journal.permsc.ru/index.php/echo/article/download/echo2020n2p6/1678/3034) (ГИ УрО РАН) | Калибровка геомеханической модели по оседаниям (Верхнекамское и Саскачеван), приконтурные породы глубоких горизонтов, напряжения по ГИС | Отечественная практика и параметры по соли Верхнекамья | ★★ |

## 5. Вероятностные подходы

| Источник | О чём | Для чего | Приоритет |
| --- | --- | --- | --- |
| [Probabilistic analysis of a gas storage cavity in spatially random rock salt](https://upcommons.upc.edu/entities/publication/de0306b2-ff07-4fb6-b480-34f4e60dd212) (CIMNE) | Случайные поля (разложение Карунена–Лоэва), закон BGRa, глобальная чувствительность, subset simulation для вероятности отказа | Ближе всего к нашей стохастике, но для механики; методы генерации полей и оценки малых вероятностей | ★★★ |
| [Brunetta et al., 2025](https://rpsonline.com.sg/proceedings/isgsr2025/html/P184.html) (ISGSR) | Метамодель POD + RBF по 200 расчётам Abaqus с Munson–Dawson, Монте-Карло для вероятности отказа | Образец суррогата устойчивости (этап 7) | ★★★ |
| [Zeitler, Ross, 2021](https://www.sandia.gov/research/publications/details/parametric-study-of-sansmic-input-and-resulting-impact-on-predicted-cavern-2021-11-01/) | Параметрическое исследование SANSMIC | Какие входы сильнее всего влияют на форму | ★★ |

**Наша ниша:** вероятностные работы относятся к механике готовой камеры; стохастический
прогноз самой формы при выщелачивании (Монте-Карло по геологии и режиму) в найденных
источниках не встречается. Это стоит проверить отдельным поиском перед публикацией.

## 6. Инструменты

| Инструмент | Для чего | Этап |
| --- | --- | --- |
| [FEniCSx](https://fenicsproject.org) | МКЭ: ползучесть, напряжения | 6 |
| [Gmsh](https://gmsh.info) | Сетка из R(z, φ) | 6 |
| [GSTools](https://geostat-framework.org) | Коррелированные случайные поля | 5 |
| [SALib](https://salib.readthedocs.io) | Индексы Соболя, чувствительность | 5 |
| [PyMC](https://www.pymc.io) | Байесовская калибровка | 8 |
| [SANSMIC](https://github.com/sandialabs/sansmic) | Бенчмарк выщелачивания | 2.4 |
| [SafeInCave](https://github.com/ADMIRE-Public/SafeInCave) | Возможная основа геомеханики | 6 |

## Что взять в работу сейчас

1. **Этап 2.4:** design document и руководство SANSMIC — модель струи и пример задачи для
   сверки.
2. **Калибровка k(ψ):** Cohen et al. 2020 и статья Energy 2023 — диапазон roof_ratio и n.
3. **Нерастворимые:** статья Scientific Reports 2018 — сверить нашу модель насыпи.
4. **Перед этапом 6:** посмотреть SafeInCave — возможно, не придётся писать обвязку
   FEniCSx с нуля.

## Пополнение

Новую работу добавлять в нужный раздел таблицей: ссылка, о чём (своими словами), для чего
нам, приоритет. Если источник меняет решение — запись в `decisions.md`.
