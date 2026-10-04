# Статус внедрения GLM-5.3-Flash

Обновлено: **2026-10-04**, часовой пояс `Asia/Yekaterinburg`.
План: [GLM53_FLASH_IMPLEMENTATION_PLAN.md](GLM53_FLASH_IMPLEMENTATION_PLAN.md).
Исходные данные: [GLM53_FLASH_GGUF_INVENTORY.json](GLM53_FLASH_GGUF_INVENTORY.json).
Основная тестовая модель: [инспекция Uncensored-IQ3_XXS](GLM53_FLASH_IQ3_XXS_INSPECTION.json).

Этот файл хранит фактический прогресс и точку продолжения для следующего агента.
План определяет объём работ и критерии готовности; статус фиксирует, что уже сделано,
чем это проверено и что делать дальше. Пример в конце файла не является выполненной работой.

## Текущее состояние

| Поле | Значение |
|---|---|
| Общий статус | P0, P2 и P3 в работе: P0.1, статическая P0.2, scaffold P0.3a, P2.1a, runner P2.1b.1, P2.2–P2.4, P2.5a/b/c/d/e/f, P2.6 и reference P3.2a DONE в пределах проверок журнала; P1, P4–P6 не начаты |
| Последняя проверенная ревизия Strata | `444f442`: реализация из журнала уже закоммичена; последующий аудит и смена тестовой модели — в документации |
| Последняя выполненная работа | AUDIT-01: проверен текущий код и новый GGUF; 151 Python-тест, 3 Node-проверки, 1 protocol CTest прошли |
| Следующая задача | `P0.3b`: архив Unsloth с проверенным hash, сборка реальных llama/oracle/CUDA targets и проверка графов |
| Активная задача / исполнитель | Нет; требуется продолжение P0.3b |
| Блокеры | Старый блокер коммита снят фактом наличия `444f442`. Для P0.3b ещё нужно подтвердить архив/хеш и реальную сборку; прежняя ошибка HTTP сохранена в истории, в этом аудите сеть не проверялась |
| Основная тестовая модель | `H:\GLM-5.3-Flash-GGUF\GLM-5.3-Flash-Uncensored-IQ3_XXS.gguf` |
| Дополнительный профиль | `H:\GLM-5.3-Flash-GGUF\UD-Q3_K_XL`; прежние отчёты сохраняются отдельно |
| Стенд | Windows, Ryzen 9 9950X, 128 ГиБ RAM, RTX 5090 32 ГиБ |
| GLM backend / setup | `--check-only` готов; `backends/glm5next/` содержит CMake/tokenizer scaffold, inference backend и создание профиля отсутствуют |
| Закреплённая зависимость GLM | Для аудита выбран Unsloth `86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9`; сборка/хеш архива/production pin ещё не выполнены |
| Последняя проверенная конфигурация запуска GLM | Нет, модель не запускалась |
| Измеренная скорость GLM / пиковая RAM / VRAM | Не измерены |

Новый основной GGUF проверен 2026-10-04: **1 файл, 1412 тензоров, 112,310 ГиБ**,
45 основных блоков и 1 MTP; имена/формы прошли существующий loader contract.
Основная модель: 109,687 ГиБ весов, из них 102,322 ГиБ экспертов; MTP: 2,614 ГиБ,
из них 2,443 ГиБ экспертов. Полный хеш и содержимое весовых payloads не проверялись.

Вопреки имени `IQ3_XXS`, фактические routed types — IQ2_S/IQ3_S/IQ4_XS и
Q2_K/Q3_K для MTP. Среди остальных весов есть BF16. Проверка CUDA должна учитывать
новый набор типов. Шаблон отличается четырьмя заменами `[0]` → `.0`; 18 rendered
prompts и 72 проверки IDs/round-trip совпали с прежним Python frontend.
[Сравнение файлов](GLM53_FLASH_IQ3_XXS_COMPARISON.json),
[локальные проверки](GLM53_FLASH_IQ3_XXS_LOCAL_CHECKS.json).
Это не независимый tokenizer oracle и не запуск модели.

Прежний UD-Q3_K_XL: 4 части, 1412 тензоров, 137,404 ГиБ; metadata-only первая
часть и проверка её хеша описаны в исходной инвентаризации. Его
[GLM53_FLASH_INSPECTION.json](GLM53_FLASH_INSPECTION.json) и старые планы чтения
относятся только к UD-Q3_K_XL. Не использовать их offsets для нового одиночного файла.

При каждом возобновлении сверять эту сводку с рабочим деревом: данные файла могут
отставать от изменений, сделанных другим агентом или пользователем.

## Этапы

| Этап | Статус | Подтверждённый результат | Что осталось для завершения |
|---|---|---|---|
| Подготовка | DONE | План и инвентаризация сохранены | — |
| P0. Совместимость и эталон | IN_PROGRESS | P0.1/P0.2 проверены на обеих моделях; scaffold P0.3a DONE, protocol CTest повторно прошёл | Реальная сборка кандидата, хеш архива, trace MTP off, GPU fixtures и tokenizer oracle |
| P1. Основной GPU engine | TODO | Нет | Запуск с подгрузкой матриц, GPU-аудит, baseline и память |
| P2. Токенизация и API | IN_PROGRESS | P2.1a/P2.1b.1/P2.2–P2.4/P2.5a,b,c,d,e,f/P2.6 DONE на fixtures/mock; повторная общая проверка 151 Python + 3 Node, 72 локальные проверки нового GGUF | Реальная tokenizer parity, template oracle, runtime backend selection и фактический INFO, HTTP и полная модель |
| P3. Конвейер и кэш | IN_PROGRESS | P3.2a DONE: reference byte-range/cache-key contract; fixture-тесты и 1032 матрицы нового профиля проверены по заголовкам | Runtime-интеграция, асинхронная доставка, cache parity, отмена и измерение перекрытия |
| P4. Сессии | TODO | Нет | Полный hybrid state, архивы, restore, A → B → A |
| P5. Native MTP | TODO | В GGUF присутствуют веса; исполнения MTP нет | Draft/verify/rollback, sampling, сессии и A/B скорости |
| P6. Замеры и выпуск профиля | TODO | Нет | Воспроизводимые замеры, регрессии Qwen/DeepSeek, setup и документация |

## Правила заполнения для агента

1. Перед работой прочитать план, эту сводку и последние записи журнала; проверить
   `git status --short` и текущий HEAD. Не перезаписывать чужие изменения.
2. Выбрать конкретный пункт плана, присвоить ему ID вида `P0.1` и записать название.
   ID обозначает подпункт, а не завершение всего этапа. Перед редактированием кода
   указать активную задачу, исполнителя и перевести соответствующий этап в `IN_PROGRESS`.
3. После законченного изменения или проверки добавить запись в журнал. Указывать
   фактические файлы, команды, результаты и ограничения. Обновлять статус также
   перед передачей работы, остановкой или переключением задачи.
4. Разделять «код написан», «собралось», «проверено на fixture», «проверено на полной
   модели» и «измерено ускорение». Если проверки ещё не выполнены, использовать `VERIFY`.
5. Отмечать `DONE` только при выполнении критерия конкретной задачи. Этап целиком
   получает `DONE`, когда выполнены все его обязательные пункты и критерий из плана.
   Тогда синхронизировать соответствующие чекбоксы плана.
6. Для проверки сохранять точную команду, рабочую директорию, exit code и результат.
   Если есть лог/JSON, давать ссылку на существующий файл. Отсутствующий лог или
   невыполненный тест так и обозначать; не подставлять ожидаемый результат вместо фактического.
7. Для замеров записывать commit и незакоммиченные изменения, SHA backend-зависимости,
   модель/квантование, контекст, batch/ubatch, cache budget, MTP, sampling и состояние
   прогрева. Указывать число повторов, единицы, медиану/диапазон и исходные результаты.
   Скорость DeepSeek не считать измерением GLM; cache misses не считать чтениями SSD.
8. При препятствии записать точную ошибку, команду воспроизведения, уже проверенные
   варианты и условие разблокировки. Не называть обычную следующую задачу блокером.
9. Сохранять историю. При отмене изменения, регрессии или опровержении результата
   добавить новую запись со ссылкой на прежнюю, исправить сводку и чекбоксы плана.
10. Завершать обновление конкретным следующим действием, ожидаемым артефактом и
    способом проверки. Следующему агенту должно быть понятно, с какого файла начать.

| Статус | Значение |
|---|---|
| `TODO` | Исполнение задачи ещё не началось |
| `IN_PROGRESS` | Работа началась, критерий готовности ещё не выполнен |
| `VERIFY` | Реализация подготовлена, обязательная проверка ожидается или не пройдена |
| `BLOCKED` | Есть конкретное препятствие; указаны причина и условие продолжения |
| `DONE` | Критерий выполнен, приведено подтверждение |

Процент готовности не обязателен: таблица этапов и проверяемые результаты точнее.
Не оставлять активного исполнителя после завершения его работы; незаконченная задача
может оставаться `IN_PROGRESS` с пометкой «исполнитель: нет, требуется продолжение».

## Точка продолжения

**Следующая задача — P0.3b: сборка реального кандидата и проверка графов.** Начать с
[инструкции сборки](../../backends/glm5next/README.md) и
[отчёта P0.2](GLM53_FLASH_LOADER_COMPATIBILITY.md).

1. Получить исходники Unsloth `86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9` в отдельную
   директорию, зафиксировать хеш архива. Не менять зависимость DeepSeek.
2. Использовать scaffold `backends/glm5next/` с проверенным локальным архивом и
   SHA-256; собрать реальный tokenizer target. Сохранить build record и compiler/CUDA,
   SHA и патчи; сохранить исходные настройки точности из плана. Пройденный protocol
   CTest не является сборкой llama/oracle/CUDA.
3. На собранном коде подтвердить исключение блока 45 из обычного графа при MTP off.
   P0.2 подтвердил разделение по исходникам/заголовкам, без runtime trace.
4. Проверить KDA rollback, kpool, sparse attention и большие индексы; затем GPU
   fixtures и tokenizer oracle. Ранее HTTP из shell получил `WinError 10013`;
   это историческая ошибка, а не результат новой проверки доступности сети.
   Сначала проверить наличие локального архива/исходников выбранной ревизии.
5. Для oracle использовать основной файл
   `H:\GLM-5.3-Flash-GGUF\GLM-5.3-Flash-Uncensored-IQ3_XXS.gguf`
   и отдельный `GLM53_FLASH_IQ3_XXS_TOKENIZER_PARITY.json`; команда обновлена в README.
   Далее проверить CUDA для новых IQ2_S/IQ3_S/Q2_K и BF16, помимо общих типов.
   Повторная реализация уже готового инспектора или frontend не требуется.

**Ожидаемый результат:** воспроизводимая изолированная сборка по полному SHA,
хеш архива, точная команда и логи проверок. Весь P0 пока IN_PROGRESS.

## Подтверждённый журнал

Новые записи добавлять в конец этого раздела, перед разделом с шаблоном.

### PREP-01 — 2026-10-03 — Инвентаризация модели и план

- **Статус:** DONE, только подготовительный этап.
- **Что сделано:** разобраны заголовки четырёх локальных GGUF, подсчитаны веса основной
  модели и MTP, изучены точки интеграции Strata, подготовлены этапы P0–P6.
- **Артефакты:** [план](GLM53_FLASH_IMPLEMENTATION_PLAN.md) и
  [инвентаризация](GLM53_FLASH_GGUF_INVENTORY.json); теперь находятся в `docs/GLM53`.
- **Основание:** Strata `0561016`, модель по пути из сводки; полный SHA GLM backend
  ещё не выбран. Изменения engine в эту работу не входили.
- **Проверено:** в JSON зафиксированы успешное чтение заголовков, уникальность имён,
  совпадение объявленного числа тензоров и соответствие payload размерам файлов.
  SHA-256 первой части совпал с опубликованным; подробности и источник есть в плане.
- **Команда исходной инспекции:** отдельный воспроизводимый скрипт и полный лог
  команды не сохранены. Создать такой инструмент — задача P0.1; не выдавать эту
  запись за автоматический тест будущего loader.
- **Ограничения:** большие shard hashes не проверены, inference не запускалась,
  скорость/память/MTP не измерены. Наличие весов не означает готовую поддержку MTP.
- **Дальше:** выполнить P0.1 по разделу «Точка продолжения».

### P0.1-01 — 2026-10-04 00:46, Asia/Yekaterinburg — Инспектор GLM GGUF

- **Статус:** DONE, только подпункт P0.1. Исполнитель: Codex; работа завершена.
- **Изменение:** добавлен `tools/setup_glm5next.py --check-only` с JSON-отчётом
  (`--output`, `--tensor-details`). Проверяются split-номера/число частей/число тензоров,
  уникальность имён, обязательные metadata и tokenizer IDs, геометрия quant rows,
  выравнивание, пересечения и границы payload. Проверяются наличие всех блоков,
  базовые формы embedding/output, FFN/router и NextN и отделение MTP от основной модели.
  Metadata-only shard не требует padding до `data_start`. Профиль запуска не создаётся.
- **Общий reader:** `tools/gguf_reader.py` теперь сохраняет `header_end`, отклоняет
  усечённые строки (включая последний metadata value), повторные metadata keys и
  недопустимое число измерений. Данные весов не читаются.
- **Файлы:** `tools/setup_glm5next.py`, `tools/test_setup_glm5next.py`,
  `tools/gguf_reader.py`, план/статус и [JSON инспекции](GLM53_FLASH_INSPECTION.json).
- **Ревизия:** база `24dd8802b54b65f7fa5e515655d9956a672511bb`, ветка `dev`;
  перечисленные изменения проверялись до их коммита. Backend-зависимость не требуется.
- **Проверки:** рабочая директория для всех команд `C:\work\git\my-repos\Strata`.
  - `python tools/test_setup_glm5next.py` — exit 0, 12 тестов. Fixtures покрывают
    metadata-only без padding, все пять экспертных quant types, отсутствующий shard,
    усечённый payload/header, дубликаты, split/count, metadata, NextN, неправильные
    shapes/rank/row geometry/types, пересечения/выравнивание/границы, unsplit и CLI JSON/error.
  - `python tools/setup_glm5next.py --model-dir H:/GLM-5.3-Flash-GGUF/UD-Q3_K_XL --check-only --output docs/GLM53/GLM53_FLASH_INSPECTION.json`
    — exit 0. Реальная модель: 4 части, 1412 тензоров, 46 блоков = 45 + 1;
    147535921955 байт файлов, 147526401400 байт весов. Первая часть заканчивается
    на 9429859, `data_start=9429888`: отсутствие 29 байт принято корректно.
    Размеры частей, tensor type counts и main/MTP weight/expert bytes совпали с исходной инвентаризацией.
  - `python -m unittest tools.test_deepseek4 tools.test_shards` — exit 1 при импорте:
    в системном Python нет `regex` и `numpy`. Повторено существующим проектным Python:
    `.venv/Scripts/python.exe -m unittest tools.test_deepseek4 tools.test_shards`
    — exit 0, 16 тестов. Были `ResourceWarning` об unclosed file в неизменённом
    `strata_pack.py:351`; падений тестов нет.
  - Отдельные текстовые логи тестов не сохранялись; результат полной инспекции сохранён в JSON выше.
- **Сравнение с критерием:** P0.1 выполнен на fixtures и заголовках всей локальной
  модели; первый чекбокс P0 отмечен. Этап P0 целиком не завершён.
- **Не проверено:** SHA-256 payload, полный набор shapes/types по выбранному loader,
  сборка backend, GPU/inference, tokenizer oracle, скорость, память и работа MTP.
  Инспектор не подтверждает эти свойства; исходные GGUF не изменялись.
- **Коммит / ограничение среды:** `git -c safe.directory=C:/work/git/my-repos/Strata add -- tools/gguf_reader.py tools/setup_glm5next.py tools/test_setup_glm5next.py docs/GLM53/GLM53_FLASH_IMPLEMENTATION_PLAN.md docs/GLM53/GLM53_FLASH_IMPLEMENTATION_STATUS.md docs/GLM53/GLM53_FLASH_INSPECTION.json`
  завершился ошибкой `fatal: Unable to create 'C:/work/git/my-repos/Strata/.git/index.lock': Permission denied`.
  `.git` доступен среде только для чтения, повышение прав запрещено. Коммит не создан;
  изменения сохранены в рабочем дереве `dev`. Для коммита нужна среда с правом записи
  в `.git`; после восстановления доступа добавить перечисленные шесть файлов и выполнить
  `git commit -m "Add GLM GGUF admission inspector and validation fixtures"`.
- **Следующий шаг:** сохранить изменения P0.1 коммитом, затем P0.2 — таблица
  соответствия loader по разделу «Точка продолжения»; полный SHA зависимости ещё не выбран.

### P0.2-01 — 2026-10-04 01:02, Asia/Yekaterinburg — Статическое сопоставление loader

- **Статус:** DONE для сопоставления исходников и GGUF; весь P0 не завершён.
  Исполнитель: Codex, цикл 2; работа завершена.
- **Изменение:** `tools/glm5next_loader_contract.py` и флаг `--loader-contract`
  проверяют все имена/формы полного GLM Flash: KDA, DSA/MLA, mHC, indexer,
  FFN/router/shared/routed experts и NextN. Учитываются необязательные MTP
  embedding/head и концевые единичные измерения GGML. Неизвестные/пропущенные
  имена, несовместимые shapes и некорректные metadata отклоняются.
- **Основание:** Unsloth llama.cpp `86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9`;
  ссылки на исходники и таблица семейств — в
  [GLM53_FLASH_LOADER_COMPATIBILITY.md](GLM53_FLASH_LOADER_COMPATIBILITY.md).
  Это SHA кандидата аудита, не pin проверенной сборки. Исходники прочитаны через
  веб-инструмент; загрузка из Python urllib получила `WinError 10013`.
- **Артефакты:** новый модуль, `tools/test_glm5next_loader_contract.py`,
  `tools/fixtures/glm5next_loader_headers.json`, флаг в `tools/setup_glm5next.py`,
  [отчёт JSON](GLM53_FLASH_LOADER_CHECK.json), документ совместимости, план/статус.
  Fixture содержит 89 описаний тензоров из реальных блоков 0/3/45 с перенумерацией,
  без payload и без генерации ожидаемых форм из проверяемого кода.
- **Ревизия Strata:** `24dd8802b54b65f7fa5e515655d9956a672511bb`, `dev`; прежние
  незакоммиченные файлы P0.1 сохранены. Проверялось рабочее дерево с P0.1 и P0.2.
- **Проверки:** рабочая директория `C:\work\git\my-repos\Strata`.
  - `python tools/setup_glm5next.py --model-dir H:/GLM-5.3-Flash-GGUF/UD-Q3_K_XL --check-only --loader-contract --output docs/GLM53/GLM53_FLASH_LOADER_CHECK.json`
    — exit 0, сопоставлены все 1412 тензоров; основная модель 1383, MTP 29.
    Main graph: блоки 0–44, 34 KDA и 11 DSA; MTP weights: блок 45.
  - `.venv/Scripts/python.exe -m unittest tools.test_glm5next_loader_contract tools.test_setup_glm5next tools.test_deepseek4 tools.test_shards`
    — exit 0, 37 тестов, в том числе 9 новых. Сохранился прежний `ResourceWarning`
    об unclosed file из неизменённого `strata_pack.py:351`; падений нет.
  - `git -c safe.directory=C:/work/git/my-repos/Strata diff --check` — exit 0.
    Отдельные текстовые логи не сохранены; JSON полной проверки сохранён по ссылке выше.
- **Граница проверки:** сопоставление статическое; поддержка quant ops на CUDA,
  фактическая загрузка/trace графа, logits, rollback, tokenizer oracle и скорость
  не проверялись. Частичные модели и fused QKV вне профиля данного контракта.
  Production pin, хеш архива и патчи сборки относятся к P0.3.
- **Коммит:** повторная команда
  `git -c safe.directory=C:/work/git/my-repos/Strata add -- tools/gguf_reader.py tools/setup_glm5next.py tools/test_setup_glm5next.py tools/glm5next_loader_contract.py tools/test_glm5next_loader_contract.py tools/fixtures/glm5next_loader_headers.json docs/GLM53/GLM53_FLASH_IMPLEMENTATION_PLAN.md docs/GLM53/GLM53_FLASH_IMPLEMENTATION_STATUS.md docs/GLM53/GLM53_FLASH_INSPECTION.json docs/GLM53/GLM53_FLASH_LOADER_CHECK.json docs/GLM53/GLM53_FLASH_LOADER_COMPATIBILITY.md`
  — exit 1: `fatal: Unable to create 'C:/work/git/my-repos/Strata/.git/index.lock': Permission denied`.
  Коммит не создан. Все 11 файлов P0.1/P0.2 остаются в рабочем дереве `dev`;
  для коммита нужна среда с записью в `.git`. После успешного `git add` выполнить
  `git commit -m "Add GLM GGUF inspector and audited loader contract"`.
- **Следующий шаг:** P0.3 по разделу «Точка продолжения».

### P2.2-01 — 2026-10-04, Asia/Yekaterinburg — Встроенный шаблон GLM

- **Статус:** DONE для адаптера шаблона и fixture-проверок. Этап P2 остаётся
  IN_PROGRESS; API и inference oracle ещё не подключены.
- **Исполнитель:** Codex. Ветка `dev`, база
  `24dd8802b54b65f7fa5e515655d9956a672511bb`; изменения P0.1/P0.2 сохранены.
- **Выбор задачи:** исходников изолированной зависимости GLM локально нет;
  ограничения загрузки для P0.3 из предыдущей записи сохраняются. Выполнен
  независимый второй пункт P2, не требующий GPU или внешних загрузок.
- **Изменение:** `serve/glm5next.py` использует существующую sandbox Jinja-среду
  с `loopcontrols` и исходный шаблон модели. Сохраняет IDs инструментов и историю
  вызывающего кода, преобразует JSON arguments в mapping, заменяет отсутствующий
  content на пустую строку. Принимает low/high/max и булев clear_thinking;
  несовместимые настройки (включая enable_thinking/xhigh) дают TemplateRequestError.
- **Артефакты:** `serve/glm5next.py`, `serve/test_glm5next.py`,
  `serve/fixtures/glm53_chat_template.jinja`, `serve/fixtures/README.md`,
  `.gitattributes`, план и статус. Fixture — точные UTF-8 байты metadata
  `tokenizer.chat_template` первой локальной части GGUF, 10648 байт,
  SHA-256 `a4fddbbf0b432101a296c17094f8bc5a2b0d30713b5b5cd92f86be78511aa724`.
  Инструкция воспроизведения извлечения — в README; чтение только заголовка.
- **Проверки:** рабочая директория `C:\work\git\my-repos\Strata`, Python из
  `.venv`, Jinja2 3.1.6.
  - `.venv/Scripts/python.exe -m unittest serve.test_glm5next` — exit 0, 10 тестов.
    Проверены точные многоходовые prompts, префикс `[gMASK]<sop>`, generation
    prompt, Unicode, effort, clear_thinking с сохранением текущего tool turn,
    типы аргументов, пустой content, несколько вызовов, сортировка результатов
    по IDs (включая batch outputs), сохранение порядка при неоднозначных IDs,
    wrapped/bare schemas и deferred tool references. Ветки с `break` исполняются.
  - `.venv/Scripts/python.exe -m unittest serve.test_glm5next serve.test_deepseek tools.test_glm5next_loader_contract tools.test_setup_glm5next tools.test_deepseek4 tools.test_shards`
    — exit 0, 54 теста. Прежний ResourceWarning из `strata_pack.py:351` остаётся;
    вывод MockEngine в тестах не является измерением скорости модели.
  - `git -c safe.directory=C:/work/git/my-repos/Strata diff --check` — exit 0.
    Отдельный лог не сохранён.
- **Граница результата:** GLMTemplate пока не выбирается сервером. Общий
  `openai_to_messages` удаляет tool IDs и объединяет high/max в xhigh; для GLM
  нужна отдельная нормализация. Runtime setup должен извлекать шаблон выбранного
  GGUF, fixture не служит fallback. Token IDs, parser, окончания генерации,
  оба API, работа полной модели и скорость не проверялись. Поэтому весь P2 и
  пункт настроек API не отмечены DONE.
- **Коммит:** команды
  `git -c safe.directory=C:/work/git/my-repos/Strata add -- .gitattributes serve/glm5next.py serve/test_glm5next.py serve/fixtures/glm53_chat_template.jinja serve/fixtures/README.md docs/GLM53/GLM53_FLASH_IMPLEMENTATION_PLAN.md docs/GLM53/GLM53_FLASH_IMPLEMENTATION_STATUS.md`
  и `git -c safe.directory=C:/work/git/my-repos/Strata commit -m "Add GLM embedded chat template adapter and fixtures"`
  обе завершились exit 1:
  `fatal: Unable to create 'C:/work/git/my-repos/Strata/.git/index.lock': Permission denied`.
  Коммит не создан: `.git` доступен только для чтения в текущей среде.
  Все изменения сохранены в рабочем дереве `dev`; для коммита нужна запись в `.git`.
- **Следующий шаг:** P0.3 по «Точке продолжения». Пока исходники недоступны,
  независимая задача — P2.3: GLM output parser в `serve/glm5next.py` и проверки
  разрывов `<think>`/`<tool_call>`/`<arg_key>`/`<arg_value>` во всех позициях chunks.

### P2.3-01 — 2026-10-04, Asia/Yekaterinburg — GLM streaming parser

- **Статус:** DONE для parser и fixture-проверок третьего пункта P2. Весь этап
  остаётся IN_PROGRESS, интеграция с endpoints и полная модель не проверены.
- **Исполнитель:** Codex. Ветка `dev`, база
  `24dd8802b54b65f7fa5e515655d9956a672511bb`; прежние изменения сохранены.
- **Изменение:** в `serve/glm5next.py` добавлен `GLMOutputParser` и выбран через
  `GLMTemplate.output_parser`. Используется общий интерфейс Event/ToolCall:
  reasoning, content, tool_call. Поддержаны reasoning из открытого prompt `<think>`
  и явно повторённый стартовый тег, несколько вызовов, `<arg_key>/<arg_value>`,
  bare/wrapped schemas, JSON arguments и сохранение строк по declared type.
  Пробелы/переносы текста сохраняются. Дубликаты аргументов, повреждённые и
  незакрытые вызовы возвращаются как текст без executable events.
- **Файлы:** `serve/glm5next.py`, `serve/test_glm5next.py`,
  `serve/fixtures/README.md`, план и статус. Формат основан на локальном
  embedded template из P2.2; дополнительных зависимостей нет.
- **Проверки:** рабочая директория `C:\work\git\my-repos\Strata`.
  - `.venv/Scripts/python.exe -m unittest serve.test_glm5next` — exit 0,
    22 теста (12 новых). Для streaming проверены все позиции разделения на
    два chunks и ширины 1/2/3/7/19, включая Unicode и маркеры. Все обрезанные
    префиксы вызова возвращены как текст. Проверены literal tags в строковых
    аргументах, JSON-looking strings по schema, malformed calls, NaN/Infinity/
    overflow для declared JSON, разные IDs нескольких вызовов, повторный finish.
    Цикл template → parsed call → tool result → continuation проверен на fixture.
  - `.venv/Scripts/python.exe -m unittest serve.test_glm5next serve.test_deepseek tools.test_glm5next_loader_contract tools.test_setup_glm5next tools.test_deepseek4 tools.test_shards`
    — exit 0, 66 тестов. Сохраняется прежний ResourceWarning из
    `strata_pack.py:351`; сообщения MockEngine не являются замером скорости.
  - `git -c safe.directory=C:/work/git/my-repos/Strata diff --check` — exit 0.
    Отдельный лог не сохранён.
- **Ограничения:** вызовы буферизуются до полной валидации даже с
  `stream_tools=True`; ранних tool_start/tool_args нет. Это parser формата,
  не полный JSON Schema validator. Для различения строки `true` и boolean
  нужен schema. Структурная последовательность `</arg_value>` + пробелы +
  `<arg_key>`/`</tool_call>` внутри raw string неоднозначна в исходном формате.
  Ответы инструментов сериализует шаблон; они не подаются в assistant output
  parser. Token-level EOS/EOT/EOM, серверная нормализация/dispatch и проверка
  на generated output полной модели остаются открытыми. Скорость не измерялась.
- **Коммит:** команды
  `git -c safe.directory=C:/work/git/my-repos/Strata add -- .gitattributes docs/GLM53 serve/fixtures serve/glm5next.py serve/test_glm5next.py tools/fixtures/glm5next_loader_headers.json tools/gguf_reader.py tools/glm5next_loader_contract.py tools/setup_glm5next.py tools/test_glm5next_loader_contract.py tools/test_setup_glm5next.py`
  и `git -c safe.directory=C:/work/git/my-repos/Strata commit -m "Add GLM compatibility checks and chat frontend fixtures"`
  обе завершились exit 1:
  `fatal: Unable to create 'C:/work/git/my-repos/Strata/.git/index.lock': Permission denied`.
  Коммит не создан; для завершения нужна среда с записью в `.git`. Изменения
  всех четырёх запусков остаются в рабочем дереве `dev`.
- **Следующий шаг:** P0.3 по «Точке продолжения». Независимый следующий шаг
  для frontend — P2.4: обработка EOS/EOT/EOM в Service/GLM backend contract,
  с mock-проверками finish reasons и отсутствия служебных маркеров в ответе.

### P2.4-01 — 2026-10-04, Asia/Yekaterinburg — EOS/EOT/EOM и завершения API

- **Статус:** DONE для Service и mock-проверок четвёртого пункта P2. Backend,
  HTTP dispatch и полная модель по-прежнему требуют интеграции/проверки.
- **Исполнитель:** Codex. Ветка `dev`, база
  `24dd8802b54b65f7fa5e515655d9956a672511bb`; предыдущие изменения сохранены.
- **Основание:** через GGUFFile повторно прочитаны metadata и соответствующие
  строки vocab первой локальной части: EOS 154820 = `<|endoftext|>`,
  EOT 154827 = `<|user|>`, EOM 154829 = `<|observation|>`. Payload не читался.
- **Изменение:** шаблоны GLM/DeepSeek объявляют `stop_token_keys`; Service
  читает IDs из tokenizer.special_ids, проверяет наличие, тип, неотрицательность
  и уникальность. Для GLM используются три ключа, для DeepSeek прежний EOS,
  для Qwen сохранён fallback. Выбор stop IDs больше не зависит от проверки
  `architecture == deepseek4`. Существующая остановка до detokenizer применена
  к трём GLM-границам; существующие serializers выбирают finish reason по
  завершённым calls, а не по одному EOM. Hardcoded IDs в runtime не добавлены.
- **Файлы:** `serve/glm5next.py`, `serve/deepseek.py`, `serve/server.py`,
  новый `serve/test_glm5next_service.py`, `serve/fixtures/README.md`, план и статус.
- **Проверки:** рабочая директория `C:\work\git\my-repos\Strata`.
  - `.venv/Scripts/python.exe -m unittest serve.test_glm5next_service serve.test_deepseek`
    — exit 0, 16 тестов, из них 9 новых GLM. Каждый EOS/EOT/EOM проверен в
    OpenAI и Anthropic streaming/collect: stop/end_turn для текста,
    tool_calls/tool_use для двух calls, корректный счётчик completion tokens,
    остановка до detokenizer, отсутствие хвоста после границы и закрытие generator.
    Дополнительно проверены EOM с обрезанным/повреждённым вызовом, length после
    полного call, ошибка metadata, IDs из metadata вместо констант, сохранение
    буквального текста маркеров в обычных токенах и Qwen fallback.
  - `.venv/Scripts/python.exe -m unittest serve.test_glm5next serve.test_glm5next_service serve.test_deepseek tools.test_glm5next_loader_contract tools.test_setup_glm5next tools.test_deepseek4 tools.test_shards`
    — exit 0, 75 тестов. Сохраняется прежний ResourceWarning из
    `strata_pack.py:351`; строки MockEngine не являются замерами производительности.
  - `git -c safe.directory=C:/work/git/my-repos/Strata diff --check` — exit 0.
    Отдельный лог не сохранён.
- **Граница результата:** tokenizer в тесте — byte mock с реальными stop IDs,
  не реализация glm4 и не oracle. Service создан явно с GLMTemplate; HTTP
  backend selection пока не включён. Будущий engine должен передавать stop ID
  серверу либо явную причину завершения, иначе завершившийся поток будет length.
  Проверки полной модели, GPU, скорости и вывода reasoning controls не выполнены.
- **Коммит:** команды
  `git -c safe.directory=C:/work/git/my-repos/Strata add -- .gitattributes docs/GLM53 serve/fixtures serve/glm5next.py serve/test_glm5next.py serve/test_glm5next_service.py serve/server.py serve/deepseek.py tools/fixtures/glm5next_loader_headers.json tools/gguf_reader.py tools/glm5next_loader_contract.py tools/setup_glm5next.py tools/test_glm5next_loader_contract.py tools/test_setup_glm5next.py`
  и `git -c safe.directory=C:/work/git/my-repos/Strata commit -m "Add GLM compatibility checks and frontend stop handling"`
  обе завершились exit 1:
  `fatal: Unable to create 'C:/work/git/my-repos/Strata/.git/index.lock': Permission denied`.
  Коммит не создан. Все изменения остаются в `dev`; для коммита нужна среда
  с записью в `.git`, которая сейчас read-only.
- **Следующий шаг:** P0.3 по «Точке продолжения». Независимая frontend-задача —
  нормализация OpenAI/Anthropic для GLM в `serve/glm5next.py` с сохранением
  tool-call IDs и low/high/max/clear_thinking; начать с общих нормализаторов
  `serve/frontend.py`, которые сейчас теряют GLM-значимые поля.

### P2.5a-01 — 2026-10-04, Asia/Yekaterinburg — Нормализация GLM API requests

- **Статус:** DONE для подпункта P2.5a: нормализаторы и dispatch через Service.
  Родительский P2.5 и пункт UI thinking остаются открытыми.
- **Исполнитель:** Codex. Ветка `dev`, база
  `24dd8802b54b65f7fa5e515655d9956a672511bb`; прежние изменения сохранены.
- **Изменение:** `openai_to_glm_messages` и `anthropic_to_glm_messages` сохраняют
  call/result IDs, JSON arguments, reasoning и различие low/high/max. Передают
  clear_thinking; chat_template_kwargs имеют приоритет. Developer становится
  system, поздние system не переписываются в user. Anthropic tool results
  сохраняют порядок, is_error передаётся модели через `Error: `. Исходный request
  не мутируется. Неподдерживаемые media/content blocks и malformed inputs дают
  ValueError/TemplateRequestError вместо потери содержимого.
- **Подключение:** GLMTemplate объявляет normalize_openai/normalize_anthropic;
  `Service.normalize_request` выбирает их через hooks. Оба generation handlers
  и Anthropic count handler используют этот метод. Для прочих шаблонов остаются
  существующие нормализаторы. Runtime выбор GLM в main пока не включён.
- **Настройки:** default max, clear_thinking=false. OpenAI принимает
  reasoning_effort/reasoning.effort, Anthropic output_config.effort; оба API —
  chat_template_kwargs с двумя GLM-настройками. Unsupported effort, enable_thinking,
  disabled thinking и budget-to-effort conversion отклоняются. Anthropic
  enabled/adaptive без budget принимаются; общий opt-in default не отключает
  GLM reasoning. Это не реализация отдельного thinking-budget механизма.
- **Файлы:** `serve/glm5next.py`, `serve/server.py`, новый
  `serve/test_glm5next_requests.py`, `serve/fixtures/README.md`, план и статус.
- **Проверки:** рабочая директория `C:\work\git\my-repos\Strata`.
  - `.venv/Scripts/python.exe -m unittest serve.test_glm5next_requests` — exit 0,
    11 тестов первого прогона до добавления двух дополнительных проверок.
  - `.venv/Scripts/python.exe -m unittest serve.test_glm5next_requests serve.test_glm5next serve.test_glm5next_service serve.test_deepseek tools.test_glm5next_loader_contract tools.test_setup_glm5next tools.test_deepseek4 tools.test_shards`
    — exit 0, 88 тестов, включая все 13 новых. Проверены IDs и сортировка
    tool results реальным embedded template, mixed block order, tool errors,
    Unicode/JSON arguments, отсутствие мутации, effort/clear_thinking и ошибки
    параметров, double-encoded lists, Service.prepare, fallback нормализаторов
    и Anthropic count handler без HTTP listener. Прежний ResourceWarning из
    `strata_pack.py:351` сохраняется, MockEngine не является замером скорости.
  - `git -c safe.directory=C:/work/git/my-repos/Strata diff --check` — exit 0.
    Отдельный лог не сохранён.
- **Граница результата:** проверены fixtures/mock и прямой вызов count handler.
  Нет проверки HTTP end-to-end, glm4 token oracle и полной модели. Shared settings
  и UI пока предлагают Qwen effort levels; max/clear_thinking в UI не добавлены.
  MCP round-trip пока теряет IDs в общем коде; native tool references в API
  adapter не поддержаны. GLM backend не зарегистрирован для запуска.
- **Коммит:** команды
  `git -c safe.directory=C:/work/git/my-repos/Strata add -- .gitattributes docs/GLM53 serve/fixtures serve/glm5next.py serve/test_glm5next.py serve/test_glm5next_service.py serve/test_glm5next_requests.py serve/server.py serve/deepseek.py tools/fixtures/glm5next_loader_headers.json tools/gguf_reader.py tools/glm5next_loader_contract.py tools/setup_glm5next.py tools/test_glm5next_loader_contract.py tools/test_setup_glm5next.py`
  и `git -c safe.directory=C:/work/git/my-repos/Strata commit -m "Add GLM compatibility checks and API request adapters"`
  обе завершились exit 1:
  `fatal: Unable to create 'C:/work/git/my-repos/Strata/.git/index.lock': Permission denied`.
  Коммит не создан, `.git` read-only. Изменения находятся в рабочем дереве `dev`.
- **Следующий шаг:** P0.3 по «Точке продолжения». Независимая задача P2.6 —
  capabilities для thinking settings: начать с `clean_shared_defaults`,
  `Service.with_shared` и формы настроек web app, добавить GLM low/high/max и
  clear_thinking без обещания отключить reasoning, проверить сохранение настроек.

### P2.6-01 — 2026-10-04, Asia/Yekaterinburg — Настройки thinking по capabilities

- **Статус:** DONE для шестого пункта P2 на mock/DOM. Весь P2 остаётся
  IN_PROGRESS, GLM runtime backend ещё не выбран при запуске.
- **Исполнитель:** Codex. Ветка `dev`, база
  `24dd8802b54b65f7fa5e515655d9956a672511bb`; прежние изменения сохранены.
- **Изменение:** GLMTemplate объявляет reasoning_capabilities: low/high/max,
  default max, clear_thinking. Service отдаёт их через health/settings,
  проверяет shared settings и восстановление файла с теми же capabilities.
  Собственные параметры запроса, включая false, имеют приоритет над shared.
  Для GLM shared effort применяется и при Anthropic thinking=adaptive/enabled;
  некорректный явный effort не подменяется default. Qwen validation сохранена.
- **Web:** Low/High/Max, переключатель Clear earlier thinking, отсутствие Off/
  Medium для GLM и подпись reasoning always on. Новые настройки отправляются
  в chat и shared defaults; старое неподдерживаемое значение заменяется default,
  новое окно без сохранённых настроек использует max. Reset учитывает модель.
- **Файлы:** `serve/glm5next.py`, `serve/server.py`, `serve/web/app.js`,
  `serve/web/index.html`, `serve/test_glm5next_settings.py`,
  `serve/test_glm5next_settings_ui.cjs`, `serve/fixtures/README.md`, план/статус.
- **Проверки:** рабочая директория `C:\work\git\my-repos\Strata`.
  - `.venv/Scripts/python.exe -m unittest serve.test_glm5next_settings` — exit 0,
    6 новых тестов: capabilities, ошибки без изменения сохранённого состояния,
    приоритет обоих API без мутации request, persistence/reload, Qwen fallback
    и прямой вызов POST settings handler с ответами 200/400.
  - `node serve/test_glm5next_settings_ui.cjs` — exit 0. Реальные функции app.js
    исполнены с DOM stand-in: видимые кнопки GLM/Qwen, initial max, миграция Off,
    сохранение High, clear switch и payload chat/shared для обеих моделей.
  - `node --check serve/web/app.js` — exit 0.
  - `.venv/Scripts/python.exe -m unittest serve.test_glm5next_settings serve.test_glm5next_requests serve.test_glm5next serve.test_glm5next_service serve.test_deepseek tools.test_glm5next_loader_contract tools.test_setup_glm5next tools.test_deepseek4 tools.test_shards`
    — exit 0, 94 теста. Прежний ResourceWarning из `strata_pack.py:351`
    сохраняется. MockEngine не является замером производительности.
  - `git -c safe.directory=C:/work/git/my-repos/Strata diff --check` — exit 0.
    Отдельный лог не сохранён.
- **Граница результата:** браузерная визуальная проверка, HTTP end-to-end с GLM,
  tokenizer oracle, полная модель и скорость не проверялись. Требуется GLMTemplate
  в Service; runtime backend selection остаётся частью P2.5. INFO/About/monitor
  и встроенный MCP round-trip также остаются открытыми.
- **Коммит:** команды
  `git -c safe.directory=C:/work/git/my-repos/Strata add -- .gitattributes docs/GLM53 serve/fixtures serve/glm5next.py serve/test_glm5next.py serve/test_glm5next_service.py serve/test_glm5next_requests.py serve/test_glm5next_settings.py serve/test_glm5next_settings_ui.cjs serve/server.py serve/deepseek.py serve/web/app.js serve/web/index.html tools/fixtures/glm5next_loader_headers.json tools/gguf_reader.py tools/glm5next_loader_contract.py tools/setup_glm5next.py tools/test_glm5next_loader_contract.py tools/test_setup_glm5next.py`
  и `git -c safe.directory=C:/work/git/my-repos/Strata commit -m "Add GLM frontend and capability-driven reasoning settings"`
  обе завершились exit 1:
  `fatal: Unable to create 'C:/work/git/my-repos/Strata/.git/index.lock': Permission denied`.
  Коммит не создан. Все изменения в рабочем дереве `dev`; требуется запись в `.git`.
- **Следующий шаг:** P0.3 по «Точке продолжения». Независимый frontend-подпункт —
  сохранить tool-call/result IDs в `run_with_mcp` и `apiMessages` web app,
  проверить несколько инструментов и продолжение через GLM template на mock.

### P2.5b-01 — 2026-10-04, Asia/Yekaterinburg — IDs в MCP continuation

- **Статус:** DONE для P2.5b на mock/Node. Родительский P2.5 остаётся открытым.
- **Исполнитель:** Codex. Ветка `dev`, база
  `24dd8802b54b65f7fa5e515655d9956a672511bb`; предыдущие изменения сохранены.
- **Найдено:** `assistantMessages`/`apiMessages` web app уже сохраняли id и
  tool_call_id. Потеря происходила в `run_with_mcp`: история для следующего
  раунда создавалась без связи результатов с вызовами. GLM template использует
  эти IDs, а новый API normalizer требует их для tool messages.
- **Изменение:** сервер сохраняет id/type в assistant.tool_calls и соответствующий
  tool_call_id в каждом результате. Несколько результатов связываются с calls
  через zip после существующей проверки полноты; отменённый неполный раунд не
  передаётся в prepare. Production-код веб-истории менять не потребовалось.
- **Файлы:** `serve/server.py`, `serve/test_glm5next_mcp.py`,
  `serve/test_glm5next_mcp_ui.cjs`, `serve/fixtures/README.md`, план и статус.
- **Проверки:** рабочая директория `C:\work\git\my-repos\Strata`.
  - `.venv/Scripts/python.exe -m unittest serve.test_glm5next_mcp` — exit 0,
    6 новых тестов. Реальный Service, embedded GLMTemplate, GLM parser и
    OpenAI serializer использованы со scripted engine и in-memory hub.
    Проверены два calls, уникальные IDs между раундами, сортировка переставленных
    результатов в prompt, совпадение IDs stream events и continuation history,
    нормализация истории, отмена, round/token limits, приоритет client tools,
    отсутствие мутации исходной истории и совместимость Qwen.
  - `.venv/Scripts/python.exe -m unittest serve.test_glm5next_mcp serve.test_glm5next_settings serve.test_glm5next_requests serve.test_glm5next serve.test_glm5next_service serve.test_deepseek tools.test_glm5next_loader_contract tools.test_setup_glm5next tools.test_deepseek4 tools.test_shards`
    — exit 0, 100 тестов. Прежний ResourceWarning из `strata_pack.py:351`
    сохраняется; числа MockEngine не являются замером скорости модели.
  - `node serve/test_glm5next_mcp_ui.cjs` — exit 0 после исправления чтения
    CRLF/LF в самом тесте (первый запуск дал SyntaxError из-за неверно выделенных
    функций). Проверены реальные web-функции обработки MCP events и построения
    истории: несколько calls/rounds, error result, стабильные IDs, JSON arguments,
    исключение skipped/unfinished, сохранение и повторное открытие чата.
  - `git -c safe.directory=C:/work/git/my-repos/Strata diff --check` — exit 0.
    Отдельный лог не сохранён.
- **Граница результата:** не запускались внешние MCP servers, HTTP listener,
  браузерная визуальная проверка и полная GLM. Новых измерений скорости нет.
  Существующая веб-история не сохраняет reasoning по раундам; это изменение
  гарантирует связь tool calls/results, а не полный reasoning replay.
- **Коммит:** команды
  `git -c safe.directory=C:/work/git/my-repos/Strata add -- .gitattributes docs/GLM53 serve tools/fixtures/glm5next_loader_headers.json tools/gguf_reader.py tools/glm5next_loader_contract.py tools/setup_glm5next.py tools/test_glm5next_loader_contract.py tools/test_setup_glm5next.py`
  и `git -c safe.directory=C:/work/git/my-repos/Strata commit -m "Add GLM frontend and preserve MCP continuation IDs"`
  обе завершились exit 1:
  `fatal: Unable to create 'C:/work/git/my-repos/Strata/.git/index.lock': Permission denied`.
  Коммит не создан. Изменения всех запусков остаются в `dev`; требуется запись в `.git`.
- **Следующий шаг:** P0.3 по «Точке продолжения». Независимый frontend-подпункт —
  reasoning replay в `assistantMessages`: сохранить reasoning_content по MCP
  раундам и проверить согласованность с GLM clear_thinking на сохранённой истории.

### P2.5c-01 — 2026-10-04, Asia/Yekaterinburg — Reasoning replay истории web app

- **Статус:** DONE для P2.5c на Node/fixtures; P2.5 целиком остаётся открытым.
- **Исполнитель:** Codex. Ветка `dev`, база
  `24dd8802b54b65f7fa5e515655d9956a672511bb`; предыдущие изменения сохранены.
- **Изменение:** GLMTemplate объявляет replay_reasoning capability. Web-функция
  assistantMessages сохраняет reasoning_content обычного ответа и распределяет
  reasoning по MCP-раундам с помощью существующих сохранённых смещений rat.
  Сохраняется и продолжение, содержащее только reasoning. Неподдерживающие эту
  capability модели используют прежнюю историю без reasoning. UI-разделители
  и крайние пробелы в MCP-раундах убираются, как в серверном run_with_mcp.
- **Совместимость старых данных:** если rat отсутствуют, неверного типа, выходят
  за диапазон или убывают, reasoning для такого MCP-ответа не распределяется
  наугад. Text/calls/results остаются в истории. clear_thinking применяется
  шаблоном, сохранённые мысли в localStorage при переключении не удаляются.
- **Файлы:** `serve/glm5next.py`, `serve/web/app.js`,
  `serve/test_glm5next_settings.py`, `serve/test_glm5next_mcp.py`,
  `serve/test_glm5next_mcp_ui.cjs`, `serve/fixtures/README.md`, план/статус.
- **Проверки:** рабочая директория `C:\work\git\my-repos\Strata`.
  - `node serve/test_glm5next_mcp_ui.cjs` — exit 0: reasoning каждого раунда,
    plain/reasoning-only answers, финальное reasoning-only продолжение,
    сохранение/восстановление, старые/некорректные offsets и fallback для других
    моделей; прежние проверки IDs и skipped/unfinished calls также прошли.
  - `.venv/Scripts/python.exe -m unittest serve.test_glm5next_mcp serve.test_glm5next_settings`
    — exit 0, 14 тестов. Два новых теста получают реальный JSON истории из
    Node `--history`, пропускают через GLM normalizer и embedded template:
    clear_thinking сохраняет reasoning текущего tool turn; после нового user
    сообщения удаляет только прежнее reasoning, сохраняя content и tool results.
  - `.venv/Scripts/python.exe -m unittest serve.test_glm5next_mcp serve.test_glm5next_settings serve.test_glm5next_requests serve.test_glm5next serve.test_glm5next_service serve.test_deepseek tools.test_glm5next_loader_contract tools.test_setup_glm5next tools.test_deepseek4 tools.test_shards`
    — exit 0, 102 теста, без skips (Node доступен). В среде без Node два новых
    межъязыковых теста помечаются skipped. Прежний ResourceWarning из
    `strata_pack.py:351` сохраняется; MockEngine не является замером скорости.
  - `node --check serve/web/app.js` и `node serve/test_glm5next_settings_ui.cjs`
    — exit 0. `git -c safe.directory=C:/work/git/my-repos/Strata diff --check` — exit 0.
    Отдельный лог не сохранён.
- **Граница результата:** браузерная визуальная проверка, HTTP/full-model
  integration, tokenizer oracle и замеры скорости не проводились. GLM runtime
  selection по-прежнему не включён. Сохранение reasoning может увеличить prompt;
  clear_thinking позволяет удалять прошлое reasoning при рендеринге.
- **Коммит:** команды
  `git -c safe.directory=C:/work/git/my-repos/Strata add -- .gitattributes docs/GLM53 serve tools/fixtures/glm5next_loader_headers.json tools/gguf_reader.py tools/glm5next_loader_contract.py tools/setup_glm5next.py tools/test_glm5next_loader_contract.py tools/test_setup_glm5next.py`
  и `git -c safe.directory=C:/work/git/my-repos/Strata commit -m "Add GLM frontend and preserve reasoning across tool rounds"`
  обе завершились exit 1:
  `fatal: Unable to create 'C:/work/git/my-repos/Strata/.git/index.lock': Permission denied`.
  Коммит не создан, `.git` read-only; все изменения остаются в `dev`.
- **Следующий шаг:** P0.3 по «Точке продолжения». Независимый остаток P2.5 —
  INFO/About/monitor по backend capabilities: начать с Service.metrics/health
  и представления модели в web app, проверить отсутствие жёстких Qwen-допущений
  на mock GLM, сохранив запрет запуска до готовности backend/tokenizer.

### P2.5d-01 — 2026-10-04, Asia/Yekaterinburg — INFO facts в About/Monitor

- **Статус:** DONE для отображения reported INFO и frontend identity на mock/Node.
  Runtime GLM INFO и выбор backend остаются в P0.3/P2.5.
- **Исполнитель:** Codex. Ветка `dev`, база
  `24dd8802b54b65f7fa5e515655d9956a672511bb`; прежние изменения сохранены.
- **Основание:** существующий DeepSeek INFO в `backends/deepseek4/main.cpp`
  содержит expert_compute/storage/pipeline/policy/speculative; Qwen INFO в
  `src/program/generate.cpp` содержит spec/mtp_max. Раньше About показывал
  экспертный конвейер только для architecture=deepseek4 и называл чужой spec MTP.
- **Изменение:** Service.engine_facts объединяет reported INFO с актуальными
  model/context/images и reasoning capabilities; architecture берётся из INFO
  или выбранного template. Никаких GPU/cache/MTP claims из template не создаётся.
  Metrics и health используют эту identity. About/Monitor показывают поля
  независимо от architecture, отличают матрицы от экспертов, used bytes от
  бюджета и сохраняют нулевые значения. Unknown speculation не называется MTP;
  legacy Qwen mtp_max сохраняет прежний смысл verify window. Сообщение о
  неподдерживаемых VRAM controls больше не привязано к имени DeepSeek.
- **Файлы:** `serve/server.py`, `serve/web/app.js`, `serve/web/index.html`,
  `serve/test_glm5next_info.py`, `serve/test_glm5next_info_ui.cjs`,
  `serve/fixtures/README.md`, план/статус.
- **Проверки:** рабочая директория `C:\work\git\my-repos\Strata`.
  - `.venv/Scripts/python.exe -m unittest serve.test_glm5next_info` — exit 0,
    4 новых теста: отсутствие выдуманных engine capabilities, metrics с нулевыми
    значениями, precedence INFO architecture/Service identity, legacy Qwen fields.
  - `node serve/test_glm5next_info_ui.cjs` — exit 0: реальные About/helpers
    на одинаковых reported fields для GLM/DeepSeek/неизвестной архитектуры,
    пустые и нулевые поля, матрицы/эксперты, used/budget, DSpark/native MTP/legacy
    Qwen и unknown speculation. DOM подменён; это не визуальный браузерный тест.
  - `.venv/Scripts/python.exe -m unittest serve.test_glm5next_info serve.test_glm5next_mcp serve.test_glm5next_settings serve.test_glm5next_requests serve.test_glm5next serve.test_glm5next_service serve.test_deepseek tools.test_glm5next_loader_contract tools.test_setup_glm5next tools.test_deepseek4 tools.test_shards`
    — exit 0, 106 тестов, без skips. Прежний ResourceWarning из
    `strata_pack.py:351` сохраняется; строки MockEngine не являются замерами.
  - `node serve/test_glm5next_mcp_ui.cjs`, `node serve/test_glm5next_settings_ui.cjs`
    и `node --check serve/web/app.js` — exit 0.
  - `git -c safe.directory=C:/work/git/my-repos/Strata diff --check` — exit 0.
    Отдельный лог не сохранён.
- **Граница результата:** INFO fixtures не доказывают поддержку операций GPU.
  Новый GLM engine INFO не реализован; backend не зарегистрирован. Полная модель,
  HTTP end-to-end, browser visual QA, tokenizer oracle и скорость не проверялись.
- **Коммит:** команды
  `git -c safe.directory=C:/work/git/my-repos/Strata add -- .gitattributes docs/GLM53 serve tools/fixtures/glm5next_loader_headers.json tools/gguf_reader.py tools/glm5next_loader_contract.py tools/setup_glm5next.py tools/test_glm5next_loader_contract.py tools/test_setup_glm5next.py`
  и `git -c safe.directory=C:/work/git/my-repos/Strata commit -m "Add GLM frontend and architecture-independent engine facts"`
  обе завершились exit 1:
  `fatal: Unable to create 'C:/work/git/my-repos/Strata/.git/index.lock': Permission denied`.
  Коммит не создан; изменения остаются в рабочем дереве `dev`, нужна запись в `.git`.
- **Следующий шаг:** P0.3 по «Точке продолжения». Независимая проверка P2.5 —
  пройти generation handlers OpenAI/Anthropic с явно созданным GLM Service на
  mock, включая streaming, ошибки options и следующий tool turn; запуск профиля
  GLM разрешать только после реализации backend и проверки tokenizer oracle.

### P2.5e-01 — 2026-10-04, Asia/Yekaterinburg — Generation handlers и приоритет client tools

- **Статус:** DONE для обработчиков на mock/in-memory IO. P2.5 целиком остаётся
  IN_PROGRESS: runtime backend и запуск полной модели не реализованы.
- **Исполнитель:** Codex. Ветка `dev`, база
  `24dd8802b54b65f7fa5e515655d9956a672511bb`; предыдущие изменения сохранены.
- **Ошибка:** GLM normalizer сохраняет OpenAI schema с обёрткой `function`,
  но `_openai` собирал client tool names только из внешнего `name`. При совпадении
  имени с MCP клиентский инструмент не исключался из MCP и мог выполняться сервером.
- **Изменение:** список исключений MCP читает имя как из wrapped, так и из bare
  schema. Добавлены проверки настоящих `do_POST`, JSON/SSE writers и normalizers
  с GLM template, scripted engine, byte tokenizer и in-memory hub/IO.
- **Файлы:** `serve/server.py`, `serve/test_glm5next_handlers.py`,
  `serve/fixtures/README.md`, план/статус.
- **Проверки:** рабочая директория `C:\work\git\my-repos\Strata`.
  - `.venv/Scripts/python.exe -m unittest serve.test_glm5next_handlers` — exit 0,
    6 тестов: JSON/SSE обоих API с Unicode/reasoning/stop; ошибочные options
    дают JSON 400 до streaming/generation; client/MCP коллизии wrapped/bare;
    обычный MCP продолжает работать; tool-result round trip OpenAI и Anthropic.
    До исправления production-кода тест коллизии падал для wrapped schema:
    множество исключений содержало `None` вместо `mock__echo`.
  - `.venv/Scripts/python.exe -m unittest serve.test_glm5next_handlers serve.test_glm5next_info serve.test_glm5next_mcp serve.test_glm5next_settings serve.test_glm5next_requests serve.test_glm5next serve.test_glm5next_service serve.test_deepseek tools.test_glm5next_loader_contract tools.test_setup_glm5next tools.test_deepseek4 tools.test_shards`
    — exit 0, 112 тестов без skips. Сохраняется прежний ResourceWarning из
    `strata_pack.py:351`. Вывод скорости MockEngine не является измерением.
  - `git -c safe.directory=C:/work/git/my-repos/Strata diff --check` — exit 0;
    только предупреждения Git о будущей LF → CRLF нормализации. Лог не сохранён.
- **Граница результата:** HTTP listener, disconnect watcher, внешний MCP,
  tokenizer oracle, полная модель и GPU не проверялись. Backend не зарегистрирован;
  скорость не измерена. Ошибочный GLM request не запускает generation; загрузка
  backend перед валидацией в Anthropic handler этой проверкой не исключается.
- **Коммит:** команды
  `git -c safe.directory=C:/work/git/my-repos/Strata add -- .gitattributes docs/GLM53 serve tools/fixtures/glm5next_loader_headers.json tools/gguf_reader.py tools/glm5next_loader_contract.py tools/setup_glm5next.py tools/test_glm5next_loader_contract.py tools/test_setup_glm5next.py`
  и `git -c safe.directory=C:/work/git/my-repos/Strata commit -m "Add GLM frontend and preserve client tool priority over MCP"`
  обе завершились exit 1:
  `fatal: Unable to create 'C:/work/git/my-repos/Strata/.git/index.lock': Permission denied`.
  Коммит не создан; изменения сохранены в рабочем дереве `dev`. Для коммита нужна
  разрешённая запись в `.git`; ограничения среды не обходились.
- **Следующий шаг:** P0.3 по «Точке продолжения»: получить закреплённые исходники,
  подготовить `backends/glm5next/`, подтвердить сборку и MTP-off graph trace.
  Реальный HTTP smoke с GLM выполнять после backend и tokenizer oracle.

### P2.1a-01 — 2026-10-04, Asia/Yekaterinburg — Python glm4 pre-tokenizer

- **Статус:** DONE для Python regex, экспорта и fixture/local smoke. P2.1 целиком
  не закрыт: точные IDs ещё не сопоставлены с oracle выбранной GLM-зависимости.
- **Исполнитель:** Codex. Ветка `dev`, база
  `24dd8802b54b65f7fa5e515655d9956a672511bb`; прежние изменения сохранены.
- **Изменение:** `tools/strata_tokenizer.py` принимает `pre=glm4`. Отдельный
  GLM4_PATTERN соответствует локальному `LLAMA_VOCAB_PRE_TYPE_CHATGLM4` в
  `third_party/llama.cpp/src/llama-vocab.cpp:406`: группы цифр по 1–3,
  combining marks вне класса букв. Экспорт `tokenizer.json` сохраняет GLM regex,
  а не Qwen regex. Режимы qwen35/joyai-llm и правила CONTROL/USER_DEFINED сохранены.
  В пояснении явно разделены lossless round-trip и сравнение token IDs.
- **Файлы:** `tools/strata_tokenizer.py`, `tools/test_glm5next_tokenizer.py`,
  план/статус. Веса и metadata исходного GGUF не изменялись.
- **Проверки:** рабочая директория `C:\work\git\my-repos\Strata`.
  - `.venv/Scripts/python.exe -m unittest tools.test_glm5next_tokenizer` — exit 0,
    7 тестов: цифры и запрещённые межгрупповые merges, combining marks,
    contractions/whitespace/multilingual splits, special-token flags без auto-BOS,
    round-trip корпуса, отличия от Qwen и unknown pre, metadata-only GGUF/export.
    Synthetic vocabulary содержит merges через неправильные границы; проверки
    ожидаемых pieces ловят ошибку, которую один round-trip не обнаруживает.
  - `.venv/Scripts/python.exe tools/strata_tokenizer.py --gguf H:/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL-00001-of-00004.gguf --out build-local/glm5next-tokenizer --check`
    — exit 0, 154880 tokens / 321649 merges / pre glm4, 16 строк, 0 ошибок.
    Экспорт сохранён локально в `build-local/glm5next-tokenizer/tokenizer/`
    (ignored build output, не часть коммита).
  - `.venv/Scripts/python.exe -c "from tools.strata_tokenizer import Tokenizer; from tools.test_glm5next_tokenizer import CORPUS; t=Tokenizer.from_gguf(r'H:/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL-00001-of-00004.gguf'); results=[t.decode(t.encode(s,parse_special=p))==s for s in CORPUS for p in (False,True)]; print('GLM local corpus:', len(results), 'cases,', results.count(False), 'failed'); assert all(results); assert t.encode('[gMASK]<sop>',parse_special=True)==[154822,154824]"`
    — exit 0, 36 случаев, 0 ошибок. Русский/английский/китайский, код, числа,
    emoji, combining marks, whitespace/control и GLM markers в обоих режимах
    parse_special. Prefix IDs сверены с токенами metadata, не с inference oracle.
  - `.venv/Scripts/python.exe -m unittest tools.test_glm5next_tokenizer serve.test_glm5next_handlers serve.test_glm5next_info serve.test_glm5next_mcp serve.test_glm5next_settings serve.test_glm5next_requests serve.test_glm5next serve.test_glm5next_service serve.test_deepseek tools.test_glm5next_loader_contract tools.test_setup_glm5next tools.test_deepseek4 tools.test_shards`
    — exit 0, 119 тестов без skips. Сохраняется прежний ResourceWarning из
    `strata_pack.py:351`; вывод скорости MockEngine не является замером.
  - `git -c safe.directory=C:/work/git/my-repos/Strata diff --check` — exit 0.
    Отдельный лог не сохранён.
- **Граница результата:** локальный llama.cpp относится к зависимости DeepSeek,
  а не к выбранному Unsloth SHA. GPU, template oracle и точные tokenizer IDs
  выбранной GLM-зависимости ещё не проверены. Backend/profile не зарегистрированы.
  Эти проверки не доказывают правильность inference и не измеряют ускорение.
- **Коммит:** команды
  `git -c safe.directory=C:/work/git/my-repos/Strata add -- .gitattributes docs/GLM53 serve tools/fixtures/glm5next_loader_headers.json tools/gguf_reader.py tools/glm5next_loader_contract.py tools/setup_glm5next.py tools/test_glm5next_loader_contract.py tools/test_setup_glm5next.py tools/strata_tokenizer.py tools/test_glm5next_tokenizer.py`
  и `git -c safe.directory=C:/work/git/my-repos/Strata commit -m "Add GLM frontend and glm4 Python tokenizer"`
  завершились exit 1:
  `fatal: Unable to create 'C:/work/git/my-repos/Strata/.git/index.lock': Permission denied`.
  Коммит не создан; изменения сохранены в рабочем дереве `dev`, нужна запись в `.git`.
- **Следующий шаг:** P0.3 — получить и собрать выбранную Unsloth-зависимость;
  затем P2.1b — vocab-only oracle на локальных metadata, сравнить точные IDs
  `CORPUS` и rendered GLM prompts при обоих parse_special, сохранить parity report.

### P3.2a-01 — 2026-10-04, Asia/Yekaterinburg — Reference-план чтения экспертных матриц

- **Статус:** DONE для reference byte-range/cache-key contract и fixtures.
  P3 остаётся IN_PROGRESS; runtime-транспорт и GPU-кэш этим изменением не реализованы.
- **Исполнитель:** Codex. Ветка `dev`, база
  `24dd8802b54b65f7fa5e515655d9956a672511bb`; предыдущие изменения сохранены.
- **Изменение:** `tools/glm5next_expert_plan.py` принимает отчёт инспектора с
  tensor_details и возвращает gate/up/down для уникальных routed IDs в порядке
  первого появления. Размер slice вычисляется из quant geometry и shape
  `[columns, rows, experts]`, down проверяется с переставленными rows/columns.
  Между slices не добавляется выравнивание. Ключ включает явные model identity,
  load generation, main/MTP, layer, expert, projection, quant/layout и file range.
  Chunk iterator ограничивает число байт чтения, сохраняя неполный последний chunk;
  перед использованием матрицы нужно собрать все chunks, так как quant block может
  быть разделён. Некорректные IDs/layout/ranges отклоняются до выдачи плана.
- **Файлы:** `tools/glm5next_expert_plan.py`, `tools/test_glm5next_expert_plan.py`,
  план/статус и два JSON-артефакта ниже. Ни профиль, ни C++ backend не изменены.
- **Проверки:** рабочая директория `C:\work\git\my-repos\Strata`.
  - `.venv/Scripts/python.exe -m unittest tools.test_glm5next_expert_plan` — exit 0,
    7 тестов: пять quant types, down shape, первый/последний expert,
    dedup и полные triples, побайтовая реконструкция синтетических payloads,
    неполные chunks, cache-key separation, ошибочные ID/layout/file ranges,
    смещения выше 32 бит. Тесты читают только небольшие synthetic GGUF.
  - `.venv/Scripts/python.exe -m tools.glm5next_expert_plan --gguf H:/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL-00001-of-00004.gguf --layer 3 --experts 0 1 2 3 4 5 6 287 0 --chunk-bytes 1048576 --output docs/GLM53/GLM53_FLASH_EXPERT_PLAN_MAIN.json`
    — exit 0, 24 матрицы / 87031808 байт адресуемых диапазонов.
    [Main plan](GLM53_FLASH_EXPERT_PLAN_MAIN.json).
  - `.venv/Scripts/python.exe -m tools.glm5next_expert_plan --gguf H:/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL-00001-of-00004.gguf --layer 45 --experts 0 1 2 3 4 5 6 287 0 --chunk-bytes 1048576 --output docs/GLM53/GLM53_FLASH_EXPERT_PLAN_MTP.json`
    — exit 0, 24 матрицы / 95420416 байт адресуемых диапазонов.
    [MTP plan](GLM53_FLASH_EXPERT_PLAN_MTP.json). В обоих планах повторный ID 0
    устранён; границы expert 287 остаются внутри исходных tensor ranges.
  - `.venv/Scripts/python.exe -m unittest tools.test_glm5next_expert_plan tools.test_glm5next_tokenizer serve.test_glm5next_handlers serve.test_glm5next_info serve.test_glm5next_mcp serve.test_glm5next_settings serve.test_glm5next_requests serve.test_glm5next serve.test_glm5next_service serve.test_deepseek tools.test_glm5next_loader_contract tools.test_setup_glm5next tools.test_deepseek4 tools.test_shards`
    — exit 0, 126 тестов без skips. Прежний ResourceWarning из
    `strata_pack.py:351` сохраняется; mock tok/s не являются замером.
  - `git -c safe.directory=C:/work/git/my-repos/Strata diff --check` — exit 0.
    Отдельный лог не сохранён; результаты локального header-only прогона — в JSON.
- **Граница результата:** локальные планы используют только заголовки и длины
  GGUF, не читают указанные 87/95 МБ весов. Суммы диапазонов не являются замером
  трафика или скорости. Асинхронные чтения, H2D, GPU, удержание файлов/CUDA events,
  runtime cache invalidation/reload не проверены. Reference key требует нового
  generation при каждой загрузке; приложение пока не использует этот код.
- **Коммит:** команды
  `git -c safe.directory=C:/work/git/my-repos/Strata add -- .gitattributes docs/GLM53 serve tools/fixtures/glm5next_loader_headers.json tools/gguf_reader.py tools/glm5next_loader_contract.py tools/setup_glm5next.py tools/test_glm5next_loader_contract.py tools/test_setup_glm5next.py tools/strata_tokenizer.py tools/test_glm5next_tokenizer.py tools/glm5next_expert_plan.py tools/test_glm5next_expert_plan.py`
  и `git -c safe.directory=C:/work/git/my-repos/Strata commit -m "Add GLM frontend and expert byte-range reference plan"`
  обе завершились exit 1:
  `fatal: Unable to create 'C:/work/git/my-repos/Strata/.git/index.lock': Permission denied`.
  Коммит не создан; изменения остаются в рабочем дереве `dev`, нужна запись в `.git`.
- **Следующий шаг:** P0.3 — изолированная сборка выбранной Unsloth-зависимости.
  После P1 применять P3.2a как reference при реализации runtime P3.2b и сравнить
  доставленные GPU bytes с диапазонами для всех пяти quant types и main/MTP.

### P0.3a-01 — 2026-10-04, Asia/Yekaterinburg — Изолированный build scaffold и tokenizer harness

- **Статус:** DONE для scaffold/archive gate и C++ protocol harness.
  Реальный target `strata-glm5next-tokenizer` — VERIFY: полного build/link с
  выбранной зависимостью не было. P0.3/P0/P2.1b остаются незавершёнными.
- **Исполнитель:** Codex. Ветка `dev`, база
  `24dd8802b54b65f7fa5e515655d9956a672511bb`; прежние изменения сохранены.
- **Изменение:** отдельный `backends/glm5next/` требует локальный архив и явно
  заданный SHA-256, сверяет байты до compiler/dependency configure, размещает
  upstream в своей build-директории. Архивный hash не выдуман и не задан по умолчанию.
  Target oracle использует vocab_only без context/GPU layers/MTP, проверяет GLM
  architecture/pre и предоставляет `ENC 0|1 hex` / `IDS` / `ERR` / `QUIT`.
  Protocol test использует mock encoder и собирается без llama/CUDA/model.
  Build record содержит requested revision, hash, compiler/CUDA и applied patches;
  это не утверждение о подлинности архива или успешной сборке.
- **Основание API:** [llama.h выбранного Unsloth SHA](https://github.com/unslothai/llama.cpp/blob/86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9/include/llama.h).
  Проверены declarations vocab_only/load_mtp и tokenizer/metadata API. Полный
  source archive по-прежнему отсутствует; это не проверка ABI/link выбранной ревизии.
- **Файлы:** `backends/glm5next/{CMakeLists.txt,SourceArchive.cmake,README.md,tokenizer_oracle.cpp,tokenizer_protocol.hpp,test_tokenizer_protocol.cpp}`,
  `tools/test_glm5next_build.py`, план/статус. DeepSeek CMake/pin не изменены.
- **Проверки:** рабочая директория `C:\work\git\my-repos\Strata`.
  - `.venv/Scripts/python.exe -m unittest tools.test_glm5next_build` — exit 0,
    4 теста: совпавший hash (включая uppercase), отсутствующий/неверный формат,
    изменившийся архив, отсутствующий файл/директория. Fixture проверяет только
    hashing/preflight, а не extraction настоящего upstream.
  - `cmake -S backends/glm5next -B build-local/glm5next-preflight -G Ninja` —
    ожидаемый exit 1: `Set STRATA_GLM_ARCHIVE_SHA256 to the reviewed archive's 64-digit SHA-256`.
    Никакой неявной подстановки зависимости DeepSeek или загрузки из сети нет.
  - `cmake -S backends/glm5next -B build-local/glm5next-protocol -G Ninja -DSTRATA_GLM_PROTOCOL_TESTS_ONLY=ON -DCMAKE_BUILD_TYPE=Release`
    — exit 0 после настройки SDK environment для процесса. Compiler:
    MSVC 19.44.35222.0, toolset directory `14.44.35207`, Windows SDK `10.0.26100.0`.
    Начальные попытки в неполном окружении дали exit 1 (`rc` не найден, затем
    `kernel32.lib` отсутствует); добавлены SDK x64 bin в PATH, MSVC include и SDK
    ucrt/shared/um в INCLUDE, MSVC x64 и SDK ucrt/um x64 в LIB. Системные настройки
    не менялись. README рекомендует инициализированное x64 Native Tools environment.
  - `cmake --build build-local/glm5next-protocol --config Release` — exit 0,
    построен только `glm5next_protocol_test.exe`.
  - `ctest --test-dir build-local/glm5next-protocol -C Release --output-on-failure --no-tests=error`
    — exit 0, 1/1 CTest. Empty input, special flag, raw UTF-8/NUL/newline, CRLF,
    некорректный hex/flag/лишние поля, лимит размера, formatting, отрицательный ID,
    recovery после ошибок и QUIT проверены с mock callback. Release checks активны.
    Локальный лог: `build-local/glm5next-protocol/Testing/Temporary/LastTest.log`.
  - `cl /nologo /Zs /EHsc /std:c++17 /Ithird_party/llama.cpp/include /Ithird_party/llama.cpp/ggml/include /FIC:/work/git/my-repos/Strata/build-local/glm5next-protocol/syntax-defines.h backends/glm5next/tokenizer_oracle.cpp`
    — exit 0 в том же полном MSVC environment. Это syntax-only с локальными
    DeepSeek llama headers, не сборка GLM oracle. Локальный forced header задаёт
    обе provenance macros строкой `syntax-check-only`; он не входит в исходники.
  - `.venv/Scripts/python.exe -m unittest tools.test_glm5next_build tools.test_glm5next_expert_plan tools.test_glm5next_tokenizer serve.test_glm5next_handlers serve.test_glm5next_info serve.test_glm5next_mcp serve.test_glm5next_settings serve.test_glm5next_requests serve.test_glm5next serve.test_glm5next_service serve.test_deepseek tools.test_glm5next_loader_contract tools.test_setup_glm5next tools.test_deepseek4 tools.test_shards`
    — exit 0, 130 тестов без skips. Сохраняется ResourceWarning из
    `strata_pack.py:351`; mock tok/s не являются замером.
  - `git -c safe.directory=C:/work/git/my-repos/Strata diff --check` — exit 0.
- **Граница результата:** не проверены extraction/build/link настоящего архива,
  CUDA, vocabulary loading, token-ID parity и inference. Архив/его проверенный
  hash всё ещё нужны. Профиль GLM не включён; parent P0.3 не закрыт.
- **Коммит:** команды
  `git -c safe.directory=C:/work/git/my-repos/Strata add -- .gitattributes docs/GLM53 serve backends/glm5next tools/fixtures/glm5next_loader_headers.json tools/gguf_reader.py tools/glm5next_loader_contract.py tools/setup_glm5next.py tools/test_glm5next_loader_contract.py tools/test_setup_glm5next.py tools/strata_tokenizer.py tools/test_glm5next_tokenizer.py tools/glm5next_expert_plan.py tools/test_glm5next_expert_plan.py tools/test_glm5next_build.py`
  и `git -c safe.directory=C:/work/git/my-repos/Strata commit -m "Add isolated GLM candidate build and tokenizer oracle harness"`
  обе завершились exit 1:
  `fatal: Unable to create 'C:/work/git/my-repos/Strata/.git/index.lock': Permission denied`.
  Коммит не создан; изменения остаются в рабочем дереве `dev`.
- **Следующий шаг:** P0.3b по обновлённой «Точке продолжения»: проверенный архив,
  настоящая сборка oracle, затем P2.1b — сохранить exact-ID parity report для
  корпуса и rendered prompts. GPU graph tests остаются обязательными отдельно.

### P2.1b.1-01 — 2026-10-04, Asia/Yekaterinburg — Exact-ID comparison runner

- **Статус:** DONE для runner/corpus/report contract на fixtures и scripted
  subprocess. P2.1b.2/P2.1/P0 oracle не закрыты: реального candidate executable нет.
- **Исполнитель:** Codex. Ветка `dev`, база
  `24dd8802b54b65f7fa5e515655d9956a672511bb`; предыдущие изменения сохранены.
- **Изменение:** `tools/check_glm5next_tokenizer.py` сначала проверяет `--version`
  oracle: exact requested_revision из loader contract и явно переданный reviewed
  archive hash. Затем берёт tokenizer и template из GGUF и сравнивает точные IDs
  Python/oracle. 18 plain inputs вынесены в общий corpus; ещё 18 — multi-turn,
  текущий tool turn с двумя вызовами/переставленными results и новый user после
  tools, с low/high/max и обоими clear_thinking. Каждый текст проверяется с
  parse_special off/on: всего 72 сравнения при настоящем запуске.
- **Отчёт:** JSON различает pass/fail/error, сохраняет текст, оба списка IDs,
  first_difference, binary/header/template SHA-256 и declared provenance.
  GGUF hash ограничен header_end; весовые payloads не хешируются. Несовпадения,
  неверный протокол/provenance, exit/timeout дают ненулевой код. Ошибка выполнения
  заменяет предыдущий pass report; output не может перезаписать model/oracle.
- **Файлы:** `tools/check_glm5next_tokenizer.py`,
  `tools/glm5next_tokenizer_corpus.py`, `tools/test_glm5next_oracle_check.py`,
  `tools/test_glm5next_tokenizer.py`, `backends/glm5next/README.md`, план/статус.
  Прежний plain corpus перенесён без изменения содержания.
- **Проверки:** рабочая директория `C:\work\git\my-repos\Strata`.
  - `.venv/Scripts/python.exe -m unittest tools.test_glm5next_oracle_check` — exit 0,
    9 тестов. Scripted subprocess проверяет hex transport UTF-8/NUL/empty text,
    special flags, mismatch/length differences, malformed/ERR/out-of-vocabulary
    ответы, exact response count, ненулевой exit и timeout, неправильный/дублирующийся
    provenance. Отдельно проверены GGUF template corpus, ID-order tool results,
    clear_thinking, header/template/binary hashes, overwrite protection и error report.
    Это тест механизма сравнения, не независимая проверка правильности token IDs.
  - `.venv/Scripts/python.exe -m unittest tools.test_glm5next_oracle_check tools.test_glm5next_build tools.test_glm5next_expert_plan tools.test_glm5next_tokenizer serve.test_glm5next_handlers serve.test_glm5next_info serve.test_glm5next_mcp serve.test_glm5next_settings serve.test_glm5next_requests serve.test_glm5next serve.test_glm5next_service serve.test_deepseek tools.test_glm5next_loader_contract tools.test_setup_glm5next tools.test_deepseek4 tools.test_shards`
    — exit 0, 139 тестов без skips. Прежний ResourceWarning из
    `strata_pack.py:351` сохраняется; mock tok/s не являются измерением.
  - `.venv/Scripts/python.exe -m tools.check_glm5next_tokenizer --help` — exit 0.
  - `git -c safe.directory=C:/work/git/my-repos/Strata diff --check` — exit 0.
    Отдельный лог не сохранён; temporary fixture reports удалены тестами.
- **Граница результата:** настоящий oracle не запускался, реальный parity JSON
  не создавался. Проверка self-reported version/hash не аутентифицирует бинарник;
  нужен воспроизводимый build из проверенного архива. Оба tokenizers получают
  одинаковые Strata-rendered prompts: это не независимый template-rendering oracle.
  GPU/inference/ускорение здесь не проверяются.
- **Коммит:** команды
  `git -c safe.directory=C:/work/git/my-repos/Strata add -- .gitattributes docs/GLM53 serve backends/glm5next tools/fixtures/glm5next_loader_headers.json tools/gguf_reader.py tools/glm5next_loader_contract.py tools/setup_glm5next.py tools/test_glm5next_loader_contract.py tools/test_setup_glm5next.py tools/strata_tokenizer.py tools/test_glm5next_tokenizer.py tools/glm5next_expert_plan.py tools/test_glm5next_expert_plan.py tools/test_glm5next_build.py tools/check_glm5next_tokenizer.py tools/glm5next_tokenizer_corpus.py tools/test_glm5next_oracle_check.py`
  и `git -c safe.directory=C:/work/git/my-repos/Strata commit -m "Add GLM tokenizer oracle parity runner and corpus"`
  обе завершились exit 1:
  `fatal: Unable to create 'C:/work/git/my-repos/Strata/.git/index.lock': Permission denied`.
  Коммит не создан; изменения остаются в рабочем дереве `dev`.
- **Следующий шаг:** P0.3b — получить проверенный архив и собрать oracle; затем
  P2.1b.2 — выполнить команду runner из `backends/glm5next/README.md` с настоящим
  archive hash, сохранить `GLM53_FLASH_TOKENIZER_PARITY.json` и разобрать расхождения.

### P2.5f-01 — 2026-10-04, Asia/Yekaterinburg — Отмена запросов и disconnect lifecycle

- **Статус:** DONE для Service/API/MCP на mock, in-memory IO и реальной очереди
  потоков. Native GLM engine/HTTP socket watcher не проверялись; P2 остаётся в работе.
- **Исполнитель:** Codex. Ветка `dev`, база
  `24dd8802b54b65f7fa5e515655d9956a672511bb`; предыдущие изменения сохранены.
- **Ошибка:** `Service.run` при отмене до generation возвращался без `done`.
  До исправления новые проверки воспроизвели OpenAI `KeyError: usage`, MCP
  `TypeError: NoneType is not subscriptable`, Anthropic без end_turn/message_stop.
  При отмене второго MCP-раунда повторно использовался предыдущий done: 196
  completion tokens вместо 98 на scripted completion.
- **Изменение:** отменённый в очереди запрос пропускает engine work, выполняет
  обычный cleanup и выдаёт `done/cancel` с нулём output tokens после освобождения
  FIFO. API получает корректный terminal event; MCP не теряет и не удваивает
  предыдущие токены. Запрос без engine work не попадает в статистику работы.
  Большая часть diff `Service.run` — отступ существующего generation block под else;
  алгоритм активной генерации и native engine token protocol не изменены.
- **Файлы:** `serve/server.py`, `serve/test_glm5next_handlers.py`,
  `serve/fixtures/README.md`, план/статус.
- **Проверки:** рабочая директория `C:\work\git\my-repos\Strata`.
  - `.venv/Scripts/python.exe -m unittest serve.test_glm5next_handlers` — exit 0
    после исправления; затем добавлена проверка освобождения FIFO перед terminal
    event. Итоговые 11 handler-тестов вошли в полный прогон ниже (5 новых).
  - `.venv/Scripts/python.exe -m unittest serve.test_glm5next_handlers serve.test_server.StatusHandover`
    — exit 0, 12 тестов. Новые проверки: pre-cancel JSON/SSE обоих API и OpenAI
    с MCP; отменённое MCP-продолжение; реальное ожидание FIFO отдельным thread;
    done после освобождения FIFO; BrokenPipeError при prefill/reasoning/partial tool
    с monitor on/off. В последнем случае engine generator закрывается под FIFO,
    parser не исполняет незаконченный call, status/trace очищены, следующий запрос
    на том же Service выдаёт `Next`. Socket watcher заменён источником cancel event.
  - `.venv/Scripts/python.exe -m unittest serve.test_glm5next_handlers tools.test_glm5next_oracle_check tools.test_glm5next_build tools.test_glm5next_expert_plan tools.test_glm5next_tokenizer serve.test_glm5next_info serve.test_glm5next_mcp serve.test_glm5next_settings serve.test_glm5next_requests serve.test_glm5next serve.test_glm5next_service serve.test_deepseek tools.test_glm5next_loader_contract tools.test_setup_glm5next tools.test_deepseek4 tools.test_shards`
    — exit 0, 144 теста без skips. Прежний ResourceWarning из
    `strata_pack.py:351` сохраняется; mock tok/s не являются замером.
  - `.venv/Scripts/python.exe -m unittest serve.test_server.StatusHandover serve.test_server.ToolCallTerminators serve.test_server.UnfinishedToolCall`
    — exit 0, 7 дополнительных серверных регрессий: передача FIFO следующему
    запросу, Qwen tool terminators и незаконченные calls/finish reasons.
  - `git -c safe.directory=C:/work/git/my-repos/Strata diff --check` — exit 0.
    Отдельный лог не сохранён.
- **Граница результата:** тестовый движок/byte tokenizer, а не полный GLM.
  In-memory writer имитирует broken pipe; реальные сокеты, native STOP/drain,
  отмена CUDA/H2D и восстановление GLM hybrid state не проверены. Ускорение не измерено.
- **Коммит:** команды
  `git -c safe.directory=C:/work/git/my-repos/Strata add -- .gitattributes docs/GLM53 serve backends/glm5next tools/fixtures/glm5next_loader_headers.json tools/gguf_reader.py tools/glm5next_loader_contract.py tools/setup_glm5next.py tools/test_glm5next_loader_contract.py tools/test_setup_glm5next.py tools/strata_tokenizer.py tools/test_glm5next_tokenizer.py tools/glm5next_expert_plan.py tools/test_glm5next_expert_plan.py tools/test_glm5next_build.py tools/check_glm5next_tokenizer.py tools/glm5next_tokenizer_corpus.py tools/test_glm5next_oracle_check.py`
  и `git -c safe.directory=C:/work/git/my-repos/Strata commit -m "Add GLM integration and complete queued cancellation responses"`
  обе завершились exit 1:
  `fatal: Unable to create 'C:/work/git/my-repos/Strata/.git/index.lock': Permission denied`.
  Коммит не создан; изменения сохранены в рабочем дереве `dev`.
- **Следующий шаг:** P0.3b — получить проверенный архив и собрать реальный oracle,
  затем запустить P2.1b.2 runner. После реализации GPU backend повторить отмену
  с реальным socket watcher и подтвердить native STOP/drain/state cleanup.

### AUDIT-01 — 2026-10-04, Asia/Yekaterinburg — Проверка реализованного и смена тестовой модели

- **Статус:** DONE для аудита и статических проверок нового файла. Этапы P0/P2/P3
  остаются IN_PROGRESS; P1/P4/P5/P6 не закрывались.
- **Исполнитель:** Codex.
- **Основание:** HEAD `444f442` (`feat(glm5next): add inspector, frontend template,
  build scaffold, and oracle harness`), чистое рабочее дерево до этого аудита.
  Предыдущие записи о невозможности коммита теперь исторические: соответствующая
  реализация присутствует в этом коммите. Сетевой доступ заново не проверялся.
- **Код проверен:** `tools/setup_glm5next.py`, loader contract, tokenizer/corpus/
  oracle runner, `serve/glm5next.py`, API/UI-проверки, reference expert planner,
  CMake/tokenizer scaffold. Полноценного inference target в GLM CMake ещё нет;
  реальный oracle и CUDA-графы не собраны и не проверены этим аудитом.
- **Изменение:** новый одиночный файл выбран для первых тестов в плане, статусе,
  loader report и README сборки. История UD-Q3_K_XL и её JSON не перезаписывались.
  Реализация engine/frontend не менялась.
- **Инспекция:** из `C:\work\git\my-repos\Strata` выполнено:

  ```powershell
  .venv/Scripts/python.exe tools/setup_glm5next.py --model-dir H:/GLM-5.3-Flash-GGUF --check-only --loader-contract --output docs/GLM53/GLM53_FLASH_IQ3_XXS_INSPECTION.json
  ```

  Exit 0, один файл / 1412 тензоров; все имена/формы соответствуют статическому
  контракту. Полный hash/payload и inference не проверялись. Отдельно посчитан
  SHA-256 только заголовка, записан в [сравнение](GLM53_FLASH_IQ3_XXS_COMPARISON.json).
- **Локальный Python frontend и адресация:** через `.venv/Scripts/python.exe -`
  выполнен разовый stdin-скрипт с `GGUFFile`, `Tokenizer.from_gguf`,
  `cases(GLMTemplate(...))`, `inspect_model(..., tensor_details=True)` и
  `plan_expert_reads`. Exit 0. Полный stdin-скрипт отдельно не сохранён; результаты —
  [GLM53_FLASH_IQ3_XXS_LOCAL_CHECKS.json](GLM53_FLASH_IQ3_XXS_LOCAL_CHECKS.json).
  Сравнены 36 входов корпуса при parse_special off/on (72 пары IDs и round-trip);
  среди них 18 prompts из фактических шаблонов обоих GGUF. Для слоёв 3–45 и IDs
  0/1/2/3/4/5/6/287 проверены 1032 матрицы, размеры пяти новых expert quant types,
  дедупликация ID 0 и границы chunks 1 МиБ. Payload не читался; это не GPU-тест
  и не независимый tokenizer/template oracle.
- **Регрессии:** рабочая директория та же.

  ```powershell
  .venv/Scripts/python.exe -m unittest serve.test_glm5next_handlers tools.test_glm5next_oracle_check tools.test_glm5next_build tools.test_glm5next_expert_plan tools.test_glm5next_tokenizer serve.test_glm5next_info serve.test_glm5next_mcp serve.test_glm5next_settings serve.test_glm5next_requests serve.test_glm5next serve.test_glm5next_service serve.test_deepseek tools.test_glm5next_loader_contract tools.test_setup_glm5next tools.test_deepseek4 tools.test_shards serve.test_server.StatusHandover serve.test_server.ToolCallTerminators serve.test_server.UnfinishedToolCall
  node serve/test_glm5next_info_ui.cjs
  node serve/test_glm5next_mcp_ui.cjs
  node serve/test_glm5next_settings_ui.cjs
  ctest --test-dir build-local/glm5next-protocol -C Release --output-on-failure --no-tests=error
  ```

  Все команды exit 0: 151 Python-тест без skips, три Node-проверки, 1/1 CTest.
  CTest запущен на существующей сборке протокола с mock encoder; новой сборки
  llama/CUDA не было. Известный ResourceWarning `strata_pack.py:351` сохранился.
  Mock tok/s в stdout не являются измерением скорости модели. Полный лог тестов
  отдельно не сохранён.
- **Не проверено:** содержимое новых весов, независимая token-ID parity, CUDA types,
  генерация, пиковая память, runtime cache/pipeline/sessions и native MTP.
- **Следующий шаг:** P0.3b по обновлённой точке продолжения, затем P2.1b.2 с новым
  одиночным GGUF. Сохранить новые runtime-результаты отдельно от UD-Q3_K_XL.

## Шаблон следующей записи

Скопировать блок в подтверждённый журнал, убрать угловые скобки и заполнить только
фактами. Если проверка не проводилась, написать «не запускалась» и оставить `VERIFY`.
Шаблон ниже не подтверждает выполнение P0.1 и не должен попадать в сводку как `DONE`.

```markdown
### P0.1-01 — <дата и время, Asia/Yekaterinburg> — Инспектор GLM GGUF

- **Статус:** <IN_PROGRESS / VERIFY / BLOCKED / DONE>.
- **Исполнитель:** <имя агента или идентификатор задачи>.
- **Пункт плана:** P0, инспектор всех частей GGUF и metadata-only shard.
- **Изменение:** <что реализовано и какое поведение теперь обеспечено>.
- **Файлы:** <реальные пути изменённых файлов или Markdown-ссылки>.
- **Ревизия:** <git SHA; перечислить относящиеся к проверке незакоммиченные изменения>.
- **Зависимость / конфигурация:** <SHA backend и параметры, если применимо; иначе «не требуется»>.
- **Проверки:**
  - Рабочая директория: <полный путь>.
  - Команда: `<точная выполненная команда>`.
  - Exit code: <фактический код или «не запускалась»>.
  - Результат: <конкретные проверки и фактические значения>.
  - Лог / JSON: <ссылка на существующий артефакт или «не сохранён»>.
- **Сравнение с критерием:** <какие требования выполнены, какие ещё нет>.
- **Не проверено / риски:** <остаток проверок и обнаруженные ограничения>.
- **Блокер:** <нет либо точная ошибка и условие разблокировки>.
- **Следующий шаг:** <одно конкретное действие, файл/команда и ожидаемый результат>.
```

Пример формулировки результата после реального успешного прогона:
«Инспектор принял metadata-only shard без padding, отклонил усечённый payload;
локальная модель определена как 4 части / 1412 тензоров. Вывод приложен в JSON».
Эту формулировку можно использовать только если перечисленные проверки действительно
выполнены и указанный JSON сохранён. Одного успешного запуска на полной модели
недостаточно для утверждения о проверке ошибочных файлов.
