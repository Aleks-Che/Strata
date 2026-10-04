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
| Общий статус | P1 DONE; P0/P2/P3/P5/P6 IN_PROGRESS. Shared async pipeline и native MTP 1/2/3 работают, точность и pipe/API проверены; P4 session reuse, длинный контекст и полная матрица регрессий ещё не готовы |
| Последняя проверенная ревизия Strata | `3b0e0f75a50d7f284e479443a4d859b691565a16` + изменения рабочего дерева; source/binary hashes в PIPELINE_MTP_VALIDATION |
| Последняя выполненная работа | P3.3b / P5.1a,2c,3a,5a / P6.1a: runtime pipeline, native draft/verify/rollback, битовая parity, подбор глубины/cache/chunk и локальный профиль MTP 1 |
| Следующая задача | P3: увеличить измеренное H2D/compute overlap; P4: hybrid session snapshots/restore. Расширить MTP замеры на другие prompts и длинный sparse-контекст |
| Активная задача / исполнитель | Нет; перечисленные подпункты завершены Codex, остаток плана открыт |
| Блокеры | Блокеров нет; архив, бинарник, профиль и baseline доступны. Остаток плана — следующая разработка |
| Основная тестовая модель | `H:\GLM-5.3-Flash-GGUF\GLM-5.3-Flash-Uncensored-IQ3_XXS.gguf` |
| Дополнительный профиль | `H:\GLM-5.3-Flash-GGUF\UD-Q3_K_XL`; прежние отчёты сохраняются отдельно |
| Стенд | Windows, Ryzen 9 9950X, 128 ГиБ RAM, RTX 5090 32 ГиБ |
| GLM backend / setup | `strata-glm5next`, INFO `glm5next-native`, Native MTP в monitor; OpenAI/Anthropic JSON/SSE, STOP/QUIT и unload проверены на полной модели. Локальный профиль: pipeline on, MTP 1, cache ceiling 512 МиБ, chunk 4 МиБ, RAM/VRAM 95%, ctx2048/batch16/threads4 |
| Закреплённая зависимость GLM | Unsloth `86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9`, архив SHA-256 `f8e524b635b726bae74fd8f84bb9249e5b09384c63207f6707c5a3f921acad99`; MSVC 19.44 / CUDA 13.0.48 / 120a; generated TF32/runtime patches, hashes в SYNC_VALIDATION; production validation остаётся неполной |
| Последняя проверенная конфигурация запуска GLM | IQ3_XXS, 45 main + 1 NextN layer, 39 prompt + 64 generated, ctx2048/batch16, F16 KV, TF32/FA off, greedy; pipeline + cache, MTP 0/1/2/3, global targets 95% |
| Измеренная скорость GLM / память | Медианы трёх прогретых повторов: MTP off 8,584; MTP 1 8,730; MTP 2 7,431; MTP 3 6,555 ток/с. Лучший из проверенных вариантов — 1/512 МиБ/4 МиБ; преимущество над off небольшое, около 1,7%. Глобальные бюджеты RAM/VRAM 95% сохранены; подробности и снимки памяти в PIPELINE_MTP |

Дополнительно к перечисленным выше компонентам выполнены **P0.3b.1 и P2.1b.2**:
[build record](GLM53_FLASH_CANDIDATE_BUILD.json),
[команды/логи](GLM53_FLASH_CANDIDATE_BUILD_TESTS.txt),
[token IDs IQ3_XXS](GLM53_FLASH_IQ3_XXS_TOKENIZER_PARITY.json),
[token IDs UD-Q3_K_XL](GLM53_FLASH_UD_Q3_K_XL_TOKENIZER_PARITY.json).
Архив находится в `build-local/llama-glm-86ebfef2.tar.gz`, oracle —
`build-local/glm5next-candidate-cuda/bin/strata-glm5next-tokenizer.exe`.
Проверка словаря не загружает веса и не исполняет CUDA-граф GLM.

**P0.3b.2a/b:** no_alloc trace обоих GGUF прошёл для контекстов 2048/4096 и
batches 1/4/16/256: блоки 0–44, MTP descriptors отсутствуют, все compute nodes
назначены CUDA. Явное размещение `token_embd.weight` на GPU обязательно: upstream
default оставляет `embd / GET_ROWS` на CPU даже при all-layer offload; negative
audit это обнаруживает. Веса и полный forward pass не исполнялись.
Отдельные **22 CUDA fixtures** KDA/mHC/I32 прошли с CPU/scalar references,
max abs `1,4305115e-6`, NMSE `5,3275651e-15`; это синтетические значения.
[Manifest](GLM53_FLASH_GRAPH_VALIDATION.json),
[команды/CTest](GLM53_FLASH_GRAPH_VALIDATION_TESTS.txt),
[численные результаты](GLM53_FLASH_OPS_CHECK.json).

**P0.3b.2c.1/.2:** настоящий loader/graph/memory кандидата исполнен на маленьком
synthetic GLM: **98/98** проверок hybrid rollback, pending save/restore, kpool,
CPU/GPU logits и dense/sparse attention; **108/108** matrix fixtures охватывают
все типы обоих GGUF + F16. До исправления 10 state/logit сравнений не проходили:
F32 MMF кандидата использует TF32 MMA, игнорируя `NVIDIA_TF32_OVERRIDE=0`.
Изолированная GLM сборка теперь компилирует generated copy с проверкой этого
флага; оригинальные 3599 файлов архива не изменены. При прежних порогах max abs
снизился с `1.3842e-3` до `2.9802e-7`. [Manifest](GLM53_FLASH_STATE_VALIDATION.json),
[до патча](GLM53_FLASH_STATE_BEFORE_TF32_FIX.json), [state](GLM53_FLASH_STATE_CHECK.json),
[матрицы](GLM53_FLASH_QUANT_CHECK.json), [CTest/команды](GLM53_FLASH_STATE_VALIDATION_TESTS.txt).
Эти synthetic checks не исполняли реальные GGUF weights. Следующий P1.1a уже
проверил реальный forward с F16 KV и streaming, как описано ниже.

**P1.1a:** `strata-glm5next-smoke` исполнил IQ3_XXS на CUDA с mmap demand-loading,
без eager prefetch, с одним pinned staging 16 МиБ. GPU-аудит проверяет весь
scheduled graph до вычислений и отвергает CPU compute/full-expert-copy fallback.
Три прогона по 64 токена дали одинаковые token IDs и бит-в-бит одинаковые logits.
Повторный synchronous decode — **3,333 ток/с**, исходный candidate selected-copy
reference — **3,274 ток/с**; это не независимая реализация GLM. Первый замер —
**1,288 ток/с**, файловый кэш ОС не очищался. [Manifest](GLM53_FLASH_SYNC_VALIDATION.json),
[repeat](GLM53_FLASH_SYNC_REPEAT.json), [reference](GLM53_FLASH_SYNC_CANDIDATE_REFERENCE.json),
[команды/CTest](GLM53_FLASH_SYNC_VALIDATION_TESTS.txt).
P1.1b/P2.5h затем подключили production pipe/profile и оба HTTP API; P3.5h —
runtime cache и RAM/VRAM targets. P3.3b/P5 далее подключили pipeline и native MTP
с real-state rollback. Сессии и длинный sparse-контекст остаются открытыми.

**P3.3b / P5 / P6.1a:** общий async pipeline, native NextN draft/verify/rollback,
раздельные main/MTP cache keys, target sample-and-match и короткие CUDA matvec
пакеты. Реальные MTP 1/2/3, forced rejection, seeded sampling и cancel/recovery
совпали с baseline; baseline совпал с сохранёнными P1 logits бит-в-бит.
Подбор параметров, ограниченное измеренное overlap, отчёты и остаток работ:
[конвейер и MTP](GLM53_FLASH_PIPELINE_MTP.md),
[manifest](GLM53_FLASH_PIPELINE_MTP_VALIDATION.json).

Новый основной GGUF проверен 2026-10-04: **1 файл, 1412 тензоров, 112,310 ГиБ**,
45 основных блоков и 1 MTP; имена/формы прошли существующий loader contract.
Основная модель: 109,687 ГиБ весов, из них 102,322 ГиБ экспертов; MTP: 2,614 ГиБ,
из них 2,443 ГиБ экспертов. Полный хеш и численные значения весов не проверялись.
В P3.4b выбранные payloads слоёв 3/11/45 обоих профилей сравнивались после GPU-доставки
с обычным чтением файлов: по 72 матрицы в mmap/native/auto, все сравнения прошли.
[IQ3_XXS](GLM53_FLASH_IQ3_XXS_GPU_TRANSFER.json),
[UD-Q3_K_XL](GLM53_FLASH_UD_Q3_K_XL_GPU_TRANSFER.json).

P3.4c повторил эти диапазоны через отдельный LRU cache: cold/hit/eviction/reload/
invalidation, **2160 сравнений** на двух моделях и трёх режимах. Hit не добавляет
source/H2D bytes. Поколения меняются в ключах checker, настоящего model reload нет.
[IQ3_XXS cache](GLM53_FLASH_IQ3_XXS_GPU_CACHE.json),
[UD-Q3_K_XL cache](GLM53_FLASH_UD_Q3_K_XL_GPU_CACHE.json).

P3.4d проверил эти же 72 матрицы каждого профиля через `ExpertDispatch`, с mixed
hit/miss/bypass, all-hit, отменой и сменой generation keys: LRU/frequency ×
prefill/decode × mmap/native/auto, **8064 сравнений** в 576 сценариях. Это packed
bytes с independent stdio baseline; GLM graph и численные outputs не проверены.
[IQ3_XXS dispatch](GLM53_FLASH_IQ3_XXS_GPU_DISPATCH.json),
[UD-Q3_K_XL dispatch](GLM53_FLASH_UD_Q3_K_XL_GPU_DISPATCH.json).

Вопреки имени `IQ3_XXS`, фактические routed types — IQ2_S/IQ3_S/IQ4_XS и
Q2_K/Q3_K для MTP. Среди остальных весов есть BF16. Проверка CUDA должна учитывать
новый набор типов. Шаблон отличается четырьмя заменами `[0]` → `.0`; 18 rendered
prompts и 72 проверки IDs/round-trip совпали с прежним Python frontend.
[Сравнение файлов](GLM53_FLASH_IQ3_XXS_COMPARISON.json),
[локальные проверки](GLM53_FLASH_IQ3_XXS_LOCAL_CHECKS.json).
Это не независимый tokenizer oracle и не запуск модели.

Новая независимая проверка token IDs P2.1b.2 использует собранный llama oracle:
по **80/80** совпадений на обоих GGUF (22 plain inputs + 18 rendered prompts,
каждый с parse_special off/on). Старый код на четырёх новых китайских строках
дал 8 расхождений из 80; он не учитывал `glm4` ignore_merges. Исправлена выдача
целого vocab piece перед BPE. Шаблон по-прежнему рендерится Strata для обеих сторон;
независимый template oracle и генерация не проверены.

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
| P0. Совместимость и эталон | IN_PROGRESS | Прежние graph/ops/state/quant/tokenizer checks и P1.1a real mixed-quant logits/F16 forward; 10/10 CTest | Real-state rollback, длинный sparse-контекст, независимый template oracle |
| P1. Основной GPU engine | DONE | P1.1a/b: baseline, pipe/profile/setup, 64 токена, GPU-only, F16 KV, exact logits и отмена реального графа; 12/12 CTest | Дальнейший async transport относится к P3 |
| P2. Токенизация и API | IN_PROGRESS | P2.1a/b DONE: реальная tokenizer parity 80/80 на каждом GGUF после ignore_merges; P2.2–P2.4/P2.5a,b,c,d,e,f/P2.6 на fixtures/mock; P2.5g — 30 HTTP-сценариев с mock engine; P2.5h — native GLM JSON/SSE/disconnect/unload | Независимый template oracle, расширенные реальные tools/диалоги; runtime selection, INFO и HTTP с GPU engine уже проверены P2.5h |
| P3. Конвейер и кэш | IN_PROGRESS | Общий reader/pipeline, native planner P3.1b и transport lifetime P3.3a; cache/dispatch P3.2b.1/.2, controller P3.5a–h, byte checks P3.4a–d, telemetry/bench P3.6a/P3.7a,b; P3.3b — runtime async pipeline с router lookahead, exact logits, CUDA trace | Увеличить пока небольшое overlap, проверить cold I/O и readers/chunks на других workloads, подобрать refresh (сейчас 500 мс) |
| P4. Сессии | TODO | Нет | Полный hybrid state, архивы, restore, A → B → A |
| P5. Native MTP | IN_PROGRESS | P5.1a/2c/3a/5a: shared-weight NextN context, INFO/monitor, main/MTP cache ceilings, depths 1/2/3, draft/verify/rollback, seeded sampling, stop/cancel; real logits bit-exact | Session restore/reuse, index sharing между draft iterations, длинный sparse-контекст и реальные tool/EOS диалоги |
| P6. Замеры и выпуск профиля | IN_PROGRESS | P6.1a: глубины 0/1/2/3, cache 256/512, chunks 4/8 на одном prompt; профиль MTP 1/512/4, pipe/API/fixtures и документация | Другие prompts/контексты, full-model Qwen/DeepSeek регрессии, Linux/HIP, полная release matrix |

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

**Следующая задача — P3.3b: shared pipeline в реальном GLM runtime.**

1. Начать с `backends/glm5next/runtime_memory.cpp`: cache hits уже идут D2D в
   scratch, misses — последовательно через pinned 16 МиБ. Подключить проверенный
   shared transport/pipeline к misses; сохранять mapping owners, MMQ padding,
   lease events и действующие global RAM/VRAM budgets. Сейчас copy/compute overlap нет.
2. Сохранить strict GPU audit и TF32/FA/MTP off. Generated patches не изменяют
   оригинальные файлы audited Unsloth `86ebfef2`; не менять зависимости DeepSeek.
3. Артефакты текущей базы: [memory manifest](GLM53_FLASH_MEMORY_VALIDATION.json),
   [команды](GLM53_FLASH_MEMORY_TARGETS_TESTS.txt), [README](../../backends/glm5next/README.md).
   Бинарники/архив в `build-local/glm5next-candidate-cuda` и
   `build-local/llama-glm-86ebfef2.tar.gz`; локальный профиль `strata-glm5next.json`.
4. После изменения сравнить 64 IDs и все 9 912 320 F32 logits с
   `build-local/glm-sync-iq3-repeat.f32`, затем pipe и HTTP cancellation/recovery.
   Снимать раздельно warmup, prefill, decode, source/H2D, global NVML/RAM;
   различать новый процесс и повтор запроса с уже обученным expert cache.
5. P0.3b.2c.3 (real-state rollback и длинный sparse-контекст), независимый
   template oracle, P4 sessions и native MTP остаются отдельными работами.

**Ожидаемый результат:** измеренное перекрытие этапов загрузки при сохранённой
численной точности и ограниченном расходе памяти; сначала отдельный небольшой
подпункт P3, затем полное copy/compute overlap.

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

### P3.1a-01 — 2026-10-04, Asia/Yekaterinburg — Общий native reader экспертных матриц

- **Статус:** DONE для P3.1a, исполнитель Codex. Весь P3 остаётся IN_PROGRESS.
  Работа выполнена в `dev`, база `2f05974e4b41ed7555182fdad2e111bc11fcc46e`;
  изменения не закоммичены. Рабочее дерево перед началом было чистым.
- **Выбор задачи:** P0.3b пока недоступен: локальный архив кандидата не найден
  в просмотренных `build-local/glm5next*` и `third_party`. Команда ниже из корня
  репозитория завершилась exit 1 с `urllib.error.URLError: <urlopen error
  [WinError 10013] ...>`; загрузка не началась:

  ```powershell
  .venv/Scripts/python.exe -c "import urllib.request; r=urllib.request.urlopen('https://github.com/unslothai/llama.cpp/archive/86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9.tar.gz',timeout=15); print(r.status); print(r.headers.get('Content-Length')); r.close()"
  ```

- **Изменение:** существующий reader выделен в `backends/common/expert_file.hpp`;
  `backends/deepseek4/expert_file.hpp` перенаправляет на него прежние loader hooks
  и pipeline. Добавлен `Request::read_at` с явным 64-битным offset для GLM range
  planner. Он не разыменовывает mmap и удерживается вызывающим кодом вместе с
  `Source`. Проверяются границы, переполнение, размер native chunk, null destination,
  повторные/перекрывающиеся регистрации. При неожиданной ошибке ожидания I/O
  отменяется и дожидается завершения до освобождения буфера/OVERLAPPED.
  Сигнал отмены, замеченный после завершения чтения, возвращает false.
- **Файлы:** общий reader, `backends/common/test_expert_file.cpp` и README;
  compatibility include DeepSeek; `backends/glm5next/CMakeLists.txt`, его README;
  план/статус и [сохранённый CTest log](GLM53_FLASH_NATIVE_READER_TESTS.txt).
  Зависимость, графы и настройки DeepSeek/Qwen не менялись.
- **Сборка и проверки:** рабочая директория `C:\work\git\my-repos\Strata`.
  В `cmd` сначала выполнен
  `call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"`.
  MSVC **19.44.35222.0**, SDK **10.0.26100.0**, x64 Release.

  ```text
  cmake -S backends/glm5next -B build-local/glm5next-reader -G Ninja -DSTRATA_GLM_PROTOCOL_TESTS_ONLY=ON -DCMAKE_BUILD_TYPE=Release "-DCMAKE_MAKE_PROGRAM=C:/Users/Aleks/AppData/Local/Programs/Python/Python312/Scripts/ninja.exe"
  cmake --build build-local/glm5next-reader --config Release
  ctest --test-dir build-local/glm5next-reader -C Release -V --no-tests=error
  ```

  Все команды exit 0; **3/3 CTest**, без skips. Новый тест выполняется дважды:
  через общий include и через compatibility include DeepSeek. Каждый проверяет
  **384 чтения матриц**: восемь layout (`IQ2_S`, `IQ3_S`, `IQ4_XS`, `Q2_K`, `Q3_K`,
  `IQ3_XXS`, `Q6_K`, `Q4_K`), восемь выбранных экспертов, смешанные gate/up/down,
  chunks 997/65536 байт, последний неполный chunk, guards и padding между тензорами.
  Это синтетические байты с соответствующей геометрией, а не валидные значения
  деквантования или чтение весов локальной модели.
  Дополнительно: **160 чтений четырьмя threads** после unmap/закрытия исходного fd;
  независимость старого/new Source при повторной регистрации адреса и закрытие
  последнего retained handle; sparse-file offset **2³² + 123**; short read после
  усечения, pre-cancel, reuse после ошибок/отмены, отклонение неверных диапазонов.
- **CUDA-регрессия общего кода:** после того же `vcvars64.bat` выполнено:

  ```text
  cl /nologo /EHsc /std:c++17 /O2 /MD /Ithird_party\llama.cpp\ggml\include /Ibuild-local\cuda-13.0\include /Ibuild-local\cuda-13.0\include\cccl backends\deepseek4\test_expert_pipeline.cpp /Fobuild-local\glm5next-reader\deepseek-pipeline.obj /Febuild-local\glm5next-reader\deepseek-pipeline.exe /link build-local\cuda-13.0\lib\x64\cudart.lib psapi.lib
  set "PATH=%CD%\build-local\cuda-13.0\bin;%CD%\build-local\cuda-13.0\bin\x64;%PATH%"
  build-local\glm5next-reader\deepseek-pipeline.exe
  ```

  Компиляция и запуск exit 0, вывод:
  `Pipeline overlap, file queue, byte parity, ordered reuse and cancellation passed`.
  Это новый executable из существующего `test_expert_pipeline.cpp`, использующий
  изменённый общий reader; проверены прежние CUDA H2D/D2D/event/cancel fixtures.
  Полный backend DeepSeek не пересобирался, генерация на модели не запускалась.
  Отдельный log этого запуска не сохранён. `nvidia-smi --query-gpu=name,memory.total
  --format=csv,noheader` ранее вывел `Failed to initialize NVML: Unknown Error`;
  ошибка NVML не помешала фактическому CUDA-тесту.
- **Python-регрессии:**

  ```text
  .venv/Scripts/python.exe -m unittest tools.test_glm5next_expert_plan tools.test_glm5next_build tools.test_glm5next_loader_contract tools.test_setup_glm5next tools.test_deepseek4 tools.test_shards
  ```

  Exit 0, **48 тестов**, без skips. Прежний `ResourceWarning` в
  `strata_pack.py:351` сохраняется; отдельный log не сохранён.
  `git -c safe.directory=C:/work/git/my-repos/Strata diff --check` — exit 0.
- **Граница результата:** готов общий Windows native reader и проверена регрессия
  транспорта DeepSeek. Native I/O для Linux не добавлялось; прежний mmap fallback
  сохранён. Нет GLM graph integration, runtime cache-key/invalidation, GPU-проверки
  восьми GLM layouts, реальной модели, logits или замера скорости. Отмена именно
  незавершённого Windows ReadFile и ветка ошибки WaitForSingleObject не воспроизводились
  детерминированно; native fixtures подтверждают pre-cancel и повторное использование.
- **Следующий шаг:** P0.3b — получить проверенный архив кандидата и собрать oracle
  по `backends/glm5next/README.md`, затем P2.1b.2. При подключении GLM транспорта
  использовать общий `Source`/`read_at`, сохранив lifetime до завершения native I/O;
  отдельно связать staging/cache lifetime с CUDA events и проверить восемь layouts на GPU.

### P3.4a-01 — 2026-10-04, Asia/Yekaterinburg — Общий CUDA pipeline и GLM byte fixtures

- **Статус:** DONE для P3.4a, исполнитель Codex, цикл 2. Весь P3 IN_PROGRESS.
  Ветка `dev`, база `d0152f41cb956d739ae7f72c36ce450500b4d268`, дерево перед
  началом чистое. P3.1a из предыдущей записи уже присутствует в HEAD; новые
  изменения P3.4a не закоммичены. Повторной проверки сети в этом цикле не было.
- **Изменение:** `StrataExpertPipeline` перенесён из DeepSeek в
  `backends/common/expert_pipeline.hpp`; `StrataExpertSlice` выделен в
  `backends/common/expert_slice.hpp`. DeepSeek сохраняет прежний include/API и
  использует этот же код. Общий pipeline больше не зависит от llama/ggml.
  Программно сравнён с `git show HEAD:backends/deepseek4/expert_pipeline.hpp`:
  единственное отличие реализации — include `expert_slice.hpp` вместо
  `expert_transfer.h`; scheduling/events/counters/cancel не менялись.
- **Новый тест:** `backends/glm5next/test_expert_bytes.cpp`, opt-in CMake-флаг
  `STRATA_GLM_TRANSPORT_TESTS_CUDA`. Собирается отдельно от Unsloth и модели,
  требует C++ compiler, CUDA runtime/toolkit и доступный GPU. Отсутствие GPU
  приводит к ошибке, не к успешному skip. Обычная protocol/reader-сборка
  сохраняет работу без CUDA. Обновлены README общего транспорта и GLM, план/статус.
- **Что проверено:** на Windows **18 cases / 432 матрицы**: три смешанные группы
  gate/up/down, по восемь выбранных экспертов из десяти fixture-экспертов (включая
  первый и последний), mmap/native/auto и prefill/decode reader policies.
  Layout соответствует матрицам 4096×2048, down — 2048×4096, с block sizes восьми
  типов `IQ2_S/IQ3_S/IQ4_XS/Q2_K/Q3_K/IQ3_XXS/Q6_K/Q4_K`.
  Данные синтетические, с изменением байтов по абсолютному offset; чтение полной
  модели и деквантование не выполняются.
  - Native/mmap → четыре pinned staging slots → отдельный H2D stream → D2D
    в два чередующихся consumer streams → D2H и полное сравнение с исходником.
  - Chunk **262161 байт** разрывает quant blocks; проверены неполные хвосты,
    padding между тензорами, последний эксперт до EOF и guards по 37 байт вокруг
    каждой GPU-матрицы. Все 24 gate/up/down slices каждого case доставлены.
  - Native mode читает с `PAGE_NOACCESS` на mmap и после удаления registry entry:
    байты приходят через удержанные Source handles. В auto-prefill этого прогона
    весь источник прочитан native, auto-decode — через mmap; это подтверждают
    счётчики, но это не измерение физических чтений SSD.
  - Source/H2D/D2D bytes и chunks точно равны плану; unused=0, groups=1;
    read_peak ограничен четырьмя, для decode вне explicit-native равен одному.
- **Сборка:** рабочая директория `C:\work\git\my-repos\Strata`. Сначала в `cmd`:
  `call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"`,
  затем `set VSLANG=1033` и
  `set "PATH=%CD%\build-local\cuda-13.0\bin;%CD%\build-local\cuda-13.0\bin\x64;%PATH%"`.
  Фактические команды:

  ```text
  cmake -S backends/glm5next -B build-local/glm5next-transport -G Ninja -DSTRATA_GLM_PROTOCOL_TESTS_ONLY=ON -DSTRATA_GLM_TRANSPORT_TESTS_CUDA=ON -DCMAKE_BUILD_TYPE=Release "-DCUDAToolkit_ROOT=%CD%/build-local/cuda-13.0" "-DCMAKE_MAKE_PROGRAM=C:/Users/Aleks/AppData/Local/Programs/Python/Python312/Scripts/ninja.exe"
  cmake --build build-local/glm5next-transport --config Release
  ctest --test-dir build-local/glm5next-transport -C Release -V --no-tests=error
  ```

  Все команды exit 0; **4/4 CTest**, без skips. Новый GPU-тест сообщает
  `NVIDIA GeForce RTX 5090`, CUDA runtime/driver **13000/13000**. CMake обнаружил
  toolkit **13.0.48**, MSVC **19.44.35222.0**, SDK **10.0.26100.0**.
  Лог: [GLM53_FLASH_GPU_BYTE_TESTS.txt](GLM53_FLASH_GPU_BYTE_TESTS.txt).
  Времена CTest относятся к fixtures, не являются скоростью инференса.
- **Регрессия DeepSeek:** `cmd /c build-local\check-deepseek-reader.cmd` — exit 0.
  Скрипт повторил точные `cl`/PATH/executable команды из P3.1a-01 после `vcvars64.bat`,
  заново собрав `backends/deepseek4/test_expert_pipeline.cpp` с новыми includes.
  Вывод: `Pipeline overlap, file queue, byte parity, ordered reuse and cancellation passed`.
  Проверены прежние event/reuse/cancel/overlap fixtures общего pipeline.
  Полный DeepSeek backend и генерация на модели не запускались; отдельного лога нет.
- **Python-регрессии:** из той же директории:

  ```text
  .venv/Scripts/python.exe -m unittest tools.test_glm5next_build tools.test_glm5next_expert_plan tools.test_glm5next_loader_contract tools.test_setup_glm5next tools.test_deepseek4 tools.test_shards
  ```

  Exit 0, **48 тестов**, без skips; прежний `ResourceWarning` в
  `strata_pack.py:351` сохраняется. Отдельный лог не сохранён.
  `git -c safe.directory=C:/work/git/my-repos/Strata diff --check` — exit 0.
- **Граница результата:** транспорт packed bytes проверен на GPU; это не
  подтверждение GLM dequantization/MoE/logits, graph integration, runtime cache
  keys, main/MTP isolation, сессий или ускорения. Linux-ветка GPU fixture
  (только host-memory, шесть cases) не запускалась. Router IDs заданы fixture,
  dedup и GGUF offsets будущего native GLM planner здесь не проверяются.
- **Следующий шаг:** P0.3b → P2.1b.2 по точке продолжения. После появления GLM
  graph hooks подключить общий pipeline к плану gate/up/down и повторить byte parity
  на диапазонах реального GGUF, затем проверить cache keys/events и численные outputs.

### P3.4b-01 — 2026-10-04, Asia/Yekaterinburg — GPU-доставка реальных GGUF-диапазонов

- **Статус:** DONE для P3.4b и пункта P3 о byte parity экспертных матриц.
  Исполнитель Codex, цикл 3. Весь P3 остаётся IN_PROGRESS; GLM graph не подключён.
  Ветка `dev`, база `13a0376a630adfb786518692b5281cd644c0ce81`, дерево перед работой
  чистое. P3.4a уже в HEAD; новые изменения не закоммичены. Сеть не перепроверялась.
- **Реализация:** `tools/check_glm5next_transfer.py` запускает существующий GGUF
  inspector и `plan_expert_reads`, затем `strata-glm5next-transfer-check` из
  `backends/glm5next/check_expert_ranges.cpp`. Новый Windows C++ target включается
  `STRATA_GLM_TRANSPORT_TESTS_CUDA`; Unsloth и модельный loader ему не требуются.
  Манифест передаёт 64-битные offsets/length и UTF-8 paths в hex, без shell parsing.
  C++ проверяет диапазоны относительно фактической длины read-only файла,
  ограничивает матрицу 256 МиБ, chunk 16 МиБ, план 4096 матрицами и 1048576 chunks.
  Общий pipeline выполняет native/mmap → pinned staging → H2D → D2D; после D2H
  каждый байт и guards сравниваются с отдельным `fseek`/`fread` исходного файла.
  Источники и mapping остаются живы до завершения. Код общего pipeline не менялся.
- **Runner:** требует результат каждого индекса и точное соответствие source/H2D/
  D2D/chunk counters плану. Проверяет режим native/mmap, exit code и timeout;
  ошибочный запуск заменяет прежний успешный отчёт на `status=error`, exit 1.
  Запрещает перезапись GGUF/частей/checker, в том числе через hardlink. Сохраняет
  ranges, types, header hashes всех shards, binary hash, sizes/mtime, GPU/runtime;
  изменения sizes/mtime исходников или hash бинарника во время запуска отклоняются.
- **Сборка и CTest:** рабочая директория `C:\work\git\my-repos\Strata`.
  Выполнен `cmd /c build-local\test-glm5next-transport.cmd`, то есть команды сборки
  P3.4a-01 (те же vcvars64, PATH, CMake flags и каталог) после добавления нового target.
  Команда запускалась с выводом в `build-local/glm5next-range-build.log`.
  Exit 0; **5/5 CTest** без skips. Новый `glm5next_range_parser` проверил формат,
  лимиты, испорченные записи/paths и offsets >4 ГиБ. Повторно прошли 432 synthetic
  GPU comparisons и native-reader fixtures.
  [Сохранённый CTest log](GLM53_FLASH_REAL_TRANSFER_TESTS.txt).
  MSVC **19.44.35222.0**, SDK **10.0.26100.0**, toolkit **13.0.48**, RTX 5090,
  CUDA runtime/driver **13000/13000**. Архив кандидата не использовался.
- **Реальные файлы:** после
  `set "PATH=%CD%\build-local\cuda-13.0\bin;%CD%\build-local\cuda-13.0\bin\x64;%PATH%"`
  выполнены команды из корня репозитория:

  ```text
  .venv\Scripts\python.exe -m tools.check_glm5next_transfer --gguf H:/GLM-5.3-Flash-GGUF/GLM-5.3-Flash-Uncensored-IQ3_XXS.gguf --checker build-local/glm5next-transport/strata-glm5next-transfer-check.exe --layers 3 11 45 --experts 0 1 2 3 4 5 6 287 --modes mmap native auto --chunk-bytes 262161 --timeout 120 --output docs/GLM53/GLM53_FLASH_IQ3_XXS_GPU_TRANSFER.json
  .venv\Scripts\python.exe -m tools.check_glm5next_transfer --gguf H:/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL-00001-of-00004.gguf --checker build-local/glm5next-transport/strata-glm5next-transfer-check.exe --layers 3 11 45 --experts 0 1 2 3 4 5 6 287 --modes mmap native auto --chunk-bytes 262161 --timeout 120 --output docs/GLM53/GLM53_FLASH_UD_Q3_K_XL_GPU_TRANSFER.json
  ```

  Обе команды exit 0, `status=pass`; по **72 матрицы × 3 режима**, всего
  **432 сравнения реальных payloads**. Слои 3/11 — main, 45 — MTP weights,
  не исполнение MTP; восемь IDs, для каждого вся тройка gate/up/down.

  | Профиль | Types выбранных матриц | Байт за режим H2D (= D2D) | Chunks за режим |
  |---|---|---:|---:|
  | Uncensored-IQ3_XXS | IQ2_S/IQ3_S/IQ4_XS/Q2_K/Q3_K | 238026752 | 936 |
  | UD-Q3_K_XL | IQ3_XXS/IQ4_XS/Q3_K/Q4_K/Q6_K | 308805632 | 1200 |

  Максимальный выбранный offset — **120531408704** байт в одиночном файле и
  **48237218240** в shard UD-Q3_K_XL. Во всех native/auto-прогонах источник читался
  через native path, в mmap — через mapped memory. Это не физические чтения SSD.
  Guard regions по 37 байт сохранились; chunks 262161 байт включают неполные хвосты.
  Binary SHA-256: `1dc8b4460e641beae81d62db1ef5c7ca8be2239249c0b9f95724f78fe1c29e58`.
  Результаты и точные ranges:
  [IQ3_XXS](GLM53_FLASH_IQ3_XXS_GPU_TRANSFER.json),
  [UD-Q3_K_XL](GLM53_FLASH_UD_Q3_K_XL_GPU_TRANSFER.json).
- **Python:** `.venv/Scripts/python.exe -m unittest tools.test_glm5next_transfer_check`
  — exit 0, 8 новых тестов: формат/Unicode/offsets, missing/reordered/results/counters,
  source mode, child failure/timeout, limits/path escape, provenance/dedup/main-MTP,
  изменение файла/checker, stale report и защита от перезаписи/hardlink.
  Это проверки runner со scripted child, не GPU evidence.
  Затем из того же каталога:

  ```text
  .venv/Scripts/python.exe -m unittest tools.test_glm5next_transfer_check tools.test_glm5next_build tools.test_glm5next_expert_plan tools.test_glm5next_loader_contract tools.test_setup_glm5next tools.test_deepseek4 tools.test_shards
  ```

  Exit 0, **56 тестов** без skips. Прежний ResourceWarning `strata_pack.py:351`
  сохраняется; отдельного Python log нет. `git -c safe.directory=C:/work/git/my-repos/Strata diff --check`
  — exit 0.
- **Ограничения:** проверены выбранные диапазоны, не все экспертные веса. Hashes
  заголовков не удостоверяют полный payload; authenticity весов не проверялась.
  Нет деквантования/MoE/logits, GLM graph hooks, cache keys/invalidation,
  исполнения MTP, сессий или замера скорости. Checker синхронизирует GPU после
  каждой матрицы для сравнения, поэтому не является benchmark. Полный DeepSeek/Qwen
  inference не запускался; общий runtime в этом цикле не менялся.
- **Следующий шаг:** P0.3b → P2.1b.2; после сборки GLM graph связать native planner
  и cache keys с общим pipeline (P3.2b), используя сохранённые byte reports как
  контроль транспорта, затем подтвердить численную корректность.

### P3.2b.1-01 — 2026-10-04, Asia/Yekaterinburg — GPU cache с полными ключами и CUDA lifetime

- **Статус:** DONE для компонента P3.2b.1. Исполнитель Codex, цикл 4.
  Весь P3.2b/P3 остаётся IN_PROGRESS: GLM loader/router/graph ещё не подключён.
  Ветка `dev`, база `64e817c1c0260861723411aa2fea2afb0455c107`, дерево перед работой
  чистое. P3.4b уже в HEAD; новые изменения не закоммичены. Сеть не перепроверялась.
- **Реализация:** `backends/glm5next/expert_cache.hpp` содержит `ExpertKey` и
  `ExpertCache`. Native key повторяет 12 полей Python reference contract:
  полная model identity, generation, main/MTP, layer, expert, gate/up/down, quant,
  columns/rows, shard, offset и bytes. Структурно неверные ключи отклоняются;
  quant geometry проверяет loader, а не cache. Генерацию при каждом reload должен
  менять вызывающий код; архитектура/путь или mmap pointer не служат полной identity.
- **Поведение:** byte budget и LRU среди доступных entries; `Lease` защищает
  матрицы текущего плана. Upload callback вызывается только на miss и может
  использовать общий `StrataExpertPipeline::transfer`; source owner удерживается
  через shared ownership. Ready event упорядочивает hit на другом CUDA stream.
  Перед освобождением lease после последнего consumer записывается отдельный event;
  незавершённые consumers и uploads исключают вытеснение. Если места нет, cache
  возвращает пустой lease для uncached fallback, не ждёт pending events при поиске
  места и не превышает установленный бюджет новыми allocations.
- **Reload/ошибки:** `invalidate(model, generation)` сохраняет другие модели и
  поколения. Активные leases остаются валидными и учитываются в resident bytes
  после invalidation и даже удаления cache object. Явная invalidation/teardown
  может ждать GPU; ошибка uploader дренирует stream и не публикует частично
  загруженную матрицу. Снижение бюджета возвращает false, пока leases/events
  мешают trim; можно повторить после завершения. Счётчики: hits/misses/bypasses/
  evictions/invalidations. Общий transport и код DeepSeek/Qwen не менялись.
- **Файлы:** новый header, `backends/glm5next/test_expert_cache.cpp`, CMake target
  `glm5next_expert_cache_test` под прежним opt-in `STRATA_GLM_TRANSPORT_TESTS_CUDA`;
  GLM README с правилами API/lifetime, план/статус и
  [GLM53_FLASH_GPU_CACHE_TESTS.txt](GLM53_FLASH_GPU_CACHE_TESTS.txt).
- **Проверки:** рабочая директория `C:\work\git\my-repos\Strata`.
  `cmd /c build-local\test-glm5next-transport.cmd` — exit 0; команды внутри
  совпадают с P3.4a-01 (`vcvars64.bat`, CUDA PATH, CMake configure/build и verbose CTest).
  Вывод сборки направлен в `build-local/glm5next-cache-build.log`; CTest log сохранён
  по ссылке выше. **6/6 CTest**, без skips; новый cache test выполнен на **RTX 5090**.
  MSVC **19.44.35222.0**, SDK **10.0.26100.0**, toolkit **13.0.48**;
  повторный byte test сообщает CUDA runtime/driver **13000/13000**.
  Новый тест подтвердил:
  - Все 12 полей различают GPU entries (базовый ключ + 12 независимых изменений);
    повторные hits сохраняют разные исходные bytes без повторного uploader.
  - Model/generation invalidation не удаляет другой model/generation; main/MTP,
    projection, layout и offsets >4 ГиБ не смешиваются. Неверные ключи отклоняются.
  - LRU, budget shrink, защита активного плана, bypass и deferred trim.
  - Искусственно задержанный upload и consumer в двух streams: hit ждёт ready;
    освобождённый на CPU lease не разрешает преждевременное GPU-вытеснение;
    miss возвращает bypass до истечения 3-секундного контрольного gate timeout.
    Это функциональная проверка отсутствия host wait, не benchmark.
  - Удержание source owner до retirement, reload с ещё живым lease старого
    поколения, сохранение accounting, teardown cache при живом lease.
  - Ошибка uploader после enqueue не оставляет entry/VRAM/source owner;
    следующий upload на том же ключе проходит.
  - Общий pipeline загрузил **1048593 байта** с chunk **65553**; повторный cache hit
    побайтово совпал с исходником, H2D counter остался равен одному upload.
  Повторно прошли прежние 432 synthetic GPU comparisons, native-reader и parser fixtures.
- **Python-регрессии:**

  ```text
  .venv/Scripts/python.exe -m unittest tools.test_glm5next_expert_plan tools.test_glm5next_build
  ```

  Exit 0, **11 тестов**, без skips. Отдельный Python log не сохранён.
  `git -c safe.directory=C:/work/git/my-repos/Strata diff --check` — exit 0.
- **Граница результата:** standalone CUDA component и synthetic fixtures; этот
  cache ещё не используется модельным GLM graph или P3.4b real-range checker.
  Нет cached parity реальных весов, kernel/logit checks, частотного допуска,
  переиспользования allocations, общего VRAM controller или измерения скорости.
  Byte budget считает payload allocations, не CUDA-event/allocator overhead.
  Один host owner и CUDA device на cache; lease consumers должны ставиться на
  stream этого lease до release. Linux/HIP/multi-GPU и отказ CUDA allocator не
  проверялись. Ошибки численного kernel execution также вне этой проверки.
- **Следующий шаг:** P0.3b → P2.1b.2; затем GLM loader/graph должен создавать полные
  ключи с новым generation при reload и интегрировать `ExpertCache::get` с
  pipeline miss/bypass. До model graph можно отдельно расширить P3.4b checker
  для real-range cached parity и reload/forced eviction; итоговый P3.2b не закрывать
  без проверки этой связки и реального пути inference.

### P3.5a-01 — 2026-10-04, Asia/Yekaterinburg — Частотный допуск в GPU cache

- **Статус:** DONE для отдельного компонента P3.5a; P3 остаётся IN_PROGRESS.
  Исполнитель Codex, цикл 5. Ветка `dev`, база
  `a752b933a8accc300b901e7ff139c9459894b719`, исходное дерево чистое;
  перечисленные изменения не закоммичены. P0.3b и сетевой блокер не перепроверялись.
- **Реализация:** алгоритм DeepSeek вынесен в `backends/common/expert_frequency.hpp`
  как шаблон ключа/hash; прежний header стал compatibility include, pointer-key API
  и правила счёта DeepSeek сохранены. GLM использует все 12 полей `ExpertKey`.
  `ExpertCache(limit, {true, decay_period, max_keys})` включает frequency admission;
  прежний конструктор сохраняет LRU без таблицы истории.
- **Поведение:** учитываются hits и валидные misses, включая bypass. Счётчики
  насыщаются на 255, делятся на два каждый период обращений; при лимите ключей
  новый ключ очищает историю без изменения residency. При наличии места miss
  допускается сразу. При нехватке места его score должен быть не ниже score
  каждого необходимого LRU victim. Выбор всех victims предшествует вытеснению:
  отказ по частоте или нехватка доступных bytes не удаляет часть кэша.
  Leases/pending events защищены. Отказ не вызывает uploader, возвращает пустой
  lease для полного uncached transfer. Trim бюджета игнорирует score.
  Invalidation удаляет также nonresident history только нужных model/generation.
  Добавлены `admissions`, `admission_rejects` и `history_size()`.
- **Файлы:** общий header и его DeepSeek wrapper; `backends/glm5next/expert_cache.hpp`,
  `test_expert_cache.cpp`, `CMakeLists.txt`; README общего/GLM модуля, план/статус и
  [CTest log](GLM53_FLASH_FREQUENCY_CACHE_TESTS.txt).
- **Проверки:** рабочая директория `C:\work\git\my-repos\Strata`.
  `cmd /c build-local\test-glm5next-transport.cmd` — exit 0, **7/7 CTest**, без skips.
  Скрипт использует `vcvars64.bat`, CUDA DLL PATH и команды из GLM README:
  `cmake -S backends/glm5next -B build-local/glm5next-transport -G Ninja
  -DSTRATA_GLM_PROTOCOL_TESTS_ONLY=ON -DSTRATA_GLM_TRANSPORT_TESTS_CUDA=ON
  -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=C:/work/git/my-repos/Strata/build-local/cuda-13.0
  -DCMAKE_MAKE_PROGRAM=C:/Users/Aleks/AppData/Local/Programs/Python/Python312/Scripts/ninja.exe`,
  `cmake --build build-local/glm5next-transport --config Release`,
  `ctest --test-dir build-local/glm5next-transport -C Release -V --no-tests=error`.
  MSVC **19.44.35222.0**, Windows SDK **10.0.26100.0**, GPU **RTX 5090**,
  CUDA runtime/driver **13000/13000**. Полный локальный build log:
  `build-local/glm5next-frequency-build.log`; CTest output сохранён по ссылке выше.
  Подтверждены отдельные key fields, selective invalidation, cold rejection,
  promotion после повторных misses, ties, decay, bound, trim, counters,
  mixed-size multi-victim admission, защита leases и pending events в двух streams.
  Bypass доставил **1048593 байта** через общий pipeline с chunk **65553**;
  сохранённая hot matrix дала hit с исходными bytes без повторной загрузки.
  Повторно прошли 432 synthetic GPU comparisons, reader/parser/protocol fixtures
  и неизменённый тест истории DeepSeek (decay/saturation/bound).
- **Python:** `.venv/Scripts/python.exe -m unittest tools.test_glm5next_expert_plan tools.test_glm5next_build`
  — exit 0, **11 тестов**, без skips; отдельный Python log не сохранялся.
- **Ограничения:** проверены synthetic bytes и отдельный cache API, не GLM graph,
  logits или реальные cached weights. Скорость генерации и overhead политики не
  измерялись. Значения 4096/131072 не настроены по GLM workload; лимит ограничивает
  число ключей, не объём строк/host allocator. Main/MTP имеют отдельные ключи, но
  общий clock decay. Под давлением выполняется сортировка доступных LRU victims;
  allocation reuse и общий VRAM controller отсутствуют. Linux/HIP/multi-GPU,
  полные Qwen/DeepSeek inference и сборка Unsloth в этом цикле не запускались.
- **Следующий шаг:** P0.3b → P2.1b.2. Для cache до model graph можно расширить
  P3.4b checker на real-range cached parity/reload/forced eviction, затем подключить
  loader/router к `ExpertCache::get` и uncached bypass. Весь P3.2b/P3 не закрывать
  без численной проверки реального inference и измерения перекрытия.

### P3.4c-01 — 2026-10-04, Asia/Yekaterinburg — Реальные GGUF через GPU cache

- **Статус:** DONE только для P3.4c, P3/P3.2b остаются IN_PROGRESS. Codex, цикл 6.
  Ветка `dev`, база `8b1a95c25092ad04e81fc9dc09ffc148387351db`, исходное дерево
  чистое; изменения этого цикла не закоммичены. P0.3b/сеть не перепроверялись.
- **Реализация:** `tools/check_glm5next_transfer.py --cache-check` передаёт native
  checker полные поля cache key через отдельный `GLM_CACHE_RANGES_V1`; прежний
  manifest и обычные проверки сохранены. Каждый диапазон получает отдельный LRU
  cache с бюджетом ровно одной матрицы. Проверяются cold generation 1, hit на
  другом CUDA stream, generation 2 с принудительным вытеснением, повторная загрузка
  generation 1 с вытеснением generation 2, затем invalidation и generation 3.
  Все пять результатов сравниваются с независимым stdio baseline; verification
  destinations имеют guards. Miss вызывает общий pipeline с retained mapping owner;
  hit не вызывает uploader и не добавляет source/H2D bytes. Cache и transport
  counters проверяются native-кодом по этапам и Python runner по всем строкам отчёта.
- **Identity:** SHA-256 списка наблюдаемых shard paths, sizes, mtimes и header hashes.
  Это идентификатор набора файлов в checker, не полный hash весов и не production
  loader/session fingerprint. JSON schema 2 хранит identity scope, сценарий,
  executable hash, диапазоны, GPU и counters. Ошибка заменяет прежний success report.
- **Файлы:** `backends/glm5next/check_expert_ranges.cpp`, GLM README,
  `tools/check_glm5next_transfer.py`, `tools/test_glm5next_transfer_check.py`, план,
  статус, два JSON ниже и [CTest log](GLM53_FLASH_REAL_CACHE_TESTS.txt).
- **Сборка и fixtures:** рабочая директория всех команд `C:\work\git\my-repos\Strata`.
  `cmd /c build-local\test-glm5next-transport.cmd` — exit 0, **7/7 CTest**, без skips;
  configure/build/CTest и compiler environment те же, что в P3.5a-01.
  MSVC **19.44.35222.0**, SDK **10.0.26100.0**, RTX 5090, CUDA runtime/driver
  **13000/13000**. Полный локальный build log —
  `build-local/glm5next-cached-ranges-build.log`. Parser fixtures проверили новые
  поля, shard/path agreement, malformed identity/branch/projection, missing fields,
  64-bit offsets и переполнение; прежние 432 synthetic GPU comparisons, cache
  fixtures, reader, tokenizer protocol и DeepSeek frequency regression прошли.
- **Python:** `.venv/Scripts/python.exe -m unittest tools.test_glm5next_transfer_check tools.test_glm5next_expert_plan`
  — exit 0, **20 тестов**, без skips. Добавлены точный manifest/counters contract,
  отказ при пропущенных этапах, лишнем H2D, неверных counters и старом protocol,
  валидация ключей, identity по изменённому источнику и замена stale success report.
  Отдельный Python log не сохранялся.
- **Реальные веса:** перед командами CUDA DLL PATH дополнен
  `build-local/cuda-13.0/bin` и `build-local/cuda-13.0/bin/x64`.

  ```text
  .venv/Scripts/python.exe -m tools.check_glm5next_transfer --gguf H:/GLM-5.3-Flash-GGUF/GLM-5.3-Flash-Uncensored-IQ3_XXS.gguf --checker build-local/glm5next-transport/strata-glm5next-transfer-check.exe --layers 3 11 45 --experts 0 1 2 3 4 5 6 287 --modes mmap native auto --chunk-bytes 262161 --timeout 120 --cache-check --output docs/GLM53/GLM53_FLASH_IQ3_XXS_GPU_CACHE.json
  .venv/Scripts/python.exe -m tools.check_glm5next_transfer --gguf H:/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL-00001-of-00004.gguf --checker build-local/glm5next-transport/strata-glm5next-transfer-check.exe --layers 3 11 45 --experts 0 1 2 3 4 5 6 287 --modes mmap native auto --chunk-bytes 262161 --timeout 120 --cache-check --output docs/GLM53/GLM53_FLASH_UD_Q3_K_XL_GPU_CACHE.json
  ```

  Обе команды — exit 0, `status=pass`:
  [IQ3_XXS](GLM53_FLASH_IQ3_XXS_GPU_CACHE.json),
  [UD-Q3_K_XL](GLM53_FLASH_UD_Q3_K_XL_GPU_CACHE.json).
  В каждом из трёх режимов на модель: **72 матрицы, 360 сравнений bytes, 72 hits,
  288 misses/admissions, 144 evictions, 72 invalidations**, bypass=0;
  hit source/H2D=0. Всего **2160 сравнений**, gate/up/down, main/MTP, все восемь
  routed quant types двух профилей. Source/H2D/D2D pipeline на режим:
  IQ3_XXS **952107008 байт / 3744 chunks**, UD-Q3_K_XL **1235222528 байт / 4800 chunks**.
  Дополнительные cache-to-verification D2D copies в pipeline counters не входят.
  Native/mmap counters описывают способ доступа, не физические SSD reads.
- **Регрессия обычного checker:** первая команда выше повторена без `--cache-check`,
  с `--output build-local/glm5next-uncached-regression.json` — exit 0, 72 матрицы
  в каждом из трёх режимов. Старые P3.4b JSON не перезаписывались.
  `git -c safe.directory=C:/work/git/my-repos/Strata -c core.safecrlf=false diff --check` — exit 0.
- **Граница результата:** выбранные реальные payloads, не все веса. Смена generation
  симулирует loader key; файл не remap/reload. Кэш пересоздаётся для каждой матрицы,
  mixed-range планирование и frequency admission на реальных весах не проверены.
  Сравнения синхронизируют streams: это не benchmark и не проверка перекрытия.
  Нет деквантования, численных logits, GLM graph integration, настоящего MTP,
  отмены inference, сессий или измеренной скорости. Полные Qwen/DeepSeek inference,
  Linux/HIP/multi-GPU не запускались. Общий runtime и политика cache не менялись.
- **Следующий шаг:** P0.3b — получить архив закреплённого кандидата, собрать
  llama/oracle/CUDA targets по `backends/glm5next/README.md`, сохранить hash/build
  evidence, затем P2.1b.2. При интеграции GLM loader/graph использовать real-range
  cache reports как контроль транспорта, отдельно проверять численные outputs.

### P3.1b-01 — 2026-10-04, Asia/Yekaterinburg — Native planner выбранных экспертов

- **Статус:** DONE для компонента P3.1b; P3 остаётся IN_PROGRESS. Codex, цикл 7.
  Ветка `dev`, база `b4cc3d2ce313e051cae9517085e1292838fe5077`, исходное дерево
  чистое; изменения не закоммичены. Поиск `*86ebfef*` в build-local/third_party
  не нашёл архив кандидата; сеть и P0.3b не перепроверялись.
- **Реализация:** `backends/glm5next/expert_plan.hpp` принимает model identity,
  generation, layer/main/total/leading-dense block counts, три tensor layouts,
  shard geometry и router IDs. Проверяет восемь routed quant types, целые quant
  rows, tensor byte counts, транспонированную down shape, expert counts, file bounds,
  перекрытия тензоров и overflow. Dense layers и неверные IDs отклоняются.
  Metadata-only shard без padding допустим, пока на него не ссылается тензор.
- **План:** dedup IDs в порядке первого обращения; на каждый уникальный ID
  возвращается полная тройка gate/up/down с 12 полями cache key. Порядок входных
  tensor descriptors не меняет порядок проекций. 64-bit offsets не округляются
  между экспертами. Пустой route list даёт пустой план после валидации layouts.
  `max_routes=4096` по умолчанию ограничивает вход, включая повторы; выход не более
  трёх ключей на ID. Этот dedup относится к доставке весов: graph обязан сохранить
  отдельные router weights каждого токена. Планировщик не читает payload и не
  владеет mmap, файлами, GPU allocations или событиями.
- **Общий ключ:** прежние `ExpertKey`/`ExpertKeyHash` вынесены без изменения
  семантики из cache в `expert_key.hpp`, не зависящий от CUDA. И cache, и planner
  используют одну структуру. Остальной cache и общий транспорт не менялись.
- **Интеграция проверки:** `test_expert_bytes.cpp` теперь получает диапазоны из
  native planner по IDs с повторами; независимые fixture offsets/bytes остаются
  эталоном. Все восемь уникальных экспертов и их 24 матрицы доставляются через
  общий pipeline. Это интеграция с transport fixture, не с модельным router.
- **Файлы:** новые `expert_key.hpp`, `expert_plan.hpp`, `test_expert_plan.cpp`;
  `expert_cache.hpp`, `test_expert_bytes.cpp`, CMake и README в `backends/glm5next`;
  план/статус и [CTest log](GLM53_FLASH_NATIVE_PLAN_TESTS.txt).
- **Проверки:** рабочая директория `C:\work\git\my-repos\Strata`.
  `cmd /c build-local\test-glm5next-transport.cmd` — exit 0, configure/build/CTest
  с параметрами из P3.5a-01. Затем после окончательной правки выполнена чистая
  пересборка всех потребителей нового header:

  ```text
  cmd /c build-local\check-glm-native-plan.cmd
  ```

  Скрипт вызывает `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`,
  добавляет `build-local/cuda-13.0/bin` и `bin/x64` в PATH, выполняет:

  ```text
  cmake --build build-local/glm5next-transport --config Release --clean-first
  ctest --test-dir build-local/glm5next-transport -C Release -V --no-tests=error
  ```

  Exit 0, **8/8 CTest**, без skips. Полный локальный лог:
  `build-local/glm5next-native-plan-clean-build.log`; CTest output сохранён выше.
  MSVC **19.44.35222.0**, SDK **10.0.26100.0**, RTX 5090, CUDA runtime/driver
  **13000/13000**. CPU test не включает/не линкует CUDA:
  **384 ключа** по независимым fixtures, восемь типов, main/MTP, split shards,
  metadata-only, offsets >4 ГиБ, точный EOF, dedup и malformed layouts/routes.
  GPU: **18 случаев / 432 матрицы**, mmap/native/auto × prefill/decode × три mixed
  группы типов; guards, неполные chunks, два consumer streams. Счётчики source,
  H2D/D2D/chunks совпали с планом. Native case по-прежнему закрывает доступ к mmap
  через PAGE_NOACCESS и удаляет registry entry после старта: jobs удерживают handles.
  Также прошли прежние cache/frequency/lifetime tests, reader compatibility,
  range parser и tokenizer protocol fixtures. Python-тесты в этом цикле не запускались.
- **Ограничения:** новая native адресация проверена на synthetic descriptors/bytes,
  не по полной GLM loader metadata и не на реальном router output. P3.4c JSON
  сохраняются как прежнее evidence другого пути (Python planner). Нет численных
  kernels/logits, graph hooks, общей политики VRAM, отмены графа, измерения скорости
  или overlap. Planner не заменяет GGUF inspector; loader должен передавать
  проверенные metadata и удерживать реальные источники. Полные Qwen/DeepSeek
  inference, Unsloth build, Linux/HIP/multi-GPU не проверялись.
- **Следующий шаг:** P0.3b → P2.1b.2; после появления GLM loader/router заполнить
  `ExpertLayerLayout` из реальных tensor descriptors, связать `plan_experts` с
  cache leases и pipeline, проверить outputs. До этого можно отдельно проверить
  совпадение native/Python планов по заголовкам обоих реальных GGUF.

### P3.3a-01 — 2026-10-04, Asia/Yekaterinburg — Удержание источников и отмена transport plan

- **Статус:** DONE для отдельного компонента P3.3a; P3 остаётся IN_PROGRESS.
  Codex, цикл 8; ветка `dev`, база `9a6787e78239d77112ede43e09709f1db0d7a06b`.
  Исходное дерево чистое, изменения не закоммичены. Поиск `*86ebfef*` в
  build-local/third_party ничего не нашёл; сеть и P0.3b не перепроверялись.
- **Реализация:** `backends/glm5next/expert_transport.hpp` добавляет `ExpertTransport`
  и `ExpertSourceView`. `begin` связывает полные ключи плана с offset-zero mappings
  по model/generation/shard, проверяет owner, pointer/range overflow, дубликаты,
  лимиты **12288 матриц / 1048576 chunks** до отправки jobs. Ссылки на используемые
  source owners сохраняются в адаптере. Owner обязан удерживать mapping и native
  registration; один file handle не гарантирует жизнь mmap.
- **Жизненный цикл:** `transfer` принимает только следующий индекс и ненулевой
  destination; nested begin и неправильный порядок не потребляют jobs. `cancel`
  дренирует/отменяет остаток и затем освобождает sources, повторный cancel допустим.
  `finish` требует полного потребления; при неполном плане сначала дренирует остаток,
  затем сообщает ошибку. Destructor при обычном выходе/исключении сначала завершает
  workers и GPU ring, затем удаляет source owners. Ошибка самого pipeline удаляет
  его до освобождения источников; после неё адаптер нужно создать заново.
- **Граница API:** один host owner/device, вызовы сериализованы. Сигнал отмены
  запроса должен обрабатываться этим owner между transfers; конкурентный вызов
  cancel из другого потока не поддерживается. Destination memory, streams и cache
  leases остаются ответственностью вызывающего graph до завершения consumers.
  Адаптер использует прежние slot events общего pipeline, не обещает мгновенную
  отмену in-flight native read. Cache-hit filtering/admission и GLM graph hooks
  пока отсутствуют. Общий pipeline, cache и DeepSeek/Qwen runtime не менялись.
- **Файлы:** новый header и `backends/glm5next/test_expert_transport.cpp`, CMake,
  GLM README, план/статус и [CTest log](GLM53_FLASH_TRANSPORT_CANCEL_TESTS.txt).
- **Проверки:** рабочая директория `C:\work\git\my-repos\Strata`.
  `cmd /c build-local\test-glm5next-transport.cmd` — exit 0, **9/9 CTest**, без skips.
  Configure/build/CTest параметры и `vcvars64.bat`/CUDA PATH совпадают с P3.5a-01;
  итоговые команды `cmake --build build-local/glm5next-transport --config Release`
  и `ctest --test-dir build-local/glm5next-transport -C Release -V --no-tests=error`.
  Локальный build log: `build-local/glm5next-cancel-build.log`; CTest output сохранён
  по ссылке выше. MSVC **19.44.35222.0**, SDK **10.0.26100.0**, RTX 5090,
  CUDA runtime/driver **13000/13000**.
- **Новые fixtures:** native planner выдаёт тройки gate/up/down по восьми уникальным
  IDs; mixed IQ2_S/IQ3_S/IQ4_XS, chunk **4093**, реальные временные файлы/mappings
  с synthetic bytes. Для mmap/native/auto × prefill/decode: отмена после **0/1/7**
  матриц и полное завершение **24**, всего **24 плана / 192 сравнения доставленных
  матриц** плюс guards и нетронутый suffix. До transfers внешние владельцы mapping
  удаляются; weak references подтверждают жизнь источника до drain и удаление после.
  Перед отменой нулевого prefix проверено заполнение ring; `unused>0` подтверждает
  отброшенный prefetch. Новый generation/source успешно стартует после каждого cancel.
  Native case использует PAGE_NOACCESS на mmap. Отдельно прошли exception teardown
  с prefetched ring, invalid bindings, model/generation isolation, matrix/chunk
  limits, ordered transfer, неполный finish и пустой план.
- **Регрессии:** прежние 384 native keys, 432 GPU matrix comparisons, cache lifetime/
  frequency tests, native reader/DeepSeek compatibility, range parser и mock tokenizer
  protocol прошли. `git -c safe.directory=C:/work/git/my-repos/Strata -c core.safecrlf=false diff --check`
  — exit 0. Python-тесты в этом цикле не запускались.
- **Ограничения:** отмена проверена на отдельном transport adapter, не в GLM
  inference или HTTP disconnect. Actual I/O/CUDA failure injection не выполнялась;
  протестированные ошибки относятся к binding/order/незавершённому плану и внешнему
  исключению. Нет численных kernels/logits, MTP/sessions, измерения скорости,
  latency отмены или overlap timeline. Linux/HIP/multi-GPU, полный DeepSeek/Qwen
  inference и Unsloth build не запускались.
- **Следующий шаг:** P0.3b → P2.1b.2. После появления GLM graph связать native
  planner с удерживаемыми cache leases и ordered miss/bypass plan через
  `ExpertTransport`; проверить отмену реальных prefill/decode и численную parity.

### P3.5b-01 — 2026-10-04, Asia/Yekaterinburg — Предварительная защита матриц текущего плана

- **Статус:** DONE для компонента P3.5b; P3 остаётся IN_PROGRESS. Codex, цикл 9.
  Ветка `dev`, база `9cd57f2b74c8ec275b591d90b495a0c178e7957c`, исходное дерево
  чистое; изменения не закоммичены. P0.3b и его сетевой блокер не перепроверялись.
- **Проблема:** consumer lease защищает entry лишь после `get`. Если первый ключ
  плана отсутствует, его загрузка может вытеснить resident entry, который потребуется
  позже в том же плане. Нужна защита всей уже resident части до обработки misses.
- **Реализация:** `ExpertCache::protect_plan(keys)` возвращает move-only `PlanPins`.
  Проверяются все ключи и лимит 12288; дубликаты удерживаются один раз, отсутствующие
  ключи не выделяют память. Pins не предоставляют указателей на GPU bytes, не ставят
  CUDA work, не ждут ready events, не меняют LRU/frequency или access counters.
  Они удерживают allocation/source через shared ownership и исключают entry из
  кандидатов на вытеснение. Overlapping guards могут независимо завершаться;
  `release()` идемпотентен, destructor снимает защиту при исключении/отмене.
- **Lifetime:** actual consumers обязаны брать обычный `get()` lease с ready wait
  и consumer event. Leases вновь загруженных misses также нужно удерживать до конца
  плана. Invalidation удаляет lookup entry, но его bytes/source остаются учтены,
  пока жив pin/lease; reload не превышает бюджет. Pins могут пережить cache object.
  Снятие последней ссылки на retired entry может ждать GPU. Это защита residency,
  не snapshot и не доступ к старому generation после reload; loader должен пересоздать
  план при смене поколения. API остаётся для одного host owner/device.
- **Файлы:** `backends/glm5next/expert_cache.hpp`, `test_expert_cache.cpp`, GLM README,
  план/статус и [CTest log](GLM53_FLASH_PLAN_PINS_TESTS.txt).
- **Проверки:** рабочая директория `C:\work\git\my-repos\Strata`.
  `cmd /c build-local\check-glm-native-plan.cmd` — exit 0. Скрипт и compiler/CUDA PATH
  описаны в P3.1b-01; выполнены:

  ```text
  cmake --build build-local/glm5next-transport --config Release --clean-first
  ctest --test-dir build-local/glm5next-transport -C Release -V --no-tests=error
  ```

  Чистая пересборка всех targets; **9/9 CTest**, без skips. MSVC **19.44.35222.0**,
  SDK **10.0.26100.0**, GPU **RTX 5090**, CUDA runtime/driver **13000/13000**.
  Полный локальный log: `build-local/glm5next-plan-pins-build.log`; CTest сохранён выше.
  Новые fixtures подтверждают:
  - Ранний miss C вытесняет неплановый B, сохраняя поздний hit A с исходными bytes
    и без повторного uploader; проверено отдельно для LRU и frequency admission.
  - Plan pin вместе с lease нового miss не допускает третью allocation и защищает
    от budget shrink; после release trim полностью освобождает VRAM.
  - Dedup/absent keys, неизменность counters/history size и LRU, пустой план,
    неверные ключи и превышение лимита; неуспешная validation не оставляет pins.
  - Overlapping pins, generation invalidation с сохранением resident accounting,
    bypass нового generation до освобождения старого и последующий успешный reload.
  - Cache teardown при живых pin/consumer lease, исходные GPU bytes и exception
    cancellation без оставшейся защиты. Pending upload не задерживает protect_plan;
    фактический consumer по-прежнему получает правильные bytes через ready event.
  Прежние native planner, 432 GPU matrix comparisons, transport cancel/restart
  (24 плана / 192 матрицы), reader/DeepSeek compatibility, parser и mock tokenizer
  fixtures также прошли. Python-тесты в этом цикле не запускались.
- **Ограничения:** synthetic GPU fixtures и отдельный cache API. GLM graph пока
  не вызывает protect_plan; нет проверки всей связки planner/cache/miss transport
  на реальной модели, logits, MTP, скорости или overlap timeline. Pins занимают
  host metadata и удерживают VRAM до release; общего VRAM controller ещё нет.
  Полные Qwen/DeepSeek inference, Unsloth build, Linux/HIP/multi-GPU не запускались.
  Общие transport/runtime модули и политика DeepSeek/Qwen не менялись.
- **Следующий шаг:** P0.3b → P2.1b.2. Для интеграции перед первым miss создавать
  `PlanPins`, получать consumer leases, строить ordered miss/bypass plan и снимать
  защиту на всех выходах графа; проверить численную parity и отмену реального GLM.

### P3.2b.2-01 — 2026-10-04, Asia/Yekaterinburg — Cache-aware dispatch выбранных матриц

- **Статус:** DONE для отдельного компонента P3.2b.2; весь P3.2b/P3 остаётся
  IN_PROGRESS. Codex, цикл 10. Ветка `dev`, база
  `0dbd97de55a75190405436add4929251deecd4f2`, исходное дерево чистое; изменения
  не закоммичены. P0.3b и сетевой блокер в этом цикле не перепроверялись.
- **Реализация:** новый `backends/glm5next/expert_dispatch.hpp` связывает native
  keys, `ExpertCache::PlanPins`, consumer leases и `ExpertTransport`. Scope до
  первого miss защищает все resident entries и строит ordered transport plan
  только для отсутствующих ключей. Новый `ExpertCache::resident` — planning probe
  без изменения access counters/LRU/frequency и без CUDA consumer work.
  Фактический `get` вызывается один раз на копируемую матрицу: hit копируется из
  cache, admitted miss доставляется pipeline в cache и затем в destination,
  bypass доставляется полностью прямо в destination. Все leases удерживаются
  до конца scope. All-hit plan работает без source views и без source/H2D.
- **Отмена/ошибки:** cancel, неполный finish и exception destructor сначала
  дренируют остаток miss plan, затем освобождают leases/pins/source views.
  Duplicate keys, nested scope и неверный порядок отклоняются. Изменившаяся
  residency диагностируется с отменой scope; concurrent reload не поддерживается.
  Cache и transport должны пережить scope и не меняться извне во время исполнения.
  Destination memory/streams остаются живы до завершения GPU consumers.
- **Файлы:** новый header, `expert_cache.hpp`, `test_expert_transport.cpp`, GLM
  README, план/статус и [CTest log](GLM53_FLASH_DISPATCH_TESTS.txt).
- **Проверки:** рабочая директория `C:\work\git\my-repos\Strata`.
  `cmd /c build-local\check-glm-native-plan.cmd` — exit 0. Compiler environment и
  CUDA PATH те же, что в P3.1b-01; скрипт выполнил:

  ```text
  cmake --build build-local/glm5next-transport --config Release --clean-first
  ctest --test-dir build-local/glm5next-transport -C Release -V --no-tests=error
  ```

  **9/9 CTest**, без skips; MSVC **19.44.35222.0**, SDK **10.0.26100.0**,
  RTX 5090, CUDA runtime/driver **13000/13000**. Полный локальный build log:
  `build-local/glm5next-dispatch-build.log`; CTest output сохранён выше.
  Новые native-plan fixtures прошли в **шести конфигурациях**:
  mmap/native/auto × LRU/frequency, prefill reader policy, synthetic mixed
  IQ2_S/IQ3_S/IQ4_XS и guarded destinations. Подтверждены:
  - Early miss C вытесняет неплановый B, поздний hit A сохраняется, D bypass
    доставляет все bytes. Source/H2D delta равна только C+D; cache counters
    показывают 1 hit, 2 misses, 1 admission, 1 eviction и 1 bypass.
  - All-hit scope не меняет source/H2D/group counters и не требует источников.
    При нулевом бюджете все матрицы доставляются через bypass без cache allocations.
  - Frequency rejection также доставляет полную матрицу через bypass, с проверкой
    admission_rejects и H2D delta. Native mapping защищён PAGE_NOACCESS.
  - Cancel после prefix освобождает защиту для budget trim; invalid source identity,
    duplicate/nested/order errors, incomplete finish, invalidation detection,
    exception cleanup и empty scope не оставляют активный transport plan.
  Прежние 384 native keys, 432 GPU comparisons, transport cancel/restart,
  cache/pins/frequency, reader/DeepSeek compatibility, parser и tokenizer protocol
  fixtures прошли. Python-тесты в этом цикле не запускались.
  `git -c safe.directory=C:/work/git/my-repos/Strata -c core.safecrlf=false diff --check`
  — exit 0.
- **Ограничения:** dispatch копирует selected weights в destinations, не выполняет
  GLM graph и не вычисляет непосредственно из cache pointers. Pipeline D2D counter
  учитывает ring→cache/bypass; дополнительные cache→destination copies в него не
  входят. Проверка synthetic, не реальные GGUF/logits. Decode policy отдельно
  покрыта прежними transport fixtures, но новые dispatch cases используют prefill.
  Нет скорости/overlap timeline, полного runtime reload, MTP, GPU/I/O fault injection
  или полного Qwen/DeepSeek inference. Общий pipeline и DeepSeek/Qwen не менялись.
  Linux/HIP/multi-GPU и Unsloth build не запускались.
- **Следующий шаг:** P0.3b → P2.1b.2. При появлении GLM loader/router передавать
  `plan_experts` и проверенные source views в `ExpertDispatch` на copy points графа,
  затем проверить реальные cached outputs и отмену prefill/decode. До графа можно
  прогнать dispatch на выбранных реальных GGUF ranges через отдельный checker.

### P3.4d-01 — 2026-10-04, Asia/Yekaterinburg — Dispatch на реальных GGUF ranges

- **Статус:** DONE для P3.4d; P3/P3.2b остаются IN_PROGRESS. Codex, цикл 11.
  Ветка `dev`, база `f0c8dae5aec4a3a52c64462ba6687b78eb5e9ef0`, исходное дерево
  чистое; изменения не закоммичены. P0.3b/сеть в этом цикле не перепроверялись.
- **Реализация:** в `tools/check_glm5next_transfer.py` добавлен `--dispatch-check`
  с manifest `GLM_DISPATCH_RANGES_V1` и JSON schema 3. Native checker
  `backends/glm5next/check_expert_ranges.cpp` связывает реальные ranges с
  `ExpertDispatch`, cache и transport. Полные ключи, source mappings, offsets >4 GiB,
  независимый stdio baseline и guards сохранены. Старые plain/cache режимы работают.
  Runner отклоняет отсутствующие/повторные/переставленные результаты, неверные
  counts/reader policy и неполные triples; `--cache-check` и `--dispatch-check`
  взаимно исключаются. Runtime-код pipeline/cache/dispatch в этом цикле не менялся.
- **Сценарии:** каждая gate/up/down тройка задаёт A/C/D; B — те же байты C под
  generation 2. Бюджет равен A+C. Warm A/B → mixed C/A/D должен вытеснить B,
  сохранить поздний hit A и полностью доставить D через bypass. All-hit C/A
  исполняется без source views и без source/H2D/groups. Затем отмена после A
  перед D, invalidation generation 1, загрузка generation 3 и zero-budget bypass.
  На каждой конфигурации проверены 14 GPU payloads с guards, 4 hits, 10 misses,
  5 admissions, 5 bypasses, 3 evictions, 2 invalidations; после trim VRAM cache
  освобождён. Два consumer streams чередуются с синхронизацией для сравнения.
- **Реальные результаты:** по 72 матрицы на профиль, слои 3/11/45, IDs 0–6/287,
  chunk 262161 bytes. 24 тройки × LRU/frequency × prefill/decode × mmap/native/auto
  × 2 профиля = **576 сценариев / 8064 byte comparisons**, все прошли на RTX 5090,
  CUDA runtime/driver **13000/13000**. Все восемь routed quant types обоих профилей
  представлены. Reports содержат actual counters, header hashes, размеры/mtime
  файлов и SHA-256 checker `55e177374aa8dd4cd3a06e35990c4128f8478de493e571b0868f0d6863fab72e`:
  [IQ3_XXS](GLM53_FLASH_IQ3_XXS_GPU_DISPATCH.json),
  [UD-Q3_K_XL](GLM53_FLASH_UD_Q3_K_XL_GPU_DISPATCH.json).
- **Команды:** рабочая директория `C:\work\git\my-repos\Strata`. Все финальные
  команды ниже — exit 0. Для Python GPU runner PATH дополнен каталогами
  `build-local\cuda-13.0\bin` и `build-local\cuda-13.0\bin\x64`.

  ```text
  .venv\Scripts\python.exe -m unittest tools.test_glm5next_transfer_check
  cmd /c build-local\check-glm-native-plan.cmd
  .venv\Scripts\python.exe -m tools.check_glm5next_transfer --gguf H:/GLM-5.3-Flash-GGUF/GLM-5.3-Flash-Uncensored-IQ3_XXS.gguf --checker build-local/glm5next-transport/strata-glm5next-transfer-check.exe --dispatch-check --layers 3 11 45 --experts 0 1 2 3 4 5 6 287 --modes mmap native auto --chunk-bytes 262161 --timeout 120 --output docs/GLM53/GLM53_FLASH_IQ3_XXS_GPU_DISPATCH.json
  .venv\Scripts\python.exe -m tools.check_glm5next_transfer --gguf H:/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL-00001-of-00004.gguf --checker build-local/glm5next-transport/strata-glm5next-transfer-check.exe --dispatch-check --layers 3 11 45 --experts 0 1 2 3 4 5 6 287 --modes mmap native auto --chunk-bytes 262161 --timeout 120 --output docs/GLM53/GLM53_FLASH_UD_Q3_K_XL_GPU_DISPATCH.json
  ```

  Python **17/17** — scripted subprocess/runner contracts, без GPU; вывод в
  консоли, отдельный log не сохранялся. Helper выполнил чистую пересборку
  `cmake --build build-local/glm5next-transport --config Release --clean-first`
  и `ctest --test-dir build-local/glm5next-transport -C Release -V --no-tests=error`:
  **9/9 CTest**, без skips. [Сохранённый CTest](GLM53_FLASH_REAL_DISPATCH_TESTS.txt),
  полный локальный build log `build-local/glm5next-real-dispatch-build.log`.
  Новые native parser cases и прежние cache/transport/planner, 432 synthetic GPU
  comparisons, reader/DeepSeek frequency compatibility и tokenizer protocol прошли.
  Дополнительно тем же runner и checker, на IQ3_XXS с `--experts 287 --modes mmap native auto`:
  plain `--layers 3 --output build-local/glm5next-dispatch-plain-regression.json`
  (9 сравнений); `--cache-check --layers 45 --output build-local/glm5next-dispatch-cache-regression.json`
  (45 сравнений) — exit 0, оба локальных JSON сохранены.
- **Исправление проверки:** первый real-file прогон вышел с exit 1 на assertion
  reader policy в auto prefill. Проверка ошибочно требовала только native bytes;
  общий pipeline использует mmap для resident pages. Условие исправлено, добавлен
  Python case со смешанными native/mmap bytes; финальные прогоны обоих профилей
  прошли. Явные mmap/native и auto decode по-прежнему проверяются строго.
- **Ограничения:** последовательное сравнение packed bytes, без dequantization,
  численного GLM graph, logits, MTP compute, скорости или overlap timeline.
  Generation keys моделируются внутри checker; реального model reload/remap нет.
  Проверены frequency admission и capacity bypass; frequency rejection на реальных
  матрицах отдельно не проверялся (есть synthetic P3.2b.2). Cancel может добавить
  read-ahead, поэтому полный source/H2D total ограничен диапазоном, а для завершённых
  фаз проверяется точная delta. Ring D2D не включает cache→destination. Native read
  не считается физическим SSD I/O. Полные Qwen/DeepSeek inference, Unsloth build,
  Linux/HIP/multi-GPU и HTTP runtime не запускались.
- **Следующий шаг:** P0.3b → P2.1b.2: получить проверенный архив кандидата,
  собрать oracle/CUDA targets в отдельном build tree, сохранить реальный tokenizer
  parity report. Затем связать dispatch с copy points GLM graph и проверить
  численные outputs/отмену; byte checker этого критерия не заменяет.

### P3.5c-01 — 2026-10-04, Asia/Yekaterinburg — Контроллер бюджета и резерва VRAM

- **Статус:** DONE для компонента P3.5c; P3/P3.5 остаются IN_PROGRESS. Codex,
  цикл 12. Ветка `dev`, база `0ab7dd5601fff5a58a15ef0db0aa82b6d853f637`, исходное
  дерево чистое. Изменения этого цикла не закоммичены.
- **Реализация:** `backends/glm5next/expert_memory.hpp` управляет одним общим
  main/MTP cache на GPU: configured byte cap, reserve и optional total-device
  usage target. Существующая арифметика `StrataVramPolicy` перенесена без изменения
  поведения в `backends/common/vram_policy.hpp`; DeepSeek header стал compatibility
  include. Его исходный `test_vram_policy.cpp` добавлен в GLM CTest. Qwen и
  DeepSeek runtime policies/профили не менялись.
- **Контракт:** caller передаёт probe global free/total для CUDA device cache и
  вызывает refresh между dispatch scopes после размещения fixed weights, state,
  ring/workspace. Режимы 0 (configured bytes) и 2 (device usage) поддержаны;
  matrix-count mode 1 отклоняется для mixed quants. Target ограничен configured
  cap и общей reserve-арифметикой. Pins/leases/pending CUDA events препятствуют
  trim, retired allocations остаются учтены; status сообщает deferred bytes.
  Snapshot отражает последний refresh, не атомарную резервацию VRAM.
- **Недоступная память:** до первого valid sample admission приостановлен.
  False/некорректный/противоречащий resident bytes sample оставляет cached hits,
  но запрещает новые allocations и вытеснения ради misses. Исключение probe
  сохраняет паузу и передаётся вызывающему коду. Dispatch доставляет misses через
  обычный bypass. `paused_bypasses` отдельно считает такие решения. Valid refresh
  возобновляет admission; destructor controller не включает его молча после
  ошибки. Явный возврат к manual mode требует set_budget/set_admission_enabled.
- **Проверки:** рабочая директория `C:\work\git\my-repos\Strata`.
  `cmd /c build-local\check-glm-native-plan.cmd` — exit 0. Скрипт с тем же x64 MSVC
  environment/CUDA PATH, что P3.1b-01, выполнил:

  ```text
  cmake --build build-local/glm5next-transport --config Release --clean-first
  ctest --test-dir build-local/glm5next-transport -C Release -V --no-tests=error
  ```

  Чистая пересборка, **10/10 CTest**, без skips. RTX 5090, CUDA runtime/driver
  **13000/13000**. [CTest log](GLM53_FLASH_MEMORY_TESTS.txt); полный локальный
  log `build-local/glm5next-memory-build.log`. GPU allocations/bytes реальные,
  значения global memory в новых fixtures **подставные**. Проверены:
  - LRU/frequency, общий main/MTP budget, configured cap, pressure trim/recovery,
    invalidation с живыми pins/leases и сохранение учёта retired bytes.
  - Deferred trim pending GPU upload/consumer без ожидания host gate, затем
    освобождение после завершения CUDA events; total-device target mode.
  - False, zero-total, free > total, used < resident и throwing probes;
    сохранение hits, отсутствие новых admissions, pause counters и recovery.
  - Missing probe, invalid reserve/mode/target. Поддержка другого CUDA device
    отклоняется кодом; второй GPU для проверки не использовался.
  - Dispatch при недоступном sample: полный bypass payload с guards, cached hit
    без повторного H2D, режимы mmap/native/auto × LRU/frequency (6 cases).
  Прежние planner/cache/pins/transport, 432 synthetic GPU comparisons, reader,
  DeepSeek frequency и новый VRAM-policy compatibility test также прошли.
  Python, real GGUF dispatch и полный DeepSeek/Qwen inference в этом цикле
  повторно не запускались; их прежние результаты не считаются проверкой controller.
- **Повтор P0.3b:** выполнен read-only запрос, exit 1:

  ```text
  .venv\Scripts\python.exe -c "import urllib.request; r=urllib.request.urlopen('https://codeload.github.com/unslothai/llama.cpp/tar.gz/86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9', timeout=20); print(r.status); print(r.headers.get('Content-Length')); print(len(r.read(64)))"
  ```

  `urllib.error.URLError: <urlopen error [WinError 10013] ...>` при sock.connect;
  архив не получен, сборка кандидата не запускалась. Вывод в консоли, отдельный
  log не сохранялся. Просмотренные build-local/third_party по-прежнему не содержат
  готового архива кандидата; условие разблокировки — доступный проверенный архив.
- **Ограничения:** controller ещё не подключён к GLM graph/loader и реальному
  global-memory probe. Для WDDM требуется PCI-matched NVML, а не CUDA per-process
  fallback. Нет pressure от другого приложения, измерений RAM/VRAM/скорости,
  гарантии от внешней allocation между sample и cudaMalloc или предрезервации
  ещё не созданного workspace. Частота refresh и INFO/settings — работа runtime.
  Linux/HIP/multi-GPU и полная модель не проверялись.
- **Следующий шаг:** P0.3b → P2.1b.2: получить архив, собрать oracle/CUDA и записать
  реальный tokenizer parity report. При GLM runtime-интеграции подключить global
  sampler, один main/MTP cache/controller на GPU и refresh в безопасных границах,
  затем проверить внешнее pressure, численные outputs и фактическую память.

### P3.5d-01 — 2026-10-04, Asia/Yekaterinburg — Live global-memory probe

- **Статус:** DONE для P3.5d; P3/P3.5 остаются IN_PROGRESS. Codex, цикл 13.
  Ветка `dev`, база `f2266a80fedaaefc3254a0599f3f24b5567f0274`, исходное дерево
  чистое; изменения не закоммичены. P0.3b/сеть не перепроверялись.
- **Реализация:** Windows reader из DeepSeek вынесен в
  `backends/common/device_memory.hpp`; прежний `device_memory.inc` включает его.
  DLL загружается только из System32, NVML handle определяется через CUDA PCI
  bus ID, не по совпадению CUDA/NVML ordinal. Самpler сериализует вызовы mutex,
  сохраняет диагностическую стадию/API code, обнуляет outputs при failure,
  сбрасывает handle после read error/invalid values для следующего rebind.
  Partial symbols/init failure не вызывают shutdown чужой NVML initialization.
  Успешный init балансируется shutdown, DLL освобождается при destruction.
  NVML SDK/link dependency и CUDA per-process fallback под Windows не добавлены.
- **GLM:** `global_memory.hpp` создаёт owning closure с shared reader;
  новый трёхаргументный `ExpertMemoryController(cache, cap, policy)` использует
  его по умолчанию. Прежний injected probe constructor сохранён. При отсутствии
  NVML controller сохраняет paused admission из P3.5c. Инициализацию, завершившуюся
  ошибкой, можно повторить пересозданием reader/controller; обычные read errors
  повторяют PCI lookup при следующем sample. Non-Windows factory использует
  cudaMemGetInfo только для уже выбранного device; этот путь не проверялся.
- **Файлы:** общие reader и README; DeepSeek compatibility include; GLM probe,
  controller, CMake, `test_global_memory.cpp`, README, план/статус и
  [CTest log](GLM53_FLASH_GLOBAL_MEMORY_TESTS.txt).
- **Проверки:** рабочая директория `C:\work\git\my-repos\Strata`.
  `cmd /c build-local\check-glm-native-plan.cmd` — exit 0; выполнены:

  ```text
  cmake --build build-local/glm5next-transport --config Release --clean-first
  ctest --test-dir build-local/glm5next-transport -C Release -V --no-tests=error
  ```

  Чистая сборка, **12/12 CTest**, без skips. Стенд RTX 5090, CUDA runtime/driver
  **13000/13000**, прежние x64 MSVC/SDK и CUDA PATH из P3.1b-01. Полный локальный
  build log: `build-local/glm5next-global-memory-build.log`.
  - Два новых targets проверяют прямой include и `deepseek4/device_memory.inc`.
    Fixtures: два разных PCI identity с разными данными, cached handle и rebind,
    missing symbols, ошибки init/PCI/handle/read, null handle, total=0,
    free > total/unsupported sentinel, recovery и balanced shutdown.
    По 400 concurrent samples на target проверяют сериализацию lookup/read.
  - **Реальный NVML был доступен** в обоих targets, `available=1`, stage/code=0.
    В финальном запуске зафиксированы free **32 066 412 544 bytes**, total
    **34 190 917 632 bytes**. Это мгновенный global sample, не память модели.
    DeepSeek wrapper вернул такой же корректный sample.
  - Production factory/default controller получил valid sample, target **1 МиБ**,
    допустил **64 bytes** cache fixture, bytes проверены после CUDA upload;
    `paused_bypasses=0`. Live smoke допускает unavailable результат лишь при
    zero outputs/paused admission, но в этом запуске исполнилась available ветка.
  Прежние cache/controller, dispatch, reader, planner, frequency/VRAM-policy
  compatibility и 432 synthetic GPU byte comparisons прошли. Первичная сборка
  выявила Win32 min/max macros в новом header; добавлен NOMINMAX до Windows headers,
  финальная чистая сборка прошла. `git diff --check` — exit 0.
- **Ограничения:** GLM graph/loader и периодические refresh ещё отсутствуют;
  нет pressure от другого процесса, измерений inference RAM/VRAM/скорости или
  внешнего CUDA timeline. Полный DeepSeek backend с nvcc и Qwen/DeepSeek inference,
  реальные GGUF, Python suites, Linux/HIP и физический multi-GPU не запускались.
  Наличие live NVML сегодня не отменяет необходимость failure handling: предыдущая
  ошибка окружения не воспроизводилась в этом smoke. Reader сообщает snapshot,
  не резервирует VRAM против одновременных allocations других процессов.
- **Следующий шаг:** P0.3b → P2.1b.2: получить проверенный архив, собрать
  candidate/oracle/CUDA и сохранить tokenizer parity. При runtime-интеграции
  вызывать live controller между dispatch scopes после размещения workspace,
  затем проверить внешнее pressure, численные GLM outputs и фактическую память.

### P3.5e-01 — 2026-10-04, Asia/Yekaterinburg — Внешнее потребление VRAM и восстановление

- **Статус:** DONE для P3.5e, весь P3/P3.5 остаётся IN_PROGRESS. Codex, цикл 14.
  Ветка `dev`, база `ab6a0f2e94a35aa0fa6e89c1f89410e5d8c309e9`, исходное дерево
  чистое; изменения не закоммичены. P0.3b/сеть в этом цикле не перепроверялись.
- **Реализация:** новый `backends/glm5next/check_memory_pressure.cpp` и opt-in
  CUDA target `strata-glm5next-memory-pressure-check`. Python runner
  `tools/check_glm5next_memory.py` запускает два разных дочерних процесса:
  controller и holder на одном CUDA PCI device. После инициализации обоих
  contexts controller выбирает target usage = ceil(global used / МиБ) + 128 МиБ,
  mode 2, reserve 128 МиБ, cache cap 64 МиБ. Holder выделяет и заполняет 256 МиБ.
  Это пересечение заданного usage target при наличии свободной физической VRAM,
  не попытка вызвать OOM. Код рабочего cache/controller/NVML reader не менялся.
- **Проверенный сценарий:** warm main/MTP (два synthetic payload по 32 МиБ) →
  main pin → ALLOC отдельного процесса → refresh с target 0, evict idle MTP,
  retain main/deferred 32 МиБ → проверка bytes main hit → release pin/trim до 0 →
  FREE holder → восстановление target 64 МиБ и повторная загрузка обеих матриц.
  **LRU и frequency прошли**, по 5 полных byte comparisons (10 всего),
  counters hits/misses/evictions и размеры cache/deferred совпали со сценарием.
  Реальный NVML, PCI GPU и CUDA allocations, данные матриц синтетические.
- **Артефакт:** [external-memory JSON](GLM53_FLASH_EXTERNAL_MEMORY.json).
  Сохранены время UTC, checker/runner SHA-256, размеры allocations, reserve/mode,
  фактический device target, PCI identity, CUDA versions, отдельные PIDs,
  global free/total и cache counters по всем фазам обеих политик. Это отдельный
  эксперимент с памятью; результатов скорости или полной GLM в JSON нет.
- **Протокол/cleanup:** строгий порядок READY/START/ALLOC/PRESSURE/TRIM/FREE/
  RECOVER/QUIT, сверка GPU процессов и обязательные результаты обеих политик.
  Ожидание pressure/recovery sample ограничено 8 сек, worker responses — timeout
  runner (20 сек). На ошибке закрывается stdin обоих детей, allocation освобождается
  при EOF/exit; только не завершившийся собственный ребёнок принудительно
  завершается спустя 3 сек. Stale success JSON заменяется error report.
  Проверка не добавлена в default CTest из-за отдельной VRAM allocation.
- **Команды/результаты:** cwd `C:\work\git\my-repos\Strata`, все финальные команды
  ниже — exit 0. Для GPU runner PATH дополнен `build-local\cuda-13.0\bin` и
  `build-local\cuda-13.0\bin\x64`.

  ```text
  .venv\Scripts\python.exe -m unittest tools.test_glm5next_memory_check
  cmd /c build-local\check-glm-native-plan.cmd
  .venv\Scripts\python.exe -m tools.check_glm5next_memory --checker build-local/glm5next-transport/strata-glm5next-memory-pressure-check.exe --output docs/GLM53/GLM53_FLASH_EXTERNAL_MEMORY.json
  ```

  **7/7 Python tests**: validation, ошибочные phases/counters, early exit/stderr,
  timeout и завершение собственного ребёнка, лишний output/неуспешный exit,
  обе политики, замена stale report и защита checker от перезаписи. Это scripted
  subprocess tests, не GPU evidence; отдельный log Python не сохранялся.
  Helper сделал чистую сборку `cmake --build build-local/glm5next-transport
  --config Release --clean-first`, затем `ctest --test-dir build-local/glm5next-transport
  -C Release -V --no-tests=error`: **12/12 CTest**, без skips. Все прежние GPU
  transport/cache/controller/global-memory и DeepSeek compatibility tests прошли.
  [CTest log](GLM53_FLASH_PRESSURE_TESTS.txt); полный локальный build log:
  `build-local/glm5next-pressure-build.log`. Windows, RTX 5090, CUDA runtime/driver
  **13000/13000**, прежний MSVC/SDK environment. `git diff --check` — exit 0.
- **Ограничения:** нет GLM graph/реальных expert weights, численных logits, MTP
  compute, pipeline dispatch под внешним pressure, длительной нагрузки, OOM или
  гонки чужой allocation между sample и cudaMalloc. Проверены control boundaries
  отдельного checker; refresh cadence runtime остаётся задачей. Активные сторонние
  приложения могут мешать ожидаемому восстановлению target и привести к failure.
  Linux/HIP/multi-GPU, full DeepSeek/Qwen inference и Unsloth build не запускались.
- **Следующий шаг:** P0.3b → P2.1b.2: получить проверенный архив, собрать
  candidate/oracle/CUDA и записать tokenizer parity; затем подключить controller
  к безопасным границам реального GLM graph и проверить численные outputs.

### P3.6a-01 — 2026-10-04, Asia/Yekaterinburg — Ожидания CPU и staging telemetry

- **Статус:** DONE для P3.6a; общий пункт counters/timeline и P3 остаются
  IN_PROGRESS. Codex, цикл 15. Ветка `dev`, база
  `d83283ea2dc9e52257cfa8a690ab803559101bd7`, исходное дерево чистое; изменения
  не закоммичены. P0.3b/архив/сеть в этом цикле не перепроверялись.
- **Реализация:** `backends/common/expert_pipeline.hpp` разделяет прежний CPU
  `wait_us` на `slot_wait_us` (reader ждёт CUDA event предыдущего consumer слота)
  и `consumer_wait_us` (consumer ждёт публикацию следующей передачи). Старое
  значение сохранено как точная сумма новых полей для совместимости DeepSeek.
  Эти поля не измеряют GPU duration: события остаются без timing, дополнительных
  CUDA synchronizations не добавлено.
- **Память/очередь:** `pinned_bytes` и `device_ring_bytes` показывают fixed capacity
  четырёх выделенных слотов. `reader_owned_bytes`/`reader_owned_peak` — payload
  jobs, назначенных readers, включая ожидание освобождения слота. `queued_bytes`/
  `queued_peak` — опубликованные H2D payloads до consumption, включая ещё
  выполняющиеся copies. `unused_bytes` считает опубликованный отменённый остаток
  вместе с прежним счётчиком unused chunks. Current bytes очищаются при finish,
  peaks/totals сохраняются между планами. Арифметический overflow capacity
  отклоняется до CUDA allocation. `ExpertTransport::counters()` уже возвращает
  расширенный snapshot; schema backend INFO не менялась.
- **Проверки:** cwd `C:\work\git\my-repos\Strata`.

  ```text
  cmd /c build-local\check-glm-native-plan.cmd
  cmd /c build-local\check-deepseek-reader.cmd
  ```

  Обе команды — exit 0. Первая сделала clean rebuild всех targets и
  `ctest --test-dir build-local/glm5next-transport -C Release -V --no-tests=error`:
  **12/12 CTest**, без skips, RTX 5090, CUDA runtime/driver **13000/13000**.
  [CTest log](GLM53_FLASH_TELEMETRY_TESTS.txt); полный локальный build log
  `build-local/glm5next-telemetry-build.log`. Подтверждено:
  - Fixture с chunk 65553 bytes и payload 65570 bytes: fixed pinned/ring по
    **262212 bytes**, reader peak **65553**, queue peak **65570**, отменённый
    остаток **65570 bytes / 2 chunks**. После finish current bytes равны нулю.
  - Restart доставляет исходные bytes, повторно использует pipeline и не сбрасывает
    cumulative peak/unused totals. Empty plan не меняет peak. Переполнение
    staging capacity отклонено. Legacy wait равен сумме новых компонентов.
  - Все 432 прежних GPU matrix comparisons дополнительно проверяют capacity,
    reader/queue peak bounds и нулевые current/unused bytes после полного consumption.
  Вторая команда напрямую пересобрала `backends/deepseek4/test_expert_pipeline.cpp`
  через x64 MSVC (`/std:c++17 /O2 /MD`, cudart.lib, psapi.lib; environment из
  P3.1a-01) и запустила `build-local/glm5next-reader/deepseek-pipeline.exe`.
  Полное кольцо подтверждено как 4×chunk queued bytes; прошли blocked-consumer
  fixture, byte parity, native/auto queue, reuse/cancel/restart и wait-sum checks.
  [DeepSeek log](GLM53_FLASH_TELEMETRY_DEEPSEEK_TESTS.txt), локальный оригинал
  `build-local/glm5next-telemetry-deepseek.log`. `git diff --check` — exit 0.
- **Ограничения:** CPU timers включают call/lock/condition-variable overhead и
  суммируются по потокам, поэтому не равны времени запроса. Occupancy peaks не
  являются дополнительными allocations сверх fixed staging; из fixed capacities
  исключены CUDA allocator/events overhead, cache, destinations и source mappings.
  Новые поля не измеряют GPU H2D/compute duration и не подтверждают overlap
  timeline или скорость GLM. Реальные GGUF/внешнее pressure, Python, полный
  Qwen/DeepSeek inference, GLM graph, Unsloth build, Linux/HIP/multi-GPU в этом
  цикле не запускались. Планирование, reader policies и порядок CUDA work не менялись.
- **Следующий шаг:** P0.3b → P2.1b.2: получить проверенный архив, собрать
  candidate/oracle/CUDA и сохранить tokenizer parity. После интеграции графа
  вывести эти counters в INFO/monitor и измерить H2D/compute по CUDA timeline.

### P3.5f-01 — 2026-10-04, Asia/Yekaterinburg — Cache allocation OOM bypass

- **Статус:** DONE в пределах компонента; общий P3 остаётся IN_PROGRESS.
- **Исполнитель:** Codex, автоматический цикл 16.
- **Пункт плана:** P3.5, работа byte cache при изменении доступной VRAM.
- **Ревизия:** `b540f1ea27deb82d1a98ac61d32afb6bff6f1244`, ветка `dev`;
  до начала дерево чистое. Изменения этого пункта оставлены в рабочем дереве.
- **Изменение:** только `cudaErrorMemoryAllocation` от allocator матрицы кэша
  возвращает empty lease без upload/admission и увеличивает `allocation_bypasses`
  вместе с `bypasses`. Существующий dispatch переносит полную матрицу через bypass
  в заранее выделенный destination. Последующие misses могут снова выделять память.
  Planned hits сохраняют pins; уже удалённые LRU/frequency victims не восстанавливаются.
  Non-OOM allocation, event и upload errors продолжают выбрасываться. CUDA last error
  очищается только если равен `cudaErrorMemoryAllocation`. Retry, изменение бюджета
  или CPU compute fallback не добавлены.
- **Файлы:** `backends/glm5next/expert_cache.hpp`, `test_expert_cache.cpp`,
  `test_expert_transport.cpp`, `README.md`; план, этот статус и
  [CTest log](GLM53_FLASH_ALLOCATION_TESTS.txt).
- **Проверки:** рабочая директория `C:\work\git\my-repos\Strata`.
  `cmd /c build-local\check-glm-native-plan.cmd > build-local/glm5next-allocation-build.log 2>&1`
  — exit 0 после исправления начальной C2440 (неоднозначная перегрузка cudaMalloc
  в default argument заменена lambda). Helper запускает x64 MSVC environment,
  `cmake --build build-local/glm5next-transport --config Release --clean-first`, затем
  `ctest --test-dir build-local/glm5next-transport -C Release -V --no-tests=error`.
  **12/12 CTest, 0 failed**, 3,91 с; RTX 5090, CUDA runtime/driver 13000/13000.
  - Cache LRU/frequency: injected OOM без upload, admission и утечки source owner;
    pins и eviction accounting, hit при отказе allocator, успешное восстановление.
    `cudaErrorInvalidValue` и upload exception с текстом OOM не превращаются в bypass.
  - Dispatch: 12 сочетаний mmap/native/auto × LRU/frequency × prefill/decode.
    После OOM miss идут hit и успешный miss; полные bytes и guard regions совпадают,
    source/H2D totals включают ровно два misses, hit не читает source повторно.
    Проверены all-hit после восстановления, release pins/leases, propagation
    non-OOM error, drain/cancel и успешный повторный запуск того же transport.
  - Прежние CTest, включая NVML/DeepSeek include compatibility и 432 transport
    matrix comparisons, прошли. `git diff --check` — exit 0.
- **Блокер P0.3b перепроверен:** команда
  `.venv\Scripts\python.exe -c "import urllib.request; r=urllib.request.urlopen('https://codeload.github.com/unslothai/llama.cpp/tar.gz/86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9', timeout=20); print(r.status); print(len(r.read(64)))"`
  — exit 1, `URLError` с `PermissionError [WinError 10013]` при `sock.connect`.
  Архив не получен, hash/build/oracle не выполнены.
- **Ограничения:** allocation errors внедрены callback; физическая VRAM не
  исчерпывалась, очистка реального CUDA OOM last-error этим fixture не проверена.
  Buffers/transfers выполнялись на GPU, payloads синтетические. Event failure
  не внедрялся. Полные GLM/Qwen/DeepSeek inference, реальные GGUF, внешнее pressure,
  Linux/HIP/multi-GPU в этом цикле не запускались. Скорость GLM не измерена.
- **Следующий шаг:** P0.3b → P2.1b.2: получить проверенный архив Unsloth,
  собрать candidate/oracle/CUDA и сохранить tokenizer parity.

### P3.7a-01 — 2026-10-04, Asia/Yekaterinburg — Warm transport benchmark

- **Статус:** DONE в пределах warm component benchmark; общий пункт подбора
  reader settings остаётся IN_PROGRESS до GLM graph/compute и cold-I/O проверки.
- **Исполнитель:** Codex, автоматический цикл 17.
- **Ревизия:** `730e2b8c405953a08f88a312bd2c1c6b7f609a9f`, ветка `dev`;
  дерево до начала чистое. Изменения P3.7a оставлены в рабочем дереве.
- **Изменение:** `check_expert_ranges.cpp --benchmark READERS DECODE WARMUPS REPEATS`
  и runner `tools/benchmark_glm5next_transfer.py` сравнивают mmap/native/auto,
  1/2/4 readers и prefill/decode. Все cases запускаются отдельными процессами,
  порядок перемешан с seed 17; 2 warmups + 5 измерений на case. Время включает
  pipeline start, ordered transfers, finish и последний consumer stream sync;
  per-matrix sync отсутствует. Allocation, stdio baseline, полное сравнение mmap
  с baseline, warmups, memset и D2H/byte checks вынесены из таймера. Каждый проход
  проверяет полный payload и guard regions; counters должны совпасть с планом.
  JSON хранит raw samples, median/min/max, CPU wait counters, source/H2D/D2D,
  CUDA/GPU identity, binary hash и размеры/mtime/header hashes моделей.
  Destinations ограничены 512 МиБ; expected/actual host buffers каждый равен
  destination size. При 1 МиБ chunk pinned/ring capacity — по 4 МиБ.
  Decoder mmap/auto использует одного активного reader при любом configured
  count; native decode использует заданное число. Общий pipeline и defaults
  runtime не менялись.
- **Файлы:** `backends/glm5next/check_expert_ranges.cpp`, `README.md`,
  `tools/benchmark_glm5next_transfer.py`, `tools/test_glm5next_transfer_benchmark.py`,
  план, статус, CTest log и два JSON ниже.
- **Проверки:** рабочая директория `C:\work\git\my-repos\Strata`.
  - `cmd /c build-local\check-glm-native-plan.cmd > build-local/glm5next-benchmark-build.log 2>&1`
    — exit 0; clean MSVC/CUDA build, `ctest --test-dir build-local/glm5next-transport -C Release -V --no-tests=error`:
    **12/12 tests**, 3,69 с. Добавлены проверки CLI bounds benchmark;
    прежние cache/transport/NVML/DeepSeek include checks прошли.
    [CTest log](GLM53_FLASH_BENCHMARK_TESTS.txt).
  - `.venv\Scripts\python.exe -m unittest tools.test_glm5next_transfer_benchmark tools.test_glm5next_transfer_check`
    — exit 0, **23 tests**. Проверены protocol/config/sample consistency,
    missing verification, неверные bytes/chunks/reader policy, limits до запуска,
    timeout/process failure, error report и защита output от перезаписи checker.
  - Для следующих двух команд PATH дополнен `build-local/cuda-13.0/bin` и
    `build-local/cuda-13.0/bin/x64`:
    `.venv\Scripts\python.exe -m tools.benchmark_glm5next_transfer --gguf H:\GLM-5.3-Flash-GGUF\GLM-5.3-Flash-Uncensored-IQ3_XXS.gguf --checker build-local\glm5next-transport\strata-glm5next-transfer-check.exe --output docs\GLM53\GLM53_FLASH_IQ3_XXS_TRANSPORT_BENCHMARK.json`
    — exit 0.
    `.venv\Scripts\python.exe -m tools.benchmark_glm5next_transfer --gguf H:\GLM-5.3-Flash-GGUF\UD-Q3_K_XL\GLM-5.3-Flash-UD-Q3_K_XL-00001-of-00004.gguf --checker build-local\glm5next-transport\strata-glm5next-transfer-check.exe --output docs\GLM53\GLM53_FLASH_UD_Q3_K_XL_TRANSPORT_BENCHMARK.json`
    — exit 0.
    Каждый профиль: layers 3/11/45, IDs 0–6/287, полные gate/up/down,
    72 matrices × 18 cases × 7 passes = **9072 byte comparisons**;
    всего **18144**, без ошибок. `git diff --check` — exit 0.
- **Измеренный результат:** Windows, Ryzen 9 9950X, RTX 5090, CUDA runtime/driver
  13000/13000, chunk 1 МиБ, GPU cache отсутствует. Для основного IQ3_XXS выбранный
  payload — 227 МиБ; mmap prefill с 1/2/4 readers: median **18,301 / 17,380 / 17,582 мс**.
  Наименьшая prefill median — 2 readers, samples 17,294–17,589 мс;
  mmap decode с 1 configured/active reader — **18,029 мс** (17,978–18,386).
  Для UD-Q3_K_XL payload — 294,5 МиБ; mmap prefill с 1/2/4 readers:
  **23,230 / 23,011 / 23,159 мс**. Наименьшая decode median — auto с одним
  reader, **23,353 мс**, но sample ranges пересекаются с mmap; строгого преимущества
  здесь не установлено. Native в этих warm cases медленнее, а auto prefill
  читает mmap bytes, сохраняя overhead выбора политики. Причина разницы отдельно
  не профилировалась. Все measurements сохранены:
  [IQ3_XXS](GLM53_FLASH_IQ3_XXS_TRANSPORT_BENCHMARK.json),
  [UD-Q3_K_XL](GLM53_FLASH_UD_Q3_K_XL_TRANSPORT_BENCHMARK.json).
- **Ограничения:** это host wall time доставки через RAM/H2D/D2D, не чистая
  PCIe bandwidth и не токены/с. Прогреты выбранные диапазоны, не вся модель;
  ОС cache не сбрасывался и физический SSD I/O не измерялся. Chunk sweep,
  реальные router distributions, CUDA compute/overlap и inference не проверены.
  Политики runtime по локальному минимуму не изменялись. Linux/HIP/multi-GPU,
  полный Qwen/DeepSeek/GLM и MTP не запускались.
- **Блокер:** P0.3b сохраняется: архив Unsloth недоступен, последняя попытка
  в цикле 16 завершилась `WinError 10013`; в цикле 17 сеть не перепроверялась.
- **Следующий шаг:** P0.3b → P2.1b.2: получить проверенный архив,
  собрать candidate/oracle/CUDA и сохранить tokenizer parity; затем повторить
  выбор reader settings на routed workloads вместе с compute.

### P2.5g-01 — 2026-10-04, Asia/Yekaterinburg — GLM frontend через loopback HTTP

- **Статус:** DONE для HTTP/frontend с mock engine; общий P2 остаётся IN_PROGRESS.
- **Исполнитель:** Codex, автоматический цикл 18.
- **Ревизия:** `05dc32d03d232161987496e0296366ad4c4cc413`, ветка `dev`;
  дерево до начала чистое. P2.5g оставлен в рабочем дереве.
- **Пункт плана:** P2.5, снять ограничение прежних in-memory handler tests,
  проверить реальные JSON/SSE-соединения, disconnect watcher и очередь.
- **Изменение:** добавлен `serve/test_glm5next_http.py`. Production `Server`,
  handlers и `_watch_client` работают на настоящих TCP-сокетах `127.0.0.1:0`
  (порт назначает ОС). Tokenizer побайтовый, engine подставной, template GLM fixture.
  HTTP обработчики, watcher и runtime настройки не изменялись. Test server
  удерживает request threads до завершения и проверяет отсутствие неожиданного
  исключения, утёкшего listener и новых `strata-client-watch` threads.
- **Файлы:** `serve/test_glm5next_http.py`, `backends/glm5next/README.md`,
  план, этот статус и [лог тестов](GLM53_FLASH_HTTP_TESTS.txt).
- **Проверки:** Windows, Python 3.12.7, рабочая директория
  `C:\work\git\my-repos\Strata`.
  Команда:
  `cmd /c ".venv\Scripts\python.exe -m unittest serve.test_glm5next_http serve.test_glm5next_handlers serve.test_glm5next_service serve.test_glm5next_requests serve.test_glm5next_mcp serve.test_server.ClientHangUp -v > build-local\glm5next-http-tests.log 2>&1"`
  — **exit 0, 48 tests, 20,618 с**, без skip/failure.
  Новые 5 test methods выполняют **30 сценариев / 56 HTTP-запросов**:
  - OpenAI/Anthropic × JSON/SSE: Unicode/reasoning, stop/end_turn, HTTP/1.0,
    JSON Content-Length и SSE terminal event ровно один раз.
  - Client tool-call → tool-result → continuation для обоих API в JSON/SSE;
    IDs/arguments и rendered tool_response сохранены. Реальные tools не исполнялись.
  - Неверные effort/enable_thinking/clear_thinking дают HTTP 400 JSON до начала
    stream/generation; следующий валидный запрос проходит.
  - OpenAI/Anthropic × JSON/SSE × prefill/reasoning/partial tool: mock после
    выбранного prefix перестаёт выдавать токены и heartbeat. Открытое тихое
    соединение выдерживает watcher poll; после socket shutdown/close настоящий
    watcher устанавливает cancel, engine заканчивается до тестового timeout.
    Проверены monitor state `disconnected`, engine close при занятом FIFO,
    освобождение busy/queued, один учёт usage/history и успешный следующий запрос.
  - Отключённый клиент, ожидающий занятой FIFO, не начинает engine и не создаёт
    usage/history. Проверены оба API и JSON/SSE; следующий запрос проходит.
  Смежный прогон включает прежние GLM handlers/service/normalizer/MCP/web-history
  проверки и два существующих Qwen-template `ClientHangUp` socket tests.
  `git diff --check` — exit 0.
- **Ограничения:** это HTTP интеграция GLM frontend с mock, не GLM inference.
  GPU cleanup, реальные GLM tokenizer IDs, backend protocol, независимый template
  oracle и модель не проверены. Telemetry sampler/archive scheduler не стартовали;
  listener создавался напрямую из production Server с production handler.
  Linux, proxy/TLS, реальные MCP процессы и GUI в этом пункте не проверялись.
  Изменения движка не потребовались; CTest/CUDA в этом цикле не запускались.
- **Блокер:** P0.3b не снят; последняя загрузка архива Unsloth в цикле 16
  получила `WinError 10013`. В цикле 18 внешний доступ не перепроверялся;
  локальные TCP-соединения разрешены и прошли проверки.
- **Следующий шаг:** P0.3b → P2.1b.2: получить проверенный архив, собрать
  candidate/oracle/CUDA и сохранить tokenizer parity. После появления backend
  повторить HTTP disconnect/FIFO сценарии с реальным GPU cleanup.

### P3.7b-01 — 2026-10-04, Asia/Yekaterinburg — Размеры chunk и staging

- **Статус:** DONE для warm component sweep; общий P3 и окончательный выбор
  параметров для GLM inference остаются IN_PROGRESS.
- **Исполнитель:** Codex, автоматический цикл 19.
- **Ревизия:** `2a7503be11b8311c380fcf127a8d6d3a8e1a8594`, ветка `dev`;
  дерево до начала чистое. P3.7b оставлен в рабочем дереве.
- **Пункт плана:** P3, подбор reader/staging параметров; проверить предложенный
  размер 16 МиБ и меньшие chunks с одинаковыми прочими настройками.
- **Изменение:** runner `tools/benchmark_glm5next_transfer.py` принимает
  `--chunk-sweep` (до 8 разных размеров, не более 16 МиБ), `--modes`, `--readers`
  и `--phases`. CLI запрещает одновременные chunk-sweep/chunk-bytes; selections
  валидируются до model inspection. Старый вызов сохраняет 18 cases при 1 МиБ.
  Schema 2 содержит chunk size каждого run, hash runner и сравнения внутри
  неизменных mode/phase/configured-readers. Baseline — 1 МиБ, если он включён,
  иначе наименьший chunk. Сравниваются wall-time medians и расход pinned/ring;
  single-phase subset также поддержан. Порядок всего набора перемешан с seed.
- **Файлы:** `tools/benchmark_glm5next_transfer.py`,
  `tools/test_glm5next_transfer_benchmark.py`, `backends/glm5next/README.md`,
  план, этот статус, [Python log](GLM53_FLASH_CHUNK_TESTS.txt) и два JSON ниже.
- **Проверки:** cwd `C:\work\git\my-repos\Strata`, Windows, Python 3.12.7.
  `cmd /c ".venv\Scripts\python.exe -m unittest tools.test_glm5next_transfer_benchmark tools.test_glm5next_transfer_check -v > build-local\glm5next-chunk-tests.log 2>&1"`
  — **exit 0, 26 tests, 0,778 с**. Новые проверки охватывают bounds/duplicate
  selections, воспроизводимость seed, разделение сравнений по всем фиксированным
  факторам, fallback baseline, ratios/capacities, single-phase report и прежние
  defaults. Прежние malformed protocol, timeout/error и byte-counter checks прошли.
  GPU checker взят из проверенной сборки P3.7a; C++/CUDA исходники не менялись,
  повторная сборка/CTest не запускались.
- **Реальные замеры:** PATH дополнен `build-local/cuda-13.0/bin` и
  `build-local/cuda-13.0/bin/x64`. Обе команды — exit 0:
  `.venv\Scripts\python.exe -m tools.benchmark_glm5next_transfer --gguf H:\GLM-5.3-Flash-GGUF\GLM-5.3-Flash-Uncensored-IQ3_XXS.gguf --checker build-local\glm5next-transport\strata-glm5next-transfer-check.exe --chunk-sweep 262144 1048576 4194304 16777216 --readers 1 2 --seed 19 --output docs\GLM53\GLM53_FLASH_IQ3_XXS_CHUNK_BENCHMARK.json`
  `.venv\Scripts\python.exe -m tools.benchmark_glm5next_transfer --gguf H:\GLM-5.3-Flash-GGUF\UD-Q3_K_XL\GLM-5.3-Flash-UD-Q3_K_XL-00001-of-00004.gguf --checker build-local\glm5next-transport\strata-glm5next-transfer-check.exe --chunk-sweep 262144 1048576 4194304 16777216 --readers 1 2 --seed 19 --output docs\GLM53\GLM53_FLASH_UD_Q3_K_XL_CHUNK_BENCHMARK.json`
  Стенд: Ryzen 9 9950X, RTX 5090, CUDA runtime/driver 13000/13000.
  На профиль: 72 matrices (layers 3/11/45, IDs 0–6/287) × 48 cases ×
  (2 warmup + 5 samples) = **24192 полных byte comparisons**, всего **48384**.
  Source/H2D/D2D/chunk counters и guards прошли. После прогона проверены
  status/case counts и совпадение hashes runner/checker с текущими файлами.
  `git diff --check` — exit 0.
- **Результат, median мс:** одинаковый mmap mode, prefill с двумя readers,
  decode с одним. Измеряется весь выбранный payload, не один эксперт:

  | Профиль / фаза | 256 КиБ | 1 МиБ | 4 МиБ | 16 МиБ |
  |---|---:|---:|---:|---:|
  | IQ3_XXS, 227 МиБ, prefill | 45,238 | 17,806 | 11,912 | 11,452 |
  | IQ3_XXS, 227 МиБ, decode | 46,325 | 18,273 | 12,061 | 11,596 |
  | UD-Q3_K_XL, 294,5 МиБ, prefill | 58,777 | 23,176 | 16,809 | 14,900 |
  | UD-Q3_K_XL, 294,5 МиБ, decode | 60,383 | 23,222 | 16,362 | 14,938 |

  Pinned staging и device ring по отдельности занимают **1/4/16/64 МиБ** для
  этих четырёх chunk sizes. Основной профиль даёт 936/264/80/72 chunks,
  дополнительный — 1200/344/112/72. На 16 МиБ каждая выбранная матрица помещается
  в один chunk. 4 МиБ — кандидат для будущего сравнения с compute: расход каждого
  staging/ring pool вчетверо меньше, чем на 16 МиБ. Нельзя переносить эти medians
  на скорость генерации. Native с одним reader не показывает такого же эффекта;
  все сочетания и диапазоны samples сохранены в
  [IQ3_XXS JSON](GLM53_FLASH_IQ3_XXS_CHUNK_BENCHMARK.json) и
  [UD-Q3_K_XL JSON](GLM53_FLASH_UD_Q3_K_XL_CHUNK_BENCHMARK.json).
- **Ограничения:** выбранные диапазоны предварительно прогреты; cache отсутствует,
  GPU compute/overlap, реальные routing traces, OS disk I/O и полная модель не
  измерялись. Таймер включает CPU scheduling/read, H2D/D2D и final sync; allocation,
  warmup и D2H/byte validation снаружи. Ratios — описательные, не статистическая
  гарантия. Runtime defaults не менялись. Linux/HIP/multi-GPU и inference
  Qwen/DeepSeek/GLM/MTP в этом цикле не запускались.
- **Блокер:** P0.3b сохраняется, последняя попытка получить архив Unsloth в цикле
  16 завершилась `WinError 10013`; в цикле 19 внешний доступ не перепроверялся.
- **Следующий шаг:** P0.3b → P2.1b.2: получить проверенный архив и собрать
  candidate/oracle/CUDA. После подключения compute сравнить 4/16 МиБ с учётом
  доступной VRAM и фактической скорости генерации.

### P5.2a-01 — 2026-10-04, Asia/Yekaterinburg — Main/MTP cache policy

- **Статус:** DONE для независимого cache-компонента; P5 IN_PROGRESS только
  в части подготовки памяти. Native MTP execution не реализован.
- **Исполнитель:** Codex, автоматический цикл 20.
- **Ревизия:** `ddb3424f3c913f1d93949b70ea495c7cde99eb7e`, ветка `dev`;
  до начала дерево чистое, изменения P5.2a оставлены в рабочем дереве.
- **Изменение:** `ExpertCache::set_branch_budgets(main, mtp)` задаёт отдельные
  byte ceilings под общим global budget. Accounting Entry хранит ветку и
  удерживает её bytes после invalidation, пока остаются leases/pins/CUDA consumers.
  `resident_bytes(Branch)`/`byte_budget(Branch)` показывают состояние по веткам.
  Избыточная ветка освобождает свои idle LRU entries; global pressure может
  освобождать обе ветки. Нулевой ceiling отключает admission этой ветки;
  uncached dispatch переносит матрицы полностью. Protected hits остаются доступны.
  При deferred excess новые misses могут временно bypass в обеих ветках.
  Четвёртое поле `Admission.separate_branches=true` создаёт независимую MTP
  frequency history: отдельные decay clock и max_keys bound. Общее число history
  keys ограничено 2×max_keys, `history_size()` показывает сумму. Cross-branch
  eviction при этом использует LRU, поскольку frequency clocks независимы;
  same-history admission сравнивает scores и планирует victims атомарно.
  Invalidation чистит обе истории. Старые defaults сохраняют shared history
  и отсутствие branch ceilings; DeepSeek/common pipeline исходники не менялись.
- **Файлы:** `backends/glm5next/expert_cache.hpp`, `test_expert_cache.cpp`,
  `test_expert_transport.cpp`, `README.md`; план, статус и
  [CTest log](GLM53_FLASH_BRANCH_CACHE_TESTS.txt).
- **Проверки:** cwd `C:\work\git\my-repos\Strata`.
  `cmd /c build-local\check-glm-native-plan.cmd > build-local/glm5next-branch-cache-build.log 2>&1`
  — **exit 0**, clean MSVC/CUDA build и **12/12 CTest, 3,71 с**, без skip/failure.
  Helper использует `cmake --build build-local/glm5next-transport --config Release --clean-first`
  и `ctest --test-dir build-local/glm5next-transport -C Release -V --no-tests=error`.
  Стенд — Windows / RTX 5090 / CUDA runtime/driver 13000/13000.
  - LRU/frequency: MTP ceiling при свободном global budget не вытесняет main,
    zero/oversized MTP miss не вызывает upload, pinned hit сохраняется,
    trim завершается после release; retired MTP lease блокирует over-budget reload.
    Глобальный cap остаётся главным и может уменьшиться ниже branch ceilings.
  - Independent frequency: 32 MTP обращения не старят main history; проверены
    cross-branch global LRU, очистка обеих histories и отсутствие частичного
    eviction при отказе mixed-size frequency admission.
  - Pending CUDA fixture: branch trim не блокируется и не освобождает allocation
    до завершения upload/consumer events; после завершения bytes освобождены,
    main продолжает работать при нулевом MTP ceiling.
  - 12 dispatch сочетаний mmap/native/auto × LRU/frequency × prefill/decode:
    warm main → MTP bypass/main hit/MTP bypass → включение MTP admission →
    all-hit → global trim. **84 full matrix comparisons** и guards прошли;
    source/H2D bytes включают только misses, MTP quota не теряет весов.
  - Прежние key/budget/frequency/OOM/reload/memory-controller, NVML compatibility,
    byte transport и parser tests прошли. `git diff --check` — exit 0.
- **Блокер P0.3b перепроверен:** Python запрос
  `urllib.request.urlopen('https://codeload.github.com/unslothai/llama.cpp/tar.gz/86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9', timeout=20)`
  снова завершился `URLError` / `PermissionError [WinError 10013]` на sock.connect.
  Архив не получен; exit code Python отдельно не сохранялся, поскольку та же
  shell-команда затем читала исходники. Hash/build/oracle остаются невыполненными.
- **Ограничения:** branch limits — ceilings, не reserved shares; configuration
  менять только между dispatch scopes. Memory controller управляет общим cap;
  callers отдельно читают branch accounting. Реальные GLM/MTP графы, model outputs,
  draft/verify/rollback, sampling и ускорение не проверялись. Payloads синтетические,
  реальные GGUF, внешнее pressure, Linux/HIP/multi-GPU и полный Qwen/DeepSeek
  inference в этом цикле не запускались. Нет основания включать MTP в профиле.
- **Следующий шаг:** P0.3b → P2.1b.2: получить проверенный архив, собрать
  candidate/oracle/CUDA и сохранить tokenizer parity; после graph integration
  подключить branch ceilings к native MTP и измерить фактический cache hit rate.

### P5.2b-01 — 2026-10-04, Asia/Yekaterinburg — Branch pressure в memory controller

- **Статус:** DONE для controller accounting; общий P5 остаётся IN_PROGRESS,
  native MTP execution отсутствует.
- **Исполнитель:** Codex, автоматический цикл 21.
- **Ревизия:** `56d3a88871ec3c2d1a0094c2b1565830ffdfcad1`, ветка `dev`;
  дерево до начала чистое, P5.2b оставлен в рабочем дереве.
- **Пункт плана:** P5.2, согласовать раздельные main/MTP limits с общим VRAM
  controller и подготовить корректную статистику для последующего INFO/monitor.
- **Воспроизведённая ошибка:** закреплённый MTP entry 64 байта при ceiling
  16 байт требует освобождения 48 байт, но при соблюдённом global budget
  контроллер раньше показывал `trim_complete=false`, `deferred=0`.
  Новый regression test до исправления провалил `glm5next_expert_cache`:
  `FAIL: controller omitted deferred MTP bytes when the global budget fits`.
  Команда `cmd /c build-local\check-glm-native-plan.cmd > build-local/glm5next-branch-status-before.log 2>&1`
  вернула shell exit 1; 11/12 CTest прошли, один упал ожидаемо.
- **Изменение:** `Status.main`/`Status.mtp` содержат `limit/resident/deferred`,
  `global_deferred` хранит прежнее превышение resident над global target.
  Итоговый `deferred = max(global_deferred, main.deferred + mtp.deferred)`:
  excess разных веток складывается, но освобождаемые branch bytes повторно
  не прибавляются к global excess. Это нижняя граница необходимых bytes;
  цельная матрица при eviction может освободить больше. При unlimited branch
  ceilings прежнее значение deferred сохраняется. Snapshot обновляется при
  создании, refresh, невалидном/throwing probe и после trim, включая exception
  path. Отмена admissions при probe failure и политика освобождения не менялись.
- **Файлы:** `backends/glm5next/expert_memory.hpp`, `test_expert_cache.cpp`,
  `README.md`; план, этот статус и [CTest log](GLM53_FLASH_BRANCH_STATUS_TESTS.txt).
- **Проверки:** cwd `C:\work\git\my-repos\Strata`, Windows, RTX 5090,
  CUDA runtime/driver 13000/13000. Финальная команда:
  `cmd /c "build-local\check-glm-native-plan.cmd > build-local\glm5next-branch-status-build.log 2>&1"`
  — **exit 0**, clean MSVC/CUDA build и **12/12 CTest, 3,75 с**, без skip/failure.
  Helper выполняет `cmake --build build-local/glm5next-transport --config Release --clean-first`
  и `ctest --test-dir build-local/glm5next-transport -C Release -V --no-tests=error`.
  - LRU/frequency: branch-only excess 48, overlapping global excess 32
    (итог 48, не 80), обе ветки 16+48 (итог 64), global-dominant excess 80.
  - Failed/throwing probe сохраняет корректный snapshot и закрывает admission.
    Invalidation с retained leases, release одной ветки и recovery отражены
    в resident/deferred; snapshot не меняется без refresh. После release MTP
    pressure обнуляется и новая main матрица загружается с правильными bytes.
  - CUDA callback удерживает pending MTP upload/consumer без host lease;
    controller показывает branch excess при свободном global cap, не блокирует
    и не освобождает используемые bytes. После завершения событий trim и
    snapshot очищаются. Main работает при нулевом MTP ceiling.
  - Прежние dispatch (включая 84 branch matrix comparisons), byte transport,
    cache, memory-policy, OOM, NVML/DeepSeek include compatibility tests прошли.
    `git diff --check` — exit 0.
- **Ограничения:** новые pressure snapshots проверены с injected memory samples
  и реальными CUDA allocations/transfers. CUDA failure во время самого trim
  не внедрялся. Live NVML smoke прошёл в общем CTest; внешнее pressure и реальные
  GGUF в этом цикле не запускались. GLM INFO/monitor, полный GPU graph, native
  MTP, скорость генерации, Linux/HIP/multi-GPU и full-model regressions не проверены.
- **Блокер:** P0.3b сохраняется: последняя попытка загрузки архива Unsloth
  в цикле 20 получила `WinError 10013`; в цикле 21 внешний доступ не перепроверялся.
- **Следующий шаг:** P0.3b → P2.1b.2: получить проверенный архив и собрать
  candidate/oracle/CUDA. После интеграции backend передать branch snapshots
  в INFO/monitor и проверить их при настоящих main/MTP переключениях.

### P3.5g-01 — 2026-10-04, Asia/Yekaterinburg — Периодический опрос VRAM

- **Статус:** DONE для API периодического refresh; общий P3 остаётся IN_PROGRESS.
- **Исполнитель:** Codex. База `f23352b8c91c07dba3621a2c64f78a3e1301ee9a`,
  ветка `dev`, дерево до начала чистое; изменения оставлены в рабочем дереве.
- **Пункт плана:** P3.5, ограничить частоту global-memory probe на границах
  dispatch, сохранив общий VRAM reserve и поведение при ошибках.
- **Изменение:** `ExpertMemoryController::refresh_if_due(interval)` использует
  steady clock, опрашивает память при первом вызове и на границе периода.
  `refresh()` всегда выполняет опрос и сдвигает следующий срок; вызывать после
  известных выделений fixed/state/workspace. Интервал выбирает caller, default
  не введён. Фоновых потоков нет. При пропуске меняется только счётчик
  `skipped_refreshes`, snapshot, admission и deferred trim не меняются.
  Failed/throwing probe также сдвигает срок: нет повторов на каждом слое,
  новые allocations остаются запрещены до успешного sample, hits доступны.
  Нулевой интервал опрашивает каждый раз, отрицательный отклоняется без изменений.
  Необязательный timestamp позволяет детерминированные тесты без sleep;
  обратный ход переданного времени вызывает свежий sample.
- **Файлы:** `backends/glm5next/expert_memory.hpp`, `test_expert_cache.cpp`,
  `README.md`, план, статус и [лог сборки/CTest](GLM53_FLASH_REFRESH_TESTS.txt).
- **Проверки:** cwd `C:\work\git\my-repos\Strata`, Windows / RTX 5090,
  CUDA runtime/driver 13000/13000. Команда
  `cmd /c "build-local\check-glm-native-plan.cmd > build-local\glm5next-refresh-build.log 2>&1"`
  — **exit 0**, clean MSVC build, **12/12 CTest, 3,83 с**.
  Helper запускает `cmake --build build-local/glm5next-transport --config Release --clean-first`
  и `ctest --test-dir build-local/glm5next-transport -C Release -V --no-tests=error`.
  - Новые LRU/frequency fixtures: по 10 probe attempts и 203 пропущенных опроса;
    проверены первая выборка, точная граница периода, forced reset, zero/negative
    intervals, rollback времени, failed/throwing probe и восстановление admission.
  - С реальными CUDA allocations проверены bytes main/MTP, защита held lease,
    deferred trim после release и сохранение hits во время паузы admission.
    Время и memory samples в этих fixtures синтетические.
  - Прежние cache/dispatch/transport, OOM, branch snapshots и DeepSeek include
    compatibility tests прошли. Live NVML probe/controller smoke вернул valid=1.
  - `git -c safe.directory=C:/work/git/my-repos/Strata -c core.safecrlf=false diff --check`
    — exit 0.
- **P0.3b перепроверен:** в проверенных build-local каталогах и именах Downloads
  подходящего закреплённого архива не найдено. Команда
  `python -c "import urllib.request; u='https://codeload.github.com/unslothai/llama.cpp/tar.gz/86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9'; r=urllib.request.urlopen(u, timeout=20); print(r.status); r.close()"`
  — exit 1, `URLError` / `PermissionError [WinError 10013]` на socket connect.
  Архив не загружен, hash/candidate build/tokenizer oracle не выполнены.
  Отдельный файл лога сетевой попытки не сохранялся.
- **Ограничения:** периодическая выборка подключена только как API контроллера,
  вызовов из GLM graph ещё нет. Между samples внешнее pressure может измениться;
  остаётся прежний allocation-OOM bypass. Deferred trim выполняется при следующем
  фактическом refresh или принудительно. Интервал требует подбора с настоящим
  inference. Полный GGUF, скорость генерации, внешнее pressure, Linux/HIP и
  full-model регрессии Qwen/DeepSeek в этом цикле не проверялись.
- **Следующий шаг:** P0.3b → P2.1b.2 по `backends/glm5next/README.md`: получить
  закреплённый архив, собрать candidate/oracle/CUDA и сохранить tokenizer parity.
  При интеграции GPU graph использовать periodic refresh между dispatch scopes
  и forced refresh после известных крупных allocations; затем измерить интервал.

### P0.3b.1-01 / P2.1b.2-01 — 2026-10-04 14:03, Asia/Yekaterinburg — Сборка кандидата и tokenizer parity

- **Статус:** DONE для сборки P0.3b.1 и token-ID parity P2.1b.2. Исполнитель:
  Codex; работа завершена. P0.3b/P0/P2 целиком остаются IN_PROGRESS.
- **Ревизия:** `7a721e712340d0d7901226bad56290ffe6494ca0`, ветка `dev`, исходное
  рабочее дерево чистое; изменения ниже проверены до коммита.
- **Пункт плана:** получить закреплённую зависимость, собрать реальные llama/oracle/
  CUDA targets и проверить точные token IDs на двух GGUF.
- **Снятый блокер:** обычный `curl.exe --fail --location --connect-timeout 15
  --max-time 180 --output build-local/llama-glm-86ebfef2.tar.gz
  https://codeload.github.com/unslothai/llama.cpp/tar.gz/86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9`
  получил exit 1 / curl(7), socket connect недоступен. Та же команда с разрешённым
  сетевым доступом вне sandbox завершилась exit 0. Прежний WinError 10013 больше
  не блокирует продолжение: архив сохранён локально. Отдельный лог curl не сохранён.
- **Зависимость:** Unsloth `86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9`, архив
  37 493 950 байт, SHA-256
  `f8e524b635b726bae74fd8f84bb9249e5b09384c63207f6707c5a3f921acad99`.
  Корень архива содержит точный SHA; 3599 обычных файлов после распаковки сравнены
  с архивом SHA-256, пропусков/изменений нет. Это записанный локально hash загрузки
  по точному HTTPS commit URL, не независимо опубликованный checksum. Production
  pin пока не принят, DeepSeek pin не менялся. Патчей зависимости нет.
- **Сборка:** новая директория `build-local/glm5next-candidate-cuda`, MSVC
  19.44.35222.0, Windows SDK 10.0.26100.0, CUDA 13.0.48, RTX 5090. Кандидат меняет
  запрос `120` на фактический `120a`; CMake harness теперь сохраняет оба значения,
  читая actual architecture из `ggml-cuda`. Собраны `llama`, `ggml-cuda` и
  `bin/strata-glm5next-tokenizer.exe`; upstream предупреждения не скрывались.
- **Исправление:** добавлено `glm4` ignore_merges — целый pre-tokenized vocab piece
  принимается перед greedy BPE, включая слова длиннее heap threshold. Если целого
  piece нет, остаётся обычный BPE; поиск подстрок не добавлялся. Экспортируются
  флаг и корректная ссылка на SHA GLM. Qwen/JOYAI сохраняют прежние правила.
- **Файлы:** `backends/glm5next/CMakeLists.txt`, `README.md`,
  `tools/strata_tokenizer.py`, `glm5next_tokenizer_corpus.py`,
  `test_glm5next_tokenizer.py`, `test_glm5next_oracle_check.py`, план/статус;
  [build record](GLM53_FLASH_CANDIDATE_BUILD.json),
  [команды и логи](GLM53_FLASH_CANDIDATE_BUILD_TESTS.txt),
  [до исправления](GLM53_FLASH_TOKENIZER_BEFORE.json),
  [IQ3_XXS parity](GLM53_FLASH_IQ3_XXS_TOKENIZER_PARITY.json),
  [UD-Q3_K_XL parity](GLM53_FLASH_UD_Q3_K_XL_TOKENIZER_PARITY.json).
- **Проверки:** cwd `C:\work\git\my-repos\Strata`.
  - `cmd /c "build-local\build-glm-candidate.cmd > build-local\glm5next-candidate-build.log 2>&1"`
    — exit 0, полная компиляция и link, **6/6 CTest**, 0,16 с для CTest.
    Helper с точными configure/build/ctest командами сохранён в текстовом отчёте.
    Из лога исключён только список включаемых MSVC headers (10 469 строк);
    SHA полного локального лога сохранён. Это CTest protocol/planner/readers,
    а не численные CUDA fixtures.
  - После уточнения architecture record та же helper-команда с перенаправлением
    в `build-local\glm5next-candidate-recheck.log` — exit 0, `ninja: no work to do`,
    **6/6 CTest**, 0,15 с. Build record сообщает requested=120 / actual=120a.
  - `python -m unittest tools.test_glm5next_tokenizer tools.test_glm5next_oracle_check tools.test_glm5next_build -v`
    — exit 0, **21 тест**, 1,084 с. Новый fixture перед исправлением показывал
    недостижимый через merges целый token и отсутствие флага в pack metadata.
  - `python -m unittest tools.test_deepseek4 serve.test_detok serve.test_glm5next -v`
    — exit 0, **39 тестов**, 10,821 с, без skipped; Qwen heap/detokenizer,
    JOYAI boundaries и GLM template/parser. Это не full-model регрессия.
  - `python build-local/check-glm-tokenizer-before.py` с сохранённой исходной
    версией Python tokenizer — exit 1, **8 несовпадений из 80**. Скрипт и hash
    исходного tokenizer сохранены в отчётах. Все восемь — четыре добавленные строки
    × parse_special off/on; прежние 72 проверки пропускали ошибку. Например,
    ` 参考`: старый Python `[26767, 224, 98580]`, llama `[99855]`.
  - Обе точные команды `python -m tools.check_glm5next_tokenizer ...` находятся
    в `tokenizer_commands` build record и используют собранный binary/hash.
    **Exit 0, 80/80** на основном IQ3_XXS и **exit 0, 80/80** на UD-Q3_K_XL.
    В JSON сохранены тексты, оба ID-списка и hashes binary/GGUF header/template.
    Словарь загружался из указанного файла/первого shard, без model forward pass.
  - `git -c safe.directory=C:/work/git/my-repos/Strata -c core.safecrlf=false diff --check`
    — exit 0.
- **Не проверено:** GPU graph trace и MTP-off runtime boundary, numerical
  KDA/DSA/mHC/MoE, state rollback/kpool/sparse attention, полная генерация, память,
  скорость, MTP, Linux/HIP и full-model регрессии. CUDA kernels скомпилированы,
  но oracle не исполняет их. Оба токенизатора получали текст из Strata Jinja;
  независимый template renderer не проверялся. GGUF не менялись.
- **Следующий шаг:** P0.3b.2 — добавить graph/validation target в
  `backends/glm5next/CMakeLists.txt` на имеющемся кандидате, получить trace блоков
  0–44 при MTP off и численные GPU fixtures с TF32 off / flash attention off;
  затем P1 demand-loading и baseline короткой генерации.

### P0.3b.2a-01 / P0.3b.2b-01 — 2026-10-04 14:55, Asia/Yekaterinburg — Графы и численные CUDA fixtures

- **Статус:** DONE только для P0.3b.2a/b. Исполнитель: Codex; работа завершена.
  P0.3b.2/P0 остаются IN_PROGRESS, inference backend ещё отсутствует.
- **Ревизия:** база `7a721e712340d0d7901226bad56290ffe6494ca0`, ветка `dev`;
  изменения P0.3b.1/P2.1b.2 уже были в рабочем дереве и сохранены. Новые изменения
  также проверены без коммита. Зависимость прежняя, патчей нет: повторное сравнение
  всех 3599 обычных файлов с архивом не нашло изменений/пропусков.
- **Код:** `backends/glm5next/check_graph.cpp`, `check_ops.cpp`, CMake и README;
  option `STRATA_GLM_GRAPH_CHECK=ON` собирает два checker и добавляет numerical
  CTest. Зависимость DeepSeek и runtime dispatch не менялись.
- **Graph checker:** metadata-only загрузка настоящих GGUF с `no_alloc=true`,
  `load_mode=NONE`, `load_mtp=false`, проверка нулевых weight buffer bytes/data,
  внутренний `graph_reserve(..., split_only=true)`. Контексты 2048/4096,
  batch/ubatch 256, проверяемые n_tokens/n_outputs 1/4/16/256, seq=1, n_rs_seq=4,
  KV F16, flash attention off, TF32 off. Все tensor weight references и выходы
  слоёв ограничены 0–44; descriptors блока 45 отсутствуют.
- **Результат графов:** 8/8 для каждого GGUF, всего **16/16**. При ctx2048 —
  6797 nodes / 4258 compute на GPU; ctx4096 — 7316 / 4501. В каждом графе
  34 KDA, 126 MUL_MAT_ID и по 90 mHC PRE/COMB/POST; sparse indexer 0/11 узлов.
  Все compute nodes проверены; JSON хранит counts и сокращённый layer trace.
  Предполагается all-GPU weight placement без реальных весов; это не проверка
  вместимости VRAM или scheduler после подключения streaming.
- **Обнаружено и учтено:** первый no_alloc запуск с AUTO/mmap завершился
  `GGML_ASSERT(!ml.no_alloc)`; checker переключён на NONE. All-layer offload
  оставляет embedding на CPU; добавлен override `token_embd.weight` → CUDA0.
  Контрольный `--cpu-embedding` возвращает **exit 1**, во всех восьми графах
  обнаружен единственный CPU compute node `embd / GET_ROWS`. Эти условия нужно
  сохранить/перепроверить при runtime интеграции, upstream source не патчился.
- **Numerical checker:** прямой CUDA backend без fallback scheduler, отдельный
  CPU reference (4 threads), seed `0x53f1a5`. **22/22** сценария:
  7 KDA CPU/CUDA с 64 heads × 128 и tokens 1/4/15/16/17/33/64; 2 сравнения с
  независимой double scalar recurrence; 3 microbatch 1/4/16; 5 откатов snapshots
  с принятием 0–4 из четырёх draft-токенов; 4 mHC PRE/COMB/POST с width4096,
  streams4, Sinkhorn20, tokens1/4/17/256; exact I32 gather с -1, нечётными
  payload values >2^24 и INT32_MAX. Промежуточные mHC и KDA states тоже сравнены.
  Пороги: finite, NMSE≤1e-7 и max_abs≤5e-4 одновременно; измеренный максимум
  max_abs=`1.430511474609375e-6`, NMSE=`5.327565010884206e-15`.
- **Windows CTest:** первоначально численный test зависал при загрузке DLL
  (timeout 120 s), хотя прямой запуск проходил. В окружении одновременно Path
  и PATH; CTest при добавлении TF32 терял CUDA prefix. Добавлено явное
  `ENVIRONMENT_MODIFICATION` с compiler bin и bin/x64; повторная сборка/CTest
  прошла **7/7**, включая 22 numerical cases. Исходные logs ошибок остались
  локально в build-local, сводка и финальный полный CTest сохранены в отчёте.
- **Команды и артефакты:** cwd `C:\work\git\my-repos\Strata`;
  `cmd /c "build-local\build-glm-graph.cmd > build-local\glm5next-graph-ops-verified-build.log 2>&1"`
  — exit 0. Полное содержимое helper, configure/build/CTest и точные три
  graph-команды и ops-команда с exit codes сохранены в
  [текстовом отчёте](GLM53_FLASH_GRAPH_VALIDATION_TESTS.txt).
  [Manifest](GLM53_FLASH_GRAPH_VALIDATION.json) содержит compiler/target, binary,
  source и report hashes; [IQ3_XXS](GLM53_FLASH_IQ3_XXS_GRAPH_CHECK.json),
  [UD-Q3_K_XL](GLM53_FLASH_UD_Q3_K_XL_GRAPH_CHECK.json) — exit 0;
  [CPU negative](GLM53_FLASH_GRAPH_CPU_NEGATIVE.json) — ожидаемый exit 1;
  [numerical ops](GLM53_FLASH_OPS_CHECK.json) — exit 0.
- **Границы:** веса GGUF не загружались/не исполнялись, synthetic fixtures не
  заменяют logits полной модели. KDA rollback не включает conv/KV/kpool и
  model `seq_rm`; native MTP, partial pool, sparse DSA, mixed-quant MoE,
  генерация, скорость/пиковая RAM/VRAM, сессии, Linux/HIP и full-model регрессии
  Qwen/DeepSeek не проверены. Production pin пока не принят.
- **Дальше:** P0.3b.2c — начать с state fixture на существующем кандидате:
  неполный kpool и hybrid seq_rm/rollback против чистого продолжения; сохранить
  JSON результатов. После DSA/MoE numerical — P1 demand-loading и logits.

### P0.3b.2c.1-01 / P0.3b.2c.2-01 — 2026-10-04 15:52, Asia/Yekaterinburg — Hybrid state, матрицы и TF32 patch

- **Статус:** DONE только для synthetic state/matrix подпунктов. Исполнитель:
  Codex; работа завершена. P0/P0.3b.2 остаются IN_PROGRESS; следующий этап P1.1
  нужен для проверки реальных весов и P0.3b.2c.3.
- **Ревизия:** база `7a721e712340d0d7901226bad56290ffe6494ca0`, ветка `dev`.
  Прежние незакоммиченные P0.3b.1/.2a/.2b и P2.1b.2 сохранены. Новые файлы:
  `backends/glm5next/check_state.cpp`, `synthetic_glm.hpp`, `check_quant.cpp`,
  `CandidatePatches.cmake`; изменены CMake/README/план/статус. Коммита нет.
- **Synthetic model:** детерминированный GGUF с одним KDA и одним DSA блоком,
  dense/shared FFN, 4 routed experts (2 selected), mHC=4, width256/vocab64,
  F32 веса, KDA head128 × 2, conv4; indexer32 × 128, kpool4/top-k8.
  Dense reference отличается только top-k512. Seed `0x53c001`, файлы создаются
  checker в build tree; их размеры/hashes сохранены. Это модель для тестирования
  архитектурного кода, она не имеет языкового качества скачанного GLM.
- **State checks:** настоящие llama loader/forward/hybrid memory, ctx512,
  n_batch128/ubatch32 (отдельно16), n_rs_seq8, seq1, KV F32, FA off, TF32 off.
  **98/98:** prefill/tokenwise; ubatch16/32; fused/unfused indexer; GPU/CPU;
  dense/sparse первые11 tokens; scalar softmax pooling по resident positions;
  prefix7/8/9/15/16/17 × accepted0–4 — по три проверки (чистое продолжение,
  pending rollback save/restore, содержимое pooled keys); отказ слишком глубокого
  удаления без изменения serialized state. Аудит исполняемых GPU compute nodes
  не обнаружил CPU fallback; CPU — отдельный reference context.
- **Обнаруженный дефект кандидата:** F32 MMF вызывает явные
  `mma.sync...f32.tf32.tf32.f32`, которые не отключаются настройкой cuBLAS
  `NVIDIA_TF32_OVERRIDE=0`. Без патча corrected checker возвращает **exit 1**,
  **10 failed comparisons**, max abs `0.0013841986656188965`. Отключение fusion
  и принудительный cuBLAS F32 не устранили остальные расхождения; custom MMF
  продолжал выбираться. Исходные диагностические логи сохранены в build-local.
- **Исправление:** `CandidatePatches.cmake` проверяет SHA оригинального `mmf.cu`
  и anchor, создаёт `strata-glm-mmf.cu` в build tree и заменяет только source
  target `ggml-cuda`. На NVIDIA при TF32 override `0` F32 MMF не выбирается.
  Тот же predicate используется CUDA graph sync planning; F32 routed fallback
  может синхронизировать stream. F16/BF16/quantized selection не меняется.
  Патч `cuda-f32-mmf-respect-tf32-override` записан в build provenance с original/
  generated hashes. Повторный audit **3599/3599** файлов архива подтвердил, что
  extracted dependency не изменена; DeepSeek pin и исходники не затронуты.
- **После исправления:** при прежних порогах max_abs≤5e-4 и NMSE≤1e-7 все 98
  сравнений проходят, max abs `2.980232238769531e-7`, max NMSE
  `1.6071131714486772e-13`. Порог не подгонялся под исходные ошибки.
- **Matrix checks:** **108/108**: F32/F16/BF16/Q8_0/Q6_K/IQ2_S/IQ3_S/IQ3_XXS/
  IQ4_XS/Q2_K/Q3_K/Q4_K × tokens1/4/17 × MUL_MAT/routed broadcast/routed per-route.
  Матрицы 512→64, 16 экспертов, 8 selected; synthetic weights packed через ggml,
  seeded inputs, прямое CUDA execution без scheduler fallback. Эталон — CPU
  backend и double scalar dot по CPU-dequantized packed weights. NMSE≤1e-10 для
  F32 (регрессия TF32), ≤1e-4 для остальных (допускают внутреннее квантование
  активаций). Worst CUDA/CPU NMSE `6.586060349843735e-5`, CUDA/scalar
  `5.436200211477851e-5`, F32/scalar `1.6865453360159302e-14`.
- **Исправления checker/build:** первоначальный общий status был неверным из-за
  ссылки на results, инвалидированной добавлением ключей в ordered_json. Расчёт
  перенесён до вставок; финальные JSON независимо проверены по всем строкам.
  Начальный 2-head indexer fixture отклонялся GPU-аудитом — сохранена настоящая
  поддерживаемая геометрия32 × 128. Локализованный MSVC `/showIncludes` не сообщил
  Ninja о смене header; добавлен explicit OBJECT_DEPENDS на synthetic_glm.hpp.
  Первоначальные aggregate successes не используются как результат проверки.
- **Проверка:** cwd `C:\work\git\my-repos\Strata`.
  `cmd /c "build-local\build-glm-graph.cmd > build-local\glm5next-state-quant-verified-build.log 2>&1"`
  — **exit 0**, **9/9 CTest**, 3,61 с CTest. Это также повторяет прежние22 ops
  fixtures и protocol/planner/reader checks. Отдельные state/quant запуски —
  **exit 0** каждый; before-patch state — **exit 1**. Полные команды/helper,
  compiler/CUDA/120a, binary/source/report hashes:
  [manifest](GLM53_FLASH_STATE_VALIDATION.json), [лог](GLM53_FLASH_STATE_VALIDATION_TESTS.txt),
  [до патча](GLM53_FLASH_STATE_BEFORE_TF32_FIX.json),
  [state](GLM53_FLASH_STATE_CHECK.json), [матрицы](GLM53_FLASH_QUANT_CHECK.json).
  Из build log исключены только строки списка MSVC includes; hash исходного
  полного лога сохранён. `git diff --check` прошёл.
- **Ограничения:** нет real-GGUF forward, 45-layer logits, F16 KV baseline,
  численного streamed dispatch, длинного контекста, нескольких одновременных
  сессий, native MTP, Linux/HIP, full-model Qwen/DeepSeek регрессий. Скорость и
  пиковая память не измерялись. Время checker не является tokens/s модели.
- **Дальше:** P1.1 — отдельный GLM executable с синхронным demand-loading и
  GPU-only аудитом; начать с адаптации проверенных DeepSeek loader/scheduler hooks
  к GLM SHA и shared transport. Реальные generation/logits/memory reports —
  следующий проверяемый результат.

### P1.1a-01 — 2026-10-04 — Реальная синхронная генерация GLM IQ3_XXS

- **Статус / исполнитель:** DONE в scope P1.1a; Codex, работа завершена.
  P1 в целом IN_PROGRESS: production protocol/profile ещё не реализованы.
- **Изменение:** добавлен `strata-glm5next-smoke` с параметрами из GGUF,
  UTF-8 rendered prompt/raw IDs, greedy decode, сохранением F32 logits и JSON
  метрик. CUDA размещает все вычисления модели, включая embedding и экспертов.
  Routed weights остаются в mmap, eager prefetch отключён. Actual router IDs
  определяют диапазоны gate/up/down; pinned staging ограничен 16 МиБ.
  Копирование последовательное, без expert cache и MTP. Синхронизация защищает
  переиспользование GPU scratch и pinned staging.
- **Guardrails:** до исполнения проверяется весь scheduled graph; CPU compute
  отклоняется с именем узла/op/backend. Full-expert-copy fallback запрещён.
  Pageable fallback при выделении CUDA host staging отклоняется. Hook включён
  только диагностическим runner; CPU-reference fixtures продолжают работать.
- **Файлы:** `backends/glm5next/{smoke.cpp,sync_runtime.h,sync_runtime.inc,
  gpu_only_audit.inc,RuntimePatches.cmake,CMakeLists.txt,README.md}`; plan/status
  и `GLM53_FLASH_SYNC_*` reports. Архивные исходники не правились: повторное
  сравнение **3599/3599** файлов с tar.gz не выявило изменений.
- **Ревизия / сборка:** начальная база `7a721e712340d0d7901226bad56290ffe6494ca0`,
  `dev`; при завершении HEAD `3f7289dbb2c09de7a53386db5595396226fcbe5a`
  содержит прежние graph/state/quant/tokenizer изменения. Агент index/commits
  не менял; новые изменения P1.1a без commit. Unsloth SHA прежний,
  MSVC 19.44.35222.0 / CUDA 13.0.48 / target 120a, driver 581.80. Добавлена
  `STRATA_GLM_SYNC_BASELINE=ON`; TF32/FA/MTP off, KV F16, ctx2048, batch/ubatch16,
  `GGML_OP_OFFLOAD_MIN_BATCH=1`, один owner thread/sequence.
- **Проверка сборки:** cwd `C:\work\git\my-repos\Strata`.
  `cmd /c "build-local\build-glm-graph.cmd > build-local\glm-sync-build-final.log 2>&1"`
  — **exit 0**, **10/10 CTest**, 3,84 с. Новые 5 fixture checks: два
  streamed/resident F16-KV сравнения (dense/sparse), два candidate-copy сравнения,
  negative CPU embedding (отказ до compute/transfer). Прежние 22 ops/98 state/
  108 quant cases также проходят. [Полные команды и лог](GLM53_FLASH_SYNC_VALIDATION_TESTS.txt).
- **Реальная модель:** `H:\GLM-5.3-Flash-GGUF\GLM-5.3-Flash-Uncensored-IQ3_XXS.gguf`.
  Prompt из embedded template: 39 токенов, reasoning_effort low; 64 output tokens.
  Три команды `strata-glm5next-smoke --model ... --prompt-file ... --n-predict 64
  --ctx 2048 --batch 16 --report ... --logits ...`, у третьей `--candidate-copy`;
  **exit 0** все три. Точные аргументы, PATH, build helper и SHA в manifest/логе.
  Сохранены token IDs, полный sampled vocabulary для 64 позиций и текст ответа.
- **Численный результат:** все 9 912 320 logits конечны; IDs и F32 bytes
  совпадают бит-в-бит между первым/повторным synchronous и исходным candidate
  selected-copy reference. Общий raw-logit SHA-256
  `1bb10028852e351734b952e99cb94e6d56295142cbba01754899b8264fd0a550`.
  Это проверяет доставку весов при одинаковом graph/kernel, не независимую GLM.
  GPU-аудит: 378 routed matmuls prefill, 7938 decode; CPU compute 0.
- **Скорость:** первый замер prefill 0,642 / decode **1,288 ток/с**, повторный
  prefill 3,046 / decode **3,333 ток/с**; candidate-copy decode 3,274 ток/с.
  OS cache не очищался, первый замер не называется контролируемым cold-cache.
  Decode считает 63 single-token forward + synchronize (первый output из prefill),
  без sampling/output I/O/memory-query. Повторный decode 18,904 с, из них
  source memcpy 12,289 с, H2D 4,943 с; передано 192 299 904 000 байт с padding.
- **Память:** загрузка первого прогона 5,531 с; resident weights 7541,55 МиБ,
  CUDA scratch 1228,88 МиБ. Peak process working set 70,437 ГиБ включает mmap
  pages; max sampled CUDA device usage 10,588 ГиБ включает видимую CUDA чужую
  нагрузку, не является NVML peak. Наблюдения `nvidia-smi` во время первого
  запуска 13587/13607 МиБ; непрерывный мониторинг физического пика не выполнялся.
  Память после load/context/prefill и каждого decode сохранена в JSON.
- **Исправление harness:** первая synthetic попытка обнаружила инвалидированную
  ссылку `ordered_json` на memory_samples; runner переведён на map-backed json
  до запуска реального GGUF. После первого real run добавлены проверка pinned
  fallback и reference-copy selector; final build и оба повторных run прошли.
  Бинарник первого прогона имеет отдельный SHA в отчёте.
- **Артефакты:** [manifest](GLM53_FLASH_SYNC_VALIDATION.json),
  [fixtures](GLM53_FLASH_SYNC_FIXTURE_CHECK.json), [первый](GLM53_FLASH_SYNC_FIRST_RUN.json),
  [повторный](GLM53_FLASH_SYNC_REPEAT.json), [reference](GLM53_FLASH_SYNC_CANDIDATE_REFERENCE.json).
  Raw F32 остаются в `build-local/glm-sync-iq3-{cold,repeat,reference}.f32`.
- **Не проверено:** production protocol/API/cancellation, async/cache integration,
  MTP, несколько сессий, real-state rollback, длинный sparse-контекст, другие
  prompts, генерация UD-Q3_K_XL, Linux/HIP и full-model Qwen/DeepSeek регрессии.
- **Следующий шаг:** P1.1b — выделить runtime из smoke runner и подключить
  существующий протокол Strata/profile, сверяя результат с сохранённым baseline.

### P1.1b / P2.5h — 2026-10-04 — Pipe engine, профиль и реальный HTTP API

- **Статус:** DONE, только эти подпункты; исполнитель Codex.
- **Ревизия:** `f76442cfb20045486a6620f57ebc577a3873c6be` + изменения рабочего дерева.
- **Изменение:** общий `runtime.hpp`, native `main.cpp` и `protocol.hpp`;
  GEN/PP/T/DONE/ERR, STOP/QUIT, ограниченная очередь, sampling с seed,
  очистка полного hybrid state на каждом запросе и после отмены. INFO не
  объявляет MTP, session reuse, cache-admin или vram-control.
  `serve/server.py` выбирает GLM template до запуска engine. Setup экспортирует
  tokenizer и отдельный localhost profile, проверяет backend identity и
  отказывается перезаписывать существующие profile/pack.
- **Проверки:** 12/12 CTest (4,26 с), 54 существующих Python checks + 3 новых
  profile checks. Реальный native pipe: 9 сценариев, повторный greedy и seeded
  sampling, invalid token, STOP в prefill/decode и последующие чистые запросы;
  process exit 0. Реальный HTTP: OpenAI/Anthropic JSON/SSE по 64 токена,
  disconnect/recovery в prefill/decode; unload, server exit 0. Ответы совпали
  с сохранённым P1.1a baseline. GPU-only, F16 KV, ctx2048/batch16,
  TF32/FA/MTP off; на этом этапе экспертный кэш выключен.
- **Артефакты / команды / точные конфигурации:**
  [manifest](GLM53_FLASH_PROTOCOL_VALIDATION.json),
  [pipe](GLM53_FLASH_PIPE_CHECK.json), [API](GLM53_FLASH_API_CHECK.json),
  [fixture](GLM53_FLASH_PIPE_FIXTURE_CHECK.json),
  [команды и CTest](GLM53_FLASH_PROTOCOL_TESTS.txt).
  Эти отчёты сохраняют профиль до последующего включения memory targets.
- **Исправления harness:** передавать CUDA DLL directories в `child_env`;
  сохранить process handle до `engine.close()`; HTTP monitor использует
  `live.state`, а не `live.phase`. Ошибки harness не выдаются за успешные проверки.
- **Осталось:** shared async overlap, MTP, snapshots/session restore,
  real-state rollback, длинный sparse-контекст и full-model Qwen/DeepSeek regressions.
- **Следующий шаг:** P3.5h — подключить cache/controller к реальному графу,
  настроить запрошенные пользователем бюджеты памяти и проверить все logits.

### P3.5h — 2026-10-04 — Runtime cache и глобальные бюджеты RAM/VRAM

- **Статус:** DONE, только подпункт P3.5h; исполнитель Codex, работа завершена.
- **Запрос:** сначала 90%, затем пользователь разрешил 95% RAM и VRAM.
  Локальный профиль `strata-glm5next.json` сохранён с обоими targets 95.
- **Ревизия:** работа начата от `f76442c`; код уже в `ebd0ca1`. Финальные
  source/binary hashes и отдельные benchmark binary hashes записаны в manifest.
- **Изменение:** `runtime_memory.hpp/.cpp`, copy hook `sync_runtime.h/.inc`,
  `main.cpp`, `smoke.cpp`, `runtime.hpp`, `expert_cache.hpp`, setup и CMake.
  Контроллер использует global PCI-matched NVML, включает чужую нагрузку,
  fixed/state/scratch/allocator overhead; refresh 500 мс и перед запросом.
  RAM warmup читает mmap pages до target или конца всех expert tensors.
  Windows working set maximum учитывает другую занятую RAM; исходный limit
  восстанавливается при выгрузке. Изменение minimum требовало privilege 1314;
  исправлено сохранением прежнего minimum, дополнительная привилегия не нужна.
- **Cache contract:** полные model/generation/layer/expert/projection/quant/shape
  keys, `runtime-tensor:` namespace с tensor-relative offsets и точным MMQ
  padding. Владелец model mapping удерживается shared ownership. Cache hits
  копируются D2D; misses идут через один pinned buffer 16 МиБ; GPU audit сохранён.
  Copy/compute overlap и MTP остаются выключенными.
- **Измерение / исправление:** первая интеграция на 90% дала 1,732 ток/с,
  хотя H2D уменьшился вдвое. Устранены полный поиск/опрос всех eviction events
  на каждом miss и частые cudaMalloc/free: явный LRU и private CUDA memory pool
  при наличии поддержки. Следующий 90% run — 4,313 ток/с. Это один sample.
- **95%, новые процессы:** 3,781 и 4,012 ток/с; медиана **3,897** по 63 timed
  decode steps для 64 output tokens. Warmup отдельно: 46,67 и 64,40 с.
  39 prompt, ctx2048/batch16, greedy, F16 KV, TF32/FA/MTP off; OS cache не очищался.
  Все 9 912 320 logits каждого capture совпали с P1.1a бит-в-бит, SHA-256
  `1bb10028852e351734b952e99cb94e6d56295142cbba01754899b8264fd0a550`.
- **Память:** во время этих двух decode runs NVML **30,12–30,31 ГиБ**
  (94,59–95,20% от видимых 31,84 ГиБ); global RAM **115,47–119,58 ГиБ**
  (91,97–95,24% от доступных ОС 125,56 ГиБ). Во втором run прогреты все
  **102,322 ГиБ** routed experts: дальнейших expert pages для заполнения RAM нет.
  Target включает систему; это не резервирование и не точный постоянный процент.
- **Повтор в одном pipe engine:** 4,641, затем **6,306 ток/с**, те же 64 IDs.
  Вычислено как 63 / DONE.decode_seconds; включает sampling/pipe output,
  в отличие от smoke timer вокруг llama_decode. Expert cache сохраняется между
  запросами, hybrid sequence state полностью очищается. Один такой повтор.
- **Проверки:** 12/12 candidate CTest, 13/13 cache/transport CTest, 54 existing
  Python + 4 final profile tests. Native pipe 9 сценариев и реальные OpenAI/
  Anthropic JSON/SSE, disconnect/recovery, unload/server exit 0 прошли с 95%.
  После тестов server/engine остановлены, GPU memory освобождена.
- **Артефакты / точные команды:** [manifest](GLM53_FLASH_MEMORY_VALIDATION.json),
  [команды](GLM53_FLASH_MEMORY_TARGETS_TESTS.txt), [95 first](GLM53_FLASH_MEMORY_95_FIRST.json),
  [95 repeat](GLM53_FLASH_MEMORY_95_REPEAT.json), [90 pool](GLM53_FLASH_MEMORY_90_POOL.json),
  [pipe](GLM53_FLASH_MEMORY95_PIPE.json), [API](GLM53_FLASH_MEMORY95_API.json).
  Raw F32 остались в `build-local/glm-memory{90-pool,95-first,95-repeat}.f32`.
- **Не проверено:** async overlap, настройка периода controller, session restore,
  native MTP, real-state rollback, длинный sparse-контекст, Linux/HIP runtime,
  другие prompts, UD-Q3_K_XL generation и full-model Qwen/DeepSeek regression.
- **Дальше:** P3.3b — shared async pipeline для misses, exact logits и
  cancel/recovery на реальном GPU runtime; затем отдельный замер перекрытия compute.

### P3.3b / P5.1a,2c,3a,5a / P6.1a — 2026-10-04 — Pipeline, MTP и подбор профиля

- **Статус:** DONE для перечисленных подпунктов; исполнитель Codex. Этапы P3/P5/P6
  целиком не закрыты, P4 session reuse ещё не реализован.
- **Ревизия:** `3b0e0f75a50d7f284e479443a4d859b691565a16` + рабочее дерево.
  Точные source/binary hashes и связь разных сборок с отчётами — в manifest.
- **Конвейер:** runtime использует общий StrataExpertPipeline через transport/
  dispatch. Scheduler после router IDs планирует gate/up/down с теми же IDs.
  Cache leases удерживаются, misses читаются вперёд в четыре bounded slots,
  H2D идёт отдельным stream; CUDA events защищают reuse и завершение запросов.
  Main/MTP имеют отдельные ключи/частоты и ceiling под общей VRAM целью.
- **MTP:** native NextN context разделяет model weights с main, использует
  target feature rows, draft/verify и точный hybrid rollback. Поддержаны глубины
  1/2/3, target sample-and-match, seeded sampling, bonus/length/stop/cancel,
  INFO/monitor и proposed/accepted/time telemetry. Запросы начинают с чистого state.
- **Ошибка и исправление:** исходный batched CUDA path менял logits (max abs
  1,8882 на 16 токенах) и затем IDs. Короткие MUL_MAT и quantized MUL_MAT_ID
  пакеты теперь используют одиночную арифметику. Один повтор synthetic BF16
  routed теста выявил нестабильность нового dispatch для неквантованных экспертов;
  для них восстановлен исходный путь. Реальные IQ3_XXS routed weights квантованы.
  Финальные 108 сравнений — 90 exact и 18 original routed numerical — прошли,
  включая восемь повторов отдельного теста после исправления.
- **Точность на полной модели:** 39 prompt + 64 output; native MTP 1/2/3, forced
  first/middle/all, 32-token seeded sampling, draft cancellation/clean next прошли.
  Все сравниваемые logits бит-в-бит. Baseline с MTP weights/n_rs_seq=3 также
  совпал с сохранёнными P1 **9 912 320 F32 logits** бит-в-бит.
- **Подбор скорости:** отдельные процессы, один warmup + три повтора на глубину,
  greedy, F16 KV, ctx2048/batch16, TF32/FA off, budgets 95%, chunks 4 МиБ,
  MTP cache ceiling 512 МиБ. Медианы: off **8,584**, 1 **8,730**, 2 **7,431**,
  3 **6,555 ток/с**. Формула 63 / DONE.decode_seconds включает sampling/draft/
  verify/repair/output; исключает load, warmup и prefill. OS cache не очищался.
  Cache 256 МиБ дал 7,287 ток/с, chunk 8 МиБ — 8,457 (по пять повторов).
  По одному prompt нельзя подтвердить универсальную оптимальность; выигрыш
  MTP 1 над off около 1,7% и сопоставим с разбросом.
- **Рабочий профиль:** `strata-glm5next.json` — pipeline 1, MTP 1, ceiling 512 МиБ,
  chunk 4 МиБ, RAM/VRAM 95%, threads4. CLI/setup без локального профиля сохраняют
  opt-in. Ни один другой модельный профиль не менялся.
- **CUDA trace:** шесть target graphs, объединённые H2D spans 1853,812 мс,
  compute spans 775,289 мс, пересечение **2,166 мс**. Это небольшое пересечение
  интервалов streams, не оценка одновременной занятости SM и не throughput benchmark.
- **Проверки:** финальная сборка и 14/14 candidate CTest (10,75 с), 13/13
  shared transport/cache CTest (4,65 с), 85 Python tests (5,833 с); реальный
  pipe 9 сценариев и HTTP 8 сценариев на профиле MTP 1, exit 0 и unload.
  Synthetic MTP fixture отдельно проверяет stop на первом/принятом/отклонённом
  токене. Pipe/API и real trace предшествуют ограничению BF16 dispatch;
  manifest сохраняет их собственные binary hashes, не приписывает им новую сборку.
- **Артефакты:** [описание и таблицы](GLM53_FLASH_PIPELINE_MTP.md),
  [manifest](GLM53_FLASH_PIPELINE_MTP_VALIDATION.json),
  [команды/логи](GLM53_FLASH_PIPELINE_MTP_TESTS.txt),
  [MTP](GLM53_FLASH_MTP_CHECK.json), [CUDA trace](GLM53_FLASH_PIPELINE_GPU_TRACE.json),
  [benchmark](GLM53_FLASH_MTP_BENCHMARK.json),
  [pipe](GLM53_FLASH_PIPELINE_MTP_PIPE.json), [API](GLM53_FLASH_PIPELINE_MTP_API.json).
- **Осталось:** повышение степени overlap, session snapshots/restore,
  index sharing между draft iterations, длинный sparse-контекст, другие prompts/
  кванты, реальные tools/EOS, Linux/HIP и full-model Qwen/DeepSeek regression.

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
