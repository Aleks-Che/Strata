# План внедрения Hy3

Дата исследования: **2026-10-06**, `Asia/Yekaterinburg`.
Репозиторий: `e26522b4dd5cd5f40034accb6ec3c2b7b6e3f00f` с существующими
незакоммиченными изменениями. Ход выполнения:
[HY3_IMPLEMENTATION_STATUS.md](HY3_IMPLEMENTATION_STATUS.md).

Цель — запустить `H:\models\hy3\Hy3-Q3_K_M-mtp.gguf` в Strata на RTX 5090
и 128 ГиБ RAM: вычисления модели на GPU, ограниченный кэш экспертов,
асинхронная доставка SSD/RAM → VRAM, диалоги, tools, сессии и native MTP.
Использовать до 95% доступной системе RAM/VRAM, оставляя место для ОС,
состояния модели и временных буферов. Профили Qwen, DeepSeek, GLM и Step
должны сохранить своё поведение.

Обновление HY3-01, 2026-10-06: выполнены P0.1–P0.3 и проверки tokenizer/template;
скомпилированный no-allocation loader проверен с MTP off/on. Исправлена регистрация
лишних MTP-весов при off отдельным generated patch. CUDA/граф остаются следующими.
Отчёты и точные границы проверки — в [статусе](HY3_IMPLEMENTATION_STATUS.md).

Обновление HY3-03, 2026-10-06: CUDA admission P0 завершён; native pipe engine
генерирует на полном GGUF. P1.1–P1.3 выполнены: синхронный перенос выбранных
экспертов даёт побитово одинаковые logits и greedy IDs с native selected-copy.
Измерены TTFT, prefill/decode, H2D и память; cancellation/unload/reload проверены.
HY3-04 добавил отказ pinned allocation, реальную ошибку ReadFile после частичного
переноса, simulated RAM/VRAM pressure и восстановление; всего32 runtime checks.
Исправлена граница исключений C++ при отказе memory admission на MSVC.
P1.4 остаётся частично открытым для настоящего driver OOM и внешнего pressure.
Базовый pinned-режим дал 1,219 токена/с на 37 decode-шагов английского и кода;
это последовательный correctness-прогон, не оптимизированный профиль или A/B.
Ниже размеры весов рассчитаны по tensor directory; бюджеты и настройки —
исходные варианты экспериментов, а не готовый быстрый профиль.

Обновление HY3-05, 2026-10-06: P2.4/P2.5 выполнены в проверенном объёме.
Отдельный профиль подключает Hy3 к Service, OpenAI/Anthropic JSON/SSE и веб-чату.
Полный GGUF прошёл HTTP tool/result continuation с точными native IDs,
отмену по disconnect и следующий запрос. 276 CPU tests и144 native template
comparisons прошли. Кэш и асинхронный конвейер P3 остаются следующим этапом;
скоростные defaults не выбраны.

Обновление HY3-06, 2026-10-06: реализован opt-in GPU matrix cache с отдельной
Hy3 identity/registry и переиспользованием проверенного Step controller.
100 runtime и25 allocation/policy checks прошли. Full-model logits/IDs,
пересылки, память и варианты cap0/8/12/16 ГиБ отражены в статусе.
Cache8 прошёл HTTP tools/disconnect/recovery и22 synthetic pipe checks.
Копирование остаётся синхронным; следующий этап — P3.2 async pipeline.

Обновление HY3-07, 2026-10-06: bounded pipeline по реальным router IDs прошёл
246 runtime checks и полный GGUF. При cache8/readers2/chunk4 прогретая пара
English/code дала3,100 ток/с против1,869/1,842 synchronous, exact logits/IDs.
CUDA event trace подтвердил небольшое H2D/compute перекрытие; весь выигрыш
ему не приписывается. HTTP tools, disconnect/recovery и22 pipe checks прошли.
Следующий этап — P3.3 tensor batching и prefill policy.

Обновление HY3-08, 2026-10-06: реализован opt-in tensor batching с удержанием
hits/fills до единственного delivery fence. Итоговые472 runtime checks и
полный English/code ABBA прошли; среднее decode2,804→2,930 ток/с, разброс
между процессами большой, default batch0 сохранён. Первый runtime run имел
один intermittent resident/native mismatch при pipeline off; повторы прошли,
причина не установлена. Отчёт FAIL и точные границы проверки сохранены в статусе.
Длинный prompt362 прошёл exact parity, STOP/recovery в обоих режимах;
первый prefill121,454→105,085 с — одно последовательное наблюдение на вариант.
HTTP JSON/SSE, tools и disconnect/recovery с batching также прошли.

## 1. Проверенные исходные данные

### 1.1. Файл, основная модель и MTP

В каталоге модели найден один файл, split metadata отсутствует.

| Параметр | Значение |
|---|---|
| Размер GGUF | 136 454 632 928 байт = **127,083 ГиБ** |
| Версия / alignment | GGUF v3 / 32 байта |
| Metadata / tensors | 43 записи / 1298 тензоров |
| `header_end` / `data_start` | 5 161 421 / 5 161 440 |
| Сумма tensor payloads | 136 449 471 488 байт |
| Основная модель | `blk.0..79` и общие embedding/output, 1278 тензоров |
| MTP | `blk.80`, 20 тензоров, включая четыре `nextn.*` |

Проверены уникальность имён, известные размеры квантов, alignment,
отсутствие пересечений диапазонов и выходов за конец файла. Последний payload
заканчивается точно на EOF. Это проверка структуры, не содержимого весов.

| Группа | Байты | ГиБ |
|---|---:|---:|
| Основная модель, все веса | 134 592 752 896 | 125,349 |
| В том числе routed experts | 125 354 115 072 | 116,745 |
| Остальные основные веса | 9 238 637 824 | 8,604 |
| MTP, все дополнительные веса | 1 856 718 592 | 1,729 |
| В том числе routed experts MTP | 1 717 567 488 | 1,600 |
| Остальные веса MTP | 139 151 104 | 0,130 |

SHA-256 **только первых `header_end` байт**, без alignment padding и payloads:
`f3307357f0b6ab163f188f7d19ba57d2960ca70a7491b511b23b2745a9500ca9`.
Полный SHA-256 локальных весов не рассчитан. На
[странице одноимённого файла AngelSlim](https://huggingface.co/AngelSlim/Hy3-GGUF/blob/main/Hy3-Q3_K_M-mtp.gguf)
опубликован полный SHA-256
`275a60f1269c8f120fe61c96c2307f30df740db691e8f481513bb5c92417b703`.
Совпадение локального файла с этой публикацией ещё не установлено.

### 1.2. Архитектура и реальные кванты

| Параметр локального GGUF | Значение |
|---|---|
| `general.architecture` / `general.name` | `hy_v3` / `Hy3` |
| `block_count` / `nextn_predict_layers` | 81 / 1: **80 основных + 1 MTP** |
| Attention | Full GQA, 64 query heads, 8 KV heads, head dim 128 |
| Hidden / dense FFN | 4096 / 13312 |
| Основные FFN | `blk.0` dense; `blk.1..79` MoE |
| MoE | 192 routed experts, top-8, один постоянно активный shared expert |
| Routed / shared FFN | 1536 / 1536 |
| Router | sigmoid (`gating_func=2`), selection bias, norm=true, scale≈2,826 |
| Q/K norm / epsilon | На голову, длина 128 / ≈1e-5 |
| RoPE base | 11 158 840 |
| Контекст в metadata | 262 144; доступный на этом ПК контекст ещё не определён |
| Tokenizer | GPT-2 BPE, pre-tokenizer `hunyuan-dense` |
| Vocabulary / merges | 120 832 / 119 758 |

Full attention и порядок операций подтверждаются чтением выбранного ниже
`hy-v3.cpp`: Q/K RMSNorm **до RoPE**, RoPE типа NeoX; dense SwiGLU,
routed MoE плюс shared SwiGLU. Отдельного множителя включения shared expert нет.
Это другой граф, чем `hunyuan-moe`, и не Step с sliding window или GLM с DSA.
Публичный [config Tencent](https://huggingface.co/tencent/Hy3/blob/main/config.json)
согласуется с числом основных слоёв, dense-prefix, GQA, MoE и одним MTP.
Числовую эквивалентность предстоит проверить.

Название `Q3_K_M` не задаёт единый тип хранения:

| Тип | Число тензоров во всём GGUF |
|---|---:|
| F32 | 488 |
| Q8_0 | 568 |
| IQ3_XXS | 156 |
| IQ4_XS | 77 |
| Q6_K | 5 |
| Q3_K | 2 |
| Q4_K | 1 |
| Q5_K | 1 |

Основные gate/up experts преимущественно IQ3_XXS, down — IQ4_XS;
есть исключения IQ4_XS/Q5_K/Q6_K. В MTP gate/up — Q3_K, down — Q4_K.
Attention, shared FFN и dense FFN — Q8_0; embedding и output — Q6_K;
router, его bias и нормы — F32. Размер строки и expert stride брать из
каждого TensorInfo, не из имени файла или одного «общего» кванта.

Для типичного основного блока одна gate/up/down тройка занимает
2 408 448 + 2 408 448 + 3 342 336 = **8 159 232 байта**.
Без попаданий в GPU-кэш top-8 по всем 79 основным MoE-блокам требуют
**5 223 088 128 байт = 4,864 ГиБ** экспертных весов за decode-проход.
Это расчёт объёма H2D при однократном копировании выбранных матриц,
без MTP, attention и служебных данных; фактический трафик ещё не измерен.

### 1.3. Память на этом ПК

`GlobalMemoryStatusEx` показал 125,555 ГиБ физической RAM, доступной Windows.
`nvidia-smi` показал RTX 5090, total 32 607 МиБ, driver 581.80.
95% этих объёмов — **119,277 ГиБ RAM** и **30 976,65 МиБ VRAM**.
Лимит относится к системе целиком, включая другие процессы.

Полная основная модель уже больше 95%-бюджета RAM. Нельзя заранее копировать
весь GGUF в heap или закреплять весь mmap. План размещения:

1. Основные non-routed веса ≈8,604 ГиБ держать на GPU, если помещаются вместе
   с KV и workspace; позволять ОС вытеснять исходные file-backed страницы.
2. Часть routed matrices держать постоянно в VRAM; остальные обслуживать
   через ограниченный RAM working set и небольшой pinned ring.
3. Для исключения SSD-чтений исследовать раздельное владение весами в RAM
   и VRAM: GPU-resident матрицы не должны требовать постоянной RAM-копии.
   Наличие GPU-кэша само по себе не освобождает heap или locked pages.
4. Режим mmap оставить рабочим вариантом при нехватке RAM; отдельно измерять
   page faults, disk reads и влияние повторного чтения вытесненных экспертов.

Для 80 основных слоёв чистый KV payload на токен равен
`80 × 2(K,V) × 8 × 128 × sizeof(KV)`.
При context4096 это **2,5 ГиБ F32** или **1,25 ГиБ F16**;
дополнительный MTP KV — 32/16 МиБ соответственно.
При 262144 основной F16 KV потребует **80 ГиБ**, ещё без весов.
Максимум из metadata не является подходящим стартовым контекстом.
Оценки не включают padding, batch workspace, graph buffers и allocator reserve.

## 2. Основа реализации

В основном `third_party/llama.cpp` обнаружены старые Hunyuan architectures,
но нет регистрации `hy_v3`. Перенаправление Hy3 в `hunyuan-moe` недопустимо:
различаются router, dense-prefix, порядок Q/K norm и MTP.

Подходящий **кандидат для проверки** уже лежит локально:

- Unsloth llama.cpp `86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9`;
- архив `build-local/llama-glm-86ebfef2.tar.gz`, 37 493 950 байт;
- SHA-256 архива `f8e524b635b726bae74fd8f84bb9249e5b09384c63207f6707c5a3f921acad99`;
- `src/models/hy-v3.cpp`, SHA-256
  `224921b8ce6f9be02dc1252ef6847a93f81386021b24d7d08eb25444e787d62e`.

Хэши проверены локально; loader в архиве совпадает с распакованным файлом
в `build-local/glm5next-candidate-cuda/_deps/glm5next_candidate-src`.
Он содержит основной граф, `graph_mtp`, поддержку общего embedding/output
и фильтрацию KV основной модели и MTP через API зависимости.
Это подтверждает наличие кода, но не успешную загрузку нашего GGUF.

Upstream [PR llama.cpp #25395](https://github.com/ggml-org/llama.cpp/pull/25395)
добавил Hy3 и `draft-mtp`. Автор приводит внешний тест на RTX 5090:
5,81–6,31 токена/с без speculation и 7,97 с MTP. Условия не воспроизведены
в Strata; это основание проверить подход, а не обещание скорости.
[AngelSlim предупреждает о несовместимости ранних GGUF](https://huggingface.co/AngelSlim/Hy3-GGUF#build)
с новым upstream. Поэтому сначала проверять tensor names/shapes, порядок
операций и tokenizer, затем закреплять dependency и patch hashes.
При несовместимости оформить отдельный проверяемый адаптер; исходный GGUF
не переписывать и новые веса автоматически не скачивать.

Использовать отдельные `backends/hy3` и `build-local/hy3-cuda`.
Собирать dependency из проверенного архива в собственном `_deps`.
Не использовать изменяемый GLM/Step build как общую библиотеку.
TF32 и routed-stride исправления Step/GLM рассмотреть для Hy3 только после
проверки нужных путей и числовых тестов на его формах и квантах.

## 3. Что переиспользовать в Strata

| Существующая часть | Применение и ограничение |
|---|---|
| [GGUF reader](../../tools/gguf_reader.py) | Читать metadata/directory; добавить отдельный строгий Hy3 contract |
| [Общий transport](../../backends/common/README.md) | Проверенные диапазоны, file reads, pinned ring, CUDA events, статистика |
| [GLM backend](../../backends/glm5next/) | Экспертные диапазоны, mixed quants, memory pressure, GPU audit и MTP-эксперименты |
| [Step backend](../../backends/step35/) | Native pipe, selected-copy, кэш матриц, batching/reuse, проверки отмены и HTTP |
| [Step adapter](../../serve/step35.py) | Образец API-преобразования и потокового parser; грамматика Hy3 отдельная |
| [Устройство Strata](../DETAILS.md) | Принципы размещения, перекрытия копий и вычислений, измерения |

Не переносить размеры слоёв, expert counts, KV state, шаблоны, EOS и patch
offsets из другой модели. CPU в первом профиле обслуживает I/O, scheduling,
tokenizer и sampler; все матричные вычисления модели выполняются на GPU.
CPU-expert режим, если понадобится, оценивать позднее отдельным профилем.

## 4. Этапы и критерии приёмки

### P0. Формат, dependency и небольшие проверочные программы

- **P0.1:** сохранить инвентаризацию GGUF в JSON: metadata, names/shapes/types,
  offsets/bytes, main/MTP, hashes заголовка и шаблона, все range checks.
  **DONE HY3-01:** постоянный инспектор и [отчёт](HY3_INSPECTION.json).
- **P0.2:** строгий loader contract для данного файла: 80+1, dense0/MoE1..79,
  MoE80+NextN, `exp_probs_b` без `.bias`, формы всех матриц и норм.
  Отказывать при пропущенных обязательных весах; upstream `NOT_REQUIRED`
  не считать достаточной проверкой. Добавить malformed/truncated fixtures.
- **P0.3:** изолированная CMake-сборка и vocabulary-only/tokenizer oracle;
  manifest содержит source SHA, archive SHA, compiler/CUDA, flags, patch hashes.
- **P0.4:** синтетические CUDA fixtures: Q8_0/Q6_K и все expert quants,
  отдельные gate/up/down strides, sigmoid+bias top-8, нормализация/scale,
  dense/shared, Q/K norm→NeoX RoPE→GQA. Сравнить с CPU/F32 oracle,
  зафиксировать допуски, проверить batch1 и несколько токенов.
  **DONE HY3-02:** 150/150 kernel cases и 86/86 graph/state checks на RTX5090;
  [ядра](HY3_CUDA_KERNELS_VALIDATION.json), [граф](HY3_CUDA_GRAPH_VALIDATION.json).
  Размеры графа уменьшены; mixed-quant сравнения имеют явно заданные допуски.
  Это не проверка logits полного checkpoint и не измерение его скорости.
- **P0.5:** проверить main/MTP load flags и KV filters. MTP-off должен исключать
  вычисление блока80; отдельно подтвердить, какие веса реально аллоцируются.
  Возможность пропустить NextN projections не доказывает пропуск MoE80.
  **HY3-01:** эта проблема подтверждена compiled no_alloc oracle и исправлена:
  off исключает все веса MTP, on регистрирует все20.
  **DONE HY3-02:** на синтетическом GGUF проверены реальные weight allocations,
  main/MTP KV filters, execution, post-final-norm hidden и точный rollback/restore.
  MTP-off не регистрирует тензоры MTP; main logits/hidden побитово совпадают с on.

**Готово:** воспроизводимая сборка, строгий contract и числовые fixtures.
Парсинг metadata и наличие `hy-v3.cpp` сами по себе P0 не закрывают.

### P1. Native inference без MTP

- **P1.1:** `strata-hy3` с pipe-протоколом Strata, загрузкой по validated ranges,
  bounded mmap/reader, точными INFO capabilities и отменой запроса.
  **DONE HY3-03:** Windows pipe, native contract до allocations, ENC/GEN/STOP/QUIT,
  свежий KV на запрос; 22 protocol checks. HTTP и сохранение сессий не включены.
- **P1.2:** синхронно копировать на GPU только выбранные gate/up/down матрицы;
  сначала без adaptive cache, pipeline, MTP и сохранения сессий.
  Проверять GPU execution всех model matmuls, включая router и shared FFN.
  **DONE HY3-03:** native file read → pinned16 МиБ → GPU, синхронные fences,
  проверка GPU bytes, mixed types и переноса больше staging; 19 runtime checks.
  GPU audit отклоняет CPU math и whole-expert copies до выполнения.
- **P1.3:** сравнить logits и greedy token IDs с графом выбранной зависимости
  на тех же квантах и input IDs. Референс допускает медленное offload-исполнение;
  не требовать полного GPU-resident размещения 125 ГиБ весов.
  Корпус: русский/английский/китайский, код, числа, короткий и длинный prompt.
  **DONE HY3-03:** шесть prompts, включая 362 токена, и два повтора в каждом
  режиме; все token IDs и полные logits совпали побитово. F32 KV, batch17,
  context2048; TF32, CUDA fusion, MTP/cache/pipeline выключены.
- **P1.4:** измерить load, prefill, TTFT, decode, реальные H2D bytes,
  working set/commit/pinned RAM, global VRAM; проверить OOM/error recovery,
  cancellation и unload/reload. Не запускать тяжёлый тест при отсутствии бюджета.
  **ЧАСТИЧНО HY3-03/HY3-04:** метрики, sampled global память, отмена/повтор и два
  load/unload прошли; [полный отчёт](HY3_MODEL_VALIDATION.json). Host working set
  ограничен по system target93%, global guard95%. В HY3-04 проверены injected
  pinned OOM, ReadFile error, partial-copy cancel, simulated pressure,
  unload/reload и точные logits после каждого отказа —
  [32 runtime checks](HY3_RECOVERY_VALIDATION.json). Реальное заполнение памяти,
  driver OOM/device loss и восстановление при внешнем pressure не проверены.

**Готово:** осмысленная генерация и сравнение с oracle на полном локальном GGUF,
без скрытого CPU matmul; сохранён воспроизводимый baseline.

### P2. Tokenizer, шаблон, reasoning, tools и HTTP

- **P2.1:** точное BPE/UTF-8/token-piece соответствие oracle, `parse_special`,
  special IDs, BOS и EOG. GGUF EOS=120025, BOS=120000, PAD=120002,
  `seperator_token_id`=120007. Не заменять pre-tokenizer на Qwen/DeepSeek.
  Публичный config задаёт EOD=120026, но локальная vocabulary на этом ID
  содержит placeholder; EOG-набор получить и проверить через выбранный loader.
- **P2.2:** renderer локального Jinja template; режимы `no_think`/`low`/`high`,
  `preserved_thinking`, несколько system-сообщений, tool results,
  `raw_last_assistant`, generation prefix. Тестировать fixtures against oracle.
  **DONE HY3-04:** `Hy3Template`, неизменённый fixture10223 bytes, все144
  native rendering/token-ID comparisons прошли для нового adapter.
- **P2.3:** потоковый parser для `<think:opensource>`, `<tool_calls:opensource>`,
  `<tool_call:opensource>`, `<tool_sep:opensource>`, `<arg_key:opensource>`
  и `<arg_value:opensource>` с закрывающими тегами. Проверять разрыв любого
  тега/UTF-8 между chunks, несколько вызовов, строки и JSON-значения аргументов,
  escaping, неполный вывод и stop внутри reasoning/toolcall.
  **DONE HY3-04, CPU:** `Hy3OutputParser`/`Hy3StopParser`, 17 tests и7327
  fragmentation sequences. Вызовы выдаются после проверки целой группы;
  malformed/truncated/undeclared calls остаются текстом. Нативные raw strings
  не XML-escaped; неоднозначные delimiter/follower sequences отвергаются
  renderer. Base-type conversion не заменяет полную JSON Schema validation.
- **P2.4:** отдельный `serve/hy3.py`: OpenAI/Anthropic request/response,
  reasoning, JSON/SSE, usage/finish_reason, tools, отмена и следующий запрос.
  Преобразовать строковые OpenAI arguments в mapping до Jinja `.items()`;
  сохранять call IDs и порядок results. Ошибочные arguments не исполнять.
  **DONE HY3-05:** JSON/SSE обоих API, request validation, реальные token-ID
  comparisons, локальный tool/result round-trip, disconnect и следующий запрос.
  Effort low/high проверен на scripted HTTP и template fixtures; full-model
  HTTP corpus использует no_think. Подробности и границы — в статусе.
- **P2.5:** отдельный профиль, список моделей, настройки UI и smoke API/web chat;
  initial bind `127.0.0.1`. Capability флаги включать после соответствующих тестов.
  **DONE HY3-05:** exporter в новый каталог, строгая проверка tokenizer/template,
  /health, /v1/models, /settings, переключатели Off/Low/High; реальный browser
  chat через CLI-профиль ответил4. Это экспериментальный synchronous профиль.

**Готово:** fixtures совпадают с шаблоном/oracle, полный цикл toolcall→result→answer
проходит через оба API, потоковый и обычный ответы согласованы.

### P3. Кэш и асинхронный конвейер STRATA

- **P3.1:** matrix cache с ключом model generation/layer/tensor/expert/type;
  byte-budget по реальному allocation size, pin активных entries,
  корректная eviction и release. Отдельно учитывать холодный SSD и RAM hit.
  **HY3-06: cache DONE в synchronous объёме:** identity/generation, cap,
  frequency eviction/reuse, D2D hits, OOM bypass, error invalidation и release.
  Все потребители завершаются до eviction, асинхронных leases пока нет.
  Source/H2D/cache-fill/D2D учитываются отдельно; физические SSD reads и
  попадания в OS file cache ещё не разделены инструментально.
- **P3.2:** bounded RAM→pinned→H2D→compute pipeline на общем transport.
  **HY3-07 DONE в проверенном объёме:**4 слота×4 МиБ, readers1/2, native file
  reads, CUDA events, plan pins, cancel/error drain, recovery, exact full-model
  logits/IDs и короткий speed sweep. Общий transport не изменён. Kernel-level
  profiling и длительные рандомизированные A/B остаются отдельной проверкой.
  Ждать H2D event до чтения device slot и consumer event до его перезаписи;
  source mapping и destinations живут до завершения использующих их операций.
- **P3.3:** включить batching выбранных ranges, allocation reuse и prefill
  admission по отдельности. Сверять точные GPU bytes и logits с P1 при mixed
  quants, boundary offsets, смене экспертов, отмене и выгрузке.
  **Частично HY3-06/08:** allocation reuse и opt-in tensor batching реализованы.
  Batching прошёл exact full-model checks; его speed effect невелик и пока
  не меняет default. Отдельная prefill admission policy ещё не реализована.
- **P3.4:** раздельное размещение RAM/VRAM и адаптивный горячий набор;
  бюджет учитывать вместе с non-routed, KV, scratch, staging, MTP и внешними
  процессами. Cache cap уменьшается при давлении, отменяет admission и даёт
  понятную ошибку, если неизбежный рабочий набор уже не помещается.
- **P3.5:** подобрать readers/chunk size, размер GPU-кэша и host-copy метод.
  Измерять H2D/compute overlap по CUDA timeline, не по сумме CPU wait timers.
  Дополнительную загрузку экспертов следующего слоя разрешать лишь при наличии
  обоснованного предиктора; фактический router следующего слоя ещё не известен.
- **P3.6:** новая сессия, прогретая и смена темы; page faults/disk reads,
  cache hit по байтам, peak RAM/VRAM, stress pressure и bounded cancel/drain.

**Готово:** output parity с P1, доказанное перекрытие копий и вычислений,
устойчивая память и измеренное улучшение на нескольких запросах.

### P4. Сессии и длинный контекст

- Full-attention KV snapshot/restore, prefix reuse, truncation/context shift,
  abort/retry, разные session IDs; восстановленный decode сверить с fresh prefill.
- Профили context2048/4096/8192, затем выше при наличии бюджета. Сначала F32
  для числового baseline, F16 для рабочего профиля после сравнения;
  Q8 KV и RAM-offload KV — отдельные эксперименты с проверкой качества.
- Не переносить SWA ring Step или recurrent/DSA state GLM. Для MTP потребуется
  согласованное сохранение основного KV, draft KV и состояния speculative driver.

**Готово:** несколько диалогов, восстановление и переполнение контекста
не меняют корректность и не приводят к неограниченному росту памяти.

### P5. Native MTP из этого же GGUF

- **P5.1:** использовать блок80. Loader показывает последовательность
  `enorm(embedding)` и `hnorm(post-final-norm hidden)` → concat(e,h) → eh_proj →
  полный decoder block → `nextn.shared_head_norm` → общий output.
  Сверить промежуточные hidden states и draft logits с oracle.
- **P5.2:** разделить main/draft KV. Embedding/output брать из основной модели
  с проверкой lifetime и единственного владельца; дополнительных этих матриц
  в локальном блоке80 нет. Не загружать вторую копию всего trunk ради draft.
  Проверить resident MTP ≈1,729 ГиБ и streaming MTP с отдельным cache budget.
- **P5.3:** batched target verification, dedup expert ranges между draft
  positions. Исправить и проверить hidden/token position alignment,
  acceptance, rejected suffix rollback, catch-up и bonus token.
  Один MTP-слой может предлагать несколько токенов последовательными шагами.
- **P5.4:** greedy parity при depth1/2/3, затем4 только при пользе; EOS/stop,
  каждый rejection position, длинный prompt, near-context-limit, cancellation,
  session save/restore. Sampling включать после корректной проверки распределения
  и rejection sampling; greedy acceptance не подтверждает stochastic correctness.
- **P5.5:** два A/B: одинаковый target cache для оценки стоимости алгоритма;
  затем лучший MTP-off профиль с памятью draft, возвращённой target cache,
  против лучшего MTP-on в том же общем 95%-бюджете.
  Логировать draft/verify/repair/catch-up time, accepted/emitted tokens,
  H2D bytes на выходной токен, TTFT и суммарное время запроса.

**Готово:** correctness и повторяемый выигрыш по полезным выходным токенам/с
в ограниченном бюджете. При отсутствии выигрыша MTP остаётся опцией,
а основной быстрый профиль работает без него. Чужие замеры это условие не закрывают.

### P6. Установка, профиль, регрессии и документация

- `setup_hy3.py` и подготовка отдельного профиля: проверка модели, manifest,
  dry-run оценки памяти, явные ошибки несовместимости, help/INFO.
- Серия замеров с фиксированными prompt IDs/template hash, seed, sampler,
  batch/ubatch, context/KV, cache cap, readers/chunks и source/patch SHA.
  Отдельно новый процесс, прогретые веса, новый prompt и prefix-cache hit.
- После прогрева минимум три измерения каждого варианта с чередованием порядка;
  сообщать median и диапазон, prefill tokens/s, TTFT, decode tokens/s,
  full-request latency и память. Указывать, считаются ли reasoning и EOS.
- Проверить настройки/установку/API Qwen, DeepSeek, GLM и Step в затронутых
  местах; полные GPU smoke запускать последовательно при свободном бюджете.
- Рекомендуемые default выбирать по замерам. Опубликовать поддержанные режимы,
  ограничения, команду запуска и точку отката на предыдущий профиль.

**Готово:** воспроизводимый рабочий профиль Hy3 и отчёт на этом ПК;
возможности MTP, tools и sessions включены только в проверенном объёме.

## 5. Стартовые варианты экспериментов

Native pipe CLI уже создан; команды приведены в
[README](../../backends/hy3/README.md#windows-synchronous-baseline).
Текущий baseline: context2048, batch17, F32 KV, pinned16 МиБ, один синхронный
file read; MTP/cache/pipeline/fusion выключены. Остальной перебор ниже — будущая работа.

| Параметр | Начальное значение / перебор |
|---|---|
| MTP | off для P1; затем depth1/2/3 |
| Context | 2048 baseline, 4096 первый рабочий, 8192 следующий |
| KV | F32 для oracle; F16 после числового сравнения |
| Sampler | temperature0 для correctness; затем 0,9/top_p1 по GGUF |
| GPU expert cache | 0 для sync baseline; запросы cap8/12/16 ГиБ с runtime clamp |
| Pinned staging | Небольшое ограниченное кольцо; chunks4/8/16 МиБ |
| Readers | 1, затем2; больше только по результатам |
| Session/prefix reuse | off при первом A/B; отдельный вариант позже |
| RAM / VRAM | Global cap95%; дополнительный reserve для пиков аллокаций |

Даже 16 ГиБ кэша не являются гарантированно допустимыми: к ним добавляются
8,604 ГиБ базовых весов, KV, граф, scratch и внешняя нагрузка.
95% — верхняя граница полезного размещения, а не требование заполнить память.
Перед каждым тестом заново измерять свободную память, не завершать чужие процессы.
В HY3-03 sampled global память не превысила95%; это не заменяет pressure tests.

## 6. Карта будущих изменений и порядок

Inspector, contract, CPU oracles и проверки tokenizer/template созданы в HY3-01;
CUDA fixtures — HY3-02; pipe engine, runtime/model checks — HY3-03.
HTTP checker и profile созданы в HY3-05; MTP checker остаётся будущей работой.

| Область | Планируемые файлы |
|---|---|
| Формат/contract | `tools/inspect_hy3_gguf.py`, `tools/hy3_loader_contract.py`, их CPU tests |
| Изолированный engine | `backends/hy3/CMakeLists.txt`, `main.cpp`, runtime/GPU audit и checks |
| Template/API | `serve/hy3.py`, `serve/fixtures/hy3_chat_template.jinja`, parser/API tests |
| Build/profile | `tools/setup_hy3.py`, `tools/prepare_hy3_profile.py` |
| Проверки/замеры | `tools/check_hy3_engine.py`, `check_hy3_http.py`, `check_hy3_mtp.py` |
| Отчёты | `docs/hy3/HY3_INSPECTION.json`, build/validation/benchmark reports |

Порядок: **P0 → P1 → P2/P3 → P4 → P5 → P6**. Tokenizer P2.1 можно
делать во время P0. MTP-probe после P1/P3 допустим до полной готовности P4,
но не должен объявлять поддержку MTP sessions до проверки rollback/restore.
Следующий результат — HY3-09: изоляция intermittent fixture mismatch,
профилирование чтения/admission и отдельное измерение prefill policy;
подробнее в [статусе](HY3_IMPLEMENTATION_STATUS.md).

## 7. Контроль завершения

- [x] Строгий contract принят для локального GGUF, dependency закреплена (HY3-01).
- [x] Full-model inference и logits/token parity подтверждены на GPU (HY3-03, MTP-off).
- [x] Реализованы tokenizer/template, reasoning, tools и API (HY3-04/HY3-05, границы в статусе).
- [x] Кэш и асинхронный pipeline корректны и дают измеренный эффект (HY3-06/07, короткий corpus; границы в статусе).
- [ ] Глобальные бюджеты RAM/VRAM и pressure/cancel/unload проверены.
- [ ] Сессии и заявленные контексты работают после restore/shift.
- [ ] MTP проверен на корректность и скорость; default обоснован A/B.
- [ ] Финальный быстрый профиль и замеры P6; экспериментальный профиль и CPU-регрессии готовы в HY3-05.
