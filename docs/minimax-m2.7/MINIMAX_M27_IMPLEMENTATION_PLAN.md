# План внедрения MiniMax-M2.7

Дата исследования: **2026-10-07**, `Asia/Yekaterinburg`.
Исходная ревизия Strata: `e111f63f89aa77d69aed46b2150f7e61aaa5b05d`,
с существующими незакоммиченными изменениями других backend.
Ход выполнения: [MINIMAX_M27_IMPLEMENTATION_STATUS.md](MINIMAX_M27_IMPLEMENTATION_STATUS.md).

Цель — запустить в Strata
`H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf`
на Windows, 128 ГиБ RAM и RTX 5090 32 ГиБ: вычисления модели на GPU,
ограниченные кэши экспертов, асинхронную доставку SSD/RAM → VRAM,
диалоги, reasoning, tools и сессии. Глобальный бюджет RAM/VRAM — до95%,
включая ОС, остальные процессы и временные аллокации.
Рабочие профили Qwen, DeepSeek, GLM, Step, Hy3 и MiMo сохранить.

На 2026-10-08 выполнены MM27-01–06: contract/oracles, строгий GPU runtime,
синхронная доставка выбранных экспертов и optional packed GPU cache.
Full-model A/B, pressure/cancel/unload и context2K/4K прошли; точные измерения
и открытое tiny native-after-cache reload расхождение записаны в статусе.
MM27-07 добавил55 reload stress checks, обязательный replay-vs-initial gate
и сохранение EXE; старое расхождение не воспроизвелось и остаётся OPEN.
MM27-08 добавил bounded mmap readers и три пары full-model A/B.
Гибрид mmap-decode ускорил decode повтора, но замедлил decode первого/нового;
default file сохранён. Pressure/cancel/reload и2K/4K проверки прошли.
MM27-09 добавил tensor file pipeline: read/H2D/ring delivery, explicit cache D2D
completion, 1416 fixture checks и три full-model A/B пары (+22,9–32,2% decode).
MM27-10 добавил router lookahead до трёх матриц: +17,74–31,44% decode в трёх
парах A/B, CUDA event overlap и 2424 fixture checks; lifecycle/reload PASS.
MM27-11 добавил одно D2D-ожидание на тензор и pins незавершённых fills:
+6,61–14,24% decode в трёх A/B парах; 4920 fixtures и lifecycle/reload PASS.
MM27-12 проверил caps 12/16/18/20/24, readers 1/2 и chunks 4/8/16. Кандидат 20/2/8
не улучшил полное время в трёх A/B парах; сохранён экспериментальный 18/2/4.
Длинная генерация и24 pressure/4K lifecycle checks PASS. MM27-13 исправил
избыточное вытеснение: arena освобождает целые блоки с учётом plan/fill pins.
Controlled replay,4920 pipeline,47 lifecycle,55 reload и три full-model A/B пары
прошли. MM27-14 проверил блоки8/16/32/64 МиБ: default64 сохранён,32 не дал
ускорения в трёх A/B парах у VRAM-лимита. MM27-15 добавил запас для роста:
controlled replay,1152 pipeline/cache,37 lifecycle и три повтора каждого варианта
прошли; небольшой выигрыш не изменил defaults64/reserve0.
MM27-16 реализовал optional группы gate/up/down: совместная замена, readiness,
plan pins и admission один раз за router plan. 488 проверок и три пары на
полной модели PASS; все logits совпали со старым EXE. Медиана агрегата decode
3,912→3,930 токена/с не подтверждает устойчивого ускорения; default grouping off.
[Измерения и границы MM27-16](MINIMAX_M27_GROUP_CACHE.md).
MM27-17 реализовал optional RAM LRU read-only views, удаляемых после GPU fill.
Отложенное admission уменьшило лишние mappings;816 checks и full-logit parity
1344 generated tokens PASS. Три пары: RAM64 decode3,485 против3,941 токена/с
off; default RAM0 сохранён. Long256 проверил55,859 ГиБ views без numerical FAIL.
[Измерения RAM/GPU partition](MINIMAX_M27_RAM_CACHE.md).
MM27-21 проверил API adapters через scripted HTTP; native Engine bridge и
рабочий startup profile ещё TODO. Target94%/hard guard95% проверен.
Размеры ниже рассчитаны по тензорам. **MTP-тензоров в локальном файле нет**;
первый рабочий профиль будет без speculation, MTP — отдельный этап P5.
После переноса в `H:\models\MiniMax-M2.7` найдены DFlash Q3/Q4/Q5;
[инспекция и отдельный план подключения](MINIMAX_M27_DFLASH.md).
P5.DF-02: loader и borrowed head adapter проверены для всех трёх квантов;
90 коротких GPU checks PASS. P5.DF-03 добавил offline greedy driver,
full-logit corpus parity и [измерения скорости](MINIMAX_M27_DFLASH_BENCHMARK.md).
Q3/Q4/Q5 depths1/2/4/7 проверены; strict tokenwise verification устранил
изменение routing на близких scores в этом корпусе. Live EOS/cancel/pressure,
длинный контекст, sessions и serving остаются открыты; DFlash default off.

## 1. Проверенные исходные данные

### 1.1. Локальный файл и происхождение

| Параметр | Значение |
|---|---|
| Файл | Один GGUF, split metadata отсутствует |
| Размер | **138 342 384 352 байта = 128,841 ГиБ** |
| GGUF / alignment | v3 / 32 байта |
| Metadata / tensors | 43 записи / 809 тензоров |
| `header_end` / `data_start` | 8 287 960 / 8 287 968 |
| Все payloads | 138 334 096 384 байта = 128,834 ГиБ |
| Routed experts | 135 725 580 288 байт = **126,404 ГиБ**, 186 тензоров |
| Остальные веса | 2 608 516 096 байт = **2,429 ГиБ**, 623 тензора |
| Хранимые параметры | 228 689 764 864, сумма произведений tensor shapes |
| `general.architecture` | **`minimax-m2`** |
| Basename / finetune | `MiniMax-M2.7` / `ultra-uncensored-heretic` |

Имена уникальны; типы и размеры известны, offsets выровнены,
пересечений и выходов за EOF нет. Последний payload заканчивается на EOF.
Это проверка структуры, а не содержимого весов или полного loader contract.

SHA-256 первых `header_end` байт, без padding и payloads:
`9a011b2ad6fb20db3bb0cdfe9af08665939ee1f0cb0609ffe2bfee7f31c825db`.
Полный локальный checksum не рассчитан. На
[странице одноимённого файла llmfan46](https://huggingface.co/llmfan46/MiniMax-M2.7-ultra-uncensored-heretic-GGUF/blob/main/MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf)
опубликован полный SHA-256
`a0e7f9fb737e0bda01270975c7b02019e5f650ce82f563377865ba3b8aae6c0d`.
Совпадение с локальными весами не установлено; header hash с ним не сравнивать.

Metadata ссылается на базовую `MiniMaxAI/MiniMax-M2.7`, но source/converter
revision не записаны. Проверять именно этот finetune и его template;
качество и ответы официальной модели не являются oracle для изменённых весов.
Слово `BF16` в имени относится к происхождению: **BF16 tensors здесь нет**.

### 1.2. Архитектура

| Параметр локального GGUF | Значение |
|---|---|
| Блоки | 62, индексы0..61 |
| Hidden / expert FFN | 3072 / 1536 |
| FFN | Все62 слоя — MoE, 256 routed experts, top-8 |
| Dense-prefix / shared experts | Тензоров нет; исследованный граф их не использует |
| Attention | Full GQA во всех слоях, 48 query heads / 8 KV heads |
| K/V head dimension | 128 / 128 |
| Q / K / V matrices | `[3072,6144]` / `[3072,1024]` / `[3072,1024]` |
| Attention output | `[6144,3072]` |
| Q/K norm weights | **`[6144]` / `[1024]`, норма по всей проекции** |
| RoPE dimensions / base | 64 / 5 000 000 |
| RMS epsilon | ≈1e-6 |
| Router | Sigmoid (`gating_func=2`), bias `exp_probs_b.bias` |
| Context metadata | 204 800; локальная работа на таком контексте не проверена |
| MTP metadata / tensors | NextN/MTP keys отсутствуют, соответствующих tensors нет |

Критическая особенность прочитанного `minimax-m2.cpp`: после Q/K projections
RMSNorm применяется к 6144/1024 компонентам, **до reshape на головы**,
затем частичный NeoX RoPE на64 компонентах каждой128-мерной головы.
Перенос per-head norm из Hy3 изменит вычисление, даже при допустимых shapes.
Attention использует scale `1/sqrt(128)` и обычный full KV.
SWA, recurrent state и DSA в этом локальном графе нет.

Router выбирает top-8 с correction bias, а веса выбранных экспертов берёт
из sigmoid probabilities и нормализует. Отдельный `expert_weights_scale`
в metadata не задан; локальный graph при default0 не добавляет multiplier.
Не переносить значения2,826/3 из Hy3/Step и не добавлять shared FFN.

Публичный [config MiniMax-M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7/blob/main/config.json)
согласуется с per-layer Q/K norm, размерами и MoE; он также описывает MTP:
`num_mtp_modules=3`, `mtp_transformer_layers=1`, `use_mtp=true`.
Это описание исходной модели, а не доказательство наличия весов в нашем GGUF.
Source revision finetune не установлен; локальный contract строить по файлу
и проверенному loader, расхождения с опубликованными версиями фиксировать явно.

### 1.3. Реальные кванты и экспертные диапазоны

| Тип | Число тензоров | Размещение |
|---|---:|---|
| Q4_K | 375 | Embedding, Q/K/output attention, часть V/down, все gate/up experts |
| Q6_K | 61 | Output head, 30 V matrices и 30 down-expert tensors |
| F32 | 373 | Norms, router и selection bias |

Gate/up experts: `[3072,1536,256]`; down: `[1536,3072,256]`.
Одна матрица одного эксперта содержит4 718 592 элемента:
Q4_K — **2 654 208 байт / 2,53125 МиБ**, Q6_K — **3 870 720 / 3,69140625 МиБ**.
Row/expert stride и allocation charge вычислять по каждому tensor type.
Имя `Q4_K_M` не разрешает считать все down matrices четырёхбитными.

Если GPU-кэш не дал попаданий, однократная доставка top-8 gate/up/down
во всех62 слоях требует **4 241 424 384 байта = 3,950 ГиБ**
за decode-проход. Это расчёт экспертного H2D, без служебных данных;
он не является измерением PCIe bandwidth или токенов/с.

### 1.4. Память на этом ПК

Windows сообщает **125,555 ГиБ** физической RAM, доступной системе,
NVIDIA — **32 607 МиБ VRAM**, RTX5090, driver581.80.
95%-лимиты: **119,277 ГиБ RAM / 30 976,65 МиБ VRAM** глобально.
Резерв для неизбежных пиков выделяется до admission нового кэша.

Даже routed weights126,404 ГиБ больше95%-бюджета RAM. Нельзя копировать
весь файл в heap или закреплять весь mmap. План размещения:

1. ≈2,429 ГиБ non-routed weights на GPU, KV и workspace учитывать отдельно.
2. GPU expert cache плюс bounded RAM working set; холодные данные из файла.
3. Для GPU-resident матриц исключить обязательную постоянную heap/pinned RAM-копию;
   file-backed source pages должны быть вытесняемыми. Простое добавление GPU-кэша
   не освобождает уже созданную host-копию.
4. При недостатке RAM работать через mmap/native reads и отдельно измерять
   page faults/SSD bytes; не обещать SSD-free decode без измеренного размещения.

KV payload: `62 × 2(K,V) × 8 × 128 × context × sizeof(KV)`.

| Context | F32 KV, ГиБ | F16 KV, ГиБ |
|---:|---:|---:|
| 2048 | 0,96875 | 0,484375 |
| 4096 | 1,9375 | 0,96875 |
| 8192 | 3,875 | 1,9375 |
| 32768 | 15,5 | 7,75 |
| 204800 | 96,875 | 48,4375 |

Это payload без padding, batch workspace, allocator/graph buffers и checkpoints.
204800 context не подходит как старт: даже F16 KV больше всей VRAM.
Небольшой non-routed набор оставляет место для экспертов, но достаточность
общей памяти и влияние размера кэша нужно подтвердить в P1/P3.

## 2. Какую реализацию брать за основу

`minimax-m2` уже зарегистрирован в `third_party/llama.cpp` и локальном
Unsloth candidate. Предлагается отдельный backend `backends/minimax_m2`,
engine `strata-minimax-m2`, build `build-local/minimax-m2-cuda`.
В MM27-01 создан `backends/minimax_m2` с CPU oracles и сборкой
`build-local/minimax-m2-oracles`. CUDA build добавлен в MM27-02; runtime engine остаётся планируемым.

Первый кандидат для P0:

- Unsloth llama.cpp `86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9`;
- `build-local/llama-glm-86ebfef2.tar.gz`, 37 493 950 байт;
- archive SHA-256 `f8e524b635b726bae74fd8f84bb9249e5b09384c63207f6707c5a3f921acad99`;
- `src/models/minimax-m2.cpp`, 6362 байта, SHA-256
  `6574a8618655d8164588627c1a4d2fc4fa8953b65fd783b8c1849f1c5c93638e`.

Хэши проверены; loader в архиве, распакованном GLM candidate и `third_party`
совпадает. Это не утверждение о совпадении всех dependency файлов.
Локальный loader содержит только основной graph: нет ветки `graph_mtp`
и загрузки NextN. Поддержку основного графа и MTP оценивать раздельно.
[Upstream loader](https://github.com/ggml-org/llama.cpp/blob/master/src/models/minimax-m2.cpp)
использовать как внешний материал для сравнения, не как плавающий build pin.

Распаковывать проверенный архив в собственный `_deps`, сохранять manifest.
Патчи TF32/routed strides, GPU audit, literal guards, batching и CUDA Graphs
переносить только с тестом на формах3072/1536 и Q4_K/Q6_K этой модели.
Общую dependency существующих engine автоматически не обновлять.

## 3. Что переносить из Strata

| Существующая часть | Применение и ограничение |
|---|---|
| [GGUF reader](../../tools/gguf_reader.py) | Header/ranges; отдельный MiniMax contract |
| [Общий transport](../../backends/common/README.md) | Checked ranges, file readers, pinned ring, CUDA events, counters |
| [GLM backend](../../backends/glm5next/) | Mixed-quants planner, pressure checks, pipe/API, измерения |
| [Hy3 backend](../../backends/hy3/) | No-allocation oracles, matrix cache, bounded RAM delivery, tool adapter подход |
| [MiMo backend](../../backends/mimo2/) | Cache lookup/admission, literal guard, token/batch parity и verify diagnostics |
| [Step backend](../../backends/step35/) | Изолированная сборка, runtime checks, cancellation и API fixtures |
| [Описание Strata](../DETAILS.md) | Принципы размещения и перекрытия доставки с вычислениями |

Не переносить tokenizer, EOS/BOS, norm geometry, template, MTP contract
или измеренную скорость другой модели. CPU обслуживает I/O, scheduling,
tokenizer и sampler; все model matmuls в первом профиле выполняются на GPU.

## 4. Этапы внедрения и критерии перехода

### P0. Совместимость, зависимость и эталон

- **P0.1:** inspector и JSON inventory: metadata, все809 tensor records,
  shapes/types/offsets/bytes, группы памяти, header/template hashes,
  отсутствие NextN. **DONE в MM27-01:** постоянный inspector и полный JSON inventory.
- **P0.2 (DONE, MM27-01):** строгий contract: 62 MoE-блока, все globals, separate QKV,
  flattened Q/K norms6144/1024, `exp_probs_b.bias`, expert axes и mixed quants.
  Проверить отсутствие dense/shared/MTP tensors, допустимые metadata/defaults.
  Negative fixtures: missing/extra/duplicate, shape/type/row geometry,
  truncated/overlapping ranges, 64-bit offsets и неподдержанная архитектура.
- **P0.3 (DONE, MM27-01):** изолированная сборка vocabulary/template/no-allocation loader oracles;
  manifest source/archive/patch SHA, compiler/CUDA/flags. Зарегистрировать все809
  tensors без payload reads; сформировать точный EOG/BOS policy отчёт.
- **P0.4 (DONE, MM27-02):** CUDA fixtures Q4_K/Q6_K/F32, dequant/matmul/routed gather,
  gate/up/down strides, repeated experts и batch1/2/8. CPU/F32 reference
  для малых tensors, фиксированные числовые допуски и TF32 policy.
- **P0.5 (DONE для F32 activation baseline, MM27-02):** tiny full graph: per-layer Q/K RMSNorm до reshape, partial NeoX RoPE64,
  GQA48/8, sigmoid+bias selection, unbiased normalized weights, residuals.
  Проверить prefill/decode, masks/positions, FA off/on и первый generated token.
  Получено217/217 PASS с `STRATA_MM27_QUANT_F32=1`. Обычный MMVQ/MMQ путь
  дал9 logit FAIL; не считать его прошедшим тот же строгий gate. См. статус MM27-02.

**Результат P0:** строгий contract, oracles и numerical fixtures готовы для
F32 activation baseline. В MM27-02 добавлен opt-in reference режим: Q4_K/Q6_K
распаковываются на GPU и умножаются через cuBLAS F32 без Q8-квантования активаций.
Это проверка точности, не выбранный быстрый default и не ускорение.
Полный checkpoint и влияние точности/производительности обычных quant kernels — P1/P6.
Регистрация `minimax-m2` сама по себе не закрывает проверку модели.

### P1. Основная модель и синхронная доставка

- **P1.1:** native pipe с verified file ranges, bounded staging, selected-copy
  только активных gate/up/down. Сначала без MTP, adaptive cache, pipeline,
  CUDA Graphs и prefix/session reuse; исходные quants сохранить.
- **P1.2:** GPU execution audit всех model matmuls, router и output head;
  CPU literals/shape operations учитывать отдельно от вычисления весов.
  Для большого файла использовать mmap/native reads, не full heap load.
- **P1.3:** сравнить logits и greedy token IDs полного GGUF с графом выбранного
  pin на тех же весах/input IDs. Медленный offload oracle допустим;
  полностью размещать модель на GPU или деквантовать весь checkpoint в F32 не требуется.
  Корпус: русский/английский/китайский, код/числа, короткий и длинный prompt.
- **P1.4:** load, prefill, TTFT, decode, H2D/SSD bytes, память; cancel→recovery,
  ошибка→следующий запрос, unload/reload. Проверить prompt sizes около границ
  batch/ubatch, разные длины истории и первый decode после prefill.

**Готово:** корректный full-model baseline и отчёт на этом ПК, без CPU-expert fallback.

### P2. Tokenizer, reasoning, tools и API

MM27-19: P2.3 и входная canonical history часть P2.5 реализованы отдельно в
`serve/minimax_m2_history.py`: leading instructions, lossless reasoning/content,
ID/result correlation и порядок tool responses.33 unit methods,293 prompts /
71189 IDs и96 legacy cases PASS. Для native oracle исправлено округление JSON
float в отдельном target; исходный raw сохранён. Late instructions, неполные
result groups и неэкранированные структурные tool-теги отклоняются явно.
[Контракт, ограничения и результаты](MINIMAX_M27_HISTORY.md).
MM27-20: P2.4 реализован отдельным `serve/minimax_m2_tools.py`: atomic groups,
schema types/validation, raw strings, stop/truncation/UTF-8 и bounded buffer.
51 unit methods,281 native groups +281 histories /22079 IDs,8 completion replays,
3966 checks PASS. [Контракт и ограничения](MINIMAX_M27_TOOLS.md).
MM27-21: API adapters и scripted tool cycle проверены через real loopback
JSON/SSE OpenAI/Anthropic.191 tests,16 cycles,32 completion replays и272 report
checks PASS;32 API prompts /14284 IDs совпали с native oracle.
[Контракт и ограничения](MINIMAX_M27_API.md). Native Engine JSONL bridge,
startup profile, live model tool cycle и web chat ещё не подключены.

- **P2.1:** GPT-2 BPE с pre-tokenizer `minimax-m2`, vocab200064/merges199744.
  BOS200034=`]~!b[`, EOS/PAD200020=`[e~[`, UNK200021=`]!d~[`.
  Ролевой token `]~b]` имеет ID200019. Проверить `parse_special`, UTF-8,
  пробелы/CRLF, числа, contractions, точные IDs/token pieces и EOG через oracle.
- **P2.2:** воспроизвести встроенный Jinja template, SHA-256
  `893d908f7b5cc65fdde270dcae5ea1a99647c6a7ce572ae874a57b7160069566`.
  Он сам печатает BOS/system header, роли `user/ai/tool`, `[e~[` и generation
  prefix `]~b]ai\n<think>\n`. Проверить отсутствие двойного BOS.
  Официальный [generation config](https://huggingface.co/MiniMaxAI/MiniMax-M2.7/blob/main/generation_config.json)
  задаёт BOS200019, который отличается от локального GGUF; не подменять им200034.
- **P2.3:** сохранение reasoning только после последнего user turn, extraction
  `<think>...</think>` из history, default identity, первый system,
  `current_date/current_location`. Несколько system/developer сообщений
  нормализовать явно, не терять их молча. Встроенный template не читает
  `enable_thinking`: отдельный no-thinking template — другой проверяемый профиль,
  а не уже работающий переключатель, заимствованный из MiMo.
- **P2.4:** потоковый parser `<minimax:tool_call>`, `<invoke name="...">`,
  `<parameter name="...">` и `<response>`. Проверить schema types, строки/JSON,
  escaping/XML-подобный текст, несколько calls, порванные теги/UTF-8,
  незакрытый блок, stop/EOS в reasoning и tools.
- **P2.5:** canonical tool history: OpenAI JSON-string arguments→mapping для
  `.items()`, function name и call ID/result correlation. Template проверяет
  `last_tool_call.name`, включая доступ к последнему элементу списка;
  вложенный OpenAI `function` и несколько результатов требуют отдельных fixtures.
- **P2.6:** отдельный adapter OpenAI/Anthropic JSON/SSE, usage/finish_reason,
  reasoning/content, отмена/recovery и web chat. Profile text-only;
  media tokens в vocabulary не означают наличие vision/audio weights.

**Готово:** tokenizer/template совпадают с oracle; оба API и цикл
toolcall→result→answer работают, не теряя инструкции, ID и части ответа.

### P3. Кэши и асинхронный конвейер STRATA

MM27-05 добавил bounded GPU matrix cache, MM27-06 — packed allocations,
MM27-08 — bounded mmap working set. Универсального ускорения от mmap нет:
decode повтора быстрее, первого/нового запроса медленнее. MM27-09 добавил async file pipeline
текущей матрицы, MM27-10 — lookahead следующих матриц того же router.
MM27-11 объединил D2D-ожидания в пределах тензора (P3.3).
MM27-12 проверил cache/readers/chunks и длинную генерацию (P3.5).
MM27-13 добавил освобождение целых физических блоков с меньшей потерей live
cache и защитой plan/fill pins; controlled pressure, byte/logit parity и A/B PASS.
MM27-14 добавил настраиваемый размер блока8/16/32/64; меньшая потеря при единичном
освобождении подтверждена, speedup малых блоков у VRAM-лимита не подтверждён.
MM27-15 проверил запас роста:20 циклов frees/allocations стали0 в replay;
на полной модели выигрыш мал и defaults сохранены. MM27-16 добавил group
admission/replacement/pins трёх матриц эксперта. Density/bytes/logits/pressure
и три пары A/B прошли; устойчивого ускорения не получено, grouping default off.
MM27-17 реализовал P3.4 RAM LRU/partition в пределах собственных mapped views;
short A/B дал замедление, полезный speedup и policy длинных сессий открыты.
Под нагрузкой RAM лучше
проверять file-backed pages: доступный commit на этом ПК меньше свободной
физической RAM, а большие GPU allocations тоже увеличивают process commit.
Для RAM admission учитывать оба бюджета; не строить второй полный heap copy
поверх mmap. Реальный live budget остаётся95%, тестовая нагрузка MM27-04 —90%.


- **P3.1:** matrix cache с model generation/layer/tensor/expert/type key,
  byte-budget реальных allocations, pin используемых entries, корректная eviction.
  Проверить mixed Q4_K/Q6_K и физические guard/alignment bytes.
  Реализован синхронный вариант MM27-05. MM27-09 добавил явное ожидание
  cache D2D и защиту будущих попаданий в кэш на время плана текущего тензора.
- **P3.2:** bounded file/mmap→pinned ring→H2D→compute с CUDA events.
  До чтения device payload ждать H2D, до перезаписи слота — consumer completion;
  источники и destinations живут до завершения операций. Cancel/unload drains.
  MM27-09 реализовал bounded current-tensor file pipeline, event lifetime,
  pressure/fault/reload и H2D/delivery trace. MM27-10 добавил lookahead до трёх
  матриц и пересечение H2D/compute CUDA event intervals; кольца не увеличены.
- **P3.3:** reusable allocations, batch ranges, early refill, host-copy методы
  и grouping gate/up/down включать по одному. Bytes/logits parity с P1,
  включая последний chunk и смену selected experts.
  MM27-11 добавил opt-in tensor D2D batch, pins новых fills, queued faults,
  byte/logit parity и полный A/B; defaults сохранены.
  MM27-16 добавил opt-in semantic groups с readiness после copy fence,
  целиком закреплёнными тройками и запретом позднего admission. 488 checks
  и три пары PASS, но defaults сохранены из-за отсутствия устойчивого speedup.
- **P3.4:** managed RAM working set и GPU-resident expert partition, admission
  с global RAM/VRAM caps. Не дублировать весь routed набор в locked/heap memory.
  При внешнем давлении уменьшать cache, прекращать admission и возвращать
  понятную ошибку, если базовый рабочий набор уже не помещается.
  MM27-08 реализовал process working-set target94%, restore на release,
  три mapped reader варианта и проверки95%/pressure/cancel/reload/2K/4K.
  MM27-17 реализовал RAM LRU read-only views и удаление overlap после GPU fill.
  Admission ждёт ранее наблюдённого GPU refusal; protected readers не теряют
  source при trim/drop/cancel.816 checks, три короткие пары и long256 parity
  PASS; короткий workload медленнее, default0. Это partition собственных views,
  не гарантия отсутствия file pages GPU-экспертов в системном кеше Windows.
- **P3.5:** перебрать GPU cache12/16/20/24 ГиБ с live clamp,
  readers1/2, chunks4/8/16 МиБ, prefill admission и frequency decay.
  Замерять hit по байтам, disk reads/page faults, H2D и consumer wait;
  CUDA timeline должна подтвердить overlap, CPU timers его не доказывают.
- **P3.6:** cold/warm, новый prompt/смена темы, давление, отмена и reload.
  CUDA Graphs включать только после проверки dynamic expert IDs,
  адресов буферов, event lifetime и repeated-request parity.

**Готово:** bounded memory, корректность P1 и повторяемое ускорение;
сумму времени параллельных reader threads не выдавать за wall time.

### P4. Сессии, KV и длинный контекст

- Full-attention KV snapshot/restore, prefix reuse, truncation/shift,
  session isolation, отмена и retry. Сверять восстановленный ответ с fresh prefill.
- Context2K/4K/8K, затем16K/32K при бюджете. F32 KV для oracle,
  F16/Q8 — отдельный quality/performance A/B и проверка FA compatibility.
- Учитывать снижение expert cache при росте KV. KV offload в RAM — отдельный
  вариант с измерением PCIe трафика и pressure; RAM уже является ограничением.
- Не переносить SWA ring из Step/MiMo; сохранённое состояние отвергать при
  смене model/template/quant/KV configuration. Позднее добавить draft state для P5.

**Готово:** заявленные контексты и сохранение диалогов проверены; context204800
остаётся metadata-возможностью до отдельного измерения и реализации размещения.

### P5. Native MTP и DFlash — дополнительный этап

Для **native MTP** остаются два пробела: нет локальных NextN weights и нет
MiniMax MTP graph/loader в выбранном candidate. Основной текстовый профиль
выпускается без MTP.

Для **DFlash** найдены три локальных sidecar Q3/Q4/Q5. Статическая инспекция
и full draft SHA-256 выполнены в P5.DF-01; общий DFlash graph уже есть в candidate.
P5.DF-02 проверил loader oracle, borrowed embedding/head, согласование
vocabulary200055/200064, захват пяти target features и короткий verification/rollback.
P5.DF-03 реализовал offline greedy driver и измерил Q3/Q4/Q5 depths1/2/4/7
на heretic target с настоящим chat template. Итоговый screen побитно совпал
с serial target после добавления opt-in tokenwise verification; исходный
batched path провалил logit gate на router tie. Результаты повторений и границы
вывода — [benchmark](MINIMAX_M27_DFLASH_BENCHMARK.md).
Следующие gates: live EOS/cancel/pressure/recovery, длинный контекст и sessions.
Подробный список и размеры — [MINIMAX_M27_DFLASH.md](MINIMAX_M27_DFLASH.md).

- **P5.1:** установить фактическую доступность MTP checkpoint/sidecar и точный
  контракт трёх модулей официального M2.7. Config-флаг не доказывает наличие
  опубликованных весов. Зафиксировать revision/checksum, shapes, tokenizer,
  shared embedding/head, norms и соответствие hidden states finetune.
  Draft базового M2.7 допустим лишь как отдельно проверяемый кандидат для этого
  heretic checkpoint; совпадение имени архитектуры недостаточно.
- **P5.2:** реализовать/adaptировать graph, loader, main/draft state и размещение
  без второй копии trunk. Hidden boundary, порядок fusion/norm и KV topology
  установить по reference; не угадывать по GLM/Hy3/MiMo.
- **P5.3:** batched verification, acceptance/reject, rollback/catch-up,
  bonus token, EOS/stop/cancel/session restore. Greedy parity проверять от
  первого output token, включая prefill, все reject positions и длинный контекст.
  Sampling включать после корректного rejection sampling.
- **P5.4:** depth1/2/3, сначала одинаковый target cache, затем лучший off
  с возвращённой draft-памятью против лучшего on в том же95%-бюджете.
  Считать полезные output tokens/s, TTFT/request time, draft/verify/repair,
  H2D на принятый токен и acceptance. MTP default — только при выигрыше.

Локальные DFlash рассматриваются как отдельный метод со своей проверкой;
не называть его native MTP. Отсутствие готового P5
не блокирует P0–P4/P6 и оптимизации без speculation.

### P6. Установка, измерения, defaults и регрессии

- Отдельные setup/profile/help/INFO, architecture `minimax-m2`, checkpoint
  identity, text-only и mtp=false. Запуск на `127.0.0.1`; внешний bind только с API key.
- Записать exact prompt IDs/template hash, sampler/seed, context/KV,
  batch/ubatch, FA/graphs, cache/readers/chunks, dependency/patch SHA,
  driver/compiler/CUDA. Сохранять CLI и machine-readable reports.
- Минимум три замера после прогрева с чередованием порядка A/B; отдельно
  новый процесс, warmed weights, новый prompt и prefix cache hit.
  Публиковать median/диапазон load, prefill tokens/s, TTFT, decode tokens/s,
  request latency, counts, peak RAM/VRAM, SSD/H2D; явно учитывать reasoning/EOS.
- Выбрать быстрые defaults по полезной скорости и стабильности. Re-quantization
  или изменение template — отдельный артефакт/quality test, исходный GGUF сохранить.
- Регрессии затронутых common/setup/API частей для существующих моделей;
  GPU smoke проводить последовательно при наличии бюджета.

**Готово:** воспроизводимый text-only профиль исходного GGUF, измеренные
настройки, ограничения и способ отката. MTP допускается только в проверенном объёме.

## 5. Начальные настройки эксперимента

Это исходные варианты проверок; текущий experimental CLI и проверенные
настройки приведены в статусе и backend README. Таблица не задаёт defaults.

| Параметр | Начало / перебор |
|---|---|
| MTP / multimodal | off / disabled |
| Context | 2048 baseline, 4096 рабочий кандидат, затем8192 |
| KV / FA / CUDA Graphs | F32 / off / off; последующие варианты после parity |
| Sampler | temperature0 для correctness; затем1,0/top_p0,95/top_k40 как reference-кандидат |
| GPU expert cache | 0 для sync baseline; cap12/16/20/24 ГиБ с runtime clamp |
| RAM cache | Бюджет после ОС, других процессов и transient buffers |
| Pipeline | reader1/chunk8 МиБ как старт; readers1/2, chunks4/8/16 |
| Prefix/session reuse | off в первом A/B |
| Global caps | До95% RAM/VRAM, плюс резерв до выделения пиковых buffers |

Temperature в GGUF не задана;1,0 взята из официального generation config,
её пригодность для finetune проверить. 95% — потолок полезного размещения.
Например, non-routed2,429 +cache24 +F32 KV4K1,938 ≈**28,367 ГиБ** ещё
не включает workspace, graph, staging и desktop; cap24 не гарантированно допустим.
Перед тестами заново измерять внешнюю нагрузку, чужие процессы не завершать.

## 6. Карта изменений и порядок

Inspector, contract, CPU oracles и checks созданы в MM27-01, CUDA checks —
в MM27-02, experimental engine/cache — в MM27-03–06. API/profile из таблицы
остаются планируемыми. Точные результаты — в статусе.

| Область | Предлагаемые файлы |
|---|---|
| Формат | `tools/inspect_minimax_m2_gguf.py`, `tools/minimax_m2_loader_contract.py`, CPU tests |
| Backend/oracles | `backends/minimax_m2/CMakeLists.txt`, `main.cpp`, loader/tokenizer/template/kernel/graph checks |
| Template/API | `serve/minimax_m2.py`, `serve/fixtures/minimax_m27_chat_template.jinja`, parser/API tests |
| Setup/profile | `tools/setup_minimax_m2.py`, `tools/prepare_minimax_m2_profile.py` |
| Model checks | `tools/check_minimax_m2_engine.py`, `check_minimax_m2_http.py`, позднее MTP checks |
| Отчёты | `docs/minimax-m2.7/MINIMAX_M27_INSPECTION.json`, validation/benchmark reports |

Порядок: **P0 → P1 → P2/P3 → P4 → P6**; tokenizer P2.1 можно начать в P0.
P5 требует отдельного решения по весам и реализации. Inspector, строгий contract и CPU oracles завершены в MM27-01.
MM27-02 завершил numerical CUDA fixtures для F32 activation baseline.
MM27-03 реализовал P1.1/P1.2 и первый corpus P1.3: selected file delivery,
49 tiny checks и5 полных сравнений с bit-exact logits. P1.4 выполнен частично;
context512, prompts44–190,8 generated tokens не закрывают длительные/context/pressure
проверки. Подробные измерения и ограничения — в статусе.
MM27-04 добавил live memory guard и прошёл40 lifecycle/context checks:
полные2K/4K окна, cancel/pressure recovery, unload/reload и реальную нагрузку≈90%.
Long-input stress не заменяет long-answer quality, physical SSD measurements
и sessions. MM27-05 добавил optional GPU cache:56 tiny cache checks,
сравнение cache off/on на полном GGUF и сохранённый uncached runtime regression.
MM27-06 добавил packed GPU allocations с hard backing cap:14 arena checks,
112 cache checks,3 пары full-model allocator A/B и29 cache-on lifecycle checks,
включая pressure и2K/4K. Decode ускорился ещё на11,8–16,7% относительно
отдельных CUDA allocations при том же cap18ГиБ и bit-exact full-model logits.
В двух ранних tiny native-after-cache reload comparisons были расхождения;
три поздних instrumented runs прошли без изменения engine math. Причина OPEN,
defaults сохранены. MM27-07 усилил reload checks и артефакты воспроизведения:
55 stress +36 lifecycle checks PASS; без нового engine/performance изменения.
Расхождение остаётся OPEN. MM27-08 реализовал три bounded mmap readers,
504 cache checks и три full-model A/B пары;55 reload,18 fixture lifecycle
и29 full-model lifecycle checks PASS. Mmap-decode ускоряет decode повтора, но decode первого
и нового запроса медленнее; default file сохранён. P3.4 выполнен частично.
MM27-09 реализовал tensor file pipeline: 1416 fixture checks, 55 reload, 18 tiny
и 29 full lifecycle checks; три A/B пары дали +22,9–32,2% decode. Добавлено явное
ожидание D2D, исходный native FAIL остаётся OPEN. MM27-10 добавил router
lookahead, CUDA event overlap и +17,74–31,44% decode в трёх A/B парах.
MM27-11 объединил D2D-ожидания с сохранением lifetime: +6,61–14,24% decode,
4920 fixtures и full-model lifecycle/reload PASS. MM27-12 выполнил подбор
cache/readers/chunks, три подтверждающих A/B пары, длинную генерацию и 24
pressure/4K lifecycle checks. Увеличение cap/chunk не улучшило полное время.
MM27-13 реализовал освобождение целых физических блоков и проверил pressure
replay: при снижении бюджета256→255 МиБ сохранено192 МиБ вместо полного сброса.
4920 pipeline,47 lifecycle,55 reload checks и три A/B пары PASS.
MM27-14 проверил размеры блоков8/16/32/64: default64 сохранён, маленькие блоки
часто выделялись заново у VRAM-лимита. MM27-15 добавил запас роста с прежним
порогом сокращения:50 arena/reclaim,1152 cache/pipeline,37 lifecycle checks и
13 full-model workloads PASS. Defaults64/reserve0 сохранены. MM27-16 реализовал
плотные группы gate/up/down: 488 checks и три full-model A/B пары PASS,
grouping default off, устойчивого ускорения нет. MM27-17 добавил P3.4 RAM LRU
и удаление views после GPU fill;816 checks и1344-token parity PASS. RAM64
замедлил короткий corpus, default RAM0 сохранён. Long256 достиг55,859 ГиБ views,
но законченных ответов и natural EOS не дал. MM27-18 снял предел256 в пользу
свободного context budget: английский ответ и повтор завершились на414 tokens,
17 live JSONL budget/EOS/cancel/recovery scenarios PASS. Русский запрос1..96
зациклился в reasoning и остановился по length1536: полный completion gate FAIL.
13 parser/tokenizer tests и18 real-output replay checks PASS; parser пока
не подключён к API. MM27-18b добавил optional request-local sampling:
T1/top_p0,95/top_k40 завершил русский запрос на398/567/1092 tokens при трёх seed.
Все семь полных запросов завершились по EOS; greedy regression и seeded repeat
побитные,48 CPU/14 pipe/48 parser checks PASS. **Ручной quality gate FAIL:**
английский final ошибочно указал отношение рассеяния10 вместо≈4,35.
[Sampling, измерения и границы](MINIMAX_M27_SAMPLING.md).
MM27-19 реализовал history normalization и canonical input calls/results;
MM27-20 — standalone P2.4 tool parser, MM27-21 — API adapters и scripted HTTP
tool cycle. Следующий этап — resident Engine поверх native JSONL и startup
profile, затем live GPU tools/API/web chat. Качество и independent
model oracle остаются gates P6. Sampling opt-in, default greedy и18/2/4 сохраняются.
Контроль без cache/pipeline побитно повторил1024 tokens и204865536 logits
русского цикла. Повторная проверка того же транспорта не заменяет проверку
sampling/качества и независимого model/framework reference.
При повторении numerical FAIL приоритет получает трасса первого расхождения
в сохранённой сборке. До установления причины default-профиль не утверждать.

## 7. Контроль завершения

- [x] Локальный GGUF принят строгим contract, dependency/patches закреплены (MM27-01).
- [x] Per-layer Q/K norm, RoPE и mixed-quants CUDA fixtures прошли (MM27-02, F32 activation baseline; быстрый путь имеет отдельный FAIL по логитам).
- [ ] Full-model GPU baseline совпадает с oracle от первого output token.
- [ ] Tokenizer/template, reasoning, tools, оба API и web chat проверены.
- [ ] Cache/pipeline bounded, output parity и измеренный выигрыш подтверждены.
- [x] Синхронный GPU matrix cache, tiny fault/byte checks и full-model off/on comparisons (MM27-05).
- [x] Packed GPU cache: hard backing cap, measured allocator A/B и full-model cache-on pressure/2K/4K (MM27-06).
- [ ] Установлена причина tiny native-after-cache reload discrepancy MM27-06 и проверено исправление; tolerances сохраняются.
- [x] Дополнительный reload stress, обязательный replay gate и сохранение запускаемого EXE (MM27-07); это не закрывает причину старого FAIL.
- [x] Bounded mmap readers: byte/logit parity,3 A/B, pressure/cancel/reload и2K/4K (MM27-08); универсального speedup нет, default file сохранён.
- [x] Tensor file pipeline: bounded ring/events, cache pins/drain, worker faults, full A/B и pressure/2K/4K (MM27-09); перекрытие с compute добавлено следующим этапом.
- [x] Router lookahead до трёх матриц, H2D/compute CUDA event trace, full A/B и 2424 fixtures (MM27-10).
- [x] Tensor D2D batch с pins незавершённых fills, faults/pressure/reload, 4920 fixtures и три full A/B пары (MM27-11).
- [x] Подбор cache/readers/chunks, три подтверждающих A/B пары, длинная генерация и 24 pressure/4K checks (MM27-12); прежний экспериментальный набор сохранён.
- [x] Освобождение целых arena blocks с учётом pins, controlled pressure replay,4920 pipeline,47 lifecycle,55 reload checks и три A/B пары (MM27-13).
- [x] Настраиваемые блоки8/16/32/64,1872 fixture checks,37 lifecycle и три A/B пары64↔32 (MM27-14); default64 сохранён.
- [x] Запас роста новых блоков, controlled oscillation,1152 cache/pipeline,37 lifecycle и три повторения каждого варианта (MM27-15); defaults64/reserve0 сохранены.
- [x] MM27-16: семантическая группа gate/up/down, плотность, readiness/pins, correctness/pressure и три пары измерений; ускорение не подтверждено, default off.
- [x] MM27-17/P3.4: bounded RAM view LRU, отложенное admission, partition с GPU-кешем, pressure/lifecycle/parity и short A/B; default0 из-за замедления.
- [x] MM27-18: output budget в пределах context, natural EOS английского ответа, полный повтор, live JSONL cancel/recovery и отдельный reasoning parser; [область проверки](MINIMAX_M27_COMPLETION.md).
- [x] MM27-18b: optional seeded sampling, завершение фиксированного RU/EN/ZH корпуса, bit-exact greedy/repeat и JSONL cancel/recovery; [результаты](MINIMAX_M27_SAMPLING.md). Это не общий quality PASS: ручная проверка одного final не прошла; исходный greedy цикл сохранён.
- [x] MM27-19/P2.3 и входная часть P2.5: история instructions/reasoning, canonical typed calls, ID/result correlation, порядок результатов, native prompt/token parity; [контракт и ограничения](MINIMAX_M27_HISTORY.md). API пока не подключён.
- [x] MM27-20/P2.4: отдельный потоковый parser tools со schema validation, atomic multi-calls, raw strings, UTF-8/stop/truncation/limit, native wire/history round trips и replay; [границы поддержки](MINIMAX_M27_TOOLS.md). Live model tool generation и оба API остаются TODO.
- [x] MM27-21/часть P2.5/P2.6: adapters обоих API, JSON/SSE через real loopback, scripted ID/result/answer cycle, usage/finish/UTF-8/cancel/recovery и native tokenizer replay; [проверенный scope](MINIMAX_M27_API.md). Native Engine bridge, startup profile и live GPU API ещё TODO.
- [ ] Широкая answer-quality проверка и независимый full-model oracle; причина фактической ошибки английского sampled final MM27-18b не установлена. До P6 sampler default остаётся greedy.
- [ ] P3.4/P3.5: полезное ускорение RAM policy на длинных законченных ответах, page residency/SSD traffic; MM27-17 выигрыша не подтвердил.
- [ ] Pressure/cancel/unload, sessions и заявленные контексты проверены.
- [ ] Есть изолированный профиль, замеры defaults и регрессии.
- [x] P5.DF-01: локальные DFlash Q3/Q4/Q5, static ranges/tokenizer checks и full draft SHA-256; новый путь target проверен.
- [x] P5.DF-02: no-allocation loader, borrowed head/vocab adapter, MASK, короткий GPU feature/block/rollback probe для Q3/Q4/Q5.
- [x] P5.DF-03a: offline greedy DFlash driver, corpus full-logit parity, acceptance и измерение полезной скорости Q3/Q4/Q5 depths1/2/4/7; исправлена batch-sensitive arithmetic в проверенном корпусе.
- [ ] P5.DF-03b: live GPU EOS/cancel/pressure/recovery, длинный контекст, sessions и полезный speedup для serving. Native MTP по-прежнему требует весов/backend.
