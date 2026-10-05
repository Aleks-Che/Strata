# План внедрения Step-3.7-Flash

Дата: **2026-10-05**, `Asia/Yekaterinburg`.
Исходная ревизия Strata: `97e016de11754cccd1b5550232c6c198cdbe0723`.
Статус и точка продолжения: [STEP37_FLASH_IMPLEMENTATION_STATUS.md](STEP37_FLASH_IMPLEMENTATION_STATUS.md).

Основная модель: `H:\models\Step-3.7-Flash\UD-Q4_K_S`.
Входной файл: `Step-3.7-Flash-UD-Q4_K_S-00001-of-00004.gguf`.
Стенд: Windows, Ryzen 9 9950X, 128 ГиБ RAM, RTX 5090 32 ГиБ.

Цель — добавить отдельный Step backend с GPU-вычислениями, асинхронной доставкой
выбранных экспертов RAM → VRAM, ограниченным GPU-кэшем, текстовым API, состоянием
сессий и проверенным native MTP. Профили Qwen, DeepSeek и GLM должны продолжать
работать. Vision — отдельное расширение после текстового пути.

Это **план**, а не отчёт о работающей поддержке. При подготовке прочитаны локальные
заголовки, проверены диапазоны тензоров, изучены код и первичные источники.
При первоначальной подготовке Step не загружалась в engine. Текущие результаты
реализации приведены ниже и в статусе. Рабочие настройки других моделей не
изменялись; новые веса не скачивались.

Начата реализация: P0.1–P0.5, P1.1–P1.3 и P2.1 выполнены, собраны отдельные CPU/CUDA
vocabulary oracles. На реальном словаре каждый прошёл 2190/2190 exact-ID/byte
comparisons. На RTX 5090 прошли 78/78 synthetic matrix/router cases с двумя
Step-local CUDA patches. Также прошли 75/75 native synthetic graph/state checks
и 215/215 template checks. STEP-04 добавляет 27/27 transport, 22/22 pipe checks и
полную UD-Q4_K_S: два завершённых ответа, exact native/pinned logits и A→B→A.
Скорость baseline без cache/pipeline: 3,70–4,05 токена/с, повтор A — 4,57.
Это последовательные наблюдения, не доказательство ускорения транспорта.
STEP-05 добавляет sync GPU cache и main memory controls: 41/41 transport,
11/11 cache/budget, 16 exact full-model responses. На повторных запросах в
off/on/on/off сравнении decode4,315 → 6,806 токена/с; cache остаётся opt-in,
первичное заполнение может ухудшать TTFT. STEP-06 добавляет shared async ring:
92 runtime/18 ring/14 budget cases, 32 exact полных ответа и CUDA timeline.
Финальный paired decode с cacheauto: 6,242 → 6,718 токена/с (+7,6%). Расширение P1.4/P3.5 остаётся.
STEP-08: optional prefill admission off, 111 runtime cases, 13/13 CTest,
28 exact reference comparisons. На medium511 полный request wall 60,137 → 43,550 с;
после такого prefill decode медленнее. Default прежний; результаты и границы — в статусе.
STEP-09: P2.2–P2.3 adapter/parser готовы, 148 Python tests, 215 native template
comparisons, реальный tool dialogue с локальным result stub. STEP-10 добавляет
явный HTTP profile/template selection, native EOG, stop strings и capabilities.
232 Python tests PASS; exported tokenizer 2190/2190 native comparisons.
OpenAI/Anthropic JSON/SSE и real-model disconnect проверки описаны в статусе;
интерактивный web chat и внешний MCP остаются отдельными gates P2.5–P2.6.

Уточнённый приоритет пользователя: **сохранить работоспособность существующего
приложения**. Отдельные Step source/build/profile; общие изменения только с
регрессией Qwen/DeepSeek/GLM. Не подключать Step к текущему запуску до проверки
полной модели. Уже существующие test failures отделять воспроизводимым baseline.

## 1. Проверенные исходные данные

### 1.1. Локальные GGUF

GGUF v3, alignment32. Имена частей, `split.no`, `split.count=4`,
`split.tensors.count=754` согласованы. Все 754 имени уникальны; рассчитанные
диапазоны не перекрываются и укладываются в свои файлы. У трёх частей с весами
последний тензор заканчивается точно на конце файла. Численное содержимое
весов и полные SHA-256 этих больших частей пока не проверены.

| Часть | Размер файла, байт | Тензоров |
|---|---:|---:|
| `00001-of-00004` | 5 232 064 | 0, метаданные и токенизатор |
| `00002-of-00004` | 49 972 080 128 | 351 |
| `00003-of-00004` | 49 973 587 520 | 316 |
| `00004-of-00004` | 14 212 292 736 | 87 |
| Всего | **114 163 192 448**, или **106,323 ГиБ** | **754** |

Первая часть допустима без весов: `header_end=5 232 063`,
`data_start=5 232 064`, длина файла `5 232 064`. Загрузчик должен открывать весь
split-набор, а не считать первый файл маленькой самостоятельной моделью.

SHA-256 первой части, рассчитанный локально:
`13104cf4d9cbb9ddefa67d927ba8f3a8508aa35ec508c48b6886e476abfb8fb8`.
Совпадение с опубликованным checksum не установлено; это локальный fingerprint.

| Данные | Байты | ГиБ |
|---|---:|---:|
| Все tensor payloads | 114 157 912 576 | 106,318 |
| Routed experts основной модели | 107 017 666 560 | 99,668 |
| Остальные веса основной модели | 7 140 246 016 | 6,650 |

В этом наборе 126 тензоров Q4_K, 361 Q8_0, 266 F32, один Q6_K.
Все routed gate/up/down — **Q4_K**; имя `UD-Q4_K_S` не означает, что все веса
имеют этот тип. `output.weight` — Q6_K, embedding — Q8_0; router и его bias — F32.

### 1.2. Архитектура: GGUF `step35`, пользовательское имя Step-3.7-Flash

| Параметр локального GGUF | Значение |
|---|---|
| `general.architecture` | `step35` |
| `general.name` | `Step-3.7-Flash` |
| Блоки основной модели | 45, индексы 0..44 |
| Attention | 12 full, 33 sliding-window; full в 0, 4, …, 44 |
| Sliding window | 512 |
| Query heads | 64 в full, 96 в sliding |
| KV heads / ширина головы | 8 / 128 |
| Embedding / dense FFN | 4096 / 11264 |
| FFN | 3 начальных dense, затем 42 MoE |
| MoE | 288 экспертов, top-8, shared FFN 1280 |
| Expert FFN | 1280 |
| Router | sigmoid, bias выбора, нормализация весов, scale3 |
| RoPE base full / sliding | 5 000 000 / 10 000 |
| SwiGLU clamp | routed7 / shared16 в блоках 43 и 44; остальные0 |
| Контекст в metadata | 262 144; работа на этом ПК не проверена |
| Токенизатор | GPT-2 BPE, pre-tokenizer `deepseek-v3` |
| Словарь / merges | 128 896 / 127 741 |
| GGUF BOS / EOS / PAD IDs | 0 / 128007 / 2 |
| MTP в локальных файлах | Не найден: нет NextN tensors, блоков ≥45 и NextN metadata |

В официальном [config.json](https://huggingface.co/stepfun-ai/Step-3.7-Flash/raw/main/config.json)
текстовая часть тоже относится к Step3p5. Он задаёт partial RoPE 0.5 для full
attention, 1.0 для sliding, масштабирование llama3 и **3 NextN слоя**.
В GGUF есть общий `rope_freqs.weight` shape `[64]`; соответствие этих факторов
официальной формуле проверять в P0, а не заменять одной глобальной RoPE-настройкой.
Все 45 локальных блоков относятся к основной модели: вычитать из них три MTP
слоя нельзя.

Step не использует GLM KDA/DSA/mHC. Нужны собственные GQA/SWA, head-wise attention
gate, Q/K RMSNorm, router bias и clamp. Совпадения embedding 4096 и top-8 недостаточно
для переноса формул GLM или его размеров буферов.

### 1.3. Размеры экспертных матриц и предварительная память

Один тензор gate/up имеет GGUF shape `[4096,1280,288]`, down — `[1280,4096,288]`.
Q4_K хранит 256 элементов в 144 байтах. Матрица одного эксперта занимает
**2 949 120 байт = 2,8125 МиБ**, тройка — 8,4375 МиБ.

Без cache hits расчётный трафик routed weights на один decode-проход:
`42 × 8 × 3 × 2,8125 МиБ = 2,768555 ГиБ`. Это расчёт по заголовкам,
не замер H2D, дискового I/O или прогноз токенов/с.

Остальные 6,650 ГиБ весов — только часть постоянных VRAM allocations. Дополнительно
нужны KV, scratch, CUDA graphs, staging, cache metadata и временное переупаковывание.
Для F16 K/V идеальный payload можно оценить как
`4096 × (12 × context + 33 × min(context,512))` байт: при context2048 это 162 МиБ.
Оценка предполагает отдельный ограниченный SWA-cache; реальные reserve, padding,
microbatch и rollback могут требовать больше. Проверить фактические allocations.

Модель меньше установленной RAM, но это не обещает отсутствия page faults:
нужны память ОС, других приложений и рабочие буферы. Не закреплять весь mmap
в pinned RAM и не создавать полную дублирующую копию весов по умолчанию.

### 1.4. MTP и vision поставляются отдельно

В указанной директории и её родителе найден только четырёхчастный основной GGUF.
На [официальной странице GGUF](https://huggingface.co/stepfun-ai/Step-3.7-Flash-GGUF/tree/main)
опубликованы `Step3.7-flash-mtp-Q8_0.gguf` (~3,71 GB), BF16 draft и
`mmproj-step3.7-flash-f16.gguf` (~3,97 GB). Это размеры страницы публикации,
не объёмы runtime и не проверенные локальные файлы.

Для P5 выбрать и проверить совместимый draft: архитектуру, число голов,
vocabulary/token IDs, embeddings, tensor layout, revision и квантование.
Не предполагать совместимость официального draft с Unsloth trunk только по имени.
Количество обученных MTP-голов и длина предложения — разные параметры.
Vision требует mmproj, preprocessing и отдельного бюджета; одних тегов
`vision-language` в основном GGUF для изображений недостаточно.

## 2. Основа реализации и существующие ограничения

В локальном `third_party/llama.cpp/src/models/step35.cpp` и в зависимости
GLM-кандидата уже есть `llama_model_step35`: per-layer heads, SWA, gate, clamps,
main/MTP tensor loading. Оба просмотренных файла имеют SHA-256
`4c42074b6f859b5734572eb0b1f1f07c8b783e20bfeb3a357871c77a8255c8b3`.
Это наличие кода, а не проверка его работы с данным split-набором.

Первый кандидат для P0 — изолированная сборка уже доступного Unsloth source
`86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9`, описанного в
[`SourceArchive.cmake`](../../backends/glm5next/SourceArchive.cmake).
Сравнить с основной веткой llama.cpp и с указанной в
[официальной инструкции](https://huggingface.co/stepfun-ai/Step-3.7-Flash#64-llamacpp)
веткой StepFun `step3.7`. Окончательный Step pin выбрать после loader/tokenizer/
GPU fixtures, сохранить полный commit SHA, archive SHA-256 и patch manifest.
Подвижные `main/master/step3.7` не являются production pin.

Создать отдельные `backends/step35/`, build directory и профиль `strata-step37.json`.
Планируемый binary — `strata-step35`, wire architecture — `step35`, UI name —
Step-3.7-Flash. Различать модели семейства по проверенному fingerprint и параметрам,
а не только по общей строке `step35`. Backend и binary уже есть; профиль/API
пока не подключены.

Существующие препятствия в Strata:

- `deepseek-v3` добавлен в `tools/strata_tokenizer.py` и проверен exact-ID oracle.
  Для template нужны Step `fromjson`/нормализация string tool arguments.
- `serve/server.py:configured_template()` не поддерживает `step35`.
  Добавить Step template/parser и routing запросов, не выдавать его за Qwen/GLM.
- GLM runtime/model loader и generated patches содержат архитектурные условия.
  Переиспользовать общий транспорт, но проверить все scheduler hooks и shapes.
- Ошибки upstream tool parsing и сочетания MTP с multimodal state опубликованы
  в [#24181](https://github.com/ggml-org/llama.cpp/issues/24181) и
  [#25129](https://github.com/ggml-org/llama.cpp/issues/25129).
  Это источники regression cases, а не доказанные ошибки выбранной будущей сборки.

## 3. Что переиспользовать

| Компонент Strata | Применение к Step |
|---|---|
| `backends/common/expert_file.hpp`, `expert_pipeline.hpp` | mmap/native reader, bounded pinned ring, H2D stream, cancellation и ownership |
| GLM `expert_key/plan/transport/dispatch` | План матриц по router IDs и ranges; заменить GLM-only валидацию адаптером Step |
| `expert_cache`, `expert_slab`, frequency policy | GPU residency, leases/events, bounded cache, eviction/OOM bypass |
| Общие memory policies и GLM runtime controller | Глобальный RAM/VRAM budget, reserve, уменьшение кэша при росте состояния |
| Protocol, cancellation, OpenAI/Anthropic transport | Отдельные Step INFO/capabilities, template и output parser |
| Fixtures и benchmark runners GLM | Методика byte parity, logits, clocks и paired controls; новые Step references |

Key должен включать model fingerprint, load generation, shard, layer, expert,
projection, quant/shape/range и main/draft namespace. Файлы, leases и CUDA events
должны жить до завершения всех использующих их операций. Разделять общий код
небольшими изменениями с регрессией существующих backend.

Опыт GLM: MTP, shared scratch, compaction, дополнительные readers, affinity и
streaming memcpy не дают универсального выигрыша. Их default для Step выбирается
по новым измерениям. [Последняя методика](../GLM53/GLM53_FLASH_HOST_PIPELINE.md).

## 4. Этапы и критерии готовности

### P0. Инвентаризация, dependency и эталоны

- [x] **P0.1:** первичная проверка четырёх заголовков, диапазонов и типов — PREP-01 в статусе.
- [x] **P0.2:** повторяемый `tools/setup_step35.py --inspect` и JSON-инвентарь.
  Проверять пропуск/повтор/перестановку shards, unsupported quant, overflow,
  пустую первую часть, согласованность metadata и полный список тензоров.
- [x] **P0.3:** таблица соответствия каждому из 754 тензоров loader-кандидата.
  Учесть 64/96 heads, full/SWA RoPE, biases, clamps, Q/K norms, shared expert.
- [x] **P0.4:** отдельная CUDA-сборка и зафиксированный source/patch manifest.
  GPU fixtures Q4_K/Q8_0/Q6_K/F32: обычный/routed matmul, top-8, batch1/4/17,
  padded strides; scalar/CPU reference допустим только как oracle теста.
- [x] **P0.5:** tokenizer/template oracle из выбранной зависимости; synthetic
  Step graph с gate/MoE/SWA, microbatch parity, границы окна 511/512/513 и 1024.
  Сравнить full-prefill и разбивку, RoPE positions, cache clear и rollback.
  STEP-03: 75/75 graph/state и 215/215 template checks. Точный rollback использует
  checkpoint до продолжения; native tail removal после оборота SWA не bit-exact.

**Готово:** проверенный loader, воспроизводимая сборка, exact token IDs и численно
проверенные операции/состояние. Наличие `step35.cpp` само по себе P0 не закрывает.

### P1. Основная модель и синхронная доставка экспертов

- [x] **P1.1:** отдельный native pipe engine с version/INFO, tokenize/decode
  interface Strata, bounds checking, max-context и clean shutdown.
- [x] **P1.2:** non-routed weights, router и все матричные вычисления на GPU;
  experts в mmap, на GPU — только выбранные матрицы. CPU обслуживает файлы,
  токенизацию, sampling и scheduling. Строгий GPU audit без скрытого CPU fallback.
- [x] **P1.3:** простой последовательный selected-copy baseline, bounded staging,
  MTP off. Сначала сохранить независимый oracle, затем reference IDs и F32 logits
  полной модели; повторные запросы, greedy и детерминированный sampling.
- [ ] **P1.4:** реальные RAM/VRAM после load/prefill/decode, согласованный budget95,
  OOM/неподдержанный op дают диагностическую ошибку, а не повреждённое состояние.

**Готово:** минимум два различных текстовых prompts дают проверенные ответы
на полной UD-Q4_K_S; все сравниваемые logits конечны и укладываются в заранее
зафиксированный критерий. Для изменений только транспорта требовать bit-exact.

### P2. Tokenizer, reasoning, tools и API

- [x] **P2.1:** `deepseek-v3` BPE: exact IDs против native oracle для русского,
  английского, CJK, кода, цифр, Unicode, пробелов, control/user-defined tokens,
  encode/decode и `parse_special`. Проверить отсутствие двойного BOS.
- [x] **P2.2:** использовать встроенный Jinja template выбранного GGUF.
  Его SHA-256: `f428623fc81c940c35be3509fbffc086b4b4360d8800e46103e6f34d02891633`.
  Шаблон заканчивает generation prompt на `<|im_start|>assistant\n<think>\n`.
  Проверить history, reasoning replay, tools, несколько tool responses,
  continuation, системные сообщения и `reasoning_effort` low/medium/high.
  STEP-09: production StepTemplate + request adapters; 215 native comparisons,
  реальный pipe tool dialogue. HTTP registration относится к P2.4–P2.6.
- [x] **P2.3:** отдельный потоковый Step parser: `<think>`, `</think>`,
  `<tool_call><function=...><parameter=...>`, schema-directed аргументы,
  несколько вызовов и обычный текст. Не путать с GLM/DeepSeek grammar;
  тестировать все границы chunks, Unicode и неполный последний tag.
  STEP-09: 3415 parser sequences, typed calls/replay на реальной модели PASS.
  Calls буферизуются целиком; full JSON Schema validation и early tool deltas
  не заявляются. HTTP stop/cancel/serialization остаются отдельными gates.
- [x] **P2.4:** stop/EOG по фактическому tokenizer oracle. GGUF EOS 128007,
  а официальный config перечисляет 1/2/128007 — разрешить различие явно;
  не считать PAD 2 terminator автоматически. Stop strings, length и tool_calls
  корректно отображать в finish/stop_reason обоих API.
  STEP-10: native/exported EOG [1,128007], PAD2 excluded; обе API JSON/SSE,
  Unicode, stop strings и length/tool_calls проверены по loopback HTTP.
- [ ] **P2.5:** OpenAI/Anthropic JSON/SSE, web chat, MCP continuation, request
  normalization, INFO/Monitor, startup/unload и profile selection.
  Отсутствующие vision/session/MTP capabilities сообщать честно.
  STEP-10 частично: оба API с полной моделью, local MCP stub, settings/INFO,
  isolated profile и CLI lazy startup проверены. Web asset отдаётся; интерактивный
  browser, внешний MCP и HTTP unload/reload полной модели ещё не закрыты.
- [ ] **P2.6:** STOP/disconnect при prefill, decode и tools; чистый следующий
  запрос, отсутствие зависших workers и доступов к освобождённым mappings.
  STEP-10 частично: настоящий socket disconnect на prefill/reasoning/partial
  tool output и точный следующий запрос; mock queued disconnect/FIFO cleanup.
  Отмена выполняющегося внешнего MCP и длинного HTTP rollover остаются.

**Готово:** полные HTTP диалоги на реальной модели, tools и отмена проверены;
reasoning не теряется в истории и не смешивается с видимым ответом. Уровень
reasoning не выдаётся за гарантированный лимит числа thinking-токенов.

### P3. Кэш и конвейер STRATA

- [x] **P3.1:** range planner для трёх Q4_K матриц каждого из выбранных 8 experts,
  dedup маршрутов prefill/verify, offsets относительно правильного shard.
  STEP-05: main text branch использует native dedup IDs и Step registry с
  generation, tensor type/shape, реальными shard/offset; transport/logits exact.
- [x] **P3.2:** cache hit → D2D/прямое использование; miss → bounded host staging
  → async H2D → consumer event. Router lookahead только после фактических IDs.
  STEP-06: shared async ring, ready/used events, router-derived plan и cache
  residency pins проверены. D2D/scratch bridge остаётся синхронным.
  STEP-12: opt-in reuse allocation вытесненной незакреплённой записи того же
  размера:23 cache checks,117 runtime cases и40 full-model responses PASS.
  Paired decode gain: +5,45% без MTP, +11,64% с MTP2. В pipe/HTTP default ещё
  не включён; результаты — `STEP37_FLASH_CACHE_REUSE.md/json`.
- [ ] **P3.3:** события перед reuse host/device/scratch, 1/2 readers, chunks
  4/8/16 МиБ как измеряемые варианты; early host refill — отдельный opt-in тест.
  STEP-06: все конфигурации прошли fixture, throughput1/2readers измерен на8 МиБ;
  early refill выключен, full-model chunk4/16 tuning остаётся.
- [x] **P3.4:** реальные матрицы из всех shards и ранних/средних/поздних слоёв:
  byte parity, tails, guards, два streams, hit/miss/eviction/cancel/reload.
  Затем полные logits/IDs против P1 при pipeline on/off.
  STEP-06: fixture bytes/guards/cancel/reload и32 full-model responses exact;
  STEP-07: 81 реальная матрица из всех payload shards, 973 byte/guard/cache
  comparisons и reload прошли; 12 context/pressure requests exact.
- [ ] **P3.5:** memory controller, main/draft бюджеты, OOM bypass, безопасная
  реакция на внешнюю VRAM/RAM нагрузку и рост KV; без overcommit retired leases.
  Main cache: global sample/trim, target95 с reserve, RAM admission и cache-allocation
  OOM bypass проверены. STEP-07: реальные 128 МиБ VRAM + 2 ГиБ RAM pressure,
  shrink/recovery и context4096/prompt2591 exact. Startup OOM, предельные режимы
  и async cache leases остаются.
- [ ] **P3.6:** раздельные counters CPU read/memcpy, H2D/D2D, cache hit/miss,
  slot/consumer waits, prefill/decode, paging/SSD. Перекрытие подтвердить CUDA
  timeline; CPU wall-time sums не считать GPU duration.
  STEP-06: read/H2D/D2D payload, slot/consumer waits и CUDA overlap записаны.
  Отдельные paging/SSD metrics и полные GPU D2D durations ещё не измерены.
  STEP-07 выявил большой prefill cache-fill traffic при hit fraction 4,8%;
  STEP-08: optional phase-aware admission реализован, 28 exact reference comparisons,
  paired medium511 prefill −29,8%, полный запрос −27,6%; decode после medium медленнее.
  Default сохранён; broader workloads и GPU/paging counters остаются.

**Готово:** конвейер доставляет точные веса, сохраняет logits и отмену, имеет
проверенные memory bounds и измеренное перекрытие. Ускорение указывать только
при отдельном сравнении с синхронным контролем на том же hardware.

### P4. Сессии и длинный контекст

- [ ] **P4.1:** snapshot/restore включает full KV и SWA ring, positions,
  sequence metadata, RNG/sampling, частично обработанный prefill; при MTP — draft state.
- [ ] **P4.2:** exact-prefix reuse, A → B → A, fork/append, common prefix,
  eviction/reload, model mismatch; холодная генерация служит независимым эталоном.
- [ ] **P4.3:** rollback при переполнении окна и batch verification: 511/512/513,
  несколько оборотов ring; сохранять перезаписываемые позиции до решения verify.
- [ ] **P4.4:** ступени 2K/4K/8K/16K/32K, далее только по memory admission;
  262K — отдельная конечная проверка, не автоматически доступный режим.

**Готово:** восстановленная сессия продолжает тот же поток при тех же условиях,
измерены экономия prefill и память; неподдерживаемые state operations отклоняются.

### P5. Native MTP

- [x] **P5.1:** получить совместимый sidecar, проверить checksum и loader contract,
  vocabulary, embeddings/hidden interface, число реально присутствующих NextN-голов.
  Отсутствие draft не должно мешать MTP-off и P1–P4.
  STEP-11: официальный Q8_0 проверен; 3 головы, vocabulary/hidden width и общий
  embedding совместимы. [Результаты](STEP37_FLASH_MTP_TRIALS.md).
- [ ] **P5.2:** отдельный draft context и Step-specific передача hidden state;
  несколько обученных heads, catch-up, positions и attention state сверить
  с oracle. GLM `mtp.hpp` использовать как образец тестов, не готовую формулу Step.
- [ ] **P5.3:** target verify, accepted prefix, reject-first/middle/all,
  repair и EOS/stop/cancel. Greedy outputs должны совпадать с MTP-off;
  stochastic режим обязан сохранять распределение и воспроизводимость своего RNG.
- [ ] **P5.4:** корректный rollback full/SWA KV обеих ветвей, особенно через 512;
  поддержка session restore после P4. Sharing scratch/embeddings разрешать только
  при доказанном lifetime и численном совпадении; учитывать потерю main cache.
- [ ] **P5.5:** сравнить off/1/2/3, доступные квантования draft, cache budget,
  draft/verify/repair time и accepted tokens per round. Сначала Q8_0 как кандидат,
  более грубые варианты — отдельные измерения после появления совместимых файлов.

**Готово:** точность/rollback на полной модели проверены, стоимость MTP измерена.
Включать MTP по умолчанию только при воспроизводимом выигрыше end-to-end на
нескольких prompts. Высокий acceptance сам по себе недостаточен.
Возможен итог P5: MTP работает, но default остаётся off.

STEP-11: P5.2–P5.5 частично выполнены в отдельном greedy checker. Проверены
off/1/2/3, 60 MTP requests с exact reference IDs и варианты RAM/VRAM размещения.
Финальное сравнение: 9,146 → 10,602 токена/с. Production pipe/HTTP MTP, stochastic
sampling, cancel, sessions и SWA rollback через512 ещё не реализованы/не проверены;
этап P5 целиком не закрыт.

### P6. Установка, замеры и рекомендуемый профиль

- [ ] **P6.1:** non-interactive setup, split admission и отдельные model/tokenizer/
  session paths, backend identity; не менять профили других моделей.
  Локальный сервер по умолчанию 127.0.0.1; LAN требует API key согласно AGENTS.md.
- [ ] **P6.2:** cold-start, warm same prompt, alternating A/B и длинный prefill
  измерять отдельно. Короткие fixture ответы дополнить 256 и более выходных токенов.
  Начать с 4 warmup + 5 timed, расширять при дрейфе контрольных значений.
- [ ] **P6.3:** одинаковые prompts/token counts, sampling и memory budgets;
  контроль до/после, медиана/диапазон, TTFT, prefill/decode, wall time,
  RAM/VRAM maxima, paging и CUDA timeline. Сохранять raw JSON, commands и hashes.
- [ ] **P6.4:** регрессии общих модулей Qwen/DeepSeek/GLM; при их изменении —
  соответствующие полные модели, а не только CPU fixtures. CUDA/Windows —
  первый стенд, Linux/HIP — отдельная матрица с явными неподтверждёнными клетками.
- [ ] **P6.5:** default выбирать по скорости и корректности. Отдельно фиксировать
  экономию памяти; не выдавать её за прирост токенов/с. Обновить setup/docs/UI.

**Готово:** воспроизводимый текстовый профиль этого ПК, опубликованные локальные замеры,
пройденные обязательные регрессии, честная таблица capabilities и ограничений.

### P7. Vision — отдельное расширение

- [ ] **P7.1:** проверить совместимый mmproj, encoder/preprocessor, image token
  positions, image-size/patch policy и дополнительные GPU allocations.
- [ ] **P7.2:** OpenAI/Anthropic image input, web UI, token accounting,
  cancellation; reference logits/ответы для нескольких изображений и текстовой истории.
- [ ] **P7.3:** отдельная матрица vision × MTP × session reuse. Не включать
  комбинацию только потому, что её компоненты прошли отдельные тесты.

**Готово:** полные изображения проходят через реальный encoder и модель;
текстовый baseline и его скорость повторно проверены. P7 не блокирует выпуск
текстового backend, но до него нельзя объявлять поддержку изображений.

## 5. Стартовые настройки для экспериментов

Это предложения для первого запуска, **не измеренный оптимальный профиль**.

| Настройка | Начало | Дальше |
|---|---|---|
| Контекст / sequences | 2048 /1 | 4K → 8K → 16K → 32K по памяти и parity |
| Batch / ubatch / CPU threads | 16 /16 /4 | 32/64/128 и другое число threads по замерам |
| KV | F16 | Квантизация только после отдельной проверки точности |
| FA / TF32 | off /off для диагностического baseline | Сравнить FA on и режимы precision с Step oracle; ограничения GLM не считать законом Step |
| MTP / vision | off /off | P5 /P7 |
| Expert source | mmap, bounded pinned staging | Native reads — отдельный Windows эксперимент |
| Конвейер | сначала sync, затем4 slots × 4 МиБ | Сверить8/16 МиБ и1/2 readers |
| GPU expert cache | начальный ceiling 8 ГиБ | 12/16/20 только если хватает реального reserve |
| Global RAM/VRAM target | 95% по разрешению пользователя | Проверять пики, оставлять резерв ОС/desktop/scratch |

95% — целевой бюджет, а не обещание жёсткого лимита всей ОС. Не заполнять память
бесполезными копиями ради процента; выделять её полезному cache и рабочему набору.
Понижать cache/context/batch при нехватке памяти, не снимать контроль бюджета.

## 6. Карта изменений и порядок

| Планируемый путь | Ответственность |
|---|---|
| `backends/step35/` | Pin/build, loader, native engine, GPU audit, Step state/MTP |
| `backends/common/` | Только проверенные общие transport/cache/memory компоненты |
| `tools/setup_step35.py`, `tools/test_setup_step35.py` | Inventory/admission, отдельный профиль, setup tests |
| `tools/strata_tokenizer.py`, Step tokenizer fixtures | `deepseek-v3`, export, native oracle parity |
| `serve/step35.py`, `serve/server.py`, соответствующие tests | Template, XML tools, reasoning, API/capabilities |
| `tools/check_step35_*`, `tools/benchmark_step35_*` | Byte/logits/state/pipe/HTTP checks и воспроизводимые benchmarks |
| `docs/Step-3.7-Flash/` | Inventory, manifests, raw results, план и текущий статус |

Порядок: **P0 → P1 + P2 → P3 → P4 → P5 → P6**. Текстовый MTP-off профиль можно
проверять до получения draft. Vision P7 после текстового baseline; P5 требует
минимум проверенного SWA rollback из P4. Не ждать завершения всего backend,
чтобы проверить tokenizer, transport и API fixtures.

## 7. Контроль завершения

- [x] Проверить локальные заголовки и подготовить план/статус.
- [ ] P0: reproducible inspector, loader contract, pin, tokenizer/graph oracles.
- [ ] P1: точная полная модель на GPU с bounded selected-copy baseline.
- [ ] P2: правильные reasoning/tools и оба текстовых API.
- [ ] P3: безопасный асинхронный конвейер и GPU-кэш.
- [ ] P4: full/SWA state, sessions, проверенные контексты.
- [ ] P5: совместимый native MTP и измеренный выбор default.
- [ ] P6: профиль, регрессии, raw measurements и документация.
- [ ] P7: vision, если реализуется полный мультимодальный путь.
