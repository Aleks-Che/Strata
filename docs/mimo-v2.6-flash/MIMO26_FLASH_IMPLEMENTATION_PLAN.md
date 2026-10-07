# План внедрения MiMo-V2.6-Flash-RL

Дата исследования: **2026-10-06**, `Asia/Yekaterinburg`.
Исходная ревизия Strata: `1979cb6607c07faec22ab3c195c8c8e1e29d78f9`,
рабочее дерево содержит существующие изменения других backend.
Ход выполнения: [MIMO26_FLASH_IMPLEMENTATION_STATUS.md](MIMO26_FLASH_IMPLEMENTATION_STATUS.md).

Цель — добавить в Strata текстовую модель
`H:\models\mimo-v2.6-flash\MiMo-V2.6-Flash-RL-GSQ-RCO-3.5bit.gguf`:
вычисления модели на GPU, ограниченные RAM/VRAM-кэши экспертов,
асинхронную доставку SSD/RAM → VRAM, диалоги, reasoning, tools и сессии.
Стенд — Windows, 128 ГиБ RAM, RTX 5090 32 ГиБ; целевой глобальный бюджет
до 95% RAM/VRAM. Существующие профили Qwen, DeepSeek, GLM, Step и Hy3 сохранить.

MIMO-01/02 закрыли compatibility и числовые CUDA fixtures P0.
MIMO-03/04 добавили GPU baseline и bounded cache на полном GGUF; MIMO-05 —
асинхронный транспорт и проверку defaults reader1/chunk8. Измеренная прогретая
скорость pipeline7,053 против sync5,103–5,160 ток/с в контрольной серии.
Условия, память и ограничения приведены в статусе. Существенный H2D/compute
overlap, API, сессии и установочный профиль ещё требуют работы.
**В данном GGUF нет MTP, vision и audio weights.** Рабочий текстовый профиль
без них — самостоятельный результат; P5/P7 описывают отдельные расширения.

## 1. Проверенные исходные данные

### 1.1. Локальный GGUF и происхождение

При PREP-01 в каталоге был один GGUF; у основного файла split metadata отсутствует.
В MIMO-06 дополнительно проверены пять MTP/DFlash sidecars, см. P5.

| Параметр | Значение |
|---|---|
| Размер файла | **134 982 426 368 байт = 125,712 ГиБ** |
| GGUF / alignment | v3 / 32 байта |
| Metadata / tensors | 49 записей / 472 тензора |
| `header_end` / `data_start` | 5 979 134 / 5 979 136 |
| Tensor payloads | 134 976 447 232 байта = 125,707 ГиБ |
| Routed experts | 122 909 884 416 байт = **114,469 ГиБ**, 141 тензор |
| Остальные веса | 12 066 562 816 байт = **11,238 ГиБ**, 331 тензор |
| Число хранимых параметров | 308 778 780 864, по произведениям tensor shapes |
| `general.architecture` / name | `mimo2` / `Mimo` |
| `mimo2.nextn_predict_layers` | **0** |
| `mimo2.export_scope` | `text_trunk_without_mtp_or_modality_companions` |

Все имена уникальны; размеры известных типов, выравнивание и границы проверены,
пересечений нет. Последний payload заканчивается точно на EOF.
Это не проверка содержимого весов и не полный loader contract.

SHA-256 первых `header_end` байт, без padding и payloads:
`46e0ff64f95482e89997ba7e661d36a7061a7a5961e033fe5baa6b241e23269a`.
Полный SHA-256 файла не рассчитан; идентичность опубликованным весам не установлена.

Локальная metadata указывает источник `XiaomiMiMo/MiMo-V2.6-Flash-RL`,
source revision `3b38d063180c3e4aed9691fdc735f3d10b266ee4`,
converter revision `58367713a6935c0810103378144008df32e3d5db`.
Это записанные конвертером сведения, не независимая проверка происхождения.
Не смешивать RL с MOPD или прежней MiMo-V2-Flash по сходству имён.

[Издатель этого GSQ-RCO GGUF](https://huggingface.co/pfeifferj/MiMo-V2.6-Flash-RL-GSQ-RCO-GGUF)
описывает независимую community-конверсию текстовой модели без MTP и модальностей;
указывает проверку с upstream llama.cpp `58367713`. Это полезный reference pin,
но его работа на нашем ПК ещё не воспроизведена. Оценки качества издателя
не являются локальными тестами Strata.

### 1.2. Архитектура основной модели

| Параметр локального GGUF | Значение |
|---|---|
| Блоки | 48, индексы 0..47 |
| FFN | Dense в блоке0; MoE в 1..47 |
| Hidden / dense FFN / expert FFN | 4096 / 16384 / 2048 |
| Routed experts | 256 на MoE-блок, top-8 |
| Shared experts | Тензоров нет; исследованный граф их не использует |
| Router | Sigmoid, correction bias, одна группа / одна выбранная группа |
| Attention | 9 full и 39 sliding-window (SWA) |
| Full blocks | **0, 5, 11, 17, 23, 29, 35, 41, 47** |
| SWA window | 128 |
| Query heads | 64 в каждом слое |
| KV heads | 4 в full; 8 в SWA |
| Q/K head dimension / V dimension | **192 / 128** |
| QKV | Fused `attn_qkv.weight`; full `[4096,13568]`, SWA `[4096,14848]` |
| Attention output | `[8192,4096]` |
| RoPE dimensions / base full / base SWA | 64 / 10 000 000 / 10 000 |
| RMS epsilon / attention value scale | ≈1e-6 / ≈0,707 |
| Attention sinks | 39 F32 `[64]`, только SWA |
| Context metadata | 1 048 576; локальная поддержка этого контекста не проверена |

Список full-слоёв брать из массива metadata: начало схемы нельзя заменять
простым правилом «каждый шестой от нуля». K/V длины и KV heads также
обрабатываются по слоям, без общего stride от Step или Hy3.

В прочитанном `mimo2.cpp`: RMSNorm перед attention, fused Q/K/V views,
частичный NeoX RoPE, attention scale `1/sqrt(192)`, sinks в softmax,
масштабирование attention output, residual и dense/MoE SwiGLU.
Q/K norm weights отсутствуют. Router нормализует выбранные sigmoid weights;
bias используется для выбора экспертов. `expert_weights_scale` в GGUF отсутствует:
проверить default loader и не переносить multiplier2,826/3 из Hy3/Step.

Закреплённый [config Xiaomi для RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL/blob/3b38d063180c3e4aed9691fdc735f3d10b266ee4/config.json)
согласуется с hybrid pattern, разными K/V dimensions, отсутствием shared experts
и тремя NextN-слоями исходного checkpoint. Последнее **не относится к этому
экспорту**, где NextN=0. В MIMO-01 сверены точный source config и файл upstream
loader: [46 проверок, 0 расхождений](MIMO26_FLASH_REFERENCE_VALIDATION.json).
Conversion pipeline целиком не проверен, payload не сравнивался с источником.

### 1.3. Реальные кванты и доставка экспертов

| Тип | Число тензоров | Где используется |
|---|---:|---|
| Q2_K | 50 | Routed gate/up/down |
| Q3_K | 74 | Routed gate/up/down |
| MXFP4 | 17 | Часть routed down |
| BF16 | 101 | Attention, embedding/output, dense FFN |
| F32 | 230 | Router, bias, нормы, attention sinks |

GSQ-RCO — способ получения весов, а не отдельный GGML type.
В этом файле нет Q2_0, Q8_0, IQ-типов или FP8 tensors. Описание MXFP4/FP8
в `source_quantization` относится к исходному checkpoint, не ко всем
локальным тензорам. Имена форматов проверять по каждой записи directory.

Одна экспертная матрица содержит `4096 × 2048` элементов:
Q2_K — **2 752 512 байт / 2,625 МиБ**, Q3_K — **3 604 480 / 3,4375 МиБ**,
MXFP4 — **4 456 448 / 4,25 МиБ**. Gate/up shape `[4096,2048,256]`,
down `[2048,4096,256]`; память и row strides отличаются по типам.

Если все выбранные эксперты промахнулись мимо VRAM-кэша, топ-8 во всех
47 MoE-слоях требуют **3 840 933 888 байт = 3,577 ГиБ** экспертных весов
за decode-проход. Это расчёт H2D при однократной доставке каждой матрицы,
без служебных данных, не измерение пропускной способности или токенов/с.

### 1.4. RAM, VRAM и KV

Windows сообщает 134 813 700 096 байт физической RAM (**125,555 ГиБ**),
`nvidia-smi` — RTX 5090, **32 607 МиБ** total, driver581.80.
95%-бюджеты: **119,277 ГиБ RAM / 30 976,65 МиБ VRAM** глобально,
включая другие процессы и графический рабочий стол.

Файл немного больше всей доступной Windows RAM и заметно больше 95%-бюджета.
Предварительная полная heap-копия или закрепление всего mmap не подходят.
План: non-routed веса ≈11,238 ГиБ на GPU; routed matrices распределять
между GPU-кэшем и ограниченным RAM working set, холодные читать из файла.
Не держать лишние heap/pinned копии GPU-resident матриц; исходные file-backed
страницы должны оставаться вытесняемыми. Размер mmap не равен resident RAM.

KV оценивать раздельно для full и SWA. Для идеального compact SWA без padding:
`bytes = (9×4×context + 39×8×min(context,128)) × (192+128) × sizeof(KV)`.

| Context | Compact SWA, F32 / F16, ГиБ | Все слои хранят полный context, F32 / F16, ГиБ |
|---:|---:|---:|
| 2048 | 0,1355 / 0,0677 | 0,8496 / 0,4248 |
| 4096 | 0,2234 / 0,1117 | 1,6992 / 0,8496 |
| 8192 | 0,3992 / 0,1996 | 3,3984 / 1,6992 |
| 1 048 576 | 45,0476 / 22,5238 | 435 / 217,5 |

Compact столбец — нижняя оценка payload, **не измерение allocator**.
Реальные SWA buffers могут включать ubatch, padding, запас под rollback,
несколько sequences и checkpoints. Узнать фактическое размещение в P0/P1.
1M context не помещается в заданный VRAM-бюджет с базовыми весами даже при
этой F16-оценке. Начинать с 2K/4K, длинный контекст рассматривать отдельно.

## 2. Основа реализации

`mimo2` уже зарегистрирован в `third_party/llama.cpp`. Его loader совпадает
байт-в-байт с локальным candidate Unsloth, используемым GLM/Step/Hy3:

- source SHA `86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9`;
- архив `build-local/llama-glm-86ebfef2.tar.gz`;
- archive SHA-256 `f8e524b635b726bae74fd8f84bb9249e5b09384c63207f6707c5a3f921acad99`;
- `src/models/mimo2.cpp`, 17 030 байт, SHA-256
  `2ec5caa11fb9ab7604b798a17cd80244427596581c06cdb033762b6d56f50823`.

Хэши архива, файла внутри архива и двух локальных копий проверены.
Совпадение одного loader не означает совпадение всех dependency исходников.
CUDA source содержит MXFP4 MMVQ/MMQ и Flash Attention вариант K192/V128.
В MIMO-02 выполнены synthetic numerical checks на RTX5090, включая FA off/on;
полный GGUF ещё не запускался. Native Blackwell MXFP4 MMQ с FP4-активациями
не прошёл заданный допуск; выбран точный GPU baseline, см. README backend.

Первый кандидат — этот локальный архив в **новом изолированном build**.
P0 должен сопоставить его с revision `58367713a6935c0810103378144008df32e3d5db`,
заявленной издателем, включая loader, converter, router, attention и KV.
В MIMO-01 получен `src/models/mimo2.cpp` этой ревизии: он совпадает с кандидатом
байт-в-байт. Эквивалентность converter и всего дерева двух pins не установлена.
Для CUDA/runtime продолжить выбор dependency по числовым проверкам,
зафиксировать source/archive/patch hashes; не обновлять общую dependency всех моделей.

Созданы [CPU/CUDA checks](../../backends/mimo2/README.md),
build — `build-local/mimo2-oracles` и `build-local/mimo2-cuda`.
В MIMO-03 создан отдельный pipe engine `strata-mimo2`; установочный профиль
и HTTP adapter пока запланированы.
Патчи TF32, routed strides и CUDA Graphs переносить только после проверки
применимости к MiMo. Известные проблемы графов Hy3 не доказывают ни наличие,
ни отсутствие такой же проблемы у MiMo.

## 3. Что переиспользовать

| Существующая часть | Использование |
|---|---|
| [GGUF reader](../../tools/gguf_reader.py) | Header/directory без чтения payload; отдельный MiMo loader contract |
| [Общий transport](../../backends/common/README.md) | Checked ranges, readers, pinned ring, CUDA events и статистика |
| [Step backend](../../backends/step35/) | Примеры hybrid full/SWA fixtures, snapshot/rollback, GPU matrix cache |
| [GLM backend](../../backends/glm5next/) | Mixed-quants dispatch, ranges, memory pressure, pipe/API и замеры |
| [Hy3 backend](../../backends/hy3/) | Строгий contract, oracles, GPU audit, pipeline, bounded RAM cache подход |
| [Step API adapter](../../serve/step35.py) | Структура API-преобразования и потокового parser |
| [Устройство Strata](../DETAILS.md) | Размещение экспертов и перекрытие копий с вычислениями |

Переиспользовать инфраструктуру, сохраняя MiMo-specific geometry и state.
В первом профиле CPU занят tokenizer, sampler, I/O и scheduling;
матричные вычисления модели, включая router и экспертов, выполняются на GPU.
Наличие поддержки `mimo2` в llama.cpp не означает готовый Strata backend.

## 4. Этапы внедрения и критерии перехода

### P0. Формат, loader contract и эталоны

- **P0.1 — DONE (MIMO-01):** постоянный inspector/JSON: metadata, все names/shapes/types/offsets,
  группы памяти, full/SWA pattern, tokenizer/template hashes, source pins.
  [Отчёт инспекции](MIMO26_FLASH_INSPECTION.json), payload не читается.
- **P0.2 — DONE (MIMO-01):** строгий contract для text trunk: 48 блоков, dense0/MoE1..47,
  fused QKV двух форм, 39 sinks, отсутствие shared и NextN tensors,
  `exp_probs_b.bias`, KV heads4/8, K192/V128, RoPE64 и каждый quant type.
  Проверить обязательные веса независимо от `TENSOR_NOT_REQUIRED` loader.
  Negative fixtures: неизвестные metadata, missing/extra tensor, несовпадение
  pattern/shapes, quant row geometry, duplicate/overlap/truncated/64-bit ranges.
- **P0.3 — DONE, CPU (MIMO-01):** изолированная сборка, manifest source/compiler/CUDA/flags/patches;
  vocabulary-only, template и no-allocation loader oracles. Регистрация должна
  покрывать все 472 tensors, не добавлять MTP и shared-expert placeholders.
  [Сборка](MIMO26_FLASH_BUILD_VALIDATION.json),
  [регистрация 472 тензоров при load_mtp=false/true](MIMO26_FLASH_LOADER_VALIDATION.json).
- **P0.4 — DONE (MIMO-02):** числовые CUDA fixtures для Q2_K/Q3_K/MXFP4 и BF16/F32 matmuls:
  dense, routed batch1/multitoken, sigmoid+bias/top-8/normalization,
  разные strides gate/up/down и repeated expert IDs. Проверить TF32 policy.
  [96/96 PASS](MIMO26_FLASH_CUDA_KERNELS_VALIDATION.json); synthetic K512/M64,
  experts256/top-8. Закреплены TF32/F32, routed strides и MXFP4 precision policies.
- **P0.5 — DONE (MIMO-02):** tiny full+SWA graph: fused QKV offsets, partial NeoX RoPE,
  scale `1/sqrt(192)`, value scale0,707, sinks, masks, KV4/8.
  Сравнить с CPU/F32 oracle при FA off/on, если конкретный путь поддерживается.
  Обязательны позиции127/128/129 и255/256/257, prefill/decode и rollback.
  [132/132 PASS](MIMO26_FLASH_CUDA_GRAPH_VALIDATION.json): 3 блока, hidden256,
  experts16/top-8; реальные heads64, KV4/8, K192/V128. F32 и mixed-quants,
  CPU/F32 dequantized oracle, FA off/on, SWA eviction и state restore.
  CUDA Graphs/reuse отключены. Это проверка synthetic graph, не полного engine.

**Готово:** воспроизводимые contract/oracles и числовые fixtures.
Наличие parser и llama architecture entry само по себе P0 не закрывает.

### P1. Native GPU baseline без speculation

- **P1.1:** native pipe engine с validated file ranges, синхронным selected-copy
  gate/up/down, bounded staging и точными INFO capabilities; без MTP,
  adaptive cache, pipeline и prefix reuse в первом baseline.
- **P1.2:** execution audit: все model matmuls на GPU, без скрытой CPU-expert
  обработки. Non-routed BF16 weights остаются в исходном формате.
- **P1.3:** на полном GGUF сравнить logits и greedy token IDs с графом
  выбранного llama pin на тех же quants/input IDs. Медленный offload oracle
  допустим: полностью разместить 125 ГиБ на одной GPU невозможно.
  Корпус: русский, английский, китайский, код, числа, короткие/длинные prompts;
  обязательно перейти SWA128, недостаточно короткого ответа ниже окна.
- **P1.4:** load/prefill/TTFT/decode, H2D bytes, process/global memory;
  cancel→следующий запрос, ошибка→recovery, unload/reload. Измерить фактический
  KV allocation и резерв под ubatch до увеличения expert cache.

**Готово:** корректный full-model baseline и проверяемый отчёт на этом ПК.

**MIMO-03: выполнено для context512, batch8, F32 KV, FA on и greedy.**
Два синхронных copy режима совпали побитно на6 случаях корпуса (включая249-token
prompt через SWA128) и3 повторах по32 токена. Проверены cancel/error→fresh,
GPU-only audit, EOF/ranges и unload. Результаты и ограничения измерений —
[статус MIMO-03](MIMO26_FLASH_IMPLEMENTATION_STATUS.md).
API, большой контекст, cache/pipeline и tuning не входят в этот baseline.
Не объявлять tokens/s по скоростям отдельного kernel или другого checkpoint.

### P2. Tokenizer, reasoning, tools и API

- **P2.1:** GPT-2 BPE с pre-tokenizer `qwen2`: 152576 tokens / 151387 merges,
  EOS151645 (`<|im_end|>`), PAD151643 (`<|endoftext|>`), add_bos=false.
  Получить полный EOG-набор через oracle; не копировать stop IDs из другой модели
  и не считать наличие audio tokens поддержкой audio. Проверить UTF-8,
  special-token escaping, exact IDs и обратную сборку token pieces.
  **MIMO-01: IDs/bytes PASS 614/614.** Изолирована native-нормализация `</s>`;
  BOS11 не вставляется. Полный EOG записан в отчёте. В MIMO-03 runtime
  останавливается только по EOS151645; BOS/PAD/FIM и `</s>` не останавливают ответ.
- **P2.2:** сохранить и воспроизвести **встроенный** Jinja template.
  `enable_thinking=false` даёт prefix `<think></think>`; проверить default
  thinking, `reasoning_content` в history, system/tools и tool messages.
  Template SHA-256: `11ea52e156de38a458e6b7720ad45915d65b97d4ec979a09f55e3c9bd1b4d059`.
  **MIMO-01: raw Jinja/template tokens PASS 85/85.** HTTP-нормализация arguments,
  потоковый parser и полный tool round-trip ещё не реализованы.
- **P2.3:** потоковый parser для `<think>`, `<tool_call><function=NAME>`,
  `<parameter=KEY>VALUE</parameter>` и закрывающих тегов. Строковые arguments
  в template выводятся иначе, чем mapping: задать явную нормализацию API.
  Проверить типы по schema, строки с JSON/XML, Unicode, несколько calls,
  куски тега/UTF-8 на границах chunks, незакрытые блоки, EOS и stop.
- **P2.4:** отдельный adapter OpenAI/Anthropic JSON/SSE: reasoning/content,
  tool call IDs/results, usage/finish_reason, отмена и восстановление.
  Проверить полный цикл tools→result→answer, историю и отсутствие утечки тегов.
- **P2.5:** text-only capability в профиле/UI; image/audio/video запросы
  отклонять до renderer. Template умеет печатать media placeholders, но
  соответствующих весов нет. Не возвращать фиктивно успешный multimodal ответ.

**Готово:** tokenizer/template fixtures совпадают с oracle; оба API и веб-чат
проверены на текстовых диалогах, reasoning и tool continuation.

### P3. Кэши и конвейер STRATA

**MIMO-05: P3.1 и транспорт P3.2 проверены; P3 целиком остаётся IN PROGRESS.**
Добавлены bounded LRU, mixed-quants byte audit, reuse allocations, file/mmap,
decode-only admission, ограничение working set и async reader/H2D ring с events.
Проверены23 cache ownership cases и212 runtime cases, полный corpus/повторы
с bit-exact transport parity. История замеров и defaults — в
[статусе](MIMO26_FLASH_IMPLEMENTATION_STATUS.md). CUDA trace показывает небольшой
overlap; существенное перекрытие H2D/compute, pressure и широкий benchmark TODO.

Разбор от2026-10-07: [перенос оптимизаций GLM/Step/DeepSeek](MIMO26_FLASH_OPTIMIZATION_TRANSFER_REVIEW.md).
MIMO-07 реализовал общие блоки16 МиБ и physical accounting. Собственный ABBA
дал6,434→6,952 ток/с (+8,1%), payload9,855→12,370 ГиБ; default16 проверен
на полном корпусе с bit-exact logits. [Условия и ограничения](MIMO26_FLASH_SLAB_CACHE.md).
MIMO-08 добавил tensor delivery:6,591→7,299 ток/с в ABBA (+10,7%), либо+5,3%
относительно последнего прогретого контроля. [Условия](MIMO26_FLASH_TENSOR_BATCH.md).
MIMO-09 добавил частотный допуск decay65536:6,923→7,692 ток/с в ABBA (+11,1%),
к быстрейшему контролю+8,2%. H2D прогретого decode−11,8%; первые запросы тем
копируют больше весов. [Условия и ограничения](MIMO26_FLASH_FREQUENCY_CACHE.md).
MIMO-10 проверил grouping cache fills:−19,17% waits, но7,861→7,826 ток/с,
плюс незакрытая mixed synthetic диагностика; default off. В обычную сборку
добавлен literal guard, убирающий временные строки проверки logits.
Его отдельный ABBA с fill0:7,721→7,933 ток/с (+2,74%), logits/IDs exact.
[Измерения и проверки](MIMO26_FLASH_FILL_AND_GUARDS.md).
MIMO-11 исправил распределение ключей истории на MSVC: CPU microbenchmark
примерно44× быстрее. Два full-model ABBA прошли exact parity; контрольные
скорости заметно колеблются, устойчивый end-to-end прирост не подтверждён.
Хеш включён, cache defaults сохранены. [Результаты](MIMO26_FLASH_HASH_CACHE.md).
MIMO-12 заменил resident/pin lookup: CPU microbenchmark63,330→32,505 мс,
но decode8,390 против прогретого контроля8,401 ток/с. Индекс включён как
снижение CPU-затрат. [Проверки и ограничения](MIMO26_FLASH_CACHE_INDEX.md).
MIMO-13 объединил guard tails по route:10,336 ток/с, +24,02% к быстрейшему
контролю8,334; H2D operations−49,03%, corpus и224 CUDA cases PASS.
Guard batching on; early refill off после7,992→7,912 ток/с.
[Результаты и defaults](MIMO26_FLASH_HOST_PIPELINE.md).
Ближайшие отдельные опыты без MTP: D2D/compute timeline и промежуточные копии.
Allocation reuse и prefill hits уже реализованы. Новый default требует своего A/B;
проценты разных этапов нельзя складывать.

- **P3.1:** matrix key с model generation/layer/tensor/expert/type, byte-budget
  по реальным allocations. Mixed-format cache, pin активных entries,
  reuse только после завершения consumer. MIMO-07 добавил packed blocks,
  учёт padding/свободных slots и pressure/OOM unit checks. Проверить смену модели
  и длительное внешнее давление отдельно.
- **P3.2:** bounded reader→pinned staging→H2D stream→GPU compute.
  H2D event защищает чтение новых данных, consumer event — перезапись device slot.
  Mapping, cache entries и destinations живут до завершения всех пользователей.
- **P3.3:** batch ranges, allocation reuse, early host refill и host-copy варианты
  включать отдельно. Byte/logits parity с P1 на Q2_K/Q3_K/MXFP4,
  repeated experts, prefill и отмене в середине pipeline.
  MIMO-08: доставка одного scheduler input до общей fence проверена и включена;
  LRU order и синхронная публикация fills сохранены. Async fills остаются TODO.
  MIMO-09: frequency admission с периодом65536 наблюдений включён по результатам
  отдельного ABBA; pins и fenced fill сохранены, prefill off не обучает историю.
  MIMO-10: grouped reservations/fills реализованы opt-in, pending entries скрыты
  до fence, enqueue-error/recovery проверены. Скорость не выросла и один mixed
  synthetic повтор FAIL; не считать готовым recommended profile или compute overlap.
- **P3.4:** управляемый RAM working set и GPU-resident weights без обязательной
  постоянной RAM-копии. Глобальный memory admission включает other processes,
  model weights, KV, scratch, ring, allocator reserve и будущий draft.
  При давлении останавливать admission/уменьшать cache; если базовый набор
  не помещается, завершать запрос понятной ошибкой с корректным drain.
- **P3.5:** подобрать cache8/12/16 ГиБ с runtime clamp, readers1/2,
  chunks4/8/16 МиБ, prefill admission. Измерять disk reads/page faults,
  RAM/GPU hit по байтам, H2D/compute и ожидания. CPU wait sums не считать
  доказательством overlap: нужна CUDA timeline.
- **P3.6:** сравнить fresh/warm requests, смену темы и повторный prompt.
  CUDA Graphs — отдельный шаг после проверки стабильных адресов, событий,
  dynamic expert IDs и cancel/unload; performance не заменяет correctness.

**Готово:** bounded memory, parity с P1, доказанное перекрытие и повторяемый
выигрыш. Конвейер копирует выбранных экспертов; следующий router нельзя
считать известным до вычисления следующего слоя без отдельного предиктора.

### P4. Сессии и длинный контекст

- Snapshot/restore full KV и SWA ring с positions/head pointers; prefix reuse,
  truncate/shift, abort/retry и изоляция session IDs. Сравнить с fresh prefill
  до/после128 и на многократных оборотах окна, не только на первых127 токенах.
- Compact SWA может перезаписать данные, необходимые rollback: определить
  checkpoint strategy до реализации speculative decode. `seq_rm` сам по себе
  не гарантирует восстановление перезаписанного кольца.
- Профили2K/4K/8K, затем выше при бюджете. F32 KV — oracle baseline,
  F16/Q8 KV — отдельные проверки качества, FA compatibility и фактической памяти.
  Сохранённое состояние несовместимого template/model/KV configuration отвергать.

**Готово:** заявленные контексты и сессии корректны при restore/shift/cancel;
1M metadata не выдаётся за проверенную возможность профиля.

### P5. MTP / DFlash — дополнительный этап с отдельными весами

**Текущий GGUF не позволяет включить native MTP:** `nextn_predict_layers=0`,
нет NextN tensors и блоков≥48. Наличие `graph_mtp` в llama.cpp этого не меняет.
Первый рабочий профиль и P6 выпускаются с `mtp=false`.

- **P5.1:** найти или отдельно экспортировать MTP из совместимой **RL** source
  revision; проверить checksum, tokenizer, hidden/output norms, layer mapping,
  shared embedding/head, draft attention/FFN и quantization. Публичный config
  описывает три NextN. В MIMO-06 найдены3 native MTP и2 DFlash MOPD sidecars;
  header/ranges/tokenizer/full SHA проверены. DFlash weights в официальных RL/MOPD
  имеют одинаковые LFS hashes, но embedding/head локальных sidecars отличаются
  от target RL. Проверять acceptance и target parity, а не только имя архитектуры.
  Отчёты: `MIMO26_FLASH_DRAFT_INSPECTION.json`, `MIMO26_FLASH_DRAFT_RESEARCH.json`.
- **P5.2:** отдельное размещение активных draft heads, без второй копии trunk;
  граф и hidden-state boundary сверить с oracle. По локальному коду MiMo
  передаёт hidden до final output norm — не переносить post-norm контракт Hy3.
  Проверить наличие dense FFN, которое ожидает локальный `graph_mtp`.
- **P5.3:** main/draft state, batched verify, dedup expert ranges, acceptance,
  rejection repair и catch-up. Все rollback positions на SWA128 и после
  нескольких оборотов кольца; EOS/stop, отмена и session restore.
- **P5.4:** greedy depth1/2/3; stochastic sampling только после проверки
  корректного rejection sampling. A/B с одинаковым target cache и отдельно
  лучший MTP-off с возвращённой draft-памятью против лучшего on при тех же95%.
  Считать принятые выходные tokens/s, TTFT/request time, draft/verify/repair
  и H2D bytes на output token, а не только acceptance rate.

**Готово для MTP:** совместимые веса, correctness и локально измеренный выигрыш.
До этого сохраняется текстовый профиль без MTP. Offline probe из MIMO-06 —
инструмент сравнения, не завершение P5 и не serving-интеграция. DFlash требует
собственного learned MASK, target features1/12/24/36/48, partial RoPE64, sinks
и value scale0.612. Сначала проверить пакетное вычисление target относительно
последовательного baseline; не объяснять любое расхождение только квантом draft.

На локальном корпусе MIMO-06 без draft получено6,13 ток/с, лучший проверенный
draft — MTP Q4 head0/cutoff0.7 —5,55 ток/с. Выигрыш не подтверждён; общий
quality gate не пройден из-за расхождения batched target. Условия и все варианты:
[сравнение MTP/DFlash](MIMO26_FLASH_SPECULATIVE_COMPARISON.md).

### P6. Установка, замеры и рекомендуемые настройки

- Изолированные setup/profile/help/INFO с architecture `mimo2`, text-only,
  mtp=false, оценкой памяти до загрузки. Сервер изначально на `127.0.0.1`;
  для внешнего bind обязателен API key.
- Зафиксировать model/header/template hash, dependency/patch SHA, driver,
  CUDA/compiler, context/KV, FA/graphs, batch/ubatch, cache/readers/chunks,
  sampler/seed и точные prompt IDs.
- Минимум три замера после прогрева, порядок A/B чередовать. Отдельно новый
  процесс, warmed weights, новый prompt и prefix-cache hit. Публиковать median
  и диапазон prefill/decode, TTFT, request latency, generated counts, память
  и disk/H2D traffic; указать учёт reasoning/EOS.
- Рассмотреть BF16→Q8_0 non-routed как **отдельный производный артефакт**
  только после baseline и оценки качества: 11,238 ГиБ базовых весов ограничивают
  expert cache. Не менять пользовательский GGUF и не смешивать такой A/B
  с оптимизацией выполнения исходной модели.
- Регрессии затронутых setup/API/common transport для Qwen/DeepSeek/GLM/Step/Hy3;
  тяжёлые GPU tests выполнять последовательно при свободном бюджете.

**Готово:** воспроизводимый текстовый профиль и локальный отчёт; defaults
обоснованы замерами, MTP и модальности честно отключены до отдельной приёмки.

### P7. Vision/audio/video — отдельное расширение

В этом каталоге companion weights не найдены. Нужны совместимые encoders,
projectors/processors, контракт token insertion/positions, отдельные бюджеты
и проверки каждой модальности. Одних media tokens в vocabulary недостаточно.
P7 не входит в критерий готовности текущего текстового GGUF.

## 5. Исходные настройки эксперимента

CLI MiMo уже существует: context512/batch8, F32 KV, FA on, greedy и GPU cache.
Ниже сохранены исходные параметры экспериментов; context2K/4K ещё не принят.
Bounded pipeline проверен, существенный compute overlap ещё предстоит получить.
Текущие defaults и их проверки описаны в статусе и backend README.

| Параметр | Начало / варианты |
|---|---|
| MTP / multimodal | off / disabled для данного файла |
| Context | 2048 baseline, 4096 рабочий кандидат, затем8192 |
| KV / FA / CUDA Graphs | F32 / off / off для эталона; варианты включать после parity |
| Sampler | temperature0 для correctness; затем1,0/top_p0,95 по GGUF |
| GPU expert cache | 0 для reference;8/12 ГиБ и увеличенный cap с runtime clamp проверены в MIMO-04 |
| RAM cache | Динамический cap после вычета ОС, других процессов и transient buffers |
| Pipeline | Default reader1/chunk8 по замерам MIMO-05; readers1/2 проверены, chunks4/16 пока только fixtures |
| Prefix/session reuse | off для первого A/B |
| Global memory target | До95% RAM/VRAM с дополнительным резервом на пики |

95% — верхняя граница полезного размещения, не требование заполнить память.
11,238 ГиБ базовых весов +16 ГиБ expert cache +1,699 ГиБ F32 full-context KV
при4K уже дают28,937 ГиБ без graph/scratch/desktop; cap16 не гарантированно
поместится. Окончательный cap определяется после измерения выбранного KV режима.
Перед запуском проверить external load; чужие процессы не завершать.

## 6. Карта файлов и порядок работ

Inspector, contract, CPU oracles, CUDA kernel/graph checks и их tests уже созданы.
Engine уже создан; API/setup в таблице пока запланированы.

| Область | Файлы |
|---|---|
| Inspector/contract | `tools/inspect_mimo2_gguf.py`, `tools/mimo2_loader_contract.py`, CPU tests |
| Native engine/oracles | `backends/mimo2/CMakeLists.txt`, loader/tokenizer/template/graph checks, `main.cpp` |
| API/template | `serve/mimo2.py`, `serve/fixtures/mimo26_chat_template.jinja`, request/parser/HTTP tests |
| Setup/profile | `tools/setup_mimo2.py`, `tools/prepare_mimo2_profile.py` |
| Model validation | `tools/check_mimo2_engine.py`, `check_mimo2_http.py`, позднее `check_mimo2_mtp.py` |
| Отчёты | `docs/mimo-v2.6-flash/MIMO26_FLASH_INSPECTION.json`, validation/benchmark reports |

Порядок: **P0 → P1 → P2/P3 → P4 → P6**. P2.1 можно делать в P0;
MIMO-06 добавляет отдельное сравнение пяти MTP/DFlash sidecars, MIMO-07 — плотный
cache и измеренный default16; MIMO-08 — tensor delivery с проверенными lifetimes;
MIMO-09 — частотный кеш с измеренным decay65536.
MIMO-10 измерил fills (off) и устранил временные строки CPU guard (on).
MIMO-11 исправил хеш истории (on), подтвердил точную parity и сохранил оба
ABBA с оговоркой о влиянии фоновой нагрузки на end-to-end скорость.
MIMO-12 ускорил resident cache metadata (on), сохранил exact policy/logits;
прогретый decode практически прежний. MIMO-13 включил пакетную передачу guard tails
после full-model ABBA и проверки defaults; early host refill оставлен off.
MIMO-14 добавил прогретый H2D/D2D/compute trace и два варианта CUDA13 batched D2D.
Ускорение не подтверждено: поздний9,421→9,342; ранний с фиксированным cache11 ГиБ
9,619→9,418 ток/с. CUDA API batch остаётся opt-in; полные [результаты](MIMO26_FLASH_D2D_BATCH.md).
MIMO-15 добавил собственное GPU scatter-copy ядро для resident cache → scratch.
Основной ABBA9,567→10,259 ток/с (+7,23%); IDs/logits exact. Отдельный русский
ABBA при cache8 ГиБ9,131→8,035 (−12,01%); поэтому default0 сохранён, mode2 opt-in.
Условия, влияние live clamp и отдельная проверка русского
запроса — в [отчёте](MIMO26_FLASH_SCATTER_COPY.md).
MIMO-16 повторил Q4 MTP на текущем transport: без draft10,520 ток/с,
Q4 9,015, Q4 +scatter9,076. Short corpus IDs совпали, но oracle снова расходится
на18-м токене; [результаты](MIMO26_FLASH_MTP_Q4_RETEST.md). Serving остаётся off.
MIMO-17 реализовал opt-in per-column target BF16 head без полной F32-копии:
Q4 9,880→10,929 ток/с, контроль без MTP10,615; cache+2,085 ГиБ,
H2D−14,48%. Head fixture bit-exact, oracle batch2 по-прежнему расходится.
[Результаты и ограничения](MIMO26_FLASH_TARGET_HEAD.md); default0, serving off.
Следующая задача **MIMO-18**: target batch2 parity по слоям, BF16 dense/routed MMVQ;
для скорости без draft —
direct resident weights, групповые H2D, chunks4/16 и границы синхронизации.
Для chunk A/B сохранять `--fixed-pack-guards 1`; D2D задавать одинаково в обоих вариантах.
Сохраняются задачи
локализации mixed/F32 synthetic mismatch и проверки запросов при смене темы.
Для MTP отдельно проверить batched target parity. H2D/compute overlap и P2 API
ещё открыты. P7 требует modality companions.

## 7. Контроль завершения

- [x] Локальный файл принят строгим contract; CPU dependency и отдельные CUDA/runtime patches закреплены.
- [x] GPU baseline совпадает с native selected-copy reference, включая SWA128 (context512/greedy).
- [ ] Tokenizer/template, reasoning, tools, оба HTTP API и веб-чат проверены.
- [x] Mixed-quants cache/pipeline: bytes/logits parity, bounded memory, cancel/error/drain и прирост на повторном prompt проверены (context512).
- [ ] Существенный compute overlap, длительный pressure stress и производительность на разных темах подтверждены.
- [ ] Restore/shift/cancel/reload и заявленные контексты прошли проверку.
- [ ] Есть отдельный профиль, замеры defaults и регрессии других моделей.
- [ ] Дополнительно: MTP/DFlash после quality parity, полезного speedup и serving lifecycle P5.
- [ ] Дополнительно: мультимодальность только после P7.
