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

Это план, а не отчёт о работающей поддержке. Прочитаны локальный GGUF header,
tensor directory и исходники; генерация MiMo не запускалась.
Размеры ниже рассчитаны по тензорам, скорость и пиковая память не измерены.
**В данном GGUF нет MTP, vision и audio weights.** Рабочий текстовый профиль
без них — самостоятельный результат; P5/P7 описывают отдельные расширения.

## 1. Проверенные исходные данные

### 1.1. Локальный GGUF и происхождение

В каталоге найден один GGUF; split metadata отсутствует.

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

Публичный [config Xiaomi для RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL/blob/main/config.json)
согласуется с hybrid pattern, разными K/V dimensions, отсутствием shared experts
и тремя NextN-слоями исходного checkpoint. Последнее **не относится к этому
экспорту**, где NextN=0. Прочитан `main`; сверку точного source revision
и conversion mapping необходимо завершить в P0.

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
CUDA source содержит MXFP4 MMVQ/MMQ и Flash Attention вариант K192/V128;
наличие кода не доказывает правильное выполнение на нашем GGUF.

Первый кандидат — этот локальный архив в **новом изолированном build**.
P0 должен сопоставить его с revision `58367713a6935c0810103378144008df32e3d5db`,
заявленной издателем, включая loader, converter, router, attention и KV.
Точный upstream source этой ревизии в ходе подготовки не получен;
эквивалентность двух pins пока не установлена. Выбрать dependency по проверкам,
зафиксировать source/archive/patch hashes; не обновлять общую dependency всех моделей.

Новый backend — `backends/mimo2`, build — `build-local/mimo2-cuda`,
отдельный профиль/engine `strata-mimo2`. Это предлагаемые имена, файлов ещё нет.
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

- **P0.1:** постоянный inspector/JSON: metadata, все names/shapes/types/offsets,
  группы памяти, full/SWA pattern, tokenizer/template hashes, source pins.
  Подготовительное header-чтение выполнено; постоянных MiMo tools пока нет.
- **P0.2:** строгий contract для text trunk: 48 блоков, dense0/MoE1..47,
  fused QKV двух форм, 39 sinks, отсутствие shared и NextN tensors,
  `exp_probs_b.bias`, KV heads4/8, K192/V128, RoPE64 и каждый quant type.
  Проверить обязательные веса независимо от `TENSOR_NOT_REQUIRED` loader.
  Negative fixtures: неизвестные metadata, missing/extra tensor, несовпадение
  pattern/shapes, quant row geometry, duplicate/overlap/truncated/64-bit ranges.
- **P0.3:** изолированная сборка, manifest source/compiler/CUDA/flags/patches;
  vocabulary-only, template и no-allocation loader oracles. Регистрация должна
  покрывать все 472 tensors, не добавлять MTP и shared-expert placeholders.
- **P0.4:** числовые CUDA fixtures для Q2_K/Q3_K/MXFP4 и BF16/F32 matmuls:
  dense, routed batch1/multitoken, sigmoid+bias/top-8/normalization,
  разные strides gate/up/down и repeated expert IDs. Проверить TF32 policy.
- **P0.5:** tiny full+SWA graph: fused QKV offsets, partial NeoX RoPE,
  scale `1/sqrt(192)`, value scale0,707, sinks, masks, KV4/8.
  Сравнить с CPU/F32 oracle при FA off/on, если конкретный путь поддерживается.
  Обязательны позиции127/128/129 и255/256/257, prefill/decode и rollback.

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
Не объявлять tokens/s по скоростям отдельного kernel или другого checkpoint.

### P2. Tokenizer, reasoning, tools и API

- **P2.1:** GPT-2 BPE с pre-tokenizer `qwen2`: 152576 tokens / 151387 merges,
  EOS151645 (`<|im_end|>`), PAD151643 (`<|endoftext|>`), add_bos=false.
  Получить полный EOG-набор через oracle; не копировать stop IDs из другой модели
  и не считать наличие audio tokens поддержкой audio. Проверить UTF-8,
  special-token escaping, exact IDs и обратную сборку token pieces.
- **P2.2:** сохранить и воспроизвести **встроенный** Jinja template.
  `enable_thinking=false` даёт prefix `<think></think>`; проверить default
  thinking, `reasoning_content` в history, system/tools и tool messages.
  Template SHA-256: `11ea52e156de38a458e6b7720ad45915d65b97d4ec979a09f55e3c9bd1b4d059`.
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

- **P3.1:** matrix key с model generation/layer/tensor/expert/type, byte-budget
  по реальным allocations. Mixed-format cache, pin активных entries,
  reuse только после завершения consumer. Проверить eviction и смену модели.
- **P3.2:** bounded reader→pinned staging→H2D stream→GPU compute.
  H2D event защищает чтение новых данных, consumer event — перезапись device slot.
  Mapping, cache entries и destinations живут до завершения всех пользователей.
- **P3.3:** batch ranges, allocation reuse, early host refill и host-copy варианты
  включать отдельно. Byte/logits parity с P1 на Q2_K/Q3_K/MXFP4,
  repeated experts, prefill и отмене в середине pipeline.
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

### P5. MTP — дополнительный этап, нужны отдельные веса

**Текущий GGUF не позволяет включить native MTP:** `nextn_predict_layers=0`,
нет NextN tensors и блоков≥48. Наличие `graph_mtp` в llama.cpp этого не меняет.
Первый рабочий профиль и P6 выпускаются с `mtp=false`.

- **P5.1:** найти или отдельно экспортировать MTP из совместимой **RL** source
  revision; проверить checksum, tokenizer, hidden/output norms, layer mapping,
  shared embedding/head, draft attention/FFN и quantization. Публичный config
  описывает три NextN, но совместимый sidecar пока не выбран и не проверен.
  Не подключать draft от MOPD/V2 только потому, что совпадает архитектурное имя.
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
До этого сохраняется текстовый профиль без MTP; отсутствие sidecar не блокирует P0–P4/P6.

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

Это предполагаемые параметры будущего backend; готового CLI MiMo пока нет.

| Параметр | Начало / варианты |
|---|---|
| MTP / multimodal | off / disabled для данного файла |
| Context | 2048 baseline, 4096 рабочий кандидат, затем8192 |
| KV / FA / CUDA Graphs | F32 / off / off для эталона; варианты включать после parity |
| Sampler | temperature0 для correctness; затем1,0/top_p0,95 по GGUF |
| GPU expert cache | 0 для sync baseline; cap8/12/16 ГиБ после memory audit |
| RAM cache | Динамический cap после вычета ОС, других процессов и transient buffers |
| Pipeline | reader1, chunk8 МиБ как старт; затем readers1/2 и chunks4/8/16 |
| Prefix/session reuse | off для первого A/B |
| Global memory target | До95% RAM/VRAM с дополнительным резервом на пики |

95% — верхняя граница полезного размещения, не требование заполнить память.
11,238 ГиБ базовых весов +16 ГиБ expert cache +1,699 ГиБ F32 full-context KV
при4K уже дают28,937 ГиБ без graph/scratch/desktop; cap16 не гарантированно
поместится. Окончательный cap определяется после измерения выбранного KV режима.
Перед запуском проверить external load; чужие процессы не завершать.

## 6. Карта файлов и порядок работ

Ниже **предлагаемые**, ещё не созданные MiMo-specific файлы.

| Область | Файлы |
|---|---|
| Inspector/contract | `tools/inspect_mimo2_gguf.py`, `tools/mimo2_loader_contract.py`, CPU tests |
| Native engine/oracles | `backends/mimo2/CMakeLists.txt`, loader/tokenizer/template/graph checks, `main.cpp` |
| API/template | `serve/mimo2.py`, `serve/fixtures/mimo26_chat_template.jinja`, request/parser/HTTP tests |
| Setup/profile | `tools/setup_mimo2.py`, `tools/prepare_mimo2_profile.py` |
| Model validation | `tools/check_mimo2_engine.py`, `check_mimo2_http.py`, позднее `check_mimo2_mtp.py` |
| Отчёты | `docs/mimo-v2.6-flash/MIMO26_FLASH_INSPECTION.json`, validation/benchmark reports |

Порядок: **P0 → P1 → P2/P3 → P4 → P6**. P2.1 можно делать в P0;
P5 и P7 требуют отдельных артефактов. Ближайший результат — inspector,
строгий contract и vocabulary-only/no-allocation oracles, затем CUDA fixtures.

## 7. Контроль завершения

- [ ] Локальный файл принят строгим contract; dependency и patches закреплены.
- [ ] GPU baseline совпадает с oracle, включая переходы SWA128.
- [ ] Tokenizer/template, reasoning, tools, оба HTTP API и веб-чат проверены.
- [ ] Mixed-quants cache/pipeline корректны, память ограничена, эффект измерен.
- [ ] Restore/shift/cancel/reload и заявленные контексты прошли проверку.
- [ ] Есть отдельный профиль, замеры defaults и регрессии других моделей.
- [ ] Дополнительно: MTP только после получения совместимых весов и P5.
- [ ] Дополнительно: мультимодальность только после P7.
