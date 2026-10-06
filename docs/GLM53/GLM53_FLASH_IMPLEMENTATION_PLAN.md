# План внедрения GLM-5.3-Flash

Дата исследования: 3 октября 2026. Исходное состояние Strata: `0561016`.
Обновление 4 октября 2026, ревизия `444f442`: основной файл для первых тестов —
`H:\GLM-5.3-Flash-GGUF\GLM-5.3-Flash-Uncensored-IQ3_XXS.gguf`.
Прежний `H:\GLM-5.3-Flash-GGUF\UD-Q3_K_XL` остаётся дополнительным профилем
для сравнения и проверки работы с моделью больше RAM.
Ход выполнения и точка продолжения: [GLM53_FLASH_IMPLEMENTATION_STATUS.md](GLM53_FLASH_IMPLEMENTATION_STATUS.md).

Цель — добавить GLM в Strata с вычислениями модели на GPU, подгрузкой холодных
матриц SSD → RAM → GPU, кэшем экспертов, сохранением сессий, статистикой и native MTP.
Рабочие Qwen3.8-Flash-Next и DeepSeek V4 Flash должны сохранить свои профили и поведение.

Это план, а не отчёт о работающей поддержке: проверены локальные заголовки GGUF,
размеры файлов, код Strata и доступные реализации llama.cpp. GLM не запускалась;
скорость, пиковая память и корректность генерации на этом ПК ещё не измерены.

## 1. Что находится в локальной модели

### Основная тестовая модель: Uncensored-IQ3_XXS

Проверена существующим инспектором 2026-10-04: один GGUF, **120 591 950 656 байт
(112,310 ГиБ)**, 1412 тензоров, `glm5next`, 45 основных блоков и 1 MTP.
Все имена/формы прошли статический loader contract выбранной ревизии Unsloth.
[Отчёт инспектора](GLM53_FLASH_IQ3_XXS_INSPECTION.json),
[сравнение метаданных и памяти](GLM53_FLASH_IQ3_XXS_COMPARISON.json).

| Группа нового файла | Все веса, ГиБ | Routed experts, ГиБ | Остальные веса, ГиБ |
|---|---:|---:|---:|
| Основная модель | 109,687 | 102,322 | 7,365 |
| MTP-блок | 2,614 | 2,443 | 0,171 |
| Всего | 112,301 | 104,766 | 7,536 |

Файл меньше прежнего на **25,094 ГиБ**. Это освобождает место для рабочего набора
в RAM, но фактическое потребление с ОС, состоянием и буферами ещё не измерено.
Без попаданий в GPU-кэш расчётный трафик экспертов основной модели за обычный
decode-проход — **2,842 ГиБ** вместо 3,477 ГиБ у UD-Q3_K_XL; это не замер скорости.

Имя файла не описывает фактические типы: тензоров `IQ3_XXS` в нём нет.
Основные routed matrices: `IQ2_S`, `IQ3_S`, `IQ4_XS`; MTP: `Q2_K`, `Q3_K`.
Одна матрица эксперта занимает соответственно 2,5625 / 3,4375 / 4,25 / 2,625 /
3,4375 МиБ. Среди остальных весов присутствуют BF16, F32, Q6_K и Q8_0.
Для P1/P3 нужна CUDA-проверка именно этих типов, а не только типов старого файла.

Все `glm5next.*` и tokenizer metadata совпали с прежней моделью, кроме текста
`tokenizer.chat_template`: четыре замены обращения `[0]` на `.0`. Новый встроенный
template дал тот же результат на 18 диалоговых вариантах корпуса; выполнены
72 проверки token IDs/round-trip и планирование 1032 матриц по заголовкам.
[Локальные проверки](GLM53_FLASH_IQ3_XXS_LOCAL_CHECKS.json).
Это проверка существующего Python frontend, не независимый llama tokenizer oracle.
Template загружать из выбранного GGUF; fingerprints моделей и архивы сессий разделять.
Структурная совместимость не доказывает равенство весов или ответов двух моделей.

Полный хеш нового файла, численные значения весов, GPU inference и MTP пока не проверялись.
В P3.4b выбранные 72 экспертные матрицы каждого профиля прошли byte parity после
доставки на GPU в mmap/native/auto; это проверка транспорта, не вычислений модели.
Ниже исходные данные UD-Q3_K_XL сохранены как история второго профиля; его числа
137,404 ГиБ и 3,386 ГиБ MTP не относятся к основной тестовой модели.

### Дополнительный профиль: UD-Q3_K_XL

Исходный снимок: [GLM53_FLASH_GGUF_INVENTORY.json](GLM53_FLASH_GGUF_INVENTORY.json).
Размеры ниже получены из GGUF; ГиБ = 2³⁰ байт, МиБ = 2²⁰ байт.

| Часть | Размер файла, байт | Тензоров |
|---|---:|---:|
| `00001-of-00004` | 9 429 859 | 0, только метаданные и токенизатор |
| `00002-of-00004` | 49 609 243 584 | 496 |
| `00003-of-00004` | 49 623 486 016 | 469 |
| `00004-of-00004` | 48 293 762 496 | 447 |
| Всего | 147 535 921 955, или 137,404 ГиБ | 1412 |

Все заголовки прочитаны, номера частей и количество уникальных тензоров согласованы,
данные каждого тензора укладываются в соответствующий файл. Хеши трёх больших частей
не проверялись: это проверка структуры и длины, а не полная проверка содержимого весов.

SHA-256 первой части:
`d72a357e8ccb0091d451866044b485bd3f562515c3f283b27775234bd0bccb77`.
Он совпал с [опубликованной первой частью Unsloth](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/blob/main/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL-00001-of-00004.gguf).

Есть особенность установщика: первая часть не содержит весов и заканчивается
на 29 байт раньше округлённого `data_start`. Это допустимо для части без тензоров.
Существующий `tools/setup_deepseek4.py:inspect_model()` начинает проверку длины с
`data_start` и при переносе этого кода ошибочно объявит файл недокопированным.
Нужно проверять фактический конец метаданных и отдельно диапазоны существующих
тензоров. Дописывать выравнивание в оригинальный GGUF не требуется.

### Общая архитектура двух профилей

| Параметр из локального GGUF | Значение |
|---|---|
| `general.architecture` | `glm5next` |
| Блоки | 45 основных + 1 NextN/MTP; `block_count=46` |
| Основной attention | 34 KDA, 11 DSA/MLA; DSA в блоках 3, 7, …, 43 |
| FFN | 3 начальных dense-блока, затем 42 MoE-блока |
| MoE | 288 экспертов, 8 выбранных на токен, 1 shared expert |
| Ширина | embedding 4096; expert FFN 2048; dense FFN 12288 |
| Индексатор attention | 32 головы, ширина 128, top-k 2048, kpool 4 |
| mHC | 4 потока, 20 итераций Sinkhorn |
| Контекст в метаданных | 1 048 576; доступность такого контекста на этом ПК не проверена |
| Токенизатор | GPT-2 BPE, pre-tokenizer `glm4`, словарь 154 880 |
| MTP | `nextn_predict_layers=1`, веса находятся в `blk.45.*` |

Разделение 45 + 1 и параметры гибридной архитектуры также согласуются с
[официальным config.json](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16/raw/main/config.json).
Нельзя трактовать 46 блоков как 46 обычных слоёв или заменить KDA механизмом Qwen.

### Весовые данные и память UD-Q3_K_XL

| Группа | Все веса, ГиБ | Routed experts, ГиБ | Остальные веса, ГиБ |
|---|---:|---:|---:|
| Основная модель | 134,009 | 125,174 | 8,835 |
| MTP-блок | 3,386 | 3,199 | 0,187 |
| Всего | 137,395 | 128,373 | 9,022 |

Это объём квантованных весов, а не требование к VRAM: отдельно нужны состояния
attention, рабочие тензоры, staging, GPU-кэш и возможное переупаковывание весов.
MTP можно загружать частично: его 3,199 ГиБ routed weights подходят для того же
матричного конвейера. Общее потребление MTP нельзя оценить только по 0,187 ГиБ
оставшихся весов; нужно учитывать его состояние и буферы проверки кандидатов.

`UD-Q3_K_XL` смешивает типы. В экспертных тензорах есть `IQ3_XXS`, `IQ4_XS`,
`Q6_K`, а в MTP — `Q3_K` и `Q4_K`. Одна матрица одного эксперта занимает
соответственно 3,0625 / 4,25 / 6,5625 / 3,4375 / 4,5 МиБ до служебного выравнивания.
Размер слота нужно получать из типа и shape конкретного тензора.

ПК: Ryzen 9 9950X, 128 ГиБ RAM, RTX 5090 32 ГиБ, Windows, CUDA 13.0;
модель на H: Samsung 990 PRO 4TB. Этот же стенд описан в
[замерах частотного кэша DeepSeek](../deepseek-v4-flash-0731/DEEPSEEK4_FREQUENCY_CACHE.md).
Даже основные routed weights почти занимают всю доступную RAM. Прогретый рабочий
набор может обслуживаться из RAM без SSD, но это нельзя обещать для любого нового
запроса, длинного prefill и смены сессии. Кэш ОС может вытеснять страницы.

При 8 из 288 экспертов и нулевых попаданиях в GPU-кэш расчётный объём матриц
на один обычный проход decode основной модели — около **3,477 ГиБ**:
`125,173828 × 8 / 288`. Это оценка трафика весов, не прогноз токенов/с;
она не учитывает padding, повторное использование и объединение маршрутов при MTP.

## 2. Какую реализацию брать за основу

Текущий DeepSeek backend объявляет pin llama.cpp
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. В локальной зависимости нет GLM-5.3.
Подстановка нового имени модели в существующий launcher недостаточна.

На дату исследования есть два несовместимых соглашения GGUF:

| Реализация | Формат и статус | Следствие для наших файлов |
|---|---|---|
| Unsloth, `glm5next/upstream` | `glm5next`; PR #27754 закрыт 1 октября | Первый кандидат для уже скачанного GGUF |
| Основной llama.cpp, PR #27773 | `glm5-next`; merged 30 сентября | Нужен отдельный адаптер совместимости либо другой GGUF |

В обсуждении прямо подтверждено, что основная реализация пока не читает старые
Unsloth GGUF. Для ветки Unsloth опубликованы условия корректности:
`NVIDIA_TF32_OVERRIDE=0` и `-fa off`; их следует взять как исходный режим и
перепроверить на выбранном commit. Это не универсальное ограничение всех будущих
реализаций GLM. [PR #27754](https://github.com/ggml-org/llama.cpp/pull/27754).

Отличаются также имена тензоров: локальные `indexer_compressor_ape` / `_gate`
против `indexer.kpool_ape` / `_gate` в основной реализации. Простого переименования
архитектуры недостаточно без проверки shape, layout и формул. Основная реализация
использует гибридную память с индексатором; MTP вынесен отдельно.
[PR #27773](https://github.com/ggml-org/llama.cpp/pull/27773),
[код основной реализации](https://github.com/ggml-org/llama.cpp/blob/master/src/models/glm5-next.cpp).

**Решение для первого внедрения:** отдельный backend `backends/glm5next/`, отдельная
директория сборки и зависимость с полным commit SHA и хешем архива. Начать с проверки
ветки Unsloth на существующих файлах. Окончательный SHA выбрать после P0, не закреплять
подвижную ветку как production-зависимость. Существующий pin DeepSeek оставить.

Перенос на основную ветку llama.cpp — возможный следующий шаг с таблицей соответствия
метаданных/тензоров и проверкой численной эквивалентности. Оригинальные GGUF обоих профилей
сохранять неизменными. CMake-патчи Strata требуют адаптеров к конкретным версиям:
проверять найденные точки вставки и прекращать сборку при несовпадении.

## 3. Что переносить из Strata

| Компонент | Действие |
|---|---|
| `expert_pipeline.hpp`, `expert_file.hpp` | Повторно использовать очереди, staging, native reads, mmap, отмену и срок жизни файлов |
| `cuda_expert_transfer.inc`, `expert_reuse.hpp` | GPU-кэш матриц, arena/dynamic режимы, CUDA events перед повторным использованием памяти |
| `expert_frequency.hpp` | Частотный допуск с LRU-вытеснением; отдельный учёт основной модели и MTP |
| `expert_pipeline_sched.inc`, `expert_pipeline_scope.inc` | Планировать gate/up/down после получения router IDs; адаптировать точки графа GLM |
| `gpu_only_audit.inc` | Запретить CPU-исполнение вычислительных узлов при prefill, decode и MTP |
| `vram_policy.hpp`, `vram_control.hpp` | Общий бюджет VRAM и оперативное уменьшение кэшей при росте состояния |
| Протокол backend, сервер, conversation cache | Сохранить интерфейсы; добавить GLM-специфичную сериализацию и capabilities |
| DSpark и DeepSeek output parser | Заменить на native GLM MTP и отдельный GLM parser |

Транспорт переносить небольшими общими модулями в `backends/common/` по мере проверки,
с адаптерами к каждой версии ggml. Избегать трёх расходящихся копий всего движка.
Выделение общего кода должно сопровождаться регрессией DeepSeek до изменения его профиля.
Описание существующего транспорта: [конвейер](../deepseek-v4-flash-0731/DEEPSEEK4_EXPERT_PIPELINE.md),
[очередь чтения и CUDA events](../deepseek-v4-flash-0731/DEEPSEEK4_EXPERT_PIPELINE_V2.md).

## 4. Этапы внедрения и критерии перехода

### P0. Совместимость файлов, зависимость и эталон

- [x] Сделать GLM-инспектор на основе `tools/gguf_reader.py`: одиночный GGUF и все четыре части,
  уникальные имена, shapes, типы, диапазоны, NextN и metadata-only shard.
- [x] Сопоставить все обязательные тензоры с выбранным loader; отдельно проверить
  `block_count=46`, отсутствие обычного прохода через MTP при MTP off.
  Статическая проверка P0.2: [отчёт](GLM53_FLASH_LOADER_COMPATIBILITY.md).
  Обе проверки повторены для Uncensored-IQ3_XXS; [новый JSON](GLM53_FLASH_IQ3_XXS_INSPECTION.json).
  Trace собранного кандидата проверен в P0.3b.2a без выделения памяти под веса;
  исполнения полного графа ещё нет.
- [ ] Собрать изолированную зависимость; записать полный SHA, CUDA/compiler, patch set.
  Проверить исправления KDA rollback, kpool state, sparse attention и больших индексов.
  - [x] P0.3a: отдельный `backends/glm5next/` CMake scaffold, обязательная проверка
    SHA-256 локального архива и vocab-only oracle harness; protocol CTest и archive
    fixtures прошли. Исходник oracle проверен синтаксически с локальными headers.
  - [ ] P0.3b: получить архив кандидата, записать проверенный hash, собрать реальные
    llama/oracle/CUDA targets и проверить графы. P0.3a не подтверждает такую сборку.
    - [x] P0.3b.1: архив `86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9` получен,
      SHA-256 записан; 3599 исходных файлов совпали с архивом. Изолированная сборка
      llama/oracle/CUDA прошла на MSVC 19.44 / CUDA 13.0.48, target `120a`, без
      патчей зависимости. 6/6 CTest; [build record](GLM53_FLASH_CANDIDATE_BUILD.json).
    - [ ] P0.3b.2: trace обычного графа без блока 45, KDA rollback, kpool,
      sparse attention и большие индексы. Сборка не доказывает численную корректность.
      - [x] P0.3b.2a: no_alloc trace двух GGUF, контексты 2048/4096 × batches
        1/4/16/256, MTP off, блоки 0–44, GPU-only scheduler audit. 16/16 графов;
        negative case с CPU embedding отклонён. Без загрузки/вычисления весов.
      - [x] P0.3b.2b: 22 численных CUDA fixtures KDA/mHC/I32 gather, включая
        scalar KDA reference, microbatch и snapshots с принятием 0–4 токенов.
        [Отчёты и параметры](GLM53_FLASH_GRAPH_VALIDATION.json), 7/7 CTest.
        Полный hybrid rollback и kpool пока не проверены.
      - [ ] P0.3b.2c: неполный kpool, hybrid seq_rm/rollback, численная sparse DSA
        и mixed-quant MoE; затем baseline logits при реальном исполнении.
        - [x] P0.3b.2c.1: настоящий граф synthetic GLM (KDA + DSA + mHC + MoE),
          98 проверок logits/state/kpool; 30 rollback-вариантов, pending save/restore,
          scalar pool reference, CPU/GPU и dense/sparse сравнения. Исправлено
          игнорирование TF32-off в F32 MMF кандидата локальным CUDA-патчем;
          10 прежних расхождений устранены без изменения порогов.
        - [x] P0.3b.2c.2: 108 CUDA matrix fixtures для всех типов обоих GGUF + F16,
          обычный/routed matmul, 8 выбранных экспертов, tokens 1/4/17;
          CPU и scalar references. [Отчёты](GLM53_FLASH_STATE_VALIDATION.json).
        - [ ] P0.3b.2c.3: реальная 45-layer модель, F16 KV и mixed-quant streamed
          outputs/logits/state. P1.1a: реальный forward и F16 KV проверены,
          64 токена и 9 912 320 logits бит-в-бит совпали с candidate selected-copy
          reference. Остались real-state rollback и длинный sparse-контекст;
          [отчёт](GLM53_FLASH_SYNC_VALIDATION.json).
- [ ] Подготовить GPU fixtures для KDA/DSA/mHC/MoE и сравнения logits, включая разбивку
  prefill на микробатчи и переход от полного к разреженному attention.
- [ ] Зафиксировать корректный tokenizer/template oracle из той же зависимости.

**Готово:** loader понимает локальный формат и строит правильный граф;
тестовые GPU-графы и токенизация согласованы с эталоном. Обход несовместимости
путём запуска экспертов основной модели на CPU не входит в план.

### P1. Основная модель, синхронная подгрузка, строгий GPU-режим

- [x] Создать `strata-glm5next` с существующим протоколом Strata и отдельным профилем.
  Архитектурные параметры читать из GGUF, а не из констант DeepSeek на 43 слоя.
  P1.1a: отдельный диагностический `strata-glm5next-smoke` уже исполняет модель;
  P1.1b/P2.5h: production executable/profile/setup и оба HTTP API проверены
  на IQ3_XXS, включая STOP/disconnect, seeded sampling, clean state и unload.
  См. [protocol manifest](GLM53_FLASH_PROTOCOL_VALIDATION.json).
- [x] Разместить attention, KDA, mHC, dense/shared FFN и router на GPU.
  Routed matrices держать в отображениях файлов и доставлять по фактическим IDs.
- [x] Сначала реализовать простой последовательный режим копирования как базу
  проверки. Все матричные операции, включая streamed experts, выполняются на GPU.
- [x] Отключить полное чтение/фиксацию всех экспертов при загрузке модели;
  ограничить pinned RAM staging и не создавать вторую полную копию mmap в RAM.
- [x] Добавить GPU-аудит и отказ с именем неподдержанного op, а не скрытый fallback.
  CPU обслуживает файлы, токенизацию, sampling и диспетчеризацию; его загрузка сама
  по себе не доказывает CPU-вычисление слоёв.
- [x] Снять фактические VRAM/RAM после загрузки, после prefill и во время decode.
  Использовать исходные настройки точности P0; не копировать принудительный FA on
  из `backends/deepseek4/main.cpp`.

P1.1a: IQ3_XXS, 39 prompt + 64 generated, ctx2048/batch16, F16 KV,
TF32/FA/MTP off: **1,288 ток/с** первый замер и **3,333 ток/с** повторный.
Это синхронный режим без VRAM expert cache; файловый кэш ОС не очищался.
Все logits конечны, CPU compute отсутствует; 10/10 CTest. Команды, память и
baseline: [manifest](GLM53_FLASH_SYNC_VALIDATION.json).

**Готово:** воспроизводимая генерация 64–128 токенов, без NaN/Inf, без CPU compute
узлов модели и без обязательной предзагрузки всех экспертов в RAM. Есть baseline logits/token IDs
и показатели памяти для дальнейших оптимизаций.

### P2. Токенизация, диалоги, tools и API

- [x] Добавить `glm4` в `tools/strata_tokenizer.py` и сравнить IDs с oracle
  на русском, английском, китайском,
  коде, числах, emoji и специальных токенах. Одного encode/decode round-trip мало.
  - [x] P2.1a: Python glm4 regex и экспорт pre_pattern; 7 fixture-тестов и
    round-trip на локальном GGUF. Qwen/JOYAI режимы сохранены.
  - [x] P2.1b: точные IDs совпали с собранной Unsloth-зависимостью на обоих
    GGUF после добавления `ignore_merges`. Это не разрешает запуск GLM-профиля
    без проверки графов и интеграции engine.
    - [x] P2.1b.1: runner `tools/check_glm5next_tokenizer.py`, проверки provenance,
      72 сравнения plain/rendered inputs, JSON с IDs и расхождениями; проверено
      на scripted subprocess, без численного oracle.
    - [x] P2.1b.2: по 80 сравнений plain/rendered text, parse_special off/on,
      на каждом GGUF; все совпали. Старый Python дал 8 расхождений на расширенном
      корпусе; GLM принимает целый vocab piece перед greedy merges. Экспортируется
      `ignore_merges`, добавлена регрессия Qwen/JOYAI. Отчёты:
      [IQ3_XXS](GLM53_FLASH_IQ3_XXS_TOKENIZER_PARITY.json),
      [UD-Q3_K_XL](GLM53_FLASH_UD_Q3_K_XL_TOKENIZER_PARITY.json).
      Проверку template rendering с независимым oracle выполнить отдельно.
- [x] Использовать встроенный Jinja template с нужными extensions, включая `break`.
  Проверить `[gMASK]<sop>`, сериализацию нескольких сообщений и tool results.
  P2.2: отдельный `GLMTemplate` и 10 fixture-тестов готовы; подключение к API
  и сравнение с inference oracle остаются отдельными задачами P2/P0.
- [x] Создать GLM parser для `<think>`, `<tool_call>`, `<arg_key>`, `<arg_value>`
  и ответов инструментов; проверить маркеры, разорванные между streaming chunks.
  DeepSeek DSML parser этому формату не соответствует.
  P2.3: parser и цикл call/result/continuation проверены на fixtures; вызовы
  выдаются целиком после валидации. Оба API проверены на mock в P2.5;
  runtime backend и полная модель ещё не проверены.
- [x] Развести завершения: EOS 154820, EOT 154827, EOM 154829; корректно сообщать
  `stop` / `tool_calls`, не отдавать служебные границы как пользовательский текст.
  P2.4: Service и сериализаторы обоих API проверены на mock; передача stop token
  из будущего GLM backend и полная модель ещё требуют проверки.
- [ ] Подключить выбор backend, settings, INFO/About/monitor, OpenAI и Anthropic
  endpoints. Возможности определять по backend capabilities, а не условию
  `architecture == deepseek4`.
  - [x] P2.5a: отдельные нормализаторы GLM и выбор через template hooks в Service,
    обоих generation handlers и Anthropic count handler; проверки на fixtures/mock.
    Выбор GLM при запуске и полная HTTP-интеграция ещё не готовы.
  - [x] P2.5b: сохранить call/result IDs в MCP continuation; проверить несколько
    раундов, отмену, лимиты и восстановление истории web app на mock/Node.
  - [x] P2.5c: восстановить reasoning_content web history по MCP-раундам для GLM;
    проверить clear_thinking через цепочку Node history → normalizer → embedded template.
  - [x] P2.5d: отображать reported INFO в About/Monitor без архитектурных условий
    DeepSeek/Qwen; не выводить поддержку GPU/MTP из имени модели. Проверено на mock/Node.
  - [x] P2.5e: проверить generation handlers OpenAI/Anthropic, JSON/SSE, ошибки options
    и tool-result round trip на mock; сохранить приоритет client tools над MCP для
    OpenAI schemas с обёрткой function. Проверка без HTTP listener и GPU.
  - [x] P2.5f: отмена до generation и в очереди завершает API/MCP без двойного
    учёта токенов; disconnect при prefill/reasoning/partial tool освобождает slot
    для следующего запроса. Проверено на mock, in-memory writer и threads;
    socket watcher дополнительно проверен в P2.5g; GLM GPU cleanup остаётся для runtime-проверки.
  - [x] P2.5g: реальные loopback HTTP-сокеты для обоих API: JSON/SSE Unicode и
    reasoning, tool-call/result continuation, HTTP 400 до начала stream, отключение
    клиента в очереди и при prefill/reasoning/partial tool. Production watcher
    отменяет молчащий mock engine; FIFO/usage/monitor и следующий запрос проверены.
    30 сценариев, 56 HTTP-запросов на Windows. GLM GPU engine и tokenizer oracle
    не задействованы; это интеграция HTTP с существующим GLM frontend.
- [x] Отобразить реальные настройки thinking: локальный template использует
  `reasoning_effort` low/high/max и `clear_thinking`. В нём нет `enable_thinking`
  или вставки `/nothink`; обещать отключение reasoning через Qwen-переключатель нельзя.
  P2.6: template capabilities, серверная валидация/сохранение и web controls
  проверены на mock и Node DOM; запуск с полной GLM ещё не выполнен.

**Готово:** чат, streaming, отмена, несколько tool calls и продолжение после tool
result работают через оба API; токены и шаблон совпадают с oracle.

### P3. Полный конвейер и кэш матриц

- [x] Подключить SSD/mmap → ограниченный RAM staging → asynchronous H2D → GPU ring
  → compute. После готовности router IDs объединять запросы gate/up/down и удалять
  дубликаты. Сохранять все 8 выбранных экспертов и исходные квантованные байты.
  - [x] P3.1a: общий Windows native reader в `backends/common/expert_file.hpp`,
    явные 64-битные offsets, проверка диапазонов и удержание файлового источника.
    Native fixtures восьми quant layouts и регрессия существующего CUDA pipeline
    DeepSeek прошли. Подключение GLM graph, staging/H2D и cache keys ещё не выполнено.
  - [x] P3.1b: native C++ planner после router IDs — проверка трёх tensor layouts,
    eight routed quants, dedup в порядке первого обращения, полные gate/up/down
    cache keys и 64-bit ranges. CPU fixtures и доставка его планов через общий
    CUDA pipeline прошли; реальный loader/router/graph ещё не подключён.
- [ ] Разделить ключи кэша по модели/поколению загрузки, основной/MTP ветке, слою,
  эксперту и матрице. Учитывать tensor layout, quant type, file offset и размер.
  - [x] P3.2a: reference-план адресации gate/up/down и cache-key contract в
    `tools/glm5next_expert_plan.py`; dedup IDs, bounded chunks и пять quant types
    проверены на fixtures, планы main/MTP построены по локальным заголовкам.
  - [ ] P3.2b: применить этот contract в runtime-кэше и транспорте, проверить
    reload/invalidation, удержание источников и CUDA events. P3.2a не включает GPU.
    - [x] P3.2b.1: отдельный C++ GPU cache с полными ключами, byte budget, LRU,
      leases/events и удержанием source owner; GPU fixtures проверяют hit parity,
      reload, pending consumers, bypass, ошибку загрузки и общий pipeline uploader.
      GLM loader/router/graph и реальные cached outputs пока не подключены.
    - [x] P3.2b.2: native dispatch scope объединяет plan pins, consumer leases и
      ordered miss/bypass transport; hits исключены из prefetch, bypass доставляет
      полные матрицы. Mixed plans, frequency rejection, zero budget и отмена
      проверены на CUDA fixtures; GLM graph и реальные model outputs ещё отсутствуют.
- [ ] Слот освобождать по CUDA event после последнего потребителя. Отмена графа
  должна завершать или отменять чтения и не оставлять обращения к закрытым mmap.
  - [x] P3.3a: отдельный native transport adapter удерживает source mappings,
    связывает полные ключи с ordered pipeline, дренирует отменённый остаток и
    освобождает источники после чтений. Проверены partial cancel/restart и
    exception teardown на Windows CUDA fixtures; graph/cache integration и
    отмена реального GLM inference ещё не подключены.
  - [x] P3.3b: `runtime_memory.cpp` подключён к общему pipeline/dispatch; после
    router IDs планируются gate/up/down с удержанием cache hits и mapping owner.
    Четыре slots, отдельный H2D stream, CUDA events, cleanup scope и GPU-only audit.
    Реальный IQ3_XXS: exact logits; native MTP использует тот же транспорт.
    Измерения и ограничения: [pipeline/MTP](GLM53_FLASH_PIPELINE_MTP.md).
  - [x] P3.3c: host waits выбранных expert copies заменены GPU event dependencies;
    режим 2 ставит одну пару событий на матрицу. 16/16 CTest и full-model logits
    bit-exact; пять повторов после четырёх warmup: MTP 1 вырос с 8,510 до
    9,586 ток/с. Глобальные RAM/VRAM targets 95% сохранены. Подробности и
    ограничения в [отчёте](GLM53_FLASH_PIPELINE_MTP.md).
  - [x] P3.3d: проверены decode readers/write-combined, раздельное старение main/MTP
    frequency history и cache allocator. Добавлены pool telemetry и ограничение
    RAM warmup по hard working-set cap. Финальный sweep на IQ3_XXS / RTX 5090:
    MTP off/1/2/3 — 9,902/9,843/8,948/7,600 ток/с; прежний профиль на том же
    бинарнике — 9,457. Локально выбраны MTP off, native allocator, decay131072;
    17 candidate + 13 transport/cache CTest, 83 Python tests, 11 real-model
    checks; baseline 9 912 320 F32 logits bit-exact. Один prompt, 64 output,
    четыре warmup + пять повторов; [результаты и ограничения](GLM53_FLASH_PIPELINE_MTP.md).
  - [x] P3.3e: packed cache в CUDA blocks, размер 16 МиБ выбран вместо 64 МиБ;
    последний блок уменьшается под доступную VRAM, пустой освобождается сразу.
    MTP off: 9,865 → 11,764 ток/с (+19,26%) на одной сборке, четыре warmup + пять
    повторов. Main cache 12,902 → 17,307 ГиБ. 18 candidate / 13 transport CTest,
    83 Python tests, 9 912 320 F32 logits bit-exact, 9 native pipe и 8 HTTP cases.
    Локальный профиль обновлён. Первый запрос медленнее; один prompt / 64 output.
    [Проверки и ограничения](GLM53_FLASH_SLAB_VALIDATION.json).
  - [x] P3.3f: опциональное сохранение частых expert IDs и восстановление VRAM
    cache, model/layout validation, ограниченные admission scores, атомарная
    запись; отдельный benchmark первого ответа и нового prompt. 37 запросов,
    19 candidate / 13 transport CTest, 83 Python tests, полные logits bit-exact,
    9 native pipe и 8 HTTP cases. Медиана первого запроса A 68,68 → 97,85 с,
    несмотря на снижение source/H2D на 22,36%; новый режим оставлен выключенным
    по умолчанию. [Результаты и ограничения](GLM53_FLASH_WARM_VALIDATION.json).
  - [x] P3.3g: опциональный разовый RAM scan с пропуском GPU cache, оценкой
    резидентности/бюджета и page-fault telemetry. 21 поисковый + 24 финальных
    запроса: 11,244 → 11,436 ток/с после четырёх warmup, но общий
    выигрыш не подтверждён; новый режим выключен. Повторный scan и VirtualUnlock
    отклонены. 20 candidate CTest, 83 Python, полные logits bit-exact,
    9 native pipe / 8 HTTP cases. [Ограничения](GLM53_FLASH_HOST_VALIDATION.json).
  - [x] P3.3h: реальные GLM mappings подключены к native file reader; добавлены
    системные paging/disk counters и уплотнение idle GPU slabs без новых VRAM
    allocations или чтения RAM. 59 ответов по двум references, 22 candidate /
    13 transport CTest, 83 Python, 9 native pipe cases. Ни native read, ни
    compaction, ни reader2 не дали устойчивого ускорения; defaults сохранены.
    Выявлены большие задержки подготовки весов при малом дисковом I/O и большом
    числе transition faults. [Измерения и ограничения](GLM53_FLASH_COMPACTION.md).
  - [x] P3.3i: опциональная private/hybrid загрузка и mmap только для экспертов;
    Windows RAM budget до загрузки, проверка backing buffers и commit telemetry.
    60 ответов A/B, повторный контроль, readers1/2: нового устойчивого прироста
    ток/с нет. N=0 уменьшил measured working set на 3,437 ГиБ; полная private
    загрузка GGUF не поместилась в commit budget. 26 candidate CTest, 83 Python,
    9 912 320 F32 logits bit-exact, 9 native pipe cases. Defaults сохранены.
    [Методика и ограничения](GLM53_FLASH_RAM_LOADING.md).
  - [x] P3.3j: раздельная защита host/device slots и optional early host refill;
    диагностические memcpy/SSE2/AVX2 и CPU affinity. 77 точных benchmark
    ответов, 32 candidate / 14 transport CTest, 83 Python, полные F32 logits
    bit-exact и 9 native pipe cases. Новый режим 12,565 ток/с при контролях
    12,935/11,833; устойчивого прироста нет, defaults сохранены.
    [Измерения и ограничения](GLM53_FLASH_HOST_PIPELINE.md).
- [x] Проверить побайтовое равенство доставленных матриц для экспертных типов обоих
  профилей: IQ2_S, IQ3_S, IQ4_XS, Q2_K, Q3_K, а также прежних IQ3_XXS, Q6_K, Q4_K;
  padding/alignment и последний неполный chunk; отдельно проверить
  полную тройку gate/up/down и границы файловых диапазонов.
  - [x] P3.4a: общий `backends/common/expert_pipeline.hpp` и GPU fixtures восьми
    layout: 432 сравнения матриц, mmap/native/auto, prefill/decode policies,
    два consumer streams, guards, неполные chunks и проверка счётчиков.
    CUDA-регрессия DeepSeek прошла. Данные P3.4a синтетические; GLM graph,
    деквантование и численная корректность остаются для интеграционной проверки.
  - [x] P3.4b: `tools/check_glm5next_transfer.py` и Windows CUDA range checker;
    реальные GGUF обоих профилей, слои 3/11/45 и IDs 0–6/287: 72 матрицы на модель
    × 3 режима, 432 сравнения всего. Диапазоны и payload прошли общий pipeline;
    GPU bytes совпали с независимым stdio-чтением, счётчики совпали с планом.
    JSON для каждого профиля сохранён; graph hooks, cache и MoE kernels не проверены.
  - [x] P3.4c: тот же checker с `--cache-check` проверяет cold/hit/forced eviction/
    generation reload/invalidation на выбранных реальных матрицах обоих профилей.
    72 матрицы × 5 этапов × 3 режима × 2 модели = 2160 сравнений bytes;
    hits не добавляют source/H2D bytes. Это отдельный LRU cache на каждую матрицу,
    без GLM graph, численных kernels и настоящего model reload.
  - [x] P3.4d: `--dispatch-check` проверяет связку ExpertDispatch/cache/transport
    на реальных матрицах обоих профилей: mixed hit/miss/bypass, all-hit без sources,
    отмена, invalidation/generation keys и zero budget. LRU/frequency × prefill/decode
    × mmap/native/auto: 8064 byte comparisons на RTX 5090. GLM graph, численные
    outputs и настоящий model reload не проверены; скорость не измерялась.
- [ ] Перенести bounded read queue, совместное планирование трёх матриц, защиту
  нужных текущему плану entries, LRU и frequency admission, общий VRAM reserve.
  - [x] P3.5a: общий bounded frequency history с decay и opt-in допуск в отдельном
    GLM GPU cache. Полные ключи, учёт bypass, атомарный выбор нескольких LRU victims,
    invalidation и CUDA lifetime проверены на fixtures; DeepSeek history regression
    прошла. Подключение к GLM graph, общий VRAM controller и замеры ещё не выполнены.
  - [x] P3.5b: `ExpertCache::protect_plan` предварительно удерживает все resident
    entries текущего плана, чтобы ранний miss не вытеснил поздний hit. Проверены
    LRU/frequency, overlapping pins, trim, reload/invalidation и отмена. Реальный
    GLM graph должен связать эту защиту с consumer leases и планом miss/bypass.
  - [x] P3.5c: общий с DeepSeek расчёт VRAM reserve и GLM cache controller с
    configured cap/total-device target, deferred trim и паузой admission при
    недоступном sample. Main/MTP, retired allocations, CUDA events и bypass bytes
    проверены на GPU fixtures с подставными memory samples. Live probe добавлен
    в P3.5d; период refresh и GLM runtime ещё не подключены.
  - [x] P3.5d: общий Windows PCI-matched NVML reader и owning live probe по умолчанию
    в GLM controller; ошибки API, stale outputs, rebind и concurrency проверены
    fixtures, реальный NVML → controller → cache smoke прошёл на RTX 5090.
    DeepSeek compatibility include проверен. Внешнее pressure и GLM graph не проверены.
  - [x] P3.5e: отдельный CUDA holder 256 МиБ и controller с кэшем 64 МиБ проверяют
    реальное внешнее pressure, защиту main pin, trim MTP и восстановление после free.
    LRU/frequency прошли на RTX 5090; 10 полных byte comparisons синтетических
    матриц. JSON с targets/samples/PIDs/hashes сохранён. GLM graph/OOM не проверены.
  - [x] P3.5f: отказ выделения cache matrix с `cudaErrorMemoryAllocation` переводит
    miss в полный uncached transfer, с отдельным `allocation_bypasses` counter.
    Иные ошибки остаются видимыми. Injected OOM, lifetime/accounting, pins,
    recovery и dispatch byte parity проверены на RTX 5090 для LRU/frequency,
    mmap/native/auto, prefill/decode. Физическое исчерпание VRAM и GLM graph не проверены.
  - [x] P3.5g: opt-in `refresh_if_due` ограничивает частоту global-memory probe;
    явный refresh обходит интервал. Первый вызов, граница периода, deferred trim,
    failed/throwing probe и recovery проверены для LRU/frequency на CUDA fixtures.
    Между samples сохраняются snapshot и admission; ошибки тоже ограничены по частоте.
    Выбор интервала, вызовы из GLM graph и влияние на скорость остаются для runtime.
  - [x] P3.5h: runtime cache + global RAM/VRAM targets 90/95%, NVML/controller
    каждые 500 мс, bounded mmap warmup/Windows working set maximum, retained
    model owners, exact MMQ padding, явный LRU и private CUDA pool. На IQ3_XXS
    все 9 912 320 logits совпали с baseline; 95%: 3,781/4,012 ток/с в новых
    процессах, 6,306 ток/с один повтор в том же pipe engine. Native pipe и оба
    HTTP API прошли с targets 95%. Это synchronous cache, overlap ещё нет.
    [Manifest](GLM53_FLASH_MEMORY_VALIDATION.json),
    [команды](GLM53_FLASH_MEMORY_TARGETS_TESTS.txt).
- [ ] Сравнить mmap/native/auto; отдельно подобрать число читателей prefill/decode.
  Прогретый режим должен читать RAM, если страницы там есть. Native read или cache
  miss сами по себе не равны физическому обращению к SSD.
  - [x] P3.7a: воспроизводимый benchmark выбранных прогретых диапазонов реальных
    GGUF: mmap/native/auto × 1/2/4 readers × prefill/decode; два warmups и пять
    samples, полные bytes/guards проверены вне таймера. По 18 конфигураций и
    9072 matrix comparisons на профиль RTX 5090. JSON содержит все samples,
    counters, hashes и фактическую decode policy. Runtime defaults не менялись;
    cold I/O, влияние compute, routing и окончательный подбор остаются открытыми.
  - [x] P3.7b: chunk/staging sweep 256 КиБ/1/4/16 МиБ на обоих GGUF;
    сравнения сгруппированы при одинаковых mode/phase/readers, указан расход
    pinned/ring. По 48 конфигураций и 24192 byte comparisons на профиль.
    CLI поддерживает выбор modes/readers/phases; результаты и raw samples
    сохранены. Это warm transport; runtime defaults и compute не изменялись.
- [ ] Добавить счётчики source bytes, H2D bytes, GPU hits, admission/bypass,
  ожидания read/H2D/compute, evictions и пик staging. Подтвердить перекрытие работ
  CUDA timeline, а не только наличием `Async` в имени вызова.
  - [x] P3.6a: общий pipeline разделяет CPU slot/consumer waits, сохраняет legacy
    wait total и считает fixed pinned/ring bytes, reader/queue high-water marks,
    текущие payload bytes и abandoned bytes. GPU fixtures и отдельная DeepSeek
    CUDA-регрессия прошли. GPU durations/timeline, INFO и GLM graph ещё не готовы.

**Готово:** результаты совпадают с последовательной базой P1 при одинаковых
вычислительных настройках; forced misses, вытеснение, mixed quants и отмена безопасны.
Есть измеренное перекрытие копирования с вычислением. Предсказание будущего router
не требуется: предзагрузка ограничена уже известными зависимостями графа.

### P4. Кэширование и выгрузка сессий

- [ ] Подключить существующую политику conversation cache и дисковые архивы,
  сохраняя полный GLM state: KDA convolution/recurrent, MLA cache, indexer keys/gates,
  kpool/позиции, sampler/RNG и состояние MTP, когда оно включено.
- [ ] Проверить возможности upstream state API; не считать обычный KV snapshot
  достаточным для гибридной модели. Исправить очистку состояния при неудачном restore.
- [ ] Версионировать снимок по model fingerprint, архитектуре, tokenizer/template,
  backend/state ABI и параметрам контекста/точности. Несовместимый снимок отклонять
  с пересчётом prompt. Файлы сохранять атомарно, с проверкой целостности.
- [ ] Начать с одной активной вычислительной сессии и переключения сохранённых чатов.
  Параллельные sequence IDs включать только после отдельной проверки hybrid memory.
- [ ] Выполнить A → B → A, archive → restart → restore, дописывание prefix,
  неполный kpool, отмену prefill/decode, повреждённый архив и очистку сессии.

**Готово:** продолжение после restore совпадает с непрерывным запуском в режиме
детерминированной проверки; нет смешивания чатов и утечек RAM/VRAM.

### P5. Native MTP с частичной загрузкой

В локальном `blk.45` есть NextN projection/norms, DSA и MoE. Использовать их;
DSpark DeepSeek не совместим с hidden states и архитектурой GLM.
Исходная upstream-ссылка — [PR #27917](https://github.com/ggml-org/llama.cpp/pull/27917).
На 2026-10-05 изучен заменяющий его [PR #29928](https://github.com/ggml-org/llama.cpp/pull/29928);
источники, ограничения и сопоставление с нашей веткой сохранены в
[исследовании](GLM53_FLASH_MTP_EXTERNAL_RESEARCH.md). Обновление pin и MTP index
sharing не входят в перенос сокращённого catch-up.

- [ ] Подключить MTP выбранной совместимой ветки с проверкой feature tensors,
  общим embedding/output там, где это предусмотрено моделью, и без дублирования
  основных весов. Проверить `index_share_for_mtp_iteration` по официальному config.
  - [x] P5.1a: native NextN context разделяет один model с target; feature rows,
    shared embedding/output и `load_mtp` подключены. В официальном config флаг
    index sharing равен true; выбранный кандидат пока пересчитывает draft indexer.
    Это ограничение отмечено в [отчёте](GLM53_FLASH_PIPELINE_MTP.md).
- [ ] Передавать expert weights MTP через тот же конвейер; задать отдельный лимит
  кэша и историю частот под общим бюджетом VRAM.
  - [x] P5.2a: optional main/MTP byte ceilings и независимые frequency history
    под общим ExpertCache budget; учёт retired leases, pins/events, deferred trim
    и invalidation. CUDA fixtures и 84 synthetic matrix comparisons через dispatch
    проверили MTP bypass при нулевом лимите и восстановление admission, сохраняя
    main hits. Default policy не менялась. Native MTP graph/draft ещё не подключены.
  - [x] P5.2b: memory-controller status учитывает main/MTP ceilings, resident и
    deferred bytes, включая retired leases и pending CUDA events. Совместное
    превышение global/branch не считается дважды; исправлен нулевой deferred
    при незавершённом branch trim. CUDA LRU/frequency и probe failure/recovery
    checks прошли. Вывод в GLM INFO/monitor и MTP graph ещё не подключены.
  - [x] P5.2c: реальные main/MTP graphs используют общие transport/cache/controller,
    раздельные keys/frequency и MTP byte ceiling под global VRAM target.
    Параметры `--mtp-cache-mib`, `--expert-pipeline` и размер chunk доступны в setup/profile.
- [ ] Начать с одного draft token, затем 2 и 3. Проверять кандидатов основной
  моделью, поддержать частичное принятие и корректную обработку стоп-токенов/tools.
  - [x] P5.3a: глубины 1/2/3, target sample-and-match, bonus token и ограничение
    длины подключены. Forced first/middle/all, seeded sampling, отмена/clean next
    прошли на полном GGUF; stop branches отдельно проверены на native fixture.
    Сохранение сессий и real tool-call/EOS coverage остаются отдельными проверками.
  - [x] P5.3b: catch-up/prefill черновика заполняют MLA/indexer/pooled-key cache
    без неиспользуемых attention/FFN/logits. Прежний путь доступен через
    `STRATA_GLM_MTP_CACHE_ONLY=0`. 90 synthetic state comparisons и следующие
    draft logits/hidden совпали бит-в-бит; full GGUF MTP1/2/3, forced rejection,
    seeded sampling и recovery также сохранили logits. 114 benchmark responses,
    9 pipe и 8 HTTP cases прошли. Устойчивое общее ускорение не подтверждено,
    рабочий MTP off сохранён. [Проверки и замеры](GLM53_FLASH_MTP_CACHE_ONLY.md).
  - [x] P5.3c: optional token-batch MMVQ для 2–4 токенов с арифметикой одиночного
    decode; 486 matrix cases bit-exact, full GGUF MTP1/2/3 и rollback прошли.
    98 benchmark responses совпали; MTP1 с новым путём 9,654/10,187 ток/с на
    двух prompts. Контроли зависели от памяти; устойчивого общего процента
    ускорения нет. Локальный token batch on, MTP off.
  - [x] P5.3d: сравнение в repeated-B/compaction режиме 12,748 ток/с; optional
    общий target/draft scratch освобождает 993,553 МиБ, полный GGUF bit-exact.
    Убраны временные строки из проверок logits; 28 CTest, 83 Python,
    108 benchmark responses и 9 итоговых pipe cases прошли. Новая сборка:
    MTP off 13,373, MTP1 11,082, shared MTP1 10,644 ток/с по медианам;
    устойчивого ускорения MTP нет, defaults off сохранены.
    [Замеры и ограничения](GLM53_FLASH_COMPACTION_MTP.md).
    [Проверки и ограничения](GLM53_FLASH_MMVQ_TOKEN_BATCH.md).
- [ ] Реализовать откат всех ветвей состояния на точное число принятых токенов,
  включая KDA и pooled indexer. Проверить reject-first, reject-middle, accept-all,
  отмену раунда и сохранение сессии после каждого случая.
- [ ] Проверить greedy-equivalence и корректность speculative sampling.
  Одинаковый seed сам по себе не гарантирует одинаковую последовательность при
  разных схемах расходования RNG; отдельно тестировать алгоритм принятия/коррекции.
  - [x] P5.5a: real 64-token greedy и 32-token seeded equivalence, включая все
    сравниваемые logits бит-в-бит. Найден и исправлен drift короткого CUDA batch:
    matmul 2–4 колонок использует путь одиночного decode. Добавлены 108 batch-vs-single
    CUDA matrix comparisons; исходный pin не изменён, применяется generated patch.
- [ ] В monitor вывести draft proposed/accepted, среднюю длину принятия, время draft,
  verify/rollback и полезные выходные токены/с. Включать MTP в рабочем профиле только
  после положительного A/B по итоговой скорости, а не по одному acceptance rate.

**Готово:** MTP сохраняет корректность основной модели и сессий; опубликованы
реальные затраты памяти и выигрыш/проигрыш на этом ПК. Если ускорения нет, остаётся
доступный переключатель с default off до дальнейшей настройки.

### P6. Замеры, регрессии и готовый профиль

- [ ] Обобщить harness `tools/benchmark_deepseek4.py` или добавить GLM-адаптер:
  фиксированные tokenized prompts, seed, sampling и версия backend в результатах.
  - [x] P6.1a: `tools/benchmark_glm5next_mtp.py` проверяет сохранённые IDs в
    отдельных warmed pipe engines; сравнение MTP 0/1/2/3, ceiling 256/512 МиБ,
    chunks 4/8 МиБ. Лучший измеренный вариант 1/512/4 установлен в локальном
    профиле с RAM/VRAM 95%. Это один 39-token prompt и 64 output; расширенная
    workload matrix остаётся открытой. [Результаты](GLM53_FLASH_PIPELINE_MTP.md).
- [ ] Измерить prefill 256 / 4096 / 16384 токена и decode 128–256 токенов;
  затем 32K/64K при достаточной памяти. Отдельно проверить смену темы и чата.
- [ ] Разделить первый проход, повторный prefill, восстановленный prefix и warm
  decode. Для warm — минимум три повтора, медиана и диапазон; помечать состояние
  файлового кэша ОС, не называть запуск cold без подтверждения физических чтений.
- [ ] Последовательно сравнить synchronous/async, LRU/frequency и MTP off/1/2/3.
  Менять по одному фактору. Зафиксировать token parity для транспортных изменений.
- [ ] Собирать TTFT, prefill/decode/итоговую скорость, RAM/VRAM peak, CPU процесса
  и системы, GPU utilization, реальные чтения диска с моделью и H2D. Сопоставление
  H: с физическим диском определять при запуске; сейчас это PhysicalDrive6.
- [ ] Прогнать Qwen и DeepSeek по прежним профилям: загрузка, генерация, API,
  tools, сессии, MTP/DSpark. Для изменённого общего транспорта — также его unit tests.
- [ ] Добавить GLM setup/launcher, справку параметров, результаты в `bench/results/`
  и краткую инструкцию в `docs`. Локальный сервер по умолчанию слушает 127.0.0.1.

**Готово:** воспроизводимый рабочий профиль GLM, численные результаты на данном ПК,
успешные регрессии и простой возврат на прежние профили Qwen/DeepSeek.

## 5. Исходные настройки эксперимента

Это предлагаемые стартовые значения для Uncensored-IQ3_XXS, пока не измеренные
рабочие настройки. UD-Q3_K_XL проверять отдельным профилем с собственными замерами.

| Настройка | Начало | Последующая проверка |
|---|---|---|
| Активные последовательности | 1 | Много чатов через сохранение/восстановление |
| Контекст | 4096 | 8192 → 16384 → 32768/65536 по памяти и точности |
| Batch / ubatch | 256 / 256 | 512, затем 1024; независимо оценивать workspace |
| Кэш матриц основной модели | до 8 ГиБ | 12/16 ГиБ, только если вмещаются реальные пики |
| Свободный резерв VRAM | 1,5–2 ГиБ | Подбор по allocation peaks и рабочему столу |
| Staging / кольцо | 16 МиБ на слот, 4 слота | Проверить фактические размеры и chunking |
| Читатели | prefill 2, decode 1 | 1/2/4 с контролем CPU и физического I/O |
| Политика кэша | LRU для baseline | Frequency после проверки parity |
| MTP | off | 1 → 2 → 3 draft token после P4 |
| Точность | настройки совместимой ветки P0 | Любое ускорение FA/TF32 — после численной проверки |

Выбранный GPU-кэш — потолок, а не обязательное выделение. Если attention/workspace
не помещаются, сначала уменьшать cache/batch/context; CPU compute fallback запрещён.
Квантизацию KV и изменение indexer top-k не включать в начальную оптимизацию:
это отдельные изменения точности, требующие проверки качества.

## 6. Карта изменений и порядок работ

| Файлы/директории | Планируемая ответственность |
|---|---|
| `backends/glm5next/` (новое) | CMake pin/patches, GPU engine, hybrid state, native MTP |
| `backends/common/` (новое, поэтапно) | Общий транспорт и политики кэша с адаптерами ggml |
| `tools/setup_glm5next.py` (новое) | Admission локальных shards, бюджет памяти, отдельный профиль |
| `tools/strata_tokenizer.py`, GLM tests | `glm4` и эталонные token IDs |
| `serve/glm5next.py` (новое), `serve/server.py` | Шаблон/stream parser, завершения, API и выбор backend |
| `serve/vram_settings.py`, `serve/web/app.js` | Capabilities, настройки памяти/MTP и статистика |
| GLM validation/benchmark tools | GPU parity, pipeline, sessions, MTP и воспроизводимые замеры |
| `docs/`, `bench/results/` | Установка, ограничения и измеренные результаты |

Порядок: **P0 → P1 + P2 → P3 → P4 → P5 → P6**. Первые API-проверки выполнять
сразу после P1/P2; регрессии запускать при каждом выделении общего кода.
Критерий первого запуска — правильный короткий диалог на GPU с demand-loading.
Завершение всего плана — конвейер, сессии, статистика, проверенный MTP и замеры.

Мультимодальность — отдельное расширение после текстового пути: в указанной
директории нет vision projector. Понадобятся совместимый mmproj, GLM preprocessing,
отдельный бюджет VRAM и тесты изображений. Заявленный в metadata контекст 1M также
требует самостоятельного исследования памяти и длительных проверок; этот план
не считает его доступным только по наличию числа в GGUF.

## 7. Контроль завершения

- [x] Проверить локальные GGUF, сохранить инвентаризацию и изучить совместимость.
- [x] Составить план с сохранением Qwen/DeepSeek и ограничением GPU compute.
- [ ] P0: закреплённая совместимая зависимость и эталоны.
- [ ] P1–P2: основной GPU engine и полноценный текстовый API.
- [ ] P3–P4: матричный конвейер, кэш и безопасное переключение сессий.
- [ ] P5: native MTP, rollback и измерение полезного ускорения.
- [ ] P6: профиль этого ПК, регрессии и опубликованные замеры.
