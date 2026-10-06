# Статус внедрения Hy3

Обновлено: **2026-10-06**, `Asia/Yekaterinburg`.
План: [HY3_IMPLEMENTATION_PLAN.md](HY3_IMPLEMENTATION_PLAN.md).

Этот файл хранит проверенный прогресс и следующую задачу. **Полный GGUF Hy3
генерирует через отдельный native pipe engine.** HY3-01 добавил inspector, строгий loader contract,
изолированные CPU oracles, проверку tokenizer/template и исправление MTP load flags.
HY3-02 закрыл P0 числовыми CUDA fixtures и проверками main/MTP graph/state.
HY3-03 добавил синхронный selected-copy runtime, подтвердил точное совпадение
token IDs/logits и измерил baseline на полном GGUF. HY3-04 проверил восстановление после
управляемых отказов и добавил отдельный template/reasoning/tool parser. HY3-05
подключил отдельный профиль к Service, обоим HTTP API и веб-чату; полный GGUF
прошёл tool/result continuation и disconnect/recovery. HY3-06 добавил bounded
GPU matrix cache с exact logits/IDs и сокращением H2D. HY3-07 подключил
bounded pipeline с реальными router IDs, native file readers и CUDA events.
HY3-08 добавил opt-in tensor batching с удержанием cache entries до завершения
копий. Его небольшой speed effect требует расширенных замеров; default остаётся off.
HY3-09 воспроизвёл intermittent resident/native mismatch, добавил диагностику
и раздельные таймеры доставки. CUDA graphs отключены в Hy3 как обход;
внутренняя причина сбоя ещё не исправлена. Асинхронный pipeline сохранён.
HY3-10 добавил native MTP: resident блок80, отдельный draft KV, пакетную
проверку черновиков и откат. Depth1 прошёл полный6-prompt corpus и HTTP;
depth2/3 оставлены для экспериментов. HY3-11 добавил управляемый RAM-кэш:
correctness полного GGUF и HTTP прошла, LRU замедлил короткий benchmark.
HY3-12 добавил frequency admission RAM и защиту от prefill scan;
на третьей повторной en/code паре получил3,461 ток/с против2,854 pooled off.
Это эффект прогретого короткого corpus. HY3-13 добавил GPU prefill policy и
проверил её вместе с увеличением VRAM-кэша и tensor batching: прогретая en/code
пара ускорилась с3,467 до4,533 ток/с без MTP. HY3-14 добавил плотное размещение
матриц в VRAM: около15,2 ГиБ вместо10,3 ГиБ и5,755 против4,388 ток/с в новом
сравнении без MTP. HY3-15 добавил общий cache main/MTP и shared scratch:
main-кэш вырос13,464→14,675 ГиБ, но resident MTP оказался быстрее
(6,474 против5,854 ток/с). Быстрый resident профиль сохранён.
Условия и ограничения — ниже.

## Текущее состояние

| Область | Статус |
|---|---|
| Подготовка PREP-01 | DONE: исследование, план и этот статус |
| Репозиторий HY3-14 | HEAD `819ad13` с существующими изменениями; отдельный измеренный engine `19369d58…`, source hashes и build manifest в отчётах |
| HY3-15, размещение MTP | Общий cache и shared scratch реализованы;177 MTP/796 runtime PASS; скорость streamed ниже resident, опция для экспериментов |
| Модель | `H:\models\hy3\Hy3-Q3_K_M-mtp.gguf`, один файл, 127,083 ГиБ |
| Формат | `hy_v3`, GGUF v3, 1298 тензоров, 43 metadata records |
| Состав | 80 основных блоков, 1 MTP; dense0, MoE1..79, MoE80+NextN |
| MTP | HY3-10: native driver depth1/2/3, resident блок80; отдельный KV, verify/rollback/catch-up; depth1 прошёл corpus, lifecycle и HTTP |
| Candidate dependency | Unsloth `86ebfef2`; CPU и CUDA13.0/sm_120, 3 correctness patches + изолированный runtime patch |
| Inspector/loader contract | DONE: все1298 тензоров, JSON и отрицательные CPU fixtures |
| Tokenizer / template | 2194 / 144 проверки, 0 расхождений с native oracles |
| CUDA admission | 150/150 kernel cases, 86/86 graph/state checks; CPU fallback в GPU-графах отсутствует |
| Native engine | DONE HY3-03: Windows pipe, GPU-only, sync selected-copy, MTP-off |
| Восстановление после отказов | HY3-04: pinned OOM, partial ReadFile/cancel, simulated RAM/VRAM, unload/reload; 32/32 checks |
| Reasoning/tool parser | HY3-04: 17 tests /7327 fragmentation sequences, adapter144/144 native comparisons |
| HTTP / UI | HY3-05 PASS: OpenAI/Anthropic JSON/SSE, tools, cancellation, отдельный профиль и реальный browser chat |
| Кэш / pipeline / sessions | HY3-06 cache и HY3-07 bounded pipeline PASS; sessions TODO |
| Полный GGUF | 6 prompts + 2 повтора в каждом режиме; точные logits/IDs, STOP/recovery PASS |
| Tool/result/answer на полном GGUF | HY3-04 direct pipe; HY3-05 HTTP PASS: `get_temperature({"city":"Paris"})` → локальный результат17 → `17 C`; точные IDs |
| Локальная скорость, токенов/с | pinned **1,219**, native **2,086** на 37 decode-шагов en+code; последовательные наблюдения, не A/B |
| HY3-06 cache sweep, прогретые en+code | cache8 **1,821**; cache-off **1,360/1,377** до/после,37 decode-шагов на пару; около+33% в этом коротком прогоне |
| HY3-07 pipeline sweep, прогретые en+code | readers2 **3,100**, readers1 **2,523**, sync **1,869/1,842**; cache8 во всех вариантах; около+67% в этом коротком прогоне |
| HY3-08 tensor batch, короткий ABBA | off **2,804**, on **2,930** в среднем; около+4,5%, но большой разброс между процессами; default off |
| HY3-09 CUDA graphs off, короткий ABBA | graphs on **3,000**, off **2,905** ток/с; −3,2% decode, +1,6% времени пары; обход correctness-сбоя |
| Настройки baseline | Context2048/batch17/F32 KV, pinned16 МиБ; MTP/cache/pipeline/TF32/fusion off |
| Сохранённые профили HY3-11 | `build-local/hy3-http-ram-mtp0/hy3.json` и `hy3-http-ram-mtp1/hy3.json`: GPU cache8192 МиБ, readers2, RAM auto; указывают на сохранённый LRU engine9422341b в `build-local/hy3-ram-lru` |
| Tensor batch | Доступен через `--pipeline-batch 1`; старый профиль HY3-08 требует его старый checksummed exe |
| Сохранённый HY3-08 для сравнения | `build-local/hy3-baseline-batch0/hy3.json`; CUDA graphs on, известное intermittent ограничение |
| Сохранённый HY3-07 rollback | `build-local/hy3-baseline-pipeline2/hy3.json` и точный старый exe |
| Сохранённый synchronous rollback | `build-local/hy3-baseline-cache8/hy3.json` и неизменённый exe HY3-06 |
| Профили HY3-12 | `build-local/hy3-http-frequency-mtp0/hy3.json` и `hy3-http-frequency-mtp1/hy3.json`: RAM cap64 ГиБ/frequency, GPU cache8 ГиБ/readers2; отдельный checksummed exe a1117b6d |
| HY3-12, третья en/code пара | RAM off2,766/2,948, frequency3,461, frequency+MTP1 3,698 ток/с; первые2 прохода с заполнением медленнее |
| HY3-13, третья en/code пара | Контроль3,396/3,541; GPU decode policy + cache11 + tensor batch4,460/4,608 ток/с; pooled3,467→4,533 (+30,7%), без MTP |
| Профиль HY3-13 | `build-local/hy3-http-speed13/hy3.json`: cache11264 МиБ, policy decode, readers2/chunk4/batch1, RAM cap65536 МиБ/frequency, MTP0; отдельный exe7be63e00 |
| HY3-14, плотное GPU размещение | Матрицы15,218–15,279 ГиБ / backing15,688–15,750 ГиБ; pooled5,755 против4,388 ток/с (+31,1%), H2D−23,3%; последние2 en/code пары после3 прогревочных |
| Профиль HY3-14 без MTP | `build-local/hy3-http-arena14/hy3.json`: arena cap16384 МиБ, policy decode, readers2/chunk4/batch1, RAM cap65536 МиБ/frequency; отдельный exe19369d58 |
| HY3-14 MTP off/on/off | Depth1: 6,424 против5,869 ток/с (+9,5% decode), полное время пары−2,0%; acceptance89,5%; GPU hot matrices13,461 вместо15,16–15,22 ГиБ; отдельный профиль `build-local/hy3-http-arena14-mtp1/hy3.json` |
| Следующая задача | **Расширить speed corpus со сменой темы и длинной генерацией; измерить оставшиеся ожидания конвейера. Причина CUDA graphs остаётся открытой** |

Созданы изолированный `backends/hy3`, tools/tests и отчёты. В общем Python
tokenizer добавлен `hunyuan-dense`; действующие профили и модель не изменялись.
Регрессии tokenizer Step/GLM и проверка разбиения цифр Qwen прошли.

## Подтверждено при подготовке

- File size **136 454 632 928**, header_end **5 161 421**, data_start **5 161 440**.
  1298 уникальных tensor names; все размеры известны, offsets выровнены,
  диапазоны не пересекаются, последний payload заканчивается на EOF.
- Trunk payload **134 592 752 896 байт (125,349 ГиБ)**:
  routed **125 354 115 072 (116,745 ГиБ)**, остальные **9 238 637 824 (8,604 ГиБ)**.
- MTP payload **1 856 718 592 байта (1,729 ГиБ)**:
  routed **1 717 567 488**, остальные **139 151 104**.
- Несмотря на название файла, routed weights используют IQ3_XXS/IQ4_XS
  и K-quants. Полная таблица типов и архитектуры находится в плане.
- SHA-256 header без padding/payloads:
  `f3307357f0b6ab163f188f7d19ba57d2960ca70a7491b511b23b2745a9500ca9`.
- SHA-256 UTF-8 строки `tokenizer.chat_template`:
  `7fc351fee674c13754656ba7f33a3ca426bfb7231039f48444360a3f3c5ecf3e`.
- Архив `build-local/llama-glm-86ebfef2.tar.gz` — 37 493 950 байт,
  SHA-256 `f8e524b635b726bae74fd8f84bb9249e5b09384c63207f6707c5a3f921acad99`.
  `hy-v3.cpp` в нём и в распакованном GLM candidate совпадает:
  `224921b8ce6f9be02dc1252ef6847a93f81386021b24d7d08eb25444e787d62e`.
- Candidate содержит основной граф и native MTP; основной `third_party`
  не содержит регистрации `hy_v3`. Это результат чтения исходников,
  не результат компиляции или проверки совместимости на весах.
- Windows сообщает **134 813 700 096 байт RAM = 125,555 ГиБ** через
  `GlobalMemoryStatusEx`; NVIDIA сообщает **32 607 МиБ VRAM**, RTX 5090,
  driver581.80. Лимиты95%: **119,277 ГиБ RAM / 30 976,65 МиБ VRAM**.
  Во время подготовки `nvidia-smi` показал 31 463 МиБ занятой VRAM;
  это внешняя нагрузка, не потребление Hy3. Перед тестом проверять заново.

Публичные материалы и ссылки с оговорками о совместимости приведены в плане.
Чужие токены/с не переносить в таблицу локальных результатов.

## Не проверено

- HY3-09: внутренняя причина intermittent resident/native расхождения.
  Воспроизведено на mixed batch4/17 и F32 batch17; веса/native-pinned совпали.
  CUDA graphs отключены как обход, history fixtures прошли. Graph-enabled
  контроль итоговой сборки также прошёл: воспроизводимость непостоянна.
  Все FAIL сохранены; проблема не считается исправленной внутри dependency.
- Полный checksum весов и соответствие опубликованному AngelSlim файлу.
- Независимый полноразмерный CPU/F32 oracle: full-model parity проверен между
  двумя путями доставки одного native GPU graph; GPU-resident oracle — на fixtures.
- Flash Attention, F16 KV, production контекст длиннее проверенных 362 prompt tokens.
- Полноразмерная generation-проверка low/high и stochastic sampling через HTTP;
  эти effort modes проверены template fixtures и scripted HTTP, реальный corpus — no_think.
- Kernel-level GPU timeline, физический SSD traffic, длительные рандомизированные A/B, настоящий
  driver OOM/device-loss recovery и внешнее давление памяти. Pinned OOM и
  RAM/VRAM availability injection проверены; Source-read bytes не равны SSD traffic.
- MTP: расширенный/randomized corpus, stochastic speculation и production
  sessions. Native driver реализован в HY3-10; depth3 имеет сохранённое
  greedy расхождение на коротком mixed fixture, в рабочий профиль не выбран.
  При temperature>0 используется проверяемый target-only fallback.
- Установка через общий setup, финальные быстрые defaults и full-model регрессии
  других моделей; отдельный Hy3-профиль, HTTP/web chat и CPU-регрессии уже проверены.

## Таблица этапов

| Этап | Статус | Условие перехода |
|---|---|---|
| PREP-01 | DONE | Проверены заголовок/исходники; создана документация |
| P0 — contract/dependency/oracles | DONE HY3-01/HY3-02 | Header/contract/oracles, CUDA kernels, synthetic main/MTP graph и KV |
| P1 — sync GPU baseline | P1.1–P1.3 DONE; P1.4 controlled failures PASS | Полный GGUF/parity/метрики готовы; настоящий driver OOM и внешний pressure не проверены |
| P2 — tokenizer/template/API | DONE HY3-04/HY3-05 в проверенном объёме | Native parity, request adapters, оба API, tools, cancel/recovery, профиль и web smoke |
| P3 — cache/pipeline | Cache HY3-06, pipeline HY3-07, tensor batching HY3-08 и GPU prefill policy HY3-13 готовы в проверенном объёме | Физические SSD counters, расширенные A/B и диагностика intermittent fixture mismatch остаются |
| P4 — sessions/context | TODO | Fresh vs restored/shifted parity и bounded memory |
| P5 — native MTP | HY3-10 частично: depth1, corpus, lifecycle и HTTP PASS | Сессии, stochastic speculation, отдельный streaming budget и расширенное A/B остаются |
| P6 — profile/release checks | Частично: experimental profile/CPU regressions HY3-05 | Остались финальные замеры/defaults и release checks |

`DONE` относится только к указанному объёму. `IN PROGRESS` не означает,
что модель уже может генерировать. Не закрывать P0 по факту успешного чтения header.

## Точка продолжения: HY3-13 — GPU prefill policy и пакетная доставка

На полном GGUF выбран отдельный профиль без MTP: cache11 ГиБ,
`--gpu-cache-policy decode --pipeline-readers 2 --pipeline-batch 1`, RAM cap64 ГиБ.
Повторные короткие замеры дали4,460/4,608 ток/с против3,396/3,541 у контроля
cache8/all/batch0. Это третья en/code пара после двух обучающих пар в каждом
новом процессе; OS file cache не очищался. Холодный первый запрос не ускорен
гарантированно. Engine defaults сохранены; профиль включается явно.

1. Проверить длинную генерацию и смену тем с одинаковой историей запросов,
   отдельно от повторов коротких prompts. RAM/GPU eviction уже ограничены
   бюджетом, но этот benchmark не устанавливает скорость на новых темах.
2. Повторно профилировать лучшую конфигурацию отдельно от speed runs.
   Средний decode H2D остался99,370 ГиБ на37 полезных токенов; следующие
   кандидаты — оставшиеся route/tensor fences и перекрытие доставки с compute.
   Не складывать CPU-времена параллельных readers с elapsed time.
3. CUDA graphs остаются off. Не включать их по одному успешному повтору;
   исходная intermittent проблема dependency не устранена.

### Архивная точка после HY3-12 — частотный RAM-кэш

MTP depth1 завершён в объёме greedy, fresh requests, cancel/recovery и HTTP.
LRU runtime HY3-11 прошёл correctness; HY3-12 добавил frequency admission. Профили HY3-10 с тем же
checksum перенаправлены на сохранённый `build-local/hy3-mtp-accepted/strata-hy3.exe`.
Сравнения RAM выполнены отдельными сборками, старые результаты сохранены.

### Следующий эксперимент HY3-13 — GPU prefill policy

1. Сохранить HY3-09 cache8/readers2/chunk4/batch0, CUDA graphs off как контроль.
   Использовать отдельный диагностический `--profile-delivery` для выбора
   следующей оптимизации; performance A/B проводить с выключенными таймерами.
2. Проверить отдельную GPU prefill admission policy: ограничить заполнение кэша
   одноразовыми prefill matrices, сохранить существующие hits и корректную
   доставку всех misses. Не смешивать policy с tensor batching или MTP.
   HY3-09 измерил45,813 с cache admission из116,449 с длинного prefill,
   включая14,162 с allocation и25,016 с free; это основной кандидат.
   Проверить cold/warm/eviction, длинный prefill, точные logits/IDs и STOP/recovery.
3. CUDA graphs не включать по одному успешному повтору. Для дальнейшей
   диагностики есть `--runtime-probe history-graphs`, веса и intermediate
   captures. Причину нужно подтвердить на исходном uncaptured graph;
   callbacks меняют разбиение. Не ослаблять exact comparisons.

### Архивная точка перед HY3-09

1. Изолировать исходный mixed batch4 resident/native FAIL HY3-08. Воспроизводить
   исходную последовательность F32→mixed и смену batch/contexts; при расхождении
   новый checker сохраняет полные logits. Сверить промежуточные тензоры и CUDA
   graph reuse, не ослаблять точное сравнение и не объявлять причину по предположению.
2. Сохранить cache8/readers2/chunk4/batch0 как контроль. Разделить время native
   file read, cache admission/allocator и ожидание producer; CPU суммы двух
   readers не считать elapsed time или физическим SSD traffic.
3. Отдельно измерить prefill admission policy: не смешивать изменение eviction
   с batching. Повторить чередующиеся запросы в стабильных условиях, включая
   длинный prefill, exact logits/IDs, tools и STOP/recovery. MTP пока off.

### Архивная точка перед HY3-08

1. Сохранить HY3-07 cache8/readers2/chunk4 как контрольный вариант. Проверить
   доставку всех выбранных matrices одного tensor с одним fence вместо fence
   на каждую матрицу и cache fill. Не перезаписывать scheduler scratch до
   завершения прежних потребителей; держать cache pins до drain, включая ошибки.
2. На F32/mixed/wide fixtures проверить exact bytes/logits, cache hit/miss,
   eviction, частичные ошибки, STOP/recovery. Затем повторить полный GGUF
   и HTTP tools с exact IDs. MTP и sessions оставить off.
3. Измерить batching отдельно от изменения admission/cache policy. Сравнить
   warm English/code и длинный prefill; контролировать RAM/VRAM95%.
4. CUDA events HY3-07 показали лишь небольшой H2D/compute overlap. Не объяснять
   весь выигрыш +67% этим overlap: read/H2D concurrency и host overhead требуют
   отдельного профилирования. Физический SSD traffic всё ещё не измерен.

### Архивная точка перед HY3-07

1. Подключить bounded `backends/common/expert_pipeline.hpp` по примеру Step
   к Hy3 cache miss path. Сохранить synchronous cache8 как контрольный вариант.
2. Начать с двух staging slots и одного reader; H2D/read overlap включать
   только с явными CUDA events. До eviction/перезаписи scratch завершать
   потребителей; сохранять mappings/cache pins до полного drain.
3. Проверить exact bytes/logits/IDs на mixed quants и полном GGUF, cold/warm
   cache, partial read/copy failure, cancel и следующий запрос. Потом A/B
   одинакового corpus, включая HTTP tools. MTP и sessions оставить off.
4. По отдельности измерить tensor batching и prefill admission. В HY3-06
   cap12/16 ГиБ не ускорили прогретую пару относительно8; полезно проверить
   CUDA allocation overhead и timeline, не считать каждый лишний байт VRAM
   полезным кэшем. Разделение реального SSD traffic и OS cache hits ещё TODO.

### Архивная точка перед HY3-06

1. Добавить bounded GPU matrix cache в selected-copy путь, по образцу Step,
   с ключом model generation/layer/tensor/expert/type и явным ограничением VRAM.
   Начать с MTP-off, synchronous transport и нулевого кэша как контрольного варианта.
2. Проверить точные bytes/logits/greedy IDs на F32 и mixed fixtures, batch1/17,
   hit/miss/eviction, cancel/recovery и unload/reload. Старые ссылки кэша
   не должны переживать смену model generation.
3. Считать GPU hit/miss, bytes from file/RAM и реальные H2D bytes отдельно;
   подтвердить сокращение H2D при повторе одинакового corpus полного GGUF.
   Не выдавать уменьшение source-read bytes за физический SSD traffic.
4. Перебрать cap8/12/16 ГиБ только после runtime clamp с учётом global95%,
   non-routed8,604 ГиБ, KV/graph/scratch и внешней нагрузки. Сначала parity,
   затем повторные A/B скорости; после этого — P3.2 async pipeline.

### Архивная точка перед HY3-05

1. Добавить OpenAI/Anthropic request adapters в `serve/hy3.py`: call/result IDs,
   text blocks, `no_think/low/high`, `preserved_thinking`, точные ошибки на
   unsupported options. Определять старт reasoning по фактическому Hy3 prefix.
2. Подключить Hy3 к Service только через отдельный профиль; точные control-token
   spellings и EOG120025 проверять до запуска. Не менять default других моделей.
3. Проверить оба API через loopback JSON/SSE: tools/result/answer, stop/length,
   usage, cancellation/disconnect и следующий запрос. Затем smoke полного GGUF.
4. Сохранить P1 baseline для P3 кэша/конвейера. Настоящий driver OOM/device loss
   и внешний pressure остаются отдельными непроверенными режимами; тестовая
   имитация доступной памяти не считается hardware stress test.

### Архивная точка перед HY3-04

1. Закрыть оставшиеся P1.4 failure paths: управляемые ошибки reader/allocations,
   достижение бюджета без реального заполнения всей RAM, отмена/drain, следующий
   запрос и выгрузка. Отчёт должен различать ожидаемый отказ и повреждение runtime.
2. Добавить потоковый reasoning/tool parser Hy3 по P2.3: разрывы тегов и UTF-8,
   несколько вызовов, JSON arguments, malformed/incomplete output и round-trip.
3. Подключить отдельный `serve/hy3.py` к проверенному pipe: оба API, JSON/SSE,
   usage/finish_reason, отмена и повтор. Начальный bind `127.0.0.1`.
4. Сохранить этот P1 baseline для дальнейшего P3: GPU expert cache и overlap
   должны уменьшать измеренные **4,865 ГиБ H2D на decode-шаг**, сохраняя parity.
   Проверять бюджет перед каждым тестом; чужие процессы не останавливать.

### Архивная точка перед HY3-03

1. Создать `strata-hy3` native pipe engine. Использовать строгий loader contract
   до weight allocations и отдельно собранную зависимость Hy3.
2. Реализовать синхронный selected-copy для routed gate/up/down; dense/shared,
   attention, router и все остальные model matmuls выполнять на GPU.
   Начать с MTP-off, без adaptive cache/pipeline/sessions.
3. Взять bounded reader/pinned transport из `backends/common`, добавить Hy3
   диапазоны и mixed-quants. Учитывать main routed116,745 ГиБ, non-routed8,604 ГиБ;
   весь main payload125,349 ГиБ нельзя одновременно разместить в RAM.
4. Сверить logits/greedy IDs с native-графом на тех же квантах и токенах;
   затем измерить load, TTFT, prefill/decode и H2D bytes на полном файле.
5. Перед запуском проверить свободные RAM/VRAM и соблюдать общий лимит95%,
   не останавливая чужие процессы. Скорость Hy3 ещё не измерена.

### Архивная точка перед HY3-02

1. Сохранить текущий CPU baseline и build manifest. Добавить отдельную CUDA
   конфигурацию `build-local/hy3-cuda`, не переключать working builds GLM/Step.
2. Реализовать P0.4 fixtures для всех local mixed quants и правильных expert
   strides, sigmoid+selection bias, top-8/norm/scale, dense/shared FFN,
   Q/K norm до NeoX RoPE и full GQA. Сравнить с CPU/F32 reference.
3. Проверить, нужны ли correctness patches TF32/routed-strides из Step на
   выбранных CUDA путях; переносить только с source hashes и failing/passing fixture.
4. Проверить main/MTP KV filters, hidden-state convention, rollback/restore
   на маленьком синтетическом Hy3. Существующий load-flags patch уже проверен
   в compiled no_alloc; вычислительную корректность он не подтверждает.
5. После P0.4/P0.5 перейти к P1 sync selected-copy на полном файле. До allocations
   заново проверить свободные RAM/VRAM и соблюдать global95%, не останавливая
   чужие процессы. Inference tokens/s появятся только на этом этапе.

### Архивная точка перед HY3-01

1. Создать `tools/inspect_hy3_gguf.py` поверх существующего `GGUFFile` и
   `tools/hy3_loader_contract.py`. Сохранить `docs/hy3/HY3_INSPECTION.json`:
   полный directory, metadata summaries, bytes по группам, header/template hashes.
2. Проверить контракт **для всех 1298 тензоров**, обязательность dense0,
   MoE1..79, MoE80, NextN80, общих embedding/output/norm. Сравнить с локальным
   `hy-v3.cpp`; не маскировать отсутствующие веса его `NOT_REQUIRED` flags.
3. Добавить CPU fixtures для неверной архитектуры, main/MTP boundary,
   shapes/quants, duplicate/overlap/truncated ranges и 64-bit offsets.
4. Создать изолированный `backends/hy3/CMakeLists.txt`, закрепить проверенный
   archive SHA и loader hash, распаковывать только в собственный build directory.
   Сначала собрать vocabulary-only oracle: это не требует загрузки весов на GPU.
5. Сверить BPE, BOS/EOS/EOG и template. Особое внимание `hunyuan-dense`,
   `:opensource` tokens, `no_think/low/high` и placeholder на ID120026.
6. Обновить этот статус командами, PASS/FAIL, manifest и точкой продолжения.
   Следом — P0.4/P0.5 CUDA fixtures и аудит пропуска MTP, затем P1.

Этот список выполнен в HY3-01 в объёме header/loader/tokenizer/template.
В HY3-02 добавлены числовые GPU проверки; в HY3-03 — generation engine и full-model baseline.

## Подтверждённый журнал

### PREP-01 — 2026-10-06 — Инспекция и план

**Действия:** прочитаны инструкции репозитория, план/статус GLM и Step,
общий transport, candidate loader, локальные GGUF metadata/tensor directory.
Проверены архив dependency, GPU total и доступный Windows объём RAM.
Изучены официальный config Tencent, публикация AngelSlim и upstream Hy3 PR.

**Результат:** структурные проверки прошли; main/MTP разделены,
бюджеты и ограничения отражены в плане. Вычисления на GPU не запускались,
tensor payloads не читались и не хэшировались. Веса не скачивались.

**Воспроизведение header-проверки**, PowerShell из корня репозитория:

```powershell
@'
from pathlib import Path
from collections import Counter
import hashlib
import re
from tools.gguf_reader import GGUFFile

p = Path(r'H:\models\hy3\Hy3-Q3_K_M-mtp.gguf')
g = GGUFFile(p)
size = p.stat().st_size
assert g.version == 3 and g.alignment == 32
assert g.metadata['general.architecture'] == 'hy_v3'
assert g.metadata['hy_v3.block_count'] == 81
assert g.metadata['hy_v3.nextn_predict_layers'] == 1
assert len(g.tensors) == len({t.name for t in g.tensors}) == 1298
previous_end = 0
for t in sorted(g.tensors, key=lambda t: t.offset):
    n = t.expected_bytes()
    assert n is not None and n > 0, t.name
    assert t.offset % g.alignment == 0, t.name
    assert t.offset >= previous_end, t.name
    assert g.data_start + t.offset + n <= size, t.name
    previous_end = t.offset + n
assert g.data_start + previous_end == size

groups = Counter()
for t in g.tensors:
    match = re.match(r'blk\.(\d+)\.', t.name)
    scope = 'mtp' if match and int(match[1]) >= 80 else 'main'
    kind = 'routed' if '_exps.weight' in t.name else 'other'
    groups[f'{scope}_{kind}'] += t.expected_bytes()
expected = dict(main_routed=125354115072, main_other=9238637824,
                mtp_routed=1717567488, mtp_other=139151104)
assert dict(groups) == expected, groups
assert size == 136454632928
assert (g.header_end, g.data_start) == (5161421, 5161440)
assert sum(groups.values()) == 136449471488
with p.open('rb') as stream:
    header_sha = hashlib.sha256(stream.read(g.header_end)).hexdigest()
assert header_sha == 'f3307357f0b6ab163f188f7d19ba57d2960ca70a7491b511b23b2745a9500ca9'
template_sha = hashlib.sha256(g.metadata['tokenizer.chat_template'].encode('utf-8')).hexdigest()
assert template_sha == '7fc351fee674c13754656ba7f33a3ca426bfb7231039f48444360a3f3c5ecf3e'
print('PASS header/ranges; bytes:', dict(groups))
print('Tensor types:', dict(Counter(t.type_name for t in g.tensors)))
print('Header SHA-256:', header_sha)
print('Template SHA-256:', template_sha)
'@ | python -X utf8 -
```

Это проверка зафиксированного локального файла. Она намеренно не является
универсальным Hy3 loader contract и не проверяет содержимое матриц.

**Проверка документации:** приведённый Python-код повторно выполнен из этого
Markdown, exit0, `PASS header/ranges`; размеры групп и оба хэша совпали.
Локальные ссылки, парность code fences и отсутствие trailing whitespace — PASS.
Арифметика main+MTP, top-8 H2D и KV оценок — PASS. Это не inference-тесты.

### HY3-01 / P0.1–P0.3, часть P0.5/P2 — 2026-10-06 — Contract и compiled oracles

**Основание:** Strata `9fd4153e372d75dd638eaf90c9c1f1f180dc6057`, локальный
GGUF с тем же header hash. Изолированная Release/Ninja CPU-сборка MSVC
19.44.35222.0 из закреплённого архива Unsloth; CUDA не включалась.
[Build manifest](HY3_BUILD_MANIFEST.json).

**Что изменено:**

- [Inspector](../../tools/inspect_hy3_gguf.py) и
  [loader contract](../../tools/hy3_loader_contract.py): полный tensor directory,
  main/MTP, mixed quants, byte ranges/strides, hashes и проверка обязательных весов.
  Неизвестные архитектурные metadata и неподдержанные layouts отклоняются.
- [Изолированный backend](../../backends/hy3/README.md): tokenizer/template
  oracles и настоящий compiled loader в `no_alloc=true`. Он проверяет,
  что dummy weight buffers нулевые, не читает payloads и не строит inference context.
- Python tokenizer поддерживает `hunyuan-dense` с ordered splits из pinned
  oracle. Добавлен только Hy3-specific пропуск отсутствующего CR символа после BPE.
- Отдельные corpus/checkers для tokenizer, шаблона и регистрации весов;
  CPU fixtures проверяют invalid metadata/shapes/quants, отсутствие MTP-норм,
  bias naming, duplicate/overlap/truncation, >4 ГиБ offsets и защиту output paths.

**Результаты:**

| Проверка | Результат |
|---|---|
| Static admission локального файла | PASS, 1298 тензоров, все диапазоны и формы |
| Python unit/regression tests | PASS, 31 тест, включая Step/GLM и Qwen digit split |
| CTest version checks | PASS, 2/2 |
| Tokenizer vs native oracle | PASS, **2194** случая, 0 несовпадений IDs/decoded bytes |
| Native Jinja vs Python + token IDs | PASS, **144** случая, 0 несовпадений |
| Compiled loader, MTP on | PASS, 1278 main + 20 MTP, точные logical byte totals |
| Compiled loader, MTP off после patch | PASS, те же1278 main, **0 MTP**, dummy weight buffers0 |
| GPU logits / inference / tokens/s | SKIPPED: CUDA runtime ещё не реализован |

**Найденные особенности и исправление:**

1. В локальном vocabulary нет отдельного byte-mapped `\r` (`č`). Native
   tokenizer пропускает его; это не lossless round-trip. Корпус содержит
   244 случая с CR, и отчёт явно учитывает ожидаемую потерю этого символа.
   LF, остальные control bytes и варианты tokenizer других моделей сохранены.
2. Native EOG set — **[120025]**. ID120026 из публичного config здесь placeholder,
   а не дополнительный stop. BOS добавляет шаблон, oracle не дублирует его.
3. Исходный loader при `load_mtp=false` продолжал регистрировать **9** optional
   MTP tensors, **1 740 784 384 байта ≈1,621 ГиБ**. Это подтверждено запуском
   compiled no_alloc, не только чтением исходников.
4. [MtpLoadFlags.cmake](../../backends/hy3/MtpLoadFlags.cmake) генерирует отдельную
   translation unit и передаёт skip flag optional FFN/NextN tensors. Проверка
   повторена на обоих режимах. Trunk byte totals не изменились; MTP-on по-прежнему
   регистрирует все20. Generated source SHA-256:
   `0a40aca60138dde0f30ffc0eae0f8fbcacabebc2d7803afaadf69b580870d60d`.
   Устранено лишнее включение весов в план загрузки; реальная экономия resident
   RAM/VRAM и correctness графа пока не измерялись.

**Артефакты:** [инвентаризация](HY3_INSPECTION.json),
[исходная регистрация](HY3_LOADER_UPSTREAM_VALIDATION.json),
[регистрация после исправления](HY3_LOADER_VALIDATION.json),
[tokenizer validation](HY3_TOKENIZER_VALIDATION.json),
[template validation](HY3_TEMPLATE_VALIDATION.json).
В исходном отчёте поле `status=pass` означает успешную регистрацию в no_alloc,
а не корректный пропуск MTP: его `load_mtp=false` запись сохраняет найденные9 тензоров.

**Воспроизведение:** сборку выполнять в MSVC2022 developer shell с Windows SDK.
Обычный shell первоначально не находил `rc/mt`; среда `vcvars64.bat` исправила
сборку. Смена глобальных настроек Git или установка зависимостей не требовалась.

```powershell
cmake -S backends/hy3 -B build-local/hy3-oracles -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_HY3_ARCHIVE=C:/work/git/my-repos/Strata/build-local/llama-glm-86ebfef2.tar.gz
cmake --build build-local/hy3-oracles --target strata-hy3-tokenizer strata-hy3-template strata-hy3-loader -j 4
ctest --test-dir build-local/hy3-oracles --output-on-failure
python -m unittest tools.test_hy3_gguf tools.test_hy3_tokenizer tools.test_step35_tokenizer tools.test_glm5next_tokenizer
python tools/inspect_hy3_gguf.py --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --output docs/hy3/HY3_INSPECTION.json
python tools/check_hy3_loader.py --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --oracle build-local/hy3-oracles/bin/strata-hy3-loader.exe --output docs/hy3/HY3_LOADER_VALIDATION.json
python tools/check_hy3_tokenizer.py --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --oracle build-local/hy3-oracles/bin/strata-hy3-tokenizer.exe --output docs/hy3/HY3_TOKENIZER_VALIDATION.json
python tools/check_hy3_template.py --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --oracle build-local/hy3-oracles/bin/strata-hy3-template.exe --tokenizer-oracle build-local/hy3-oracles/bin/strata-hy3-tokenizer.exe --output docs/hy3/HY3_TEMPLATE_VALIDATION.json
```

Все финальные команды завершились exit0. Стартовый P0.3 закрыт только для
CPU oracles; CUDA-сборка, граф, KV/rollback, native generation engine, API и
измеренный профиль остаются следующими этапами. Не включать Hy3 в production
список моделей по результатам этих проверок.

**Следующий шаг:** HY3-02 — CUDA fixtures и числовое сравнение графа;
подробная точка продолжения приведена выше.

### HY3-02 / P0.4–P0.5 — 2026-10-06 — CUDA kernels, граф и KV/MTP state

**Основание:** тот же repository HEAD `9fd4153e372d75dd638eaf90c9c1f1f180dc6057`,
изменения HY3-01/HY3-02 ещё без commit. Архив и исходный loader не изменялись.
Полный локальный GGUF в этой итерации не читался. Тестовые GGUF создаются
в новом каталоге `build-local/hy3-tests` и содержат только синтетические веса.

**Сборка:** отдельный `build-local/hy3-cuda`, MSVC19.44.35222.0, CUDA13.0.48,
Ninja/Release, architecture120, RTX5090/driver581.80.
Сохранён [CUDA manifest](HY3_CUDA_BUILD_MANIFEST.json); CPU baseline сохранён.
Generated patches: `hy3-mtp-load-flags`, `cuda-f32-mmf-respect-tf32-override`,
`cuda-routed-input-strides`. Каждый патч проверяет hash исходника и создаёт
собственную translation unit в build directory.

**Что изменено:**

- `check_kernels.cpp`: F32 и все7 квантованных типов локальной модели,
  192 experts/top8, batch1/4/17, обычный/routed matmul, broadcast/per-route input,
  compact/padded strides. Scalar reference использует распакованные веса и
  double accumulation; CPU reference запускается явно, CUDA fallback отсутствует.
  Router проверяет sigmoid, selection bias, выбранные IDs, нормировку исходных
  вероятностей и множитель2,826.
- `synthetic_hy3.hpp` и `check_graph.cpp`: два основных блока и один MTP,
  dense0/MoE1/MTP2, width256, head_dim128, Q4/KV1, experts16/top8, FF256,
  vocab64, full F32 KV, context2048, flash attention выключен.
  F32 и mixed GGUF используют native loader и граф Hy3. Mixed включает
  Q8_0/Q6_K, основной IQ3_XXS/IQ4_XS и MTP Q3_K/Q4_K.
- Scalar проверки Q/K RMSNorm→NeoX RoPE, causal full GQA, dense/shared/routed
  SwiGLU и суммы shared+routed. MTP: enorm/hnorm, concat(e,h), eh_proj,
  общий output head, post-final-norm hidden; проверена передача hidden
  основного графа и повторного MTP шага.
- Main KV содержит только слои0/1, MTP KV — только слой2. В обоих контекстах
  save→A→restore→B→restore→A и removal/replay дают побитово одинаковые
  logits/hidden. MTP-off не загружает тензоры блока2; основной граф совпадает
  побитово с MTP-loaded вариантом. GPU scheduler audit: **0 CPU math nodes**.
- `tools/check_hy3_cuda.py`: новый output directory, hashes бинарника/manifest,
  логи и sampled memory; контроль общей RAM/VRAM95% с остановкой только
  созданного тестового процесса. В проверках запас памяти не исчерпан.

**Исправления подтверждены до/после:**

| CUDA-конфигурация | Результат | Причина |
|---|---|---|
| Без двух CUDA patches, TF32 override=0 | 136/150, exit1 | Custom F32 MMF игнорирует override; 14 расхождений |
| Только strict-F32 | 146/150, exit1 | 4 padded routed F32 cases неверно вычисляют физический шаг входа |
| Strict-F32 + routed-strides | **150/150, exit0** | Совпадение с reference в заданных допусках |
| Native main/MTP graph/state | **86/86, exit0** | F32 64 checks, mixed22; no CPU fallback |

Артефакты: [до CUDA patches](HY3_CUDA_KERNELS_UNPATCHED.json),
[только strict-F32](HY3_CUDA_KERNELS_STRICT_ONLY.json),
[исправленные kernels](HY3_CUDA_KERNELS_VALIDATION.json),
[graph/state](HY3_CUDA_GRAPH_VALIDATION.json).
Все содержат параметры, hashes бинарника/manifest, допуски, ошибки и memory samples.

**Числовые результаты:** максимальная ошибка F32 logits CPU/CUDA —
`8,35e-7` main и `3,58e-7` MTP. Для mixed logits — `0,02416` main и
`0,01142` MTP, NMSE соответственно `1,693e-4` и `7,188e-5`.
Допуски заданы до первого запуска: mixed abs≤0,03 logits/≤0,12 hidden,
NMSE≤2e-3; CPU и CUDA используют разные способы квантизации активаций.
Это числовое сравнение с допуском, не побитовый parity полного checkpoint.
Restore/replay и MTP-off/on требуют точного совпадения и проходят с F32/mixed.

**Память:** в финальном kernel прогоне sampled global VRAM максимум2872 МиБ,
global RAM23,104 ГиБ; process peak working set567,676 МиБ. В финальном graph
прогоне —3120 МиБ,21,423 ГиБ и565,133 МиБ соответственно. Global показатели
включают другие процессы; опрос RAM раз в0,5с, GPU раз в2с может пропустить
краткий пик. Эти числа не оценивают память inference полного GGUF.

**Воспроизведение:** команды CUDA configure/build и оба запуска приведены в
[backends/hy3/README.md](../../backends/hy3/README.md#cuda-admission).
Локальные финальные каталоги: `build-local/hy3-tests/kernels-patched` и
`build-local/hy3-tests/graph-final`. Старые неудачные прогоны сохранены.
Обе CUDA опции включены по умолчанию в новой CUDA-конфигурации; для существующего
cache README явно задаёт ON. Регрессии: **31 Python test PASS**, CPU CTest **2/2**.

**Не закрыто:** полный GGUF, pipe engine, выбранные копии экспертов,
Flash Attention/F16 KV, длинный контекст, speculative accept/reject driver,
API, production sessions и производительность. Синтетический MTP graph не
является готовым MTP-ускорением Strata. **Токенов/с пока нет.**

**Следующий шаг:** HY3-03/P1 — sync selected-copy baseline полного GGUF,
MTP-off, с logits/greedy parity и измерением скорости/памяти.

### HY3-03 / P1.1–P1.3, часть P1.4 — 2026-10-06 — Полный GGUF и sync baseline

**Основание:** Strata `51125bdbc03f700608584d650e94666bec18c16b`, рабочее дерево
чистое в начале итерации; изменения HY3-03 пока без commit. Тот же GGUF и
header/template hashes. RTX5090, driver581.80, 125,555 ГиБ доступной системе RAM,
MSVC19.44.35222, CUDA13.0.48, sm_120, Release/Ninja. Dependency — Unsloth
`86ebfef2`, отдельная сборка `build-local/hy3-cuda`.
[Manifest с generated source hashes](HY3_RUNTIME_BUILD_MANIFEST.json).

**Что изменено:**

- `strata-hy3.exe`: stdin/stdout pipe ENC/GEN/STOP/QUIT, native sampler,
  progress и request metrics. Каждый GEN начинает с чистого KV и sampler;
  session ID обеспечивает изоляцию, но не сохраняет KV между запросами.
- Native contract проверяет metadata, все1298 тензоров, main/MTP boundary,
  shapes/quants/ranges до загрузки весов. MTP-off исключает блок80.
  Non-routed weights находятся на GPU; математика CPU запрещена scheduler audit.
- Новый путь selected ranges: native file read через общий `expert_file.hpp`
  → pinned16 МиБ → GPU. GPU fence перед перезаписью staging; не более одного
  чтения в полёте. Проверяются размеры/offsets и MMQ padding до512 байт.
  Whole-expert copies отклоняются до копирования. Кэша и overlap пока нет.
- Loader использует demand mmap без предварительного чтения всего файла.
  Existing GLM `HostWorkingSetBudget` ограничивает mapped resident pages
  по system target93%, учитывая другие процессы. Global RAM/VRAM guard95%; VRAM
  читается через общий NVML helper по PCI identity, а не process view CUDA/WDDM.
- Отдельный model checker сохраняет prompt IDs, logits hashes, метрики,
  memory samples, проверяет сравнение режимов, повтор, отмену и свежий запрос.
  Исправлено закрытие тестового pipe после принудительного завершения child:
  broken pipe больше не скрывает исходную ошибку и отчёт memory monitor.
- Добавлены явные CMake header dependencies: локальный MSVC с русскими include
  diagnostics пропускал rebuild после изменения header. Архив dependency,
  исходный third_party и действующие engine/profile других моделей не менялись.

**Проверки:**

| Проверка | Результат |
|---|---|
| Native preallocation admission полного GGUF | PASS; blocks81/vocab120832, точные main/MTP bytes, weight allocations0 |
| Runtime: resident/native/pinned, bytes и logits | **19/19 PASS**, exit0; F32/mixed, batch1/4/17, transfer >16 МиБ |
| Pipe: greedy/seeded repeat, malformed input, STOP/QUIT и повтор | **22/22 PASS**, exit0 |
| Полный GGUF: native selected-copy vs pinned-file | **PASS**, exit0; 6 prompts + 2 повтора на режим, все IDs и logits побитово одинаковы |
| Полный GGUF: STOP → свежий запрос, unload/reload | PASS в обоих режимах; process exit0, memory monitor без ошибок |
| CUDA kernel regression с runtime patch | **150/150 PASS**, exit0 |
| Synthetic main/MTP graph/state regression | **86/86 PASS**, exit0 |
| Python contract/tokenizer/Step/GLM + pipe cleanup regression | **32 tests PASS** |
| CPU CTest version checks | **2/2 PASS** |

Артефакты: [runtime](HY3_RUNTIME_VALIDATION.json), [pipe](HY3_PIPE_VALIDATION.json),
[полный GGUF](HY3_MODEL_VALIDATION.json),
[kernel regression](HY3_RUNTIME_KERNELS_REGRESSION.json),
[graph regression](HY3_RUNTIME_GRAPH_REGRESSION.json).
Полный отчёт содержит SHA-256 проверенного engine
`d6fd9dbffd4985feea1b925d58dcb3550c4b580e687ca07e550bf6742d540600`.
Raw logits и stderr остаются в `build-local/hy3-tests/model-bounded`.

**Числовая оговорка:** с включёнными candidate CUDA fusions fixture F32/batch4
дал resident/native max_abs **0,00105551**; pinned/native при этом совпал.
[Диагностический FAIL](HY3_RUNTIME_FUSION_DIAGNOSTIC.json) сохранён.
С `GGML_CUDA_DISABLE_FUSION=1` все19 runtime checks проходят с точными bytes/logits.
Конкретная fusion ещё не изолирована; runtime принудительно выключает fusions,
TF32 и оставляет `GGML_OP_OFFLOAD_MIN_BATCH=1`. P0 graph callbacks меняют условия
планирования и не заменяют эту проверку без evaluation callbacks.

**Измерения полного GGUF:** template `no_think`, greedy temperature0,
context2048, batch/ubatch17, F32 KV, FA off, MTP/cache/pipeline/fusions off.
Сначала native, затем pinned; OS file cache не очищался, порядок не чередовался.
Это correctness-baseline с наблюдениями времени, **не контролируемый speed A/B**.

| Запрос | Prompt / output tokens¹ | Native TTFT, с | Pinned TTFT, с | Native decode, токенов/с² | Pinned decode, токенов/с² |
|---|---:|---:|---:|---:|---:|
| Русский, 2+2 | 34 / 2 | 16,032 | 10,144 | 1,828 | 1,330 |
| Английский, пять простых чисел | 25 / 15 | 6,157 | 8,955 | 2,005 | 1,305 |
| Китайский, столица | 24 / 3 | 5,246 | 8,538 | 2,194 | 1,299 |
| Python, сложение двух чисел | 28 / 24 | 6,156 | 9,953 | 2,139 | 1,172 |
| Числа, 123+456 | 27 / 2 | 5,230 | 11,043 | 2,248 | 1,235 |
| Длинный prompt, цвет лодки | 362 / 2 | 143,773 | 134,579 | 1,044 | 1,283 |

¹ Включая EOS, reasoning выключен. Ответы: `4`, `2, 3, 5, 7, 11.`, `北京。`,
функция `add`, `579`, `blue`. Это проверка этих запросов, не оценка качества модели.
² `decode_steps / decode_forward_seconds`: первый token получен из prefill
и не включён, последующий EOS включён. В коротких ответах всего один-два decode
шага. Для en+code вместе37 шагов: native **2,086**, pinned **1,219 токена/с**.
Повторы русского после длинного запроса/отмены: native1,179/1,239,
pinned1,316/1,320; все logits совпали с первым русским ответом.

Load до READY: native **2,914 с**, pinned **4,822 с**; demand loading означает,
что сюда не входит чтение всех routed weights. Prefill длинного prompt:
native143,770 с /2,518 токена/с, pinned134,576 с /2,690 токена/с.
Каждый decode-шаг переносит около **4,865 ГиБ H2D** выбранных экспертов.
Для pinned source-read bytes равны H2D bytes; ОС может обслужить чтения из
file cache, поэтому это не измерение физического SSD traffic.

| Sampled memory peak³ | Native | Pinned |
|---|---:|---:|
| Global physical RAM, ГиБ | 117,094 | 36,006 |
| Global VRAM, МиБ | 14 022 | 14 336 |
| Process working set, ГиБ | 99,032 | 9,257 |
| Process private commit, ГиБ | 12,635 | 12,651 |
| Собственный pinned staging, МиБ | 0 | 16 |

³ Global значения включают другие процессы, RAM опрашивается примерно раз0,5с,
GPU раз2с; короткий пик может быть пропущен. Указанный staging — только новый
transport buffer, не сумма всех внутренних driver allocations. mmap virtual size
не равен working set. Оба прогона уложились в95%; неиспользованная VRAM оставлена
для будущего полезного expert cache. Рабочий набор native ограничен93%-target,
иначе mapped pages при длинном prompt вытесняют доступную RAM.

**Воспроизведение:** configure/build и команды в
[README](../../backends/hy3/README.md#windows-synchronous-baseline).
Финальные каталоги: `runtime-bounded`, `pipe-bounded`, `model-bounded`,
`runtime-kernels`, `runtime-graph` внутри `build-local/hy3-tests`.
Full-model команда:

```powershell
python tools/check_hy3_model.py --engine build-local/hy3-cuda/bin/strata-hy3.exe --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --cuda-bin build-local/cuda-13.0/bin/x64 --output-dir build-local/hy3-tests/model-new --predict 24
```

**Не закрыто:** injected OOM/read-error recovery, внешний pressure, контекст
длиннее362 prompt tokens, F16 KV/FA, HTTP/UI/parser, persistent sessions,
expert cache/pipeline и speculative MTP. Новый синхронный transport пока не даёт
ускорения в этом прогоне; его назначение — проверенная основа для P3.
Оптимальные defaults ещё не выбраны. **Следующий шаг:** HY3-04, оставшиеся
P1.4 failure/recovery fixtures, затем P2 parser/API с сохранением этого baseline.

### HY3-04 / P1 recovery, P2.2–P2.3 — 2026-10-06 — Отказы, parser и tool round-trip

**Основание:** тот же HEAD `51125bdbc03f700608584d650e94666bec18c16b` с
незакоммиченными изменениями HY3-03; они сохранены. Та же Release/Ninja сборка,
MSVC19.44.35222, CUDA13.0.48/sm_120, RTX5090/driver581.80 и checkpoint.
[Build manifest](HY3_RECOVERY_BUILD_MANIFEST.json). Скоростные настройки
и архитектура вычислений не менялись: MTP/cache/pipeline/fusion off, F32 KV.

**Отказы runtime и найденная ошибка:**

- Добавлено13 проверок к прежним19. Pinned allocation возвращает injected
  `cudaErrorMemoryAllocation` до upload; после отказа тот же context повторяет
  исходные logits. На следующем selected read после успешного H2D временно
  заменяется handle только синтетического fixture: настоящий Win32 `ReadFile`
  отказывает, handle восстанавливается, запрос повторяется с точными logits.
- Отмена после частичного upload, simulated RAM/VRAM availability0, заведомо
  невозможный GPU reserve, пустой registry после unload и reload — PASS.
  Неполная копия не публикуется как успешная. Test caps только уменьшают
  доступность памяти, не отключают реальные лимиты и не заполняют RAM/VRAM.
  Эти функции не доступны через CLI, env или пользовательский pipe.
- Тест RAM admission сначала завершил процесс с **exit3221226356 /0xc0000374**.
  Throwing helpers были объявлены `extern "C"`; при MSVC `/EHsc` такой вызов
  считается не бросающим исключение, что нарушало нужный unwind path.
  Переведены на C++ linkage. Тот же тест теперь возвращает ожидаемый отказ,
  следующий запрос и выгрузка проходят. [Диагностика до исправления](HY3_RECOVERY_PRE_FIX.json)
  хранит exit code и binary hash; процесса хватило только до RAM pressure,
  поэтому отдельного `runtime-report.json` у этого неудачного прогона нет.
- [Финальный runtime report](HY3_RECOVERY_VALIDATION.json): **32/32 PASS**, exit0.
  Partial ReadFile/cancel возвращали управление примерно за1 мс в tiny fixture;
  это не гарантия latency при задержанном SSD I/O. Registry/source lifetime
  и reload проверены. [Pipe regression](HY3_RECOVERY_PIPE_VALIDATION.json): **22/22 PASS**.

Реальное исчерпание VRAM внутри driver, device loss и внешнее давление памяти
не воспроизводились. Успешная инъекция pinned OOM не доказывает восстановление
после любого CUDA allocation failure. Прочие модели/shared transport не менялись.

**Template и output parser:**

- Новый [serve/hy3.py](../../serve/hy3.py) и точный fixture10223 bytes.
  Renderer проверяет SHA-256, принимает native `no_think/low/high`, правила
  replay/raw-last-assistant/training/fallback, нормализует JSON-string arguments
  перед `.items()` без изменения исходных messages. Неизвестные опции отвергаются.
- Потоковый parser разбирает reasoning и namespace `:opensource`, сохраняет
  raw strings и whitespace; typed arguments проверяет как конечный JSON нужного
  базового типа. Duplicate keys/arguments, неизвестные functions, ошибочные и
  незавершённые группы остаются текстом. Весь tool group проверяется до выдачи
  первого вызова; tool markup внутри reasoning не становится действием.
- Проверены разрывы каждого тега и UTF-8, несколько calls, JSON с delimiter
  strings внутри кавычек, stop в reasoning/tool, перекрывающиеся stop sequences,
  финальные неполные теги, повторный finish и ограничение group buffer1 МиСимвол.
  Нативный формат не экранирует raw values; неоднозначная пара value delimiter
  и следующего key/call delimiter в строках history отвергается. Полная JSON
  Schema validation не добавлена; raw strings задаются string schemas.
- [17 CPU tests /7327 fragmentation sequences](HY3_PARSER_VALIDATION.json) — PASS.
  Новый runtime renderer [совпал с native Jinja и BPE](HY3_ADAPTER_TEMPLATE_VALIDATION.json)
  во всех **144** случаях. Python regression: **100 tests PASS**, включая
  Step/GLM/DeepSeek adapters, Hy3 contract/tokenizer и pipe cleanup.

**Проверка полного GGUF:** [tool dialogue report](HY3_TOOL_DIALOGUE_VALIDATION.json).
`tools/check_hy3_dialogue.py` использует полный read-only checkpoint, native
pipe и локальный фиксированный result stub; HTTP и внешние tools не вызываются.
Template `no_think`, greedy temperature0, pinned sync16 МиБ, context2048/batch17.

1. User просит температуру в Paris через `get_temperature`. Prompt211 токенов,
   output19 с EOS. Модель выдала ровно один полный вызов с `{"city":"Paris"}`;
   parser вернул typed tool event без постороннего content.
2. В history добавлен этот call и локальный result
   `{"city":"Paris","temperature_c":17}` с тем же call ID. Prompt253 токена,
   output3 с EOS: **`17 C`**, без повторного tool call.
3. Для обоих фактических prompts строки и token IDs совпали с compiled
   native template/tokenizer oracles. Оба finish=`stop`, engine exit0,
   memory monitor без ошибок. Это тест протокола на фиктивной температуре,
   а не запрос реальной погоды.

Prefill75,647/96,621с; generation13,463/1,581с соответственно. Это единичная
correctness-проверка, не новый benchmark или ускорение. Sampled global peak:
RAM **33,437 ГиБ**, VRAM **14 258 МиБ**; process private commit12,648 ГиБ.
Монитор RAM≈0,5с/GPU≈2с может пропустить короткий пик; лимит95% не пересечён.
Проверенный engine SHA-256:
`f5cf4dd25611a9173e3cb5d2b4828204babbb959b71f73b99fbb46edb02180c7`.
Исторические HY3-03 reports/binary hashes сохранены и относятся к прежней сборке.

**Воспроизведение**, после обычной CUDA build из README:

```powershell
python tools/check_hy3_cuda.py --build build-local/hy3-cuda --kind runtime --cuda-bin build-local/cuda-13.0/bin/x64 --output-dir build-local/hy3-tests/recovery-new
python -m unittest serve.test_hy3 serve.test_step35 serve.test_glm5next serve.test_deepseek tools.test_hy3_gguf tools.test_hy3_tokenizer tools.test_hy3_engine tools.test_step35_tokenizer tools.test_glm5next_tokenizer
python tools/check_hy3_template.py --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --oracle build-local/hy3-oracles/bin/strata-hy3-template.exe --tokenizer-oracle build-local/hy3-oracles/bin/strata-hy3-tokenizer.exe --runtime-adapter --output docs/hy3/HY3_ADAPTER_TEMPLATE_VALIDATION.json
python tools/check_hy3_dialogue.py --engine build-local/hy3-cuda/bin/strata-hy3.exe --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --cuda-bin build-local/cuda-13.0/bin/x64 --template-oracle build-local/hy3-oracles/bin/strata-hy3-template.exe --tokenizer-oracle build-local/hy3-oracles/bin/strata-hy3-tokenizer.exe --output-dir build-local/hy3-tests/dialogue-new
```

Финальные локальные каталоги: `runtime-recovery-03`, `pipe-recovery-01`,
`dialogue-01` внутри `build-local/hy3-tests`. Неудачные recovery-01/02 сохранены.

**Следующий шаг:** HY3-05 — P2.4/P2.5, request adapters и отдельный профиль,
подключение к Service/HTTP, JSON/SSE/tool round-trip/cancellation. Parser импортируется
отдельно и пока не включён в сервер. UI, sessions, cache/pipeline и MTP остаются TODO.

### HY3-05 / P2.4–P2.5 — 2026-10-06 — Service, HTTP и веб-чат

**Основание:** HEAD `9995b3fd3395cd839869a23a72ccc12ff56a544f`, изменения
этого этапа без commit. C++ engine не менялся, SHA-256
`f5cf4dd25611a9173e3cb5d2b4828204babbb959b71f73b99fbb46edb02180c7`.
Header/template/dependency — те же, что в HY3-04. Текущие Python/JS hashes,
список regression modules и результаты сохранены в
[HY3_HTTP_VALIDATION.json](HY3_HTTP_VALIDATION.json). Старые HY3-04 parser
и model reports сохранены как исторические результаты прежнего Python-кода.

**Изменено:** OpenAI/Anthropic request adapters в `serve/hy3.py`, регистрация
архитектуры в Service, определение начального reasoning по реальному prefix,
проверка special token spellings/types/metadata до engine startup. На этом
tokenizer только EOS120025 завершает генерацию; PAD/placeholder не обрывают ответ.
Tools сохраняют call/result IDs и порядок history; строковые OpenAI arguments
разбираются в конечный JSON object до Jinja. Media, hard reasoning budgets,
чужие template options и некорректные arguments получают HTTP400 до engine.

Режим по умолчанию — `no_think`, доступны `low/high` и `preserved_thinking`.
В Anthropic disabled означает no_think, enabled/adaptive — high;
`output_config.effort` уточняет effort, затем применяются явные template kwargs.
Противоречивые thinking/effort отвергаются. Общие настройки не перезаписывают
явный Anthropic thinking. Другие модели сохраняют прежние ветви нормализации.
В веб-настройках Hy3 показывает Off/Low/High и скрывает Medium/Max.

`tools/prepare_hy3_profile.py` создаёт tokenizer и `hy3.json` только в новом
каталоге, проверяет header и identity сборки. Рабочий локальный профиль:
`build-local/hy3-http-p2/hy3.json`. Настройки: host127.0.0.1, context2048,
batch17, F32 KV, synchronous pinned16 МиБ, temperature0, MTP/cache/pipeline off.
`fit_max_tokens=true` уменьшает только лимит ответа, не обрезает prompt.
Это experimental baseline; ускоренные defaults ещё не подбирались.

**CPU и регрессии — PASS:**

- **276 unittest tests**, 0 failures/errors/skips: Hy3 parser/adapters/profile,
  loopback JSON/SSE и регрессии Step, GLM, DeepSeek, общего Service, lifecycle,
  structured output, detokenizer и tokenizer. Точный список22 modules — в отчёте.
- В Hy3 loopback проверены оба API × JSON/SSE × три effort режима; Unicode,
  usage/EOG, tools/result/answer, stop/length, HTTP400 и следующий запрос,
  disconnect в очереди/prefill/reasoning/partial tool и восстановление.
  Эти сочетания используют scripted engine, а не полный GGUF.
- **144/144** native template/token-ID comparisons для текущего adapter:
  [HY3_HTTP_TEMPLATE_VALIDATION.json](HY3_HTTP_TEMPLATE_VALIDATION.json).
- Два Node checks веб-настроек: Hy3/GLM/Step/Qwen и существующий GLM/Qwen.
  В исходном расширенном прогоне обнаружены два устаревших ожидания в
  `serve/test_glm5next_profile.py`: не учтены существующие pipeline/MTP args
  и ошибочно предполагалось, что memory flags стоят последними. Исправлены
  только тесты; повтор всех276 прошёл. GLM exporter/defaults не менялись.

**Полный GGUF, production Service/StrataEngine, loopback HTTP — PASS:**
[HY3_HTTP_MODEL_VALIDATION.json](HY3_HTTP_MODEL_VALIDATION.json).

1. Вопрос `Сколько будет 2 + 2? Ответь одной цифрой.` прошёл через обе API
   формы и оба transport режима:34 prompt tokens,2 output tokens с EOS,
   ответ4. Все четыре prompts/outputs совпали по IDs.
2. OpenAI JSON вернул один `get_temperature({"city":"Paris"})`,211 prompt /
   19 output tokens с EOS. Его ID и локальный результат17 переданы через
   Anthropic SSE;253 prompt /3 output tokens с EOS, ответ **`17 C`**.
   Оба фактических prompts и полные output IDs точно совпали с HY3-04
   direct-dialogue reference на том же бинарном файле. Внешних tools не было.
3. Клиент отключён во время prefill778-token запроса после первого PP17;
   runtime вернул `cancel`, очередь освободилась через **3,753с** после
   закрытия сокета. Следующий34-token запрос воспроизвёл те же IDs/ответ4.
   Итого7 успешных генераций и1 отмена; engine exit0.
4. /health, /v1/models, /settings и HTML веб-чата отдали HTTP200 и Hy3 capabilities.

На RTX5090, Windows/driver581.80,128GB установленной RAM, указанном GGUF и
greedy no_think: tool call prefill **76,178с**, generation **15,396с**;
continuation prefill **94,639с**, generation **1,604с**. Это correctness-прогон
без performance A/B. Sampled global peak RAM **35,403 ГиБ**, VRAM **14 360 МиБ**,
ошибок95% monitor нет. Интервалы RAM≈0,5с/GPU≈2с не исключают короткие пики.
Сохранён прежний P1 baseline; ускорение этим этапом не заявляется.

**Реальный веб-чат — PASS:** server.main с этим CLI-профилем, browser UI,
temperature0/max_tokens32; вопрос выше дал видимый ответ4 и состояние Idle.
Reasoning не попал в обычный текст. В интерфейсе Off выбран по умолчанию,
Low/High доступны. Запрос/ответ, память и exit codes:
[HY3_HTTP_UI_VALIDATION.json](HY3_HTTP_UI_VALIDATION.json);
[снимок](HY3_HTTP_UI_SMOKE.png). Browser tab и тестовые server/engine закрыты,
оба exit0. UI-скорость на двух output tokens не используется как benchmark.

**Воспроизведение:** команды экспорта, запуска, HTTP и CPU checks находятся в
[backend README](../../backends/hy3/README.md#experimental-http-profile).
Локальный уже созданный профиль запускается так:

```powershell
python -m serve.server --engine strata --config build-local/hy3-http-p2/hy3.json --port 8094
python tools/check_hy3_http.py --profile build-local/hy3-http-p2/hy3.json --reference docs/hy3/HY3_TOOL_DIALOGUE_VALIDATION.json --output-dir build-local/hy3-tests/http-new
python -m unittest serve.test_hy3 serve.test_hy3_http tools.test_hy3_profile
node serve/test_hy3_settings_ui.cjs
python tools/check_hy3_template.py --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --oracle build-local/hy3-oracles/bin/strata-hy3-template.exe --tokenizer-oracle build-local/hy3-oracles/bin/strata-hy3-tokenizer.exe --runtime-adapter --output docs/hy3/HY3_HTTP_TEMPLATE_VALIDATION.json
```

Server и checker запускать по очереди: checker сам создаёт engine и временный
loopback listener. Raw logs: `build-local/hy3-tests/http-01`, UI-runner/report:
`build-local/hy3-tests/ui-01` и `ui-smoke-runner.py` рядом. Для повтора выбирать
новый output directory. Все перечисленные финальные команды завершились с exit0.

**Не закрыто:** full-model low/high/stochastic corpus, production sessions,
F16 KV/Flash Attention, длинный контекст, driver OOM/external pressure,
общий installer и speed defaults. Следующий шаг **HY3-06 / P3.1** — bounded
GPU expert cache: exact logits/IDs, hit/miss/eviction и снижение H2D;
затем P3.2 async pipeline и контролируемые A/B.

### HY3-06 / P3.1 — 2026-10-06 — Bounded GPU matrix cache

**Основание:** HEAD `4336d41a7b25b21db80ea5bfa7d9e0d799d0bd14`, изменения
этого этапа без commit. Новый engine SHA-256:
`8da251920bf0d034f52978192ea1e10fc412896193057355502d981e974858b9`.
В private build добавлен patch marker `hy3-bounded-matrix-cache`;
Unsloth source, основные correctness patches, GGUF и template не менялись.
Предыдущий бинарный файл HY3-05 сохранён в
`build-local/hy3-baseline-p2/strata-hy3.exe` (SHA начинается с `f5cf4dd2`).

**Изменено:** `backends/hy3/cache_runtime.inc` подключает неизменённый
`step35::ExpertCache` к Hy3 scheduler. Ключ состоит из поколения модели,
ID зарегистрированного тензора и expert ID; registry проверяет name/layer,
type, shape, strides, живое file mapping и фактический offset. Новый begin
и release освобождают старые entries. Cache hit копирует веса D2D в scratch,
miss использует прежний file→pinned→GPU путь, затем заполняет cache slot.
Реальный MMQ padding следующего эксперта включён в запись, до512 байт.
Каждая копия завершена до eviction/reuse; overlap пока отсутствует.

Аллокации учитываются с округлением64 КиБ; controller использует decaying
frequency, bounded victim scan и повторное использование равных allocations.
Global NVML free/total ограничивает budget с reserve5%+256 МиБ, RAM сохраняет
target93% и guard95%. Cache OOM пропускает admission; ошибка заполнения
сбрасывает записи. При pressure cache сначала уменьшается, а неизбежный
недостаток памяти отказывает до compute. Реальный driver OOM/device loss
не заявлен как проверенный.

CLI получил `--expert-cache-mib 0..16384`, только для pinned mode; default0
сохраняет baseline. Новый профиль `build-local/hy3-http-cache8/hy3.json`
явно задаёт8192 МиБ. Остальные параметры: context2048, batch17, F32 KV,
temperature0/no_think, pinned16 МиБ; MTP/pipeline/fusion/TF32 off.
Действующие профили других моделей не изменялись. В JSON метриках отдельно
записаны cache hit/miss/eviction/reuse/oom, resident/budget, source/H2D,
D2D hits и D2D cache fill. Source-read bytes не являются SSD traffic.

**Проверки — PASS:**

- [HY3_CACHE_ALLOCATION_VALIDATION.json](HY3_CACHE_ALLOCATION_VALIDATION.json):
  **25/25** CUDA allocator/policy cases, включая simulated pressure/OOM,
  pins, reuse и131200 сравнений victim decisions. Проверяется тот же Step
  controller, собранный внутри изолированного Hy3 build.
- [HY3_CACHE_RUNTIME_VALIDATION.json](HY3_CACHE_RUNTIME_VALIDATION.json):
  **100/100** runtime cases. Включены прежние32 и68 новых: F32/mixed,
  batch1/17, cold/warm/new generation, byte/logit parity, eviction/reuse,
  OOM bypass, незавершённое заполнение, partial read/cancel и recovery,
  simulated RAM/VRAM trim. На полностью прогретых synthetic fixtures H2D=0,
  все observer-проверки GPU bytes совпали с source.
- [HY3_CACHE_PYTHON_VALIDATION.json](HY3_CACHE_PYTHON_VALIDATION.json):
  **44 unittest tests** и6 CLI admission checks. Profile exporter сохраняет
  выбранный cap и отвергает недопустимый до создания файлов; Hy3 HTTP/parser
  регрессии прошли. Этот отчёт также хранит hashes изменённых исходников.
- [HY3_CACHE_MODEL_VALIDATION.json](HY3_CACHE_MODEL_VALIDATION.json):
  полный GGUF, **20 генераций** в sweep0/8/12/16/0 ГиБ: English/code дважды
  в каждом процессе, fresh KV и greedy, cap24 output tokens. Все полные
  logit hashes и token IDs совпали с историческим native baseline HY3-03.
  Каждый nonzero cap дополнительно прошёл STOP и повтор English с exact
  logits/IDs: ещё3 отмены и3 успешных восстановления. Все engine exit0,
  sampled95% monitors без ошибок.
- [HY3_CACHE_HTTP_VALIDATION.json](HY3_CACHE_HTTP_VALIDATION.json):
  полный GGUF с cache8 через production Service/StrataEngine. OpenAI и
  Anthropic JSON/SSE дали ответ `4`; tools вернули `get_temperature(Paris)`
  и продолжение `17 C`, prompt/output IDs точно совпали с HY3-04. Всего
  7 завершённых запросов. Disconnect во время prefill запроса из 778 токенов
  дал DONE cancel и снятие busy через 3,679 с; следующий запрос воспроизвёл
  эталон. Sampled peak RAM **38,237 ГиБ**, VRAM **26 701 МиБ**; guard без
  ошибок, engine exit0, временный loopback listener закрыт.
- [HY3_CACHE_PIPE_VALIDATION.json](HY3_CACHE_PIPE_VALIDATION.json):
  **22/22** synthetic pipe lifecycle checks с cap8 МиБ: A/B/A, повтор seeded
  sampling, ошибки запросов, STOP/recovery, изоляция session keys и QUIT
  во время запроса. Engine exit0; memory guard без ошибок.

**Короткий capacity sweep**, RTX5090/driver581.80, Windows,128GB установленной
RAM, `Hy3-Q3_K_M-mtp.gguf`. В таблице второй проход English/code в каждом
процессе:15+24 output tokens с EOS, **37 decode-шагов** после первых токенов
из prefill. Скорость =37 / сумма decode_forward_seconds. Prompt25/28,
KV очищается на каждый запрос; GPU cache/history сохраняются внутри процесса.
OS file cache не сбрасывался; это последовательный sweep с cache-off до/после,
не рандомизированное длительное сравнение. Точные расчёты:
[HY3_CACHE_BENCHMARK.json](HY3_CACHE_BENCHMARK.json).

| Запрошенный cap, ГиБ | Cache resident в конце, ГиБ | Decode, ток/с | H2D на decode-шаг, ГиБ | Sampled peak VRAM, МиБ | Sampled peak RAM, ГиБ |
|---:|---:|---:|---:|---:|---:|
| 0, до sweep | 0 | 1,360 | 4,865 | 13 917 | 37,463 |
| 8 | 8,000 | **1,821** | 3,077 | 26 519 | 39,719 |
| 12 | 10,416 | 1,803 | 2,835 | 30 230 | 34,862 |
| 16 | 10,394 | 1,762 | 2,773 | 30 233 | 37,939 |
| 0, после sweep | 0 | 1,377 | 4,865 | 14 003 | 33,761 |

Cache8 дал **+33,1% decode throughput** относительно объединённых cache-off
повторов и **−36,8% H2D** на этом corpus. Полное время прогретой пары с prefill:
37,391с вместо45,020/44,610с без кэша. Cap12/16 сократили H2D сильнее, но
не ускорили эту пару; runtime ограничил реальный кэш примерно10,4 ГиБ.
Размеры всех CUDA/driver allocations не равны сумме payload cache slots:
общая VRAM выше. Причина разницы отдельно не профилировалась. Guard использует
NVML total−free (системное reserve может отличаться от nvidia-smi memory.used
в sampled peak колонке); для больших cap request snapshots около30 720 МиБ.

Для следующего этапа выбран **cache8**: лучший результат этого sweep и меньше
VRAM. Это не универсально оптимальный размер; длинные/другие темы и concurrent
memory load потребуют дополнительных замеров. Автоматический default exporter
остался0; быстрый вариант включён только в отдельном экспериментальном профиле.

HTTP-прогон также показал, что ускорение decode не означает сокращения TTFT.
Tool call: prefill **92,011 с**, generation **10,247 с**; continuation:
prefill **102,437 с**, generation **1,594 с**. В историческом HY3-05 без кэша
prefill был 76,178/94,639 с. Это разные последовательные прогоны, не отдельный
контролируемый prefill A/B, но улучшение длинного prefill пока не подтверждено.
Текущий путь разбивает ranges на матрицы и синхронизирует каждую доставку;
batching и отдельную prefill admission policy следует измерить до выбора
общих speed defaults.

**Воспроизведение:** build/CUDA/model commands — в
[backend README](../../backends/hy3/README.md#experimental-http-profile).
Локальный запуск созданного cache8-профиля:

```powershell
python -m serve.server --engine strata --config build-local/hy3-http-cache8/hy3.json --port 8094
python tools/check_hy3_http.py --profile build-local/hy3-http-cache8/hy3.json --reference docs/hy3/HY3_TOOL_DIALOGUE_VALIDATION.json --output-dir build-local/hy3-tests/cache-http-new --allow-engine-change
```

Server и HTTP checker запускать по очереди: checker создаёт свой listener.
Для экспорта на новую сборку использовать новый directory и
`tools/prepare_hy3_profile.py ... --expert-cache-mib 8192`. Откат доставки —
cap0 на этой же сборке, без смены модели и template. Старые profile/checksum
HY3-05 остаются историческими; helper проверяет checksum текущего exe.
Raw run directories: `cache-unit-01`, `cache-runtime-01`, `cache-model-01`,
`cache-http-01`, `cache-pipe-01` в `build-local/hy3-tests`. Последняя проверка
запущена через `python -X utf8 build-local/check-hy3-cache-pipe.py`, который
вызывает существующий `tools.check_hy3_engine.fixture` с cap8 МиБ и Monitor.
Для её повтора нужен новый output directory. Сборка и перечисленные финальные
проверки — exit0.

**Не закрыто:** async read/H2D/compute overlap, tensor batching, отдельная
prefill admission policy, физические SSD/RAM-hit counters, full-model
stochastic/long-context corpus и hardware device loss. Следующая задача
**HY3-07 / P3.2** — bounded pipeline с доказанным timeline overlap и exact
parity относительно synchronous cache8. MTP остаётся off.

### HY3-07 / P3.2 — 2026-10-06 — Bounded expert pipeline

**Основание:** HEAD `4336d41a7b25b21db80ea5bfa7d9e0d799d0bd14` с изменениями
HY3-06; новый этап без commit. Engine SHA-256:
`68288b37ee631bea8e8b45f9e13aa33615f5d1002d54963b994542670de1e0d3`.
Patch marker `hy3-bounded-pipeline`. Dependency, GGUF, template, граф вычислений
и математические kernels не менялись. Engine HY3-06 с SHA `8da25192…` и профиль
с его точным checksum сохранены в `build-local/hy3-baseline-cache8`.

**Изменено:** Hy3 использует неизменённые `common/expert_pipeline.hpp`,
Step cache controller/PlanPins и Step GPU event tracer. Отдельные
`pipeline_runtime.inc`/`pipeline_sched.inc` строят план gate/up/down по реальным
router IDs и будущему порядку scheduler splits только для того же route.
Registry проверяет generation, tensor metadata и live file mapping; веса
никогда не предсказываются. Кэш удерживает будущие hits до завершения плана.

CLI `--pipeline-readers 0..2`, `--pipeline-chunk-mib 4|8|16`; default readers0.
Nonzero readers требуют pinned mode и nonzero cache cap. Стартовый рабочий
вариант использует **4 слота по4 МиБ**, то есть16 МиБ pinned RAM и16 МиБ GPU
ring. В плане первоначально предлагались2 слота; общий transport имеет фиксированные4,
поэтому сохранён его проверенный механизм без изменения общих backend-файлов.
Синхронный host buffer освобождается при включении pipeline. Перед allocation
проверяются общий95%-budget и запас; ненужная RAM искусственно не заполняется.

Miss читает зарегистрированный файл → pinned slot → независимый H2D stream;
вычисления получают данные D2D в обычной точке scheduler. Ready/used CUDA events
защищают slot ownership. Перед перезаписью scratch и после каждой доставки/fill
пока остаются fences. Plan scope завершает reader/H2D работу на каждом выходе
из graph, включая STOP и ошибки. Отказ producer требует пересоздания ring перед
следующим запросом; mappings/pins не освобождаются до drain. Незавершённый fill
инвалидирует кэш. Ошибки device loss/настоящий driver OOM не моделировались.

**Проверки — PASS:**

- [HY3_PIPELINE_RUNTIME_VALIDATION.json](HY3_PIPELINE_RUNTIME_VALIDATION.json):
  **246/246**, включая прежние100 и146 новых. F32/mixed, batch1/17, readers1/2,
  cold/warm/eviction, точные GPU bytes/logits, bounded16+16 МиБ ring,
  STOP после частичной доставки, реальный invalid-handle ReadFile до старта
  workers, injected submission failure после двух H2D, incomplete fill,
  auto recovery, simulated RAM/VRAM pressure. Wide matrix проверяет несколько
  chunks. После drain queued/reader-owned=0. Режимы8/16 МиБ chunk не сравнивались
  на полном GGUF; здесь рабочий chunk4 МиБ.
- [HY3_PIPELINE_PYTHON_VALIDATION.json](HY3_PIPELINE_PYTHON_VALIDATION.json):
  **45 CPU tests +7 CLI admission checks**, hashes исходников и engine.
- [HY3_PIPELINE_PIPE_VALIDATION.json](HY3_PIPELINE_PIPE_VALIDATION.json):
  **22/22**, synthetic cache8 МиБ/readers2: A/B/A, seeded sampling, отказ
  некорректных команд, STOP/recovery, session isolation и QUIT во время запроса.
- [HY3_PIPELINE_MODEL_VALIDATION.json](HY3_PIPELINE_MODEL_VALIDATION.json):
  полный GGUF, cache8192 МиБ, readers0/1/2/0. По два прохода English/code в каждом
  процессе: **16 генераций**, затем4 отмены и4 восстановления. Все полные
  token IDs и logit hashes совпали с историческим HY3-03 baseline. Каждый
  процесс exit0; memory guards без ошибок; незавершённых pipeline payloads нет.
- [HY3_PIPELINE_HTTP_VALIDATION.json](HY3_PIPELINE_HTTP_VALIDATION.json):
  полный GGUF, cache8/readers2/chunk4, production Service/StrataEngine.
  **7 завершённых запросов:** OpenAI/Anthropic JSON/SSE, math, tool call и
  продолжение `17 C`, точные prompt/output IDs. Disconnect на prefill запроса
  из778 токенов дал DONE cancel и снятие busy через **2,118 с**, следующий запрос
  восстановил эталон. Engine exit0, временный loopback listener закрыт.
  Sampled peak RAM **36,408 ГиБ**, VRAM **26 923 МиБ**, guard без ошибок.

**Измерения:** RTX5090/driver581.80, Windows,128GB установленной RAM,
context2048/batch17/F32 KV, temperature0/no_think, MTP/fusion/TF32 off.
В таблице прогретая пара English/code, prompt25/28, output15/24 с EOS:
**37 decode-шагов**; первые токены получены из prefill и в decode throughput
не входят. Fresh KV на запрос, cache/history сохраняются. Trace выключен.
Один последовательный sweep с synchronous повторами до/после, OS file cache
не сбрасывался. Это короткое сравнение, не универсальный optimum или длинный
рандомизированный benchmark. Расчёты: [HY3_PIPELINE_BENCHMARK.json](HY3_PIPELINE_BENCHMARK.json).

| Readers | Decode, ток/с | Prefill пары, с | Полное время пары, с | H2D/decode, ГиБ | Peak VRAM, МиБ | Peak RAM, ГиБ |
|---:|---:|---:|---:|---:|---:|---:|
| 0, до | 1,869 | 17,245 | 37,066 | 3,077 | 26 481 | 34,103 |
| 1 | 2,523 | 12,590 | 27,279 | 3,080 | 26 514 | 33,335 |
| 2 | **3,100** | **9,261** | **21,224** | 3,080 | 26 541 | 33,942 |
| 0, после | 1,842 | 16,958 | 37,068 | 3,077 | 26 506 | 35,750 |

Readers2 дали **+67,0% decode throughput** против объединённых synchronous
повторов (1,856 ток/с) и **−42,7% времени пары**. H2D почти одинаков: улучшение
не связано с дополнительным сокращением объёма весов. С pipeline residency
составил8191,188 МиБ против8192 без него; PlanPins немного меняют решения
eviction, поэтому это сравнение полных способов доставки, не только числа
потоков. Peak — sampled global usage, включая другие процессы.

Отдельный HTTP correctness-прогон: tool call prompt211/output19 — prefill
**47,634 с**, generation **6,218 с**; tool result prompt253/output3 — prefill
**59,206 с**, generation **0,871 с**. В предыдущем HY3-06 cache8 без pipeline
prefill был92,011/102,437 с. Это последовательные наблюдения разных прогонов,
не самостоятельный рандомизированный HTTP A/B; основной speed вывод выше
основан на одинаковом English/code corpus с synchronous повторами до/после.

**CUDA trace — PASS:** [HY3_PIPELINE_TRACE_VALIDATION.json](HY3_PIPELINE_TRACE_VALIDATION.json).
Отдельный полный English-запрос, readers2/cache8,4 instrumented graphs,
exact logits/IDs, engine exit0. H2D/compute overlap по событиям:
**1,056;0,753;0,095;0,433 мс** (сумма2,337 мс). Первые два graphs — prefill,
следующие — decode. Это небольшое перекрытие; оно не объясняет весь прирост
скорости. Измеряются интервалы на реальных streams, включая scheduling gaps,
а не kernel occupancy. CPU read/wait суммы также не доказывают GPU overlap.
Диагностическое время не смешивается с benchmark. Tiny warm fixture дал0 overlap,
что тоже отражено в raw fixture trace.

**Профили:** созданы отдельные `build-local/hy3-http-pipeline1/hy3.json` и
`build-local/hy3-http-pipeline2/hy3.json`. По этому sweep выбран readers2/chunk4,
cache8; автоматический default exporter остаётся pipeline0/cache0.
Для rollback имеется точный synchronous exe/profile HY3-06 выше.

```powershell
python -m serve.server --engine strata --config build-local/hy3-http-pipeline2/hy3.json --port 8094
python tools/check_hy3_http.py --profile build-local/hy3-http-pipeline2/hy3.json --reference docs/hy3/HY3_TOOL_DIALOGUE_VALIDATION.json --output-dir build-local/hy3-tests/pipeline-http-new --allow-engine-change
```

Server и checker запускать по очереди; checker создаёт собственный loopback
listener. Build и model sweep команды — в [backend README](../../backends/hy3/README.md).
Raw directories: `pipeline-runtime-01`, `pipeline-model-01`, `pipeline-pipe-01`,
`pipeline-trace-01`, `pipeline-http-01` в `build-local/hy3-tests`. Local runners:
`build-local/check-hy3-pipeline-pipe.py`, `check-hy3-pipeline-trace.py`,
`summarize-hy3-pipeline.py`; для повтора выбирать новые output directories.
Сборка и перечисленные финальные проверки завершились с exit0. Source/profile
checksums сверены с протестированным exe; `git diff --check` прошёл.

**Не закрыто:** tensor batching, отдельная prefill admission policy, длинный
контекст/stochastic full-model corpus, physical SSD/OS cache counters,
hardware device loss, production sessions и MTP. Следующая задача **HY3-08 / P3.3** —
сократить число fences на доставку tensor с сохранением exact parity.

### HY3-08 / P3.3 — 2026-10-06 — Группировка копий одного tensor

**Основание:** HEAD `4336d41a7b25b21db80ea5bfa7d9e0d799d0bd14`, изменения
HY3-06/07/08 без commit. Engine SHA-256:
`e24dc96ac7edfb308d0574b07a2e0f29d80f202387ef7dfc640ff105b0c0078a`.
Patch marker `hy3-tensor-batch-copy`. Dependency, GGUF, template, математические
kernels, common pipeline и Step cache controller не изменены в этом этапе.
Старый exe HY3-07 с SHA `68288b37…` и отдельный профиль с точным checksum
сохранены в `build-local/hy3-baseline-pipeline2`.

**Изменено:** `--pipeline-batch 1` ставит копии всех выбранных experts одного
scheduler input tensor в consumer stream, затем выполняет один delivery fence.
Перед перезаписью scratch остаётся синхронизация с прежними вычислениями.
PlanPins защищают cache hits; новые cache destinations удерживаются отдельными
pins до завершения всех копий. Cache fill следует за ring→scratch copy в том же
stream. При STOP или исключении stream сначала завершается, затем освобождаются
pins; незавершённый fill инвалидирует кэш. Общий ring остаётся4×4 МиБ.

Режим требует nonzero pipeline readers. Native default — `--pipeline-batch 0`;
exporter включает его только по явному `--pipeline-batch`. Добавлены counters
числа tensor batches, delivery fences, максимального числа удерживаемых fills
и CPU wall time batch. Delivery fences не включают синхронизацию перед scratch,
graph-end и возможную внутреннюю синхронизацию CUDA allocator.

**Проверки и обнаруженное ограничение:**

- [HY3_BATCH_RUNTIME_VALIDATION.json](HY3_BATCH_RUNTIME_VALIDATION.json):
  финальный полный набор **472/472 PASS**, включая226 новых проверок.
  F32/mixed/wide, batch1/17, cold/warm/eviction, точные GPU bytes/logits,
  несколько pending fills, ReadFile/submission/fill errors, STOP после двух
  поставленных копий, recovery и injected allocation OOM bypass.
- Первый полный запуск дал **471/472**:
  [HY3_BATCH_INITIAL_RUNTIME_FAILURE.json](HY3_BATCH_INITIAL_RUNTIME_FAILURE.json).
  Старый случай `mixed/batch=4/native_vs_resident` при **выключенном pipeline**:
  max abs `0.016343072056770325` на1728 элементах. Native/pinned совпали точно,
  все новые batch cases прошли. Два следующих полных запуска прошли472/472;
  checksum mixed fixture одинаков. Исходный FAIL сохранён, причина **не найдена**.
- [HY3_BATCH_REFERENCE_VALIDATION.json](HY3_BATCH_REFERENCE_VALIDATION.json):
  **60/60 PASS** —32 resident/native сравнения на4 reloads и28 native repeats
  с fresh KV, batch4/pipeline0. Focused probe не воспроизводит всю исходную
  историю F32→mixed и смену contexts, поэтому не доказывает устранение ошибки.
  Runtime checker теперь сохраняет оба полных массива logits при расхождении.
- [HY3_BATCH_PYTHON_VALIDATION.json](HY3_BATCH_PYTHON_VALIDATION.json):
  **46 CPU tests +6 CLI admission checks PASS**, hashes engine/исходников.
- [HY3_BATCH_PIPE_VALIDATION.json](HY3_BATCH_PIPE_VALIDATION.json):
  **22/22 PASS**, synthetic batch1/cache8 МиБ/readers2: A/B/A, seeded sampling,
  invalid commands, STOP/recovery, session isolation и QUIT во время запроса.
- [HY3_BATCH_MODEL_VALIDATION.json](HY3_BATCH_MODEL_VALIDATION.json):
  полный GGUF, порядок batch **0/1/1/0**, по2 English/code прохода:
  **16 генераций +4 отмены +4 восстановления**. Все полные token IDs и logit
  hashes совпали с историческим HY3-03. Каждый процесс exit0; guard без ошибок.
- [HY3_BATCH_LONG_VALIDATION.json](HY3_BATCH_LONG_VALIDATION.json):
  prompt362/output2, batch0/1, в каждом режиме обычный запрос, STOP и повтор.
  Все полные IDs/logit hashes совпали с HY3-03, оба процесса exit0,
  sampled95% guards без ошибок. Подробные времена ниже.
- [HY3_BATCH_HTTP_VALIDATION.json](HY3_BATCH_HTTP_VALIDATION.json):
  **7 завершённых запросов PASS**, production Service/StrataEngine с batch1.
  Math JSON/SSE через OpenAI/Anthropic дали одинаковые prompt/output IDs;
  первый math-запрос задаёт локальный эталон. Tool call/result совпали
  с историческими native IDs: `get_temperature({"city":"Paris"})` → `17 C`.
  Disconnect на prefill778 токенов освободил busy через **2,031 с**;
  следующий math-запрос восстановил эталон. Engine exit0, loopback listener
  закрыт. Peak RAM **36,497 ГиБ**, VRAM **26 415 МиБ**, guard без ошибок.

**Скорость:** RTX5090/driver581.80, Windows,128GB установленной RAM,
cache8192 МиБ/readers2/chunk4, context2048/prompt batch17/F32 KV,
temperature0/no_think, MTP/fusion/TF32 off. Второй English/code проход каждого
процесса: prompt25/28, output15/24 с EOS, **37 decode forwards** без первых
токенов из prefill. Fresh KV, сохранённый cache, OS file cache не сбрасывался.
Raw расчёты: [HY3_BATCH_BENCHMARK.json](HY3_BATCH_BENCHMARK.json).

| Tensor batch | Decode, ток/с | Prefill пары, с | Полное время пары, с | Delivery fences на37 decode | Peak VRAM, МиБ | Peak RAM, ГиБ |
|---|---:|---:|---:|---:|---:|---:|
| 0, до | 3,061 | 9,472 | 21,586 | 73 945 | 26 559 | 34,976 |
| 1, первый | 3,227 | 9,323 | 20,814 | 8 769 | 26 556 | 35,108 |
| 1, второй | 2,683 | 10,851 | 24,670 | 8 769 | 26 817 | 38,116 |
| 0, после | 2,587 | 11,459 | 25,793 | 73 945 | 26 694 | 38,495 |

По объединённым decode durations: **2,804 → 2,930 ток/с (+4,5%)**;
среднее полное время пары **23,690 → 22,742 с (−4,0%)**. Delivery fences
сократились в **8,43 раза**, но передаваемый H2D объём одинаков —3,080 ГиБ
на decode forward. Cache residency в конце8191,188 МиБ у всех вариантов.
Peak — sampled global usage, включая другие процессы, ниже95% лимита.

Разброс скорости между процессами больше среднего выигрыша. К концу sweep
выросли и CPU read time sums; это сумма работы двух readers, не elapsed time
и не физическое SSD время. Причина дрейфа не установлена. Короткий ABBA
подтверждает работу batching, но недостаточен для изменения defaults.

**Длинный prefill:** отдельный последовательный batch0→1, тот же GGUF и
настройки,362 prompt tokens. Первый запрос начинается с пустым GPU cache;
восстановительный запрос использует cache history после STOP и fresh KV.

| Tensor batch | Первый prefill, с | Prefill после STOP, с | Первый request, с | Prefill delivery fences | Peak VRAM, МиБ | Peak RAM, ГиБ |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 121,454 | 136,679 | 122,049 | 512 113 | 30 234 | 42,097 |
| 1 | 105,085 | 102,664 | 105,490 | 5 214 | 30 223 | 44,941 |

Первый prefill с batching занял на13,5% меньше времени. Это одно наблюдение
на вариант без обратного порядка; оно не устанавливает стабильный выигрыш.
Первый prefill передал822,472/822,509 ГиБ через H2D: объём почти одинаков,
но не идентичен. Более долгие pending pins немного меняют cache eviction,
поэтому сравнивается весь режим доставки. Выход всего2 токена с EOS;
decode throughput этого запроса не используется для вывода о скорости.
STOP завершился через5,929/3,576 с; это wall time отменяемого pipe-запроса,
включая его начальный prefill, а не задержка от отправки STOP.

В отдельном HTTP correctness-прогоне tool call prompt211/output19 дал
prefill **48,260 с**, generation **5,895 с**; tool result prompt253/output3 —
prefill **52,560 с**, generation **0,822 с**. HTTP A/B в этом этапе не проводился;
эти наблюдения не заменяют короткое ABBA сравнение выше.

**Профили:** контроль `build-local/hy3-http-batch0/hy3.json`; эксперимент
`build-local/hy3-http-batch1/hy3.json`. Оба используют новый checksummed exe,
cache8/readers2/chunk4; различается только tensor batching. Старые профили,
которые указывают на пересобранный exe с прежним checksum, не обновлялись:
для rollback использовать сохранённый `hy3-baseline-pipeline2/hy3.json`.

```powershell
cmd /c build-local\build-hy3-cache.bat
python -X utf8 tools/check_hy3_cache_model.py --engine build-local/hy3-cuda/bin/strata-hy3.exe --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --reference docs/hy3/HY3_MODEL_VALIDATION.json --cuda-bin build-local/cuda-13.0/bin/x64 --cache-mib 8192 --pipeline-readers 2 --pipeline-batch 0 1 1 0 --output-dir build-local/hy3-tests/batch-model-new
python -X utf8 tools/check_hy3_cache_model.py --engine build-local/hy3-cuda/bin/strata-hy3.exe --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --reference docs/hy3/HY3_MODEL_VALIDATION.json --cuda-bin build-local/cuda-13.0/bin/x64 --cache-mib 8192 --pipeline-readers 2 --pipeline-batch 0 1 --prompts long --repeats 1 --output-dir build-local/hy3-tests/batch-long-new
python -X utf8 tools/check_hy3_http.py --profile build-local/hy3-http-batch1/hy3.json --reference docs/hy3/HY3_TOOL_DIALOGUE_VALIDATION.json --output-dir build-local/hy3-tests/batch-http-new --allow-engine-change
python -m serve.server --engine strata --config build-local/hy3-http-batch1/hy3.json --port 8094
```

Не запускать server и GPU checkers одновременно. Runtime и focused probe
команды — в [backend README](../../backends/hy3/README.md). Raw directories:
`batch-runtime-01/02/03`, `batch-reference-01`, `batch-pipe-01`, `batch-model-01`, `batch-long-01`
и `batch-http-01` в `build-local/hy3-tests`. Для повторов выбирать новые output
directories. Сборка, финальные runtime/model/HTTP проверки завершились с exit0;
первый runtime FAIL описан отдельно выше. Source/profile hashes совпали
с протестированным exe, `git diff --check` прошёл.

**Следующий этап HY3-09:** воспроизвести intermittent resident/native mismatch
с исходной историей тестов и сравнить промежуточные тензоры. Затем измерить
read/admission/allocator costs перед отдельной prefill admission policy.
MTP, production sessions, длинный контекст и длительный рандомизированный
speed benchmark остаются открыты. Tensor batching пока экспериментальный.

### HY3-09 / P3.3 — 2026-10-06 — Диагностика CUDA graphs и времени доставки

**Основание:** HEAD `4336d41a7b25b21db80ea5bfa7d9e0d799d0bd14`, изменения
HY3-06/07/08 уже находились в index в начале этапа; HY3-09 добавлен без commit.
Engine SHA-256:
`b5fcc067554cd4d8acbe8318a468eea96cb1fa9a141107980996071579117365`.
Новые markers: `hy3-delivery-profile`, `hy3-disable-cuda-graphs`.
Dependency, GGUF, tokenizer/template, математические kernels и общие
Step cache/pipeline headers не изменены. Исходный HY3-08 exe с SHA `e24dc96a…`
и checksummed профиль сохранены в `build-local/hy3-baseline-batch0`.

**Расхождение воспроизведено, точная причина не исправлена.**
[HY3_GRAPH_INVESTIGATION.json](HY3_GRAPH_INVESTIGATION.json) хранит результаты
всех вариантов; полный FAIL с logits и промежуточными сравнениями —
[HY3_GRAPH_FAILURE_DIAGNOSTIC.json](HY3_GRAPH_FAILURE_DIAGNOSTIC.json).

- Шестой запуск неизменённого полного набора HY3-08 воспроизвёл
  `mixed/batch=4/native_vs_resident`: max abs **0,0163430721**, с первого токена.
  Первые5 запусков прошли472/472. Native/pinned снова совпали побитово.
- Новый `--history-probe` повторяет исходные F32→mixed→wide и batch1/4/17
  в одном процессе, сохраняя смену моделей/contexts и порядок копий.
  На пятом цикле два запуска воспроизвели mixed/batch17, max abs **0,0183075666**.
- После сбоя все загруженные resident/mapped веса совпали по GPU bytes.
  Повтор в тех же contexts точно сохранил обе разные последовательности logits.
  Fresh contexts с callback после каждой F32-операции дали одинаковые78
  промежуточных тензоров; captured resident совпал с прежним resident, captured
  native изменился. Callback меняет разбиение графа, поэтому сам по себе
  не локализует неисправную операцию и не заменяет исходное сравнение.
- `GGML_CUDA_PDL=0` не устранил ошибку: F32/batch17 на14-м цикле,
  max abs **0,0010767281**. Принудительная перепроверка свойств CUDA graph
  вместо UID fast path тоже дала F32/batch17 FAIL. Этот экспериментальный
  переключатель удалён из исходников; его binary сохранён локально в
  `build-local/hy3-history-diagnostic`.
- Восемь полных запусков с `GGML_CUDA_DISABLE_GRAPHS=1` прошли472/472;
  отдельные32 history cycles —448/448. На итоговой сборке с отключёнными graphs
  также **448/448 PASS**:
  [HY3_GRAPH_HISTORY_VALIDATION.json](HY3_GRAPH_HISTORY_VALIDATION.json).
- Повторное включение graphs в итоговом checker тоже прошло448/448:
  [HY3_GRAPH_ENABLED_RECHECK.json](HY3_GRAPH_ENABLED_RECHECK.json).
  Сбой зависит от истории и не воспроизводится каждый раз; этот PASS
  **не отменяет прежние FAIL и не доказывает исправление**.

**Изменено:** isolated Hy3 runtime задаёт `GGML_CUDA_DISABLE_GRAPHS=1` до
инициализации backend, INFO показывает `cuda_graphs=0`. Это консервативный
обход наблюдаемого сбоя. Native file readers, H2D stream/ring, GPU cache и
tensor batching работают независимо от capture/replay и сохранены.
В production engine нет переключателя для возврата непроверенного режима.
Только тестовый `--runtime-probe history-graphs` включает graphs для диагностики.
Другие backends не менялись. Внутренняя причина в CUDA graph path ещё открыта.

Добавлен opt-in `--profile-delivery 1`, требующий pipeline readers.
JSON request metrics содержит `prefill_delivery`/`decode_delivery`: cache
get/admit, victim scan, CUDA allocate/free, refresh/probe, PlanPins/trim,
ожидание scratch и delivery fences. Default off не читает часы этих scopes.
Включение таймеров не меняет cache policy. Times — inclusive CPU wall;
admit включает victim/allocate, refresh включает probe. Их нельзя складывать.
Source-read CPU time суммируется по двум readers; это не elapsed или SSD time.
Диагностические времена не смешиваются со speed benchmark.

**Проверки:**

- [HY3_GRAPH_RUNTIME_VALIDATION.json](HY3_GRAPH_RUNTIME_VALIDATION.json):
  **484/484 PASS**, прежние472 и12 новых для profiler off/on, tensor batch0/1,
  точных logits, eviction counters и сброса таймеров без потери cache residency.
- [HY3_GRAPH_PYTHON_VALIDATION.json](HY3_GRAPH_PYTHON_VALIDATION.json):
  **46 CPU tests +6 CLI checks PASS**, engine/source hashes.
- [HY3_GRAPH_MODEL_VALIDATION.json](HY3_GRAPH_MODEL_VALIDATION.json):
  старый/new/new/старый exe, graphs1/0/0/1; **16 English/code генераций,
  4 отмены и4 восстановления**, все full IDs/logit hashes совпали с HY3-03.
  Каждый процесс exit0, memory guards без ошибок.
- [HY3_DELIVERY_PROFILE_VALIDATION.json](HY3_DELIVERY_PROFILE_VALIDATION.json):
  полный GGUF,362 prompt tokens, profiler on, обычный запрос, STOP и recovery.
  Полные IDs/logit hashes совпали с HY3-03. Engine exit0, guard без ошибок,
  peak RAM **40,185 ГиБ**, VRAM **28 862 МиБ**.
- [HY3_GRAPH_HTTP_VALIDATION.json](HY3_GRAPH_HTTP_VALIDATION.json):
  **7 запросов и1 disconnect/recovery PASS**, OpenAI/Anthropic JSON/SSE.
  Math IDs совпали между API и после отмены; tool call/result IDs точно
  совпали с сохранённым native reference. Отмена prefill завершилась через
  **2,390 с** после disconnect. Engine exit0, memory guard без ошибок;
  sampled global peak RAM **40,240 ГиБ**, VRAM **30 212 МиБ**, ниже95%.
  Tool call prompt211/output19: prefill55,284 с, generation7,202 с;
  tool result prompt253/output3: prefill64,457 с, generation1,002 с.
  Это correctness-прогон, не дополнительное A/B измерение скорости.

**Цена обхода:** [HY3_GRAPH_BENCHMARK.json](HY3_GRAPH_BENCHMARK.json).
RTX5090/driver581.80, Windows,128GB RAM, cache8/readers2/chunk4/tensor batch0,
context2048/prompt batch17/F32 KV, temperature0/no_think, MTP/TF32/fusion off.
Delivery profiler и CUDA trace выключены. Второй English/code проход:
prompt25/28, output15/24 с EOS,37 decode forwards без первых prefill tokens.
Fresh KV, cache сохраняется, OS file cache не сбрасывался.

| CUDA graphs | Decode, ток/с | Prefill пары, с | Полное время пары, с | Peak VRAM, МиБ | Peak RAM, ГиБ |
|---|---:|---:|---:|---:|---:|
| on, HY3-08 до | 2,993 | 9,586 | 21,977 | 26 264 | 37,250 |
| off, HY3-09 первый | 2,892 | 9,418 | 22,240 | 26 138 | 37,343 |
| off, HY3-09 второй | 2,917 | 9,572 | 22,282 | 26 203 | 37,353 |
| on, HY3-08 после | 3,008 | 9,498 | 21,824 | 26 266 | 37,184 |

По объединённым decode durations **3,000 → 2,905 ток/с (−3,19%)**;
среднее время пары **21,900 → 22,261 с (+1,65%)**. H2D одинаков во всех
вариантах —3,080 ГиБ/decode forward. Сравниваются два exe: новая сборка
отключает graphs и содержит неактивные diagnostic timers; это сравнение
рабочих сборок, не доказательство нулевого overhead дополнительных веток.
Короткий стабильный ABBA не заменяет длинный corpus. Выбран graphs off
из-за наблюдавшегося correctness-сбоя, а не ради ускорения.

**Профиль длинного prefill:** тот же cache8/readers2/chunk4/tensor batch0,
CUDA graphs off. Первый prefill **116,449 с**, повтор после STOP **109,058 с**.
Это отдельный instrumented correctness-прогон, не сравнение скорости.

| Измеренный scope первого prefill | CPU wall, с | Интерпретация |
|---|---:|---|
| Cache admission | **45,813** | Inclusive: включает выбор жертвы и allocation/free |
| CUDA allocation | 14,162 | Внутри admission;96 777 новых allocation calls |
| CUDA free | 25,016 | Включает возможные ожидания GPU/driver внутри API |
| Victim scan | 5,772 | Внутри admission;10 150 944 просмотренных кандидата |
| Cache lookup | 2,760 | Поиск и обновление частот |
| Memory probe | 0,174 | Внутри refresh0,177 с; не основной источник затрат |
| Delivery fences | 49,078 | CPU ожидание окончания копий, не чистое время GPU DMA |
| Producer wait | 7,045 | Ожидание готовности очередного payload |
| Source read sum | 146,012 | Сумма двух одновременно работающих readers, не elapsed/SSD time |

Prefill выполнил94 711 allocation reuses и526 903 599 616 байт cache fills.
Несмотря на reuse, cache admission занял около39% prefill wall time;
allocator/free scopes вложены и отдельно к нему не прибавляются. Потоки
чтения могут работать одновременно с этими scopes. Следующий эксперимент —
ограничить admission во время prefill, сохранив hits и доставку всех misses.
Изменять95%-guard ради скорости эти результаты не обосновывают.

**Воспроизведение:**

```powershell
cmd /c build-local\build-hy3-cache.bat
python -X utf8 tools/check_hy3_cuda.py --build build-local/hy3-cuda --kind runtime --runtime-probe history --cuda-bin build-local/cuda-13.0/bin/x64 --output-dir build-local/hy3-tests/history-new
python -X utf8 tools/check_hy3_cuda.py --build build-local/hy3-cuda --kind runtime --runtime-probe history-graphs --cuda-bin build-local/cuda-13.0/bin/x64 --output-dir build-local/hy3-tests/history-graphs-new
python -X utf8 tools/check_hy3_cache_model.py --engine build-local/hy3-cuda/bin/strata-hy3.exe --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --reference docs/hy3/HY3_MODEL_VALIDATION.json --cuda-bin build-local/cuda-13.0/bin/x64 --cache-mib 8192 --pipeline-readers 2 --pipeline-batch 0 --prompts long --repeats 1 --profile-delivery --output-dir build-local/hy3-tests/delivery-profile-new
python -X utf8 tools/check_hy3_http.py --profile build-local/hy3-http-graphsafe/hy3.json --reference docs/hy3/HY3_TOOL_DIALOGUE_VALIDATION.json --output-dir build-local/hy3-tests/graphsafe-http-new --allow-engine-change
python -m serve.server --engine strata --config build-local/hy3-http-graphsafe/hy3.json --port 8094
```

Второй probe намеренно включает проблемный режим только в checker; возможен
FAIL с сохранённой диагностикой. Не ослаблять bit-exact admission и не удалять
неудачные отчёты. Новый профиль: `build-local/hy3-http-graphsafe/hy3.json`,
cache8192 МиБ/readers2/chunk4/batch0, F32 KV/context2048/batch17, MTP off.

Не запускать server и GPU checkers одновременно. Raw directories:
`graphsafe-runtime-01`, `graphsafe-history-01`, `graphsafe-reproduce-01`,
`graph-model-01`, `delivery-profile-01`, `graphsafe-http-01` внутри
`build-local/hy3-tests`. Локальный ABBA helper
`build-local/check-hy3-graph-abba.py` сохраняет точный порядок старый/new/new/старый;
для повтора выбрать новый output directory в helper. Старый engine находится
в `build-local/hy3-baseline-batch0`; его нельзя заменить новой сборкой при A/B.
Все финальные проверки завершились с exit0. Source/profile hashes сверены
с протестированным exe, локальные ссылки и `git diff --check` проверены.
HTTP test listener и его engine завершены; фоновый server не оставлен.


### HY3-10 / P5 — 2026-10-06 — Native MTP с resident draft-блоком

**Основание:** HEAD `4336d41a7b25b21db80ea5bfa7d9e0d799d0bd14`,
незакоммиченные изменения HY3-09 сохранены. Пользователь явно перенёс
приоритет с prefill policy на MTP. Контрольный HY3-09 exe/profile сохранён
в `build-local/hy3-baseline-graphsafe` (SHA `b5fcc067…`).
Итоговый engine SHA-256:
`c87706ae8c83640ff08f3a21fe5cce8eaa2aaf751f0f7b90f1d56ac528108e89`.
Первый depth sweep использовал SHA `4bb1fc0f…`, сохранённый в
`build-local/hy3-mtp-initial`. Затем из проверок finite убраны временные
строки на каждый logit, добавлены bounds draft batch и счётчики в cancelled
DONE. Удалено вычисление лишнего токена после предложенного EOS: EOS проверяется
logits предыдущего токена. Есть отдельный rejected-draft-EOS fixture. Промежуточный
exe `95ef9057…` и его полный6-prompt corpus сохранены в
`build-local/hy3-mtp-before-eog` и
[HY3_MTP_CORPUS_BEFORE_EOG.json](HY3_MTP_CORPUS_BEFORE_EOG.json). Математика, веса, dependency и общие CUDA kernels не менялись.

**Реализация:** `--mtp 0..3`, default0. Один model owner загружает блок80
целиком в VRAM (1,729 ГиБ), общий embedding/output не дублируется. Main/draft
contexts имеют отдельный F32 KV. Target experts используют прежний cache
и async pipeline; resident draft experts не включаются в mapped registry.
Global95% guard учитывает оба контекста, resident MTP и cache.

Driver в `backends/hy3/mtp.hpp` связывает x[p] с post-final-norm h[p−1],
переносит hidden через границы prefill chunks, предлагает1..3 черновика,
проверяет их вместе с carry token, откатывает rejected KV suffix и обновляет
draft history по подтверждённым target features. Target sampler вызывается
один раз на выдаваемый токен. Он учитывает greedy penalties; draft greedy
может иметь другую историю penalties, что влияет на acceptance, не на verify.
Fresh request, STOP и ошибка очищают оба KV. Temperature>0 использует прежний
target-only sampling, а не непроверенный stochastic speculative algorithm.

INFO показывает `mtp`, `spec`, `mtp_storage=resident`; request JSON — фактический
`mtp_depth`, proposed/accepted/delivered, rounds, rejection counts и времена
prefill/draft/verify/repair. DONE передаёт delivered drafts и offered drafts.
С MTP `decode_steps` означает verify batches; скорость ниже считается по
подтверждённым выходным токенам после первого prefill sample, делённым на
полное `generation_wall_ms`, включая draft/verify/repair/sampling/output.

**Проверки итоговой сборки:**

- [HY3_MTP_RUNTIME_VALIDATION.json](HY3_MTP_RUNTIME_VALIDATION.json):
  **59 synthetic comparisons PASS**, F32 и два mixed layouts, depth1/2/3,
  forced reject0/1/2, accept-all, bonus/output limits, stop, cancel/recovery,
  greedy penalties и край context128. MTP-loaded target-only совпал побитово.
- [HY3_MTP_OFF_REGRESSION.json](HY3_MTP_OFF_REGRESSION.json):
  **484 runtime checks PASS** для существующего cache/pipeline, profiler,
  отказов и bit-exact selected-copy. CUDA graphs остаются off.
- [HY3_MTP_PYTHON_VALIDATION.json](HY3_MTP_PYTHON_VALIDATION.json):
  **47 CPU tests +9 CLI checks PASS**, проверены profile identity/depth bounds.
- [HY3_MTP_PIPE_VALIDATION.json](HY3_MTP_PIPE_VALIDATION.json):
  **22 checks PASS** при depth1; greedy и seeded sampling IDs совпали с
  тем же pipe fixture при MTP-off. Включены A/B/A isolation, invalid requests,
  STOP, recovery, session key isolation и QUIT во время prefill.

**Полный GGUF и HTTP:**
[HY3_MTP_MODEL_VALIDATION.json](HY3_MTP_MODEL_VALIDATION.json) —
12 генераций,6 prompts (en/ru/zh/code/numbers/long) × off/depth1, exact greedy IDs;
off logits также совпали с историческим hash. Каждый режим прошёл STOP
в prefill и decode с последующим восстановлением. Seeded temperature0,7
fallback дал одинаковые IDs и hash logits в обоих режимах. Guards без ошибок,
exit0. Long prompt362: prefill93,806 с off и152,425 с on; это одиночный
correctness-прогон, выигрыша long prefill он не показывает. Global peak RAM
40,372/48,741 ГиБ, VRAM27 007/30 250 МиБ.
[HY3_MTP_HTTP_VALIDATION.json](HY3_MTP_HTTP_VALIDATION.json) —
7 ответов и disconnect/recovery PASS: OpenAI/Anthropic JSON/SSE, точные
math/tool/result IDs. Listener и engine завершены после проверки.

**Численная проверка:** MTP-off на полном GGUF сохраняет исторические
bit-exact logits. При батче target verification logits отличаются от serial
(в первом corpus max abs до2,240, NMSE до0,01011), поэтому битовое совпадение
с serial MTP-on не объявляется. Независимый
[HY3_MTP_LOGITS_VALIDATION.json](HY3_MTP_LOGITS_VALIDATION.json) загрузил
**только target, без MTP weights/hidden/KV rollback**, и выполнил тот же English
continuation батчами1/2/3/4. Все четыре массива logits побитово совпали с
соответствующими depth0/1/2/3 из sweep. Это отделяет эффект размера батча
от ошибок драйвера для этого запроса, но не доказывает universal greedy parity.

В synthetic checker используются существующие Hy3 graph limits:
F32 abs≤5e−4/NMSE≤1e−7, mixed abs≤0,03/NMSE≤2e−3, и точные greedy IDs.
Near-limit mixed fixture дал max abs0,016832/NMSE4,49e−5; независимый
**target-only batch4** дал ровно то же расхождение. Начальные FAIL harness
сохранены в `mtp-fixture-01/02`: первый ошибочно ограничивал forced accept3
оставшимися двумя draft slots; второй использовал чужой NMSE limit до
проверки по Hy3 batch oracle. Итоговые проверки не меняют существующие
bit-exact cache/transport требования.

**Ограничение depth3:**
[HY3_MTP_PIPE_DEPTH3_DIAGNOSTIC.json](HY3_MTP_PIPE_DEPTH3_DIAGNOSTIC.json)
сохраняет FAIL cross-mode greedy comparison. Оба режима прошли по22 pipe
checks, seeded sampling совпал, но на prompt IDs `[11,18,25,32]` mixed fixture
serial/depth3 разошлись на восьмом выходном токене. Depth1 на том же fixture
совпал полностью. Поэтому depth3 не выбран рабочим профилем, P5.4 для всех
глубин не считается завершённым. Depth2/3 доступны для диагностики.

**Первый sweep:**
[HY3_MTP_DEPTH_SWEEP.json](HY3_MTP_DEPTH_SWEEP.json) и
[HY3_MTP_BENCHMARK.json](HY3_MTP_BENCHMARK.json). RTX5090/581.80, Windows,
128GB RAM, cache8192 МиБ/readers2/chunk4/batch-copy0, context2048/batch17,
F32 KV, temperature0/no_think, CUDA graphs/TF32/fusion/profiler off.
Logits записываются во всех вариантах, OS file cache не сбрасывался.
Свежий процесс для каждого depth; два en/code прохода, затем STOP/recovery.
Все20 генераций и5 восстановлений совпали по IDs с HY3-03; off logits — exact.
Таблица использует второй проход: prompt25/28, output15/24 с EOS,
37 полезных токенов после первых prefill samples.

| MTP depth | Decode, ток/с | Prefill пары, с | Время пары, с | Accepted/offered | Peak VRAM, МиБ | Peak RAM, ГиБ |
|---|---:|---:|---:|---:|---:|---:|
| 0, до | 2,868 | 9,595 | 22,501 | 0/0 | 26 346 | 36,591 |
| 1 | 3,185 | 9,743 | 21,366 | 17/19 | 28 100 | 38,434 |
| 2 | 3,223 | 10,027 | 21,512 | 24/27 | 28 125 | 38,988 |
| 3 | 2,297 | 12,201 | 28,312 | 25/35 | 30 233 | 42,877 |
| 0, после | 2,755 | 10,473 | 23,910 | 0/0 | 30 249 | 42,341 |

Объединённый off —2,810 ток/с; depth1 +13,3%, depth2 +14,7% в этом коротком
sweep. Depth1 имеет меньшее полное время пары; разница depth1/2 мала относительно
разброса. Global peaks включают другие процессы; поздний off отличается
по памяти и времени. Это не randomized benchmark и не гарантия ускорения
на других prompts. Сравнивался одинаковый target cache cap, **не одинаковая
полная память**. Последующий
[HY3_MTP_BUDGET_VALIDATION.json](HY3_MTP_BUDGET_VALIDATION.json) сравнил
cache10/MTP0 и cache8/MTP1: все16 генераций и4 восстановления exact IDs.
Во время третьего процесса выполнялась CPU-сборка RAM-кэша, поэтому этот
прогон оставлен как capacity/correctness evidence, не как чистый speed benchmark.

**Профили:** новый off — `build-local/hy3-http-mtp0/hy3.json`;
проверенный MTP1 — `build-local/hy3-http-mtp1/hy3.json`.
Оба cache8/readers2/chunk4/batch-copy0, temperature0, context2048/F32 KV;
MTP1 добавляет resident draft. Старые profiles с checksum прежнего exe
не переписывались; rollback использует сохранённый executable.

**Воспроизведение (по одному GPU-процессу):**

```powershell
cmd /c build-local\build-hy3-cache.bat
python -X utf8 tools/check_hy3_cuda.py --build build-local/hy3-cuda --kind mtp --cuda-bin build-local/cuda-13.0/bin/x64 --output-dir build-local/hy3-tests/mtp-check-new
python -X utf8 tools/check_hy3_mtp.py --engine build-local/hy3-cuda/bin/strata-hy3.exe --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --reference docs/hy3/HY3_MODEL_VALIDATION.json --cuda-bin build-local/cuda-13.0/bin/x64 --depths 0 1 --prompts en ru zh code numbers long --repeats 1 --lifecycle --output-dir build-local/hy3-tests/mtp-corpus-new
python -X utf8 tools/prepare_hy3_profile.py --model H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --engine build-local/hy3-cuda/bin/strata-hy3.exe --output-dir build-local/hy3-http-mtp-new --cuda-dir build-local/cuda-13.0 --expert-cache-mib 8192 --pipeline-readers 2 --mtp 1
python -X utf8 tools/check_hy3_http.py --profile build-local/hy3-http-mtp1/hy3.json --reference docs/hy3/HY3_TOOL_DIALOGUE_VALIDATION.json --output-dir build-local/hy3-tests/mtp-http-new --allow-engine-change
python -m serve.server --engine strata --config build-local/hy3-http-mtp1/hy3.json --port 8094
```

`--variants 10240:0 8192:1 8192:1 10240:0` в MTP checker задаёт явный
порядок cache/depth для сравнения с возвратом памяти draft в target cache.
Первый вариант должен быть off. Raw logs находятся в `build-local/hy3-tests`;
для повторов выбирать новые output directories. Проверка независимых logits:
`tools/check_hy3_mtp_logits.py --help` — нужен исходный sweep с сохранёнными
`logits.f32` после English recovery, не только скопированный JSON.

**Не закрыто:** streamed MTP и отдельный draft cache budget; сессии;
stochastic speculation; depth3 synthetic greedy mismatch; чистый повтор
при возврате draft memory в target cache и длинный randomized speed corpus. GPU prefill policy HY3-13
и внутренняя причина CUDA graphs остаются отдельными задачами.

### HY3-11 / P3 — 2026-10-06 — Управляемый RAM-кэш матриц

**Основание:** тот же HEAD `4336d41a…`, без commit; изменения HY3-09/10 сохранены.
Пользователь явно запросил RAM-кэш после наблюдения низкой занятой RAM.
Перед изменением Windows counters показали81,305 ГиБ standby-cache и
81,312 ГиБ available; почти вся «доступная» RAM уже содержала файловый кэш.
Это замер всей системы, не доказательство принадлежности всех страниц Hy3.

**Реализация:** `backends/hy3/host_cache.hpp`, `--ram-cache-mib auto|N`;
0 по умолчанию сохраняет прежний путь, новые профили включают auto.
Cache хранит неизменяемые chunks выбранных expert matrices с точным MMQ
padding. Key: live Source identity,64-bit offset,размер. Один Source handle
удерживается до удаления entries; новый model/cache generation очищает RAM.
LRU вытесняет старые записи, копируемая reader-потоком запись защищена.
Отменённое или неудачное чтение не публикует частичный payload.
Вытесненный буфер того же размера переиспользуется; при pressure trim
память освобождается без резервного пула. Allocation/metadata OOM пропускает
admission, сохраняя доставку уже прочитанного блока.

Это pageable RAM-копии, не CUDA pinned100GB и не предварительная загрузка
всего GGUF. Наполнение происходит по запросам router; GPU cache hits обходят
RAM tier. Один и два async readers, synchronous pinned путь и tensor batching
используют одинаковый кэш. Общий pipeline получил необязательный SourceRead
callback; другие backends его не передают и сохраняют прежнюю ветку.

Budget рассчитывается из физической available RAM и commit headroom:
оставляет7% total+512 МиБ физической памяти и512 МиБ commit; давление памяти
уменьшает cache до независимого95% global guard.
Максимальный cap фиксируется по headroom при создании; paging старых
страниц не даёт разрешения раздувать private allocations дальше. Размер payload не является
измерением residency: Windows вправе вытеснять pageable pages. `ram_cache`
в request JSON показывает bytes/budget/entries/pending, hits/misses,
hit/file/fill bytes, evictions/rejected/OOM. Отдельные prefill/decode bytes.
File bytes — объём native reads, включая возможные hits файлового кэша ОС;
физический SSD traffic этим счётчиком не измеряется.

Engine SHA-256:
`9422341b55e21665522c6e4c5a149eee894e95d64a13f840ec8877f715255b30`.
Новый patch marker `hy3-managed-ram-cache`; source hashes сохранены в
[HY3_RAM_PYTHON_VALIDATION.json](HY3_RAM_PYTHON_VALIDATION.json).

**Проверки:**

- [HY3_RAM_CPU_VALIDATION.json](HY3_RAM_CPU_VALIDATION.json):36 checks,
  включая1 200 concurrent native reads, exact bytes, LRU, file identity,
  давление RAM/commit, partial cancel, I/O error, fallback и reset.
- [HY3_RAM_RUNTIME_VALIDATION.json](HY3_RAM_RUNTIME_VALIDATION.json):
  **544 checks PASS**, из них60 новых RAM-проверок: F32/mixed,batch1/17,
  readers0/1/2, tensor batching, холодные/прогретые logits и каждый GPU-copy
  побитово. Прогретый cache при tiny VRAM-cache обслужил все source bytes
  без native file reads; новая generation освободила entries.
- [HY3_RAM_MTP_VALIDATION.json](HY3_RAM_MTP_VALIDATION.json):59 MTP fixtures PASS.
- [HY3_RAM_PIPE_VALIDATION.json](HY3_RAM_PIPE_VALIDATION.json):5×22 checks PASS,
  RAM off/sync/readers1/readers2/MTP1, exact greedy и seeded sampling IDs,
  отмена, восстановление, invalid requests, изоляция и QUIT.
- [HY3_RAM_PYTHON_VALIDATION.json](HY3_RAM_PYTHON_VALIDATION.json):48 CPU tests
  и9 native CLI checks PASS; export auto/cap и отказ неверных параметров.

**Полный GGUF:**
[HY3_RAM_MODEL_VALIDATION.json](HY3_RAM_MODEL_VALIDATION.json) —12 генераций:
en/ru/zh/code/numbers/long при RAM auto и MTP0/1. Все greedy IDs совпали с
историческим эталоном, off logits также побитово. Каждый режим прошёл
prefill/decode STOP с восстановлением и seeded sampling fallback; guards без
ошибок, оба exit0. Decode cancel settle1,25/2,26 мс после STOP send.
Long prompt362: prefill98,047/97,759 с, cache payload74,668/72,259 ГиБ;
global peak RAM110,709/109,118 ГиБ, VRAM27 178/28 589 МиБ. Эти одиночные
наблюдения не доказывают эффект MTP или RAM-кэша на prefill throughput.
В long request RAM обслужила741,3/742,3 ГиБ повторных чтений, native reads —
50,3/54,1 ГиБ; физический SSD traffic не измерен.

[HY3_RAM_HTTP_VALIDATION.json](HY3_RAM_HTTP_VALIDATION.json):7 ответов и
1 disconnect/recovery PASS при RAM auto + MTP1. Проверены OpenAI/Anthropic
JSON/SSE, точные math/tool-call/tool-result IDs, отмена prefill и следующий
запрос. Disconnect settle2,769 с; exit0, memory guard без ошибок. Listener закрыт.
Финальное сравнение LRU RAM off/auto/auto/off завершено:
[HY3_RAM_LRU_BENCHMARK.json](HY3_RAM_LRU_BENCHMARK.json), все16 генераций
и4 восстановления exact. Прогретая en/code пара: off2,873/2,719,
auto2,123/2,384 ток/с. Улучшение против Windows file cache не получено.
Профили `build-local/hy3-http-ram-mtp0/hy3.json` и
`build-local/hy3-http-ram-mtp1/hy3.json`: target cache8 ГиБ/readers2/chunk4,
RAM auto, context2048/batch17/F32 KV; graphs/TF32/fusion/batch-copy off.
Первое заполнение имеет стоимость allocation+copy; скорость не объявляется
лучше файлового кэша Windows до завершения сравнения.

Первый вариант без allocation reuse сохранён в
[HY3_RAM_INITIAL_BENCHMARK.json](HY3_RAM_INITIAL_BENCHMARK.json), engine
`9a586165…` в `build-local/hy3-ram-initial`. Все16 коротких генераций и4
восстановления exact logits/IDs. Вторые en/code проходы:

| RAM cache | Decode, ток/с | Prefill пары, с | Полное время пары, с | Global peak RAM, ГиБ |
|---|---:|---:|---:|---:|
| off до | 2,081 | 14,031 | 31,812 | 47,054 |
| auto первый | 2,031 | 13,799 | 32,018 | 116,405 |
| auto второй | 2,254 | 11,780 | 28,197 | 116,095 |
| off после | 2,770 | 10,227 | 23,586 | 41,691 |

Pooled off2,376 против auto2,136 ток/с: около−10,1%, с большим разбросом
во времени. Кэш удерживал около75 ГиБ payload, система использовала до92,71%
RAM; все memory guards прошли. Повторные allocations при eviction — гипотеза
части накладных расходов, не отдельный измеренный causal contribution.
В итоговой сборке добавлены reuse counters и сохранение буфера того же размера;
этот первоначальный результат не выдаётся за результат новой сборки.

**Не закрыто на HY3-11:** предварительный прогрев и frequency/admission policy RAM,
ограничение admission GPU во время prefill (перенесено в HY3-13), physical SSD counters,
длинный randomized corpus и реальное внешнее давление памяти. RAM-cache
не закрепляет страницы ОС и не гарантирует заполнение до95% на коротком запросе.

### HY3-12 — 2026-10-06, Asia/Yekaterinburg — Частотный RAM-кэш

Основание: тот же commit4336d41a и сохранённые dirty changes HY3-09/10/11.
Пользователь попросил оставлять востребованных экспертов в RAM, редких —
в файле, и проверить пользу занятой памяти. В HY3-11 RAM hit bytes были
ненулевыми и составляли около88% source bytes в одном прогретом проходе,
но LRU проиграл Windows file cache по throughput. Allocation payload не
выдаётся за физическую residency, native reads — за физический SSD traffic.

Изменено: `host_cache.hpp` получил frequency admission и decay. На первом
обращении chunk проходит только через pinned staging, на повторном decode
становится кандидатом в RAM. При нехватке места сравнивается частота с64
кандидатами из LRU tail; равная частота сохраняет resident entry. Весь набор
victims выбирается до eviction; блок того же размера переиспользуется.
История ограничена131072 keys, счётчики делятся на2 каждые131072 decode
source accesses. История переживает eviction/reset counters, но weak file
identity исключает наследование статистики новым Source по прежнему адресу.
VRAM hits не обучают RAM tier: учитывается спрос на данные, которых нет в VRAM.

Prefill читает уже сохранённые chunks, но не заполняет RAM-кэш и не обновляет
частоту/recency. Фаза задаётся main явно после drain: MTP verify batch остаётся
decode независимо от числа токенов. Добавлены counters prefill/frequency bypass,
history size, victim candidates и отдельные prefill/decode fill bytes.
`--ram-cache-policy frequency|lru` экспортируется в отдельные профили.
При включённом RAM tier выбран frequency; сам RAM tier по умолчанию выключен.
Роутинг модели, число экспертов и вычисления не менялись.

Первичный двухпроходный benchmark сохранён в
[HY3_FREQUENCY_INITIAL_BENCHMARK.json](HY3_FREQUENCY_INITIAL_BENCHMARK.json),
engine `e47e5a97…` в `build-local/hy3-frequency-initial`. Вторые en/code пары:
RAM off2,810/2,885; frequency auto2,342/2,384; LRU auto2,424 ток/с.
Frequency продолжал прогрев: во второй паре добавил21,946 ГиБ, к её концу
удерживал51,479 ГиБ. Ускорение не получено; третий повтор выделен отдельно.
Auto budgets различались со стартовым headroom, поэтому финальный опыт
использует числовой RAM cap64 ГиБ и3 повтора, GPU cache8 ГиБ/readers2/chunk4.

Финальная сборка
`a1117b6d714699ed5e57a25724fa36f4e9618d92edc6ccecf75611314ee2f861`
сохранена в `build-local/hy3-frequency/strata-hy3.exe`.
Добавлено восстановление latched reader error до смены фазы следующего запроса.
`ram_reused_payload_bytes` показывает сохранённые chunks, из которых уже было
хотя бы одно чтение после admission; `ram_unreused_payload_bytes` — сохранённые
chunks без такого чтения. Gauges переживают reset counters и уменьшаются при
вытеснении; это диагностика повторного использования, не OS residency.

Проверки финальной сборки, все PASS/exit0:

- [CPU](HY3_FREQUENCY_CPU_VALIDATION.json):193 checks,1200 concurrent reads,
  cold-scan resistance, promotion, decay при смене нагрузки, multi-victim admission,
  bounded history, exact bytes, reuse gauges, RAM/commit pressure, cancel/error.
- [Runtime](HY3_FREQUENCY_RUNTIME_VALIDATION.json):652 checks; LRU/frequency,
  F32/mixed, batch1/17, readers0/1/2; exact logits и GPU-copy bytes. В12 сочетаниях
  read error перед новой фазой восстановлен, prefill не публикует RAM entries.
- [MTP](HY3_FREQUENCY_MTP_VALIDATION.json):59 fixtures.
- [Pipe](HY3_FREQUENCY_PIPE_VALIDATION.json):6×22 checks, RAM off/sync/readers1/2,
  MTP1 и LRU; greedy/seeded IDs, STOP/recovery, errors/session isolation/QUIT.
- [Python/CLI](HY3_FREQUENCY_PYTHON_VALIDATION.json):49 tests и13 native CLI checks.
  Сохранены команды, source hashes, exe/manifest SHA и raw logs.

Команды: `cmd /c build-local\build-hy3-cache.bat` (log
`build-local/hy3-frequency-build-final.log`),
`python -X utf8 build-local/check-hy3-frequency-final.py` — последовательные
CPU/runtime/MTP/pipe/CLI проверки и export. Профили
`build-local/hy3-http-frequency-mtp0/hy3.json` и
`build-local/hy3-http-frequency-mtp1/hy3.json` указывают на сохранённый exe:
RAM cap65536 МиБ/frequency, GPU cache8192 МиБ/readers2/chunk4, context2048,
batch17/F32 KV, temperature0; graphs/TF32/fusion/batch-copy off.
Это экспериментальные профили, глобальный RAM-cache default остаётся0.
Полный GGUF: [HY3_FREQUENCY_MODEL_VALIDATION.json](HY3_FREQUENCY_MODEL_VALIDATION.json)
и компактный [HY3_FREQUENCY_BENCHMARK.json](HY3_FREQUENCY_BENCHMARK.json).
**24 генерации PASS**: en/code×3 повтора×4 режима; все IDs exact,
MTP-off logits побитово равны историческому oracle. В каждом процессе прошли
prefill/decode STOP с восстановлением и seeded sampling fallback; все exit0,
monitor_error=null. Повтор HTTP с новой policy отдельно не проводился
(API code не изменялся, pipe/export проверены).

Команда полного опыта:

```powershell
python -X utf8 tools/check_hy3_mtp.py --engine build-local/hy3-cuda/bin/strata-hy3.exe --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --reference docs/hy3/HY3_MODEL_VALIDATION.json --cuda-bin build-local/cuda-13.0/bin/x64 --pipeline-readers 2 --variants 8192:0 8192:0:65536:frequency 8192:1:65536:frequency 8192:0 --repeats 3 --lifecycle --output-dir build-local/hy3-tests/frequency-benchmark-final
```

RTX5090,125,555 ГиБ RAM, один процесс за раз; GPU cache8 ГиБ,
readers2/chunk4, context2048/batch17/F32 KV, temperature0, profiler off.
Третья en/code пара в каждом процессе,37 useful decode tokens:

| Режим | Decode, ток/с | Prefill пары, с | Полное время пары, с | RAM payload в конце, ГиБ |
|---|---:|---:|---:|---:|
| RAM off до | 2,766 | 9,589 | 22,965 | 0 |
| Frequency, MTP0 | 3,461 | 10,186 | 20,876 | 51,809 |
| Frequency, MTP1 | 3,698 | 11,265 | 21,269 | 52,947 |
| RAM off после | 2,948 | 9,666 | 22,214 | 0 |

Pooled off2,854 ток/с → frequency **+21,26% decode** в этом коротком
прогретом опыте. Полное время пары без MTP сократилось примерно на7,6%.
MTP1 добавил6,85% decode, но полное время пары оказалось немного больше:
prefill11,265 против10,186 с. Это не основание объявлять MTP глобально быстрее.
Поэтому оба профиля сохранены отдельно, глобальные defaults не переключены.

**Цена прогрева:** первые две пары frequency MTP0 дали2,101/2,383 ток/с
и добавили29,534/21,946 ГиБ в RAM. После второй пары из51,479 ГиБ payload
только33,611 ГиБ уже обслужили хотя бы одно чтение после admission.
После третьей пары из51,809 ГиБ уже использованы51,268 ГиБ, осталось0,541 ГиБ
без cache hit. Добавлено только0,330 ГиБ/132 allocations; decode получил
113,068 ГиБ из RAM и0,356 ГиБ через ReadFile — около99,69% decode source bytes
из managed cache. H2D113,424 ГиБ такой же, как у RAM off: этот tier сокращает
native source reads, но не объём передачи на GPU. В модели этого короткого
corpus RAM eviction не потребовался; замену горячих/холодных entries проверили
на CPU fixtures. Пользу при новой теме или тесном RAM cap ещё нужно измерить.

Sampled global peaks: MTP0 **86,462 ГиБ RAM /26 768 МиБ VRAM**, MTP1
**89,935 ГиБ /28 801 МиБ**. Для MTP1 фактический budget был ограничен
headroom ниже64 ГиБ (62,793 ГиБ на третьей паре). Все95% guards прошли;
RAM≈0,5с/GPU≈2с sampling не исключает короткие пики. Decode STOP settle
1,25/2,25 мс. RAM read/fill counters и retained payload не являются
измерением физического SSD traffic или гарантированной OS residency.

Ограничение замера: последовательный короткий повторяющийся corpus, OS cache
не очищался; по одному процессу на frequency/MTP и два off-контроля, без
confidence interval. Ускорение после прогрева нельзя переносить на первый
запрос, перезапуск модели, другую тему или длинный diverse corpus.

Не закрыто: physical SSD/residency telemetry, длительный разнообразный corpus,
реальное внешнее давление RAM и GPU prefill admission. Частотный RAM-кэш
не обещает выигрыш по скорости или заполнение памяти до95%.

### HY3-13 — 2026-10-06, Asia/Yekaterinburg — GPU prefill policy и быстрый профиль без MTP

**Основание.** RTX5090/32607 МиБ, Ryzen9 9950X, Windows,125,555 ГиБ RAM;
тот же полный127,083 ГиБ GGUF/header `f3307357…`, dependency86ebfef2,
CUDA13.0/sm_120/Release. В ходе проверки HEAD стал `819ad13`; старые
профили и exe не перезаписаны. Измеренный engine сохранён отдельно:
`build-local/hy3-gpu-prefill/strata-hy3.exe`, SHA-256
`7be63e00a3a553063fb8c4172f6dcb373ceff6daa7732dc4a21e8b7c85b26247`.
Manifest и hashes generated runtime находятся в correctness-отчётах.

**Изменено.** `--gpu-cache-policy all|decode`, прежний default `all`.
В `decode` prefill обслуживает существующие GPU hits, но не делает admission
и не меняет history/LRU. Фаза задаётся driver после drain; MTP verify batch
остаётся decode. Policy действует в sync, per-matrix и tensor-batch pipeline,
при RAM cache on/off. В общем Step `ExpertCache::get` появился параметр
`train=true`; прежние callers сохраняют своё поведение. В JSON добавлены
policy, bypass и отдельные prefill/decode GPU fill bytes; exporter и sweep
принимают новый флаг. Router IDs, веса, квантизация и вычисления не менялись.

**Почему это проверялось.** Отдельный instrumented run старого a111 engine:
[HY3_GPU_PREFILL_PROFILE_DIAGNOSTIC.json](HY3_GPU_PREFILL_PROFILE_DIAGNOSTIC.json).
На третьей en/code паре delivery waits заняли6,415 с из12,473 с generation;
GPU admission —1,326 с включительно. Это диагностические CPU scopes,
их нельзя складывать с parallel reader times или сравнивать по throughput
с обычным benchmark. PCIe под нагрузкой показал5.0×16. Для полного GGUF
startup compute buffer948,757 МиБ; предположение о5 ГиБ scratch не подтвердилось.

**Скорость без instrumentation.**
[HY3_GPU_PREFILL_BENCHMARK.json](HY3_GPU_PREFILL_BENCHMARK.json) хранит36
измеренных генераций, параметры, IDs/logit hashes, память, raw paths/SHA
и STOP/recovery. Context2048, batch17, F32 KV, temperature0, MTP0,
readers2/chunk4, RAM cap65536 МиБ/frequency; graphs/fusion/TF32 off.
Каждая строка — третий проход en/code в отдельном процессе после двух
обучающих проходов;37 полезных decode tokens на пару. OS cache не очищался.

| Вариант в порядке выполнения | Decode ток/с | Prefill пары, с | Вся пара, с |
|---|---:|---:|---:|
| Контроль: cache8/all/batch0 | 3,396 | 11,211 | 22,104 |
| cache8/decode/batch0 | 3,679 | 9,797 | 19,855 |
| cache11/decode/batch0 | 3,750 | 9,429 | 19,295 |
| cache11/decode/batch1 | 4,460 | 9,342 | 17,639 |
| Повтор cache11/decode/batch1 | 4,608 | 9,180 | 17,209 |
| Обратный контроль cache8/all/batch0 | 3,541 | 10,042 | 20,491 |

Pooled двух контролей и двух выбранных runs: **3,467→4,533 ток/с (+30,7%)**.
Среднее время полной пары **21,298→17,424 с (−18,2%)**. Это короткое
последовательное сравнение с обратным контролем, не доверительный интервал
и не доказательство той же скорости на произвольной новой теме.
Первые пары лучшего профиля дали1,637/2,039 ток/с, контроля1,912/1,954:
ускорение холодного старта не установлено, RAM admission сначала стоит времени.

В парном cap11 сравнении batching сократил decode fences **73482→8769**;
скорость3,750→4,460 (+18,9%). Это уже существующий transport mode,
выбранный заново после появления RAM cache и GPU decode policy.
Средний H2D лучшего профиля **99,370 ГиБ на37 токенов** против113,424 у
контроля (−12,4%); около2,686 ГиБ на токен всё ещё пересылаются на GPU.
На третьей паре обоих лучших runs native source reads в decode равны0,
RAM cache обслуживает все H2D misses.97,2–97,6% оставшегося RAM payload
уже использовались после admission. Эти counters не измеряют SSD traffic
или физическую резидентность страниц Windows.

Пики всех коротких runs: **89,524 ГиБ RAM /30252 МиБ VRAM**; выбранные
профили дали88,563/87,656 ГиБ RAM и30252/30222 МиБ VRAM. Лимит95% сохранён.
Capacity — верхняя граница, controller уменьшает её по реальному headroom.

**Проверки кода.** Сборка `cmd /c build-local\build-hy3-cache.bat`, exit0.
29 allocator/policy cases, включая lookup без обучения;
796 runtime cases, из них144 новых cold/decode/warm-prefill checks;
59 MTP fixtures и50 Python tests прошли. Отчёты:
[cache](HY3_GPU_PREFILL_CACHE_VALIDATION.json),
[runtime](HY3_GPU_PREFILL_RUNTIME_VALIDATION.json),
[MTP](HY3_GPU_PREFILL_MTP_VALIDATION.json),
[Python](HY3_GPU_PREFILL_PYTHON_VALIDATION.json).
В36 benchmark генерациях token IDs и все logits побитово совпали с
историческим native reference; `prefill_gpu_fill_bytes=0` в decode policy.
Полноразмерные модели других backends в этой итерации не запускались.

Итоговый профиль прошёл полный6-prompt corpus: en/ru/zh/code/numbers/long,
побитовые logits и точные IDs; STOP во время prefill и generation,
оба recovery и seeded sampling также PASS, exit0.
[HY3_GPU_PREFILL_MODEL_VALIDATION.json](HY3_GPU_PREFILL_MODEL_VALIDATION.json).
Длинный prompt362 обработан за70,128 с в этой последовательности;
это correctness-наблюдение, не парное измерение ускорения длинного prefill.
Пик75,757 ГиБ RAM /30249 МиБ VRAM. Шесть synthetic pipe вариантов
(RAM off/on, sync/readers1/2, batch0/1, MTP0/1) прошли по22 checks;
четыре неверных значения policy отклонены до загрузки модели:
[HY3_GPU_PREFILL_PIPE_VALIDATION.json](HY3_GPU_PREFILL_PIPE_VALIDATION.json).
Повторный полный HTTP/tool-dialogue прогон новой конфигурации не выполнялся;
Python HTTP regressions входят в указанные50 tests.

**Выбранный отдельный профиль.**
`build-local/hy3-http-speed13/hy3.json`, SHA-256
`0c028edfc5e81f401bd1d9f3d5c82b1989f0d618c8c959bfa6e9e7a37669dcc3`,
ссылается на сохранённый exe7be63e00. MTP0, cache11264 МиБ/decode,
readers2/chunk4/batch1, RAM cap65536 МиБ/frequency. Engine defaults и
старые HY3-12/MTP профили сохранены. Identity, checksum и аргументы:
[HY3_GPU_PREFILL_PROFILE_VALIDATION.json](HY3_GPU_PREFILL_PROFILE_VALIDATION.json).
Запуск явно:

```powershell
python -m serve.server --engine strata --config build-local/hy3-http-speed13/hy3.json --port 8094
```

**Воспроизведение измерений.**

```powershell
python tools/check_hy3_mtp.py --engine build-local/hy3-gpu-prefill/strata-hy3.exe --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --reference docs/hy3/HY3_MODEL_VALIDATION.json --cuda-bin build-local/cuda-13.0/bin/x64 --variants 8192:0:65536 --gpu-cache-policy all decode --repeats 3 --output-dir build-local/hy3-tests/policy-new
python tools/check_hy3_mtp.py --engine build-local/hy3-gpu-prefill/strata-hy3.exe --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --reference docs/hy3/HY3_MODEL_VALIDATION.json --cuda-bin build-local/cuda-13.0/bin/x64 --variants 11264:0:65536 --gpu-cache-policy decode --pipeline-batch 0 1 --repeats 3 --output-dir build-local/hy3-tests/cap11-new
```

Затем повторить второй вариант cap11/batch1 и контроль cap8/all/batch0
в двух новых процессах, как записано в raw reports. Не включать
`--profile-delivery` в speed comparison.

**Осталось.** Длительная генерация и рандомизированная смена тем,
скорость при вытеснении RAM payload, новый GPU timeline лучшего профиля.
Следующий шаг — измерить оставшиеся ожидания по router/tensor boundaries
с теми же exact checks; не приписывать весь generation wall time PCIe.

### HY3-14 / P3.4 — 2026-10-06, Asia/Yekaterinburg — Плотное размещение матриц в VRAM

**Основание.** HEAD `819ad13` с существующими изменениями; dependency `86ebfef2`,
тот же GGUF/header и RTX5090. Собранный engine SHA-256
`19369d5809e594269c25932ccd5c3d6d099afe34de02cd3ef68819ea0e9878e8`
сохранён отдельно: `build-local/hy3-gpu-arena/strata-hy3.exe`.
Команды сборки, exit0, manifest и hashes исходников:
[HY3_GPU_ARENA_BUILD_VALIDATION.json](HY3_GPU_ARENA_BUILD_VALIDATION.json).

**Чем была занята VRAM.** Старый engine7be63e00 с MTP0, context2048,
batch17/F32 KV, readers2/chunk4/batch1 и RAM cache off сравнивался при GPU
cap1 МиБ (ни одна матрица не помещается) и cap11264 МиБ. После English
полезный cache payload составил0 и10,263 ГиБ, global used —14,282 и29,995 ГиБ.
Разница сверх payload — около5,45 ГиБ. Это оценка между двумя процессами,
включающая колебания внешней нагрузки, а не точная классификация каждого байта:
[HY3_VRAM_BASELINE_DIAGNOSTIC.json](HY3_VRAM_BASELINE_DIAGNOSTIC.json).

Отдельный allocation probe подтвердил накладные расходы на этом Windows/CUDA:
948 матриц с теми же размерами после округления занимают2,461 ГиБ payload.
Отдельные `cudaMalloc` дали CUDA delta3,734 ГиБ в обоих повторах; один общий
allocation —2,461 ГиБ. Все диапазоны были заполнены и освобождены между
вариантами. Это проверка размещения, не benchmark inference:
[HY3_VRAM_ALLOCATION_DIAGNOSTIC.json](HY3_VRAM_ALLOCATION_DIAGNOSTIC.json).

Backend breakdown нового engine показывает8,604 ГиБ model buffers,
1,250 ГиБ target context/KV и0,927 ГиБ compute; GPU transport ring —0,016 ГиБ.
К этому добавляются hot matrices, их allocation overhead и CUDA/driver/desktop.
Payload кэша — сами часто используемые матрицы, а не дополнительная копия
полного набора экспертов. Редкие матрицы остаются в GGUF и доставляются по спросу.

**Что перенесено из соседних реализаций.** DeepSeek fixed cache выделяет одну
arena и ищет подходящую по размеру жертву в64 старых entries; его dynamic mode
использует individual allocations. Step использует отдельные allocations и
тот же controller, от которого начинал HY3. В HY3 добавлены растущие блоки
до64 МиБ и отдельные size classes, чтобы освобождать пустые блоки при давлении.
Режим `--gpu-cache-allocator arena` включается явно; `individual` и поведение
Step по умолчанию сохранены. Shared target/draft scratch из DeepSeek не решает
этот случай: измерения HY3 здесь выполняются с MTP0.

Замена ищет до64 unpinned entries того же размера, с частотой не выше входящей
матрицы. Она работает и при отказе выделить новый блок. Уже запланированные
hits защищены pins; слот переиспользуется после завершения GPU consumers.
Пустой блок сразу возвращается CUDA. При сильном давлении синхронизированная
граница decode может очистить весь необязательный кэш для возврата блоков.
Перед новым allocation сохраняются global reserve5%+256 МиБ и независимый95% guard.

В предварительной реализации controller учитывал лишь live payload. Он мог
ошибочно считать пустые слоты внутри своих блоков внешней занятой памятью,
сокращая live cache до9,341 ГиБ при15,688 ГиБ backing. Исправленный refresh
учитывает backing целиком; заполнение уже выделенных слотов не требует новой
VRAM. Старые диагностические прогоны сохранены:
[HY3_GPU_ARENA_EXPERIMENTS.json](HY3_GPU_ARENA_EXPERIMENTS.json).
В первых двух были сторонние MiMo GPU fixtures на старте; они не используются
как итоговое сравнение скорости.

Добавлены JSON gauges `cache_backing_bytes`, `cache_backing_slack_bytes`,
block allocation/free counters и отдельный `cache_arena_budget_rejects`.
Последний также входит в старый `cache_oom`, поэтому рост `cache_oom` сам по
себе не означает настоящий driver OOM. В individual mode backing gauges равны0
как признак отсутствия такого учёта; это не нулевой расход GPU-кэша.

**Проверки.** 18 arena cases (плотность, сохранность соседних слотов, release,
pins, pressure, fallback reuse и заполнение holes при исчерпанном headroom),
29 общих cache cases, по796 runtime cases в arena/individual и59 MTP fixtures
прошли. Python:51 tests PASS. Полноразмерные Step/DeepSeek модели в этой
итерации не запускались. Отчёт:
[HY3_GPU_ARENA_VALIDATION.json](HY3_GPU_ARENA_VALIDATION.json).

**Скорость без MTP.** Последовательность arena16 → individual11 → arena16,
по5 пар English/code в каждом новом процессе. Context2048, batch17/F32 KV,
GPU policy decode, readers2/chunk4/batch1, RAM cap65536 МиБ/frequency,
temperature0, CUDA graphs/TF32/fusion и delivery profiler выключены.
Все30 генераций совпали с историческим reference по IDs и побитовым logits;
STOP/recovery тоже PASS. Сторонних engine-процессов в этих трёх speed runs
монитор не обнаружил. ОС file cache не очищался.

Для сравнения взяты последние2 пары после3 прогревочных. На пару приходится
37 полезных decode-токенов; первый выходной токен входит в prefill.

| Режим | Полезные матрицы, ГиБ | CUDA backing, ГиБ | Decode, ток/с | Время всей пары, с | H2D на пару, ГиБ |
|---|---:|---:|---:|---:|---:|
| Individual, cap11 | 10,270–10,325 | Отдельно не измеряется | 4,388 pooled; 4,255–4,530 | 17,617 | 97,708 |
| Arena, cap16, два процесса | 15,218–15,279 | 15,688–15,750 | 5,755 pooled; 5,659–5,869 | 14,442 | 74,985 |

Decode **+31,1%**, время полной пары **−18,0%**, H2D **−23,3%**.
У arena остаётся около0,47 ГиБ padding/пустых слотов. Это рост полезного
размещения примерно на48%, а не увеличение общей доступной VRAM.
Последние пары уже почти не читали decode misses из native source: остаток
8–20 МиБ на пару у arena и0 у individual. Эти counters не измеряют физический
SSD traffic. Первые пары arena дали2,249/2,293, control2,010 ток/с;
скорость5,755 относится к прогретому короткому corpus, не к холодному запуску.

Пики arena runs: **85,843/86,004 ГиБ RAM**, **30423/30329 МиБ VRAM**;
control92,442 ГиБ RAM /30305 МиБ VRAM. Sampled global guard95% сохранён.
NVML `used` в этих peaks и `total-free` в начальной диагностике — разные
показатели драйвера; их нельзя складывать как один точный ledger.
Полные параметры, повторы, hashes и request metrics:
[HY3_GPU_ARENA_BENCHMARK.json](HY3_GPU_ARENA_BENCHMARK.json).

**Отдельный профиль без MTP.** `build-local/hy3-http-arena14/hy3.json`,
SHA-256 `b6376f9150f2b715001ac3f8cca4528723cb492affb9a84a6d4bf49bc19be809`:
arena cap16384 МиБ, policy decode, readers2/chunk4/batch1, RAM cap65536 МиБ/frequency.
Ссылается на неизменяемую копию exe19369d58. Экспортёр проверил identity,
checksum и аргументы; профиль HY3-13 и его exe остались побитово прежними:
[HY3_GPU_ARENA_PROFILE_VALIDATION.json](HY3_GPU_ARENA_PROFILE_VALIDATION.json).

```powershell
python -m serve.server --engine strata --config build-local/hy3-http-arena14/hy3.json --port 8094
```

Полный6-prompt corpus en/ru/zh/code/numbers/long прошёл: побитовые logits,
точные IDs, STOP во время prefill/generation, оба recovery и seeded sampling.
Пик74,348 ГиБ RAM /30288 МиБ VRAM. Отчёт:
[HY3_GPU_ARENA_MODEL_VALIDATION.json](HY3_GPU_ARENA_MODEL_VALIDATION.json).
Шесть synthetic pipe вариантов (RAM off/on, readers0/1/2, batch0/1, MTP0/1)
прошли по22 checks; четыре неправильных allocator values отклонены до загрузки:
[HY3_GPU_ARENA_PIPE_VALIDATION.json](HY3_GPU_ARENA_PIPE_VALIDATION.json).
Повторный full-model HTTP/tool-dialogue с arena не выполнялся;51 Python tests
включают HTTP regressions. Длительная генерация, смена тем и реальное внешнее
pressure/residency/SSD измерение остаются открытыми.

**Дополнительный тест MTP.** По запросу пользователя выполнен MTP off/on/off
с depth1, тем же arena/RAM profile, по5 повторов и lifecycle checks в каждом
процессе. В последних2 прогретых парах:

| Режим | Полезный decode, ток/с | Полное время пары, с | Prefill пары, с | H2D пары, ГиБ | Hot matrices, ГиБ |
|---|---:|---:|---:|---:|---:|
| MTP0, контроль до/после | 5,869 pooled; 5,849–5,887 | 14,277 | 7,972 | 75,264 | 15,160–15,218 |
| MTP1 | 6,424 pooled; 6,413–6,436 | 13,996 | 8,237 | 80,235 | 13,461 |

MTP дал **+9,46% decode**, но лишь **−1,97% полного времени** этой короткой
пары. Приняты34/38 черновиков (**89,47%**):7/7 на English и10/12 на code
в каждом повторе. Постоянные GPU model buffers выросли8,604→10,333 ГиБ,
то есть на1,729 ГиБ; доступный hot set основных экспертов уменьшился.
H2D вырос на6,60%. Это сравнение реальных режимов с ценой MTP в памяти;
отдельно влияние verification compute и уменьшения cache capacity не изолировано.

Все30 greedy генераций сохранили точные reference IDs; MTP-off logits —
побитовые, для MTP-on сохранены числовые diagnostics. Prefill/decode cancel,
recovery и seeded sampling fallback прошли во всех3 процессах, exit0.
Пик MTP1:88,805 ГиБ RAM /30252 МиБ VRAM; общий пик всей последовательности
88,805 ГиБ /30446 МиБ. Сторонних engine samples нет. Новый6-prompt и HTTP
прогон с MTP1 не выполнялся; текущая MTP1 проверка ограничена English/code
и lifecycle. Отчёт:
[HY3_GPU_ARENA_MTP_BENCHMARK.json](HY3_GPU_ARENA_MTP_BENCHMARK.json).

Дополнительно сохранён `build-local/hy3-http-arena14-mtp1/hy3.json`, SHA-256
`2d9396cd412d6d4ed5dccca64ec9d1929dfc67ee7e6f561661c33677ec932742`.
Он использует тот же exe19369d58 и настройки arena14, добавляя `--mtp 1`.
Оба профиля доступны явно; универсальный engine default остаётся MTP0:
[HY3_GPU_ARENA_MTP_PROFILE_VALIDATION.json](HY3_GPU_ARENA_MTP_PROFILE_VALIDATION.json).

```powershell
python -m serve.server --engine strata --config build-local/hy3-http-arena14-mtp1/hy3.json --port 8094
```

**Воспроизведение.** Новые output directories обязательны. Для arena/control
выполнить первую команду с allocator/cap `arena/16384`, затем
`individual/11264`, затем снова `arena/16384`; не запускать параллельно.

```powershell
python tools/check_hy3_mtp.py --engine build-local/hy3-gpu-arena/strata-hy3.exe --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --reference docs/hy3/HY3_MODEL_VALIDATION.json --cuda-bin build-local/cuda-13.0/bin/x64 --variants 16384:0:65536 --gpu-cache-policy decode --gpu-cache-allocator arena --pipeline-batch 1 --repeats 5 --output-dir build-local/hy3-tests/arena-new
python tools/check_hy3_mtp.py --engine build-local/hy3-gpu-arena/strata-hy3.exe --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --reference docs/hy3/HY3_MODEL_VALIDATION.json --cuda-bin build-local/cuda-13.0/bin/x64 --variants 16384:0:65536 16384:1:65536 16384:0:65536 --gpu-cache-policy decode --gpu-cache-allocator arena --pipeline-batch 1 --repeats 5 --lifecycle --output-dir build-local/hy3-tests/arena-mtp-new
```

### HY3-15 / P3.5 — 2026-10-06, Asia/Yekaterinburg — Общий кэш main/MTP и общий compute buffer

**Результат.** Подгружаемые MTP-эксперты реализованы и прошли correctness.
Фиксированного раздела VRAM для них нет: main и MTP используют один cache
controller, arena, RAM-кэш и асинхронный конвейер. Это увеличило полезный
main-кэш, но замедлило проверенный короткий corpus. Быстрый профиль
`build-local/hy3-http-arena14-mtp1/hy3.json` с resident MTP сохранён.
Новые режимы доступны явно; ускорением они не объявляются.

**Основание и сборки.** HEAD `819ad130f49d1e088e2fbc50e1e4f5c681e1e390`
с существующими изменениями. GGUF и header hash те же, что в HY3-14.
RTX5090 32607 МиБ, Ryzen9 9950X, 125,555 ГиБ OS-visible RAM, Windows,
CUDA13.0/sm_120, Release. Две сохранённые сборки:

- `build-local/hy3-streamed-mtp/strata-hy3.exe`, SHA-256
  `ed07f1772a5108d28b0adf66ad63115486522e1aa3747e1f99c32dd5516ce913`:
  первоначальная подгрузка с отдельным draft compute buffer.
- `build-local/hy3-streamed-mtp-shared/strata-hy3.exe`, SHA-256
  `eb06b5c3861adde10964be07fd37f8a798e1e2cbe6963dec3fe9c6ee11ae1e4e`:
  итоговая сборка с дополнительным shared scratch.

Build manifests, source hashes, команды и exit codes сохранены в
[HY3_STREAMED_MTP_VALIDATION.json](HY3_STREAMED_MTP_VALIDATION.json) и
[HY3_SHARED_MTP_VALIDATION.json](HY3_SHARED_MTP_VALIDATION.json).
Старые exe и профили не перезаписывались.

**Реализация.** `--mtp-experts resident|streamed` выбирает размещение трёх
routed tensors блока80. По умолчанию `resident`; `streamed` требует MTP1..3.
Контракт отдельно считает `mtp_routed_bytes`, loader оставляет эти матрицы
в native-file/mmap storage и регистрирует их в общем cache generation.
Ключи main/MTP различаются tensor ID; частоты, лимит и вытеснение общие.
Dense/NextN weights, embeddings/output и отдельные KV остаются на GPU.
Ни полного копирования экспертного тензора, ни CPU MoE вычислений нет.

Дополнительный `--mtp-shared-scratch 1` разрешён со streamed MTP. Механизм
владения compute buffer перенесён из локальной реализации DeepSeek; исходный
`ggml-alloc.c` проверен тем же SHA-256. Основной и draft контексты завершают
вычисления и копируют выходы в host до переключения; общий буфер создаётся
только до первого графа. При увеличении allocation разделяется безопасно.
Общие буферы не затрагивают KV, model weights и sampler state.

Метрики: `main_cache_bytes + mtp_cache_bytes == cache_bytes`,
`draft_gpu_context_bytes`, `draft_gpu_compute_bytes`, `mtp_shared_scratch`,
`mtp_scratch_saved_bytes`. Последнее поле означает первоначально освобождённый
allocation; активное разделение проверяется отдельно. При активном sharing
compute учитывается у target, draft compute равен0, чтобы не считать дважды.

**Память полной модели.** Подгрузка уменьшила model buffers с10,33336 до
8,73375 ГиБ: минус1638 МиБ. Но отдельный draft compute вырос с10,36 до651,49
МиБ, поэтому первый вариант освободил только996,87 МиБ суммарно.
Shared scratch убрал дополнительный draft allocation651,49 МиБ; общий
target compute остался948,76 МиБ. В последнем прогретом повторе:

| Режим итоговой сборки | Main cache, ГиБ | MTP cache, ГиБ | Backing общего cache, ГиБ |
|---|---:|---:|---:|
| MTP0 | 15,097 | 0 | 15,562 |
| MTP1 resident | 13,464 | 0; MTP weights в model buffer | 13,875 |
| MTP1 streamed + shared scratch | 14,675 | 0,305 | 15,438 |

Это увеличение main-кэша на1,211 ГиБ. MTP использует частотный отбор без
заранее закреплённых слотов. В итоговом сравнении sampled peak VRAM30378 МиБ,
RAM90,775 ГиБ; в первоначальном VRAM30564 МиБ, RAM90,712 ГиБ. Глобальные
95% guards не сработали. Пики включают другие системные процессы; монитор
не обнаружил сторонних engine процессов во время обоих сравнений.

**Измерение скорости.** Каждый процесс:5 одинаковых English/code пар,
37 useful decode tokens на пару; первые3 пары обучают кэши, последние2
сведены как total tokens / total decode time. Context2048, batch17, F32 KV,
temperature0, pipeline readers2/chunk4MiB/batch1, arena cap16384MiB,
GPU policy decode, RAM cap65536MiB/frequency. CUDA graphs/fusion/TF32 off.
Процессы последовательные; файловый кэш Windows не очищался. Полное время
пары ниже — сумма prefill и generation, без загрузки модели.

| Итоговая сборка eb06b5c3 | Decode, ток/с | Диапазон двух пар | Полная пара, с | Decode H2D, ГиБ/пару |
|---|---:|---:|---:|---:|
| MTP0 | 5,633 | 5,612–5,655 | 14,868 | 75,659 |
| MTP1 resident | **6,474** | 6,470–6,478 | **13,791** | 80,142 |
| MTP1 streamed + shared scratch | 5,854 | 5,735–5,978 | 16,677 | 77,489 |

Shared streaming снизил H2D примерно на3,3%, но decode оказался на9,6%
медленнее resident. Причина задержки не установлена одним счётчиком H2D;
увеличение VRAM-кэша само по себе не доказывает прирост скорости.
Первый вариант без sharing дал5,298 против6,428 ток/с resident в своей
сборке; его два последних повтора были нестабильнее (5,021–5,607).
Сравнивать5,298 и5,854 как строгий A/B эффекта sharing нельзя: это разные
последовательные серии. В обеих сериях resident быстрее своего кандидата.
Холодные пары итоговой сборки:2,233 /2,373 /2,435 ток/с соответственно;
прогретые числа не относятся к первому запросу или смене темы.

Отчёты со всеми повторами, настройками, памятью и request metrics:
[HY3_STREAMED_MTP_BENCHMARK.json](HY3_STREAMED_MTP_BENCHMARK.json),
[HY3_SHARED_MTP_BENCHMARK.json](HY3_SHARED_MTP_BENCHMARK.json).

**Correctness.** Итоговая сборка:177 MTP проверок (по59 для resident,
streamed и streamed+shared),796 runtime cases с arena, allocator sharing
parity/growth/lifetime,32 Python tests,4 pipeline/storage режима по22 pipe
сценария и4 отрицательных CLI проверки — PASS, exit0. Первоначальная сборка
дополнительно прошла29 проверок общего cache controller.
В MTP fixtures logits, IDs и draft acceptance между тремя размещениями
точные, включая forced rejection, cancel/recovery, penalties и границу KV.
На полном GGUF в каждой серии30 генераций: greedy IDs точные; MTP0 logits
побитово совпали с историческим reference;10 streamed запросов побитово
совпали с resident MTP той же сборки и сохранили proposed/accepted counts.
На последних2 парах MTP принял34/38 черновиков (89,47%). Prefill/decode
cancel, recovery и seed42/temperature0,7 fallback прошли в обеих сериях.
Это не утверждение о побитовом равенстве MTP и последовательного MTP0:
их batched CUDA logits могут различаться.

**Профиль.** `build-local/hy3-http-streamed15/hy3.json` экспортирован для
явных экспериментов с MTP1, streamed experts и shared scratch. Identity,
header, tokenizer/template проверены экспортером. Он не заменяет быстрый
resident профиль и не запускает сервер автоматически. Команда:

```powershell
python -m serve.server --engine strata --config build-local/hy3-http-streamed15/hy3.json --port 8094
```

Воспроизведение итогового сравнения (новый output directory обязателен):

```powershell
python tools/check_hy3_mtp.py --engine build-local/hy3-streamed-mtp-shared/strata-hy3.exe --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --reference docs/hy3/HY3_MODEL_VALIDATION.json --cuda-bin build-local/cuda-13.0/bin/x64 --variants 16384:0:65536 16384:1:65536 --mtp-experts resident streamed --mtp-shared-scratch 1 --gpu-cache-policy decode --gpu-cache-allocator arena --pipeline-batch 1 --repeats 5 --lifecycle --output-dir build-local/hy3-tests/streamed-mtp-new
```

**Не закрыто.** Новый6-prompt corpus, HTTP inference и длинная смена тем не
перезапускались; full-model comparison ограничен en/code и lifecycle.
Shared scratch сейчас проверен на Windows/CUDA0. Depth2/3 проверены fixtures,
но новые full-model speed series используют только depth1.
Следующий шаг — разделить время main verify на ожидание доставки и CUDA
compute при одинаковом маршруте и прогретом RAM, затем повторить A/B.
До измеренного выигрыша выбирать resident MTP для скорости.

## Правила обновления

- Каждая следующая запись имеет ID `HY3-01`, `HY3-02`, дату, исходный commit,
  dirty state/patch hashes, команды, exit code, результаты и нерешённые вопросы.
- Хранить JSON/logs с параметрами модели, dependency и шаблона; в статусе давать
  ссылки на реально созданные отчёты. Планируемые пути не выдавать за готовые.
- Время измерять отдельно для load/prefill/decode/request; tokens/s считать
  по принятым выходным токенам. Draft throughput не выдавать за output throughput.
- Записывать sampler/seed, prompt/generated counts, context/KV, batch/ubatch,
  MTP depth/acceptance/repair, cache/readers/chunks, GPU/driver/build flags.
- Память: global physical RAM и available, process commit/working set,
  pinned buffers, GPU used/total, cache allocation и peak. mmap file size
  не считать ни resident RAM, ни гарантированно свободной RAM.
- Оптимизация считается принятой после correctness и повторных A/B с одинаковыми
  условиями. Изменение cache capacity из-за MTP учитывать отдельным сравнением.
- Явно указывать пропущенные проверки. При неудаче оставить рабочий baseline
  и записать конкретную следующую проверку; не менять `TODO` на `DONE` по намерению.

## Шаблон следующей записи

```text
### HY3-NN / Pn.m — YYYY-MM-DD HH:MM, Asia/Yekaterinburg — Название
Основание: commit, dirty state, dependency/patch SHA, модель/header/template hash.
Изменено: файлы и поведение.
Команды: точные команды сборки и проверки.
Проверки: PASS/FAIL/SKIPPED, exit codes, fixtures/model corpus.
Результаты: correctness, tokens/s/TTFT, память, cache/H2D/SSD, условия и повторы.
Артефакты: ссылки на созданные JSON/logs.
Не закрыто: ошибки, ограничения и непроверенные режимы.
Следующий шаг: одна конкретная задача и критерий приёмки.
```
