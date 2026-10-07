# Статус внедрения MiMo-V2.6-Flash-RL

Обновлено: **2026-10-07**, `Asia/Yekaterinburg`.
План: [MIMO26_FLASH_IMPLEMENTATION_PLAN.md](MIMO26_FLASH_IMPLEMENTATION_PLAN.md).

Файл хранит фактический прогресс и точку продолжения.
**P0, P1, P3.1 и транспорт P3.2 реализованы: GPU engine с cache и async pipeline.**
Проверены context512, F32 KV, FA on, greedy, file/mmap, sync и bounded pipeline.
Defaults engine: cache до14 ГиБ с live clamp, блоки16 МиБ, mmap, один reader,
chunk8 МиБ, доставка по тензорам, decode-only frequency admission с decay65536.
CPU literal guard включён; экспериментальные групповые fills выключены.
API и установочный профиль ещё не готовы.
Существенный overlap H2D/compute пока не подтверждён; P3 в целом не закрыт.
Из P2 проверены IDs/bytes/tokenizer/raw Jinja.
Наличие `mimo2` в llama.cpp не означает готовую интеграцию в Strata.

## Текущее состояние

| Область | Статус |
|---|---|
| PREP-01 | DONE: исследование, план и этот файл |
| Исходная ревизия Strata | `1979cb6607c07faec22ab3c195c8c8e1e29d78f9`, с существующими изменениями |
| Модель | `H:\models\mimo-v2.6-flash\MiMo-V2.6-Flash-RL-GSQ-RCO-3.5bit.gguf` |
| Файл / архитектура | 125,712 ГиБ / `mimo2`, 472 тензора |
| Trunk | 48 слоёв: dense0 +47 MoE; 9 full +39 SWA attention |
| MTP / DFlash | В main GGUF нет; найдены3 MTP и2 DFlash sidecars. Offline probe добавлен; serving off |
| Vision/audio/video | Веса отсутствуют; локальный экспорт text-only |
| Header/ranges/contract | PASS: постоянный inspector, все 472 тензора, negative fixtures |
| Candidate | Изолированные CPU oracles Unsloth `86ebfef2`; native регистрация без weight allocation PASS |
| CUDA build | Unsloth `86ebfef2`, CUDA13.0.48, sm120; 3 локальные precision/stride поправки |
| CUDA операции | PASS 96/96: F32/BF16/Q2_K/Q3_K/MXFP4 и router, experts256/top-8 |
| Native MiMo graph | PASS 132/132: F32/mixed synthetic weights, FA off/on, full/SWA128 и state |
| Pinned source | Config `3b38d063`: 46 проверок PASS; `mimo2.cpp` `58367713` совпал с кандидатом |
| Tokenizer / stop | PASS 614/614 IDs + bytes; runtime EOS151645, без BOS/PAD/FIM stops |
| Raw Jinja / API | PASS 85/85 template checks; HTTP/parser/tools round-trip TODO |
| Python / native checks | 40 unit/regression tests; 3 CPU CTest; 23 cache +212 runtime +96 kernels +132 graph PASS |
| Engine | PASS: 6 full-model corpus cases; A/B/A с4×32 токена, bit-exact transport parity |
| Cache / pipeline / sessions | Bounded LRU/decaying frequency и async transport проверены; существенный compute overlap / sessions TODO |
| Плотный cache, MIMO-07 | Default16; payload9,855→12,370 ГиБ, итоговый ABBA6,434→6,952 ток/с (+8,1%), logits bit-exact |
| Доставка по тензорам, MIMO-08 | Default on; ABBA6,591→7,299 ток/с (+10,7%); к последнему прогретому контролю+5,3%, logits bit-exact |
| Частотный кеш, MIMO-09 | Default decay65536; ABBA6,923→7,692 ток/с (+11,1%), к быстрейшему контролю+8,2%; exact logits |
| Fills / CPU guard, MIMO-10 | Literal guard:7,721→7,933 ток/с (+2,74%), включён. Grouped fills:−0,45%, default off, mixed diagnostic открыт |
| Локальная скорость | Прогретый sync5,103–5,160; pipeline reader1:7,053, reader2:6,914 ток/с; условия MIMO-05 ниже |
| Память / TTFT | Pipeline corpus с trace: peak RSS93,42 ГиБ; request-end global VRAM30,00 ГиБ; TTFT6,57с EN /39,64с long249 |
| Рекомендуемый профиль | Engine defaults выбраны по локальным опытам; setup/API profile и P6 benchmark ещё TODO |
| Следующая задача | Cache lookup/CPU overhead и адаптация к смене темы; mixed/F32 диагностика, speculative quality и P2 API |

В PREP-01 созданы план/статус; MIMO-01 добавил `backends/mimo2`, MiMo tools/tests
и `qwen2` branch общего Python tokenizer. Действующие профили и GGUF не менялись.
MIMO-02 добавил CUDA fixtures и MiMo-local generated patches, не меняя `_deps`.
MIMO-03 добавил отдельный синхронный engine и full-model validation.
MIMO-04 добавил GPU expert cache, выбор reader/admission, RAM budget и замеры.
MIMO-05 добавил bounded async transport, route pins, диагностику и defaults reader1.
MIMO-06 проверил все пять MTP/DFlash sidecars отдельным GPU probe. На этом корпусе
без draft быстрее; основной engine и его defaults сохранены.
MIMO-07 добавил плотное размещение expert cache, измерил отдельный ABBA без MTP
и включил default16. [Результаты и ограничения](MIMO26_FLASH_SLAB_CACHE.md).
MIMO-08 объединил доставку экспертов одного scheduler input, сохранив прежние
LRU/admission и синхронные cache fills. [Результаты](MIMO26_FLASH_TENSOR_BATCH.md).
MIMO-09 добавил частотный допуск с забыванием истории; снизил H2D прогретого
decode на11,8% и объём cache fills на82,9%. [Результаты](MIMO26_FLASH_FREQUENCY_CACHE.md).
MIMO-10 проверил grouped fills и оставил их off; убрал временные строки CPU
проверки logits. [Измерения и диагностика](MIMO26_FLASH_FILL_AND_GUARDS.md).
Изменения Hy3/common transport относятся к другой работе и не являются MiMo validation.

## Подтверждено при подготовке

- File size **134 982 426 368 байт**, header_end **5 979 134**,
  data_start **5 979 136**, alignment32, GGUFv3, metadata49, tensors472.
- Имена уникальны, размеры всех типов известны, offsets выровнены;
  диапазоны не пересекаются и не выходят за EOF, последний заканчивается на EOF.
- Payload **134 976 447 232 байта (125,707 ГиБ)**;
  routed **122 909 884 416 (114,469 ГиБ)**,
  остальные **12 066 562 816 (11,238 ГиБ)**.
- Типы: Q2_K50, Q3_K74, MXFP4 17, BF16 101, F32 230.
  Routed tensors141; non-routed331. Source FP8 metadata не означает локальные FP8 weights.
- 48 слоёв, MoE1..47, 256 экспертов/top-8, shared tensors отсутствуют.
  Full layers **0,5,11,17,23,29,35,41,47**, остальные39 — SWA128.
  K192/V128, KV heads4/8, fused QKV, partial RoPE64, SWA attention sinks.
- Header SHA-256 без padding и payloads:
  `46e0ff64f95482e89997ba7e661d36a7061a7a5961e033fe5baa6b241e23269a`.
- Template SHA-256 по UTF-8 строке, длина3867 символов:
  `11ea52e156de38a458e6b7720ad45915d65b97d4ec979a09f55e3c9bd1b4d059`.
- GPT-2 BPE `qwen2`, vocab152576, merges151387, EOS151645, PAD151643,
  add_bos=false. Native oracle EOG: 128247,151643,151645,151662,151663,151664;
  native BOS11 — fallback на обычную запятую, он не вставляется.
- Export scope явно исключает MTP и modality companions. Нет `.nextn.`
  tensors и блоков≥48. На PREP-01 других файлов не было; при MIMO-06 найдены пять sidecars, см. отдельный отчёт ниже.
- Local source revision из metadata: `3b38d063180c3e4aed9691fdc735f3d10b266ee4`;
  converter revision: `58367713a6935c0810103378144008df32e3d5db`.
  Это provenance claims конвертера, не доказательство идентичности payload.
- `mimo2.cpp` из `third_party`, распакованного GLM candidate и архива
  `build-local/llama-glm-86ebfef2.tar.gz` совпадает, SHA-256
  `2ec5caa11fb9ab7604b798a17cd80244427596581c06cdb033762b6d56f50823`.
  Архив SHA-256: `f8e524b635b726bae74fd8f84bb9249e5b09384c63207f6707c5a3f921acad99`.
- Windows RAM **125,555 ГиБ**, RTX5090 total **32 607 МиБ**, driver581.80.
  95% global caps: **119,277 ГиБ / 30 976,65 МиБ**.
  Снимок до тестов: available RAM≈105,99 ГиБ, GPU used2399 МиБ;
  это внешняя нагрузка и не потребление MiMo. Перед запуском измерить заново.

Внешние источники с оговорками и расчёты KV/H2D приведены в плане.
Подтверждение отсутствия MTP относится к **этому экспорту**, не к исходному
checkpoint Xiaomi. Основное текстовое внедрение может продолжаться без sidecar.

## Не проверено

- Полный hash payload и совпадение с опубликованным GGUF.
- Полный converter pipeline и эквивалентность всех исходников dependency двух pins.
  Точные config `3b38d063` и `mimo2.cpp` `58367713` уже получены и сверены.
- Независимый Transformers oracle и широкая оценка качества полного GGUF.
  Transport parity на том же pinned graph/quants подтверждена, включая SWA128.
- Потоковый parser, API arguments normalization, tools round-trip и HTTP/UI.
  IDs/EOG discovery/raw renderer и runtime stop policy уже проверены.
- Full-model context2K/4K, длительная работа при внешнем memory pressure,
  process commit и physical SSD I/O. Замерены working set/global memory,
  TTFT/prefill/decode и native read/H2D bytes, cancel/invalid-request recovery.
- CUDA Graphs, существенный overlap H2D/compute, sessions, другие cache policies
  сверх LRU/decay65536. Prefill admission on/off и file/mmap readers проверены.
- Serving MTP/DFlash, native heads2/3 и multimodal companions. Структура пяти sidecars и offline greedy сравнение вынесены в MIMO-06.

## Таблица этапов

| Этап | Статус | Условие готовности |
|---|---|---|
| PREP-01 | DONE | Исследование и документация сохранены |
| P0 — compatibility/oracles | DONE: MIMO-01 +MIMO-02 | Contract/oracles, 96 kernel и132 graph checks PASS |
| P1 — GPU baseline | DONE для context512/greedy | Full-model transport parity через SWA128, cancel/recovery, отчёт скорости/памяти |
| P2 — tokenizer/template/API | IN PROGRESS: IDs/raw template/stop PASS | Parser/tools, оба API и web chat |
| P3 — cache/pipeline | IN PROGRESS: P3.1 и транспорт P3.2 проверены; P3.3–P3.6 частично | Bytes/logits parity и memory PASS; существенный compute overlap, pressure stress и широкий benchmark TODO |
| P4 — sessions/context | TODO | Restore/shift/cancel parity для full и SWA |
| P5 — MTP / DFlash | EXPERIMENTAL: пять sidecars, отдельный GPU probe | Требуются quality parity, полезный speedup и serving lifecycle |
| P6 — profile/regressions | TODO | Измеренные defaults, отдельный text-only профиль и регрессии |
| P7 — multimodal | OUT_OF_SCOPE текущего GGUF | Companions и отдельная приёмка каждой модальности |

P5/P7 не блокируют P0–P4/P6. `DONE` означает только указанную проверенную часть;
наличие внешней реализации не закрывает native engine или MTP.

## Точка продолжения: MIMO-11

Разбор переноса от7 октября: [GLM / Step / DeepSeek](MIMO26_FLASH_OPTIMIZATION_TRANSFER_REVIEW.md).
Три шага выполнены: [MIMO-07](MIMO26_FLASH_SLAB_CACHE.md) — плотный cache;
[MIMO-08](MIMO26_FLASH_TENSOR_BATCH.md) — tensor delivery;
[MIMO-09](MIMO26_FLASH_FREQUENCY_CACHE.md) — частотный допуск.
Их ABBA имеют разные контроли; проценты нельзя складывать.
[MIMO-10](MIMO26_FLASH_FILL_AND_GUARDS.md) проверил grouped fills без speedup
и добавил literal guard. Далее cache lookup/CPU overhead и численная диагностика.

1. Разобрать расхождение batched target с последовательным greedy из MIMO-06.
   Сначала replay известных target tokens без draft, затем kernel/attention isolation.
   Первая гипотеза — tokenwise small-batch dispatch из GLM; для MiMo дополнительно
   проверить MXFP4, формы4096×2048/2048×4096 и batch8. Причина ещё не установлена.
   Не включать speculative serving до проверки correctness и полезной скорости.
2. Tensor delivery уже уменьшает waits: одна scratch fence и одна delivery
   fence на scheduler input. Defaults сохраняют синхронные cache fills.
   Grouped fills реализованы с ownership pending reservations, но остаются
   off: ABBA не показал ускорения, новый mixed synthetic mismatch не объяснён.
   Scratch может содержать живые активации. Сохранить `--expert-readers 0`
   и контроль `STRATA_MIMO_PIPELINE_BATCH=0`.
3. Route pins уже защищают будущие hits. Любая новая асинхронная запись cache
   потребует защиты fills/consumers до их завершения, повторения bytes/logits,
   cancel/error/drain checks. CUDA Graphs одновременно не включать.
4. Уточнить CPU/driver затраты и kernel timeline: текущий CUDA event trace показывает
   лишь небольшой overlap. Проверить chunks4/16 на полной модели, разные темы,
   physical SSD I/O и длительный external pressure. Reader1/2 при chunk8 измерены.
5. Cache14 запрашивает14 ГиБ allocations. В ABBA MIMO-07 live clamp оставлял
   около12,9 ГиБ; payload с блоками16 составил12,37 ГиБ вместо9,855.
   Размер32 не измерен. Новые варианты сравнивать минимум тремя прогретыми
   повторами и возвратом к контролю; учитывать ненулевой system disk I/O.
   MIMO-09 выбрал decay65536; периоды16384/131072 не измерены. На первых
   запросах тем частота увеличила H2D, на прогретых уменьшила. Проверить поток
   разных однократных запросов и возвращение к прежней теме отдельно.
6. P2: отдельный API adapter/parser/tools и setup profile. До него использовать
   только документированный pipe. Контексты выше512 требуют отдельной проверки.
7. Локализовать повторяющийся F32 synthetic mismatch: в MIMO-08 он возник
   в старом `file_pipe_1_256`, в MIMO-07 — `mmap_pipe_2_256`, с одинаковой
   ошибкой. Повтор PASS не доказывает исправление; численные пороги не ослаблять.
   В MIMO-10 отдельно возник mixed/file_pipe_1_8 max_abs0,0007548183;
   байты совпали, повтор244/244 PASS, оба сохранённых старых контроля212/212.
   Не объявлять эти две сигнатуры одной причиной без локализации.

Не запускать GLM/Step/Hy3 engine с этим файлом как проверку поддержки MiMo.
Не выделять весь объём модели в RAM. Наличие sidecar не означает готовность MTP/DFlash serving.
Тяжёлые тесты начинать после нового global memory admission; чужие процессы не останавливать.

## Подтверждённый журнал

### MIMO-10 / P3 — 2026-10-07 — grouped fills и literal guard

Добавлен opt-in FillBatch: резервирование без GPU submit, невидимые pending
entries, копирование только surviving reservations, fence перед публикацией
и cleanup на ошибке. Ticket защищает от повторного ключа/адреса. CPU tests
сравнивают policy/residency/payload с прежним алгоритмом на16 000 операций.
Включать через `STRATA_MIMO_CACHE_FILL_BATCH=1`; default остаётся0.

С тем же GGUF/RTX5090/128 ГБ RAM, decay65536 и tensor delivery1, ABBA
**7,861→7,826 ток/с (−0,45%)**. Host fences−19,17%, полезного speedup нет.
Corpus6 prompts,3 509 248 logits и ABBA1536 IDs/all logits exact.
Native runtime244/244 сначала отвергнут wrapper из-за старого ожидаемого
count212; wrapper исправлен, добавлен Python regression. Затем243/244:
mixed/file_pipe_1_8/logits, max_abs0,0007548183 при совпавших bytes. Повтор
244/244 и два сохранённых pre-change212/212 прошли. Причина остаётся открытой.

Отдельно добавлена перегрузка `require(bool,const char *)`. Проверка конечных
logits больше не создаёт временную string для каждого элемента; прежние
NaN/Inf и динамические сообщения сохранены. MSVC Release microbenchmark:
152576→0 allocations и3,22→0,057 мс на одну строку из152576 значений.
Это время CPU guard, не общей генерации. Изменение включено в обычную сборку.
Его отдельный binary ABBA с fill0 дал **7,721→7,933 ток/с (+2,74%)**,
к быстрейшему контролю+2,63%. Процессы:7,729 /7,978 /7,889 /7,713.
Время вне decode3,877→0,345 мс/output (−91,11%); суммарный forward почти
не изменился (+0,15%). Все48 запросов/1536 IDs и logits exact; corpus6 prompts
с3 509 248 exact logits и error/cancel recovery PASS. Peak VRAM29,506 ГиБ/92,66%,
RAM107,752 ГиБ/85,82%. Default/unset-env проверка EN/SWA249 — в validation.

27 cache +56 slab +21 frequency +17 fill-batch +11 literal cases,
5000/5000/16000 mixed operations и29 Python tests PASS. CUDA runtime counts
212 для контроля и244 для grouped mode включают32 новых error/recovery checks;
исходные numerical FAIL и wrapper error не удалены.
[Полные замеры и ограничения](MIMO26_FLASH_FILL_AND_GUARDS.md),
[validation](MIMO26_FLASH_OPTIMIZATION10_VALIDATION.json).

Общий transport, frequency helper, другие backend и model weights не менялись.
Новый compute overlap, MTP serving, API, sessions и context выше512 не проверены.

### MIMO-09 / P3 — 2026-10-07 — частотный допуск

Общая ограниченная частотная история подключена к MiMo с полными ключами
generation/tensor/expert. Hits и misses обучают историю; guard tails и prefill
с admission off её не меняют. Кандидат выбирается среди64 старых unpinned
entries, с учётом размера allocation. Редкий newcomer сохраняет более частую
запись. Route pins, синхронная публикация fills, GPU-only audit и global95% сохранены.
`STRATA_MIMO_CACHE_DECAY=65536` включён по умолчанию; `0` — прежний LRU.

Windows, RTX5090/128 ГБ RAM, тот же GGUF, slab16 и tensor delivery в обоих
режимах. Same-binary ABBA **6,923→7,692 ток/с (+11,1%)**; относительно
быстрейшего контроля+8,2%. Процессы:7,112 /7,826 /7,563 /6,744.
Decode H2D прогретых запросов−11,8%, cache-fill D2D−82,9%, host fences−58,9%.
Reserved cache практически одинаков:12,708/12,711 ГиБ. Пик нового режима:
VRAM29,506 ГиБ/92,66%, RAM111,540 ГиБ/88,84%.
Первые запросы тем увеличили H2D с251,799 до347,100 ГиБ: сохраняется цена
адаптации частотной истории. Они сохранены отдельно, не смешаны с warm throughput.
[Условия и ограничения](MIMO26_FLASH_FREQUENCY_CACHE.md),
[замеры](MIMO26_FLASH_FREQUENCY_BENCHMARK.json).

27 cache,56 slab и21 frequency cases, две серии по5000 смешанных операций,
28 Python tests PASS. Runtime212/212 при decay0 и65536; corpus6 prompts,
3 509 248 exact logits, cancel/error recovery и unload PASS. ABBA48 запросов,
1536 IDs и все logits exact. После default-only пересборки запуск без env/CLI
overrides EN/SWA249 и recovery PASS;1 220 608 logits exact.
Сборки, исходники, команды и hashes — в [validation](MIMO26_FLASH_FREQUENCY_VALIDATION.json).

Общий frequency header и другие backend не менялись. Прежняя F32 synthetic
ошибка в этой серии не возникла; её причина остаётся открытой. Пороги не ослаблены.
Периоды16384/131072, длинный context и внешний pressure stress не измерены;
MTP, API, sessions и существенный compute overlap этим этапом не закрыты.

### MIMO-08 / P3.3 — 2026-10-07 — tensor delivery

Selected matrices и guard tails одного scheduler input доставляются до общей
copy fence. Route pins сохраняют sources, scratch fence остаётся перед tensor.
Затем воспроизводится исходный LRU/fill order; fills публикуются синхронно.
`STRATA_MIMO_PIPELINE_BATCH=1` включён по умолчанию; `0` оставлен для контроля.
Общий transport, арифметика, memory95% и GPU-only compute не менялись.

На том же GGUF/RTX5090/128 ГБ RAM, slab16 в обоих режимах, same-binary ABBA:
**6,591→7,299 ток/с (+10,7%)**. Последний прогретый контроль дал6,933,
относительно него прибавка5,3%. Четыре процесса:6,282 /7,388 /7,212 /6,933.
Ранний русский контроль медленнее позднего; все его запросы включены в итог.
Host waits доставки/fills−74,92%, дополнительные scratch fences−90,41%
(prefill+decode); decode H2D практически одинаков. Пик нового режима:
VRAM29,506 ГиБ/92,66%, RAM113,454 ГиБ/90,36%.
[Условия и ограничения](MIMO26_FLASH_TENSOR_BATCH.md),
[полные замеры](MIMO26_FLASH_TENSOR_BATCH_BENCHMARK.json).

27 cache +56 slab cases,5000 смешанных операций и28 Python tests PASS.
Новый runtime212/212, corpus6 prompts с3 509 248 exact logits, error/cancel
recovery и unload PASS. ABBA48 запросов/1536 токенов и все logits bit-exact.
Измеренная сборка сохранена отдельно от default-only shipping build;
проверка unset env/CLI на EN/SWA249 и recovery записана в
[validation](MIMO26_FLASH_TENSOR_BATCH_VALIDATION.json).

Первый выключенный контроль дал211/212 из-за прежнего F32 mismatch
`file_pipe_1_256`; доставленные bytes совпали. Повтор212/212 без изменения
арифметики/порогов. Этот дефект не решён. MTP serving, API, sessions и
существенный compute overlap этим этапом не закрыты.

### MIMO-07 / P3 — 2026-10-07 — плотное размещение cache

Добавлен allocator общих GPU-блоков, выравнивание slots256 байт и physical
accounting целых блоков. Индекс неполных блоков сокращает поиск свободного места.
Pins, fenced fill, byte audit, global95% и GPU-only compute сохранены.
Defaults pipe engine: `STRATA_MIMO_CACHE_SLAB_MIB=16`; `0` возвращает контроль.
Групповые копии, частотная история и MTP этим изменением не реализованы.

Итоговый same-binary ABBA0/16/16/0, три темы, один прогрев и три запроса×32
на тему: **6,434→6,952 ток/с (+8,1%)**. Payload9,855→12,370 ГиБ,
decode H2D−16,7%, peak global VRAM29,513 ГиБ (92,68%), RAM109,894 ГиБ
(87,53%). Ненулевые system disk reads сохранены; модельный I/O не изолирован.
Нельзя сравнивать эти скорости напрямую с другим тестом MIMO-05/MIMO-06.
[Условия и результаты](MIMO26_FLASH_SLAB_CACHE.md),
[ABBA](MIMO26_FLASH_SLAB_BENCHMARK.json).

23 cache +56 slab cases и5000 смешанных операций,28 Python tests PASS.
Итоговый runtime212/212, полный corpus6 prompts с3 509 248 точными logits,
cancel/error recovery и unload PASS. Все1536 токенов и logits ABBA совпали.
После пересборки с default16 без env/CLI overrides полный corpus повторён:
logits/IDs и recovery PASS. Измеренная и итоговая сборки сохранены раздельно.
[Hashes и проверки](MIMO26_FLASH_SLAB_BUILD_VALIDATION.json).

Первый slabOFF runtime повторил прежнее F32 `mmap_pipe_2_256` расхождение;
его причина не решена, численные пороги не ослаблены. Первый packed runtime
требовал адаптации8 small-cache occupancy assertions; численные проверки прошли.
Uncached native corpus остановился по RAM guard95%; для итогового сравнения
использован managed cache0. Эти неудачи не включены в скорость и сохранены в
[журнале опытов](MIMO26_FLASH_SLAB_EXPERIMENTS.json). P3 целиком остаётся открытым.

### MIMO-06 / P5 — 2026-10-06 — сравнение пяти MTP/DFlash sidecars

Добавлен opt-in GPU probe с загрузкой MTP head0 и DFlash, target feature capture,
confidence cutoff, пакетной проверкой и rollback. Все пять файлов проверены
статически и запущены на локальной модели; сохранены полные hashes sidecars.
Основной executable не менялся, `STRATA_MIMO_SPEC_PROBE` возвращён в OFF.
[Сравнение и условия](MIMO26_FLASH_SPECULATIVE_COMPARISON.md),
[сборка и проверки](MIMO26_FLASH_SPECULATIVE_BUILD_VALIDATION.json).

Три prompt ×32 токена, по три прогретых повтора после первого; context512,
batch8, F32 KV, полный физический KV для rollback, FA on, reader1/chunk8 МиБ,
cache до14 ГиБ с live clamp. Итоговая скорость: без draft **6,13 ток/с**;
MTP Q4/Q8/BF16 — **5,55/5,48/4,88**; DFlash Q8/BF16 — **5,26/4,66**.
Draft использует cutoff0.7, MTP depth1, DFlash depth7. Все варианты совпали
с baseline по выходным IDs этого корпуса. Q4 — самый экономный кандидат для
дальнейших опытов, но default остаётся без draft. Исторические7,05 ток/с MIMO-05
получены другим тестом и не являются baseline этого сравнения.
[Основные замеры](MIMO26_FLASH_SPECULATIVE_BENCHMARK.json),
[перебор, shared head и boundary](MIMO26_FLASH_SPECULATIVE_EXPERIMENTS.json).

Широкая приёмка качества не пройдена: replay известных baseline-токенов без
draft-весов расходится с последовательным target на18-м токене русского ответа
при batch2 и batch8, до первого rollback. Cutoff0.7 совпадает на тестовом корпусе,
но не исправляет этот источник расхождения. Native heads2/3, stochastic correction
и serving lifecycle не реализованы. Следующий этап — локализация batched target.
[Диагностика](MIMO26_FLASH_SPECULATIVE_DIAGNOSTIC.json).

28 Python tests и10 проверок verified prefix PASS. В завершённых итоговых прогонах
GPU audit не обнаружил CPU model nodes, очереди drained, global RAM/VRAM ниже95%.
Ранний опыт с shared head остановлен guard при позднем CUDA allocation; после
переноса прогрева verify workspace перед cache fill повторные прогоны PASS.
Остановленный опыт не включён в скорость. Общий transport, Hy3/Step и main.cpp
этим этапом не изменялись.

### MIMO-05 / P3.2 — 2026-10-06 — bounded pipeline и выбранные defaults

Стенд: тот же GGUF, Windows/RTX5090 32 ГБ/128 ГБ RAM, CUDA13.0.48/sm120,
Unsloth `86ebfef2`; GPU graph/quants/precision прежние. Common transport и
GLM working-set helper использованы без изменения. Hy3/Step работа сохранена.
[Сборка и hashes](MIMO26_FLASH_PIPELINE_BUILD_VALIDATION.json).

После настоящих router IDs планируются selected gate/up/down matrices этого
слоя. Один/два reader читают file/mmap в4 pinned slots; отдельный H2D stream
загружает device ring. События ready/used защищают слот. Producer не пишет
scheduler scratch; ring→scratch D2D идёт после завершения предыдущего consumer.
Будущие resident hits защищены route pins, cache fill публикуется после fence.
Хвосты512 байт входят в план копирования и проверяются byte observer.
Cancel/error/unload дожидаются workers/CUDA и снимают pins; после injected
worker error следующий план пересоздаёт транспорт и восстанавливает работу.

При chunk8 дополнительно используются32 МиБ pinned RAM и32 МиБ device ring.
Budget проверяется перед каждым route/decode, с global95% guard и прежним
working-set cap. Каждая матрица уже не вызывает отдельный NVML query.
Все матричные операции остаются на GPU, non-GPU compute audit активен.

В synthetic tests обнаружено нестабильное начальное состояние: иногда первый
decode нового context расходился с reference, включая sync cache, хотя bytes
совпадали. Reference и pipe явно очищали KV, а новые fixture contexts — нет.
Теперь context initialization тоже synchronizes/clears KV. **Шесть последовательных
запусков по212/212 PASS**, допуски не менялись. Отключение PDL/fusion не устранило
ошибку, их policy оставлена прежней. Точный нижележащий механизм зависимости
от неочищенного storage не локализован полностью; исходные FAIL сохранены в
[diagnostic](MIMO26_FLASH_PIPELINE_DIAGNOSTIC.json), стабильные повторения —
в [runtime validation](MIMO26_FLASH_PIPELINE_RUNTIME_VALIDATION.json).

Проверено:

- 23 cache ownership/budget cases, включая pins/eviction/bypass/nested leases.
- 212 runtime cases: F32/mixed, file/mmap, readers1/2, chunks4/16, caches8/256 МиБ,
 273 positions через SWA128, bit-exact bytes/logits, cancel/error/drain/recovery.
- 96 kernels и132 graph checks,40 Python tests,3 CPU CTest.
- [Полная модель](MIMO26_FLASH_PIPELINE_MODEL_VALIDATION.json):6 prompts,
 23 output tokens и все logits bit-exact native; long prompt249, invalid/cancel
 recovery, clean unload. Это transport parity одного pinned graph, не независимый
 Transformers oracle и не широкая оценка качества.
- [Implicit defaults](MIMO26_FLASH_PIPELINE_DEFAULTS_VALIDATION.json): reader1,
 а также автоматическое отключение pipeline для native/cache0, bit-exact fixture.

Измерение скорости: context512/batch8, F32 KV, FA on, greedy, TF32/graphs off;
одинаковый prompt24,4 fresh-KV запроса по32 output tokens (31 decode forward),
logits export on, trace off. Первый запрос исключён из прогретой статистики.
Порядок sync→reader1→reader2→sync; native reference запускался перед первым sync.
OS file cache не сбрасывался. [Полные замеры](MIMO26_FLASH_PIPELINE_BENCHMARK.json).

| Вариант, cache14 с clamp | Decode median; диапазон, ток/с | Aggregate3 warm, ток/с | Warm TTFT, с |
|---|---:|---:|---:|
| Sync до pipeline | 5,108;5,053–5,148 | 5,103 | 3,504–3,738 |
| Pipeline reader1, chunk8 | 7,047;7,023–7,089 | **7,053** | **2,554–2,654** |
| Pipeline reader2, chunk8 | 6,913;6,909–6,921 | 6,914 | 2,602–2,620 |
| Sync после pipeline | 5,173;5,124–5,183 | 5,160 | 3,469–3,548 |

Reader1 дал **+36,7–38,2%** к sync в этой серии. Его средняя CPU time/elapsed
за процесс с load/lifecycle —1,48 ядра против1,86 у reader2 и≈1,00 у sync.
CPU обслуживает memcpy, paging, scheduler и CUDA submission; эти time samples
не определяют долю каждой причины. CPU matrix nodes не исполнялись.

Defaults: **reader1/chunk8/mmap, requested cache14 ГиБ, prefill admission off**.
Chunk8 — лучший измеренный вариант по числу readers; chunks4/16 ещё не ранжированы
по скорости на полной модели. `--expert-readers 0` возвращает sync режим.
В corpus с trace sampled RSS93,42 ГиБ, global RAM peak≈94,00%, request-end
global VRAM30,00 ГиБ (≈94,21% total). Это включает другие процессы; sampling
может пропускать пики. Разные темы/первые запросы медленнее warm repeated prompt.

В final trace первые два prefill graphs: overlap0,177/0 мс; два decode graphs:
**0/0,050 мс**, при H2D83,36/51,29 мс. Это интервалы CUDA events на streams,
с возможными CPU submission gaps, не профиль отдельных kernels. Значительного
перекрытия H2D/SM пока не показано. Выигрыш согласуется с prefetch и уменьшением
синхронной host/driver работы, но точная доля причин не измерена. P3 остаётся
IN PROGRESS; следующий опыт — сокращение D2D/fences с сохранением lifetimes.

### MIMO-04 / P3.1 — 2026-10-06 — GPU expert cache и ограничение памяти

Основание: Strata `819ad130`, dependency `86ebfef2`, тот же локальный GGUF,
RTX5090/Windows/128 ГБ RAM, CUDA13.0.48/sm120. Common file/device memory и
GLM host working set helper использованы без правок; Hy3/Step изменения сохранены.
[Сборка, команды и hashes](MIMO26_FLASH_CACHE_BUILD_VALIDATION.json).

Реализован отдельный MiMo LRU по matrix identity: generation, зарегистрированный
tensor (name/type/shape/stride/file range), expert ID. Quantized bytes не меняются.
Кэш обслуживает Q2_K/Q3_K/MXFP4, проверяет размер при hit, повторно использует
совместимые allocations и учитывает payload отдельно от выделенной памяти.
Успешный fenced fill предшествует публикации entry; ошибки/отмена не оставляют
частичных entries. H2D/D2D заканчиваются до reuse/eviction. Unload/new generation
сбрасывают mappings/cache. Это **синхронный cache, без асинхронного overlap**.

Обнаружены и исправлены две проблемы бюджета:

- На Windows/WDDM cudaMalloc размером2,75/3,60 МБ занимал примерно4 МиБ.
  Учёт64 КиБ недооценивал расход. Теперь runtime явно округляет allocation до2 МиБ,
  проверяет live NVML и сохраняет5% total +256 МиБ. До первого полного batch
  остаются ещё3 ГиБ для ленивых FP32 cuBLAS pools. Cache cap пересчитывается.
- Mmap без working set cap прошёл все6 corpus outputs, но95% RAM guard прервал
  следующий lifecycle request. Теперь собственный working set ограничен под94%
  global RAM с учётом других процессов. Mapping остаётся; чистые страницы можно
  вытеснять. Прежние process limits восстанавливаются при release. Corpus с12
  и14 ГиБ прошёл, включая cancel→fresh. [Диагностика](MIMO26_FLASH_CACHE_ALLOCATION_DIAGNOSTIC.json).

**Defaults engine:** `--expert-cache-mib 14336 --expert-reader mmap
--expert-cache-prefill off`, context512/batch8, F32 KV, FA on, greedy.
Prefill использует hits без admission/изменения LRU; grouped all-miss ranges
копируются вместе. Cache сохраняется между запросами, KV каждый раз очищается.
`--copy-mode native` автоматически выбирает cache0; file reader и явный cache0
остаются доступны. API/launcher profile ещё не создан.
[Проверка запуска без cache/reader/context flags](MIMO26_FLASH_CACHE_DEFAULTS_VALIDATION.json).

Проверки:

- **17 cache ownership/budget cases:** LRU/reuse, generation, OOM/fill failure,
  zero cap, pressure trim, reset counters, payload/allocation accounting и cleanup.
- **[84 runtime cases](MIMO26_FLASH_CACHE_RUNTIME_VALIDATION.json):** F32/mixed,
  file/mmap, small cache8 МиБ с eviction и256 МиБ с hits,273 позиции через128/256;
  byte observer на hits/misses, exact logits, admission off, cancel→fresh,
  reload и запрет CPU compute до выполнения графа.
- **[6 full-model cases](MIMO26_FLASH_CACHE_MODEL_VALIDATION.json):** EN/RU/ZH,
  code/numbers,249-token prompt;23 выходных токена и все их logits bit-exact
  с native reference. Cancel/error→fresh и unload PASS. Это transport corpus,
  не широкая оценка качества или проверка context2K/4K.
- **[96 kernels +132 graph](MIMO26_FLASH_CACHE_CUDA_REGRESSION.json)**,
  **40 Python tests +3 CPU oracle CTest PASS**. Numerical tests содержат CPU
  reference, но production engine не переносит вычисления матриц на CPU.

**[Замеры скорости](MIMO26_FLASH_CACHE_BENCHMARK.json):** тот же prompt24,
3 свежих KV-запроса по32 выходных токена,31 decode forward на запрос;
precision/graphs policy прежняя. Native → cache12 → cache14 выполнялись
последовательно, OS cache не сбрасывался, logits export включён. Первый запрос
прогревает данные, следующие два — warm. P6 с минимум тремя warm и чередованием
A/B ещё не закрыт; абсолютная скорость зависит от paging и внешней нагрузки.

| Режим | Decode ток/с,1/2/3 | По сумме93 forwards | H2D ГиБ/decode forward |
|---|---|---:|---:|
| Native mmap, cache0 | 2,655 /3,409 /3,437 | 3,122 | 3,578 |
| Cache12 ГиБ, mmap, prefill off | 3,328 /5,192 /5,500 | 4,445 | 1,229 |
| Cache14 ГиБ с clamp, mmap, prefill off | 3,316 /5,508 /5,737 | 4,563 | 1,191 |

В этой серии aggregate cache14 выше native примерно на46%; во втором/третьем
повторах — на62%/67%. Это результат данного prompt, не обещание такого выигрыша
на всех запросах. Все96 IDs и14 647 296 logits на каждый режим bit-exact.
В предыдущих опытах cache8/file дал2,316 ток/с, cache12/file3,436,
cache12/mmap/admission-on4,420. Они выполнены при другом прогреве/бинарнике;
таблица опытов сохранена, но эти числа не являются строгим ranking.

Cache14 в повторах реально занимал12,68–12,87 ГиБ allocations после clamp.
На полном corpus: peak process working set **95,63 ГиБ**,
минимум available global RAM **11,98 ГиБ**, sampled global VRAM **30 214 МиБ**,
максимум request-end NVML **30 720 МиБ** (около94,2% total32 607 МиБ).
Выборки около1с пропускают короткие пики; значения NVML после requests приведены
отдельно. Короткие повторные prompts дают working set≈54 ГиБ: demand mmap
загружает только затронутые pages; заполнение всей RAM пока не форсируется.
Полный корпус использовал около90% общей RAM; предыдущий corpus12 приближался
к94,4%. Длительный pressure stress и явный RAM frequency cache ещё TODO.

**CPU:** GPU audit проверяет все вычислительные nodes до каждого graph execute;
в полном corpus/повторах CPU compute rejections0 и full-copy rejections0.
CPU обслуживает tokenizer/greedy, ranges/cache, I/O, driver calls и ожидания CUDA.
В чистом repeat замере cumulative CPU time соответствует примерно одному ядру
в среднем; это не профиль причин нагрузки. Причины нужно разделить trace/profile.

`requested_bytes = h2d_bytes + cache_hit_bytes` проверено для успешных запросов.
В mmap `h2d_ms` включает faults/staging и `source_bytes=0`; это не нулевой SSD I/O.
File reader считает native read bytes, также не physical disk reads.
Не закрыты P3.2 pipeline, P2 HTTP/tools, P4 sessions/context и P6 profile.

### MIMO-03 / P1 — 2026-10-06 — Синхронный GPU engine полного GGUF

Основание: workspace с существующими изменениями, dependency `86ebfef2`,
тот же header/template hash. Архив и `_deps` не менялись. Изменены только
MiMo backend/tools/docs; shared transport используется без правок.
Build/source/binary hashes и команды —
[runtime build record](MIMO26_FLASH_RUNTIME_BUILD_VALIDATION.json).

Добавлены `contract.hpp`, `runtime.hpp`, `sync_runtime.h/.inc`, GPU audit,
generated `RuntimePatches.cmake`, `main.cpp`, `check_runtime.cpp`;
`tools/check_mimo2_engine.py`, его tests и runtime-режим CUDA runner.
Production contract отвергает fixtures до allocation и проверяет472 tensors,
EOF/ranges/types и metadata. У реального GGUF attention arrays **INT32**,
у synthetic fixtures UINT32: первое полное admission выявило это отличие,
после явной проверки типов модель загрузилась. Также добавлены explicit header
dependencies для локализованного MSVC/Ninja; изменение contract действительно
пересобирает бинарник. Неудачные ранние прогоны сохранены в build-local.

Размещение: non-routed BF16/F32 на GPU, routed quantized weights в demand mmap,
без полного prefetch. `native` сохраняет selected-copy исходного scheduler;
`pinned` читает выбранные file ranges в16-МиБ cudaHostAlloc, затем H2D/fence.
CPU выполняет tokenizer, greedy sampling, I/O и scheduling. Audit запрещает
любой CPU compute node; полный expert-copy вне selected-copy пути отвергается.
Memory admission учитывает global NVML VRAM и доступную RAM, потолок95%.

Pipe: `ENC`, `GEN`, `STOP`, `QUIT`, стартовые INFO/READY. Только greedy,
fresh full/SWA KV на каждый запрос, EOS151645, add_bos=false.
Текстовые `</s>`, FIM/PAD/BOS не являются stop. Нет MTP, cache, pipeline,
prefix/session reuse, HTTP или автоматического профиля.

Проверки:

- [Runtime fixtures](MIMO26_FLASH_RUNTIME_VALIDATION.json): **28/28 PASS**.
  F32/mixed, resident GPU/native mmap/pinned file,273 позиции через128/256;
  побайтовая проверка каждой копии, logits, отмена внутри copy и fresh recovery,
  unload/reload, truncated ranges, CPU compute rejection, RAM/VRAM admission,
  EOS policy. Pinned/native logits bit-exact.
- [Pipe fixtures](MIMO26_FLASH_PIPE_VALIDATION.json): оба режима,
  invalid IDs→fresh, cancel→fresh и clean unload.
- [Полный корпус](MIMO26_FLASH_ENGINE_VALIDATION.json): **6/6 PASS**, EN/RU/ZH,
  код, числа и249-token prompt; всего23 выходных токена на режим, все logits
  bit-exact, greedy IDs одинаковы, CPU math/full-copy rejections0.
  Ответы включают `Hello world.`, `Париж`, `巴黎。`, `391`; long/code ответы
  ограничены4 токенами — это transport test, не оценка полноты ответа.
  Cancel в prefill, invalid request→fresh, clean unload прошли оба режима.
- [Повторы полного GGUF](MIMO26_FLASH_ENGINE_BENCHMARK.json): **3×32 токена
  на режим**, prompt24, context512/batch8, KV F32, FA on, TF32/graphs/reuse off.
  Все96 token IDs и14 647 296 logit values на режим совпали бит-в-бит.
  Каждый запрос fresh KV. Native запускался первым, OS cache не сбрасывался;
  вывод logits в файл включён. Это диагностический baseline, не доказательство
  speedup и не выбор оптимальных настроек production.
- После runtime integration повторены **96/96 kernels и132/132 graph**;
  **40 Python tests и3 CPU CTest PASS**. Commands/hashes/exit codes — build record.

RTX5090 32 ГБ,128 ГБ RAM, Windows, CUDA13.0.48/sm120. Повторы decode считают
31 forward на32 выданных токена; первый токен получен после prefill.

| Режим | Decode ток/с, повторы1/2/3 | Prefill/TTFT, с, повторы1/2/3 | Request, с, повторы1/2/3 |
|---|---|---|---|
| Native mmap | 1,720 /2,651 /2,857 | 16,881/16,887; 6,362/6,370; 4,439/4,448 | 35,159 /18,309 /15,551 |
| Pinned file16 МиБ | 1,237 /1,111 /1,220 | 10,018/10,026; 12,068/12,078; 12,049/12,058 | 35,365 /40,315 /37,753 |

По сумме31×3 decode forwards: **2,292 ток/с native,1,187 ток/с pinned**.
Файловая доставка пока медленнее прогретого mmap и не объявляется ускорением.
Она сохраняет ограниченный staging и значительно меньший process working set.
Default pinned выбран для синхронного bounded baseline; режим native остаётся
доступен для сравнения. Дальнейшая скорость требует cache/pipeline.

Память на полном корпусе (выборки около1с, включают внешние процессы):

| Метрика | Native | Pinned |
|---|---:|---:|
| Peak process working set | 95,067 ГиБ | 11,798 ГиБ |
| Минимум доступной global RAM | 10,420 ГиБ | 86,505 ГиБ |
| Максимум global VRAM | 16 771 МиБ | 16 777 МиБ |

Повторы дали global VRAM maxima16 783/16 795 МиБ и working set53,966/11,798 ГиБ.
Начальная загрузка в корпусе: native7,696с, pinned3,571с; порядок прогрева разный.
Фактические GPU allocations из native log: weights11 507,57 МиБ, full KV22,50
МиБ (512 cells/9 layers), SWA KV97,50 МиБ (256 cells/39 layers), compute buffer
1 761,29 МиБ. Дополнительные CUDA/cuBLAS pools и desktop входят в global samples.
На decode доставляется примерно3,578 ГиБ/token, включая scheduler padding;
source/H2D timers измерены только в pinned. Read bytes не равны physical SSD reads.

Не закрыто: cache/pipeline, HTTP/tools/UI, contexts2K/4K, process commit/physical
disk measurements, prolonged pressure и speed defaults. Полный corpus выполнен
до исправления счётчиков invalid request и выделения неизменного EOS predicate;
финальный бинарник проверен synthetic pipe и3×32 full-model повторами.
Следующий шаг — **MIMO-04 / P3.1**, с этими режимами как transport/logits reference.

### MIMO-02 / P0.4–P0.5 — 2026-10-06 — CUDA операции и native full/SWA graph

**Основание:** Strata `819ad130f49d1e088e2fbc50e1e4f5c681e1e390`, существующие
изменения других задач сохранены. RTX5090, driver581.80, Windows x64,
MSVC19.44.35222.0, CUDA13.0.48, Ninja Release, sm120. Source/archive pin прежний;
исходный `mimo2.cpp` и `_deps` не менялись. Generated patches и source hashes —
в [отчёте сборки](MIMO26_FLASH_CUDA_BUILD_VALIDATION.json).

**Изменено:** CUDA option/manifest в CMake, `check_kernels.cpp`,
`synthetic_mimo2.hpp`, `check_graph.cpp`, три CMake-поправки,
`tools/check_mimo2_cuda.py`, `tools/test_mimo2_cuda.py`, README/план/статус.
Runner сохраняет логи, проверяет provenance и число cases, контролирует 95%
общей RAM/VRAM и ограничивает время теста600с. Чужие процессы не останавливает.

**Результаты, exit0 / PASS:**

| Проверка | Результат | Артефакт |
|---|---|---|
| CUDA kernels/router | 96/96, плотные и routed операции, padding/guards, IDs exact | [kernels](MIMO26_FLASH_CUDA_KERNELS_VALIDATION.json) |
| Native graph | 132/132, F32/mixed, CPU/F32 oracle, FA off/on, full/SWA128 | [graph](MIMO26_FLASH_CUDA_GRAPH_VALIDATION.json) |
| Сборка / CPU regression | CUDA build exit0; CPU oracles rebuild +3/3 CTest PASS | [build](MIMO26_FLASH_CUDA_BUILD_VALIDATION.json) |
| Python | 37/37 tests PASS, включая 2 новых для runner admission | [build](MIMO26_FLASH_CUDA_BUILD_VALIDATION.json) |
| Диагностика исходных ошибок | Сохранены непрошедшие cases и применённые исправления | [diagnostic](MIMO26_FLASH_CUDA_DIAGNOSTIC.json) |

Kernel fixture: K512/M64, experts256/top-8, batch1/4/17. Graph fixture:
3 блока full/SWA/full, hidden256, experts16/top-8, vocab64; настоящие
heads64, KV4/8/4, K192/V128, RoPE64, sinks и scale0,707.
CPU mixed reference использует F32-деквантизацию тех же packed weights.

Для384 позиций проверены prefill/serial/split и границы127/128/129,
255/256/257/383. SWA cache256 cells против1024 full cells; после384 токенов
SWA хранит позиции128..383, маска допускает только последние128.
Отдельно scalar-эталон проверяет fused QKV offsets, partial NeoX, маску,
`1/sqrt(192)`, sinks, value scale и отсутствие лишнего router multiplier.
GPU audit: CPU compute nodes отсутствуют, FA действительно выполняет `FLASH_ATTN_EXT`.

Максимальные расхождения logits с CPU/F32: F32 **3,76e-7**,
mixed **0,000967** (NMSE **9,38e-7**). FA on/off: F32 **0,000230**,
mixed **0,000411**. Tail rollback совпал бит-в-бит; native state restore после
другого разговора — в допуске, не бит-в-бит. Это не закрывает P4 для serving.

**Исправления по результатам тестов:**

1. Unpatched CUDA: 20/96 failures — explicit TF32 нарушал requested F32 precision;
   Blackwell MXFP4 MMQ дополнительно квантовал активации в FP4
   (максимальный NMSE0,01025 при лимите0,0001).
2. После только TF32 fix: 10/96 failures. Стал виден неправильный physical stride
   в gather токенов к экспертам; исправлен отдельно. MXFP4 MMQ для Blackwell
   переключён на GPU dequant/cuBLAS, MMVQ для малого batch сохранён.
3. Kernels прошли96/96, graph129/132. BF16 custom MMF округлял активации
   иначе, чем decode. Добавлен учёт `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32`;
   после этого 132/132. Допуски после неудачных запусков не ослаблялись.

Флаги дочернего процесса: `NVIDIA_TF32_OVERRIDE=0`,
`GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32`, `GGML_CUDA_DISABLE_GRAPHS=1`,
`LLAMA_GRAPH_REUSE_DISABLE=1`. Команды сборки/запуска — в
[README](../../backends/mimo2/README.md), точные invocation/manifest/binary hashes —
в JSON. CUDA DLLs этого toolchain находятся в `build-local/cuda-13.0/bin/x64`.

Первый запуск был отложен из-за внешней нагрузки VRAM. Финальные sampled peaks:
kernels14749МиБ, graph3498МиБ **общего** GPU usage (включая чужие задачи),
ниже95% от32607МиБ. Минимум свободной RAM в graph run103,42ГиБ;
peak process working set по выборкам≈671,2МиБ. Длительности4,91с/5,05с
относятся только к synthetic tests, не к скорости генерации модели.
Проверено также отклонение существующего рабочего каталога native graph checker
без перезаписи его отчёта; финальная сборка после этой защиты повторно прошла132/132.

**Не закрыто:** full48-layer GGUF, engine/API/cache/pipeline, runtime stop policy,
полные сессии, CUDA Graphs/reuse. Стоимость precision policies на полной модели
не измерена; это correctness baseline. Полный GGUF и его payload не загружались,
реальных TTFT/токенов/с по MiMo ещё нет.

**Следующий шаг:** MIMO-03, P1 synchronous selected-copy GPU engine и full-model baseline.

### MIMO-01 / P0.1–P0.3, часть P2 — 2026-10-06 — CPU admission и oracles

**Основание:** подготовка на Strata `1979cb66`; к фиксации отчёта HEAD уже
`819ad130`, чужие изменения сохранены. MiMo source hashes и фактическая
ревизия записаны в [отчёте сборки](MIMO26_FLASH_BUILD_VALIDATION.json).
Изолированный Unsloth `86ebfef2`, archive/loader SHA закреплены, patches=none.
Windows x64, MSVC19.44.35222.0, Ninja Release, CPU; CUDA отключена.

**Реализовано:**

- `tools/inspect_mimo2_gguf.py` и `mimo2_loader_contract.py`: names/shapes/quants,
  metadata, byte ranges, слойная геометрия, строгая production admission.
  Tiny fixture не принимается CLI; неизвестные веса/metadata и MTP отклоняются.
- `backends/mimo2`: три отдельные C++ программы — vocabulary-only tokenizer,
  native Jinja и actual loader registration с `no_alloc/load_mode=NONE`.
- `tools/strata_tokenizer.py`: отдельный `qwen2` regex без изменения других pre.
  `mimo2_tokenizer.py`: scoped normalization; `mimo2_template.py`: text-parts guard
  и Python renderer; `check_mimo2_oracles.py`: воспроизводимая сверка с native.
- `check_mimo2_reference.py`: проверка точных source config/loader и 46 сопоставлений.
  Получены только два маленьких публичных файла, исходники не выполнялись.

**Проверки, exit0 / PASS:**

| Проверка | Результат | Артефакт |
|---|---|---|
| Header/ranges/strict contract | 472 тензора, исходный header hash сохранён | [inspection](MIMO26_FLASH_INSPECTION.json) |
| Native registration | Все names/shapes/types/bytes и 48 геометрий совпали; 0 allocated weight bytes при обоих load_mtp flags | [loader](MIMO26_FLASH_LOADER_VALIDATION.json) |
| Tokenizer | 614/614 exact IDs и обратные UTF-8 bytes, 0 mismatches | [tokenizer](MIMO26_FLASH_TOKENIZER_VALIDATION.json) |
| Raw template | 84 combinations +1 ручной no-think prefix, 85/85 PASS | [template](MIMO26_FLASH_TEMPLATE_VALIDATION.json) |
| Pinned reference | 46/46 проверок; `mimo2.cpp` совпал байт-в-байт | [reference](MIMO26_FLASH_REFERENCE_VALIDATION.json) |
| Python fixtures/regressions | 17 MiMo +18 Step/GLM/Hy3, всего 35 PASS | [build/test report](MIMO26_FLASH_BUILD_VALIDATION.json) |
| CTest | 3/3 version entry points PASS | [build/test report](MIMO26_FLASH_BUILD_VALIDATION.json) |

Точные команды inspector/checkers/build находятся в
[README backend](../../backends/mimo2/README.md); команды и hashes binaries/source
также сохранены в отчёте сборки. Первичная конфигурация без Windows SDK в PATH
не нашла `rc`/`mt`; повторный `--fresh` configure/build в x64 developer shell прошёл.

**Найдено и исправлено:** первоначально 147 из 614 tokenizer cases расходились
на `</s>` при parse_special=true. Native llama-vocab повышает его NORMAL→CONTROL
по написанию. MiMo factory повторяет это только для проверенного словаря,
не меняя GGUF и другие `qwen2` модели. После исправления 0 расхождений.
Native BOS11 и дополнительные EOG/FIM IDs зафиксированы; BOS не вставляется.
Флаг `runtime_stop_policy_validated=false` оставлен явно.

**Не закрыто:** числовые CUDA fixtures, engine/API/cache/pipeline, runtime stop
policy, полный converter/source-tree audit и совпадение payload. Source config
содержит NextN3 и multimodal companions, этот экспорт — NextN0 без companions.
Регистрация с load_mtp=true не создаёт отсутствующие веса.
Inference, TTFT, токены/с и peak memory **не измерялись**.

**Следующий шаг:** MIMO-02, CUDA kernels и full/SWA128 fixtures P0.4/P0.5.

### PREP-01 — 2026-10-06 — Заголовок, архитектура и план

**Действия:** прочитаны инструкции, планы/статусы GLM, Step и Hy3,
GGUF reader, общий transport и локальный `mimo2.cpp`; изучены CUDA dispatch
для MXFP4 и K192/V128, опубликованные GSQ-RCO model card и RL config.
Проверены header, ranges, hashes dependency и размеры системной памяти.

**Результат:** инвентаризация готова для переноса в inspector; MTP и modality
weights отсутствуют; pipeline/state/format проверки расписаны в плане.
Tensor payload не читался и не хэшировался; inference и GPU kernels не запускались,
модели и новые зависимости не скачивались.

**Воспроизведение проверки** из корня репозитория, PowerShell:

```powershell
@'
from pathlib import Path
from collections import Counter
import hashlib
import re
from tools.gguf_reader import GGUFFile

p = Path(r'H:\models\mimo-v2.6-flash\MiMo-V2.6-Flash-RL-GSQ-RCO-3.5bit.gguf')
g = GGUFFile(p)
assert g.version == 3 and g.alignment == 32
assert g.metadata['general.architecture'] == 'mimo2'
assert g.metadata['mimo2.block_count'] == 48
assert g.metadata['mimo2.nextn_predict_layers'] == 0
assert g.metadata['mimo2.export_scope'] == 'text_trunk_without_mtp_or_modality_companions'
assert len(g.metadata) == 49
assert len(g.tensors) == len({t.name for t in g.tensors}) == 472
size = p.stat().st_size
end = 0
for t in sorted(g.tensors, key=lambda t: t.offset):
    n = t.expected_bytes()
    assert n is not None and n > 0, t.name
    assert t.offset % g.alignment == 0, t.name
    assert t.offset >= end, t.name
    assert g.data_start + t.offset + n <= size, t.name
    end = t.offset + n
assert g.data_start + end == size == 134982426368
assert (g.header_end, g.data_start) == (5979134, 5979136)
assert not any('nextn' in t.name for t in g.tensors)
layers = {int(m[1]) for t in g.tensors if (m := re.match(r'blk\.(\d+)\.', t.name))}
assert layers == set(range(48))
routed = [t for t in g.tensors if '_exps.weight' in t.name]
routed_bytes = sum(t.expected_bytes() for t in routed)
total_bytes = sum(t.expected_bytes() for t in g.tensors)
assert len(routed) == 141 and routed_bytes == 122909884416
assert total_bytes == 134976447232
assert total_bytes - routed_bytes == 12066562816
types = Counter(t.type_name for t in g.tensors)
assert dict(types) == dict(Q2_K=50, Q3_K=74, MXFP4=17, BF16=101, F32=230)
pattern = g.metadata['mimo2.attention.sliding_window_pattern']
assert len(pattern) == 48 and set(pattern) == {0, 1}
assert [i for i, s in enumerate(pattern) if not s] == [0, 5, 11, 17, 23, 29, 35, 41, 47]
assert g.metadata['mimo2.attention.head_count_kv'] == [8 if s else 4 for s in pattern]
with p.open('rb') as stream:
    header_sha = hashlib.sha256(stream.read(g.header_end)).hexdigest()
assert header_sha == '46e0ff64f95482e89997ba7e661d36a7061a7a5961e033fe5baa6b241e23269a'
template_sha = hashlib.sha256(g.metadata['tokenizer.chat_template'].encode('utf-8')).hexdigest()
assert template_sha == '11ea52e156de38a458e6b7720ad45915d65b97d4ec979a09f55e3c9bd1b4d059'
print('PASS header/ranges/pattern; payload bytes:', total_bytes)
print('Routed/other bytes:', routed_bytes, total_bytes-routed_bytes)
print('Types:', dict(types))
print('Header SHA-256:', header_sha)
print('Template SHA-256:', template_sha)
'@ | python -X utf8 -
```

Команда проверяет зафиксированный локальный файл, не содержимое его матриц.
Она не заменяет постоянный строгий contract, numerical fixtures или inference.

**Проверка документации:** приведённый Python-код повторно выполнен из этого
Markdown, exit0, `PASS header/ranges/pattern`; размеры, типы и hashes совпали.
Локальные ссылки, code fences и отсутствие trailing whitespace — PASS.
Арифметика payload/H2D и оценок KV — PASS. Это не измерение inference.

## Правила обновления

- Каждая запись `MIMO-01`, `MIMO-02` и далее: дата, исходный commit/dirty state,
  patch hashes, изменённые файлы, точные команды, exit codes и PASS/FAIL/SKIPPED.
- Ссылки давать на созданные JSON/logs. Запланированные файлы явно помечать
  как будущие; результаты другой модели не заносить в MiMo validation.
- Фиксировать model/header/template hashes, dependency SHA, GPU/driver/compiler,
  context/KV/FA/graphs, batch/ubatch, sampler/seed, cache/readers/chunks.
- Скорость: load/prefill/decode/request отдельно; учитывать reasoning/EOS,
  prompt/output counts, cold/warm/prefix reuse и минимум три повторения A/B.
- Память: global used/available RAM, process commit/working set, pinned buffers,
  GPU used/total, cache allocations, KV и peaks; disk/H2D bytes и page faults.
- Оптимизация принимается после correctness на тех же quants и границах SWA.
  После нового квантования это отдельный quality/performance эксперимент.
- Не объявлять MTP или мультимодальность готовыми по upstream code/config.
  Отсутствующие веса и непроверенные режимы сохранять явно в статусе.

## Шаблон следующей записи

```text
### MIMO-NN / Pn.m — YYYY-MM-DD HH:MM, Asia/Yekaterinburg — Название
Основание: commit, dirty state, dependency/patch SHA, GGUF/template hash.
Изменено: файлы и поведение.
Команды: точные build/test/benchmark команды.
Проверки: PASS/FAIL/SKIPPED, exit codes, fixtures/corpus, SWA boundaries.
Результаты: logits/token parity, TTFT/tokens/s, память, cache/H2D/SSD, повторы.
Артефакты: ссылки на существующие отчёты и логи.
Не закрыто: ошибки, ограничения, отсутствующие веса и непроверенные режимы.
Следующий шаг: конкретная задача и критерий приёмки.
```
