# Step-3.7-Flash: копирование из mmap в pinned RAM

Дата: 2026-10-06, Asia/Yekaterinburg. База
`9c457bf526e5916d513bedccf0a25878cf36d026`, поверх STEP-13–16.
STEP-17: разобрать source staging и ожидания конвейера, затем проверить
AVX2 temporal stores как отдельный opt-in кандидат. С MTP2 получено
13,803 → 14,777 токена/с (+7,06%), преимущество во всех четырёх парных раундах.
Без MTP повторный контроль не подтвердил ускорения. Defaults не переключены.

## Выбор кандидата

Отдельный screening: RAM512 МиБ, четыре cacheable pinned slots по8 МиБ,
payload2949632 байта (матрица Step Q4_K плюс512 padding),4096 копирований
на вариант, один прогрев и шесть раундов с обращением порядка.
Windows, Ryzen9 9950X, RTX5090. Это скорость CPU-копирования, не инференса:

| Вариант | ГиБ/с |
|---|---:|
| CRT memcpy | 23,994 |
| AVX2 с обычными записями | 44,727 |
| AVX2 non-temporal stores | 23,725 |

Выбран вариант с обычными записями. Гипотеза: повторно используемые buffers
выгодно оставлять в CPU-кэше; фактическое размещение cache lines этим тестом
не измерено. GPU coherence и H2D могут изменить результат, поэтому screening
не является основанием включить алгоритм по умолчанию.

## Реализация и область изменения

`host_copy.cpp` — единственный новый translation unit с `/arch:AVX2` на MSVC
или `-mavx2` на GCC. Функция не встраивается в вызывающий код. Перед включением
проверяются CPUID, OSXSAVE/AVX и XCR0 на Windows, compiler CPU dispatch на GCC.
Неподдерживаемый CPU/OS отклоняет explicit AVX2; default остаётся CRT.

Копирование выравнивает destination до64 байт коротким prefix, переносит по128
байт четырьмя vector loads/stores, затем копирует tail. Чтений за заданным
диапазоном нет. Это temporal stores, отдельный sfence для non-temporal записи
не требуется. Математика модели, padding и quantization не меняются.

В shared `expert_pipeline.hpp` добавлен необязательный HostCopy callback.
При его отсутствии прежний `std::memcpy` сохранён. Новых буферов, events,
очередей или правил владения слотами в shared primitive нет. Только Step
подключает callback; source/executable других backends не пересобирались.
При исключении callback работает прежняя обработка ошибки producer и drain.

`strata_step_pipeline_config_copy()` выбирает CRT/AVX2 между drained requests.
Без tracing смена алгоритма сохраняет ring, cache и history. Legacy config
APIs возвращают CRT и выключают profiling. Entry point имеет явный
`noexcept(false)`. Рабочий `strata-step35.exe` не пересобирается.

## Диагностика

`--host-copy crt,avx2` задаёт paired sweep. `--pipeline-profile` (только без
MTP) включает host copy wall time, Windows thread cycles, число копирований
дольше500 мкс и существующий GpuTrace на128 graphs каждого запроса. Timers
и CUDA event allocation влияют на исполнение: эта серия исследует механизм,
финальный throughput измеряется без новых profilers.

Host copy counters атомарные: owner может снимать snapshot, пока reader
работает. В `generation_io` записываются decode deltas после prefill. Thread
cycles не переводятся в миллисекунды: frequency/preemption не контролируются.
`source_ms` включает wrapper overhead; `host_copy_wall_ns` охватывает только
копирование. Все такие интервалы могут перекрываться с GPU/другими readers.

GPU events стоят на реальных H2D и compute streams. Сохранённые graph summaries
содержат H2D/compute/overlap и span; `uncovered_span_ms` включает, в частности,
неизмеренные D2D и host gaps, поэтому не является доказательством GPU idle.
Raw trace сохраняет все intervals; короткие graph summaries и hash — в отчёте.
Profiling пересоздаёт ring перед каждым запросом для нового GpuTrace; этот
режим не участвует в решении о финальной скорости.

`process_page_faults` — разница Windows process counters вокруг всего запроса,
включая prefill, CUDA/runtime allocations и diagnostics. Это soft+hard faults;
она не показывает отдельно source faults или чтение SSD. На других ОС поле
может быть null. CPU load и global RAM/VRAM monitor сохранены.

## Проверки

- 828 byte/guard cases:0..513,4095/4096/4097 и2949120/2949632 байта, смещения
  source/destination0,1,15,31,32,63; недоступная страница сразу за концом
  allocation. Ещё три checks для profiler off/on и invalid mode.
- 217 CUDA ring checks: старые72 для callback-free CRT плюс CRT/AVX2 callbacks,
  chunks4/8/16,1/2 readers, cached/WC, early on/off, wrap, tails, cancel/restart.
  Отдельно callback exception propagation/drain.
- 217 runtime cases: bytes и bit-exact logits; AVX2/legacy-CRT/AVX2 switches,
  F32/mixed weights,1/2 readers, eviction/reuse, prompt513, host counter deltas.
- 24 Python setup/shards/profile/tokenizer tests, py_compile PASS.

## Полная модель

Main: локальный Step-3.7-Flash UD-Q4_K_S106,323 ГиБ; draft: официальный Q8_0,
shared embedding,3 resident heads, исходный catch-up, MTP2/p_min0.6.
Windows/9950X/RAM128GiB/RTX5090 32607MiB, driver581.80, power limit400W.
Context2048, batch17, F32 KV, FA/TF32 off, reader1/chunk8/cached/reuse on,
prefill admission off, cache cap16384MiB с прежними global budget95 guards.
Cache scan baseline, profiling cache off. Без MTP draft не загружается.

Два P1 prompts,54/88 выходных IDs включая EOG. Каждый ответ сравнивается с
эталоном. Каждый вариант прогревается, порядок меняется через раунд. Это
короткий greedy эксперимент; HTTP/MTP, SWA rollback, stochastic sampling,
long-context/cancel/pressure admission этим этапом не заявляются.

## Что показало профилирование

Один измеряемый раунд после прогрева, среднее по двум ответам, decode:

| Метрика | CRT | AVX2 |
|---|---:|---:|
| Host copy, ГиБ/с | 20,120 | 30,000 |
| Thread cycles на байт | 0,19957 | 0,13343 |
| Host copy wall, мс/ответ | 3701,99 | 2441,16 |
| Source scope, мс/ответ | 3731,12 | 2458,46 |
| Consumer wait, мс/ответ | 1502,79 | 1059,04 |
| Producer slot wait, мс/ответ | 192,09 | 855,32 |
| GPU H2D events, мс/ответ | 2259,03 | 2600,70 |
| GPU compute events, мс/ответ | 785,49 | 781,93 |
| Decode, токенов/с | 11,442 | 11,446 |

CPU-копирование стало дешевле примерно на треть в cycles/byte, но H2D и
ожидание свободного слота выросли. В этой диагностической серии общий decode
практически не изменился. Объём H2D также немного отличается: состояние cache
и global budget не фиксируются искусственно. Складывать перекрывающиеся
интервалы нельзя. Копирований дольше 500 мкс в четырёх измеряемых ответах нет.

На первом холодном CRT-прогреве зарегистрированы 12 214 977 и 4 230 014 process
faults. В измеряемых ответах с tracing: CRT в среднем 11 104,5, AVX2 — 11 088.
Это не свидетельство чтения соответствующего количества страниц с SSD:
счётчик включает soft faults и выделения диагностических/runtime buffers.

## Скорость без диагностических timers

Throughput — суммарные выходные IDs, делённые на суммарное generation time.
«Полная пара» включает prefill и генерацию обоих prompts, но не loading и
смену конфигурации. Во всех рядах сравниваются CRT и AVX2 одной серии.

| Серия | Раунды | CRT → AVX2, токенов/с | Изменение | Полная пара CRT → AVX2, с |
|---|---:|---:|---:|---:|
| Без MTP, batch, первый sweep | 2 | 11,479 → 11,674 | +1,69% | 15,867 → 15,218 |
| Без MTP, batch, новый процесс и обратный начальный порядок | 4 | 11,268 → 11,214 | −0,48% | 16,202 → 15,817 |
| MTP2, batch+early | 4 | 13,803 → 14,777 | **+7,06%** | 14,179 → 12,899 |

MTP2: парные прибавки +9,31%, +6,68%, +3,63%, +8,71%. Acceptance у обоих
вариантов одинакова: 348/376 (92,55%). Процессорное время model process,
сумма по его потокам за полный запрос: **10,146 → 9,035 CPU-с/ответ (−10,95%)**.
Это не wall time и не оценка энергопотребления.

Без MTP первый выигрыш не воспроизвёлся: в контрольной серии три из четырёх
парных раундов отрицательные (+3,69%, −0,31%, −2,65%, −2,54%). CPU time всё же
11,600 → 10,832 CPU-с/ответ (−6,62%). Более короткое время полного запроса не
доказывает ускорение decode. Также проверенный early refill без MTP дал
нестабильные результаты: CRT 11,736, AVX2 11,488 токена/с в первом sweep;
для no-MTP прежний batch/CRT кандидат сохранён.

## Нагрузка CPU во время тестов

Все проценты нормированы на 32 логических CPU. В финальной MTP2 серии
средняя загрузка model process CRT/AVX2 — 4,47/4,38%, остальной системы —
16,81/19,52%. В no-MTP подтверждении — 4,47/4,28% и 32,21/31,82% соответственно.
Остаток включает работу ОС/драйвера, в том числе возможную работу по запросу
движка; его нельзя целиком приписывать посторонним приложениям.

После замечания пользователя и завершения model process выполнен отдельный
трёхсекундный замер: система 33,28%, Blender 23,44%, Telegram 3,08%, System
0,23%. Это отдельный снимок после теста, не разложение предыдущих запросов.
Он подтверждает сохранение значительной нагрузки после выхода Strata.
Другие процессы и приоритеты не менялись. Первый пробный снимок всех процессов
не использован: обход PID растянул интервалы, а общий трёхсекундный знаменатель
исказил per-process проценты. Сохранённый снимок использует собственный wall
interval каждого PID и не считает System Idle Process полезной загрузкой.

Фоновая нагрузка не контролировалась. Поэтому абсолютные скорости нельзя
напрямую сравнивать с предыдущими STEP; результат MTP — наблюдение этой
парной серии, а не изолированная оценка влияния AVX2 при любом фоне.

## Результат и артефакты

**72/72 ответов exact**, из них 20 warmups и 20 ответов с MTP. Это повторения
двух P1 prompts, не 72 разных задачи. Четыре model process завершились с
кодом 0; monitor95 не срабатывал. Общие пики RAM **75,24%**, VRAM
**30257 МиБ / 92,79%**. Ограничения памяти не ослаблялись.

AVX2 оставлен offline-кандидатом для MTP2/shared embedding/reuse/batch+early,
reader1/chunk8/cacheable pinned RAM. Включение: `--host-copy avx2`; диагностические
профили выключены. Для no-MTP подтверждённого decode gain нет. Default CRT,
рабочий native engine и HTTP-профиль сохранены. Перед production admission
нужны длинные/state/cancel/pressure проверки, а MTP — ещё SWA512 rollback,
stochastic и pipe/HTTP integration.

Все IDs, timings, CPU/memory, команды, source/binary hashes, unit reports и
сводки traces: [STEP37_FLASH_HOST_COPY.json](STEP37_FLASH_HOST_COPY.json).
Raw logs/traces: `build-local/step35-cuda/step17-host/`.
CPU screening source и строки результатов включены в JSON.

Сохранённый probe: `strata-step35-mtp-check-host-copy-tested.exe`, SHA-256
`6d203f6a5afa94d17793d15f6d485a3de5ed8e7525066861602ceeb60b801509`.
Рабочий `strata-step35.exe` сохранил SHA-256
`32a88bd85855395dbc79df2da323d971a9543c2847c618e7e34245a771bf9df3`.
