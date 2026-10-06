# Step-3.7-Flash: профилирование CPU-кэша

Дата: 2026-10-06, Asia/Yekaterinburg. База
`9c457bf526e5916d513bedccf0a25878cf36d026`, поверх STEP-13–15.

**Устойчивое ускорение генерации не подтверждено.** Выбор вытесняемой записи
стал дешевле на32,24%, но финальная скорость11,692 →11,521 токена/с без MTP
и13,761 →13,698 с MTP2. Сопутствующая CPU-нагрузка менялась. Direct остаётся
opt-in, прежние defaults сохранены.

STEP-16: измерить cache lookup/admission, выбор вытесняемой записи, опрос
памяти и подготовку router plan; затем проверить конкретный кандидат.
Все58 ответов совпали с эталоном. [Машинный отчёт](STEP37_FLASH_CACHE_PROFILE.json)
сохраняет команды, IDs, timing/CPU/memory samples, unit cases и hashes.

## Профилирование

`tools/check_step35_mtp.py --cache-profile` включает host timers только для
экспериментального checker. Без этого флага новые таймеры не читают clock.
`generation_io` хранит разность counters после prefill и после decode.
Это wall time CPU-потока, включая ожидания и вытеснение ОС, не CUDA timeline.

- `cache_get_ms`: device check, frequency record, lookup и LRU splice.
- `cache_admit_ms`: admission целиком, включая вложенные refresh, victim scan,
  cudaFree и cudaMalloc. `cache_victim_ms` — только поиск кандидата;
  `cache_allocate_ms`/`cache_free_ms` — операции с GPU allocations.
- `cache_refresh_ms`: memory probe, расчёт бюджета и возможный trim;
  `memory_probe_ms` — только provider глобальных RAM/VRAM readings.
- `cache_protect_ms`: построение residency pins; `cache_trim_ms` — trim.
- `plan_build_ms`: после окончания предыдущего плана, включая проверку
  metadata, contains, pins и запуск ring. `plan_end_ms`: drain, снятие pins,
  trim и учёт ring. Вложенные времена нельзя суммировать.

Первый контроль без MTP: четыре прогретых ответа, два P1 prompts ×два раунда.
Среднее на ответ: get209,307 мс, admission264,625 мс, из него victim scan219,945 мс;
plan build49,026 мс, plan end10,313 мс. Victim scan —3,510% времени генерации,
а memory probe2,695 мс —0,043%. Опрос памяти не является заметным ограничением
этой серии; его частота и все budget95 guards сохранены.

## Кандидат

`--cache-scan baseline,direct` чередует два пути в одном процессе. `baseline`
сохраняет прежний поиск. `direct` использует стабильный адрес `std::map` entry
в LRU node, избегая повторного map lookup для каждого из32 кандидатов. Score
лучшего кандидата хранится локально. При первом score0 поиск можно закончить:
score неотрицателен, история внутри scan не меняется, равные scores выбирают
первую, самую старую запись. Pins, окно32 незакреплённых кандидатов, frequency
decay, rejection, cache size и allocations остаются прежними.

При eviction/reuse LRU node удаляется перед map entry; ни один указатель
на удалённую запись не сохраняется. Переключение — между запросами после
drain. Оба новых API имеют явный `noexcept(false)` для MSVC `/EHsc`.
`cache_fast_scan` и `cache_profile` подтверждаются в RESULT. Legacy probes
принимаются только для default baseline/off; неподтверждённый direct отклоняется.

Default остаётся baseline. Стабильный `strata-step35.exe` не пересобирался;
общие pipeline/frequency helpers и другие model backends не изменены.

## Проверки

- 25 cache checks PASS:131200 совпавших решений и membership сравнение после
  каждого admission, разные размеры allocation, pins, trim, два перехода
  frequency epoch, смена режима; отдельно первый нулевой score и oldest tie.
- 204 runtime cases PASS: побайтовое сравнение GPU copies и bit-exact logits,
  включая eviction/direct с F32/mixed weights,1/2 readers, early/batch,
  cached/WC и prompt513. Включение/выключение timers не меняет результат.
- 24 Python setup/shards/profile/tokenizer tests PASS.

## Стенд и метод полной модели

Windows, Ryzen9 9950X, RAM128 ГиБ (125,555 видно), RTX5090 32607 МиБ,
driver581.80, power limit400W, PCIe5×16 под нагрузкой. Main: локальный
Step-3.7-Flash UD-Q4_K_S106,323 ГиБ; draft: официальный Q8_0, shared embedding,
три resident heads, исходный catch-up, depth2/p_min0.6.

Context2048, batch17, F32 KV, FA/TF32 off, reader1, ring slots8 МиБ,
cached pinned RAM, reuse on, prefill admission off, cache cap16384MiB с
динамическим global budget. Без MTP draft не загружается. Pipeline batch
без MTP, batch+early с MTP2. Два P1 prompts с54/88 выходными IDs включая EOG,
каждый ответ проверяется по полному списку IDs. Это короткий greedy benchmark,
не admission длинного контекста, stochastic sampling или native HTTP/MTP.

Каждый вариант прогревается на обоих prompts. Порядок baseline/direct меняется
на обратный через раунд. Финальную скорость необходимо сравнивать с выключенными
новыми timers; profiled series проверяет механизм. CPU load записывается для
каждого ответа; background load не контролируется. Global RAM/VRAM monitor
останавливает только собственный child при превышении95%.

## Результат

Парное профилирование без MTP, два раунда после прогрева:

| Среднее на ответ | Baseline | Direct |
|---|---:|---:|
| Victim scan, мс | 218,723 | 148,207 |
| Admission целиком, мс | 259,950 | 189,818 |
| Cache get, мс | 201,164 | 202,397 |
| Просмотрено кандидатов | 870872 | 872812 |
| Генерация, токенов/с | 11,891 | 11,926 |

Victim scan стал дешевле на32,24%, admission —на26,98%. Это снижение времени
отдельной операции: общая скорость изменилась лишь на+0,30%, а знаки по раундам
различались (+1,99% и−1,37%). Кэш и ring сохраняются между запросами: число
hits/admissions зависит от порядка и динамического бюджета. Совпадение policy
проверено отдельно на одинаковой последовательности131200 обращений.

Финальные четыре раунда с выключенными новыми timers:

| Режим | Baseline, токенов/с | Direct, токенов/с | Изменение |
|---|---:|---:|---:|
| Без MTP, batch | 11,692 | 11,521 | −1,46% |
| MTP2, batch+early | 13,761 | 13,698 | −0,45% |

Без MTP: три отрицательных раунда, один+0,16%; среднее время пары полных
запросов15,657 →15,884 с. Средняя сопутствующая CPU-нагрузка7,59 →14,39%
всей32-поточной CPU capacity; условия неравны. Ускорение не подтверждено,
а собственное влияние алгоритма на общую скорость не отделено от фоновой нагрузки.

С MTP2: изменения по раундам+1,20%,−0,15%,−7,90%,+5,15%; средняя сопутствующая
нагрузка11,81 →12,69%, в отдельных ответах до31,19%. Пара запросов14,280 →14,317 с.
Acceptance одинаковый348/376 (92,55%) для каждого варианта. Эти данные не дают
оснований включить direct по умолчанию. `other_percent` включает OS/driver
busy time вне model process и не доказывает влияние конкретного приложения.

Среднее source wall time без MTP3,906 →3,985 с на ответ, consumer wait
1,631 →1,708 с; эти интервалы частично перекрываются. Экономия на admission
не гарантирует такого же сокращения generation, пока producer готовит данные
параллельно. Следующий кандидат требует отдельных замеров memcpy/page faults,
H2D ready и ожиданий между слоями, прежде чем менять общий pipeline.

Всего58/58 exact responses (20 с MTP,14 warmups), четыре model process exit0.
Пик по всем сериям: RAM75,49%, VRAM30238MiB/92,735%; monitor95 не сработал.
Синтетические проверки prompt513 не заменяют MTP rollback за границей SWA512;
MTP checker остаётся ограничен480 позициями. Production admission не заявляется.

Raw: `build-local/step35-cuda/step16-cache/`. Сохранённый probe:
`strata-step35-mtp-check-direct-scan-tested.exe`, SHA-256
`5401731cd52462593b424b1eef1709743a0f2a07a6bc3a2e973896d7898c52cf`.
Стабильный engine SHA
`32a88bd85855395dbc79df2da323d971a9543c2847c618e7e34245a771bf9df3`
сохранён; private build использует статические библиотеки.
