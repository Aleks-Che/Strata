# MM27-35 — повторное использование host router IDs

Дата:2026-10-10. Реализован native opt-in `--router-host-ids 1` поверх
строгого F32-пути и matrix events из
[MM27-33](MINIMAX_M27_COPY_EVENTS.md). Default0 сохранён.

## Что изменено

Scheduler уже читает ID выбранных экспертов на CPU перед загрузкой весов.
Раньше strict Q4_K/Q6_K CUDA `MUL_MAT_ID` повторял это чтение и ожидание stream.
Новый путь передаёт собственную копию этих bytes в ту же CUDA-операцию.
Арифметика, сортировка строк, dequantization и cuBLAS остаются прежними.

`router_ids.hpp` хранит один snapshot размером до64КиБ на scheduler thread.
Проверяются точный node, weight, IDs tensor, GPU storage/buffer, type, shape,
strides, view source/offset и число экспертов. Layout исходного и скопированного
IDs tensor должен совпадать. ID проверяются на допустимый диапазон.
Snapshot используется один раз; несовпадение также его инвалидирует.
Tokenwise slices и неподдерживаемый layout возвращаются к прежнему D2H.

RAII scope очищает snapshot на входе и любом выходе из scheduler split,
включая отмену/ошибку. Reset/release также очищают его. Указатели не служат
ключами между графами, запросами или загрузками модели. При активном eval
callback scheduler не публикует snapshot: callback может изменить данные.
Данные потребляются во время CPU submission, не остаются доступны GPU после
возврата и не требуют удержания тензора между split.

Добавлены `router_ids_published`, `router_ids_hits`, `router_ids_misses`,
`router_ids_bytes`. CUDA graph policy, scheduler router synchronization,
H2D fence таблицы перестановки и events2 graph-exit drain сохранены.
Это устраняет только повторное D2H чтение ID и связанный с ним host wait.

Патч применяется к отдельному generated CUDA dispatch после проверки SHA
закреплённого upstream. Runtime-off build не получает bridge call.
Другие model backends и shared transport не изменены.

## Сборка и проверки

Кандидат: `build-local/minimax-m2-router-ids-candidate/engine.exe`, SHA-256:
`612354a12513eb1ce616ac2cc370bdd314a5c4725760bb586f6b9dd0ee1e6177`.
Рядом сохранены manifest, `router-ids-check.exe`, `cache-check.exe`.
Legacy EXE98e85e80… восстановлен по прежнему пути, events EXE13a885cc… сохранён.
Сервер принимает прежние точные SHA; новый EXE пока предназначен для native A/B.

- 97 checks: owned snapshot, одноразовое чтение, несовпадение node/storage/
  shape/stride/view/expert count, scope при исключении, смена graph при тех же
  адресах; CUDA batch1/8/16, events0/2, changing tokens/positions и повторные
  загрузки.18 пар,28800 F32 logits совпали побитно, все конечные.
- 230 cache/pipeline checks: selected bytes, cancel/recovery, partial fill,
  worker faults, controlled pressure, metadata identity, unload/reload.
  Reuse включён, попадания явно проверены на mixed-quant fixtures.
- 12 Python methods проверяют event/reuse evidence;4 CLI cases отвергают
  неверные значения и mode1 до чтения весов.

Артефакты финальной сборки: `build-local/minimax-m2-router-ids-fixtures-02`,
`build-local/minimax-m2-router-ids-cache-02`, `...router-ids-unit.log`,
`...router-ids-cli.json`. Предварительные fixtures-01/cache-01 относятся к
ранней сборке до callback guard и сохранены отдельно. Первый CLI probe без
CUDA PATH не запустился корректно; его четыре процесса завершены, результат
сохранён в `...router-ids-cli-missing-cuda-path.json`. Повтор с штатным
`runtime_environment` прошёл4/4. В численные и скоростные PASS этот probe не входит.

## Профиль измерения

RTX5090 32607МиБ, Ryzen9 9950X, RAM125,555ГиБ, Windows, driver581.80,
CUDA13.0/MSVC19.44.35222.0. Target138342384352 bytes:
`H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf`.

Context2048/batch16, cache18ГиБ/arena64МиБ/reserve0/decay65536,
readers2/chunk4/lookahead1/D2D1/events2. Strict F32 activations/KV,
TF32/FA/graphs/DFlash/MTP off. RAM expert LRU и prefix/session reuse off.
Три свежие пары процессов: off/on, on/off, off/on. В каждом A,A,B,A по128
выходных токенов; KV очищается, история expert cache между запросами остаётся.
OS file cache не контролируется. Все выходные logits сверяются с сохранённым
legacy reference MM27-32; отдельный observer останавливает процесс при>95%.

```powershell
python -X utf8 tools/check_minimax_m2_router_ids.py --out build-local/minimax-m2-router-ids-ab-NEW --repeats 3
python -X utf8 tools/check_minimax_m2_router_ids_batches.py --out build-local/minimax-m2-router-ids-batches-NEW
python -m unittest tools.test_minimax_m2_router_ids tools.test_minimax_m2_copy_events
python tools/validate_minimax_m2_router_ids.py
```

## Результаты

| Пара | Off, токенов/с | On, токенов/с | Прирост |
|---|---:|---:|---:|
| 1: off/on | 4,176444 | 4,457950 | +6,74% |
| 2: on/off | 4,294597 | 4,404567 | +2,56% |
| 3: off/on | 4,291695 | 4,470481 | +4,17% |
| Медианы | **4,291695** | **4,457950** | **+3,87%** |

Это decode throughput:508 serial forward steps на workload. Медиана полного
времени четырёх запросов **157,033→152,176с** (−3,09%). Медианы отдельных
случаев a_empty/a_repeat/b_new_topic/a_return быстрее на5,29/3,95/3,19/4,82%.
Все три пары показали положительный результат, но разброс сохраняется;
измерение не обещает тот же процент на другом корпусе или длинном контексте.

24 requests,3072 output IDs и **614596608 F32 logits** совпали с сохранённым
legacy reference побитно, без NaN/Inf. Во всех шести процессах совпали transport
и cache counters: decode H2D984,084985ГиБ, hit50,967889%, evictions40711,
fill125,150253ГиБ. Это суммарные передачи на четыре ответа, не размер резидентных
весов. Изменение скорости не связано со сменой маршрутов или cache policy.

Каждый on workload:97464 `router_ids_hits`,0 misses. Prefill2976 hits/38323200
bytes, decode94488 hits/3023616 bytes. Prefill IDs имеют исходные padded
strides: snapshot включает промежутки, а проверка/индексация сохраняют layout.
Graph-exit drains524 на workload остаются; режим не отключает GPU зависимости.

Дополнительно full-model batch1/8: четыре процесса off/on, два исходных prompt
по8 выходных токенов. Все8 ответов завершены (64 IDs суммарно). В двух парах
**32 IDs /6402048 F32 logits** совпали побитно при том же batch, transport и
cache counters одинаковы. Это короткая correctness проверка, отдельная от
скоростных измерений. Артефакты: `build-local/minimax-m2-router-ids-batches-01`.

Независимый observer:2144 samples для десяти full-model процессов; максимумы
RAM**22,213%**, VRAM**84,406%**, ошибок/превышений95% нет. Этот A/B сохраняет
прежний cache18ГиБ для сравнимости; он не является real-pressure тестом.
Все десять native PID завершены. После работы GPU1267МиБ/1%.

Итоговый [машинный отчёт](MINIMAX_M27_ROUTER_IDS_CHECK.json): **695 gates,
163 artifact/source SHA-256 PASS**. Валидатор повторно сравнил logits,
пересчитал timings, проверил transport, исходники/generated patches,
memory samples, прежние binary SHA и завершение процессов.
Raw speed corpus: `build-local/minimax-m2-router-ids-ab-01`; build logs:
`build-local/minimax-m2-router-ids-configure.log`, `...router-ids-build.log`.

## Границы

Этот этап не закрывает full-model2K/4K, prefix/archive, EOS/sampling и real
pressure для нового EXE. Tiny pressure/cancel/reload не заменяет эти проверки.
Скоростной full-model A/B относится к batch16. Короткий full-model batch1/8
проверяет parity при том же batch; он не измеряет длинный контекст или sessions.
Server defaults и его exact-EXE admission не изменены. Качество ответов,
независимый full-model oracle и старое MM27-06 reload расхождение остаются open.

Последующий [MM27-36](MINIMAX_M27_ROUTER_IDS_CONTEXT.md) закрыл эти
long/session/real-pressure/EOS/sampling/cancel проверки при batch16/ctx4K
и подключил тот же EXE к серверу как opt-in. Скоростные числа выше остаются
измерением MM27-35; defaults off сохранены.
