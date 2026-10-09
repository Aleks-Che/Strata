# MM27-33 — CUDA events и асинхронные участки графа

Перенос идеи matrix-level event dependencies из
[GLM](../GLM53/GLM53_FLASH_SUCCESSFUL_OPTIMIZATIONS.md). Дата:2026-10-09.
Режим2 дал **4,086→4,266 токена/с (+4,40%)** в медианах трёх независимых
пар на локальном MiniMax Q4_K_M. Полные logits совпали побитно.
Это native opt-in: default `--pipeline-events 0` и server EXE98e85e80…
сохранены на этом этапе. Последующий
[MM27-34](MINIMAX_M27_COPY_EVENTS_CONTEXT.md) проверил2K/4K, sessions,
EOS/real pressure/cancel и подключил mode2 к серверу как opt-in с этим EXE.

## Реализация

Native `--pipeline-events 0|1|2` требует включённых file pipeline и D2D batch:

- 0: прежние CPU fences и synchronous split compute.
- 1: одна пара CUDA events на матрицу, прежний synchronous split compute.
- 2: те же events плюс asynchronous split compute.

Перед записью в scratch backend записывает `scratch_released` в свой CUDA
stream; delivery stream ждёт это событие. После всех выбранных ranges и fills
delivery записывает `copy_ready`, backend stream ждёт его перед compute.
Пара переиспользуется после публикации соответствующих waits; каждый wait
относится к уже записанному событию. Используется event API закреплённой CUDA
dependency, где `ggml_backend_event.context` содержит `cudaEvent_t`.

Scheduler пропускает прежнее input ожидание только для выбранных экспертных
весов, доставляемых этим механизмом. User inputs, другие веса и передачи между
backend сохраняют upstream зависимости. Router IDs по-прежнему читаются с
синхронизацией; предсказания будущих экспертов нет.

Новые fills закреплены до retirement события копирования. Retirement перенесён
перед cache decisions следующей матрицы: CPU может сначала отправить compute.
Ready flags групп и RAM/GPU partition обновляются только после завершения
копии. В один момент отложена максимум одна матрица; pending slots нельзя
повторно использовать или освободить раньше. Граница router plan сохраняет
drain очереди и delivery stream. Observer получает завершённые bytes и поэтому
принудительно вызывает retirement; performance runner работает без observer.

В режиме2 графы отправляются асинхронно, но graph scope синхронизирует все
задействованные CUDA backends на любом выходе, включая ошибку и отмену. Затем
отменяется незавершённый план. Публичная граница decode остаётся синхронной:
операции не переживают scratch, graph или context. Shared transport, dtype,
математика модели, частотная политика и precision settings не изменены.

`compute_ms` в режиме2 — время CPU submission участков, а не GPU compute wall
time; это указано в `compute_ms_scope`. Сравнивать скорость нужно по
`decode_ms`, `decode_tokens_per_second` и `request_ms`. Добавлены counters
events, пропущенных scheduler waits, retirement checks/waits, observer fences,
async compute calls и graph exit fences. `pipeline_pending_copy` должен быть0
на границе завершённого запроса.

Это не устранение всех host waits. Strict F32 CUDA `MUL_MAT_ID` по-прежнему
копирует router IDs на CPU и синхронизирует stream, затем синхронизирует H2D
таблицы перестановки. Scheduler уже читает IDs для выбора экспертов; повторное
использование этих данных на этом этапе ещё не реализовано. Последующий
[MM27-35](MINIMAX_M27_ROUTER_IDS.md) добавляет отдельный native opt-in.
Математика strict F32 и исторический fast-quant logit FAIL остаются прежними.

## Профиль и методика

RTX5090 32607МиБ, RAM125,555ГиБ, Windows/driver581.80/CUDA13.0,
MSVC19.44.35222.0. Q4_K_M target138342384352 bytes из
`H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf`.
Context2048/batch16, cache18ГиБ/arena64МиБ/reserve0/decay65536,
file/readers2/chunk4/lookahead1/D2D batch1. Strict F32 activations/KV,
TF32/FA/graphs/MTP/DFlash off; RAM expert cache, grouping, prefix/archive off.

Сохранён корпус MM27-32: четыре запроса A,A,B,A, по128 output tokens;52/52/64/52
prompt tokens. KV свежий, expert cache сохраняется внутри процесса. OS file
cache не очищается. Сначала один процесс на режим0/1/2, затем отдельная
повторная серия выбранной пары. Полные logits сравниваются с сохранённым
синхронным EXE98e85e80…; каждый процесс ограничен Job/timeout и независимым
Windows/NVML observer95%. Все копии и наблюдатели завершаются до хеширования.

Основание: Strata HEAD `54467693c8354d8c64a72ec1cb083473a529b4ba` с ранее
существовавшими staged/unstaged changes; dependency
`86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9`. Каждый каталог измерений содержит
копии исходников, EXE, manifest с generated/patch hashes, запросы и сырые логи.
Model header SHA проверяется по MM27-32; checksum всего138ГБ файла не вычислен.

Это короткие фрагменты ответов, не natural-EOS или answer-quality проверка.
Медиана агрегата считается по полной сумме времени508 forward decode steps,
а не как среднее четырёх token/s. Загрузка модели в request time не включена.

## Измерения

Первичный screen: по одному свежему процессу на режим0/1/2. Decode агрегата:
**4,046 /4,046 /4,234 токена/с**. Одних matrix events оказалось недостаточно
для измеренного выигрыша; режим2 выбран для отдельной повторной серии.
Screen в итоговые медианы не включён. Порядки повторов:0→2,2→0,0→2.

| Пара | Режим0, токена/с | Режим2, токена/с | Изменение | Полное время0→2, с |
|---|---:|---:|---:|---:|
| 1 | 4,085945 | 4,265577 | +4,40% | 164,030→157,808 |
| 2 | 4,117071 | 4,290843 | +4,22% | 163,532→156,956 |
| 3 | 4,000357 | 4,163195 | +4,07% | 167,270→161,424 |
| Медиана | **4,085945** | **4,265577** | **+4,40%** | **164,030→157,808** |

Полное время корпуса сократилось на3,79%. Ниже отдельные медианы запросов;
их сумма не обязана совпадать с медианой полного времени корпуса.

| Запрос | Режим0, токена/с | Режим2, токена/с | Изменение |
|---|---:|---:|---:|
| A, пустой expert cache | 4,1765 | 4,3612 | +4,42% |
| Повтор A | 4,4739 | 4,6773 | +4,55% |
| Новая тема B | 3,8310 | 3,9759 | +3,78% |
| Возврат A | 3,9218 | 4,1096 | +4,79% |

Все девять процессов, включая screen, сохранили одинаковые cache decisions:
на workload decode H2D984,085ГиБ, hit по байтам50,968%, fills125,150ГиБ,
40711 evictions. Speedup не получен изменением кэша, точности или routing.
Каждый mode2 workload:94488 пар событий на decode,94996 async split calls,
508 graph-exit fences. С prefill:97464 пары и524 graph-exit fences.
Все retirement queries оказались ready; блокирующих retirement waits0.
Router/plan/internal CUDA waits этим счётчиком не учитываются.

## Проверки и границы

Сборка и все команды завершились с exit0. Общий итог screen+повторов:
**36 requests /4608 IDs /921894912 F32 logits** совпали с сохранённым
синхронным EXE98e85e80… побитно; все значения finite.
**1256 CUDA fixture checks**:228/228/228 для0/1/2,120 для expert groups,
240 для RAM cache64МиБ и212 для mode2 без lookahead. Проверены cuda/arena,
F32/mixed GGUF, delivery bytes, observer-free runs, pending fills,
ошибки чтения/copy/compute, отмена, injected pressure/OOM, recovery,
identity/reload/unload. Эти маленькие fixtures не заменяют full-model pressure.
Шесть CPU test methods и восемь CLI cases PASS.

Независимый validator проверил **357 gates**, hashes всех сохранённых
артефактов/исходников, счётчики зависимостей/drain, расчёт timings и memory
samples.2917 отсчётов: global RAM≤23,710%, VRAM≤85,085%; лимит95% не превышен.
В этом эксперименте не создавалось реальное давление около95%.
[Машинный итог](MINIMAX_M27_COPY_EVENTS_CHECK.json).

На момент MM27-33 оставались long-context/prefix/archive parity и live
pressure/cancel/recovery на EXE13a885cc…; они проверены MM27-34 при batch16
и context≤4096. Прежние independent oracle, answer-quality и MM27-06
discrepancy gates остаются. Результат скорости относится
к указанному корпусу, машине и профилю; это не общий performance default.

## Воспроизведение

Собрать `strata-minimax-m2-bench` и `strata-minimax-m2-cache-check`.
Кандидат13a885cc… сохранён отдельно в
`build-local/minimax-m2-copy-events-candidate/engine.exe`; принятый сервером
EXE98e85e80… и admission SHA остаются прежними.

```powershell
python -m unittest tools.test_minimax_m2_copy_events
python tools/check_minimax_m2_copy_events.py --out build-local/minimax-m2-copy-events-screen-01
python tools/check_minimax_m2_copy_events.py --out build-local/minimax-m2-copy-events-ab-01 --modes 0 2 --repeats 3
python tools/validate_minimax_m2_copy_events.py --runs build-local/minimax-m2-copy-events-screen-01 build-local/minimax-m2-copy-events-ab-01 --output docs/minimax-m2.7/MINIMAX_M27_COPY_EVENTS_CHECK.json
```

Cache fixtures используют native `cache-check.exe` с прежними pipeline flags
и новым `--pipeline-events N`. Проверены0/1/2, режим2 с группами экспертов,
режим2 с RAM cache64МиБ и режим2 с `--pipeline-lookahead 0`.
`build-local/minimax-m2-copy-events-fixture-*-01`
содержат исходные tiny GGUF и reports; stdout/stderr сохранены рядом.
Скрипт `tools/validate_minimax_m2_copy_events.py` повторно проверяет hashes,
конфигурацию, cache decisions, event/drain accounting, numerical records,
memory samples и расчёт результатов.

Для повторения выбрать новые output directories: runner не перезаписывает
существующие измерения. Сборка выполнена через
`build-local/build-minimax-m2-copy-events.bat` (vcvars64, CMake/Ninja, CUDA13.0);
логи `build-local/minimax-m2-copy-events-configure.log` и
`build-local/minimax-m2-copy-events-build.log`. После сохранения candidate
основной bench EXE восстановлен из сохранённого98e85e80… для server admission.
Чтобы испытать новый флаг, передать candidate явно в
`tools/run_minimax_m2.py --engine build-local/minimax-m2-copy-events-candidate/engine.exe -- ...`
и добавить `--pipeline-events 2` к проверенному профилю18/2/4/lookahead/D2D.

Candidate SHA-256:
`13a885cc75352bb39cc4231174da3d9f03705d5f9c150117935d6f05ea9d2401`.
Измерения: `build-local/minimax-m2-copy-events-screen-01/report.json` и
`build-local/minimax-m2-copy-events-ab-01/report.json`; retained full logits
лежат рядом. Тесты CLI/CPU:
`build-local/minimax-m2-copy-events-cli.json` и
`build-local/minimax-m2-copy-events-cpu.log`.
