# Step-3.7-Flash: групповые копии экспертов

Работа начата2026-10-05, завершение измерений2026-10-06, Asia/Yekaterinburg.
Ревизия до изменений: `9c457bf526e5916d513bedccf0a25878cf36d026`.

**Без MTP:9,546 →11,364 токена/с (+19,04%). С MTP2:11,194 →13,655
(+21,98%), с дополнительным early refill —13,941 (+24,54%).**
Контроль уже использует allocation reuse из STEP-12. Это два коротких P1 prompt
на данном ПК, а не оценка произвольного длинного диалога.

[Машинный отчёт](STEP37_FLASH_BATCH_COPY.json) сохраняет все IDs, timings,
memory peaks, source/binary hashes, команды и первую неуспешную попытку.

## Скорость и память

Финальные серии: четыре парных раунда после прогрева.

| Режим | Baseline | Batch | Batch + early |
|---|---:|---:|---:|
| Без MTP, токена/с | 9,546 | **11,364 (+19,04%)** | 11,397 (+19,39%) |
| MTP2, токена/с | 11,194 | **13,655 (+21,98%)** | **13,941 (+24,54%)** |
| Без MTP, пара с prefill | 19,023с | 16,137с (−15,17%) | 16,171с (−14,99%) |
| MTP2, пара с prefill | 17,218с | 14,362с (−16,59%) | 14,107с (−18,07%) |

Групповые копии быстрее контроля во всех четырёх раундах обеих серий.
Раннее заполнение добавило к batch только0,29% decode без MTP, а время полной
пары стало чуть больше: устойчивого дополнительного выигрыша здесь не установлено.
С MTP2 дополнительная прибавка составила2,10%; batch+early был быстрее batch
во всех четырёх раундах. Кандидат no-MTP —batch; MTP2 —batch+early. Это выбор
для следующих проверок, а не включение нового рабочего default.

Среднее число host fences на copy stream за генерацию ответа:
**71146 →8820** без MTP и **55813 →3402** с MTP2 — примерно в8,1 и16,4 раза меньше.
GPU/host ring по-прежнему4 ×8MiB. H2D-трафик остаётся близким:74,272 →76,248GiB
на ответ без MTP;73,416 →72,688GiB с MTP2/batch и72,479GiB с batch+early.
Это ускорение доставки и ожиданий; вычисления модели и routing не упрощались.

| Полная серия | Пик RAM | Пик VRAM |
|---|---:|---:|
| Sweep четырёх вариантов | 87,916GiB /70,02% | 30249MiB /92,77% |
| Подтверждение без MTP | 89,656GiB /71,41% | 30265MiB /92,82% |
| Подтверждение MTP2 | 91,971GiB /73,25% | 30214MiB /92,66% |

**84/84 ответов** успешных серий совпали с native reference, включая warmup;
из них30 с MTP. Acceptance во всех измеряемых MTP режимах348/376 =92,55%.
Ещё8 точных warmup answers первой прерванной серии сохранены отдельно и не
входят в84. Monitor95 не завершал ни один из этих запусков.

Первый завершённый sweep всех четырёх вариантов дал baseline9,700, early9,817,
batch11,606, batch+early11,843 токена/с без MTP. Для вывода выше использованы
отдельные более длинные подтверждающие серии, а не лучшие значения sweep.

## Реализация

`step_pipeline_copy_tensor()` отправляет копии всех выбранных экспертов одного
весового тензора и делает один `cudaStreamSynchronize` перед его использованием.
Прежний путь ожидал завершения каждой матрицы и каждого cache fill отдельно.
Scheduler hook применяется только к GPU `MUL_MAT_ID` с CPU-mapped экспертами,
при явно включённой опции. Native scratch dependency fence сохранён.

Все уже закэшированные источники защищены существующим router plan. Новые
записи, куда ещё идёт cache fill, получают временные residency pins. Они
удерживаются до завершения copy stream, в том числе при exception cleanup.
Блоки не могут быть вытеснены или переиспользованы во время копирования.
За пределы функции незавершённые cache reads/writes не выходят; отдельного
retired pool и новых GPU buffers для batching нет.

Раннее заполнение pinned RAM-буфера проверяется отдельно. Host slot можно
перезаписать после завершения его предыдущего H2D; device slot по-прежнему
защищён событием предыдущего потребителя. Использован уже существующий режим
общего `StrataExpertPipeline`; файлы `backends/common/` не изменены.

Четыре режима offline checker:

| `--pipeline-modes` | Групповые копии | Раннее заполнение host slot |
|---|---|---|
| `baseline` | Нет | Нет |
| `early` | Нет | Да |
| `batch` | Да | Нет |
| `batch-early` | Да | Да |

Обычный API настройки конвейера выбирает baseline. Новый расширенный API
используют checker и CUDA tests; pipe/HTTP defaults не менялись. Reuse allocation
из STEP-12 включён во **всех** сравниваемых режимах.

## Метод

Windows/9950X/128GiB/RTX5090, driver581.80, наблюдаемый power limit400W;
во время теста PCIe5 ×16. Параметры питания не менялись. Main: локальный
UD-Q4_K_S106,323GiB. MTP: официальный Q8_0 с проверкой checksum и metadata,
общий embedding, depth2, p_min0.6, исходный catch-up всех трёх голов.

Context2048, batch17, F32 KV, FA/TF32 off, pipeline1/chunk8MiB, prefill admission
off, cache reuse on, cache cap16384MiB с динамическим ограничением по глобальной
VRAM. Без MTP draft не загружается. P1 prompts содержат38/34 input tokens,
ответы54/88 tokens с EOG. Каждый запрос начинает со свежих KV/speculative state,
а GPU-кэш экспертов сохраняется в течение серии.

Первый sweep: четыре режима, по два прогревочных запроса на режим и два
измеряемых раунда. Подтверждение: baseline/batch/batch-early, такой же прогрев
и четыре измеряемых раунда; порядок конфигураций обращается на нечётных раундах.
Контрольные no-MTP и MTP2 серии работают в разных процессах.

Decode = сумма output tokens / сумма generation seconds. Время пары запросов
включает prefill и исключает load. Выбор конфигурации происходит до request
timer. Переключение только batching сохраняет ring. При смене early refill ring
создаётся заново; если нужен запас, cache controller освобождает недостающий
объём плюс64MiB, затем снова проверяет бюджет. Это может немного менять cache
residency между конфигурациями; frozen-cache replay здесь не выполнялся.

## Проверки и первая неуспешная попытка

В первой версии после8 exact warmup answers сработал guard создания нового ring:
`Step pipeline ring exceeds available memory budget`. Серия не дала измеряемых
результатов. Глобальные пики monitor: RAM69,59%, VRAM92,89%; превышения95% не было.
Причина отказа — требуемый резерв для замены ring рядом с заполненным кэшем.
Добавлено сохранение одинакового ring при смене batching и освобождение запаса
перед необходимой заменой. Исходная попытка сохранена в JSON как неуспешная.

Финальные CUDA проверки:

- **36 pipeline checks PASS:**1/2 readers ×4/8/16MiB ×early off/on, wrap всех
  размеров ring на payload66MiB+513B, tails/guards, отказ неверному source,
  отмена и повторный запуск с изменёнными исходными байтами.
- **157 runtime cases PASS:**36 новых cold/warm вариантов и4 переключения
  режимов поверх STEP-12. Сравниваются реальные GPU bytes и побитовые logits,
  F32/mixed weights, маленький и большой cache,1/2 readers, prompt513.
- **24 существующих Step Python tests PASS**, `py_compile` и `git diff --check`.

Synthetic prompt513 проверяет transport через SWA boundary без MTP. Это не
admission MTP rollback через512. Checker полной модели по-прежнему ограничен480
суммарными позициями и greedy; native pipe/HTTP, длинные запросы и cancel/recovery
с новыми опциями требуют отдельной проверки перед выбором рабочего default.

## Воспроизведение и counters

Сборка — по [backend README](../../backends/step35/README.md). Raw results:
`build-local/step35-cuda/step13-copy/`. Для повторения выбирать новые output dirs:

```powershell
$stepProbe = 'build-local/step35-cuda/bin/strata-step35-mtp-check-batch-copy-tested.exe'
$stepModel = 'H:/models/Step-3.7-Flash/UD-Q4_K_S/Step-3.7-Flash-UD-Q4_K_S-00001-of-00004.gguf'
python -X utf8 tools/check_step35_mtp.py --engine $stepProbe --model $stepModel --depths 0 --cache-reuse on --pipeline-modes baseline,batch,batch-early --cache-mib 16384 --rounds 4 --output-dir build-local/step35-cuda/repeat-batch-no-mtp
python -X utf8 tools/check_step35_mtp.py --engine $stepProbe --model $stepModel --draft build-local/models/step37-mtp/Step3.7-flash-mtp-Q8_0.gguf --draft-placement shared-embedding --depths 2 --cache-reuse on --pipeline-modes baseline,batch,batch-early --cache-mib 16384 --rounds 4 --output-dir build-local/step35-cuda/repeat-batch-mtp2
```

`generation_io.pipeline_copy_fences` считает host waits на copy stream;
не включает отдельный native scratch/backend fence. `pipeline_batch_ms` — wall
time всей групповой доставки с ожиданием source и cache admission. В batch режиме
индивидуальный `d2d_ms` не измеряется и равен0; это не отсутствие D2D. CPU counters
перекрываются во времени, их нельзя складывать как CUDA timeline.

Проверенный probe имеет SHA-256
`da5568b807e0695aab28f36fcbeb97ac49fb5b18f0bd7f17251706e7ba6bf723`.
Рабочий `strata-step35.exe` не пересобирался и сохранил SHA-256
`32a88bd85855395dbc79df2da323d971a9543c2847c618e7e34245a771bf9df3`.

Следующий performance-кандидат —1/2 readers уже с групповыми копиями: прежний
reader sweep выполнялся с поштучными ожиданиями, поэтому его вывод нельзя
автоматически переносить на новый путь. Такой дополнительный выигрыш ещё
не измерен. До переноса оптимизаций в рабочий профиль также нужны длинные
pipe/HTTP запросы, cancel/recovery и memory pressure.
