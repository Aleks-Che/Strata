# Step-3.7-Flash: повторное использование GPU-блоков кэша

Дата: 2026-10-05. Ревизия перед работой:
`c81b0c0b92d63e8c24aef24446dadb9d71dc178d`.

В отдельном checker получено **9,363 → 9,873 токена/с без MTP (+5,45%)**
и **10,355 → 11,560 с MTP2 (+11,64%)**. Изменение повторно использует GPU-блок
вытесняемого эксперта, сокращая вызовы `cudaFree`/`cudaMalloc`. Рабочий pipe/HTTP
профиль пока не включает эту опцию.

[Машинный отчёт](STEP37_FLASH_CACHE_REUSE.json) содержит все 40 ответов с IDs,
timings, counters, hashes, команды, пики памяти и результаты CUDA checks.
Это два коротких P1 prompt, повторённые в разных режимах, а не 40 разных задач.
Raw reports со всеми memory samples и stderr находятся в
`build-local/step35-cuda/step12-speed/`.

## Изменение

В `ExpertCache::admit()` старая запись уже выбрана существующим алгоритмом
частоты/давности. Если её allocation имеет тот же размер с учётом округления,
запись не закреплена текущим router plan и замена укладывается в бюджет,
её GPU-указатель передаётся новой записи. Затем caller полностью записывает
новые веса до разрешения следующего чтения.

Сохраняется действующий контракт: предыдущие D2D-чтения блока завершены.
Оптимизация не снимает CUDA synchronization и не вводит asynchronous cache
leases. Блоки другого размера выделяются прежним способом. Trim и pressure
освобождают память; отдельного пула запасных allocations нет. Счётчик growth
и проверки глобального RAM/VRAM бюджета продолжают работать и при reuse.

API `strata_step_cache_reuse()` вызывается между запросами. По умолчанию reuse
выключен; новый CLI `--cache-reuse off,on` есть у Python MTP checker, который
передаёт режим на каждый запрос. Pipe/server/UI и общие backends не изменены.

## Стенд и метод

- Windows, Ryzen 9 9950X, RTX 5090 32607 MiB, driver581.80; RAM128 GiB,
  доступный ОС объём125,555 GiB. Настройки питания не менялись.
- Main: локальный Step-3.7-Flash UD-Q4_K_S, четыре shards,106,323 GiB.
  Sidecar: официальный Q8_0 из STEP-11; его checksum и metadata повторно проверены.
- Context2048, batch17, F32 KV, FA/TF32 off, pipeline1/chunk8MiB,
  prefill admission off, cache cap16384MiB с динамическим ограничением по VRAM.
- MTP2: shared embedding, все три головы resident, upstream catch-up,
  `p_min=0.6`, demand paging. Без MTP draft вообще не загружен.
- Один процесс на каждую серию: no-MTP и MTP2. Внутри процесса кэш экспертов
  сохраняется; KV и speculative state очищаются перед каждым запросом.
- Два P1 prompt:38/34 input и54/88 output tokens, включая EOG. Для каждого
  режима два warmup requests, затем четыре измеряемых раунда с порядком
  off/on, on/off, off/on, on/off. По8 измеряемых ответов на режим.
- Decode = сумма output tokens / сумма generation time. Среднее время пары
  включает prefill, исключает загрузку модели. Monitor завершает только свой
  child при превышении95% общей RAM/VRAM; ни одна серия не прервана.

## Измерения

| Режим | Reuse off, токена/с | Reuse on, токена/с | Прирост decode | Пара с prefill, off → on |
|---|---:|---:|---:|---:|
| Без MTP, без draft | 9,363 | 9,873 | +5,45% | 19,312 → 18,402 с (−4,71%) |
| Q8 MTP2, shared embedding | 10,355 | 11,560 | +11,64% | 18,176 → 16,742 с (−7,89%) |

В серии MTP контроль менялся по мере прогрева: парный throughput off
9,843 /10,209 /10,710 /10,709; on11,376 /11,255 /11,823 /11,810 токена/с.
Выигрыш есть во всех четырёх раундах. Только последние два дают
10,709 →11,817, около+10,3%; общий результат не основан на лучшем отдельном ответе.
Два коротких prompt не устанавливают выигрыш для длинного диалога.

Среднее число новых GPU allocations за генерацию одного ответа:

| Режим | До | После | Повторно использовано блоков |
|---|---:|---:|---:|
| Без MTP | 622 | 24,625 | 614,750 |
| MTP2 | 1385,625 | 51,750 | 1278,750 |

Трафик H2D почти не изменился: no-MTP79,254 →79,575 GB, MTP78,328 →78,097 GB
в среднем на ответ (десятичные GB, только generation). Это уменьшение накладных
расходов замены записей, а не новый способ выбирать экспертов или уменьшать веса.
Число misses и admissions немного отличается из-за сохраняемого кэша, частотной
истории и текущего динамического бюджета; это не replay одного frozen cache.

Пики всей системы за серию, включая warmup:

| Серия | RAM | VRAM |
|---|---:|---:|
| Без MTP | 88,900 GiB /70,81% | 30220 MiB /92,68% |
| MTP2 | 93,370 GiB /74,37% | 30234 MiB /92,72% |

Во всех **40/40** ответах IDs совпали с P1 native reference. Из них20 с MTP;
его acceptance в измеряемой части каждого режима —348/376 =92,55%.

## Проверки и границы

- 23 CUDA cache checks PASS: reuse при недоступном новом allocation, замена
  identity и содержимого, pins/unpin, различный размер, duplicate admission,
  pressure trim, OOM и отсутствие скрытого пула.
- 117 runtime cases PASS. Шесть добавленных вариантов проверяют exact GPU bytes
  и побитовые logits: F32/mixed weights × sync/reader1/reader2, prompt513.
  Это synthetic transport/SWA check без MTP, не admission MTP rollback через512.
- 24 существующих Step Python tests PASS, `py_compile` и `git diff --check` PASS.
- Существующий `strata-step35.exe` не пересобирался, SHA-256 прежний:
  `32a88bd85855395dbc79df2da323d971a9543c2847c618e7e34245a771bf9df3`.
- Точный проверенный probe сохранён как
  `build-local/step35-cuda/bin/strata-step35-mtp-check-cache-reuse-tested.exe`, SHA-256
  `b7af43c4e6c0355eb43400ab521179819548b18b2ac00c554657da7d859a2482`.

Полная модель проверена в пределах480 позиций checker. До включения reuse
в рабочем профиле нужны native pipe/HTTP длинные запросы, cancel/recovery и
memory-pressure с этой опцией. MTP по-прежнему требует отдельных state/rollback
и integration gates из STEP-11.

## Повторить

Сборка checker — по [backend README](../../backends/step35/README.md).
Команды из корня репозитория; для нового запуска выбрать новые output directories:

```powershell
$stepProbe = 'build-local/step35-cuda/bin/strata-step35-mtp-check-cache-reuse-tested.exe'
$stepModel = 'H:/models/Step-3.7-Flash/UD-Q4_K_S/Step-3.7-Flash-UD-Q4_K_S-00001-of-00004.gguf'
python -X utf8 tools/check_step35_mtp.py --engine $stepProbe --model $stepModel --depths 0 --cache-reuse off,on --cache-mib 16384 --rounds 4 --output-dir build-local/step35-cuda/repeat-reuse-no-mtp
python -X utf8 tools/check_step35_mtp.py --engine $stepProbe --model $stepModel --draft build-local/models/step37-mtp/Step3.7-flash-mtp-Q8_0.gguf --draft-placement shared-embedding --depths 2 --cache-reuse off,on --cache-mib 16384 --rounds 4 --output-dir build-local/step35-cuda/repeat-reuse-mtp2
```

## Дальнейшие кандидаты

1. Уменьшить число синхронизаций между D2D-копиями матриц одного router plan.
   Потребуются явные consumer events/leases и сохранение защиты aliasing scratch.
   Сейчас синхронизация выполняется после каждой матрицы; ускорение этой
   следующей оптимизации ещё не измерено.
2. Проверить early host refill для Step: копировать следующую матрицу в pinned
   host buffer после H2D event, сохраняя отдельную защиту device slot.
   Shared primitive уже имеет такой режим, Step оставляет его выключенным.
3. Подбирать confidence/depth MTP после ускорения основного пути. В этой серии
   draft+catch-up заняли в среднем116ms из6142ms генерации MTP2/reuse-on (~1,9%);
   большой выигрыш только от ускорения этих вычислений текущими числами не обоснован.

Счётчики source/D2D/wait — перекрывающиеся CPU wall-time sums; их нельзя складывать
или выдавать за CUDA timeline. Необходимость следующих изменений определяется
парными измерениями и exact parity, а не загрузкой RAM ради процента.
