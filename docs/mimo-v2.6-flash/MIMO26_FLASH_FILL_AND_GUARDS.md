# MiMo: заполнение кеша и CPU-проверка logits

Проверено **2026-10-07**, Asia/Yekaterinburg. Windows, RTX5090 32 ГБ,
128 ГБ RAM; `MiMo-V2.6-Flash-RL-GSQ-RCO-3.5bit.gguf`, 134 982 426 368 байт.
CUDA13.0.48, sm120, Unsloth `86ebfef2`.

Выполнены два отдельных опыта. Групповые cache fills уменьшили число
ожиданий, но не ускорили эту конфигурацию; они оставлены выключенными.
Перегрузка проверки `require` для строкового литерала убрала временные строки
в цикле по logits и включена в обычную сборку: **7,721→7,933 ток/с (+2,74%)**
на прогретых запросах, +2,63% относительно быстрейшего контроля своей серии.

[Групповые fills: замеры](MIMO26_FLASH_FILL_BATCH_BENCHMARK.json),
[CPU guard: замеры](MIMO26_FLASH_LITERAL_GUARDS_BENCHMARK.json),
[сборки, проверки и диагностика](MIMO26_FLASH_OPTIMIZATION10_VALIDATION.json).

## Общие условия

Slab16, frequency decay65536, tensor delivery1, cache до14 ГиБ с live clamp,
mmap/reader1/chunk8 МиБ, prefill admission off. Context512, batch8, F32 KV,
FA on, greedy, обычный physical SWA. MTP и CUDA trace выключены.

В каждой ABBA-серии три темы: счёт, Python, русский текст. Первый запрос темы
помечен прогревом, затем три измеренных запроса по32 output tokens. KV свежий,
кеш и история сохраняются между темами процесса. На вариант18 измеренных
запросов и558 timed decode steps; на серию48 запросов/1536 output IDs,
включая прогревы. Все сохранённые logits и IDs сравниваются побитово.

Ток/с — сумма decode steps / сумма generation time, без prefill, загрузки
и первого output в числителе, с проверкой logits, sampling и записью F32.
System disk counters включают все процессы и диски. Пики RAM/VRAM — выборки;
runtime отдельно применяет global95%. CPU model nodes запрещены аудитом.
Проценты разных серий и прошлых MIMO-07/08/09 нельзя складывать.

## Групповые fills: эксперимент выключен

`ExpertCache::FillBatch` сначала резервирует места в прежнем порядке admission,
затем копирует только сохранившиеся reservations. Во время выбора мест GPU-копии
ещё не запущены: временную запись можно вытеснить без ожидания. Ticket отличает
повтор того же ключа даже при переиспользовании того же адреса. Pending entries
не возвращаются из `get`/`contains`, не становятся route pins и не публикуются
до общего fence. На ошибке после enqueue fence предшествует удалению записей.
При отмене до submit незапущенные reservations просто удаляются.

Scratch остаётся защищён прежней delivery fence; compute не запускается до
завершения fills. Это группировка внутри одного scheduler input, без нового
overlap compute. Future hits и guard tails по-прежнему защищены route pins.
LRU/frequency admission и physical budget сопоставлены с синхронным вариантом
на16 000 случайных mixed-size операций, включая decay и давление.

Одна сборка `7e34b3c5…`, порядок fill0/1/1/0:

| Измерение | Синхронные fills | Групповые fills |
|---|---:|---:|
| Общая скорость, ток/с | 7,861 | 7,826 |
| Host fences доставки/fills | 127 178 | 102 797 |

Ожиданий стало на19,17% меньше, скорость изменилась на−0,45%.
Процессы по порядку:7,857 /7,774 /7,879 /7,865 ток/с. На этой машине
полезный прирост не подтверждён. Полный GGUF прошёл6 prompts и3 509 248
exact logits, отмену/ошибку→fresh; ABBA1536 IDs и234 356 736 F32 значений exact.

Есть отдельная **неразрешённая численная диагностика**. Первый native runtime
прошёл244/244, но Python wrapper ожидал прежние212 cases. Проверка количества
исправлена с отдельным regression test; исходный error report сохранён.
Следующий запуск дал243/244: `mixed/file_pipe_1_8/logits`,
max_abs0,000754818320274353, NMSE7,4306189e−7, первый differing index0.
Byte audit прошёл. Ещё один запуск дал244/244, обе проверки сохранённой
сборки до изменений —212/212. Это не доказывает исправление или независимость
ошибки от группировки. Новая mixed-сигнатура не объявлена той же ошибкой,
что прежняя F32 max_abs0,0005061785 из MIMO-07/08.

`STRATA_MIMO_CACHE_FILL_BATCH=0` остаётся default. `1` — эксперимент только
при включённом tensor pipeline; sync reader и per-matrix delivery его не
используют. Новые метрики: `cache_fill_batch`, `cache_pending`,
`pipeline_fill_batches`, `pipeline_fill_submissions`. После выхода из delivery,
успешного запроса, ошибки или отмены pending должен быть0.
Fill counters считают фактически отправленные D2D и их fences, включая ошибки;
это не GPU execution time. Не считать этот режим рекомендованным ускорением.

## CPU guard: временные строки устранены

Ранее `require(bool, const std::string &)` создавал временную строку
`non-finite logits` для каждого из152576 значений. Добавлена перегрузка
`require(bool, const char *)`: на успешном пути строка не создаётся,
при ошибке выбрасывается тот же `std::runtime_error`. Динамические сообщения
остались на прежней перегрузке. Все проверки NaN/Inf сохраняются.

CPU microbenchmark на этой машине, MSVC Release, четыре прогона по100 строк
из152576 конечных значений, порядок old/new/new/old:

| Измерение | Временная string | Литерал |
|---|---:|---:|
| Выделения на одну строку logits | 152 576 | 0 |
| Время одной строки, мс, два прогона | 3,221 /3,211 | 0,0553 /0,0586 |

Число выделений измерено в отдельном CPU executable с перехватом `operator new`.
Другие стандартные библиотеки могут держать эту строку inline. Эти числа
описывают guard, не всё время inference. Проверены NaN, ±Inf, максимальный
конечный float, subnormal и прежний текст исключения для динамической строки.

Для end-to-end сравнения сохранена старая сборка `7e34b3c5…`, новая —
`75b86c2c…`. Оба варианта используют fill0. Разница кода engine между ними —
только перегрузка в `contract.hpp`; дополнительные изменения — CPU checker,
CMake target и возможность сравнить два executable в benchmark runner.
Порядок процессов old/new/new/old: **7,729 /7,978 /7,889 /7,713 ток/с**.

| Измерение | Временная string | Литерал |
|---|---:|---:|
| Общая скорость, ток/с | 7,721 | 7,933 |
| Счёт, ток/с | 9,520 | 9,946 |
| Python, ток/с | 6,801 | 6,947 |
| Русский текст, ток/с | 7,328 | 7,480 |
| Время вне decode на output token, мс | 3,877 | 0,345 |
| Сумма decode_forward за558 steps, с | 70,036 | 70,142 |
| Среднее полное время запроса, с | 7,400 | 7,302 |
| Decode H2D, ГиБ | 626,183 | 640,238 |
| Cache reserved, среднее ГиБ | 12,821 | 12,829 |
| Cache payload, среднее ГиБ | 12,443 | 12,437 |
| Peak global VRAM, ГиБ | 29,506 | 29,506 |
| Peak global RAM, ГиБ | 108,645 | 107,752 |

Время вне decode вычислено как `generation_ms - decode_forward_ms`, делённое
на число output tokens, включая первый output. Оно включает проверки logits,
sampling, экспорт и ожидания вывода; это не чистый CPU profiler. Оно снизилось
на91,11%, тогда как суммарный decode_forward изменился на+0,15%. Это согласуется
с устранением работы при обработке logits, без ускорения матричной арифметики.
H2D немного вырос; изменчивый live clamp и admission означают, что поведение
кеша не следует объявлять идентичным по всем счётчикам между процессами.

System disk reads прогретых запросов по порядку:0,155 /0,108 /0,356 /0,502 ГиБ.
Пики новой сборки:VRAM92,66%, RAM85,82%. Все48 запросов/1536 IDs и234 356 736
сравнённых F32 значений прошли exact parity. Отдельный corpus6 prompts:
3 509 248 logits exact, invalid/cancel→fresh и unload PASS.
Проверка unset env/default CLI на EN/SWA249 и recovery сохранена в shipping
validation; она проверяет, что fill batching остаётся0.

## Проверки и воспроизведение

CPU:27 cache,56 slab,21 frequency,17 fill-batch и11 literal cases PASS,
5000/5000/16000 смешанных операций соответственно;29 Python tests PASS.
CUDA runtime:212 cases контроля и244 cases grouped fills, включая32 новых
проверки ошибки после первого D2D enqueue и последующего восстановления.
Все исходные FAIL/error reports сохранены; численные допуски не менялись.
96 kernel/132 graph checks относятся к прежним этапам и здесь не повторялись.

```powershell
$mimoModel = 'H:\models\mimo-v2.6-flash\MiMo-V2.6-Flash-RL-GSQ-RCO-3.5bit.gguf'
python -m tools.benchmark_mimo2_cache --build build-local/mimo2-cuda --model $mimoModel --cuda-bin build-local/cuda-13.0/bin/x64 --axis fill --fixed-cache-decay 65536 --order 0,1,1,0 --output-dir build-local/mimo2-validation/fill-abba-new
python -m tools.benchmark_mimo2_cache --build build-local/mimo2-cuda --model $mimoModel --cuda-bin build-local/cuda-13.0/bin/x64 --axis binary --control-binary build-local/mimo2-fill-measured/strata-mimo2.exe --order 0,1,1,0 --output-dir build-local/mimo2-validation/literal-abba-new
build-local/mimo2-cuda/bin/strata-mimo2-literals-check.exe
```

Каталоги должны быть новыми. First command теперь использует CPU guard в обоих
вариантах; точная измеренная до-guard сборка и исходники сохранены в
`build-local/mimo2-fill-measured`. Shipping guard build —
`build-local/mimo2-literal-measured`. Копии бинарников требуют прежний CUDA PATH.
Контрольные runners принимают `--cache-fill-batch 0|1`, default0; прочие
benchmark axes явно выключают fills. Binary axis всегда сохраняет fill0.

Контексты выше512, длительный внешний memory pressure, другие GPU, MTP quality,
API/profile и sessions остаются отдельными задачами. Существенный overlap
H2D/compute этим этапом не подтверждён.
