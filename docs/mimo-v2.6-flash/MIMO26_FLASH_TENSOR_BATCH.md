# MiMo: доставка экспертов по тензорам

Измерено **2026-10-07**, Asia/Yekaterinburg. Windows, RTX5090 32 ГБ,
128 ГБ RAM; `MiMo-V2.6-Flash-RL-GSQ-RCO-3.5bit.gguf`, 134 982 426 368 байт.
CUDA13.0.48, sm120, Unsloth `86ebfef2`.

Итоговый ABBA дал **6,591 → 7,299 ток/с (+10,7%)** без MTP. Контроль заметно
ускорился от начала к концу серии:6,282→6,933. Если сравнивать оба новых
прогона с последним прогретым контролем, прибавка **5,3%**. Все сохранённые
logits и выходные токены совпали. Новый режим включён в defaults MiMo engine.

[Замеры](MIMO26_FLASH_TENSOR_BATCH_BENCHMARK.json),
[сборки, проверки и диагностика](MIMO26_FLASH_TENSOR_BATCH_VALIDATION.json).

## Изменение

Раньше backend ждал освобождения scratch перед каждым выбранным диапазоном,
а поток копирования — после доставки каждой матрицы эксперта. Теперь все
выбранные эксперты **одного scheduler input** доставляются до общей fence.
Перед началом сохраняется backend fence: scratch может содержать живые
активации предыдущего consumer. Producer по-прежнему пишет только device ring.

Route pins удерживают все cache sources, в том числе источники512-байтовых
guard tails. Во время доставки admission и eviction не выполняются. После
её завершения воспроизводится исходный порядок LRU touches и cache fills.
Отложенный `touch` не удваивает счётчик hits. Каждый fill остаётся синхронным
и публикуется только после завершённой копии. При ошибке или отмене runtime
дожидается завершения доставки до снятия pins; арифметика модели не менялась.

Это уменьшает число ожиданий, но не число необходимых весов. В этом ABBA
decode H2D отличался менее чем на0,1%; cache payload оставался около12,3 ГиБ.
Асинхронные cache fills, частотный допуск, CUDA Graphs и новое перекрытие
compute этим этапом не реализованы. Shared transport других backend не изменён.

## Измерения

Один executable, порядок `0,1,1,0`; в обоих режимах slab16. Три темы: счёт,
Python и русский текст. В каждом процессе для каждой темы первый запрос —
прогрев, затем три измеренных запроса по32 токена. Итого48 запросов,
1536 output IDs; на каждый режим18 измеренных запросов и558 decode steps.
Ни медленный русский контроль, ни медленный русский запрос нового режима
не удалялись из результатов.

Context512, batch8, F32 KV, FA on, greedy, обычный physical SWA основного engine,
свежий KV на каждом запросе, cache сохраняется. Cache до14 ГиБ с live clamp,
mmap, reader1, chunk8 МиБ, prefill admission off. MTP и trace выключены.
Скорость — сумма decode steps / сумма generation time; без первого output,
prefill и загрузки, с sampling и записью logits.

| Измерение | Доставка по матрицам | Доставка по тензорам |
|---|---:|---:|
| Общая скорость, ток/с | 6,591 | 7,299 |
| Счёт, ток/с | 7,961 | 8,585 |
| Python, ток/с | 6,139 | 6,532 |
| Русский текст, ток/с | 6,001 | 7,071 |
| Среднее полное время прогретого запроса, с | 8,034 | 7,264 |
| Decode H2D за558 steps, ГиБ | 680,213 | 679,552 |
| Cache reserved, среднее ГиБ | 12,883 | 12,876 |
| Cache payload, среднее ГиБ | 12,324 | 12,364 |
| Host fences доставки/fills | 1 220 541 | 306 091 |
| Дополнительные backend fences перед доставкой | 935 226 | 89 676 |
| Peak global VRAM, ГиБ | 29,506 | 29,506 |
| Peak global RAM, ГиБ | 113,674 | 113,454 |

Счётчики fences включают **prefill и decode** всех18 прогретых запросов режима.
Они исключают route drains, остальные scheduler/backend waits и producer events.
Снижение —74,92% и90,41% соответственно; это не измерение CUDA execution time.

По порядку процессов: **6,282 /7,388 /7,212 /6,933 ток/с**. System disk reads
за прогретые запросы:0,104 /0,029 /0,068 /0,062 ГиБ. Счётчики включают все
процессы и диски, не изолируют модель или H:. Разброс русского контроля
5,331→6,863 ток/с сохраняется в JSON; нельзя приписать весь ABBA-прирост
только сокращению fences. Относительно последнего контроля отдельные новые
прогоны быстрее примерно на4,0–6,6%.

Пик VRAM нового режима —92,66% доступного объёма, RAM —90,36%. Это пики
выборок; runtime дополнительно сохраняет global admission95% и существующие
резервы. CPU model nodes и полные копии expert tensors аудит не обнаружил.
Числа MIMO-07 получены другой серией и не используются как контроль здесь.

## Проверки

- 27 CPU cache cases, включая deferred LRU touch/counters,56 slab cases,
  5000 смешанных операций и28 Python tests PASS.
- Новый runtime212/212 PASS: F32/mixed, file/mmap, readers1/2, chunks4/16,
  cache8/256 МиБ, guard tails, prefill, pressure, bytes, logits, cancel/error
  recovery и drain. В завершённых pipeline cases проверено3 901 685 760 байт.
- Full-model corpus6 prompts, включая SWA249, EN/RU/ZH, код и числа:
  3 509 248 F32 logits и IDs совпали с сохранённым managed-cache reference.
  Invalid request, cancel→fresh и unload PASS.
- В ABBA все48 запросов прошли exact logits/IDs, delivery accounting,
  GPU audit, budget и drain checks. First/warm случаи сохранены раздельно.

Первый контроль с **выключенной новой доставкой** дал211/212: повторилось
прежнее F32-расхождение, теперь в `f32/file_pipe_1_256/logits`.
Max_abs0,0005061785 и NMSE5,2051e−7 совпадают с прежней диагностикой;
все доставленные байты совпали. Повтор контроля прошёл212/212 без изменения
арифметики и численных порогов. Причина не установлена; этот результат
сохранён в validation JSON и не считается исправленным дефектом.

Измеренный executable `da13e049…` и его исходники сохранены в
`build-local/mimo2-tensor-measured`. После измерений изменён только default
переключателя, затем выполнена отдельная проверка запуска без env/CLI overrides:
EN/SWA249, invalid request, cancel→fresh и unload. Её результат, hashes и
проверенные logits записаны в shipping-разделе validation JSON.

Размер контекста выше512, длительное внешнее давление памяти и другие GPU
не проверялись. MTP batched-target parity остаётся открытым.96 kernel и132 graph
checks относятся к прежнему этапу, в этом этапе заново не запускались.

## Управление и воспроизведение

`STRATA_MIMO_PIPELINE_BATCH=1` — default отдельного MiMo pipe engine.
`0` возвращает прежнюю доставку. Переменную задавать до запуска; при
`--expert-readers 0` используется прежний синхронный путь. Default slab16 сохранён.
Setup/API profile пока отсутствует.

```powershell
$mimoModel = 'H:\models\mimo-v2.6-flash\MiMo-V2.6-Flash-RL-GSQ-RCO-3.5bit.gguf'
python -m tools.benchmark_mimo2_cache --build build-local/mimo2-cuda --model $mimoModel --cuda-bin build-local/cuda-13.0/bin/x64 --axis pipeline --fixed-slab-mib 16 --order 0,1,1,0 --output-dir build-local/mimo2-validation/tensor-abba-new
```

Каталог должен быть новым. Контрольные runners `check_mimo2_cuda` и
`check_mimo2_engine` явно выбирают `--pipeline-batch 0|1` (их default0).
Для текущего режима добавить `--cache-slab-mib 16 --pipeline-batch 1`.
Следующие отдельные опыты — частотный допуск и группировка cache fills с
собственными lifetimes; порядок admission и данные нельзя менять без проверки.
