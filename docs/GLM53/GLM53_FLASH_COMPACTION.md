# GLM-5.3-Flash: чтение весов и фрагментация GPU cache

Проверка 2026-10-05, Windows / Ryzen 9 9950X / RTX 5090 32 ГиБ / RAM 128 ГиБ.
Модель — `GLM-5.3-Flash-Uncensored-IQ3_XXS.gguf`, 112,31 ГиБ.
MTP off, ctx2048/batch16/threads4, RAM/VRAM targets95, slabs16, events2,
main decay131072, chunk4 МиБ. Архив Unsloth и арифметика модели не менялись.

## Что изменено

`STRATA_GLM_EXPERT_READ_MODE=0|1|2` подключает уже существующий общий transport
к реальным mappings загрузчика GLM: mmap / native cached ReadFile / native для
холодного prefill. Auto сохраняет mmap в decode. На Windows регистрируются точные
offset-zero mappings и файловые дескрипторы; native mode отказывается запускаться,
если какой-либо expert tensor не покрыт регистрацией. Закрытие модели удаляет
регистрацию; queued reads удерживают собственный handle. Исходный архив не изменён,
генерируемая копия `llama-mmap.cpp` защищена SHA-256 исходника. Linux native mode
не реализован. `pipeline.file_bytes` и `mmap_bytes` показывают выбранный путь.

`STRATA_GLM_CACHE_COMPACT=1` уплотняет уже выделенные GPU slabs между запросами,
если в них есть хотя бы 64 МиБ свободных slots. Совместимые веса перемещаются
из разреженных блоков в заполненные; пустые блоки возвращаются CUDA. Перенос
использует D2D, не выделяет новую VRAM и не читает веса из RAM. Бюджет и NVML
guard остаются прежними. Это не изменение квантования или выбора экспертов.

Перед переносом проверяются все leases, plan pins, retired allocations и CUDA
events. Если хоть один владелец/consumer активен, операция пропускается. Сначала
проверяются все адреса и размеры и составляется план; затем завершаются D2D,
обновляются указатели кэша и освобождаются блоки. LRU, частоты и logical byte
accounting не изменяются. В лог добавлены moved/released bytes, число проходов,
пропуски и общее CPU wall time уплотнения.

Benchmark получил чередование двух сохранённых prompts и опциональные Windows
PDH counters: page reads, pages input/output, transition faults и disk read bytes.
Это разности накопительных **системных** счётчиков: они включают другие процессы.
Их нельзя называть hard faults только нашего engine или вычитать из process
page faults для получения soft faults. Учтён wrap 32-bit counters.
[Документация PDH](https://learn.microsoft.com/en-us/windows/win32/api/pdh/nf-pdh-pdhgetrawcountervalue).

## Измерения

В пилоте на prompt B, четыре warmup + три timed, native read дал **10,379 ток/с**,
mmap — **12,822 ток/с**. Медиана полного ответа — **9,053 / 7,294 с**.
Это разные последовательные процессы, файловый кэш ОС не очищался.
У native decode два активных reader, у mmap один; сравниваются готовые режимы,
а не только стоимость одного API. Native не выбран основным.

Пилот compaction на B, четыре warmup + пять timed: **12,748 ток/с**, полный
ответ **7,487 с**. Три прохода уплотнения вернули суммарно **578 МиБ** за **30,5 мс**.
Все девять ответов совпали с reference. Это улучшение размещения памяти, но
устойчивого ускорения относительно mmap пилота не получено.

Финальная пара на одном бинарнике: шесть warmup с чередованием A/B, затем по три
timed ответа на каждый prompt. A — 39 входных токенов, B — 49; выход по 64.

| Режим | A, ток/с | B, ток/с | A, полный ответ, с | B, полный ответ, с |
|---|---:|---:|---:|---:|
| Контроль, mmap / reader1 | 9,450 | 10,022 | 8,723 | 8,654 |
| Compaction, mmap / reader1 | 8,539 | 9,137 | 9,402 | 9,504 |
| Без compaction, mmap / reader2 | 7,535 | 2,487 | 12,314 | 35,133 |

В timed snapshots медиана незаполненных slots для A уменьшилась с **461 до
44 МиБ**, для B — с **461 до 97 МиБ**. Полезный main cache обычно вырос примерно
на 0,5 ГиБ, но H2D не снизился. В конце последнего запроса cache снова оказался
фрагментирован: перенос выполняется только между запросами. Общая скорость
этой пары ниже примерно на 9%; прямой таймер уплотнения — только 0–18 мс на
запрос. Эти данные не доказывают, что вся разница вызвана самим D2D переносом.

Числа относятся к данным prompts, их порядку и состоянию ОС. Значение 12,822
для повторяемого B — контроль без новой оптимизации; его нельзя выдавать за
прирост относительно прежних 9,9 ток/с на другом workload. В холодных запросах
наблюдались гигабайты дисковых чтений, в большинстве прогретых — значительно
меньше. Отдельного process-only hard-fault trace здесь нет.

Дополнительная серия reader2 использует тот же бинарник, тот же порядок A/B и
6+6 запросов. В timed run 7 скорость **1,621 ток/с**, суммарное время readers
**91,015 с**, системное чтение с дисков **49,81 МиБ**, process page faults
**1 151 083**, системные transition faults **1 810 251**. Working set находился
около hard maximum, примерно 104,5–104,8 ГиБ. Плохое время подготовки весов
нельзя объяснять только чтением десятков гигабайт с SSD: эта серия показывает
большое количество faults при малом дисковом I/O. Системные счётчики не дают
полной причинной атрибуции одному процессу или одному параметру reader count.

Windows может возвращать transition pages из RAM в working set без диска;
достижение maximum working set заставляет вытеснять другие страницы.
[Механизм working set в документации Microsoft](https://learn.microsoft.com/en-us/windows/win32/memory/working-set).
Следующее направление — проверить владение CPU-копиями GPU-resident весов и
стоимость hard working-set cap. Просто отключать cap или превышать targets95
не предлагается; repeated RAM scans из P3.3g уже дали отрицательный результат.

## Решение и проверки

Рабочий профиль сохранён: **MTP off, mmap, reader1, compaction off**, RAM/VRAM95.
Native read и compaction доступны как опциональные диагностические/экспериментальные
режимы, но устойчивого прироста скорости не показали. Обновлён backend patch
identity; режимы не включены в installer defaults.

Все **59 benchmark responses** совпали с сохранёнными references, процессы
завершились с exit0. **22 candidate CTest**, **13 transport CTest**, **83 Python
tests** прошли. Новый CUDA fixture проверяет побайтовую сохранность при переносе,
отсутствие фиктивных cache accesses/evictions, отказ при lease/pin/pending CUDA
или retired owner, отказ при неполном списке allocations и полное освобождение
после переноса. В старом tail-slab fixture исправлен захват `MiB` для MSVC.

На полной модели с compaction on прошли **9 native pipe cases**: две точные
генерации, ошибочный token ID, seeded sampling/repeat, cancel в prefill/decode
и восстановление после каждой отмены; exit0. [Отчёт](GLM53_FLASH_COMPACTION_PIPE.json).

Полные F32 logits, HTTP endpoints, длинный контекст, Linux/HIP и полные
Qwen/DeepSeek модели в этой работе повторно не проверялись. Арифметика графа не
менялась; предыдущая F32/HTTP проверка находится в P5.3b. Отдельные DeepSeek
compatibility fixtures входят в transport CTest.

Подробные времена, все token IDs, память, счётчики и hashes бинарников:
[benchmark](GLM53_FLASH_COMPACTION_BENCHMARK.json),
[manifest проверок](GLM53_FLASH_COMPACTION_VALIDATION.json).
