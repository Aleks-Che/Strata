# MIMO-15 — CUDA scatter-copy для резидентных экспертов

2026-10-07, Windows, Ryzen9 9950X, RTX5090 32GB, RAM128GB,
CUDA13.0 / MSVC19.44 / driver581.80. Модель:
`H:\models\mimo-v2.6-flash\MiMo-V2.6-Flash-RL-GSQ-RCO-3.5bit.gguf`.

**Добавлено собственное ядро копирования: `STRATA_MIMO_D2D_BATCH=2`, opt-in.**
Основной ABBA дал **9,567 → 10,259 ток/с (+7,23%)**; средний кандидат на5,48%
быстрее самого быстрого контрольного процесса. Выигрыш зависит от запроса.
Результат относится к этому PC, GGUF и коротким повторным запросам, context512.
Отдельный русский ABBA при cache8 ГиБ дал9,131→8,035 ток/с (−12,01%).
**Прежний default0 сохранён:** устойчивый выигрыш во всех проверенных условиях
не установлен, причина различия ещё открыта.

Данные: [скорость](MIMO26_FLASH_SCATTER_COPY_BENCHMARK.json),
[корректность и SHA](MIMO26_FLASH_SCATTER_COPY_VALIDATION.json),
[диагностические traces](MIMO26_FLASH_SCATTER_COPY_TRACE.json).

## Изменение

Вместо отдельных cache/guard → scratch копий один CUDA launch обрабатывает до32
диапазонов. Данные делятся на tiles16 КиБ; один блок256 потоков копирует один tile.
Выровненные адреса используют16-байтовые операции, остатки и невыровненные адреса
копируются побайтно. Небольшой512-байтовый guard занимает один блок.
Flat grid и префиксы tiles не создают пустую прямоугольную сетку для коротких tails.

Дескрипторы передаются значением в kernel arguments. Дополнительные device
allocations и pinned metadata buffers не нужны. Полный список проверяется до
первого launch; больше32 диапазонов делятся на несколько пакетов. Пустые
диапазоны пропускаются, включая последний элемент незавершённого пакета.

Источники cache защищены route pins, guard buffer живёт до завершения плана;
назначения не пересекаются. Ранний запуск сохраняет порядок из MIMO-14:
готовые resident weights отправляются до ожидания misses. Ring delivery,
его события возврата producer, scratch fence и admission/fills сохраняют прежние
lifetimes. Ошибка после enqueue и отмена завершают stream до освобождения pins.
Матричные вычисления остаются на GPU.

Mode0 — прежние отдельные копии, mode1 — CUDA13 `cudaMemcpyBatchAsync`, который
в MIMO-14 не дал ускорения. Mode2 не использует этот API. Проверена только
сборка CUDA13 на этом PC; это не проверка других GPU/toolkits.
`pipeline_d2d_kernel_launches` считает фактические CUDA launches отдельно от
логических `pipeline_d2d_batches` и числа диапазонов.

## Копирование отдельно

88 byte/canary cases проверяют выравнивание, нулевые диапазоны, границы16 байт
и16/64 КиБ, пакеты0/1/31/32/33/65/129 элементов, дырки в destination и сохранность
source. Production strides2752512/3604480/4456448 байт и512-байтовые guards
проверены для tiles16/64/256 КиБ.

Микрозамер:8 матриц +8 guards, четыре разные пары банков,256 итераций на вариант,
одна stream fence на тензор. Порядок: memcpy,16,64,256,256,64,16,memcpy.
Среднее wall time на тензор:

| Путь | Время |
|---|---:|
| Отдельные CUDA copies | 118,027 мкс |
| Scatter, tile16 КиБ | 63,163 мкс |
| Scatter, tile64 КиБ | 70,791 мкс |
| Scatter, tile256 КиБ | 74,237 мкс |

Tile16 выбран для runtime. Экономия46,5% относится только к этому microbenchmark;
в inference остаются чтение RAM, H2D, ожидания, вычисления и другие копии.

## Полная модель без трассировки

Context512, batch8, F32 KV, FA on, greedy, без MTP, CUDA Graphs и TF32.
Mmap, reader1, chunk8, slab16, decay65536, tensor delivery и packed guards on;
prefill admission, early refill и grouped fills off. На тему два исключённых
прогрева и три измеряемых ответа по32 токена. KV свежий для каждого запроса,
expert cache сохраняется. Decode:31 шаг на ответ; load/prefill/первый выходной
токен исключены, sampling и экспорт logits входят во время.

Основная серия: счёт, код, русский текст; cache request11264 МиБ.

| Проход | Путь | Decode |
|---|---|---:|
| A1 | Обычные копии | 9,726 ток/с |
| B1 | Scatter | 9,950 ток/с |
| B2 | Scatter | 10,588 ток/с |
| A2 | Обычные копии | 9,414 ток/с |

Сводная скорость считается как сумма шагов, делённая на сумму generation time.
Средний полный запрос6283,61 → 5961,72 мс; это не cold TTFT.
За18 timed requests с их prefill новый режим объединил1025522 диапазона
в88204 логических пакета /88260 CUDA launches.

В первом контроле live clamp немного менял cache: минимальный cap11203,22 МиБ,
средний физический размер11250,89 МиБ. Это не полностью изолированный ABBA.
В двух кандидатах и последнем контроле cap/allocations11264 МиБ; все сравниваемые
cache/traffic counters совпали между этими тремя процессами, включая каждый
warmup. Каждый из них передал352729085440 байт decode H2D за279 timed steps.
Положительный результат двух кандидатов сохраняется относительно обоих контролей.

| Тема, объединённые проходы | Обычные копии | Scatter |
|---|---:|---:|
| Счёт | 13,144 | 14,369 ток/с |
| Код | 7,452 | 8,747 ток/с |
| Русский текст | 9,682 | 9,216 ток/с |

Русский текст в B1 дал8,471, в B2 —10,105 ток/с. Поэтому общий выигрыш нельзя
переносить на каждый запрос; выполнены отдельные русские серии, описанные ниже.
Основной ABBA: sampled peak global VRAM29,506 ГиБ /92,66%, RAM116,876 ГиБ /93,09%.
Это секундные глобальные выборки с другими приложениями, не измерение пиков
между выборками. Рабочий cache request остаётся14336 МиБ с live clamp95%;11 ГиБ
использовались только для сравнительного эксперимента.

## Отдельные русские серии и решение о default

Тот же русский prompt, но теперь с пустого expert cache, без предшествующих
тем count/code. На процесс два прогрева и три timed ответа по32 токена.

| Cache request | A1 | B1 | B2 | A2 | Объединённые A → B |
|---|---:|---:|---:|---:|---:|
| 11264 МиБ, меняющийся live clamp | 7,465 | 9,479 | 8,718 | 7,721 | 7,591 → 9,082 ток/с |
| 8192 МиБ, cap и allocations постоянны | 8,859 | 9,006 | 7,253 | 9,421 | 9,131 → 8,035 ток/с |

Первая серия не изолирует D2D: средние physical cache9859,67/10076,00 МиБ и H2D
различаются. Во второй **все четыре процесса имеют точные cache/traffic counters**,
включая warmups; каждый передал130984413696 байт decode H2D за93 timed шага.
Scatter дал−12,01% в среднем, хотя B1 был быстрее A1. Request time7437,90→8546,42 мс.
Peak global VRAM/RAM: первая92,66%/71,61%, вторая87,36%/70,12%.
Cache8 ГиБ нужен здесь для устранения влияния live clamp, а не как рабочий профиль.

В медленном B2 выросли средние source time5,657→6,850с и consumer wait3,820→4,567с
по отношению к B1, включая prefill. Системные disk counters ненулевые и включают
другие процессы; приписывать изменение только новому CUDA kernel нельзя.
Причина разброса не доказана. Однако одного положительного ABBA недостаточно
для смены default при наличии сопоставимой отрицательной серии.
Mode2 остаётся доступным для дальнейшей локализации, default0 сохранён.

## Корректность и финальная сборка

- Три ABBA:100 ответов,3200 output token IDs и488243200 F32 logits совпали
  побитно с контролем каждой серии. Это parity transport на том же quantized graph.
- Явный mode2:6 full-model corpus cases,3509248 logits exact с сохранённым
  `frequency-corpus`: EN, длинный249-token prompt через SWA128, RU, ZH, code, numbers.
  Ошибка/cancel → свежий запрос и unload прошли.
- Сохранённый default проверен отдельно на тех же6 случаях:3509248 logits exact.
  Переменные D2D/packed guards/early refill удалены из environment; INFO подтвердил
  соответственно0/1/0 и34 МиБ staging. Рабочий cache request14336 МиБ.
- Runtime224 mode0 /228 mode2 PASS с первой попытки. Проверены реальные GPU bytes,
  guards, logits, CPU-node rejection и4 дополнительных случая ошибки уже
  отправленного пакета / drain / recovery для F32 и mixed weights.
- CUDA copy88 cases, CPU7 CTest и Python30 tests PASS. Команды, exit codes,
  hashes и raw paths сохранены в validation JSON; отдельные JSON не заменяют логи.

Скомпилированные engine sources после ABBA не менялись; финальный engine имеет
тот же SHA256 `f56fc8f97d140e743e54d06fdd443b3016a2f1944a01b82e34988ed67532b771`.
Финальный snapshot `build-local/mimo2-scatter-default-measured` дополнительно
сохраняет benchmark CLI с `--fixed-d2d-batch` для будущих сравнений chunks.
Основной GGUF и веса draft не менялись.

## Трассировка и ограничения

После основной серии сняты два code traces: два прогрева по32 выхода,
skip74 графа, затем5 decode graphs. В отдельном JSON сохранены сводки и hashes
исходных traces. При фоновой нагрузке cache менялся: контроль10,49–10,64 ГиБ,
кандидат9,08–9,64 ГиБ по концам запросов; decode H2D также различался.
Эта пара **не используется как доказательство ускорения**. Event intervals
содержат driver/CPU dispatch и накладные расходы instrumentation.

Существенный H2D/compute overlap по-прежнему не подтверждён. Прежние F32 и mixed
synthetic diagnostics не локализованы; успешные повторы не означают исправления.
Пороги численных проверок не изменены. Результаты не проверяют MTP/DFlash,
большие контексты, API или независимое качество исходного checkpoint.

## Воспроизведение

Сборка: `build-local/build-mimo2-scatter.cmd`, CUDA toolkit13.0 и MSVC x64.
Измеренная сборка/исходники сохранены в `build-local/mimo2-scatter-measured`;
engine SHA256 `f56fc8f97d140e743e54d06fdd443b3016a2f1944a01b82e34988ed67532b771`.
Финальные hashes, команды и snapshots перечислены в validation JSON.

```powershell
python -m tools.benchmark_mimo2_cache --build build-local/mimo2-cuda --cuda-bin build-local/cuda-13.0/bin/x64 --model H:/models/mimo-v2.6-flash/MiMo-V2.6-Flash-RL-GSQ-RCO-3.5bit.gguf --axis d2d --fixed-pack-guards 1 --order 0,2,2,0 --expert-cache-mib 11264 --warmups 2 --output-dir build-local/mimo2-validation/scatter-new
```

Для дальнейших chunk A/B сохранять `--fixed-pack-guards 1` и одинаковый D2D mode.
Флаг benchmark `--fixed-d2d-batch 2` позволяет отдельно повторить их со scatter-copy.
Следующий MIMO-16: chunks4/16, групповые H2D или чтение resident weights без
промежуточного scratch; каждый вариант требует собственной parity и возврата
к контролю. Непроверенные синхронизации убирать нельзя: scratch используется
также живыми активациями. API и speculative target parity остаются отдельными задачами.
