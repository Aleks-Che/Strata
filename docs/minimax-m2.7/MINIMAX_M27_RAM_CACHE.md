# MM27-17: RAM LRU и разделение с GPU-кешем

Дата: **2026-10-08**, Windows, RTX 5090 32 ГиБ, RAM 125,555 ГиБ.
[План](MINIMAX_M27_IMPLEMENTATION_PLAN.md),
[статус](MINIMAX_M27_IMPLEMENTATION_STATUS.md).

Реализован opt-in `--ram-cache-mib N`, default **0**. Это LRU read-only
отображений участков GGUF, подключённый к существующим file pipeline readers.
В heap не создаётся вторая копия routed weights; большой RAM-кеш не закрепляется
через VirtualLock/cudaHostAlloc. Pinned ring остаётся прежнего размера.
Когда копия матрицы в GPU-кеше завершена, перекрывающиеся RAM views удаляются.
При последующем GPU miss файл можно отобразить снова.

**Результат трёх коротких A/B пар:** final RAM64 даёт **3,485 против3,941
токена/с** без RAM-кеша (−11,57% aggregate forward decode). Медиана полного
времени трёх запросов **89,556 против85,944 с (+4,20%)**. Поэтому RAM-кеш
остаётся opt-in/off; его наличие не означает утверждённого быстрого профиля.

## Реализация

`backends/minimax_m2/host_cache.hpp` хранит ключ `(Source identity, offset, length)`
и read-only view с LRU-позицией. Source удерживает собственный файловый handle;
живые views удерживают Source. Bind/unbind, context replacement, abort и release
очищают кеш, поэтому новая модель не наследует записи предыдущей.

Чтение из view завершается в pinned staging до публикации H2D job. Каждый
reader удерживает view через shared ownership; trim и GPU invalidation не
освобождают адрес во время memcpy. Удалённая запись помечается retired и
перестаёт давать hits; фактический unmap ждёт последнего reader. GPU invalidation
вызывается после copy fence и поддерживает matrix и grouped cache, batched
и synchronous D2D. Никакой D2H-копии при GPU eviction не добавлено.

Prefill может использовать готовые views, но не меняет их LRU-приоритет и
не заполняет кеш новыми. Decode admission начинается после первого serial
шага, как у GPU-кеша. При включённом GPU-кеше новый RAM view разрешается только
после зарегистрированного отказа GPU admission на предыдущем обращении к
этому диапазону. Это устраняет создание временных RAM views для большинства
успешных GPU fills. История ограничена65536 диапазонами, хранит weak Source
identity и очищается при GPU admission или смене модели. Без GPU-кеша
дополнительное условие не требуется. При отказе RAM admission используется прежний ReadFile.
Ошибки чтения передаются наружу; обработчик in-page I/O exception переводит
ошибку mmap в обычную runtime error, после чего конвейер завершается.

Лимит учитывает allocation-granularity prefix и округление до страниц, а не
только полезные bytes. Начальный cap ограничен свободной физической RAM с
запасом 6% общего объёма +256 МиБ. Давление может уменьшать бюджет и возвращать
его в пределах этого cap; вытеснение страниц Windows не увеличивает cap.
Меньше256 МиБ свободного commit запрещает новые отображения и очищает кеш.
Это проверка запаса под metadata, не резервирование heap под всю модель.
Дополнительные границы: максимум65536 entries и16 МиБ на читаемый chunk.

Process working-set target94% действует и для нового RAM-кеша, обновляется
при работе runtime и восстанавливается на release. Глобальные RAM/VRAM guards
95%, запас под GPU allocations и commit не изменены. После освобождения views
память измеряется заново перед окончательной проверкой лимита.

`MapViewOfFile(FILE_MAP_READ)` создаёт read-only view; unmap удаляет страницы
view из working set этого процесса. Он не гарантирует удаления тех же file
pages из других mappings или системного кеша Windows. Поэтому здесь разделены
управляемые записи RAM/GPU, а не доказано полное отсутствие физических
дубликатов во всей ОС. См. [MapViewOfFile](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-mapviewoffile)
и [UnmapViewOfFile](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-unmapviewoffile).

## Telemetry и границы проверки

- `ram_cache_bytes/budget/cap/peak_bytes` — charged views, не физическая
  резидентность всех страниц. `ram_cache_entries/readers` — число entries
  и выполняющихся mapped reads.
- `ram_cache_hits` — повторное использование существующего view. Windows
  может вытеснить его страницы; hit не означает гарантированное чтение из RAM.
- `ram_cache_mapped_bytes/file_bytes` — успешные copies через mapped source
  и ReadFile fallback. Admissions, evictions, GPU drops и dropped charged bytes,
  prefill bypasses/rejections записываются отдельно.
- `ram_cache_gpu_waits/history_entries` — отложенные admission и объём metadata
  об отказах GPU. CPU-cache hits не изменяют GPU policy.
- Старый `file_bytes` продолжает считать доставку через file pipeline, включая
  managed source reader. Ни он, ни ReadFile fallback bytes не являются
  измерением физического трафика SSD.

Окончательный вариант прошёл **816 cases**:35 CPU file-view tests,228 matrix pipeline,
120 grouped pipeline,180 synchronous D2D,216 RAM-off regression,
18 fixture lifecycle и19 full-model pressure/lifecycle. Bytes/logits gates
не ослаблялись. Есть отдельная проверка RAM hits при отказе GPU admission,
concurrent reads/drop/clear, LRU replacement, offset/guard/EOF, разной Source
identity, physical/commit pressure, cancellation, unload/reload и восстановления.

Контекст4096 проверен на небольшой fixture с4079 input tokens. Full-model
lifecycle использует context512,31 input tokens и batch8; это не новая полная
проверка длинного контекста. В pressure test создаётся отдельная read-only
нагрузка RAM и до12 ГиБ GPU. Короткая повторяющаяся последовательность может
полностью переходить в GPU-кеш: этот тест не доказывает работу заполненного
64-ГиБ RAM-кеша под давлением.

Ранние `ram-fixture-01` и `ram-sync-fixture-01` содержат неудачное требование
RAM hits при cache churn. Маленькие матрицы могут каждый раз приниматься в
GPU и сразу удаляться из RAM; это корректно. Ожидание заменено отдельным
тестом гарантированного GPU admission bypass. Все numerical gates в этих
ранних запусках прошли. Итоговые отчёты используют исправленный checker.

## Измерение скорости

Target: `H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf`,
138342384352 байта. Header SHA-256
`9a011b2ad6fb20db3bb0cdfe9af08665939ee1f0cb0609ffe2bfee7f31c825db`;
полный checksum target не вычислялся. Dependency `86ebfef2`, CUDA13.0.48,
MSVC19.44.35222.0, architecture120/120a, driver581.80.

GPU cache18 ГиБ, arena64/reserve0, file2/chunk4/lookahead/D2D batch,
context2048/batch16, strict F32 activations/KV, FA/graphs/grouping/MTP/DFlash
off, `STRATA_MM27_TOKENWISE=0`. Корпус стандартного tuning driver: первый
английский запрос, его повтор и русский запрос со списком1..96;
52/52/343 input tokens и по32 output tokens. KV очищается между запросами;
GPU/RAM caches остаются. Каждый вариант начинается в новом процессе.
OS file cache и внешняя нагрузка не контролировались; GPU тесты и сборки
не запускались параллельно. Измерения памяти дискретные.

В первом варианте RAM view создавался на каждом decode miss. Четыре
последовательных screen запуска дали:

| Режим | Decode первого / повтора / новой темы, токена/с | Полное время, с |
|---|---|---:|
| Off до | 3,415 / 4,669 / 3,906 | 87,643 |
| Eager RAM64 | 1,661 / 3,806 / 2,773 | 101,040 |
| Eager RAM84 | 1,977 / 3,800 / 2,787 | 97,799 |
| Off после | 3,580 / 4,643 / 3,884 | 86,062 |

В первом RAM64 запросе создано19410 views, из них13137 сразу удалены по GPU
admission; на конце корпуса оставалось36,155 ГиБ views. Лимит64 ГиБ ещё не
достигнут, поэтому повышение до84 не решало проблему. Все screen token IDs
и logits совпали с сохранённым старым движком; screen не меняет defaults.

Исправлена политика: RAM admission ждёт ранее наблюдённого GPU bypass.
В первом запросе число созданий снизилось до3216, GPU drops — до1755;
первый замер decode стал3,179 токена/с. Это сократило затраты первоначального
прототипа, но не дало выигрыша над отключённым RAM-кешем. После изменения
повторены unit, pipeline/regression и lifecycle/pressure проверки.

Три final пары шли в порядке off/RAM64, RAM64/off, off/RAM64:

| Режим | Агрегаты forward decode, токена/с | Медиана | Полные времена трёх запросов, с |
|---|---|---:|---|
| Off | 3,923 / 3,982 / 3,941 | 3,941 | 87,714 / 85,581 / 85,944 |
| RAM64 | 3,485 / 3,534 / 3,485 | 3,485 | 89,556 / 89,283 / 89,760 |

Медианы по запросам: **3,445/4,649/3,899 → 3,185/3,958/3,408 токена/с**.
Агрегат = сумма `decode_forward_tokens` / сумма `decode_ms`; первая выдача
учитывается в prefill. Decode interval исключает sampling, запись logits и
внешние memory samples; `request_ms` включает эти расходы и prefill, исключая
загрузку модели. С useful throughput DFlash эти абсолютные скорости напрямую
не сопоставимы. Это короткий корпус, а не утверждение о любых длинных сессиях.

RAM64 на границах запросов удерживал **3,948/11,698/20,876 ГиБ** в
1461/4349/7773 views; history6273/7619/13439, active readers0. Глобальные
sampled peaks RAM/VRAM: off **27,75/85,37%**, RAM64 **43,08/85,37%**.
Все18 ответов,576 IDs и115236864 logits совпали со старым EXE побитно.

Отдельный финальный pressure test затронул84,5 ГиБ file-backed страниц и
держал12 ГиБ временной GPU нагрузки. В снимке занято **89,92% RAM /94,10% VRAM**;
GPU backing уменьшился **15,875→9,5625 ГиБ**, slots **15,272→9,308 ГиБ**.
Logits под нагрузкой и после восстановления совпали побитно. RAM view LRU
отдельно проверен на ограниченном cap и инъецированном physical/commit pressure.

### Длинная проверка

Один дополнительный RAM64 workload с теми же52/52/343 input tokens и
**256 output tokens на запрос** прошёл все gates. Все768 IDs и153649152 logits
побитно совпали с сохранённым MM27-12 long reference. Итоговые RAM views:
**39,445 /49,454 /41,692 ГиБ**, peak **55,859 ГиБ**. Лимит64 ГиБ не достигнут;
зафиксированные evictions в этом прогоне вызваны GPU admission. LRU на
заполненном RAM cap проверен в отдельном CPU unit, а не этим full-model corpus.

Global sampled RAM/VRAM peaks **69,88/85,52%**. Decode3,635/3,689/4,808 токена/с,
полное время261,399 с. Парного текущего RAM-off long timing здесь нет — по
этому запуску нельзя утверждать ускорение длинных ответов. Все три ответа
закончились по token limit; natural EOS и законченный reasoning/final ещё TODO.
Совокупно final implementation сравнен по **21 ответу /1344 IDs /268886016 logits**.

## Артефакты

- [RAM unit](MINIMAX_M27_RAM_UNIT_CHECK.json),
  [matrix pipeline](MINIMAX_M27_RAM_FIXTURE_CHECK.json),
  [grouped pipeline](MINIMAX_M27_RAM_GROUP_FIXTURE_CHECK.json),
  [synchronous D2D](MINIMAX_M27_RAM_SYNC_FIXTURE_CHECK.json),
  [RAM-off regression](MINIMAX_M27_RAM_OFF_REGRESSION_CHECK.json).
- [Lifecycle/pressure](MINIMAX_M27_RAM_LIFECYCLE_CHECK.json),
  [первоначальный screen](MINIMAX_M27_RAM_SCREEN_CHECK.json),
  [final три пары](MINIMAX_M27_RAM_CONFIRM_CHECK.json),
  [long256](MINIMAX_M27_RAM_LONG_CHECK.json).
- [Build manifest](MINIMAX_M27_RAM_BUILD_MANIFEST.json),
  [проверки provenance, SHA-256 исходников/EXE/отчётов](MINIMAX_M27_RAM_VALIDATION_CHECK.json).

Новый bench SHA-256 `0f3b6216b5de9aae0c6cbc6940e31929e4238a5bd62e2830817d34be9a622846`.
Снимок до изменений: `build-local/minimax-m2-ram-baseline-01` (187 hashes
проверены). Final EXE дополнительно сохранены в `build-local/minimax-m2-ram-evidence-01`.
Raw logs/logits/requests/source snapshots: `build-local/minimax-m2-ram-screen-01`,
`minimax-m2-ram-confirm-01`, `minimax-m2-ram-long-01`, `minimax-m2-ram-lifecycle-02`.
Screen хранит прежнюю eager реализацию; final reports — отложенное admission.
Shared common/Step/Hy3/GLM файлы побитно совпали со снимком до этого этапа.

## Запуск

Существующий экспериментальный pipeline18/2/4 с RAM cap64 ГиБ:

```powershell
python tools/run_minimax_m2.py -- --gguf H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --request REQUEST.json --output NEW-REPORT.json --gpu-cache-mib 18432 --gpu-cache-allocator arena --pipeline-readers 2 --pipeline-chunk-mib 4 --pipeline-lookahead 1 --pipeline-d2d-batch 1 --ram-cache-mib 65536
```

Флаг требует mode2/file и включённого pipeline. Grouping включать отдельно,
если нужен эксперимент MM27-16. JSON tuning config принимает `ram_cache_mib`;
`tools/check_minimax_m2_lifecycle.py` принимает `--ram-cache-mib`.

Из настроенной CUDA/MSVC среды собрать и проверять последовательно, каждый
раз выбирая новые выходные пути:

```powershell
cmake --build build-local/minimax-m2-cuda --target strata-minimax-m2-bench strata-minimax-m2-host-cache-check strata-minimax-m2-cache-check strata-minimax-m2-lifecycle-check -j 6
python tools/run_minimax_m2.py --engine build-local/minimax-m2-cuda/bin/strata-minimax-m2-host-cache-check.exe -- build-local/mm27-new-ram-unit
python tools/run_minimax_m2.py --engine build-local/minimax-m2-cuda/bin/strata-minimax-m2-cache-check.exe -- build-local/mm27-new-ram-fixture --pipeline-readers 2 --pipeline-chunk-mib 4 --pipeline-lookahead 1 --pipeline-d2d-batch 1 --ram-cache-mib 64
python tools/check_minimax_m2_lifecycle.py --model H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --out build-local/mm27-new-ram-lifecycle --cases fixture full_pressure --gpu-cache-mib 18432 --gpu-cache-allocator arena --pipeline-readers 2 --pipeline-chunk-mib 4 --pipeline-lookahead 1 --pipeline-d2d-batch 1 --ram-cache-mib 65536
```

Следующие границы P3.4: policy и стоимость отображения RAM views;
длинные законченные ответы и sessions; фактическая page residency и
SSD traffic. P2 API, P4 session lifecycle, P6 рабочий профиль и DFlash live gates
не закрываются добавлением RAM LRU. Старое MM27-06 reload discrepancy остаётся OPEN.
