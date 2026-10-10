# MM27-37 — асинхронная загрузка таблицы перестановки

Дата: 2026-10-10. Продолжение [MM27-36](MINIMAX_M27_ROUTER_IDS_CONTEXT.md).
**MM27-37 завершён как эксперимент: correctness PASS, устойчивый speedup
не подтверждён.** Native режим остаётся выключенным; server whitelist прежний.

## Изменение

В strict F32 пути Q4_K/Q6_K таблица перестановки создавалась в локальном
`std::vector`, копировалась на GPU, затем CPU вызывал `cudaStreamSynchronize`.
Ожидание обеспечивало время жизни исходных данных. Предыдущий MM27-35 убрал
повторное чтение ID экспертов с GPU, но оставил эту синхронизацию.

Новый native флаг `--sort-table-async 1` помещает таблицу в собственный кольцевой
буфер: 4 слота по 4096 байт, всего 16 КиБ pinned RAM на поток движка. CUDA event
после H2D защищает слот до завершения копирования. Повторное использование
проверяет event и ждёт только незавершённую копию. H2D и читающие таблицу kernels
идут в одном compute stream. Изменение следует правилам владения pinned memory
и синхронизации events из [документации NVIDIA](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html).

Размер ограничен независимо от числа запросов. Таблицы больше 4 КиБ используют
старый путь. При batch 16 и top-8 таблица занимает 1024 байта, при decode — 64.
Исходный tensor ID может иметь промежутки между строками; его `ggml_nbytes`
не равен размеру плотно упакованной таблицы. Арифметика, порядок экспертов,
device pool, доставка весов и CUDA graph policy не изменены.

Graph-exit fence сохранён. После него ring снимает отметки занятых слотов;
ранний выход, ошибка и отмена тоже сохраняют источники до завершения копий.
Смена режима, reset и release дожидаются источников. Если enqueue успешен,
но запись event завершилась ошибкой, очистка использует исходный stream.
Изменение касается только private generated MiniMax dispatch, закреплённый
upstream и shared transport не редактируются.

Режим по умолчанию выключен. Кандидат не добавлен в whitelist Python-сервера:
его отдельные long-context/session/real-pressure проверки ещё не выполнены.

## Сборка и малые проверки

Кандидат: `build-local/minimax-m2-sort-table-candidate/engine.exe`, SHA-256:
`692f2e76d97192c3357d209d2013d92b63735607e6c4eeaeac42b31c71a839ff`.

Dependency: `86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9`; CUDA 13.0,
MSVC 19.44.35222.0, Windows, RTX 5090 32607 МиБ, Ryzen 9 9950X,
RAM 125,555 ГиБ. Основной EXE `98e85e80…`, events EXE `13a885cc…` и
router EXE `612354a1…` сохранены.

Сборка: `build-local/build-minimax-m2-sort-table.bat`; логи
`build-local/minimax-m2-sort-table-configure.log` и `…sort-table-build.log`.
Перед инкрементальной сборкой основной bench сохранён; после сборки кандидат
скопирован в отдельный каталог, основной bench восстановлен.

- 106 native условий PASS: две CUDA streams, 48 копий с перезаписью исходного
  буфера, принудительно занятый слот, ограничение RAM, oversize fallback,
  очистка после исключения, close/reopen; полный F32 parity на маленькой MoE
  при batch 1/8/16 и events 0/2, три истории запросов на каждую пару.
- 236 cache/fault/reload условий PASS с events 2, router IDs и новым ring.
  Проверены сбои чтения, доставки и compute, отмена, инъекция давления памяти,
  восстановление и освобождение pinned storage.
- 17 Python методов PASS для проверки table/router/event evidence.
- 4 неверных сочетания/значения CLI отклонены до загрузки весов.

Первый fixture run сохранил точные logits, но отклонил неверное утверждение
`table_bytes == 2 * router_ids_bytes` при batch 8/16. Проверка исправлена на
точный размер упакованной таблицы; код движка после этого не менялся.
Финальные артефакты: `build-local/minimax-m2-sort-table-fixtures-02`,
`…sort-table-cache-01`, `…sort-table-unit.log`, `…sort-table-cli.json`.

Пробный полный A/B `…sort-table-ab-01` остановился на второй неточной формуле
счётчика: последний MoE-блок при prefill сохраняет только одну output-строку
на microbatch, в отличие от 61 предыдущего блока. Для этой модели ожидаемый
размер равен `64 * (183 * evaluated_prompt_tokens + pipeline_matrices / 62)`
в prefill и `11904 * (generated_tokens - 1)` в decode. Все 102432768 logits
кандидата побитно совпали с reference при отдельной повторной проверке
(`corrected-evidence.json`). Эта пара дала 4,370 → 4,530 ток/с, но в итоговую
серию не включается. Итоговый A/B выполнен в новом каталоге `…sort-table-ab-02`.

## Полная модель

Target: `H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf`,
138342384352 байта. Corpus A,A,B,A по 128 выходных токенов; контекст 2048,
batch 16, fresh KV, GPU cache 18 ГиБ/arena 64 МиБ/reserve 0/decay 65536,
readers 2/chunk 4/lookahead/D2D/events 2, router IDs включены в обоих режимах.
RAM expert cache, prefix/archive, MTP и DFlash выключены. Полные F32 logits
сравниваются с сохранённым MM27-32 reference, полученным старым EXE.

```powershell
python -X utf8 tools/check_minimax_m2_sort_table.py --out build-local/minimax-m2-sort-table-ab-NEW --repeats 3
python -X utf8 tools/check_minimax_m2_sort_table_batches.py --out build-local/minimax-m2-sort-table-batches-NEW
```

GPU-прогоны последовательные, каждый процесс ограничен Windows Job и timeout.
Независимый Windows/NVML observer останавливает прогон при RAM/VRAM выше 95%.
OS file cache не контролируется. Короткий A/B не доказывает скорость и
надёжность для длинного контекста, сессий, иных quant/backend или DFlash.

## Результат трёх пар

| Пара | Ожидание после H2D, ток/с | Pinned ring, ток/с | Разница |
|---|---:|---:|---:|
| 1: off → on | 4,465481 | 4,485353 | +0,4450% |
| 2: on → off | 4,459970 | 4,452079 | −0,1769% |
| 3: off → on | 4,475881 | 4,482868 | +0,1561% |
| Медианы режимов | 4,465481 | 4,482868 | +0,3894% |

Сумма времени четырёх запросов, медианы: 151,852 → 151,189 с (−0,4371%).
24 запроса, 3072 токена и 614596608 F32 logits совпали с reference побитно;
все значения конечны. Decode H2D 984,084985 ГиБ, hit bytes 50,967889%,
40711 вытеснений — одинаково во всех режимах и повторах.

Каждый on-корпус: 97464 table uploads / 8626944 байта (8,227 МиБ),
0 reuse waits, 0 drain waits, 0 fallback. В конце каждой фазы pending = 0;
используется 16 КиБ pinned RAM. Независимые 1820 samples в A/B:
RAM ≤22,259%, VRAM ≤84,966%; нарушения 95% и ошибки observer отсутствуют.

Малый прирост и разнонаправленные пары не подтверждают устойчивое ускорение.
Режим остаётся выключенным по умолчанию и не включается в server whitelist.
Большой long/session/pressure прогон для него откладывается до подтверждения
пользы, а проверенный router-ID профиль MM27-36 сохраняется.

CPU-таймер `compute_ms` при events 2 включает отправку операций и ожидания,
а не время GPU kernels. Его сокращение не равно ускорению вычислений: полное
время decode почти не изменилось, а ожидания доставки выросли. Это согласуется
с переносом ожидания на другие границы конвейера; отдельного GPU timeline
в этом эксперименте нет. Суммировать перекрывающиеся таймеры нельзя.
Медиана decode CPU submission: 28,667 → 14,215 с; delivery: 56,608 → 60,694 с.

## Batch 1/8 и системные чтения

Две полные короткие подсказки из скоростного корпуса, по 8 выходных токенов,
ctx 2048. Для каждого batch сравнивается off/on: ещё 32 ID и 6402048 F32
logits в двух парах совпали побитно, решения кеша и передачи экспертов прежние.
Это проверка точности при том же batch, не новый скоростной A/B и не 2K/4K
prefix/archive validation.

В этих четырёх запусках добавлены начальный/конечный raw PDH counters из
существующего `tools/windows_memory_counters.py`. Счётчики охватывают весь ПК
и все физические диски, включая загрузку модели и другие процессы.

| Запуск | Source/H2D движка, ГиБ | Физические чтения всего ПК, байт | Pages output |
|---|---:|---:|---:|
| batch 1, off | 418,371 | 10256163328 | 0 |
| batch 1, on | 418,371 | 81459200 | 0 |
| batch 8, off | 286,107 | 41189376 | 0 |
| batch 8, on | 286,107 | 40076288 | 0 |

После первого окна физические чтения составляют лишь 38,22–77,69 МиБ.
Это указывает на работу из файлового кеша Windows в этих коротких прогретых
запусках. ReadFile bytes нельзя называть физическим трафиком SSD. Здесь нет
привязки к диску H:, отдельных prefill/decode окон или исключения чужих I/O;
вывод нельзя переносить на холодный запуск, длинный контекст и другой corpus.

## Итоговая проверка и продолжение

`tools/validate_minimax_m2_sort_table.py` повторно проверил полные logits,
transport/table counters, source/generated/build/artifact SHA, limits и
завершение десяти native процессов. **903 gates / 178 hashes PASS**.
Machine-readable evidence: [MINIMAX_M27_SORT_TABLE_CHECK.json](MINIMAX_M27_SORT_TABLE_CHECK.json).
В сумме A/B и batch-проверок 2100 независимых memory samples:
RAM ≤22,350%, VRAM ≤84,966%. Инъекции нехватки памяти проверялись в fixtures;
реальный long/session/94% corpus на этом EXE не запускался.

```powershell
python -X utf8 tools/validate_minimax_m2_sort_table.py
```

Валидатор по умолчанию читает зафиксированные `…sort-table-ab-02` и
`…sort-table-batches-01`; для нового evidence его пути нужно выбирать явно
или обновить после нового прогона. Старые evidence не перезаписываются.

Следующая задача P3 — ограниченное профилирование source → pinned staging →
H2D → ring/D2D на скоростном A,A,B,A с проверенным router EXE. Нужны раздельные
prefill/decode окна, системные физические I/O counters и CUDA timeline,
чтобы выбрать изменение доставки экспертов по измерениям.
