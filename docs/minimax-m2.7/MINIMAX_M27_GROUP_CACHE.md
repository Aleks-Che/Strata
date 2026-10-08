# MM27-16: группы экспертов gate/up/down

Измерено **2026-10-08** на Windows, RTX 5090 32 ГиБ и 125,555 ГиБ доступного
ОС общего объёма RAM. План и общий прогресс:
[план](MINIMAX_M27_IMPLEMENTATION_PLAN.md),
[статус](MINIMAX_M27_IMPLEMENTATION_STATUS.md).

Реализована optional группа `(generation, layer, expert)` из трёх матриц
gate/up/down. Она занимает один слот arena, целиком заменяется и целиком
защищается от вытеснения. **488 проверок прошли**, как и три пары измерений
на полной модели с побитным сравнением всех logits со старым движком.
Устойчивого ускорения не получено: медиана агрегата decode **3,912 → 3,930
токена/с (+0,47%)**, полное время трёх запросов **86,935 → 86,154 с**.
`--cache-group-experts` по умолчанию остаётся **0**.

## Реализация и границы

- `backends/minimax_m2/group_cache.hpp` адаптирует существующую политику кеша
  для MiniMax. Immutable tensor directory при привязке модели задаёт полные
  тройки одного слоя с одинаковым количеством экспертов. Типы, strides,
  file ranges и generation продолжают проверяться runtime.
- Каждый компонент имеет выровненное на 64 КиБ место и настоящие следующие
  байты tensor guard, до 512 байт. Последний эксперт не читает за конец тензора.
  Порядок tensor payloads в GGUF не меняется; веса не перезаписываются.
- Первое admission резервирует место сразу для трёх матриц. Ready/pending
  учитываются отдельно для каждой матрицы. `commit` публикует готовность
  только после завершения CUDA copy. Повторное использование адреса всегда
  сбрасывает readiness прежнего эксперта.
- Частота группы обновляется один раз за router plan. Отказ принять группу
  на первой матрице действует до конца этого плана; следующая матрица не
  может начать позднее admission неполной тройки. Следующий план может повторить
  попытку. Это устраняет обнаруженный при тестировании случай admission на DOWN.
- Уже находящиеся в кеше группы и новые fills защищены до завершения всего
  плана. Защита одной матрицы удерживает все три. Освобождение физической памяти
  по-прежнему выбирает целые неприкреплённые блоки arena. Cancel/fault/pressure
  сначала завершают копирование и работу очереди, затем снимают pins и чистят кеш.
- Требуются ненулевой GPU cache и allocator `arena`. Полная готовность троек
  проверена с file pipeline и трёхматричным router lookahead. Без такого плана
  API допускает частично заполненные группы; это не измеренный быстрый профиль.
- Transport ring, размеры read/H2D chunks и scheduler scratch не увеличены.
  Флаг меняет размещение в кеше, но не превращает три GGUF ranges в одно чтение.
  Global RAM/VRAM guards 95% и host commit reserve сохранены.

В telemetry добавлены `cache_expert_groups`, `cache_partial_expert_groups`,
`cache_ready_matrices`, `cache_pending_matrices`, `cache_ready_bytes`,
`cache_group_admissions`, `cache_group_plan_pins_peak` и признак режима.
`cache_hits/misses` по-прежнему считают обращения к матрицам. В grouped mode
`cache_evictions/reuses` и admissions считают группы; сравнивать их напрямую
с количеством вытесненных матриц нельзя. `cache_pressure_groups` означает
физические блоки, а не новые semantic groups.

Shared Step/Hy3/common файлы на этом этапе не менялись; их hashes сверены со
снимком до правок. В CMake добавлены явные зависимости generated backend от
включаемых `.inc`/headers: первоначальная incremental сборка не подхватила
исправление `group_cache.hpp`. Итоговые проверки выполнены после пересборки.

## Проверки GPU, отмены и давления памяти

| Проверка | PASS | Условия |
|---|---:|---|
| Group cache unit | 31 | Production Q4/Q6 strides; reuse, readiness, pins, давление, abort/recovery, generation и guards |
| Grouped engine/cache/pipeline | 114 | F32/mixed fixture, caps 2/256 МиБ, readers 2/chunk 4/lookahead/D2D batch |
| Matrix mode regression | 216 | Те же fixture и pipeline, cuda/arena, группировка выключена |
| Grouped synchronous D2D | 90 | Readers 2/chunk 4/lookahead, D2D batch выключен |
| Lifecycle fixture | 18 | Context 4096, input 4079, batch 16 |
| Full target lifecycle/pressure | 19 | Context 512, input 31, batch 8, реальное и инъецированное давление |

Итого **488**. Проверяются доставленные bytes, logits, prefill без заполнения
кеша, повтор и смена темы, malformed identities, read/copy/compute failures,
cancel, pending-copy faults, unload/reload и восстановление после ошибки.
Новая полная модель не тестировалась с почти заполненным контекстом 4K на этом
этапе: 4K здесь относится к небольшой fixture. Старые MM27-06 2K/4K результаты
сохраняются отдельно; причина прежнего tiny reload расхождения остаётся OPEN.

В production-size unit test два физических блока по 64 МиБ вместили:

| Кванты тройки | Полных групп | Занято слотами, МиБ | Готовые bytes с guard, МиБ | Готовые bytes / physical |
|---|---:|---:|---:|---:|
| Q4_K / Q4_K / Q4_K | 16 | 123,000 | 121,523 | 94,94% |
| Q4_K / Q4_K / Q6_K | 14 | 124,250 | 122,575 | 95,76% |

Это реальные CUDA allocations и чтение проверяемых bytes. Сравнения скорости
с матричным кешем эта таблица не содержит. Уменьшение бюджета освобождает целый
неприкреплённый блок; pin любой его группы препятствует освобождению блока.

В full-model pressure test затронуто 85 ГиБ read-only file-backed pages и
выделено 12 ГиБ временной GPU нагрузки. В снимке под нагрузкой глобально занято
**89,95% RAM / 94,07% VRAM**. Кеш уменьшил physical backing
**15,875 → 9,3125 ГиБ** и resident slots **15,272 → 9,029 ГиБ**;
logits после этого и после восстановления побитно совпали. Это отдельный
тест давления, не конфигурация измерения скорости.

## Три пары на полной модели

Target: `H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf`,
138 342 384 352 байта. Header SHA-256
`9a011b2ad6fb20db3bb0cdfe9af08665939ee1f0cb0609ffe2bfee7f31c825db`.
Полный target checksum не вычислялся. Dependency `86ebfef2`, CUDA 13.0.48,
MSVC 19.44.35222.0, driver 581.80, requested architecture 120/effective 120a.

Оба варианта используют один новый EXE: cap 18 ГиБ, arena block 64 МиБ,
growth reserve 0, file reader, 2 readers, chunk 4 МиБ, lookahead и D2D batch.
Context 2048/batch 16, F32 activations/KV, FA/graphs/native MTP/DFlash off;
`STRATA_MM27_TOKENWISE=0`. Для numerical reference предварительно запущен
сохранённый EXE до изменений с тем же корпусом и settings.

Три запроса: английский вопрос о цвете неба, его повтор и русский вопрос о
сумме списка 1..96. Встроенный template даёт **52/52/343 input tokens**;
каждый ответ ограничен 32 токенами, естественный EOS не достигнут. KV очищается
между запросами; модель и экспертный кеш остаются загруженными. Для каждого
варианта запускается новый процесс. Порядок: matrix/group, group/matrix,
matrix/group. Сборки и другие наши GPU тесты параллельно не запускались;
OS file cache и внешняя нагрузка не контролировались.

| Запрос | Matrix, median токена/с | Group, median токена/с | Изменение |
|---|---:|---:|---:|
| Первый | 3,460 | 3,460 | +0,01% |
| Повтор | 4,664 | 4,699 | +0,76% |
| Смена темы | 3,860 | 3,817 | −1,12% |

| Режим | Агрегаты decode в трёх прогонах, токена/с | Медиана | Полное время трёх запросов, с |
|---|---|---:|---|
| Matrix | 3,969 / 3,912 / 3,903 | 3,912 | 85,431 / 86,970 / 86,935 |
| Group | 3,931 / 3,930 / 3,868 | 3,930 | 86,129 / 86,154 / 87,395 |

Агрегат = сумма `decode_forward_tokens` / сумма `decode_ms`. Первый токен
относится к prefill. Decode timing в этом harness исключает sampling, запись
logits и внешнее измерение памяти; `request_ms` включает эти расходы и prefill.
**Эти абсолютные токены/с нельзя напрямую сравнивать с useful throughput
DFlash harness** из предыдущего этапа. Общая медиана request time без загрузки
модели уменьшилась на 0,90%, но диапазоны перекрываются и преимущество по
запросам неоднородно. Доказанного ускорения для defaults нет.

Hit bytes для первого/повтора/новой темы: matrix **46,09/59,37/45,65%**,
group **46,48/60,34/43,64%**. Соответствующий H2D на decode token:
**2,130/1,605/2,147 → 2,115/1,567/2,227 ГиБ**. Выигрыш попаданий на повторе
сменяется потерей при новой теме; atomic group replacement само по себе
не обеспечивает лучшую политику выбора экспертов.

Physical arena = 18 ГиБ в обоих вариантах. На границах decode grouped cache
содержит **2154 полные группы / 6462 готовые матрицы**, примерно 17,166 ГиБ
готовых bytes в 17,388 ГиБ charged slots. Partial/pending = 0; peak дополнительных
plan fill pins = 24 при проверяемой границе 768. Matrix charged slots = 17,433 ГиБ.
Снятые пиковые значения global RAM/VRAM: matrix **27,76/85,96%**,
group **27,66/87,34%**. Это дискретные измерения. 95% остаётся общим пределом,
а скорость измерена при одинаковом cap 18 ГиБ; этот этап не выбирает новый cap
и не реализует отдельный большой RAM cache.

Все шесть запусков прошли gates и сравнение со старым EXE: **18 ответов,
576 token IDs, 115 236 864 float logits побитно совпали**. Скорость reference
запуска не включена в три пары. Default grouping off и прежний экспериментальный
18/2/4 сохранены; DFlash здесь не включался и его gates не закрывались.

## Артефакты и воспроизведение

- [Unit](MINIMAX_M27_GROUP_UNIT_CHECK.json),
  [pipeline](MINIMAX_M27_GROUP_FIXTURE_CHECK.json),
  [matrix regression](MINIMAX_M27_GROUP_MATRIX_REGRESSION_CHECK.json),
  [synchronous D2D](MINIMAX_M27_GROUP_SYNC_FIXTURE_CHECK.json).
- [Lifecycle/pressure](MINIMAX_M27_GROUP_LIFECYCLE_CHECK.json),
  [полные три пары](MINIMAX_M27_GROUP_CONFIRM_CHECK.json),
  [old reference provenance](MINIMAX_M27_GROUP_REFERENCE_MANIFEST.json).
- [Build manifest](MINIMAX_M27_GROUP_BUILD_MANIFEST.json),
  [сводка, hashes исходников/EXE/отчётов](MINIMAX_M27_GROUP_VALIDATION_CHECK.json).

Старый EXE SHA-256 `d6c669ec216815f4c0499d62443cb5ee73f0fb072c443f6575ed6920425afbb5`,
новый `b3e92f8387e1ede7292be50f0c8a8537dfea10435e71524a5f5147b1847de317`.
Старые исходники и EXE: `build-local/minimax-m2-group-baseline-01` (124 hashes
проверены). Новые EXE дополнительно сохранены в `build-local/minimax-m2-group-evidence-01`.
Raw logits, logs, requests и снимки исходников находятся в
`build-local/minimax-m2-group-{reference,confirm,lifecycle}-01`.

Ранние неудачные отчёты `group-fixture-01/02/03` не удалены. Первый содержал
неверное ожидание двух pending fills при вместимости только одной F32 тройки;
второй выявил позднее admission; третий запускал stale backend после header
правки. Его EXE сохранён как `engine-before-dependency-fix.exe`. Итоговые
114/216/90 checks выполнены свежей сборкой; logit tolerances не ослаблялись.

Для воспроизведения выбрать новые пути вывода и выполнять GPU команды
последовательно. Из настроенной CUDA/MSVC среды собрать:

```powershell
cmake --build build-local/minimax-m2-cuda --target strata-minimax-m2-bench strata-minimax-m2-group-cache-check strata-minimax-m2-cache-check strata-minimax-m2-lifecycle-check -j 6
python tools/run_minimax_m2.py --engine build-local/minimax-m2-cuda/bin/strata-minimax-m2-group-cache-check.exe -- build-local/mm27-new-group-unit.json
python tools/run_minimax_m2.py --engine build-local/minimax-m2-cuda/bin/strata-minimax-m2-cache-check.exe -- build-local/mm27-new-group-fixture --pipeline-readers 2 --pipeline-chunk-mib 4 --pipeline-lookahead 1 --pipeline-d2d-batch 1 --cache-group-experts 1
python tools/check_minimax_m2_lifecycle.py --model H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --out build-local/mm27-new-group-lifecycle --cases fixture full_pressure --gpu-cache-mib 18432 --gpu-cache-allocator arena --pipeline-readers 2 --pipeline-chunk-mib 4 --pipeline-lookahead 1 --pipeline-d2d-batch 1 --cache-group-experts 1
```

В tuning JSON сравниваются две записи с одинаковыми остальными параметрами:

```json
[
  {"name":"matrix-18","cache_mib":18432,"readers":2,"chunk_mib":4,"group_experts":false},
  {"name":"group-18","cache_mib":18432,"readers":2,"chunk_mib":4,"group_experts":true}
]
```

```powershell
python tools/tune_minimax_m2_pipeline.py --model H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --configs build-local/minimax-m2-group-confirm-config.json --out build-local/mm27-new-group-ab --repeats 3 --tokens 32 --reference-logits build-local/minimax-m2-group-reference-01/logits.f32 --reference-report build-local/minimax-m2-group-reference-01/report.json
```

Далее для P3 открыты управление RAM working set с учётом GPU-resident experts
и проверка иной admission policy на длинных законченных ответах. P2 API,
P4 sessions, P6 профиль и DFlash live lifecycle остаются отдельными пунктами.
