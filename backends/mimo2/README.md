# MiMo-V2.6-Flash-RL: GPU engine, кэш экспертов и проверки

Доступны строгий inspector, регистрация тензоров без выделения памяти под веса,
tokenizer/Jinja oracles, числовые CUDA fixtures и синхронный pipe engine полного GGUF.
Проверены context512, F32 KV, FA on, greedy, ограниченный GPU expert cache
и reader/H2D pipeline. **HTTP API, профиль запуска и сессии ещё не реализованы.**

План и результаты: [план](../../docs/mimo-v2.6-flash/MIMO26_FLASH_IMPLEMENTATION_PLAN.md),
[статус](../../docs/mimo-v2.6-flash/MIMO26_FLASH_IMPLEMENTATION_STATUS.md).

## Границы поддержки

Проверен локальный `MiMo-V2.6-Flash-RL-GSQ-RCO-3.5bit.gguf`: 48 блоков,
dense0 +47 MoE, 256 экспертов/top-8, full/SWA128, K192/V128, partial RoPE64,
BF16/F32/Q2_K/Q3_K/MXFP4. Inspector не читает payload и не доказывает
правильность численных вычислений. Другие layouts требуют отдельной проверки.
Production inspector/engine не принимают synthetic fixtures. Отдельные test binaries
`strata-mimo2-runtime-check` и `strata-mimo2-fixture-engine` принимают только
заданную трёхслойную fixture geometry и tokenizer `none`.

В этом экспорте нет MTP или multimodal weights. `load_mtp=true` проверяется
только как отрицательный случай: регистрация по-прежнему содержит 472 тензора
и ноль NextN-слоёв. Отдельные MTP/DFlash GGUF теперь проверяются экспериментальным
`strata-mimo2-spec-check`; это не serving-режим основного engine.

## Сборка CPU oracles

Отдельная сборка не меняет зависимости других backend. Требуются CMake≥3.24,
Ninja и C++17 compiler. На Windows запускать из x64 Native Tools / Developer
PowerShell для Visual Studio, чтобы были доступны MSVC и Windows SDK `rc`/`mt`.
Python-проверки используют установленные `regex` и `jinja2`.

Из корня репозитория:

```powershell
cmake -S backends/mimo2 -B build-local/mimo2-oracles -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_MIMO_ARCHIVE=C:/work/git/my-repos/Strata/build-local/llama-glm-86ebfef2.tar.gz
cmake --build build-local/mimo2-oracles --target strata-mimo2-tokenizer strata-mimo2-loader strata-mimo2-template strata-mimo2-cache-check -j 8
ctest --test-dir build-local/mimo2-oracles --output-on-failure --no-tests=error
```

Архив Unsloth revision `86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9` проверяется
по SHA-256 `f8e524b635b726bae74fd8f84bb9249e5b09384c63207f6707c5a3f921acad99`.
Он распаковывается в собственный `_deps`. Патчей dependency нет. Сборка пишет
`mimo2-build-manifest.json`; CTest проверяет запуск трёх программ и их версии.
Полные проверки выполняются отдельно:

```powershell
$mimoModel = 'H:/models/mimo-v2.6-flash/MiMo-V2.6-Flash-RL-GSQ-RCO-3.5bit.gguf'
python -X utf8 tools/inspect_mimo2_gguf.py --gguf $mimoModel --output docs/mimo-v2.6-flash/MIMO26_FLASH_INSPECTION.json
python -X utf8 tools/check_mimo2_oracles.py --gguf $mimoModel --bin-dir build-local/mimo2-oracles/bin --kind loader --output docs/mimo-v2.6-flash/MIMO26_FLASH_LOADER_VALIDATION.json
python -X utf8 tools/check_mimo2_oracles.py --gguf $mimoModel --bin-dir build-local/mimo2-oracles/bin --kind tokenizer --output docs/mimo-v2.6-flash/MIMO26_FLASH_TOKENIZER_VALIDATION.json
python -X utf8 tools/check_mimo2_oracles.py --gguf $mimoModel --bin-dir build-local/mimo2-oracles/bin --kind template --output docs/mimo-v2.6-flash/MIMO26_FLASH_TEMPLATE_VALIDATION.json
python -X utf8 -m unittest tools.test_mimo2_gguf tools.test_mimo2_tokenizer tools.test_step35_tokenizer tools.test_glm5next_tokenizer tools.test_hy3_tokenizer
```

Loader checker сопоставляет каждое имя, shape, type, byte count и геометрию
каждого слоя между Python contract и скомпилированным llama loader.
Tokenizer checker проверяет IDs и обратные UTF-8 bytes, включая Unicode,
код, числа, special tokens, комбинируемые символы и `parse_special` on/off.
Template checker сравнивает Python Jinja с независимым native Jinja на history,
thinking, system, tools, строковых/mapping arguments и tool results.
Это проверка raw template; нормализация API arguments и HTTP остаются в P2.

## Сверка опубликованной конфигурации

Исходники читаются как данные, не выполняются. После загрузки двух файлов
по закреплённым URLs из отчёта можно повторить проверку:

```powershell
python -X utf8 tools/check_mimo2_reference.py --gguf $mimoModel --config build-local/mimo2-reference/config-3b38d063.json --reference-loader build-local/mimo2-reference/mimo2-58367713.cpp --candidate-loader build-local/mimo2-oracles/_deps/mimo2_candidate-src/src/models/mimo2.cpp --output docs/mimo-v2.6-flash/MIMO26_FLASH_REFERENCE_VALIDATION.json
```

Сверены pinned config `3b38d063` и `mimo2.cpp` upstream `58367713`.
Loader совпадает с кандидатом байт-в-байт. Проверка не устанавливает
эквивалентность converter, всего dependency tree или весов.

## Особенности tokenizer и stop policy

`qwen2` использует свой regex: одна цифра и отличие от `qwen35` в combining marks.
MiMo factory в `tools/mimo2_tokenizer.py` отдельно повторяет native promotion
токена `</s>` (128247) из NORMAL в CONTROL. Без него 147 из 614 случаев
не совпадали; после исправления совпадают все 614. Исходная metadata не меняется.
Другие `qwen2` словари эту замену не получают.

Oracle сообщает BOS11 (обычная запятая) и EOG
`128247,151643,151645,151662,151663,151664`. Это эвристики данной dependency.
BOS не вставляется: GGUF задаёт `add_bos=false`, вызовы используют `add_special=false`.
Runtime останавливается только на экспортированном EOS151645. Проверены
настоящие завершения ответов и predicate: BOS11, PAD151643, `</s>`128247 и
FIM151662–151664 не останавливают генерацию. Политика не переносит эвристический
EOG-набор dependency на диалог. Потоковый parser/tools остаются в P2.

## CUDA fixtures — MIMO-02

На RTX5090, Windows, CUDA13.0.48 / sm120, MSVC19.44.35222.0 прошли
**96 проверок операций и 132 проверки графа**. Синтетические веса не читаются
из полного GGUF. Отчёты: [kernels](../../docs/mimo-v2.6-flash/MIMO26_FLASH_CUDA_KERNELS_VALIDATION.json),
[graph](../../docs/mimo-v2.6-flash/MIMO26_FLASH_CUDA_GRAPH_VALIDATION.json),
[сборка и регрессии](../../docs/mimo-v2.6-flash/MIMO26_FLASH_CUDA_BUILD_VALIDATION.json).

Операции: F32/BF16/Q2_K/Q3_K/MXFP4, dense/routed, batch1/4/17,
256 экспертов/top-8, повторяемые IDs, padding/strides и защита входных буферов.
Сверка с CPU и скалярным накоплением double по деквантованным весам.
Router: sigmoid, bias только для выбора, нормализация без дополнительного multiplier.

Граф: 3 блока full/SWA/full, hidden256, FFN512/256, 16 экспертов/top-8,
vocab64. Геометрия attention как в модели: heads64, KV4/8/4, K192/V128,
RoPE64, SWA128, sinks и value scale0,707. Проверяются F32 и mixed weights,
384 позиции prefill/decode, границы127/128/129 и255/256/257,
ограниченный SWA cache, FA off/on, append/rollback/clear/restore.
CPU-эталон mixed fixture содержит F32-деквантизацию **тех же** packed weights.
Audit не обнаружил CPU-вычислений в проверяемых GPU-графах.

Из x64 developer shell, из корня репозитория:

```powershell
cmake -S backends/mimo2 -B build-local/mimo2-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=C:/work/git/my-repos/Strata/build-local/cuda-13.0/bin/nvcc.exe -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_MIMO_ARCHIVE=C:/work/git/my-repos/Strata/build-local/llama-glm-86ebfef2.tar.gz -DSTRATA_MIMO_CUDA=ON -DSTRATA_MIMO_STRICT_F32=ON -DSTRATA_MIMO_ROUTED_STRIDES=ON -DSTRATA_MIMO_MXFP4_PRECISE=ON
cmake --build build-local/mimo2-cuda --target strata-mimo2-kernels-check strata-mimo2-graph-check -j 8
python -X utf8 tools/check_mimo2_cuda.py --build build-local/mimo2-cuda --kind kernels --output-dir build-local/mimo2-validation/kernels-new --cuda-bin build-local/cuda-13.0/bin/x64
python -X utf8 tools/check_mimo2_cuda.py --build build-local/mimo2-cuda --kind graph --output-dir build-local/mimo2-validation/graph-new --cuda-bin build-local/cuda-13.0/bin/x64
python -X utf8 -m unittest tools.test_mimo2_cuda
```

Каждый `--output-dir` должен быть новым. Runner задаёт локальные для дочернего
процесса `NVIDIA_TF32_OVERRIDE=0`, `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32`,
`GGML_CUDA_DISABLE_GRAPHS=1`, `LLAMA_GRAPH_REUSE_DISABLE=1`; пишет JSON/stdout/stderr,
hashes binary/manifest и выборки памяти с интервалом около0,5с.
При достижении 95% общей RAM/VRAM останавливается только запущенный им тест.
Выборки включают чужие процессы и не гарантируют захват каждого пика.

### Политика точности CUDA baseline

Три MiMo-local CMake-поправки создают translation units внутри build;
исходный архив, `_deps` и другие backend не меняются:

- `StrictF32.cmake`: custom F32 MMF учитывает TF32 override, BF16 MMF — запрос
  `f32` compute. BF16 веса сохраняются; для выбранного baseline prefill
  не округляет активации в BF16 дополнительно к уже сохранённым весам.
- `RoutedStrides.cmake`: исправляет physical row indexing при gather входов
  экспертов. Ошибка проявилась после переключения F32 на точный путь.
- `Mxfp4Precision.cmake`: на Blackwell отключает MXFP4 MMQ с дополнительными
  FP4-активациями; остаются MMVQ для малого batch и GPU dequantization/cuBLAS
  для остальных случаев. Q2_K/Q3_K и другие backend этим не переключаются.

[История расхождений](../../docs/mimo-v2.6-flash/MIMO26_FLASH_CUDA_DIAGNOSTIC.json):
без поправок 20/96 failures; после только TF32 fix — 10/96;
после трёх первых поправок — 96/96 kernels, но 129/132 graph.
Учет `f32` compute для BF16 снизил mixed-logit max abs с0,002447 до0,000967,
после чего прошли все132 graph cases **без ослабления допусков**.
Это выбранный режим точности. Первые полные замеры с ним приведены в статусе;
отдельная оценка стоимости каждого precision override ещё не проводилась.

CUDA Graphs, повторное использование графа и full-model throughput этим не проверены.
Native restore после уплотнения SWA совпадает в допуске, но не бит-в-бит;
короткий tail rollback в двух fixtures совпал бит-в-бит. Это не готовая система сессий.

## Синхронный engine — MIMO-03

Сборка Windows/CUDA из того же developer shell, с указанными выше archive/compiler:

```powershell
cmake -S backends/mimo2 -B build-local/mimo2-cuda -DSTRATA_MIMO_RUNTIME=ON
cmake --build build-local/mimo2-cuda --target strata-mimo2 strata-mimo2-runtime-check strata-mimo2-fixture-engine -j 8
$env:PATH = (Resolve-Path build-local/cuda-13.0/bin/x64).Path + ';' + $env:PATH
build-local/mimo2-cuda/bin/strata-mimo2.exe --native $mimoModel --serve --max-context 512 --batch-size 8 --copy-mode pinned
```

Production binary сначала проверяет GGUF metadata, точную geometry, все tensor
names/types/shapes, alignment, EOF и отсутствие пересечений. Настоящий GGUF
хранит attention arrays как **INT32**, fixtures — UINT32; проверены оба договора.
На GPU находятся все non-routed weights; BF16 сохраняется. Эксперты остаются
в demand mmap без полного prefault. Любой CPU compute node и попытка общего
копирования expert tensor вне selected-copy пути завершают запрос ошибкой.

В MIMO-03 `pinned` означал синхронное чтение выбранных диапазонов через native file reader,
`cudaHostAlloc` на16 МиБ, H2D и fence после каждого chunk. Буфер не переиспользуется
до fence; отменённое file I/O завершается до возврата. Mapping registry удаляется
при unload. `native` — эталонный selected-copy из mmap в закреплённой ревизии llama,
с тем же графом и precision policy. Это transport reference, не независимый
Transformers oracle. В MIMO-04 режим `pinned` также поддерживает mmap и GPU cache;
актуальный путь явно указан в INFO и параметрах ниже.

Глобальные RAM и VRAM проверяются перед load/context/decode; pinned-copy — также
перед ranges. Windows VRAM берётся из NVML по PCI device, без WDDM fallback на
CUDA free memory. Используется потолок95%, а не заполнение памяти до95%.
Начальный запас под scratch — консервативный admission reserve8 ГиБ;
фактические allocations и выбранный512 context зафиксированы в отчётах.

Pipe, protocol1:

- `INFO ...`, затем `READY 512 stop` при готовности; capabilities точные:
  text-only, greedy, EOS151645, MTP/session reuse off; cache/pipeline/readers указаны явно.
- `ENC 0|1 <UTF-8 hex>` → `IDS ...`; BOS не добавляется.
- `GEN <count> <comma-separated-ids>` → `PP`, затем `T <id>`, затем `DONE`.
  Sampling options и session keys пока не принимаются. Каждый запрос начинает
  с пустого full/SWA KV; `--logits-file <new-path.f32>` пишет строку F32/vocab
  перед каждым выбранным токеном, append между запросами одного процесса.
- `STOP` отменяет активный/ожидающие запросы; после drain/clear возможен новый
  запрос. `QUIT`/EOF отменяют работу и освобождают ресурсы. Ошибка даёт `ERR`.

Проверки (каталоги должны быть новыми):

```powershell
python -m tools.check_mimo2_cuda --build build-local/mimo2-cuda --kind runtime --output-dir build-local/mimo2-validation/runtime-new --cuda-bin build-local/cuda-13.0/bin/x64
python -m tools.check_mimo2_engine --build build-local/mimo2-cuda --model $mimoModel --suite corpus --output-dir build-local/mimo2-validation/corpus-new --cuda-bin build-local/cuda-13.0/bin/x64
python -m tools.check_mimo2_engine --build build-local/mimo2-cuda --model $mimoModel --suite repeat --output-dir build-local/mimo2-validation/repeat-new --cuda-bin build-local/cuda-13.0/bin/x64
```

Engine runner требует NumPy/psutil, сравнивает greedy IDs и **все** logits
выданных токенов. Corpus: EN/RU/ZH/code/numbers и249-token prompt через SWA128;
repeat: три свежих запроса по32 выходных токена, тот же prompt24 tokens.
Cancel/error→fresh и clean unload проверяются в обоих режимах. Выборки общей
памяти и process working set снимаются примерно раз в секунду; короткие пики
могут быть пропущены. `source_bytes` — native read bytes, не физический SSD I/O;
source/H2D timers заполнены только для управляемого copy-mode `pinned`.

## Кэш экспертов — MIMO-04

Default engine запрашивает14 ГиБ GPU cache, использует mmap, один reader,
pipeline chunk8 МиБ и заполнение кэша только на decode.
Live memory clamp уменьшает cap по доступной VRAM.
Prefill использует уже сохранённые матрицы, но не вытесняет ими decode
working set. Объём и результаты замеров приведены в
[статусе](../../docs/mimo-v2.6-flash/MIMO26_FLASH_IMPLEMENTATION_STATUS.md).
Запуск с defaults:

```powershell
build-local/mimo2-cuda/bin/strata-mimo2.exe --native $mimoModel --serve
```

Параметры:

- `--expert-cache-mib N`: запрошенный предел allocations, 0 отключает кэш.
  Фактический предел уменьшается по доступной global VRAM; это не payload bytes.
- `--expert-reader mmap|file`: mmap использует доступные RAM pages, file —
  native ReadFile. Pipeline доставляет данные через ограниченный pinned ring;
  синхронный mmap использует driver staging, синхронный file — staging16 МиБ.
  File reader уменьшает process working set, но в локальных повторах был медленнее.
- `--expert-cache-prefill off|on`: разрешение новых entries на prefill.
- `--copy-mode native`: исходный selected mmap copy, cache по умолчанию0;
  одинаковые GPU graph, precision и quants для transport reference.

Прежний file baseline: `--expert-cache-mib 0 --expert-reader file`.
Runner по умолчанию сохраняет uncached baseline; кэш для теста задавать явно:

```powershell
cmake --build build-local/mimo2-cuda --target strata-mimo2-cache-check strata-mimo2-runtime-check -j 8
ctest --test-dir build-local/mimo2-cuda -R mimo2_cache_ownership --output-on-failure
python -m tools.check_mimo2_engine --build build-local/mimo2-cuda --model $mimoModel --suite repeat --expert-cache-mib 14336 --expert-reader mmap --expert-cache-prefill off --output-dir build-local/mimo2-validation/cache-repeat-new --cuda-bin build-local/cuda-13.0/bin/x64
```

LRU хранит отдельные quantized matrices Q2_K/Q3_K/MXFP4. Ключ включает generation,
зарегистрированный tensor identity и expert ID; registry проверяет type/shape/
stride/file range. Entry публикуется только после успешной завершённой копии.
D2D hit/fill и H2D fence защищают адреса до eviction/reuse; pipeline дополнительно
защищает будущие cache hits через route pins до завершения доставки.
При новом model generation кэш и mapping identities сбрасываются.

Allocation probe на этой Windows/5090 выявил округление небольших cudaMalloc.
Теперь runtime явно округляет каждый allocation до2 МиБ, отдельно считает
`cache_bytes` и `cache_payload_bytes`, а live NVML оставляет5% total +256 МиБ
запаса. До первого успешного полного batch дополнительно сохраняются3 ГиБ
для ленивых FP32 cuBLAS pools. Поэтому стартовый INFO limit может быть меньше
запрошенного; после прогрева он пересчитывается.

Mmap reader ограничивает **собственный** working set с расчётом на94% общей RAM,
с учётом других процессов. Windows может вытеснять чистые страницы GGUF;
глобальный guard остаётся95%. Короткие запросы читают лишь часть экспертов,
поэтому RAM заполняется по потребности. Полный corpus уже проверен около
предела RAM; длительный стресс при меняющейся внешней нагрузке ещё предстоит.

CPU хранит экспертные веса и обслуживает I/O, tokenization, greedy selection,
scheduling и ожидания CUDA. **Матричные операции модели выполняются на GPU.**
Перед каждым графом `gpu_only_audit.inc` отвергает любой вычислительный узел
на не-GPU backend. Отдельный negative fixture проверяет этот отказ до исполнения.
CPU-эталоны в `kernels-check`/`graph-check` используются только в числовых тестах.

Метрики: `requested_bytes = h2d_bytes + cache_hit_bytes` для успешного запроса;
`cache_fill_bytes` — дополнительные D2D bytes при сохранении промахов.
В синхронном mmap `h2d_ms` включает page faults и driver staging, `source_bytes=0`;
это не отсутствие чтений SSD. File `source_bytes` также не измеряет physical I/O.
Runner сохраняет sampled working set/global memory и cumulative CPU user/system
time процесса; CPU time сам по себе не показывает выполнение матриц.

## Конвейер доставки — MIMO-05

`--expert-readers 1|2` включает ограниченный транспорт (default1), `0` сохраняет
синхронный baseline. `--expert-chunk-mib 4|8|16` задаёт размер одного из четырёх
слотов: при8 МиБ это32 МиБ pinned RAM и32 МиБ VRAM. Source reader выбирается
через прежний `--expert-reader mmap|file`. Pipeline требует managed cache mode.
При `--copy-mode native` или `--expert-cache-mib 0` readers автоматически0,
если пользователь не задал их явно. Несовместимая явная комбинация отвергается.

После чтения настоящих router IDs планируются gate/up/down только этого слоя.
Промахи читаются в pinned ring и загружаются отдельным H2D stream в device ring.
Producer не пишет scheduler scratch: тот может ещё содержать живые активации.
`ready` event защищает ring→scratch D2D; `used` event запрещает reader переиспользовать
слот до завершения consumer. Перед записью scratch завершается предыдущий
вычислительный split. Cache hits будущих копий защищены от eviction; cache fills
по-прежнему публикуются после fence. Защитные хвосты512 байт доставляются тем же
планом, но не становятся самостоятельными полными cache entries.

Успешное завершение, отмена, ошибка worker, смена модели и unload закрывают план,
дожидаются readers/CUDA и снимают pins. После ошибки worker transport пересоздаётся
для следующего запроса. Budget проверяется перед каждым route и decode; внутри
route фиксированный ring и cache cap ограничивают новые allocations.

Числовая проверка выявила зависимость первого decode в новых synthetic contexts
от истории allocations, включая синхронный reader. Отключение PDL или fusion
не исправило её. Создание контекста теперь явно очищает KV, как уже делал pipe
перед каждым запросом. После исправления шесть отдельных запусков прошли
**212/212 runtime cases**. Допуски не менялись, PDL/fusion policy осталась прежней.
Подробности и исходные FAIL сохранены в статусе и diagnostic report.

Пример сравнения с тремя прогретыми запросами после первого:

```powershell
python -m tools.check_mimo2_engine --build build-local/mimo2-cuda --model $mimoModel --suite repeat --repeats 4 --expert-cache-mib 14336 --expert-reader mmap --expert-cache-prefill off --expert-readers 1 --expert-chunk-mib 8 --output-dir build-local/mimo2-validation/pipeline-repeat-new --cuda-bin build-local/cuda-13.0/bin/x64
```

Для диагностики добавить `--trace-graphs 4` runner; прямой engine дополнительно
требует `--trace-file <new.json>`. CUDA events ставятся на реальные H2D/compute
streams. Trace меняет timing, поэтому не используется для throughput сравнения.
Пересечение интервалов graph submission не равно профилю отдельных CUDA kernels;
в первом полном trace overlap был небольшим, на двух decode graphs — нулевым.

Pipeline metrics: `pipeline_file_bytes` / `pipeline_mmap_bytes` различают ReadFile
и memcpy из mmap в pinned RAM; `source_ms` — сумма wall time всех readers,
которая может превышать elapsed. `pipeline_delivered_bytes` / `pipeline_d2d_bytes`
считают ring→scratch, `pipeline_unused_bytes` — загруженный отменённый suffix.
`h2d_ms` прежнего синхронного пути не измеряет async H2D; для него служит CUDA trace.
`pipeline_consumer_wait_us`, `pipeline_slot_wait_us`, `pipeline_submit_us` — CPU
wait/submission time, не доказательство GPU overlap. `pipeline_read_peak` —
максимум одновременно работающих readers за жизнь transport, не за один запрос.
Physical SSD I/O не измеряется.

## Отдельное сравнение MTP / DFlash

`STRATA_MIMO_SPEC_PROBE=ON` добавляет offline greedy executable и private patches
для MiMo-only NextN sidecar, пяти target features и DFlash value scale0.612.
По умолчанию опция выключена. Основной `strata-mimo2` не получает MTP/API switches.
Все модели работают на CUDA; CPU обслуживает файлы, scheduling и выбор токена.

Native MTP использует только первую обученную голову и не загружает веса двух
остальных. Их нельзя автоматически соединять по схеме Step35: для MiMo это
отдельные смещения относительно target hidden state. DFlash поддерживает
до7 предложений плюс anchor и confidence cutoff. MASK151675 берётся из draft,
а не из target. Опция `--share-target` использует RL embedding/head и отдельный
learned MASK на GPU; это другая конфигурация draft, её нужно измерять отдельно.
Выходная BF16-матрица при этом считается по столбцам на GPU, чтобы второй
контекст не создавал полную дополнительную F32-копию большой матрицы.

Контекст512, batch8, F32 KV, FA on, `swa_full=true`: все позиции теста сохранены
для rollback. В режиме draft перед заполнением expert cache выполняется отдельный
discarded batch8 с logits на всех позициях для прогрева ленивых CUDA pools.
Это входит во время загрузки; no-draft baseline не держит неиспользуемый verify
workspace. Перед каждым запросом оба KV очищаются; expert cache сохраняется.
Память учитывается глобально с пределом95%, cache до14 ГиБ с live clamp.

Из уже настроенного CUDA runtime build, в MSVC developer shell:

```powershell
cmake -S backends/mimo2 -B build-local/mimo2-cuda -DSTRATA_MIMO_SPEC_PROBE=ON
cmake --build build-local/mimo2-cuda --target strata-mimo2-spec-check strata-mimo2-spec-verify-check -j 8
ctest --test-dir build-local/mimo2-cuda -R mimo2_spec_verified_prefix --output-on-failure
python -m tools.prepare_mimo2_spec_requests --model $mimoModel --output build-local/spec-none.json --depth 0
python -m tools.check_mimo2_speculative --model $mimoModel --requests build-local/spec-none.json --output-dir build-local/spec-none
python -m tools.prepare_mimo2_spec_requests --model $mimoModel --output build-local/spec-dflash.json --depth 7 --p-min 0.7
python -m tools.check_mimo2_speculative --model $mimoModel --kind dflash --draft $mimoDraft --requests build-local/spec-dflash.json --output-dir build-local/spec-dflash --reference-dir build-local/spec-none
```

Все report/logits destinations должны быть новыми. `--suite tune` создаёт
перебор depth1/3/7 и порога0.7, `--suite boundary` — prompt длиннее SWA128.
Runner сохраняет IDs, F32 logits, acceptance, H2D, cache, timing, CPU audit и
sampled memory. Первый запрос каждого prompt/config помечен warmup и исключается
агрегатором `tools.summarize_mimo2_speculative`. Не выдавать `output_mismatch`
за качество PASS, даже если отдельные примеры совпали. Пакетный target может
отличаться от последовательного; диагностический `oracle_ids` проверяет это
без draft model и также пропускает все предложения через target.

Проверены только greedy и один sequence. Stochastic correction, HTTP streaming,
cancellation, context shift и MTP heads2/3 в этот probe не входят. Его скорости
следует сравнивать с его же baseline, а не напрямую с прежним pipeline benchmark.

Проверены23 cache ownership cases,212 runtime cases,96 kernels и132 graph checks,
полный corpus, отмена/recovery и bit-exact logits/IDs. API/profile — следующий
отдельный этап; batched D2D и более широкий pressure/context stress ещё предстоят.
