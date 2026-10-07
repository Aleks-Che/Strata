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

Кеш хранит отдельные quantized matrices Q2_K/Q3_K/MXFP4. Ключ включает generation,
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

По умолчанию `STRATA_MIMO_CACHE_SLAB_MIB=16`: несколько матриц одинакового
размера размещаются в общем блоке. `0` возвращает отдельные allocations;
`32` — экспериментальный вариант, скорость которого ещё не измерена.
Размер блока округлён до2 МиБ, slots — до256 байт; последний блок уменьшается
под остаток бюджета. `cache_bytes` включает весь блок, свободные slots и padding,
`cache_slot_bytes` — живые slots, `cache_payload_bytes` — полезные веса.
Адреса живых slots не перемещаются; пустые блоки освобождаются.
Global admission продолжает учитывать NVML, KV, workspace и прочие процессы.
Переменную задавать до запуска. В измеренном MIMO-07 A/B менялся только allocator:
tensor batching был выключен, частотная история и compaction не перенесены.
На RTX5090/128 ГБ RAM итоговый ABBA дал6,434→6,952 ток/с (+8,1%),
payload9,855→12,370 ГиБ при cache около12,9 ГиБ. Условия, проверки и
ограничения: [MIMO-07](../../docs/mimo-v2.6-flash/MIMO26_FLASH_SLAB_CACHE.md).

Воспроизводимый A/B без MTP (новый каталог результатов):

```powershell
python -m tools.benchmark_mimo2_cache --build build-local/mimo2-cuda --model $mimoModel --cuda-bin build-local/cuda-13.0/bin/x64 --order 0,16,16,0 --output-dir build-local/mimo2-validation/slab-abba-new
```

Runner сравнивает все сохранённые logits и IDs, исключает первый запрос каждого
prompt из скорости, сохраняет system disk read counters, memory samples и
счётчики cache. `tools.check_mimo2_engine` и `tools.check_mimo2_cuda` принимают
`--cache-slab-mib 0|16|32` и явно задают соответствующую переменную процессу
(default этих контрольных runners — `0`, default самого engine — `16`).
Проверки `mimo2_cache_ownership` и `mimo2_slab_ownership` выполняются без GPU;
вторая покрывает реальные размеры квантованных матриц, OOM, holes, pins,
ошибки заполнения, давление бюджета и5000 операций со смешанными размерами.

По умолчанию `STRATA_MIMO_PIPELINE_BATCH=1` объединяет доставку выбранных экспертов одного
scheduler input. Одна backend fence защищает scratch от предыдущего consumer;
одна copy-stream fence завершает доставку всех матриц и guard tails. Route pins
сохраняют cache sources. Затем воспроизводится прежний порядок LRU/admission:
каждый новый cache fill по-прежнему завершается перед публикацией. Это не
асинхронная публикация кеша и не изменение арифметики модели. `0` — контроль.

`pipeline_copy_batches` считает завершённые tensor deliveries;
`pipeline_copy_fences` — host waits доставки и cache fills, включая prefill;
`pipeline_scratch_fences` — дополнительные backend waits перед доставкой.
Счётчики не включают route drains, прочие scheduler fences и producer events.
`pipeline_batch_ms` включает source waits и admission, это wall time, не GPU time.
Контрольные runners принимают `--pipeline-batch 0|1` (default0).

Сравнение с одинаковым slab16, без MTP:

```powershell
python -m tools.benchmark_mimo2_cache --build build-local/mimo2-cuda --model $mimoModel --cuda-bin build-local/cuda-13.0/bin/x64 --axis pipeline --fixed-slab-mib 16 --order 0,1,1,0 --output-dir build-local/mimo2-validation/tensor-abba-new
```

Slab-axis runner явно выключает tensor batching, чтобы сохранять контроль
прежнего эксперимента. Оба режима не меняют лимиты global RAM/VRAM95%.
На RTX5090/128 ГБ RAM ABBA дал6,591→7,299 ток/с (+10,7%); относительно
последнего прогретого контроля6,933 прибавка5,3%. Logits/IDs совпали побитово.
[Условия, сборки и ограничения MIMO-08](../../docs/mimo-v2.6-flash/MIMO26_FLASH_TENSOR_BATCH.md).

По умолчанию `STRATA_MIMO_CACHE_DECAY=65536` включает частотный допуск:
редкая новая матрица сохраняет более частого кандидата на вытеснение. Сравниваются
64 старых unpinned entries, прежде всего того же размера allocation. Равные
частоты разрешают замену. При свободном месте допуск сохраняется. `0` возвращает
прежний LRU; `16384`/`131072` доступны для опытов, их скорость ещё не измерена.

История до65536 ключей generation/tensor/expert учитывает только выбранные
полные матрицы, включая misses, которым отказано в admission. Guard tails и
prefill при admission off её не обучают. Период65536 наблюдений — примерно58
обычных decode tokens этой модели, после него частоты лениво делятся пополам.
Reset статистики запроса сохраняет историю; новый кеш её очищает. Pins, бюджет
и завершение fill перед публикацией не менялись. Переменную задавать до запуска.

`INFO expert_cache_decay` и метрики `cache_decay`, `cache_history_keys`,
`cache_frequency_updates/rejected/candidates` показывают выбранную политику
и её работу. `check_mimo2_cuda`/`check_mimo2_engine` принимают `--cache-decay`
(default0), поэтому текущий режим проверяется с
`--cache-slab-mib 16 --pipeline-batch 1 --cache-decay 65536`.
CPU CTest `mimo2_frequency_ownership` покрывает21 случай и5000 операций.

```powershell
python -m tools.benchmark_mimo2_cache --build build-local/mimo2-cuda --model $mimoModel --cuda-bin build-local/cuda-13.0/bin/x64 --axis frequency --fixed-slab-mib 16 --order 0,65536,65536,0 --output-dir build-local/mimo2-validation/frequency-abba-new
```

Frequency axis сохраняет tensor delivery1 и slab16; остальные axes явно ставят
decay0. На RTX5090/128 ГБ RAM, без MTP, warm ABBA дал6,923→7,692 ток/с
(+11,1%), к быстрейшему контролю+8,2%. H2D прогретого decode−11,8%, fills−82,9%.
При смене темы первый запрос может копировать больше весов, пока история
адаптируется. [Измерения и ограничения MIMO-09](../../docs/mimo-v2.6-flash/MIMO26_FLASH_FREQUENCY_CACHE.md).

`STRATA_MIMO_CACHE_FILL_BATCH=0` остаётся default. Экспериментальный `1`
резервирует невидимые cache entries, затем копирует сохранившиеся reservations
одного scheduler input до общей fence. До submit их можно переиспользовать;
после submit ошибка сначала завершает копии, затем удаляет pending entries.
Требует tensor delivery1 и pipeline; sync/per-matrix delivery не меняются.
Метрики `cache_fill_batch`, `cache_pending`, `pipeline_fill_batches` и
`pipeline_fill_submissions` показывают режим и завершение копий.
Контрольные runners принимают `--cache-fill-batch 0|1` (default0).

На этой RTX5090 групповые fills снизили waits на19,17%, но ABBA дал
7,861→7,826 ток/с (−0,45%). Один повтор synthetic mixed/file/cache8 также
дал exact-logit mismatch при совпавших байтах. Режим остаётся выключенным;
повтор PASS не считается исправлением. Новые ownership tests:17 cases,
16 000 mixed operations; CUDA checker добавляет32 fill-error/recovery cases.

В обычную сборку включена перегрузка `require(bool,const char *)`, устраняющая
временные строки на успешной проверке logits. NaN/Inf и динамические ошибки
проверяются по-прежнему. На MSVC Release CPU guard152576 значений потребовал
0 вместо152576 allocations и около0,057 вместо3,22 мс; это отдельный microbenchmark.
`strata-mimo2-literals-check` / CTest `mimo2_literal_guards` проверяет11 случаев.
Отдельный ABBA полной модели с fill0:7,721→7,933 ток/с (+2,74%), все logits/IDs
exact. Время вне decode снизилось с3,877 до0,345 мс/output; forward почти не изменился.
[Замеры полной модели и диагностика MIMO-10](../../docs/mimo-v2.6-flash/MIMO26_FLASH_FILL_AND_GUARDS.md).

`MatrixHash` перемешивает generation/tensor/expert во всех битах. Это устраняет
скопление разных слоёв в одной корзине MSVC history map. В CPU-тесте141×256
ключей максимальная цепочка141→7,4 млн обращений примерно44× быстрее.
`strata-mimo2-hash-check` / CTest `mimo2_hash_history`:1488776 сравнений истории;
`--benchmark` добавляет CPU ABBA. Полная модель сохранила IDs/logits и cache
policy, но устойчивый end-to-end прирост не подтверждён из-за разброса
контролей. [Измерения MIMO-11](../../docs/mimo-v2.6-flash/MIMO26_FLASH_HASH_CACHE.md).

Резидентный `ExpertCache` использует unordered index, LRU по стабильным
указателям и pin count в самой записи. Rehash сохраняет адреса; map iterators
между операциями не удерживаются. Порядок victim scan/admission, pending tickets
и route ownership сохранены. `strata-mimo2-cache-index-check` / CTest
`mimo2_cache_index_policy` сравнивает replay с замороженным ordered-map reference;
`--benchmark` добавляет144000 обращений. CPU63,330→32,505 мс, но прогретый
decode8,390 против контроля8,401 ток/с. Индекс включён как сокращение CPU-работы,
без заявления об ускорении генерации. [MIMO-12](../../docs/mimo-v2.6-flash/MIMO26_FLASH_CACHE_INDEX.md).

`STRATA_MIMO_PACK_GUARDS=1` — default для mmap +tensor batching. Missing guard
tails до512 байт собираются по route и передаются одним H2D. Дополнительные
буферы2MiB RAM +2MiB VRAM ограничены4096 plan entries и не пересекаются со scratch;
end_plan дожидается copy stream до их повторного использования и освобождения.
Cache policy, route pins и GPU math сохранены. File/per-range используют прежний путь.
Значение0 отключает объединение. `expert_stage_mib=34` включает guard staging;
к `pipeline_device_bytes` ring добавлять `pipeline_guard_capacity_bytes`.
`pipeline_guard_ranges/batches/bytes` показывают объединение; `pipeline_chunks`
считает и обычные H2D, и guard batches. Unused guard bytes при abort учтены.

`STRATA_MIMO_EARLY_HOST_REFILL=0` — default. Значение1 разрешает ранее заполнять
host slot после его H2D; device reuse остаётся защищён consumer event.
Измерение7,992→7,912 ток/с не подтвердило speedup. Guard batching дал10,336 ток/с,
+24,02% к быстрейшему контролю8,334, H2D operations−49,03%. Два кандидата10,300/10,372;
контроли8,334/7,275 различались. [MIMO-13](../../docs/mimo-v2.6-flash/MIMO26_FLASH_HOST_PIPELINE.md).
CUDA runtime вырос до224 случаев (256 при grouped fills): добавлены scattered
F32/mixed fixtures с реальными missing guards, byte audit, cancel/error/recovery.
Оба режима224/224 и final-default full-model corpus прошли; прежние F32/mixed
диагностики остаются открытыми. CPU7 CTest/Python29 PASS.

Benchmark axes `early` и `guards` сравнивают0/1 с `--order 0,1,1,0`, slab16,
tensor batch1, decay65536 и fill0. `chunk` допускает0(control8MiB),4,16;
full-model ranking chunk4/16 ещё не выполнялся. CUDA/engine checkers принимают
`--early-host-refill 0|1` и `--pack-guards 0|1` (checker default0, явно для контроля).

Benchmark axis `fill` сравнивает0/1 с tensor delivery1 и fixed decay65536;
`binary` с `--control-binary <preserved.exe>` сравнивает два executable,
сохраняя fill0. Для этих axes явно задавать `--order 0,1,1,0`.
Прочие axes выключают fill batching, сохраняя прежние контроли.
`--warmups` (default1) задаёт число исключённых запросов на тему,
`--expert-cache-mib` (default14336) — запрос кеша для сравнений с фиксированным
бюджетом. Live clamp95% продолжает действовать; engine defaults не меняются.

`STRATA_MIMO_D2D_BATCH=2` — эксперимент MIMO-15: собственное CUDA scatter-copy ядро
для cache/guard → scratch одного тензора. Один запуск получает до32 независимых
диапазонов, tile16 КиБ; metadata передаются значением, без дополнительных GPU
allocations. Default0 сохраняет отдельные CUDA copies, mode1 — экспериментальный
CUDA13 `cudaMemcpyBatchAsync`. Режим1 требует CUDA13 и не показал ускорения в
[MIMO-14](../../docs/mimo-v2.6-flash/MIMO26_FLASH_D2D_BATCH.md).

Все источники защищены route pins; назначения не пересекаются. Пакет отправляется
до ожидания недостающих матриц; ring slots доставляются и освобождаются отдельно.
Admission/fills начинаются после прежней общей fence. Drain при ошибке/отмене
завершается до освобождения pins. Без tensor delivery эти режимы не используются.
`pipeline_d2d_batch/batches/ranges` показывают режим и логические пакеты;
`pipeline_d2d_kernel_launches` — фактические запуски scatter-copy.
CUDA/engine checkers принимают `--d2d-batch 0|1|2`; режимы1/2 требуют ещё
четырёх проверок ошибки после enqueue и восстановления (228/260 runtime cases).
`strata-mimo2-scatter-check fresh.json --benchmark` проверяет bytes/canaries,
границы пакетов и production strides; его microbenchmark не измеряет inference.

Benchmark: `--axis d2d --order 0,2,2,0 --fixed-pack-guards 1`.
На RTX5090 основной ABBA дал9,567→10,259 ток/с (+7,23%); токены/logits exact.
У первого контроля cache немного менялся; два кандидата и последний контроль
имели одинаковые cache/traffic. Отдельный русский ABBA при фиксированном cache8 ГиБ
дал9,131→8,035 ток/с (−12,01%). Поэтому default0 сохранён. Условия и различия между темами — в
[MIMO-15](../../docs/mimo-v2.6-flash/MIMO26_FLASH_SCATTER_COPY.md).
Для `chunk` сравнений со scatter задавать `--fixed-pack-guards 1 --fixed-d2d-batch 2`:
исторические defaults benchmark0 отключают эти оптимизации. При сравнении
старых binaries режим2 допустим только если оба executable его поддерживают.

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
Engine принимает `--trace-skip-graphs N`, чтобы пропустить prefill/прогрев.
Счётчик включает каждый вызов graph compute и не сбрасывается между запросами.
Trace отдельно записывает `cached_d2d`, `ring_d2d` и синхронные `fill_d2d`;
ring marker ставится после ожидания ready event, до фактической D2D-копии.
CUDA events повторно используются между графами. Первый записанный граф
создаёт пул событий; последующие могут его расширять. `span_ms` включает
паузы CPU/driver и не равен сумме времени kernels. Даже время между markers
содержит влияние инструментирования; это диагностика, не замер ускорения.
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

MIMO-16: runner принимает `--expert-cache-mib` (default14336) и `--d2d-batch 0|1|2`
(default0). Остальные параметры transport задаются явно: slab16, tensor batch1,
decay65536, guards1, fill0, early0; они сверяются с ответными метриками probe.
Нужен пересобранный `strata-mimo2-spec-check` с соответствующими полями/CLI.
Старые snapshots MIMO-06/MIMO-16 следует запускать сохранённой вместе с ними
версией runner: нынешний runner передаёт также параметры MIMO-17.
Новый измеренный build: `build-local/mimo2-mtp-q4-current-measured/build-local/mimo2-cuda`.
Q4/head0/depth1/p_min0.7 после оптимизаций:9,015 ток/с; со scatter9,076;
контроль без MTP10,520. Oracle снова расходится на18-м токене, MTP serving off.
[Условия и проверки](../../docs/mimo-v2.6-flash/MIMO26_FLASH_MTP_Q4_RETEST.md).

MIMO-17: `--target-head-columns 1` включает эксперимент только для выходной
BF16-головы target при2–8 столбцах. Каждый столбец использует обычную GPU
проекцию; F32 активации, веса sidecar и verify/rollback не меняются. Default0.
`--memory-stages` добавляет диагностический prefill8/verify2 до обычного
all-logit warmup8 и записывает глобальную память по этапам. В замере скорости
этот дополнительный прогрев выключен. Сам warmup8 перед заполнением cache
сохраняется в обоих режимах MTP.

Отдельный `strata-mimo2-head-check MODEL` (target сборки при SPEC_PROBE=ON)
проверяет настоящую выходную матрицу на одинаковых синтетических активациях:
1/2/8 столбцов, contiguous/padded inputs, независимый single-column CUDA oracle.
В JSON пишет точность, время GPU graph и глобальную дельту VRAM; последняя
включает влияние других приложений и не является точным размером CUDA pool.
Основной engine не получает этот CLI; MTP serving остаётся выключен.
[MIMO-17: результаты, ограничения и команды](../../docs/mimo-v2.6-flash/MIMO26_FLASH_TARGET_HEAD.md).

Проверены только greedy и один sequence. Stochastic correction, HTTP streaming,
cancellation, context shift и MTP heads2/3 в этот probe не входят. Его скорости
следует сравнивать с его же baseline, а не напрямую с прежним pipeline benchmark.

Проверены23 cache ownership cases,224 runtime cases,96 kernels и132 graph checks,
полный corpus, отмена/recovery и bit-exact logits/IDs. API/profile — следующий
отдельный этап; более широкий pressure/context stress ещё предстоит.
