# Статус внедрения Hy3

Обновлено: **2026-10-06**, `Asia/Yekaterinburg`.
План: [HY3_IMPLEMENTATION_PLAN.md](HY3_IMPLEMENTATION_PLAN.md).

Этот файл хранит проверенный прогресс и следующую задачу. **Генерация Hy3
в Strata пока не реализована.** HY3-01 добавил inspector, строгий loader contract,
изолированные CPU oracles, проверку tokenizer/template и исправление MTP load flags.
HY3-02 закрыл P0 числовыми CUDA fixtures и проверками main/MTP graph/state.
Генерация, скорость и пиковая память inference полного GGUF не измерялись.

## Текущее состояние

| Область | Статус |
|---|---|
| Подготовка PREP-01 | DONE: исследование, план и этот статус |
| Репозиторий HY3-01 | `9fd4153e372d75dd638eaf90c9c1f1f180dc6057`; результаты этой итерации пока без commit |
| Модель | `H:\models\hy3\Hy3-Q3_K_M-mtp.gguf`, один файл, 127,083 ГиБ |
| Формат | `hy_v3`, GGUF v3, 1298 тензоров, 43 metadata records |
| Состав | 80 основных блоков, 1 MTP; dense0, MoE1..79, MoE80+NextN |
| MTP | Полный GGUF: no_alloc off=0/on=20; синтетический граф, hidden, KV rollback/restore — PASS |
| Candidate dependency | Unsloth `86ebfef2`; CPU и CUDA13.0/sm_120, 3 generated correctness patches |
| Inspector/loader contract | DONE: все1298 тензоров, JSON и отрицательные CPU fixtures |
| Tokenizer / template | 2194 / 144 проверки, 0 расхождений с native oracles |
| CUDA admission | 150/150 kernel cases, 86/86 graph/state checks; CPU fallback в GPU-графах отсутствует |
| Native engine / HTTP / UI | TODO |
| Кэш / pipeline / sessions | TODO |
| Локальная скорость, токенов/с | **Нет измерений** |
| Рекомендуемые настройки | Не выбраны; MTP-off — исходный вариант будущего baseline |
| Следующая задача | **HY3-03: P1 — sync selected-copy engine и первый baseline полного GGUF, MTP-off** |

Созданы изолированный `backends/hy3`, tools/tests и отчёты. В общем Python
tokenizer добавлен `hunyuan-dense`; действующие профили и модель не изменялись.
Регрессии tokenizer Step/GLM и проверка разбиения цифр Qwen прошли.

## Подтверждено при подготовке

- File size **136 454 632 928**, header_end **5 161 421**, data_start **5 161 440**.
  1298 уникальных tensor names; все размеры известны, offsets выровнены,
  диапазоны не пересекаются, последний payload заканчивается на EOF.
- Trunk payload **134 592 752 896 байт (125,349 ГиБ)**:
  routed **125 354 115 072 (116,745 ГиБ)**, остальные **9 238 637 824 (8,604 ГиБ)**.
- MTP payload **1 856 718 592 байта (1,729 ГиБ)**:
  routed **1 717 567 488**, остальные **139 151 104**.
- Несмотря на название файла, routed weights используют IQ3_XXS/IQ4_XS
  и K-quants. Полная таблица типов и архитектуры находится в плане.
- SHA-256 header без padding/payloads:
  `f3307357f0b6ab163f188f7d19ba57d2960ca70a7491b511b23b2745a9500ca9`.
- SHA-256 UTF-8 строки `tokenizer.chat_template`:
  `7fc351fee674c13754656ba7f33a3ca426bfb7231039f48444360a3f3c5ecf3e`.
- Архив `build-local/llama-glm-86ebfef2.tar.gz` — 37 493 950 байт,
  SHA-256 `f8e524b635b726bae74fd8f84bb9249e5b09384c63207f6707c5a3f921acad99`.
  `hy-v3.cpp` в нём и в распакованном GLM candidate совпадает:
  `224921b8ce6f9be02dc1252ef6847a93f81386021b24d7d08eb25444e787d62e`.
- Candidate содержит основной граф и native MTP; основной `third_party`
  не содержит регистрации `hy_v3`. Это результат чтения исходников,
  не результат компиляции или проверки совместимости на весах.
- Windows сообщает **134 813 700 096 байт RAM = 125,555 ГиБ** через
  `GlobalMemoryStatusEx`; NVIDIA сообщает **32 607 МиБ VRAM**, RTX 5090,
  driver581.80. Лимиты95%: **119,277 ГиБ RAM / 30 976,65 МиБ VRAM**.
  Во время подготовки `nvidia-smi` показал 31 463 МиБ занятой VRAM;
  это внешняя нагрузка, не потребление Hy3. Перед тестом проверять заново.

Публичные материалы и ссылки с оговорками о совместимости приведены в плане.
Чужие токены/с не переносить в таблицу локальных результатов.

## Не проверено

- Полный checksum весов и соответствие опубликованному AngelSlim файлу.
- Реальные weight allocations/read и logits полного GGUF;
  no_alloc проверяет его регистрацию, CUDA fixtures используют синтетические веса.
- Flash Attention, F16 KV, числовое поведение на полном размере/длинном контексте.
- Production reasoning/tool parser, API и toolcall round-trip через сервер.
- TTFT, prefill/decode throughput, GPU timeline, peak RAM/VRAM, SSD traffic.
- MTP acceptance/скорость, speculative accept/reject driver, stochastic sampling
  и production sessions. Синтетические main/MTP KV rollback/restore проверены.
- Рабочий профиль, HTTP/web chat, установка и регрессии других моделей.

## Таблица этапов

| Этап | Статус | Условие перехода |
|---|---|---|
| PREP-01 | DONE | Проверены заголовок/исходники; создана документация |
| P0 — contract/dependency/oracles | DONE HY3-01/HY3-02 | Header/contract/oracles, CUDA kernels, synthetic main/MTP graph и KV |
| P1 — sync GPU baseline | TODO | Полный GGUF, logits/greedy parity, memory/throughput report |
| P2 — tokenizer/template/API | Tokenizer и template oracles проверены; API TODO | Production parser, tools, JSON/SSE, cancel/recovery |
| P3 — cache/pipeline | TODO | Bytes/logits parity, overlap, pressure checks и A/B |
| P4 — sessions/context | TODO | Fresh vs restored/shifted parity и bounded memory |
| P5 — native MTP | TODO | Correctness, rollback и выигрыш относительно оптимизированного off |
| P6 — profile/release checks | TODO | Воспроизводимые замеры, defaults и регрессии |

`DONE` относится только к указанному объёму. `IN PROGRESS` не означает,
что модель уже может генерировать. Не закрывать P0 по факту успешного чтения header.

## Точка продолжения: HY3-03

1. Создать `strata-hy3` native pipe engine. Использовать строгий loader contract
   до weight allocations и отдельно собранную зависимость Hy3.
2. Реализовать синхронный selected-copy для routed gate/up/down; dense/shared,
   attention, router и все остальные model matmuls выполнять на GPU.
   Начать с MTP-off, без adaptive cache/pipeline/sessions.
3. Взять bounded reader/pinned transport из `backends/common`, добавить Hy3
   диапазоны и mixed-quants. Учитывать main routed116,745 ГиБ, non-routed8,604 ГиБ;
   весь main payload125,349 ГиБ нельзя одновременно разместить в RAM.
4. Сверить logits/greedy IDs с native-графом на тех же квантах и токенах;
   затем измерить load, TTFT, prefill/decode и H2D bytes на полном файле.
5. Перед запуском проверить свободные RAM/VRAM и соблюдать общий лимит95%,
   не останавливая чужие процессы. Скорость Hy3 ещё не измерена.

### Архивная точка перед HY3-02

1. Сохранить текущий CPU baseline и build manifest. Добавить отдельную CUDA
   конфигурацию `build-local/hy3-cuda`, не переключать working builds GLM/Step.
2. Реализовать P0.4 fixtures для всех local mixed quants и правильных expert
   strides, sigmoid+selection bias, top-8/norm/scale, dense/shared FFN,
   Q/K norm до NeoX RoPE и full GQA. Сравнить с CPU/F32 reference.
3. Проверить, нужны ли correctness patches TF32/routed-strides из Step на
   выбранных CUDA путях; переносить только с source hashes и failing/passing fixture.
4. Проверить main/MTP KV filters, hidden-state convention, rollback/restore
   на маленьком синтетическом Hy3. Существующий load-flags patch уже проверен
   в compiled no_alloc; вычислительную корректность он не подтверждает.
5. После P0.4/P0.5 перейти к P1 sync selected-copy на полном файле. До allocations
   заново проверить свободные RAM/VRAM и соблюдать global95%, не останавливая
   чужие процессы. Inference tokens/s появятся только на этом этапе.

### Архивная точка перед HY3-01

1. Создать `tools/inspect_hy3_gguf.py` поверх существующего `GGUFFile` и
   `tools/hy3_loader_contract.py`. Сохранить `docs/hy3/HY3_INSPECTION.json`:
   полный directory, metadata summaries, bytes по группам, header/template hashes.
2. Проверить контракт **для всех 1298 тензоров**, обязательность dense0,
   MoE1..79, MoE80, NextN80, общих embedding/output/norm. Сравнить с локальным
   `hy-v3.cpp`; не маскировать отсутствующие веса его `NOT_REQUIRED` flags.
3. Добавить CPU fixtures для неверной архитектуры, main/MTP boundary,
   shapes/quants, duplicate/overlap/truncated ranges и 64-bit offsets.
4. Создать изолированный `backends/hy3/CMakeLists.txt`, закрепить проверенный
   archive SHA и loader hash, распаковывать только в собственный build directory.
   Сначала собрать vocabulary-only oracle: это не требует загрузки весов на GPU.
5. Сверить BPE, BOS/EOS/EOG и template. Особое внимание `hunyuan-dense`,
   `:opensource` tokens, `no_think/low/high` и placeholder на ID120026.
6. Обновить этот статус командами, PASS/FAIL, manifest и точкой продолжения.
   Следом — P0.4/P0.5 CUDA fixtures и аудит пропуска MTP, затем P1.

Этот список выполнен в HY3-01 в объёме header/loader/tokenizer/template.
В HY3-02 добавлены числовые GPU проверки; generation engine ещё отсутствует.

## Подтверждённый журнал

### PREP-01 — 2026-10-06 — Инспекция и план

**Действия:** прочитаны инструкции репозитория, план/статус GLM и Step,
общий transport, candidate loader, локальные GGUF metadata/tensor directory.
Проверены архив dependency, GPU total и доступный Windows объём RAM.
Изучены официальный config Tencent, публикация AngelSlim и upstream Hy3 PR.

**Результат:** структурные проверки прошли; main/MTP разделены,
бюджеты и ограничения отражены в плане. Вычисления на GPU не запускались,
tensor payloads не читались и не хэшировались. Веса не скачивались.

**Воспроизведение header-проверки**, PowerShell из корня репозитория:

```powershell
@'
from pathlib import Path
from collections import Counter
import hashlib
import re
from tools.gguf_reader import GGUFFile

p = Path(r'H:\models\hy3\Hy3-Q3_K_M-mtp.gguf')
g = GGUFFile(p)
size = p.stat().st_size
assert g.version == 3 and g.alignment == 32
assert g.metadata['general.architecture'] == 'hy_v3'
assert g.metadata['hy_v3.block_count'] == 81
assert g.metadata['hy_v3.nextn_predict_layers'] == 1
assert len(g.tensors) == len({t.name for t in g.tensors}) == 1298
previous_end = 0
for t in sorted(g.tensors, key=lambda t: t.offset):
    n = t.expected_bytes()
    assert n is not None and n > 0, t.name
    assert t.offset % g.alignment == 0, t.name
    assert t.offset >= previous_end, t.name
    assert g.data_start + t.offset + n <= size, t.name
    previous_end = t.offset + n
assert g.data_start + previous_end == size

groups = Counter()
for t in g.tensors:
    match = re.match(r'blk\.(\d+)\.', t.name)
    scope = 'mtp' if match and int(match[1]) >= 80 else 'main'
    kind = 'routed' if '_exps.weight' in t.name else 'other'
    groups[f'{scope}_{kind}'] += t.expected_bytes()
expected = dict(main_routed=125354115072, main_other=9238637824,
                mtp_routed=1717567488, mtp_other=139151104)
assert dict(groups) == expected, groups
assert size == 136454632928
assert (g.header_end, g.data_start) == (5161421, 5161440)
assert sum(groups.values()) == 136449471488
with p.open('rb') as stream:
    header_sha = hashlib.sha256(stream.read(g.header_end)).hexdigest()
assert header_sha == 'f3307357f0b6ab163f188f7d19ba57d2960ca70a7491b511b23b2745a9500ca9'
template_sha = hashlib.sha256(g.metadata['tokenizer.chat_template'].encode('utf-8')).hexdigest()
assert template_sha == '7fc351fee674c13754656ba7f33a3ca426bfb7231039f48444360a3f3c5ecf3e'
print('PASS header/ranges; bytes:', dict(groups))
print('Tensor types:', dict(Counter(t.type_name for t in g.tensors)))
print('Header SHA-256:', header_sha)
print('Template SHA-256:', template_sha)
'@ | python -X utf8 -
```

Это проверка зафиксированного локального файла. Она намеренно не является
универсальным Hy3 loader contract и не проверяет содержимое матриц.

**Проверка документации:** приведённый Python-код повторно выполнен из этого
Markdown, exit0, `PASS header/ranges`; размеры групп и оба хэша совпали.
Локальные ссылки, парность code fences и отсутствие trailing whitespace — PASS.
Арифметика main+MTP, top-8 H2D и KV оценок — PASS. Это не inference-тесты.

### HY3-01 / P0.1–P0.3, часть P0.5/P2 — 2026-10-06 — Contract и compiled oracles

**Основание:** Strata `9fd4153e372d75dd638eaf90c9c1f1f180dc6057`, локальный
GGUF с тем же header hash. Изолированная Release/Ninja CPU-сборка MSVC
19.44.35222.0 из закреплённого архива Unsloth; CUDA не включалась.
[Build manifest](HY3_BUILD_MANIFEST.json).

**Что изменено:**

- [Inspector](../../tools/inspect_hy3_gguf.py) и
  [loader contract](../../tools/hy3_loader_contract.py): полный tensor directory,
  main/MTP, mixed quants, byte ranges/strides, hashes и проверка обязательных весов.
  Неизвестные архитектурные metadata и неподдержанные layouts отклоняются.
- [Изолированный backend](../../backends/hy3/README.md): tokenizer/template
  oracles и настоящий compiled loader в `no_alloc=true`. Он проверяет,
  что dummy weight buffers нулевые, не читает payloads и не строит inference context.
- Python tokenizer поддерживает `hunyuan-dense` с ordered splits из pinned
  oracle. Добавлен только Hy3-specific пропуск отсутствующего CR символа после BPE.
- Отдельные corpus/checkers для tokenizer, шаблона и регистрации весов;
  CPU fixtures проверяют invalid metadata/shapes/quants, отсутствие MTP-норм,
  bias naming, duplicate/overlap/truncation, >4 ГиБ offsets и защиту output paths.

**Результаты:**

| Проверка | Результат |
|---|---|
| Static admission локального файла | PASS, 1298 тензоров, все диапазоны и формы |
| Python unit/regression tests | PASS, 31 тест, включая Step/GLM и Qwen digit split |
| CTest version checks | PASS, 2/2 |
| Tokenizer vs native oracle | PASS, **2194** случая, 0 несовпадений IDs/decoded bytes |
| Native Jinja vs Python + token IDs | PASS, **144** случая, 0 несовпадений |
| Compiled loader, MTP on | PASS, 1278 main + 20 MTP, точные logical byte totals |
| Compiled loader, MTP off после patch | PASS, те же1278 main, **0 MTP**, dummy weight buffers0 |
| GPU logits / inference / tokens/s | SKIPPED: CUDA runtime ещё не реализован |

**Найденные особенности и исправление:**

1. В локальном vocabulary нет отдельного byte-mapped `\r` (`č`). Native
   tokenizer пропускает его; это не lossless round-trip. Корпус содержит
   244 случая с CR, и отчёт явно учитывает ожидаемую потерю этого символа.
   LF, остальные control bytes и варианты tokenizer других моделей сохранены.
2. Native EOG set — **[120025]**. ID120026 из публичного config здесь placeholder,
   а не дополнительный stop. BOS добавляет шаблон, oracle не дублирует его.
3. Исходный loader при `load_mtp=false` продолжал регистрировать **9** optional
   MTP tensors, **1 740 784 384 байта ≈1,621 ГиБ**. Это подтверждено запуском
   compiled no_alloc, не только чтением исходников.
4. [MtpLoadFlags.cmake](../../backends/hy3/MtpLoadFlags.cmake) генерирует отдельную
   translation unit и передаёт skip flag optional FFN/NextN tensors. Проверка
   повторена на обоих режимах. Trunk byte totals не изменились; MTP-on по-прежнему
   регистрирует все20. Generated source SHA-256:
   `0a40aca60138dde0f30ffc0eae0f8fbcacabebc2d7803afaadf69b580870d60d`.
   Устранено лишнее включение весов в план загрузки; реальная экономия resident
   RAM/VRAM и correctness графа пока не измерялись.

**Артефакты:** [инвентаризация](HY3_INSPECTION.json),
[исходная регистрация](HY3_LOADER_UPSTREAM_VALIDATION.json),
[регистрация после исправления](HY3_LOADER_VALIDATION.json),
[tokenizer validation](HY3_TOKENIZER_VALIDATION.json),
[template validation](HY3_TEMPLATE_VALIDATION.json).
В исходном отчёте поле `status=pass` означает успешную регистрацию в no_alloc,
а не корректный пропуск MTP: его `load_mtp=false` запись сохраняет найденные9 тензоров.

**Воспроизведение:** сборку выполнять в MSVC2022 developer shell с Windows SDK.
Обычный shell первоначально не находил `rc/mt`; среда `vcvars64.bat` исправила
сборку. Смена глобальных настроек Git или установка зависимостей не требовалась.

```powershell
cmake -S backends/hy3 -B build-local/hy3-oracles -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_HY3_ARCHIVE=C:/work/git/my-repos/Strata/build-local/llama-glm-86ebfef2.tar.gz
cmake --build build-local/hy3-oracles --target strata-hy3-tokenizer strata-hy3-template strata-hy3-loader -j 4
ctest --test-dir build-local/hy3-oracles --output-on-failure
python -m unittest tools.test_hy3_gguf tools.test_hy3_tokenizer tools.test_step35_tokenizer tools.test_glm5next_tokenizer
python tools/inspect_hy3_gguf.py --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --output docs/hy3/HY3_INSPECTION.json
python tools/check_hy3_loader.py --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --oracle build-local/hy3-oracles/bin/strata-hy3-loader.exe --output docs/hy3/HY3_LOADER_VALIDATION.json
python tools/check_hy3_tokenizer.py --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --oracle build-local/hy3-oracles/bin/strata-hy3-tokenizer.exe --output docs/hy3/HY3_TOKENIZER_VALIDATION.json
python tools/check_hy3_template.py --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --oracle build-local/hy3-oracles/bin/strata-hy3-template.exe --tokenizer-oracle build-local/hy3-oracles/bin/strata-hy3-tokenizer.exe --output docs/hy3/HY3_TEMPLATE_VALIDATION.json
```

Все финальные команды завершились exit0. Стартовый P0.3 закрыт только для
CPU oracles; CUDA-сборка, граф, KV/rollback, native generation engine, API и
измеренный профиль остаются следующими этапами. Не включать Hy3 в production
список моделей по результатам этих проверок.

**Следующий шаг:** HY3-02 — CUDA fixtures и числовое сравнение графа;
подробная точка продолжения приведена выше.

### HY3-02 / P0.4–P0.5 — 2026-10-06 — CUDA kernels, граф и KV/MTP state

**Основание:** тот же repository HEAD `9fd4153e372d75dd638eaf90c9c1f1f180dc6057`,
изменения HY3-01/HY3-02 ещё без commit. Архив и исходный loader не изменялись.
Полный локальный GGUF в этой итерации не читался. Тестовые GGUF создаются
в новом каталоге `build-local/hy3-tests` и содержат только синтетические веса.

**Сборка:** отдельный `build-local/hy3-cuda`, MSVC19.44.35222.0, CUDA13.0.48,
Ninja/Release, architecture120, RTX5090/driver581.80.
Сохранён [CUDA manifest](HY3_CUDA_BUILD_MANIFEST.json); CPU baseline сохранён.
Generated patches: `hy3-mtp-load-flags`, `cuda-f32-mmf-respect-tf32-override`,
`cuda-routed-input-strides`. Каждый патч проверяет hash исходника и создаёт
собственную translation unit в build directory.

**Что изменено:**

- `check_kernels.cpp`: F32 и все7 квантованных типов локальной модели,
  192 experts/top8, batch1/4/17, обычный/routed matmul, broadcast/per-route input,
  compact/padded strides. Scalar reference использует распакованные веса и
  double accumulation; CPU reference запускается явно, CUDA fallback отсутствует.
  Router проверяет sigmoid, selection bias, выбранные IDs, нормировку исходных
  вероятностей и множитель2,826.
- `synthetic_hy3.hpp` и `check_graph.cpp`: два основных блока и один MTP,
  dense0/MoE1/MTP2, width256, head_dim128, Q4/KV1, experts16/top8, FF256,
  vocab64, full F32 KV, context2048, flash attention выключен.
  F32 и mixed GGUF используют native loader и граф Hy3. Mixed включает
  Q8_0/Q6_K, основной IQ3_XXS/IQ4_XS и MTP Q3_K/Q4_K.
- Scalar проверки Q/K RMSNorm→NeoX RoPE, causal full GQA, dense/shared/routed
  SwiGLU и суммы shared+routed. MTP: enorm/hnorm, concat(e,h), eh_proj,
  общий output head, post-final-norm hidden; проверена передача hidden
  основного графа и повторного MTP шага.
- Main KV содержит только слои0/1, MTP KV — только слой2. В обоих контекстах
  save→A→restore→B→restore→A и removal/replay дают побитово одинаковые
  logits/hidden. MTP-off не загружает тензоры блока2; основной граф совпадает
  побитово с MTP-loaded вариантом. GPU scheduler audit: **0 CPU math nodes**.
- `tools/check_hy3_cuda.py`: новый output directory, hashes бинарника/manifest,
  логи и sampled memory; контроль общей RAM/VRAM95% с остановкой только
  созданного тестового процесса. В проверках запас памяти не исчерпан.

**Исправления подтверждены до/после:**

| CUDA-конфигурация | Результат | Причина |
|---|---|---|
| Без двух CUDA patches, TF32 override=0 | 136/150, exit1 | Custom F32 MMF игнорирует override; 14 расхождений |
| Только strict-F32 | 146/150, exit1 | 4 padded routed F32 cases неверно вычисляют физический шаг входа |
| Strict-F32 + routed-strides | **150/150, exit0** | Совпадение с reference в заданных допусках |
| Native main/MTP graph/state | **86/86, exit0** | F32 64 checks, mixed22; no CPU fallback |

Артефакты: [до CUDA patches](HY3_CUDA_KERNELS_UNPATCHED.json),
[только strict-F32](HY3_CUDA_KERNELS_STRICT_ONLY.json),
[исправленные kernels](HY3_CUDA_KERNELS_VALIDATION.json),
[graph/state](HY3_CUDA_GRAPH_VALIDATION.json).
Все содержат параметры, hashes бинарника/manifest, допуски, ошибки и memory samples.

**Числовые результаты:** максимальная ошибка F32 logits CPU/CUDA —
`8,35e-7` main и `3,58e-7` MTP. Для mixed logits — `0,02416` main и
`0,01142` MTP, NMSE соответственно `1,693e-4` и `7,188e-5`.
Допуски заданы до первого запуска: mixed abs≤0,03 logits/≤0,12 hidden,
NMSE≤2e-3; CPU и CUDA используют разные способы квантизации активаций.
Это числовое сравнение с допуском, не побитовый parity полного checkpoint.
Restore/replay и MTP-off/on требуют точного совпадения и проходят с F32/mixed.

**Память:** в финальном kernel прогоне sampled global VRAM максимум2872 МиБ,
global RAM23,104 ГиБ; process peak working set567,676 МиБ. В финальном graph
прогоне —3120 МиБ,21,423 ГиБ и565,133 МиБ соответственно. Global показатели
включают другие процессы; опрос RAM раз в0,5с, GPU раз в2с может пропустить
краткий пик. Эти числа не оценивают память inference полного GGUF.

**Воспроизведение:** команды CUDA configure/build и оба запуска приведены в
[backends/hy3/README.md](../../backends/hy3/README.md#cuda-admission).
Локальные финальные каталоги: `build-local/hy3-tests/kernels-patched` и
`build-local/hy3-tests/graph-final`. Старые неудачные прогоны сохранены.
Обе CUDA опции включены по умолчанию в новой CUDA-конфигурации; для существующего
cache README явно задаёт ON. Регрессии: **31 Python test PASS**, CPU CTest **2/2**.

**Не закрыто:** полный GGUF, pipe engine, выбранные копии экспертов,
Flash Attention/F16 KV, длинный контекст, speculative accept/reject driver,
API, production sessions и производительность. Синтетический MTP graph не
является готовым MTP-ускорением Strata. **Токенов/с пока нет.**

**Следующий шаг:** HY3-03/P1 — sync selected-copy baseline полного GGUF,
MTP-off, с logits/greedy parity и измерением скорости/памяти.

## Правила обновления

- Каждая следующая запись имеет ID `HY3-01`, `HY3-02`, дату, исходный commit,
  dirty state/patch hashes, команды, exit code, результаты и нерешённые вопросы.
- Хранить JSON/logs с параметрами модели, dependency и шаблона; в статусе давать
  ссылки на реально созданные отчёты. Планируемые пути не выдавать за готовые.
- Время измерять отдельно для load/prefill/decode/request; tokens/s считать
  по принятым выходным токенам. Draft throughput не выдавать за output throughput.
- Записывать sampler/seed, prompt/generated counts, context/KV, batch/ubatch,
  MTP depth/acceptance/repair, cache/readers/chunks, GPU/driver/build flags.
- Память: global physical RAM и available, process commit/working set,
  pinned buffers, GPU used/total, cache allocation и peak. mmap file size
  не считать ни resident RAM, ни гарантированно свободной RAM.
- Оптимизация считается принятой после correctness и повторных A/B с одинаковыми
  условиями. Изменение cache capacity из-за MTP учитывать отдельным сравнением.
- Явно указывать пропущенные проверки. При неудаче оставить рабочий baseline
  и записать конкретную следующую проверку; не менять `TODO` на `DONE` по намерению.

## Шаблон следующей записи

```text
### HY3-NN / Pn.m — YYYY-MM-DD HH:MM, Asia/Yekaterinburg — Название
Основание: commit, dirty state, dependency/patch SHA, модель/header/template hash.
Изменено: файлы и поведение.
Команды: точные команды сборки и проверки.
Проверки: PASS/FAIL/SKIPPED, exit codes, fixtures/model corpus.
Результаты: correctness, tokens/s/TTFT, память, cache/H2D/SSD, условия и повторы.
Артефакты: ссылки на созданные JSON/logs.
Не закрыто: ошибки, ограничения и непроверенные режимы.
Следующий шаг: одна конкретная задача и критерий приёмки.
```
