# Статус внедрения Hy3

Обновлено: **2026-10-06**, `Asia/Yekaterinburg`.
План: [HY3_IMPLEMENTATION_PLAN.md](HY3_IMPLEMENTATION_PLAN.md).

Этот файл хранит проверенный прогресс и следующую задачу. **Полный GGUF Hy3
генерирует через отдельный native pipe engine.** HY3-01 добавил inspector, строгий loader contract,
изолированные CPU oracles, проверку tokenizer/template и исправление MTP load flags.
HY3-02 закрыл P0 числовыми CUDA fixtures и проверками main/MTP graph/state.
HY3-03 добавил синхронный selected-copy runtime, подтвердил точное совпадение
token IDs/logits и измерил baseline на полном GGUF. HY3-04 проверил восстановление после
управляемых отказов и добавил отдельный template/reasoning/tool parser. HY3-05
подключил отдельный профиль к Service, обоим HTTP API и веб-чату; полный GGUF
прошёл tool/result continuation и disconnect/recovery. HY3-06 добавил bounded
GPU matrix cache с exact logits/IDs и сокращением H2D. HY3-07 подключил
bounded pipeline с реальными router IDs, native file readers и CUDA events.
HY3-08 добавил opt-in tensor batching с удержанием cache entries до завершения
копий. Его небольшой speed effect требует расширенных замеров; default остаётся off.
Отдельная prefill policy и MTP-ускорение пока не реализованы.

## Текущее состояние

| Область | Статус |
|---|---|
| Подготовка PREP-01 | DONE: исследование, план и этот статус |
| Репозиторий HY3-08 | `4336d41a7b25b21db80ea5bfa7d9e0d799d0bd14`; изменения HY3-06/07/08 без commit |
| Модель | `H:\models\hy3\Hy3-Q3_K_M-mtp.gguf`, один файл, 127,083 ГиБ |
| Формат | `hy_v3`, GGUF v3, 1298 тензоров, 43 metadata records |
| Состав | 80 основных блоков, 1 MTP; dense0, MoE1..79, MoE80+NextN |
| MTP | Полный GGUF: no_alloc off=0/on=20; синтетический граф, hidden, KV rollback/restore — PASS |
| Candidate dependency | Unsloth `86ebfef2`; CPU и CUDA13.0/sm_120, 3 correctness patches + изолированный runtime patch |
| Inspector/loader contract | DONE: все1298 тензоров, JSON и отрицательные CPU fixtures |
| Tokenizer / template | 2194 / 144 проверки, 0 расхождений с native oracles |
| CUDA admission | 150/150 kernel cases, 86/86 graph/state checks; CPU fallback в GPU-графах отсутствует |
| Native engine | DONE HY3-03: Windows pipe, GPU-only, sync selected-copy, MTP-off |
| Восстановление после отказов | HY3-04: pinned OOM, partial ReadFile/cancel, simulated RAM/VRAM, unload/reload; 32/32 checks |
| Reasoning/tool parser | HY3-04: 17 tests /7327 fragmentation sequences, adapter144/144 native comparisons |
| HTTP / UI | HY3-05 PASS: OpenAI/Anthropic JSON/SSE, tools, cancellation, отдельный профиль и реальный browser chat |
| Кэш / pipeline / sessions | HY3-06 cache и HY3-07 bounded pipeline PASS; sessions TODO |
| Полный GGUF | 6 prompts + 2 повтора в каждом режиме; точные logits/IDs, STOP/recovery PASS |
| Tool/result/answer на полном GGUF | HY3-04 direct pipe; HY3-05 HTTP PASS: `get_temperature({"city":"Paris"})` → локальный результат17 → `17 C`; точные IDs |
| Локальная скорость, токенов/с | pinned **1,219**, native **2,086** на 37 decode-шагов en+code; последовательные наблюдения, не A/B |
| HY3-06 cache sweep, прогретые en+code | cache8 **1,821**; cache-off **1,360/1,377** до/после,37 decode-шагов на пару; около+33% в этом коротком прогоне |
| HY3-07 pipeline sweep, прогретые en+code | readers2 **3,100**, readers1 **2,523**, sync **1,869/1,842**; cache8 во всех вариантах; около+67% в этом коротком прогоне |
| HY3-08 tensor batch, короткий ABBA | off **2,804**, on **2,930** в среднем; около+4,5%, но большой разброс между процессами; default off |
| Настройки baseline | Context2048/batch17/F32 KV, pinned16 МиБ; MTP/cache/pipeline/TF32/fusion off |
| Текущий профиль pipeline | `build-local/hy3-http-batch0/hy3.json`, cache8192 МиБ, readers2, ring4×4 МиБ; batching/MTP off |
| Экспериментальный tensor batch | `build-local/hy3-http-batch1/hy3.json`, те же settings, `--pipeline-batch 1` |
| Сохранённый HY3-07 rollback | `build-local/hy3-baseline-pipeline2/hy3.json` и точный старый exe |
| Сохранённый synchronous rollback | `build-local/hy3-baseline-cache8/hy3.json` и неизменённый exe HY3-06 |
| Следующая задача | **HY3-09: изоляция intermittent resident/native mismatch и профилирование read/admission перед prefill policy** |

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

- HY3-08: причина одного intermittent `mixed/batch=4/native_vs_resident`
  расхождения при выключенном pipeline. Native/pinned совпали; повторные
  полные runtime runs и32 focused resident/native comparisons прошли.
  Исходный FAIL сохранён, проблема не считается исправленной.
- Полный checksum весов и соответствие опубликованному AngelSlim файлу.
- Независимый полноразмерный CPU/F32 oracle: full-model parity проверен между
  двумя путями доставки одного native GPU graph; GPU-resident oracle — на fixtures.
- Flash Attention, F16 KV, production контекст длиннее проверенных 362 prompt tokens.
- Полноразмерная generation-проверка low/high и stochastic sampling через HTTP;
  эти effort modes проверены template fixtures и scripted HTTP, реальный corpus — no_think.
- Kernel-level GPU timeline, физический SSD traffic, длительные рандомизированные A/B, настоящий
  driver OOM/device-loss recovery и внешнее давление памяти. Pinned OOM и
  RAM/VRAM availability injection проверены; Source-read bytes не равны SSD traffic.
- MTP acceptance/скорость, speculative accept/reject driver, stochastic sampling
  и production sessions. Синтетические main/MTP KV rollback/restore проверены.
- Установка через общий setup, финальные быстрые defaults и full-model регрессии
  других моделей; отдельный Hy3-профиль, HTTP/web chat и CPU-регрессии уже проверены.

## Таблица этапов

| Этап | Статус | Условие перехода |
|---|---|---|
| PREP-01 | DONE | Проверены заголовок/исходники; создана документация |
| P0 — contract/dependency/oracles | DONE HY3-01/HY3-02 | Header/contract/oracles, CUDA kernels, synthetic main/MTP graph и KV |
| P1 — sync GPU baseline | P1.1–P1.3 DONE; P1.4 controlled failures PASS | Полный GGUF/parity/метрики готовы; настоящий driver OOM и внешний pressure не проверены |
| P2 — tokenizer/template/API | DONE HY3-04/HY3-05 в проверенном объёме | Native parity, request adapters, оба API, tools, cancel/recovery, профиль и web smoke |
| P3 — cache/pipeline | Cache HY3-06, bounded pipeline HY3-07 и opt-in tensor batching HY3-08 готовы в проверенном объёме | Prefill policy, host-tier/SSD counters, расширенные A/B и диагностика intermittent fixture mismatch остаются |
| P4 — sessions/context | TODO | Fresh vs restored/shifted parity и bounded memory |
| P5 — native MTP | TODO | Correctness, rollback и выигрыш относительно оптимизированного off |
| P6 — profile/release checks | Частично: experimental profile/CPU regressions HY3-05 | Остались финальные замеры/defaults и release checks |

`DONE` относится только к указанному объёму. `IN PROGRESS` не означает,
что модель уже может генерировать. Не закрывать P0 по факту успешного чтения header.

## Точка продолжения: HY3-09

1. Изолировать исходный mixed batch4 resident/native FAIL HY3-08. Воспроизводить
   исходную последовательность F32→mixed и смену batch/contexts; при расхождении
   новый checker сохраняет полные logits. Сверить промежуточные тензоры и CUDA
   graph reuse, не ослаблять точное сравнение и не объявлять причину по предположению.
2. Сохранить cache8/readers2/chunk4/batch0 как контроль. Разделить время native
   file read, cache admission/allocator и ожидание producer; CPU суммы двух
   readers не считать elapsed time или физическим SSD traffic.
3. Отдельно измерить prefill admission policy: не смешивать изменение eviction
   с batching. Повторить чередующиеся запросы в стабильных условиях, включая
   длинный prefill, exact logits/IDs, tools и STOP/recovery. MTP пока off.

### Архивная точка перед HY3-08

1. Сохранить HY3-07 cache8/readers2/chunk4 как контрольный вариант. Проверить
   доставку всех выбранных matrices одного tensor с одним fence вместо fence
   на каждую матрицу и cache fill. Не перезаписывать scheduler scratch до
   завершения прежних потребителей; держать cache pins до drain, включая ошибки.
2. На F32/mixed/wide fixtures проверить exact bytes/logits, cache hit/miss,
   eviction, частичные ошибки, STOP/recovery. Затем повторить полный GGUF
   и HTTP tools с exact IDs. MTP и sessions оставить off.
3. Измерить batching отдельно от изменения admission/cache policy. Сравнить
   warm English/code и длинный prefill; контролировать RAM/VRAM95%.
4. CUDA events HY3-07 показали лишь небольшой H2D/compute overlap. Не объяснять
   весь выигрыш +67% этим overlap: read/H2D concurrency и host overhead требуют
   отдельного профилирования. Физический SSD traffic всё ещё не измерен.

### Архивная точка перед HY3-07

1. Подключить bounded `backends/common/expert_pipeline.hpp` по примеру Step
   к Hy3 cache miss path. Сохранить synchronous cache8 как контрольный вариант.
2. Начать с двух staging slots и одного reader; H2D/read overlap включать
   только с явными CUDA events. До eviction/перезаписи scratch завершать
   потребителей; сохранять mappings/cache pins до полного drain.
3. Проверить exact bytes/logits/IDs на mixed quants и полном GGUF, cold/warm
   cache, partial read/copy failure, cancel и следующий запрос. Потом A/B
   одинакового corpus, включая HTTP tools. MTP и sessions оставить off.
4. По отдельности измерить tensor batching и prefill admission. В HY3-06
   cap12/16 ГиБ не ускорили прогретую пару относительно8; полезно проверить
   CUDA allocation overhead и timeline, не считать каждый лишний байт VRAM
   полезным кэшем. Разделение реального SSD traffic и OS cache hits ещё TODO.

### Архивная точка перед HY3-06

1. Добавить bounded GPU matrix cache в selected-copy путь, по образцу Step,
   с ключом model generation/layer/tensor/expert/type и явным ограничением VRAM.
   Начать с MTP-off, synchronous transport и нулевого кэша как контрольного варианта.
2. Проверить точные bytes/logits/greedy IDs на F32 и mixed fixtures, batch1/17,
   hit/miss/eviction, cancel/recovery и unload/reload. Старые ссылки кэша
   не должны переживать смену model generation.
3. Считать GPU hit/miss, bytes from file/RAM и реальные H2D bytes отдельно;
   подтвердить сокращение H2D при повторе одинакового corpus полного GGUF.
   Не выдавать уменьшение source-read bytes за физический SSD traffic.
4. Перебрать cap8/12/16 ГиБ только после runtime clamp с учётом global95%,
   non-routed8,604 ГиБ, KV/graph/scratch и внешней нагрузки. Сначала parity,
   затем повторные A/B скорости; после этого — P3.2 async pipeline.

### Архивная точка перед HY3-05

1. Добавить OpenAI/Anthropic request adapters в `serve/hy3.py`: call/result IDs,
   text blocks, `no_think/low/high`, `preserved_thinking`, точные ошибки на
   unsupported options. Определять старт reasoning по фактическому Hy3 prefix.
2. Подключить Hy3 к Service только через отдельный профиль; точные control-token
   spellings и EOG120025 проверять до запуска. Не менять default других моделей.
3. Проверить оба API через loopback JSON/SSE: tools/result/answer, stop/length,
   usage, cancellation/disconnect и следующий запрос. Затем smoke полного GGUF.
4. Сохранить P1 baseline для P3 кэша/конвейера. Настоящий driver OOM/device loss
   и внешний pressure остаются отдельными непроверенными режимами; тестовая
   имитация доступной памяти не считается hardware stress test.

### Архивная точка перед HY3-04

1. Закрыть оставшиеся P1.4 failure paths: управляемые ошибки reader/allocations,
   достижение бюджета без реального заполнения всей RAM, отмена/drain, следующий
   запрос и выгрузка. Отчёт должен различать ожидаемый отказ и повреждение runtime.
2. Добавить потоковый reasoning/tool parser Hy3 по P2.3: разрывы тегов и UTF-8,
   несколько вызовов, JSON arguments, malformed/incomplete output и round-trip.
3. Подключить отдельный `serve/hy3.py` к проверенному pipe: оба API, JSON/SSE,
   usage/finish_reason, отмена и повтор. Начальный bind `127.0.0.1`.
4. Сохранить этот P1 baseline для дальнейшего P3: GPU expert cache и overlap
   должны уменьшать измеренные **4,865 ГиБ H2D на decode-шаг**, сохраняя parity.
   Проверять бюджет перед каждым тестом; чужие процессы не останавливать.

### Архивная точка перед HY3-03

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
В HY3-02 добавлены числовые GPU проверки; в HY3-03 — generation engine и full-model baseline.

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

### HY3-03 / P1.1–P1.3, часть P1.4 — 2026-10-06 — Полный GGUF и sync baseline

**Основание:** Strata `51125bdbc03f700608584d650e94666bec18c16b`, рабочее дерево
чистое в начале итерации; изменения HY3-03 пока без commit. Тот же GGUF и
header/template hashes. RTX5090, driver581.80, 125,555 ГиБ доступной системе RAM,
MSVC19.44.35222, CUDA13.0.48, sm_120, Release/Ninja. Dependency — Unsloth
`86ebfef2`, отдельная сборка `build-local/hy3-cuda`.
[Manifest с generated source hashes](HY3_RUNTIME_BUILD_MANIFEST.json).

**Что изменено:**

- `strata-hy3.exe`: stdin/stdout pipe ENC/GEN/STOP/QUIT, native sampler,
  progress и request metrics. Каждый GEN начинает с чистого KV и sampler;
  session ID обеспечивает изоляцию, но не сохраняет KV между запросами.
- Native contract проверяет metadata, все1298 тензоров, main/MTP boundary,
  shapes/quants/ranges до загрузки весов. MTP-off исключает блок80.
  Non-routed weights находятся на GPU; математика CPU запрещена scheduler audit.
- Новый путь selected ranges: native file read через общий `expert_file.hpp`
  → pinned16 МиБ → GPU. GPU fence перед перезаписью staging; не более одного
  чтения в полёте. Проверяются размеры/offsets и MMQ padding до512 байт.
  Whole-expert copies отклоняются до копирования. Кэша и overlap пока нет.
- Loader использует demand mmap без предварительного чтения всего файла.
  Existing GLM `HostWorkingSetBudget` ограничивает mapped resident pages
  по system target93%, учитывая другие процессы. Global RAM/VRAM guard95%; VRAM
  читается через общий NVML helper по PCI identity, а не process view CUDA/WDDM.
- Отдельный model checker сохраняет prompt IDs, logits hashes, метрики,
  memory samples, проверяет сравнение режимов, повтор, отмену и свежий запрос.
  Исправлено закрытие тестового pipe после принудительного завершения child:
  broken pipe больше не скрывает исходную ошибку и отчёт memory monitor.
- Добавлены явные CMake header dependencies: локальный MSVC с русскими include
  diagnostics пропускал rebuild после изменения header. Архив dependency,
  исходный third_party и действующие engine/profile других моделей не менялись.

**Проверки:**

| Проверка | Результат |
|---|---|
| Native preallocation admission полного GGUF | PASS; blocks81/vocab120832, точные main/MTP bytes, weight allocations0 |
| Runtime: resident/native/pinned, bytes и logits | **19/19 PASS**, exit0; F32/mixed, batch1/4/17, transfer >16 МиБ |
| Pipe: greedy/seeded repeat, malformed input, STOP/QUIT и повтор | **22/22 PASS**, exit0 |
| Полный GGUF: native selected-copy vs pinned-file | **PASS**, exit0; 6 prompts + 2 повтора на режим, все IDs и logits побитово одинаковы |
| Полный GGUF: STOP → свежий запрос, unload/reload | PASS в обоих режимах; process exit0, memory monitor без ошибок |
| CUDA kernel regression с runtime patch | **150/150 PASS**, exit0 |
| Synthetic main/MTP graph/state regression | **86/86 PASS**, exit0 |
| Python contract/tokenizer/Step/GLM + pipe cleanup regression | **32 tests PASS** |
| CPU CTest version checks | **2/2 PASS** |

Артефакты: [runtime](HY3_RUNTIME_VALIDATION.json), [pipe](HY3_PIPE_VALIDATION.json),
[полный GGUF](HY3_MODEL_VALIDATION.json),
[kernel regression](HY3_RUNTIME_KERNELS_REGRESSION.json),
[graph regression](HY3_RUNTIME_GRAPH_REGRESSION.json).
Полный отчёт содержит SHA-256 проверенного engine
`d6fd9dbffd4985feea1b925d58dcb3550c4b580e687ca07e550bf6742d540600`.
Raw logits и stderr остаются в `build-local/hy3-tests/model-bounded`.

**Числовая оговорка:** с включёнными candidate CUDA fusions fixture F32/batch4
дал resident/native max_abs **0,00105551**; pinned/native при этом совпал.
[Диагностический FAIL](HY3_RUNTIME_FUSION_DIAGNOSTIC.json) сохранён.
С `GGML_CUDA_DISABLE_FUSION=1` все19 runtime checks проходят с точными bytes/logits.
Конкретная fusion ещё не изолирована; runtime принудительно выключает fusions,
TF32 и оставляет `GGML_OP_OFFLOAD_MIN_BATCH=1`. P0 graph callbacks меняют условия
планирования и не заменяют эту проверку без evaluation callbacks.

**Измерения полного GGUF:** template `no_think`, greedy temperature0,
context2048, batch/ubatch17, F32 KV, FA off, MTP/cache/pipeline/fusions off.
Сначала native, затем pinned; OS file cache не очищался, порядок не чередовался.
Это correctness-baseline с наблюдениями времени, **не контролируемый speed A/B**.

| Запрос | Prompt / output tokens¹ | Native TTFT, с | Pinned TTFT, с | Native decode, токенов/с² | Pinned decode, токенов/с² |
|---|---:|---:|---:|---:|---:|
| Русский, 2+2 | 34 / 2 | 16,032 | 10,144 | 1,828 | 1,330 |
| Английский, пять простых чисел | 25 / 15 | 6,157 | 8,955 | 2,005 | 1,305 |
| Китайский, столица | 24 / 3 | 5,246 | 8,538 | 2,194 | 1,299 |
| Python, сложение двух чисел | 28 / 24 | 6,156 | 9,953 | 2,139 | 1,172 |
| Числа, 123+456 | 27 / 2 | 5,230 | 11,043 | 2,248 | 1,235 |
| Длинный prompt, цвет лодки | 362 / 2 | 143,773 | 134,579 | 1,044 | 1,283 |

¹ Включая EOS, reasoning выключен. Ответы: `4`, `2, 3, 5, 7, 11.`, `北京。`,
функция `add`, `579`, `blue`. Это проверка этих запросов, не оценка качества модели.
² `decode_steps / decode_forward_seconds`: первый token получен из prefill
и не включён, последующий EOS включён. В коротких ответах всего один-два decode
шага. Для en+code вместе37 шагов: native **2,086**, pinned **1,219 токена/с**.
Повторы русского после длинного запроса/отмены: native1,179/1,239,
pinned1,316/1,320; все logits совпали с первым русским ответом.

Load до READY: native **2,914 с**, pinned **4,822 с**; demand loading означает,
что сюда не входит чтение всех routed weights. Prefill длинного prompt:
native143,770 с /2,518 токена/с, pinned134,576 с /2,690 токена/с.
Каждый decode-шаг переносит около **4,865 ГиБ H2D** выбранных экспертов.
Для pinned source-read bytes равны H2D bytes; ОС может обслужить чтения из
file cache, поэтому это не измерение физического SSD traffic.

| Sampled memory peak³ | Native | Pinned |
|---|---:|---:|
| Global physical RAM, ГиБ | 117,094 | 36,006 |
| Global VRAM, МиБ | 14 022 | 14 336 |
| Process working set, ГиБ | 99,032 | 9,257 |
| Process private commit, ГиБ | 12,635 | 12,651 |
| Собственный pinned staging, МиБ | 0 | 16 |

³ Global значения включают другие процессы, RAM опрашивается примерно раз0,5с,
GPU раз2с; короткий пик может быть пропущен. Указанный staging — только новый
transport buffer, не сумма всех внутренних driver allocations. mmap virtual size
не равен working set. Оба прогона уложились в95%; неиспользованная VRAM оставлена
для будущего полезного expert cache. Рабочий набор native ограничен93%-target,
иначе mapped pages при длинном prompt вытесняют доступную RAM.

**Воспроизведение:** configure/build и команды в
[README](../../backends/hy3/README.md#windows-synchronous-baseline).
Финальные каталоги: `runtime-bounded`, `pipe-bounded`, `model-bounded`,
`runtime-kernels`, `runtime-graph` внутри `build-local/hy3-tests`.
Full-model команда:

```powershell
python tools/check_hy3_model.py --engine build-local/hy3-cuda/bin/strata-hy3.exe --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --cuda-bin build-local/cuda-13.0/bin/x64 --output-dir build-local/hy3-tests/model-new --predict 24
```

**Не закрыто:** injected OOM/read-error recovery, внешний pressure, контекст
длиннее362 prompt tokens, F16 KV/FA, HTTP/UI/parser, persistent sessions,
expert cache/pipeline и speculative MTP. Новый синхронный transport пока не даёт
ускорения в этом прогоне; его назначение — проверенная основа для P3.
Оптимальные defaults ещё не выбраны. **Следующий шаг:** HY3-04, оставшиеся
P1.4 failure/recovery fixtures, затем P2 parser/API с сохранением этого baseline.

### HY3-04 / P1 recovery, P2.2–P2.3 — 2026-10-06 — Отказы, parser и tool round-trip

**Основание:** тот же HEAD `51125bdbc03f700608584d650e94666bec18c16b` с
незакоммиченными изменениями HY3-03; они сохранены. Та же Release/Ninja сборка,
MSVC19.44.35222, CUDA13.0.48/sm_120, RTX5090/driver581.80 и checkpoint.
[Build manifest](HY3_RECOVERY_BUILD_MANIFEST.json). Скоростные настройки
и архитектура вычислений не менялись: MTP/cache/pipeline/fusion off, F32 KV.

**Отказы runtime и найденная ошибка:**

- Добавлено13 проверок к прежним19. Pinned allocation возвращает injected
  `cudaErrorMemoryAllocation` до upload; после отказа тот же context повторяет
  исходные logits. На следующем selected read после успешного H2D временно
  заменяется handle только синтетического fixture: настоящий Win32 `ReadFile`
  отказывает, handle восстанавливается, запрос повторяется с точными logits.
- Отмена после частичного upload, simulated RAM/VRAM availability0, заведомо
  невозможный GPU reserve, пустой registry после unload и reload — PASS.
  Неполная копия не публикуется как успешная. Test caps только уменьшают
  доступность памяти, не отключают реальные лимиты и не заполняют RAM/VRAM.
  Эти функции не доступны через CLI, env или пользовательский pipe.
- Тест RAM admission сначала завершил процесс с **exit3221226356 /0xc0000374**.
  Throwing helpers были объявлены `extern "C"`; при MSVC `/EHsc` такой вызов
  считается не бросающим исключение, что нарушало нужный unwind path.
  Переведены на C++ linkage. Тот же тест теперь возвращает ожидаемый отказ,
  следующий запрос и выгрузка проходят. [Диагностика до исправления](HY3_RECOVERY_PRE_FIX.json)
  хранит exit code и binary hash; процесса хватило только до RAM pressure,
  поэтому отдельного `runtime-report.json` у этого неудачного прогона нет.
- [Финальный runtime report](HY3_RECOVERY_VALIDATION.json): **32/32 PASS**, exit0.
  Partial ReadFile/cancel возвращали управление примерно за1 мс в tiny fixture;
  это не гарантия latency при задержанном SSD I/O. Registry/source lifetime
  и reload проверены. [Pipe regression](HY3_RECOVERY_PIPE_VALIDATION.json): **22/22 PASS**.

Реальное исчерпание VRAM внутри driver, device loss и внешнее давление памяти
не воспроизводились. Успешная инъекция pinned OOM не доказывает восстановление
после любого CUDA allocation failure. Прочие модели/shared transport не менялись.

**Template и output parser:**

- Новый [serve/hy3.py](../../serve/hy3.py) и точный fixture10223 bytes.
  Renderer проверяет SHA-256, принимает native `no_think/low/high`, правила
  replay/raw-last-assistant/training/fallback, нормализует JSON-string arguments
  перед `.items()` без изменения исходных messages. Неизвестные опции отвергаются.
- Потоковый parser разбирает reasoning и namespace `:opensource`, сохраняет
  raw strings и whitespace; typed arguments проверяет как конечный JSON нужного
  базового типа. Duplicate keys/arguments, неизвестные functions, ошибочные и
  незавершённые группы остаются текстом. Весь tool group проверяется до выдачи
  первого вызова; tool markup внутри reasoning не становится действием.
- Проверены разрывы каждого тега и UTF-8, несколько calls, JSON с delimiter
  strings внутри кавычек, stop в reasoning/tool, перекрывающиеся stop sequences,
  финальные неполные теги, повторный finish и ограничение group buffer1 МиСимвол.
  Нативный формат не экранирует raw values; неоднозначная пара value delimiter
  и следующего key/call delimiter в строках history отвергается. Полная JSON
  Schema validation не добавлена; raw strings задаются string schemas.
- [17 CPU tests /7327 fragmentation sequences](HY3_PARSER_VALIDATION.json) — PASS.
  Новый runtime renderer [совпал с native Jinja и BPE](HY3_ADAPTER_TEMPLATE_VALIDATION.json)
  во всех **144** случаях. Python regression: **100 tests PASS**, включая
  Step/GLM/DeepSeek adapters, Hy3 contract/tokenizer и pipe cleanup.

**Проверка полного GGUF:** [tool dialogue report](HY3_TOOL_DIALOGUE_VALIDATION.json).
`tools/check_hy3_dialogue.py` использует полный read-only checkpoint, native
pipe и локальный фиксированный result stub; HTTP и внешние tools не вызываются.
Template `no_think`, greedy temperature0, pinned sync16 МиБ, context2048/batch17.

1. User просит температуру в Paris через `get_temperature`. Prompt211 токенов,
   output19 с EOS. Модель выдала ровно один полный вызов с `{"city":"Paris"}`;
   parser вернул typed tool event без постороннего content.
2. В history добавлен этот call и локальный result
   `{"city":"Paris","temperature_c":17}` с тем же call ID. Prompt253 токена,
   output3 с EOS: **`17 C`**, без повторного tool call.
3. Для обоих фактических prompts строки и token IDs совпали с compiled
   native template/tokenizer oracles. Оба finish=`stop`, engine exit0,
   memory monitor без ошибок. Это тест протокола на фиктивной температуре,
   а не запрос реальной погоды.

Prefill75,647/96,621с; generation13,463/1,581с соответственно. Это единичная
correctness-проверка, не новый benchmark или ускорение. Sampled global peak:
RAM **33,437 ГиБ**, VRAM **14 258 МиБ**; process private commit12,648 ГиБ.
Монитор RAM≈0,5с/GPU≈2с может пропустить короткий пик; лимит95% не пересечён.
Проверенный engine SHA-256:
`f5cf4dd25611a9173e3cb5d2b4828204babbb959b71f73b99fbb46edb02180c7`.
Исторические HY3-03 reports/binary hashes сохранены и относятся к прежней сборке.

**Воспроизведение**, после обычной CUDA build из README:

```powershell
python tools/check_hy3_cuda.py --build build-local/hy3-cuda --kind runtime --cuda-bin build-local/cuda-13.0/bin/x64 --output-dir build-local/hy3-tests/recovery-new
python -m unittest serve.test_hy3 serve.test_step35 serve.test_glm5next serve.test_deepseek tools.test_hy3_gguf tools.test_hy3_tokenizer tools.test_hy3_engine tools.test_step35_tokenizer tools.test_glm5next_tokenizer
python tools/check_hy3_template.py --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --oracle build-local/hy3-oracles/bin/strata-hy3-template.exe --tokenizer-oracle build-local/hy3-oracles/bin/strata-hy3-tokenizer.exe --runtime-adapter --output docs/hy3/HY3_ADAPTER_TEMPLATE_VALIDATION.json
python tools/check_hy3_dialogue.py --engine build-local/hy3-cuda/bin/strata-hy3.exe --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --cuda-bin build-local/cuda-13.0/bin/x64 --template-oracle build-local/hy3-oracles/bin/strata-hy3-template.exe --tokenizer-oracle build-local/hy3-oracles/bin/strata-hy3-tokenizer.exe --output-dir build-local/hy3-tests/dialogue-new
```

Финальные локальные каталоги: `runtime-recovery-03`, `pipe-recovery-01`,
`dialogue-01` внутри `build-local/hy3-tests`. Неудачные recovery-01/02 сохранены.

**Следующий шаг:** HY3-05 — P2.4/P2.5, request adapters и отдельный профиль,
подключение к Service/HTTP, JSON/SSE/tool round-trip/cancellation. Parser импортируется
отдельно и пока не включён в сервер. UI, sessions, cache/pipeline и MTP остаются TODO.

### HY3-05 / P2.4–P2.5 — 2026-10-06 — Service, HTTP и веб-чат

**Основание:** HEAD `9995b3fd3395cd839869a23a72ccc12ff56a544f`, изменения
этого этапа без commit. C++ engine не менялся, SHA-256
`f5cf4dd25611a9173e3cb5d2b4828204babbb959b71f73b99fbb46edb02180c7`.
Header/template/dependency — те же, что в HY3-04. Текущие Python/JS hashes,
список regression modules и результаты сохранены в
[HY3_HTTP_VALIDATION.json](HY3_HTTP_VALIDATION.json). Старые HY3-04 parser
и model reports сохранены как исторические результаты прежнего Python-кода.

**Изменено:** OpenAI/Anthropic request adapters в `serve/hy3.py`, регистрация
архитектуры в Service, определение начального reasoning по реальному prefix,
проверка special token spellings/types/metadata до engine startup. На этом
tokenizer только EOS120025 завершает генерацию; PAD/placeholder не обрывают ответ.
Tools сохраняют call/result IDs и порядок history; строковые OpenAI arguments
разбираются в конечный JSON object до Jinja. Media, hard reasoning budgets,
чужие template options и некорректные arguments получают HTTP400 до engine.

Режим по умолчанию — `no_think`, доступны `low/high` и `preserved_thinking`.
В Anthropic disabled означает no_think, enabled/adaptive — high;
`output_config.effort` уточняет effort, затем применяются явные template kwargs.
Противоречивые thinking/effort отвергаются. Общие настройки не перезаписывают
явный Anthropic thinking. Другие модели сохраняют прежние ветви нормализации.
В веб-настройках Hy3 показывает Off/Low/High и скрывает Medium/Max.

`tools/prepare_hy3_profile.py` создаёт tokenizer и `hy3.json` только в новом
каталоге, проверяет header и identity сборки. Рабочий локальный профиль:
`build-local/hy3-http-p2/hy3.json`. Настройки: host127.0.0.1, context2048,
batch17, F32 KV, synchronous pinned16 МиБ, temperature0, MTP/cache/pipeline off.
`fit_max_tokens=true` уменьшает только лимит ответа, не обрезает prompt.
Это experimental baseline; ускоренные defaults ещё не подбирались.

**CPU и регрессии — PASS:**

- **276 unittest tests**, 0 failures/errors/skips: Hy3 parser/adapters/profile,
  loopback JSON/SSE и регрессии Step, GLM, DeepSeek, общего Service, lifecycle,
  structured output, detokenizer и tokenizer. Точный список22 modules — в отчёте.
- В Hy3 loopback проверены оба API × JSON/SSE × три effort режима; Unicode,
  usage/EOG, tools/result/answer, stop/length, HTTP400 и следующий запрос,
  disconnect в очереди/prefill/reasoning/partial tool и восстановление.
  Эти сочетания используют scripted engine, а не полный GGUF.
- **144/144** native template/token-ID comparisons для текущего adapter:
  [HY3_HTTP_TEMPLATE_VALIDATION.json](HY3_HTTP_TEMPLATE_VALIDATION.json).
- Два Node checks веб-настроек: Hy3/GLM/Step/Qwen и существующий GLM/Qwen.
  В исходном расширенном прогоне обнаружены два устаревших ожидания в
  `serve/test_glm5next_profile.py`: не учтены существующие pipeline/MTP args
  и ошибочно предполагалось, что memory flags стоят последними. Исправлены
  только тесты; повтор всех276 прошёл. GLM exporter/defaults не менялись.

**Полный GGUF, production Service/StrataEngine, loopback HTTP — PASS:**
[HY3_HTTP_MODEL_VALIDATION.json](HY3_HTTP_MODEL_VALIDATION.json).

1. Вопрос `Сколько будет 2 + 2? Ответь одной цифрой.` прошёл через обе API
   формы и оба transport режима:34 prompt tokens,2 output tokens с EOS,
   ответ4. Все четыре prompts/outputs совпали по IDs.
2. OpenAI JSON вернул один `get_temperature({"city":"Paris"})`,211 prompt /
   19 output tokens с EOS. Его ID и локальный результат17 переданы через
   Anthropic SSE;253 prompt /3 output tokens с EOS, ответ **`17 C`**.
   Оба фактических prompts и полные output IDs точно совпали с HY3-04
   direct-dialogue reference на том же бинарном файле. Внешних tools не было.
3. Клиент отключён во время prefill778-token запроса после первого PP17;
   runtime вернул `cancel`, очередь освободилась через **3,753с** после
   закрытия сокета. Следующий34-token запрос воспроизвёл те же IDs/ответ4.
   Итого7 успешных генераций и1 отмена; engine exit0.
4. /health, /v1/models, /settings и HTML веб-чата отдали HTTP200 и Hy3 capabilities.

На RTX5090, Windows/driver581.80,128GB установленной RAM, указанном GGUF и
greedy no_think: tool call prefill **76,178с**, generation **15,396с**;
continuation prefill **94,639с**, generation **1,604с**. Это correctness-прогон
без performance A/B. Sampled global peak RAM **35,403 ГиБ**, VRAM **14 360 МиБ**,
ошибок95% monitor нет. Интервалы RAM≈0,5с/GPU≈2с не исключают короткие пики.
Сохранён прежний P1 baseline; ускорение этим этапом не заявляется.

**Реальный веб-чат — PASS:** server.main с этим CLI-профилем, browser UI,
temperature0/max_tokens32; вопрос выше дал видимый ответ4 и состояние Idle.
Reasoning не попал в обычный текст. В интерфейсе Off выбран по умолчанию,
Low/High доступны. Запрос/ответ, память и exit codes:
[HY3_HTTP_UI_VALIDATION.json](HY3_HTTP_UI_VALIDATION.json);
[снимок](HY3_HTTP_UI_SMOKE.png). Browser tab и тестовые server/engine закрыты,
оба exit0. UI-скорость на двух output tokens не используется как benchmark.

**Воспроизведение:** команды экспорта, запуска, HTTP и CPU checks находятся в
[backend README](../../backends/hy3/README.md#experimental-http-profile).
Локальный уже созданный профиль запускается так:

```powershell
python -m serve.server --engine strata --config build-local/hy3-http-p2/hy3.json --port 8094
python tools/check_hy3_http.py --profile build-local/hy3-http-p2/hy3.json --reference docs/hy3/HY3_TOOL_DIALOGUE_VALIDATION.json --output-dir build-local/hy3-tests/http-new
python -m unittest serve.test_hy3 serve.test_hy3_http tools.test_hy3_profile
node serve/test_hy3_settings_ui.cjs
python tools/check_hy3_template.py --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --oracle build-local/hy3-oracles/bin/strata-hy3-template.exe --tokenizer-oracle build-local/hy3-oracles/bin/strata-hy3-tokenizer.exe --runtime-adapter --output docs/hy3/HY3_HTTP_TEMPLATE_VALIDATION.json
```

Server и checker запускать по очереди: checker сам создаёт engine и временный
loopback listener. Raw logs: `build-local/hy3-tests/http-01`, UI-runner/report:
`build-local/hy3-tests/ui-01` и `ui-smoke-runner.py` рядом. Для повтора выбирать
новый output directory. Все перечисленные финальные команды завершились с exit0.

**Не закрыто:** full-model low/high/stochastic corpus, production sessions,
F16 KV/Flash Attention, длинный контекст, driver OOM/external pressure,
общий installer и speed defaults. Следующий шаг **HY3-06 / P3.1** — bounded
GPU expert cache: exact logits/IDs, hit/miss/eviction и снижение H2D;
затем P3.2 async pipeline и контролируемые A/B.

### HY3-06 / P3.1 — 2026-10-06 — Bounded GPU matrix cache

**Основание:** HEAD `4336d41a7b25b21db80ea5bfa7d9e0d799d0bd14`, изменения
этого этапа без commit. Новый engine SHA-256:
`8da251920bf0d034f52978192ea1e10fc412896193057355502d981e974858b9`.
В private build добавлен patch marker `hy3-bounded-matrix-cache`;
Unsloth source, основные correctness patches, GGUF и template не менялись.
Предыдущий бинарный файл HY3-05 сохранён в
`build-local/hy3-baseline-p2/strata-hy3.exe` (SHA начинается с `f5cf4dd2`).

**Изменено:** `backends/hy3/cache_runtime.inc` подключает неизменённый
`step35::ExpertCache` к Hy3 scheduler. Ключ состоит из поколения модели,
ID зарегистрированного тензора и expert ID; registry проверяет name/layer,
type, shape, strides, живое file mapping и фактический offset. Новый begin
и release освобождают старые entries. Cache hit копирует веса D2D в scratch,
miss использует прежний file→pinned→GPU путь, затем заполняет cache slot.
Реальный MMQ padding следующего эксперта включён в запись, до512 байт.
Каждая копия завершена до eviction/reuse; overlap пока отсутствует.

Аллокации учитываются с округлением64 КиБ; controller использует decaying
frequency, bounded victim scan и повторное использование равных allocations.
Global NVML free/total ограничивает budget с reserve5%+256 МиБ, RAM сохраняет
target93% и guard95%. Cache OOM пропускает admission; ошибка заполнения
сбрасывает записи. При pressure cache сначала уменьшается, а неизбежный
недостаток памяти отказывает до compute. Реальный driver OOM/device loss
не заявлен как проверенный.

CLI получил `--expert-cache-mib 0..16384`, только для pinned mode; default0
сохраняет baseline. Новый профиль `build-local/hy3-http-cache8/hy3.json`
явно задаёт8192 МиБ. Остальные параметры: context2048, batch17, F32 KV,
temperature0/no_think, pinned16 МиБ; MTP/pipeline/fusion/TF32 off.
Действующие профили других моделей не изменялись. В JSON метриках отдельно
записаны cache hit/miss/eviction/reuse/oom, resident/budget, source/H2D,
D2D hits и D2D cache fill. Source-read bytes не являются SSD traffic.

**Проверки — PASS:**

- [HY3_CACHE_ALLOCATION_VALIDATION.json](HY3_CACHE_ALLOCATION_VALIDATION.json):
  **25/25** CUDA allocator/policy cases, включая simulated pressure/OOM,
  pins, reuse и131200 сравнений victim decisions. Проверяется тот же Step
  controller, собранный внутри изолированного Hy3 build.
- [HY3_CACHE_RUNTIME_VALIDATION.json](HY3_CACHE_RUNTIME_VALIDATION.json):
  **100/100** runtime cases. Включены прежние32 и68 новых: F32/mixed,
  batch1/17, cold/warm/new generation, byte/logit parity, eviction/reuse,
  OOM bypass, незавершённое заполнение, partial read/cancel и recovery,
  simulated RAM/VRAM trim. На полностью прогретых synthetic fixtures H2D=0,
  все observer-проверки GPU bytes совпали с source.
- [HY3_CACHE_PYTHON_VALIDATION.json](HY3_CACHE_PYTHON_VALIDATION.json):
  **44 unittest tests** и6 CLI admission checks. Profile exporter сохраняет
  выбранный cap и отвергает недопустимый до создания файлов; Hy3 HTTP/parser
  регрессии прошли. Этот отчёт также хранит hashes изменённых исходников.
- [HY3_CACHE_MODEL_VALIDATION.json](HY3_CACHE_MODEL_VALIDATION.json):
  полный GGUF, **20 генераций** в sweep0/8/12/16/0 ГиБ: English/code дважды
  в каждом процессе, fresh KV и greedy, cap24 output tokens. Все полные
  logit hashes и token IDs совпали с историческим native baseline HY3-03.
  Каждый nonzero cap дополнительно прошёл STOP и повтор English с exact
  logits/IDs: ещё3 отмены и3 успешных восстановления. Все engine exit0,
  sampled95% monitors без ошибок.
- [HY3_CACHE_HTTP_VALIDATION.json](HY3_CACHE_HTTP_VALIDATION.json):
  полный GGUF с cache8 через production Service/StrataEngine. OpenAI и
  Anthropic JSON/SSE дали ответ `4`; tools вернули `get_temperature(Paris)`
  и продолжение `17 C`, prompt/output IDs точно совпали с HY3-04. Всего
  7 завершённых запросов. Disconnect во время prefill запроса из 778 токенов
  дал DONE cancel и снятие busy через 3,679 с; следующий запрос воспроизвёл
  эталон. Sampled peak RAM **38,237 ГиБ**, VRAM **26 701 МиБ**; guard без
  ошибок, engine exit0, временный loopback listener закрыт.
- [HY3_CACHE_PIPE_VALIDATION.json](HY3_CACHE_PIPE_VALIDATION.json):
  **22/22** synthetic pipe lifecycle checks с cap8 МиБ: A/B/A, повтор seeded
  sampling, ошибки запросов, STOP/recovery, изоляция session keys и QUIT
  во время запроса. Engine exit0; memory guard без ошибок.

**Короткий capacity sweep**, RTX5090/driver581.80, Windows,128GB установленной
RAM, `Hy3-Q3_K_M-mtp.gguf`. В таблице второй проход English/code в каждом
процессе:15+24 output tokens с EOS, **37 decode-шагов** после первых токенов
из prefill. Скорость =37 / сумма decode_forward_seconds. Prompt25/28,
KV очищается на каждый запрос; GPU cache/history сохраняются внутри процесса.
OS file cache не сбрасывался; это последовательный sweep с cache-off до/после,
не рандомизированное длительное сравнение. Точные расчёты:
[HY3_CACHE_BENCHMARK.json](HY3_CACHE_BENCHMARK.json).

| Запрошенный cap, ГиБ | Cache resident в конце, ГиБ | Decode, ток/с | H2D на decode-шаг, ГиБ | Sampled peak VRAM, МиБ | Sampled peak RAM, ГиБ |
|---:|---:|---:|---:|---:|---:|
| 0, до sweep | 0 | 1,360 | 4,865 | 13 917 | 37,463 |
| 8 | 8,000 | **1,821** | 3,077 | 26 519 | 39,719 |
| 12 | 10,416 | 1,803 | 2,835 | 30 230 | 34,862 |
| 16 | 10,394 | 1,762 | 2,773 | 30 233 | 37,939 |
| 0, после sweep | 0 | 1,377 | 4,865 | 14 003 | 33,761 |

Cache8 дал **+33,1% decode throughput** относительно объединённых cache-off
повторов и **−36,8% H2D** на этом corpus. Полное время прогретой пары с prefill:
37,391с вместо45,020/44,610с без кэша. Cap12/16 сократили H2D сильнее, но
не ускорили эту пару; runtime ограничил реальный кэш примерно10,4 ГиБ.
Размеры всех CUDA/driver allocations не равны сумме payload cache slots:
общая VRAM выше. Причина разницы отдельно не профилировалась. Guard использует
NVML total−free (системное reserve может отличаться от nvidia-smi memory.used
в sampled peak колонке); для больших cap request snapshots около30 720 МиБ.

Для следующего этапа выбран **cache8**: лучший результат этого sweep и меньше
VRAM. Это не универсально оптимальный размер; длинные/другие темы и concurrent
memory load потребуют дополнительных замеров. Автоматический default exporter
остался0; быстрый вариант включён только в отдельном экспериментальном профиле.

HTTP-прогон также показал, что ускорение decode не означает сокращения TTFT.
Tool call: prefill **92,011 с**, generation **10,247 с**; continuation:
prefill **102,437 с**, generation **1,594 с**. В историческом HY3-05 без кэша
prefill был 76,178/94,639 с. Это разные последовательные прогоны, не отдельный
контролируемый prefill A/B, но улучшение длинного prefill пока не подтверждено.
Текущий путь разбивает ranges на матрицы и синхронизирует каждую доставку;
batching и отдельную prefill admission policy следует измерить до выбора
общих speed defaults.

**Воспроизведение:** build/CUDA/model commands — в
[backend README](../../backends/hy3/README.md#experimental-http-profile).
Локальный запуск созданного cache8-профиля:

```powershell
python -m serve.server --engine strata --config build-local/hy3-http-cache8/hy3.json --port 8094
python tools/check_hy3_http.py --profile build-local/hy3-http-cache8/hy3.json --reference docs/hy3/HY3_TOOL_DIALOGUE_VALIDATION.json --output-dir build-local/hy3-tests/cache-http-new --allow-engine-change
```

Server и HTTP checker запускать по очереди: checker создаёт свой listener.
Для экспорта на новую сборку использовать новый directory и
`tools/prepare_hy3_profile.py ... --expert-cache-mib 8192`. Откат доставки —
cap0 на этой же сборке, без смены модели и template. Старые profile/checksum
HY3-05 остаются историческими; helper проверяет checksum текущего exe.
Raw run directories: `cache-unit-01`, `cache-runtime-01`, `cache-model-01`,
`cache-http-01`, `cache-pipe-01` в `build-local/hy3-tests`. Последняя проверка
запущена через `python -X utf8 build-local/check-hy3-cache-pipe.py`, который
вызывает существующий `tools.check_hy3_engine.fixture` с cap8 МиБ и Monitor.
Для её повтора нужен новый output directory. Сборка и перечисленные финальные
проверки — exit0.

**Не закрыто:** async read/H2D/compute overlap, tensor batching, отдельная
prefill admission policy, физические SSD/RAM-hit counters, full-model
stochastic/long-context corpus и hardware device loss. Следующая задача
**HY3-07 / P3.2** — bounded pipeline с доказанным timeline overlap и exact
parity относительно synchronous cache8. MTP остаётся off.

### HY3-07 / P3.2 — 2026-10-06 — Bounded expert pipeline

**Основание:** HEAD `4336d41a7b25b21db80ea5bfa7d9e0d799d0bd14` с изменениями
HY3-06; новый этап без commit. Engine SHA-256:
`68288b37ee631bea8e8b45f9e13aa33615f5d1002d54963b994542670de1e0d3`.
Patch marker `hy3-bounded-pipeline`. Dependency, GGUF, template, граф вычислений
и математические kernels не менялись. Engine HY3-06 с SHA `8da25192…` и профиль
с его точным checksum сохранены в `build-local/hy3-baseline-cache8`.

**Изменено:** Hy3 использует неизменённые `common/expert_pipeline.hpp`,
Step cache controller/PlanPins и Step GPU event tracer. Отдельные
`pipeline_runtime.inc`/`pipeline_sched.inc` строят план gate/up/down по реальным
router IDs и будущему порядку scheduler splits только для того же route.
Registry проверяет generation, tensor metadata и live file mapping; веса
никогда не предсказываются. Кэш удерживает будущие hits до завершения плана.

CLI `--pipeline-readers 0..2`, `--pipeline-chunk-mib 4|8|16`; default readers0.
Nonzero readers требуют pinned mode и nonzero cache cap. Стартовый рабочий
вариант использует **4 слота по4 МиБ**, то есть16 МиБ pinned RAM и16 МиБ GPU
ring. В плане первоначально предлагались2 слота; общий transport имеет фиксированные4,
поэтому сохранён его проверенный механизм без изменения общих backend-файлов.
Синхронный host buffer освобождается при включении pipeline. Перед allocation
проверяются общий95%-budget и запас; ненужная RAM искусственно не заполняется.

Miss читает зарегистрированный файл → pinned slot → независимый H2D stream;
вычисления получают данные D2D в обычной точке scheduler. Ready/used CUDA events
защищают slot ownership. Перед перезаписью scratch и после каждой доставки/fill
пока остаются fences. Plan scope завершает reader/H2D работу на каждом выходе
из graph, включая STOP и ошибки. Отказ producer требует пересоздания ring перед
следующим запросом; mappings/pins не освобождаются до drain. Незавершённый fill
инвалидирует кэш. Ошибки device loss/настоящий driver OOM не моделировались.

**Проверки — PASS:**

- [HY3_PIPELINE_RUNTIME_VALIDATION.json](HY3_PIPELINE_RUNTIME_VALIDATION.json):
  **246/246**, включая прежние100 и146 новых. F32/mixed, batch1/17, readers1/2,
  cold/warm/eviction, точные GPU bytes/logits, bounded16+16 МиБ ring,
  STOP после частичной доставки, реальный invalid-handle ReadFile до старта
  workers, injected submission failure после двух H2D, incomplete fill,
  auto recovery, simulated RAM/VRAM pressure. Wide matrix проверяет несколько
  chunks. После drain queued/reader-owned=0. Режимы8/16 МиБ chunk не сравнивались
  на полном GGUF; здесь рабочий chunk4 МиБ.
- [HY3_PIPELINE_PYTHON_VALIDATION.json](HY3_PIPELINE_PYTHON_VALIDATION.json):
  **45 CPU tests +7 CLI admission checks**, hashes исходников и engine.
- [HY3_PIPELINE_PIPE_VALIDATION.json](HY3_PIPELINE_PIPE_VALIDATION.json):
  **22/22**, synthetic cache8 МиБ/readers2: A/B/A, seeded sampling, отказ
  некорректных команд, STOP/recovery, session isolation и QUIT во время запроса.
- [HY3_PIPELINE_MODEL_VALIDATION.json](HY3_PIPELINE_MODEL_VALIDATION.json):
  полный GGUF, cache8192 МиБ, readers0/1/2/0. По два прохода English/code в каждом
  процессе: **16 генераций**, затем4 отмены и4 восстановления. Все полные
  token IDs и logit hashes совпали с историческим HY3-03 baseline. Каждый
  процесс exit0; memory guards без ошибок; незавершённых pipeline payloads нет.
- [HY3_PIPELINE_HTTP_VALIDATION.json](HY3_PIPELINE_HTTP_VALIDATION.json):
  полный GGUF, cache8/readers2/chunk4, production Service/StrataEngine.
  **7 завершённых запросов:** OpenAI/Anthropic JSON/SSE, math, tool call и
  продолжение `17 C`, точные prompt/output IDs. Disconnect на prefill запроса
  из778 токенов дал DONE cancel и снятие busy через **2,118 с**, следующий запрос
  восстановил эталон. Engine exit0, временный loopback listener закрыт.
  Sampled peak RAM **36,408 ГиБ**, VRAM **26 923 МиБ**, guard без ошибок.

**Измерения:** RTX5090/driver581.80, Windows,128GB установленной RAM,
context2048/batch17/F32 KV, temperature0/no_think, MTP/fusion/TF32 off.
В таблице прогретая пара English/code, prompt25/28, output15/24 с EOS:
**37 decode-шагов**; первые токены получены из prefill и в decode throughput
не входят. Fresh KV на запрос, cache/history сохраняются. Trace выключен.
Один последовательный sweep с synchronous повторами до/после, OS file cache
не сбрасывался. Это короткое сравнение, не универсальный optimum или длинный
рандомизированный benchmark. Расчёты: [HY3_PIPELINE_BENCHMARK.json](HY3_PIPELINE_BENCHMARK.json).

| Readers | Decode, ток/с | Prefill пары, с | Полное время пары, с | H2D/decode, ГиБ | Peak VRAM, МиБ | Peak RAM, ГиБ |
|---:|---:|---:|---:|---:|---:|---:|
| 0, до | 1,869 | 17,245 | 37,066 | 3,077 | 26 481 | 34,103 |
| 1 | 2,523 | 12,590 | 27,279 | 3,080 | 26 514 | 33,335 |
| 2 | **3,100** | **9,261** | **21,224** | 3,080 | 26 541 | 33,942 |
| 0, после | 1,842 | 16,958 | 37,068 | 3,077 | 26 506 | 35,750 |

Readers2 дали **+67,0% decode throughput** против объединённых synchronous
повторов (1,856 ток/с) и **−42,7% времени пары**. H2D почти одинаков: улучшение
не связано с дополнительным сокращением объёма весов. С pipeline residency
составил8191,188 МиБ против8192 без него; PlanPins немного меняют решения
eviction, поэтому это сравнение полных способов доставки, не только числа
потоков. Peak — sampled global usage, включая другие процессы.

Отдельный HTTP correctness-прогон: tool call prompt211/output19 — prefill
**47,634 с**, generation **6,218 с**; tool result prompt253/output3 — prefill
**59,206 с**, generation **0,871 с**. В предыдущем HY3-06 cache8 без pipeline
prefill был92,011/102,437 с. Это последовательные наблюдения разных прогонов,
не самостоятельный рандомизированный HTTP A/B; основной speed вывод выше
основан на одинаковом English/code corpus с synchronous повторами до/после.

**CUDA trace — PASS:** [HY3_PIPELINE_TRACE_VALIDATION.json](HY3_PIPELINE_TRACE_VALIDATION.json).
Отдельный полный English-запрос, readers2/cache8,4 instrumented graphs,
exact logits/IDs, engine exit0. H2D/compute overlap по событиям:
**1,056;0,753;0,095;0,433 мс** (сумма2,337 мс). Первые два graphs — prefill,
следующие — decode. Это небольшое перекрытие; оно не объясняет весь прирост
скорости. Измеряются интервалы на реальных streams, включая scheduling gaps,
а не kernel occupancy. CPU read/wait суммы также не доказывают GPU overlap.
Диагностическое время не смешивается с benchmark. Tiny warm fixture дал0 overlap,
что тоже отражено в raw fixture trace.

**Профили:** созданы отдельные `build-local/hy3-http-pipeline1/hy3.json` и
`build-local/hy3-http-pipeline2/hy3.json`. По этому sweep выбран readers2/chunk4,
cache8; автоматический default exporter остаётся pipeline0/cache0.
Для rollback имеется точный synchronous exe/profile HY3-06 выше.

```powershell
python -m serve.server --engine strata --config build-local/hy3-http-pipeline2/hy3.json --port 8094
python tools/check_hy3_http.py --profile build-local/hy3-http-pipeline2/hy3.json --reference docs/hy3/HY3_TOOL_DIALOGUE_VALIDATION.json --output-dir build-local/hy3-tests/pipeline-http-new --allow-engine-change
```

Server и checker запускать по очереди; checker создаёт собственный loopback
listener. Build и model sweep команды — в [backend README](../../backends/hy3/README.md).
Raw directories: `pipeline-runtime-01`, `pipeline-model-01`, `pipeline-pipe-01`,
`pipeline-trace-01`, `pipeline-http-01` в `build-local/hy3-tests`. Local runners:
`build-local/check-hy3-pipeline-pipe.py`, `check-hy3-pipeline-trace.py`,
`summarize-hy3-pipeline.py`; для повтора выбирать новые output directories.
Сборка и перечисленные финальные проверки завершились с exit0. Source/profile
checksums сверены с протестированным exe; `git diff --check` прошёл.

**Не закрыто:** tensor batching, отдельная prefill admission policy, длинный
контекст/stochastic full-model corpus, physical SSD/OS cache counters,
hardware device loss, production sessions и MTP. Следующая задача **HY3-08 / P3.3** —
сократить число fences на доставку tensor с сохранением exact parity.

### HY3-08 / P3.3 — 2026-10-06 — Группировка копий одного tensor

**Основание:** HEAD `4336d41a7b25b21db80ea5bfa7d9e0d799d0bd14`, изменения
HY3-06/07/08 без commit. Engine SHA-256:
`e24dc96ac7edfb308d0574b07a2e0f29d80f202387ef7dfc640ff105b0c0078a`.
Patch marker `hy3-tensor-batch-copy`. Dependency, GGUF, template, математические
kernels, common pipeline и Step cache controller не изменены в этом этапе.
Старый exe HY3-07 с SHA `68288b37…` и отдельный профиль с точным checksum
сохранены в `build-local/hy3-baseline-pipeline2`.

**Изменено:** `--pipeline-batch 1` ставит копии всех выбранных experts одного
scheduler input tensor в consumer stream, затем выполняет один delivery fence.
Перед перезаписью scratch остаётся синхронизация с прежними вычислениями.
PlanPins защищают cache hits; новые cache destinations удерживаются отдельными
pins до завершения всех копий. Cache fill следует за ring→scratch copy в том же
stream. При STOP или исключении stream сначала завершается, затем освобождаются
pins; незавершённый fill инвалидирует кэш. Общий ring остаётся4×4 МиБ.

Режим требует nonzero pipeline readers. Native default — `--pipeline-batch 0`;
exporter включает его только по явному `--pipeline-batch`. Добавлены counters
числа tensor batches, delivery fences, максимального числа удерживаемых fills
и CPU wall time batch. Delivery fences не включают синхронизацию перед scratch,
graph-end и возможную внутреннюю синхронизацию CUDA allocator.

**Проверки и обнаруженное ограничение:**

- [HY3_BATCH_RUNTIME_VALIDATION.json](HY3_BATCH_RUNTIME_VALIDATION.json):
  финальный полный набор **472/472 PASS**, включая226 новых проверок.
  F32/mixed/wide, batch1/17, cold/warm/eviction, точные GPU bytes/logits,
  несколько pending fills, ReadFile/submission/fill errors, STOP после двух
  поставленных копий, recovery и injected allocation OOM bypass.
- Первый полный запуск дал **471/472**:
  [HY3_BATCH_INITIAL_RUNTIME_FAILURE.json](HY3_BATCH_INITIAL_RUNTIME_FAILURE.json).
  Старый случай `mixed/batch=4/native_vs_resident` при **выключенном pipeline**:
  max abs `0.016343072056770325` на1728 элементах. Native/pinned совпали точно,
  все новые batch cases прошли. Два следующих полных запуска прошли472/472;
  checksum mixed fixture одинаков. Исходный FAIL сохранён, причина **не найдена**.
- [HY3_BATCH_REFERENCE_VALIDATION.json](HY3_BATCH_REFERENCE_VALIDATION.json):
  **60/60 PASS** —32 resident/native сравнения на4 reloads и28 native repeats
  с fresh KV, batch4/pipeline0. Focused probe не воспроизводит всю исходную
  историю F32→mixed и смену contexts, поэтому не доказывает устранение ошибки.
  Runtime checker теперь сохраняет оба полных массива logits при расхождении.
- [HY3_BATCH_PYTHON_VALIDATION.json](HY3_BATCH_PYTHON_VALIDATION.json):
  **46 CPU tests +6 CLI admission checks PASS**, hashes engine/исходников.
- [HY3_BATCH_PIPE_VALIDATION.json](HY3_BATCH_PIPE_VALIDATION.json):
  **22/22 PASS**, synthetic batch1/cache8 МиБ/readers2: A/B/A, seeded sampling,
  invalid commands, STOP/recovery, session isolation и QUIT во время запроса.
- [HY3_BATCH_MODEL_VALIDATION.json](HY3_BATCH_MODEL_VALIDATION.json):
  полный GGUF, порядок batch **0/1/1/0**, по2 English/code прохода:
  **16 генераций +4 отмены +4 восстановления**. Все полные token IDs и logit
  hashes совпали с историческим HY3-03. Каждый процесс exit0; guard без ошибок.
- [HY3_BATCH_LONG_VALIDATION.json](HY3_BATCH_LONG_VALIDATION.json):
  prompt362/output2, batch0/1, в каждом режиме обычный запрос, STOP и повтор.
  Все полные IDs/logit hashes совпали с HY3-03, оба процесса exit0,
  sampled95% guards без ошибок. Подробные времена ниже.
- [HY3_BATCH_HTTP_VALIDATION.json](HY3_BATCH_HTTP_VALIDATION.json):
  **7 завершённых запросов PASS**, production Service/StrataEngine с batch1.
  Math JSON/SSE через OpenAI/Anthropic дали одинаковые prompt/output IDs;
  первый math-запрос задаёт локальный эталон. Tool call/result совпали
  с историческими native IDs: `get_temperature({"city":"Paris"})` → `17 C`.
  Disconnect на prefill778 токенов освободил busy через **2,031 с**;
  следующий math-запрос восстановил эталон. Engine exit0, loopback listener
  закрыт. Peak RAM **36,497 ГиБ**, VRAM **26 415 МиБ**, guard без ошибок.

**Скорость:** RTX5090/driver581.80, Windows,128GB установленной RAM,
cache8192 МиБ/readers2/chunk4, context2048/prompt batch17/F32 KV,
temperature0/no_think, MTP/fusion/TF32 off. Второй English/code проход каждого
процесса: prompt25/28, output15/24 с EOS, **37 decode forwards** без первых
токенов из prefill. Fresh KV, сохранённый cache, OS file cache не сбрасывался.
Raw расчёты: [HY3_BATCH_BENCHMARK.json](HY3_BATCH_BENCHMARK.json).

| Tensor batch | Decode, ток/с | Prefill пары, с | Полное время пары, с | Delivery fences на37 decode | Peak VRAM, МиБ | Peak RAM, ГиБ |
|---|---:|---:|---:|---:|---:|---:|
| 0, до | 3,061 | 9,472 | 21,586 | 73 945 | 26 559 | 34,976 |
| 1, первый | 3,227 | 9,323 | 20,814 | 8 769 | 26 556 | 35,108 |
| 1, второй | 2,683 | 10,851 | 24,670 | 8 769 | 26 817 | 38,116 |
| 0, после | 2,587 | 11,459 | 25,793 | 73 945 | 26 694 | 38,495 |

По объединённым decode durations: **2,804 → 2,930 ток/с (+4,5%)**;
среднее полное время пары **23,690 → 22,742 с (−4,0%)**. Delivery fences
сократились в **8,43 раза**, но передаваемый H2D объём одинаков —3,080 ГиБ
на decode forward. Cache residency в конце8191,188 МиБ у всех вариантов.
Peak — sampled global usage, включая другие процессы, ниже95% лимита.

Разброс скорости между процессами больше среднего выигрыша. К концу sweep
выросли и CPU read time sums; это сумма работы двух readers, не elapsed time
и не физическое SSD время. Причина дрейфа не установлена. Короткий ABBA
подтверждает работу batching, но недостаточен для изменения defaults.

**Длинный prefill:** отдельный последовательный batch0→1, тот же GGUF и
настройки,362 prompt tokens. Первый запрос начинается с пустым GPU cache;
восстановительный запрос использует cache history после STOP и fresh KV.

| Tensor batch | Первый prefill, с | Prefill после STOP, с | Первый request, с | Prefill delivery fences | Peak VRAM, МиБ | Peak RAM, ГиБ |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 121,454 | 136,679 | 122,049 | 512 113 | 30 234 | 42,097 |
| 1 | 105,085 | 102,664 | 105,490 | 5 214 | 30 223 | 44,941 |

Первый prefill с batching занял на13,5% меньше времени. Это одно наблюдение
на вариант без обратного порядка; оно не устанавливает стабильный выигрыш.
Первый prefill передал822,472/822,509 ГиБ через H2D: объём почти одинаков,
но не идентичен. Более долгие pending pins немного меняют cache eviction,
поэтому сравнивается весь режим доставки. Выход всего2 токена с EOS;
decode throughput этого запроса не используется для вывода о скорости.
STOP завершился через5,929/3,576 с; это wall time отменяемого pipe-запроса,
включая его начальный prefill, а не задержка от отправки STOP.

В отдельном HTTP correctness-прогоне tool call prompt211/output19 дал
prefill **48,260 с**, generation **5,895 с**; tool result prompt253/output3 —
prefill **52,560 с**, generation **0,822 с**. HTTP A/B в этом этапе не проводился;
эти наблюдения не заменяют короткое ABBA сравнение выше.

**Профили:** контроль `build-local/hy3-http-batch0/hy3.json`; эксперимент
`build-local/hy3-http-batch1/hy3.json`. Оба используют новый checksummed exe,
cache8/readers2/chunk4; различается только tensor batching. Старые профили,
которые указывают на пересобранный exe с прежним checksum, не обновлялись:
для rollback использовать сохранённый `hy3-baseline-pipeline2/hy3.json`.

```powershell
cmd /c build-local\build-hy3-cache.bat
python -X utf8 tools/check_hy3_cache_model.py --engine build-local/hy3-cuda/bin/strata-hy3.exe --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --reference docs/hy3/HY3_MODEL_VALIDATION.json --cuda-bin build-local/cuda-13.0/bin/x64 --cache-mib 8192 --pipeline-readers 2 --pipeline-batch 0 1 1 0 --output-dir build-local/hy3-tests/batch-model-new
python -X utf8 tools/check_hy3_cache_model.py --engine build-local/hy3-cuda/bin/strata-hy3.exe --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --reference docs/hy3/HY3_MODEL_VALIDATION.json --cuda-bin build-local/cuda-13.0/bin/x64 --cache-mib 8192 --pipeline-readers 2 --pipeline-batch 0 1 --prompts long --repeats 1 --output-dir build-local/hy3-tests/batch-long-new
python -X utf8 tools/check_hy3_http.py --profile build-local/hy3-http-batch1/hy3.json --reference docs/hy3/HY3_TOOL_DIALOGUE_VALIDATION.json --output-dir build-local/hy3-tests/batch-http-new --allow-engine-change
python -m serve.server --engine strata --config build-local/hy3-http-batch1/hy3.json --port 8094
```

Не запускать server и GPU checkers одновременно. Runtime и focused probe
команды — в [backend README](../../backends/hy3/README.md). Raw directories:
`batch-runtime-01/02/03`, `batch-reference-01`, `batch-pipe-01`, `batch-model-01`, `batch-long-01`
и `batch-http-01` в `build-local/hy3-tests`. Для повторов выбирать новые output
directories. Сборка, финальные runtime/model/HTTP проверки завершились с exit0;
первый runtime FAIL описан отдельно выше. Source/profile hashes совпали
с протестированным exe, `git diff --check` прошёл.

**Следующий этап HY3-09:** воспроизвести intermittent resident/native mismatch
с исходной историей тестов и сравнить промежуточные тензоры. Затем измерить
read/admission/allocator costs перед отдельной prefill admission policy.
MTP, production sessions, длинный контекст и длительный рандомизированный
speed benchmark остаются открыты. Tensor batching пока экспериментальный.

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
