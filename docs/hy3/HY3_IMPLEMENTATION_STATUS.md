# Статус внедрения Hy3

Обновлено: **2026-10-06**, `Asia/Yekaterinburg`.
План: [HY3_IMPLEMENTATION_PLAN.md](HY3_IMPLEMENTATION_PLAN.md).

Этот файл хранит проверенный прогресс и следующую задачу. **Поддержка Hy3
в Strata пока не реализована.** Выполнено чтение локального GGUF header,
проверены диапазоны тензоров, изучены исходники и внешние первоисточники.
Генерация, скорость и пиковая память модели не измерялись.

## Текущее состояние

| Область | Статус |
|---|---|
| Подготовка PREP-01 | DONE: исследование, план и этот статус |
| Репозиторий | `e26522b4dd5cd5f40034accb6ec3c2b7b6e3f00f`, рабочее дерево уже содержало изменения |
| Модель | `H:\models\hy3\Hy3-Q3_K_M-mtp.gguf`, один файл, 127,083 ГиБ |
| Формат | `hy_v3`, GGUF v3, 1298 тензоров, 43 metadata records |
| Состав | 80 основных блоков, 1 MTP; dense0, MoE1..79, MoE80+NextN |
| MTP | Веса есть в файле, 1,729 ГиБ; исполнение не проверено |
| Candidate dependency | Локальный архив Unsloth `86ebfef2`, содержит `hy-v3.cpp`; сборки Hy3 нет |
| Inspector/loader contract | Одноразовая инспекция выполнена; постоянные tools/tests ещё TODO |
| Native engine / HTTP / UI | TODO |
| Кэш / pipeline / sessions | TODO |
| Локальная скорость, токенов/с | **Нет измерений** |
| Рекомендуемые настройки | Не выбраны; MTP-off — исходный вариант будущего baseline |
| Следующая задача | **HY3-01: P0.1–P0.3, инспектор, contract и vocabulary-only oracle** |

Созданы только два документа в `docs/hy3`. Код, действующие профили и локальная
модель этой работой не изменялись. Существующие изменения GLM/Step/DeepSeek
не являются результатами внедрения Hy3.

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
- Полный автоматический loader contract; actual tensor loading и MTP-off
  allocation policy, включая routed/shared матрицы блока80.
- Сборка Hy3 на Windows/CUDA, вычислительная корректность mixed quants,
  router и Q/K norm/RoPE; соответствие logits/hidden states oracle.
- Token IDs на corpus, EOG-набор, template rendering, reasoning/tool parser.
- TTFT, prefill/decode throughput, GPU timeline, peak RAM/VRAM, SSD traffic.
- MTP acceptance, скорость, rollback, stochastic sampling, session restore.
- Рабочий профиль, HTTP/web chat, установка и регрессии других моделей.

## Таблица этапов

| Этап | Статус | Условие перехода |
|---|---|---|
| PREP-01 | DONE | Проверены заголовок/исходники; создана документация |
| P0 — contract/dependency/oracles | IN PROGRESS, только подготовительное P0.1 | Постоянный inspector, строгий contract, сборка и числовые fixtures |
| P1 — sync GPU baseline | TODO | Полный GGUF, logits/greedy parity, memory/throughput report |
| P2 — tokenizer/template/API | TODO | Oracle fixtures, tools, reasoning, JSON/SSE, cancel/recovery |
| P3 — cache/pipeline | TODO | Bytes/logits parity, overlap, pressure checks и A/B |
| P4 — sessions/context | TODO | Fresh vs restored/shifted parity и bounded memory |
| P5 — native MTP | TODO | Correctness, rollback и выигрыш относительно оптимизированного off |
| P6 — profile/release checks | TODO | Воспроизводимые замеры, defaults и регрессии |

`DONE` относится только к указанному объёму. `IN PROGRESS` не означает,
что модель уже может генерировать. Не закрывать P0 по факту успешного чтения header.

## Точка продолжения: HY3-01

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

Hy3-specific CLI и перечисленных JSON/oracle файлов пока нет. Не запускать
GLM/Step engine с новым GGUF как проверку поддержки. Полные модельные тесты
проводить после оценки свободного global memory budget; чужие процессы не останавливать.

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
