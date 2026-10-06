# Статус внедрения MiMo-V2.6-Flash-RL

Обновлено: **2026-10-06**, `Asia/Yekaterinburg`.
План: [MIMO26_FLASH_IMPLEMENTATION_PLAN.md](MIMO26_FLASH_IMPLEMENTATION_PLAN.md).

Файл хранит фактический прогресс и точку продолжения.
**MiMo backend в Strata пока не создан, генерация не запускалась.**
Проверены локальные GGUF header/ranges и доступные исходники.
Наличие `mimo2` в llama.cpp не означает готовую интеграцию в Strata.

## Текущее состояние

| Область | Статус |
|---|---|
| PREP-01 | DONE: исследование, план и этот файл |
| Исходная ревизия Strata | `1979cb6607c07faec22ab3c195c8c8e1e29d78f9`, с существующими изменениями |
| Модель | `H:\models\mimo-v2.6-flash\MiMo-V2.6-Flash-RL-GSQ-RCO-3.5bit.gguf` |
| Файл / архитектура | 125,712 ГиБ / `mimo2`, 472 тензора |
| Trunk | 48 слоёв: dense0 +47 MoE; 9 full +39 SWA attention |
| MTP | **Нет в локальном GGUF**; `nextn_predict_layers=0`, NextN tensors отсутствуют |
| Vision/audio/video | Веса отсутствуют; локальный экспорт text-only |
| Header/ranges | PASS одноразовой инспекции; постоянный loader contract ещё TODO |
| Candidate | Локальный Unsloth `86ebfef2`, `mimo2.cpp` есть; MiMo-сборки нет |
| Tokenizer / template / API | TODO: проверены только metadata и текст шаблона |
| Engine / cache / pipeline / sessions | TODO |
| Локальная скорость | **Измерений токенов/с нет** |
| Память процесса MiMo / TTFT | Не измерялись |
| Рекомендуемый профиль | Не выбран; начальный baseline будет MTP-off, text-only |
| Следующая задача | **MIMO-01: P0.1–P0.3 — inspector, contract, изолированные oracles** |

В рамках PREP-01 созданы только два документа в `docs/mimo-v2.6-flash`.
Код, действующие профили и GGUF не изменялись. Незавершённые изменения
Hy3/common transport в рабочем дереве относятся к другой работе;
их нельзя считать результатами проверки MiMo.

## Подтверждено при подготовке

- File size **134 982 426 368 байт**, header_end **5 979 134**,
  data_start **5 979 136**, alignment32, GGUFv3, metadata49, tensors472.
- Имена уникальны, размеры всех типов известны, offsets выровнены;
  диапазоны не пересекаются и не выходят за EOF, последний заканчивается на EOF.
- Payload **134 976 447 232 байта (125,707 ГиБ)**;
  routed **122 909 884 416 (114,469 ГиБ)**,
  остальные **12 066 562 816 (11,238 ГиБ)**.
- Типы: Q2_K50, Q3_K74, MXFP4 17, BF16 101, F32 230.
  Routed tensors141; non-routed331. Source FP8 metadata не означает локальные FP8 weights.
- 48 слоёв, MoE1..47, 256 экспертов/top-8, shared tensors отсутствуют.
  Full layers **0,5,11,17,23,29,35,41,47**, остальные39 — SWA128.
  K192/V128, KV heads4/8, fused QKV, partial RoPE64, SWA attention sinks.
- Header SHA-256 без padding и payloads:
  `46e0ff64f95482e89997ba7e661d36a7061a7a5961e033fe5baa6b241e23269a`.
- Template SHA-256 по UTF-8 строке, длина3867 символов:
  `11ea52e156de38a458e6b7720ad45915d65b97d4ec979a09f55e3c9bd1b4d059`.
- GPT-2 BPE `qwen2`, vocab152576, merges151387, EOS151645, PAD151643,
  add_bos=false. Полный EOG-набор oracle пока не проверен.
- Export scope явно исключает MTP и modality companions. Нет `.nextn.`
  tensors и блоков≥48; в каталоге модели других файлов не найдено.
- Local source revision из metadata: `3b38d063180c3e4aed9691fdc735f3d10b266ee4`;
  converter revision: `58367713a6935c0810103378144008df32e3d5db`.
  Это provenance claims конвертера, не доказательство идентичности payload.
- `mimo2.cpp` из `third_party`, распакованного GLM candidate и архива
  `build-local/llama-glm-86ebfef2.tar.gz` совпадает, SHA-256
  `2ec5caa11fb9ab7604b798a17cd80244427596581c06cdb033762b6d56f50823`.
  Архив SHA-256: `f8e524b635b726bae74fd8f84bb9249e5b09384c63207f6707c5a3f921acad99`.
- Windows RAM **125,555 ГиБ**, RTX5090 total **32 607 МиБ**, driver581.80.
  95% global caps: **119,277 ГиБ / 30 976,65 МиБ**.
  Снимок до тестов: available RAM≈105,99 ГиБ, GPU used2399 МиБ;
  это внешняя нагрузка и не потребление MiMo. Перед запуском измерить заново.

Внешние источники с оговорками и расчёты KV/H2D приведены в плане.
Подтверждение отсутствия MTP относится к **этому экспорту**, не к исходному
checkpoint Xiaomi. Основное текстовое внедрение может продолжаться без sidecar.

## Не проверено

- Полный hash payload и совпадение с опубликованным GGUF.
- Точный source config на revision3b38d063 и converter/source на58367713.
  Через web прочитаны опубликованные model card и RL config `main`;
  pinned URLs не удалось получить. Прямой Python HTTP также недоступен
  в текущем sandbox. Это не мешает локальным header/oracle проверкам.
- Полный автоматический loader contract и no-allocation регистрация всех весов.
- Изолированная CUDA-сборка MiMo, числовые результаты MXFP4/BF16/routed kernels,
  fused QKV, RoPE, sinks, FA и full/SWA state после оборота128.
- Tokenizer IDs/EOG, точный renderer/parser, tools и HTTP/UI.
- Полный inference, TTFT, prefill/decode, pressure/cancel/unload, peak memory.
- CUDA Graphs, асинхронный overlap, cache policies и sessions.
- Совместимый MTP sidecar, его размеры, native MTP и multimodal companions.

## Таблица этапов

| Этап | Статус | Условие готовности |
|---|---|---|
| PREP-01 | DONE | Исследование и документация сохранены |
| P0 — compatibility/oracles | IN PROGRESS: только предварительное P0.1 | Постоянный inspector, contract, сборка, числовые fixtures |
| P1 — GPU baseline | TODO | Full-model logits/greedy parity, включая SWA128, отчёт скорости/памяти |
| P2 — tokenizer/template/API | TODO | Oracle fixtures, reasoning/tools, оба API и web chat |
| P3 — cache/pipeline | TODO | Bytes/logits parity, bounded memory, overlap и повторные A/B |
| P4 — sessions/context | TODO | Restore/shift/cancel parity для full и SWA |
| P5 — MTP | WAITING_WEIGHTS, дополнительный этап | Совместимые RL draft weights, rollback и полезный speedup |
| P6 — profile/regressions | TODO | Измеренные defaults, отдельный text-only профиль и регрессии |
| P7 — multimodal | OUT_OF_SCOPE текущего GGUF | Companions и отдельная приёмка каждой модальности |

P5/P7 не блокируют P0–P4/P6. `DONE` означает только указанную проверенную часть;
наличие внешней реализации не закрывает native engine или MTP.

## Точка продолжения: MIMO-01

1. Создать `tools/inspect_mimo2_gguf.py` и `tools/mimo2_loader_contract.py`
   поверх существующего reader. Сохранить
   `docs/mimo-v2.6-flash/MIMO26_FLASH_INSPECTION.json` с полным directory,
   hashes, архитектурой и разбиением памяти. Сейчас этих файлов нет.
2. Проверить все472 tensors против локального loader: обязательные dense0,
   routed1..47, fused QKV shapes, layerwise KV, sinks только в SWA,
   `exp_probs_b.bias`, globals. Не допускать missing weight через optional flag.
3. CPU negative fixtures: повреждённые ranges, неверные shapes/quants,
   invalid pattern, неподдержанный export scope, unexpected MTP/extra tensors.
   Маленькие fixture dimensions должны допускаться отдельно от admission реальной модели.
4. Создать `backends/mimo2/CMakeLists.txt` с проверкой archive/loader hashes,
   собственными `_deps` и build manifest. Vocabulary/template/no-allocation
   loader oracles собрать прежде полной CUDA модели.
5. Сверить EOG, `qwen2` BPE и Jinja fixtures, thinking on/off, history и tools.
   Изолировать media rejection до renderer. Сопоставление pin58367713
   завершить до объявления dependency проверенной для полного inference.
6. Записать команды/PASS/FAIL и следующую задачу: P0.4/P0.5 CUDA fixtures,
   включая SWA wrap и asymmetric K/V, затем P1 full-model baseline.

Не запускать GLM/Step/Hy3 engine с этим файлом как проверку поддержки MiMo.
Не выделять весь объём модели в RAM и не включать MTP без соответствующих весов.
Тяжёлые тесты начинать после нового global memory admission; чужие процессы не останавливать.

## Подтверждённый журнал

### PREP-01 — 2026-10-06 — Заголовок, архитектура и план

**Действия:** прочитаны инструкции, планы/статусы GLM, Step и Hy3,
GGUF reader, общий transport и локальный `mimo2.cpp`; изучены CUDA dispatch
для MXFP4 и K192/V128, опубликованные GSQ-RCO model card и RL config.
Проверены header, ranges, hashes dependency и размеры системной памяти.

**Результат:** инвентаризация готова для переноса в inspector; MTP и modality
weights отсутствуют; pipeline/state/format проверки расписаны в плане.
Tensor payload не читался и не хэшировался; inference и GPU kernels не запускались,
модели и новые зависимости не скачивались.

**Воспроизведение проверки** из корня репозитория, PowerShell:

```powershell
@'
from pathlib import Path
from collections import Counter
import hashlib
import re
from tools.gguf_reader import GGUFFile

p = Path(r'H:\models\mimo-v2.6-flash\MiMo-V2.6-Flash-RL-GSQ-RCO-3.5bit.gguf')
g = GGUFFile(p)
assert g.version == 3 and g.alignment == 32
assert g.metadata['general.architecture'] == 'mimo2'
assert g.metadata['mimo2.block_count'] == 48
assert g.metadata['mimo2.nextn_predict_layers'] == 0
assert g.metadata['mimo2.export_scope'] == 'text_trunk_without_mtp_or_modality_companions'
assert len(g.metadata) == 49
assert len(g.tensors) == len({t.name for t in g.tensors}) == 472
size = p.stat().st_size
end = 0
for t in sorted(g.tensors, key=lambda t: t.offset):
    n = t.expected_bytes()
    assert n is not None and n > 0, t.name
    assert t.offset % g.alignment == 0, t.name
    assert t.offset >= end, t.name
    assert g.data_start + t.offset + n <= size, t.name
    end = t.offset + n
assert g.data_start + end == size == 134982426368
assert (g.header_end, g.data_start) == (5979134, 5979136)
assert not any('nextn' in t.name for t in g.tensors)
layers = {int(m[1]) for t in g.tensors if (m := re.match(r'blk\.(\d+)\.', t.name))}
assert layers == set(range(48))
routed = [t for t in g.tensors if '_exps.weight' in t.name]
routed_bytes = sum(t.expected_bytes() for t in routed)
total_bytes = sum(t.expected_bytes() for t in g.tensors)
assert len(routed) == 141 and routed_bytes == 122909884416
assert total_bytes == 134976447232
assert total_bytes - routed_bytes == 12066562816
types = Counter(t.type_name for t in g.tensors)
assert dict(types) == dict(Q2_K=50, Q3_K=74, MXFP4=17, BF16=101, F32=230)
pattern = g.metadata['mimo2.attention.sliding_window_pattern']
assert len(pattern) == 48 and set(pattern) == {0, 1}
assert [i for i, s in enumerate(pattern) if not s] == [0, 5, 11, 17, 23, 29, 35, 41, 47]
assert g.metadata['mimo2.attention.head_count_kv'] == [8 if s else 4 for s in pattern]
with p.open('rb') as stream:
    header_sha = hashlib.sha256(stream.read(g.header_end)).hexdigest()
assert header_sha == '46e0ff64f95482e89997ba7e661d36a7061a7a5961e033fe5baa6b241e23269a'
template_sha = hashlib.sha256(g.metadata['tokenizer.chat_template'].encode('utf-8')).hexdigest()
assert template_sha == '11ea52e156de38a458e6b7720ad45915d65b97d4ec979a09f55e3c9bd1b4d059'
print('PASS header/ranges/pattern; payload bytes:', total_bytes)
print('Routed/other bytes:', routed_bytes, total_bytes-routed_bytes)
print('Types:', dict(types))
print('Header SHA-256:', header_sha)
print('Template SHA-256:', template_sha)
'@ | python -X utf8 -
```

Команда проверяет зафиксированный локальный файл, не содержимое его матриц.
Она не заменяет постоянный строгий contract, numerical fixtures или inference.

**Проверка документации:** приведённый Python-код повторно выполнен из этого
Markdown, exit0, `PASS header/ranges/pattern`; размеры, типы и hashes совпали.
Локальные ссылки, code fences и отсутствие trailing whitespace — PASS.
Арифметика payload/H2D и оценок KV — PASS. Это не измерение inference.

## Правила обновления

- Каждая запись `MIMO-01`, `MIMO-02` и далее: дата, исходный commit/dirty state,
  patch hashes, изменённые файлы, точные команды, exit codes и PASS/FAIL/SKIPPED.
- Ссылки давать на созданные JSON/logs. Запланированные файлы явно помечать
  как будущие; результаты другой модели не заносить в MiMo validation.
- Фиксировать model/header/template hashes, dependency SHA, GPU/driver/compiler,
  context/KV/FA/graphs, batch/ubatch, sampler/seed, cache/readers/chunks.
- Скорость: load/prefill/decode/request отдельно; учитывать reasoning/EOS,
  prompt/output counts, cold/warm/prefix reuse и минимум три повторения A/B.
- Память: global used/available RAM, process commit/working set, pinned buffers,
  GPU used/total, cache allocations, KV и peaks; disk/H2D bytes и page faults.
- Оптимизация принимается после correctness на тех же quants и границах SWA.
  После нового квантования это отдельный quality/performance эксперимент.
- Не объявлять MTP или мультимодальность готовыми по upstream code/config.
  Отсутствующие веса и непроверенные режимы сохранять явно в статусе.

## Шаблон следующей записи

```text
### MIMO-NN / Pn.m — YYYY-MM-DD HH:MM, Asia/Yekaterinburg — Название
Основание: commit, dirty state, dependency/patch SHA, GGUF/template hash.
Изменено: файлы и поведение.
Команды: точные build/test/benchmark команды.
Проверки: PASS/FAIL/SKIPPED, exit codes, fixtures/corpus, SWA boundaries.
Результаты: logits/token parity, TTFT/tokens/s, память, cache/H2D/SSD, повторы.
Артефакты: ссылки на существующие отчёты и логи.
Не закрыто: ошибки, ограничения, отсутствующие веса и непроверенные режимы.
Следующий шаг: конкретная задача и критерий приёмки.
```
