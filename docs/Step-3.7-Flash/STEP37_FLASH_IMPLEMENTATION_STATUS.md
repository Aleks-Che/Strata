# Статус внедрения Step-3.7-Flash

Обновлено: **2026-10-05**, `Asia/Yekaterinburg`.
План: [STEP37_FLASH_IMPLEMENTATION_PLAN.md](STEP37_FLASH_IMPLEMENTATION_PLAN.md).

Файл хранит проверенный прогресс и точку продолжения. План задаёт критерии,
этот документ фиксирует выполненную работу. Наличие плана не означает готовую
поддержку модели. Native engine и ограниченный GPU-кэш проверены на полной
модели. Асинхронный конвейер также проверен; остаются расширенные memory-pressure проверки.
Добавлен изолированный экспериментальный HTTP-профиль; интерактивный web chat,
внешний MCP и HTTP unload/reload остаются отдельными проверками.

## Текущее состояние

| Поле | Значение |
|---|---|
| Общий статус | PREP/P0 DONE; P1 IN_PROGRESS (P1.1–P1.3 DONE); P2 IN_PROGRESS (P2.1–P2.4 DONE, P2.5–P2.6 частично); P3 IN_PROGRESS (P3.1–P3.2/P3.4 DONE); P4 TODO; P5 IN_PROGRESS (P5.1 DONE, P5.2–P5.5 частично); P6–P7 TODO |
| Проверенная ревизия Strata | `c81b0c0b92d63e8c24aef24446dadb9d71dc178d` перед STEP-12, дерево было чистое; первоначальный baseline PREP — `97e016de11754cccd1b5550232c6c198cdbe0723` |
| Последняя выполненная работа | STEP-12: opt-in GPU allocation reuse; без MTP 9,363 →9,873 (+5,45%), с MTP2 10,355 →11,560 токена/с (+11,64%); 40 exact responses, 117 runtime и23 cache checks PASS |
| Активная задача / исполнитель | Нет активного исполнителя; STEP-12 завершён в объёме offline cache/speed trials |
| Следующая задача | STEP-13 / P3,P6: подключить reuse как opt-in native pipe и проверить длинные запросы, cancel/recovery и pressure перед сменой default. P5 SWA512 rollback/stop/cancel/MTP integration и остаток P2.5–P2.6 также остаются открытыми |
| Основная модель | `H:\models\Step-3.7-Flash\UD-Q4_K_S` |
| Входной файл | `Step-3.7-Flash-UD-Q4_K_S-00001-of-00004.gguf` |
| Фактическая архитектура | GGUF `step35`; display name `Step-3.7-Flash`; 45 основных блоков, 42 MoE, 12 full + 33 SWA |
| Размер | 114 163 192 448 байт, 106,323 ГиБ; 754 тензора; routed 99,668 ГиБ, остальные 6,650 ГиБ |
| Стенд | Windows, Ryzen 9 9950X, RAM 128 ГиБ, RTX 5090 32 ГиБ |
| Разрешённый бюджет экспериментов | RAM/VRAM target 95%; реальные пики и запас для ОС учитывать отдельно |
| Step backend / профиль | Отдельный native `strata-step35`; experimental HTTP profile: `build-local/step35-cuda/step10-http-profile-8g/step37.json`, создаётся `tools/prepare_step35_profile.py`. Активный профиль приложения не заменён |
| Step dependency pin | Unsloth `86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9`; Step patches дополнены `step-prefill-cache-admission`; отдельная сборка |
| Step tokenizer / API | deepseek-v3 и exported tokenizer по 2190/2190 native checks; Step template 215/215; parser/request adapters и HTTP registration готовы. Native EOG [1,128007], PAD2 не EOG; JSON/SSE, stop strings и low/medium/high effort подключены |
| Native MTP | Официальный Q8_0 совместим; отдельный checker использует pinned common MTP driver. STEP-11: +15,9% decode от MTP; STEP-12: дополнительный paired reuse gain +11,64%, до11,560 токена/с. Это разные серии; native pipe/HTTP MTP пока не подключён |
| Vision | Локального mmproj нет; отдельный P7 |
| Измеренная скорость / runtime память Step | STEP-08: context4096, batch17, pipeline1/cacheauto/F32: prefill511 57,376 → 40,281 с; весь запрос511+16 60,137 → 43,550 с. Повторный короткий decode при prefill-off 8,505 токена/с; tradeoff и память ниже. Старый STEP-06 sync/pipeline comparison сохранён |
| Блокеры | Для P0–P3 внешних блокеров не выявлено. Draft для P5 получен; остаются integration/state gates. Для P7 по-прежнему нет mmproj |
| Приоритет совместимости | Текущие профили и бинарники моделей не изменены. Step hooks в общем server opt-in; 232 tests PASS, включая полный serve.test_server и GLM/DeepSeek regressions. Два известных старых GLM profile failures не включены в этот набор; их прежний baseline сохранён |

## Подтверждено при подготовке

1. Все четыре файла доступны для чтения, GGUF v3. `split.no=0..3`, count 4,
   total tensors 754; число тензоров по частям 0/351/316/87.
2. 754 имени уникальны, известны все quant block sizes. Диапазоны выровнены,
   не перекрываются, не выходят за длину файла. У частей с весами конец
   последнего payload совпадает с концом файла. Это не проверка значений весов.
3. Все 126 routed tensors — Q4_K, gate/up `[4096,1280,288]`,
   down `[1280,4096,288]`. Одна матрица 2,8125 МиБ, main no-hit traffic
   расчётно 2,768555 ГиБ/decode; скорость из этого не вычислялась.
4. Блоки 0..44 — основная модель; MoE 3..44. В локальном наборе нет NextN
   tensors, дополнительных blocks и vision tensors. В родительской директории
   найден только каталог основного кванта.
5. Tokenizer: `deepseek-v3`, vocab 128896, merges 127741, BOS 0, EOS 128007, PAD 2.
   Встроенный template использует `<think>`, XML function/parameter tool calls
   и generation prefix с открытым thinking-блоком. В STEP-01 oracle прошёл 2190 сравнений.
6. В `tools/strata_tokenizer.py` добавлен явный режим `deepseek-v3`; прежние режимы
   сохранили правила. STEP-10 добавляет `step35` в `configured_template()` с
   обязательной проверкой exported control tokens, metadata и template hash.
7. В локальных llama.cpp исходниках есть Step loader/graph и MTP code.
   Наличие файла не подтверждает Step inference, совместимость draft или pin.
8. Публичные config/model card и каталог официальных MTP/mmproj изучены.
   Ссылки и отделение внешних сведений от локальных измерений — в плане.

## Не проверено

- Полные checksum больших частей и совпадение с оригинальным опубликованным набором.
- Численные weights и факторы `rope_freqs`; статический loader contract 754 tensors PASS.
- Интерактивный web chat и HTTP unload/reload полной модели. API JSON/SSE,
  request adapters/parser, local tool dialogue и socket disconnect проверены;
  CLI lazy startup и отдача web HTML также проверены, без browser interaction.
- Независимый CPU/scalar oracle logits полной модели. Native/pinned selected
  transfer полной модели bit-exact (STEP-04); independent synthetic graph/state
  75/75 PASS (STEP-03), F32/Q8_0/Q6_K/Q4_K matrix/router PASS (STEP-02).
- Полное покрытие budget95: startup/native OOM, предельная внешняя нагрузка
  и контексты больше 4096. Ограниченная реальная нагрузка и запрос на 2591 токен
  при context4096 проверены в STEP-07. Cache OOM bypass и async ring проверены ранее.
- Внешний MCP и его отмена, persistent runtime sessions, SWA rollback и HTTP stop
  после длинного rollover. READY принимает session-id для изоляции, но INFO
  conversation_cache=0; восстановление KV-сессий этим не заявляется.
- MTP на длинных prompts/через SWA512, stochastic sampling, sessions и pipe/HTTP
  stop/cancel. Короткий greedy sidecar/speedup проверен в STEP-11; vision не проверен.

Результаты GLM/DeepSeek/Qwen нельзя отмечать здесь как успешные тесты Step.
Собраны CPU/CUDA vocabulary/template oracles, CUDA matrix/router и native graph checks.
Native inference полной модели выполнен; установка HTTP-профиля Step не выполнялась.

## Таблица этапов

| Этап | Статус | Что сделано | Что остаётся для завершения |
|---|---|---|---|
| PREP. Исходные данные и документы | DONE | Локальная проверка и два документа | Завершено в границах запроса на план |
| P0. Inventory, pin, oracles | DONE | P0.1–P0.5; 78/78 kernels, 75/75 graph/state, 215/215 template, tokenizer PASS | Полная модель относится к P1 |
| P1. GPU baseline | IN_PROGRESS | P1.1–P1.3 DONE; полный GGUF, exact logits/IDs. P1.4: global samples, cache trim и RAM admission реализованы | Startup/native OOM и предельная pressure остаются; bounded external pressure и context4096 прошли STEP-07 |
| P2. Tokenizer/API | IN_PROGRESS | P2.1–P2.3 DONE в adapter/pipe: native renderer 215/215, parser 3415 sequences, реальный tool dialogue | API EOG/capabilities, HTTP/profile, stop strings и disconnect/cancel |
| P3. Pipeline/cache | IN_PROGRESS | P3.1–P3.2/P3.4 DONE; shared ring, 973 real-byte checks, context4096 и bounded external pressure PASS | Prefill admission on/off проверен STEP-08; остаются startup OOM/предельная pressure, early refill, 4/16 MiB tuning и async cache leases при снятии D2D sync |
| P4. Sessions/context | TODO | В P0 проверен in-process KV checkpoint на synthetic Step | Runtime sessions, budgets, sampler/output state, полная модель и длинные контексты |
| P5. MTP | IN_PROGRESS | P5.1 DONE; официальный Q8, separate draft context/hidden/greedy verify, off/1/2/3, 60 exact MTP requests, RAM/VRAM trials в checker | Pipe/HTTP integration, SWA512 rollback, stochastic, stop/cancel/sessions, длинные и разнообразные prompts |
| P6. Profile/release | TODO | Определена методика | Setup/profile, controls/benchmarks, регрессии и capabilities |
| P7. Vision | TODO | Найден опубликованный mmproj | Отдельные encoder/input/state/memory проверки; необязателен для текстового выпуска |

## Правила обновления

1. Перед работой прочитать план, сводку и последнюю запись; проверить HEAD и
   `git status --short`. Чужие изменения сохранять.
2. Выбрать подпункт P0.2/P1.1/…; записать активную задачу и исполнителя.
   Закрытие одного подпункта не означает DONE всего этапа.
3. Разделять наличие кода, успешную сборку, fixture, полную модель и измеренный
   выигрыш. Для готового к проверке кода использовать VERIFY.
4. DONE ставить только при выполнении критерия плана и наличии доказательства.
   Синхронизировать чекбоксы, таблицу этапов, сводку и точку продолжения.
5. Сохранять command/cwd/exit code, source и binary hashes, pin/patches,
   GGUF fingerprint, parameters и результаты. Ссылки давать на существующие
   артефакты; будущие файлы помечать как планируемые.
6. Для performance сохранять cold/warm, prompt IDs, token counts, повторения,
   cache budgets, MTP depth/heads, sampling, TTFT/wall/decode, RAM/VRAM/paging.
   Не смешивать repeated prompt и A/B; cache miss не равен чтению SSD.
7. При новом ядре заранее задать numerical tolerance; не расширять её задним
   числом ради PASS. Транспортные изменения должны сохранять bytes и logits.
8. Ошибки и отменённые варианты оставлять в журнале; исправление оформлять
   новой записью. BLOCKED требует конкретной причины и условия продолжения.
9. После завершения/передачи снимать активного исполнителя, записывать один
   следующий шаг, его входы, результат и способ проверки.
10. Профили других моделей и их defaults не менять ради Step-эксперимента.
    Временные допущения и неподтверждённые backend capabilities указывать явно.

| Статус | Значение |
|---|---|
| TODO | Исполнение ещё не началось |
| IN_PROGRESS | Есть выполненная часть, критерий этапа не достигнут |
| VERIFY | Код/сборка готовы, обязательные проверки ещё не завершены |
| BLOCKED | Указаны воспроизводимое препятствие и условие разблокировки |
| DONE | Конкретный критерий выполнен и подтверждён |

## Точка продолжения

**STEP-08 — измерить admission policy для prefill.**

1. Отдельный Step backend/build сохраняется. Проверенная база:
   [manifest STEP-07](STEP37_FLASH_STEP07_BUILD_MANIFEST.json),
   [README](../../backends/step35/README.md). Основной engine остался STEP-06.
2. В запросе2591 tokens sync prefill cache hit fraction4,8%; за весь запрос
   cache-fill D2D1,36ТБ. Кандидат — opt-in ограничение новых cache admissions
   во время prefill. Cache hits и обучение frequency history должны сохраняться;
   обычное decode admission — тоже. Не менять defaults до paired measurement.
3. Передавать фазу prefill/decode явно из request owner. Нельзя считать любой
   batch<=8 decode: последний prefill microbatch также может быть коротким.
   Отмена/ошибка/следующий запрос должны корректно сбрасывать фазу.
4. Сначала fixture: mixed/F32 weights, F32/F16 KV, cold/warm/eviction,
   длинный prefill и короткий последний batch. Требовать exact bytes/logits,
   нулевые запрещённые prefill admissions и сохранённое decode admission.
5. Затем полная модель: P1 short IDs/logits и новый long reference из
   `build-local/step35-cuda/step07-context/readers-0/long-reference.f32`.
   Context4096, batch17, prompt IDs в `STEP37_FLASH_STEP07_CONTEXT.json`.
   Для скорости сравнивать полный запрос и decode отдельно; paired on/off,
   одинаковые cache budget, pipeline и prompt. STEP-07 timings не являются A/B.
6. Действующие controls: cacheauto/pipeline0 и cacheauto/pipeline1/chunk8.
   Кэш по умолчанию0, pipeline0; HTTP-профиля нет. Main/draft/full OOM,
   максимум32768 и stop после long rollover остаются отдельными gates.
7. До снятия D2D sync нужны cache consumer leases, ready/used events и retirement
   accounting. Native scratch может alias activations — нельзя писать заранее.
   Оптимизацию admission проверять независимо от смены synchronization.

**Критерий следующего шага:** точные полные logits, корректное поведение фаз
и воспроизводимое снижение общего времени запроса. При отсутствии выигрыша
сохранить результат измерения и прежний default. Общие sources/profiles других
моделей не менять; HTTP/parser/session/MTP требуют собственных проверок.

## Подтверждённый журнал

### PREP-01 / P0.1 — 2026-10-05 — Проверка локальной модели и создание плана

**Исполнитель:** Codex. **Статус:** DONE для первичной инвентаризации и документов.
**База:** `97e016de11754cccd1b5550232c6c198cdbe0723`.
**Cwd:** `C:\work\git\my-repos\Strata`.

**Изменены только:**

- `docs/Step-3.7-Flash/STEP37_FLASH_IMPLEMENTATION_PLAN.md`;
- `docs/Step-3.7-Flash/STEP37_FLASH_IMPLEMENTATION_STATUS.md`.

**Действия:** `Get-ChildItem -LiteralPath 'H:\models\Step-3.7-Flash\UD-Q4_K_S'`,
чтение через существующий `tools.gguf_reader.GGUFFile`, проверка split/ranges/
shapes, чтение template и Step loader. Проверка завершилась exit0.
Первый вывод Unicode token strings в консоль CP1251 завершился ошибкой кодировки;
повтор с `PYTHONIOENCODING=utf-8` успешен. Это ошибка вывода, не GGUF или tokenizer parity.

Команда для повторения основной структурной проверки из корня репозитория:

```powershell
$env:PYTHONIOENCODING = 'utf-8'
@'
from pathlib import Path
from collections import Counter
from tools.gguf_reader import GGUFFile
import hashlib
paths = sorted(Path(r'H:\models\Step-3.7-Flash\UD-Q4_K_S').glob('*.gguf'))
assert len(paths) == 4
names, blocks, moe = set(), set(), set()
types = Counter()
payload = routed = 0
for i, path in enumerate(paths):
    g = GGUFFile(path)
    assert (g.metadata['split.no'], g.metadata['split.count'],
            g.metadata['split.tensors.count']) == (i, 4, 754)
    end = g.header_end
    for t in sorted(g.tensors, key=lambda t: t.offset):
        assert t.name not in names
        names.add(t.name)
        n = t.expected_bytes()
        assert n is not None and t.offset % g.alignment == 0
        assert g.data_start + t.offset >= end
        end = g.data_start + t.offset + n
        assert end <= path.stat().st_size
        payload += n
        types[t.type_name] += 1
        if t.name.startswith('blk.'):
            blocks.add(int(t.name.split('.')[1]))
        if '_exps.weight' in t.name:
            assert t.type_name == 'Q4_K'
            assert t.shape == ([1280,4096,288] if 'down' in t.name else [4096,1280,288])
            routed += n
            moe.add(int(t.name.split('.')[1]))
    assert not g.tensors or end == path.stat().st_size
    print(path.name, path.stat().st_size, len(g.tensors))
assert len(names) == 754 and blocks == set(range(45)) and moe == set(range(3,45))
assert (payload, routed) == (114157912576, 107017666560)
print('PASS', dict(types), 'payload', payload, 'routed', routed)
print('first SHA256', hashlib.sha256(paths[0].read_bytes()).hexdigest())
'@ | python -
```

**Результат:** counts 0/351/316/87, 754 unique tensors, bounds PASS,
Q4_K expert shapes PASS. Факты, размеры, SHA первой части и template занесены
в план; будущий формальный inventory не объявлен готовым.

**Ограничения:** не читались все 114 ГБ весов для checksum, не выполнялись load/
forward/tokenizer oracle, не скачивались sidecar/mmproj. Source Step loader
изучен, но не проверен через сборку/граф. Оптимальные настройки ещё неизвестны.

**Далее:** P0.2, воспроизводимый inspector и CPU tests по точке продолжения.

### STEP-01 / P0.2–P0.3, P2.1 — 2026-10-05 — Inspector, loader contract и tokenizer

**Исполнитель:** Codex. **Статус:** DONE для P0.2/P0.3/P2.1;
P0.4/P0.5 остаются открытыми. **База:** `97e016de11754cccd1b5550232c6c198cdbe0723`.
**Cwd:** `C:\work\git\my-repos\Strata`.

**Реализовано:**

- `tools/setup_step35.py --inspect`: воспроизводимый read-only inventory всех
  четырёх shards, metadata-only first, split consistency, quant row geometry,
  offsets/overlap/overflow, metadata arrays, полный trunk contract. Защита от
  перезаписи любого input shard, включая aliases. Профиль не устанавливается.
- `tools/step35_loader_contract.py`: 754 tensor mappings, отдельные full/SWA
  heads и rotary dimensions, norms, gate, dense/routed/shared FFN, router bias,
  clamps. Неизвестные layouts, MTP и LongRoPE не допускаются автоматически.
- `backends/step35/`: отдельная сборка из локального архива, source/loader hash
  checks, без patches. Исполняемый файл `strata-step35-tokenizer` загружает только
  vocabulary. Pipeline, server binary и GPU graph пока отсутствуют.
- `tools/strata_tokenizer.py`: явный режим `deepseek-v3`. В выбранной зависимости
  его ordered splits совпадают с JOYAI; строка режима сохраняется как `deepseek-v3`,
  `ignore_merges` остаётся false. Алгоритмы старых режимов не изменены.
- `tools/gguf_reader.py`: проверка длины metadata array до выделения/чтения,
  чтобы испорченный count не запрашивал огромный буфер. Остальной parser сохранён.
- `tools/check_step35_tokenizer.py`: native provenance, exact token IDs,
  decoded bytes и отсутствие автоматически добавленного BOS, оба parse_special.

**Артефакты:**

- [GGUF inventory и 754 mappings](STEP37_FLASH_GGUF_INVENTORY.json).
- [Loader compatibility](STEP37_FLASH_LOADER_COMPATIBILITY.md).
- [Tokenizer parity, все 2190 cases](STEP37_FLASH_TOKENIZER_PARITY.json).
- [Build/source/binary manifest](STEP37_FLASH_BUILD_MANIFEST.json).
- [Старые GLM test failures с HEAD-модулями](STEP37_FLASH_EXISTING_FAILURES.json).

**Команды и результаты:**

```powershell
python tools/setup_step35.py --model-dir H:/models/Step-3.7-Flash/UD-Q4_K_S --inspect --output docs/Step-3.7-Flash/STEP37_FLASH_GGUF_INVENTORY.json
# exit0: 4 shards, 754 tensors, 106.322758 GiB; static trunk contract PASS

cmd /c build-local\build-step35-oracle.cmd
# exit0: MSVC configure/build; CTest 1/1 PASS (version smoke only)
# Воспроизводимые CMake команды: backends/step35/README.md

python tools/check_step35_tokenizer.py --gguf H:/models/Step-3.7-Flash/UD-Q4_K_S --oracle build-local/step35-oracle/bin/strata-step35-tokenizer.exe --output docs/Step-3.7-Flash/STEP37_FLASH_TOKENIZER_PARITY.json
# exit0: 2190 cases, 0 mismatches; native EOG [1,128007], PAD2 не EOG

python -m unittest tools.test_setup_step35 tools.test_step35_tokenizer tools.test_setup_glm5next tools.test_glm5next_tokenizer tools.test_glm5next_loader_contract tools.test_deepseek4 tools.test_glm5next_profile tools.test_setup_rope tools.test_setup_draft_vocab serve.test_deepseek serve.test_glm5next serve.test_glm5next_requests
# exit0: 125 tests PASS (20 новых Step, 105 существующих)
```

Step tests покрывают malformed headers/arrays, пропуски и перестановку shards,
duplicate tensors, неизвестные quant/names, truncation, overlap, unaligned/overflow
ranges, неверные heads/SWA/clamps, scalar clamps, metadata-only padding, singleton
dimensions, output protection и границы structural fingerprint. Отдельные BPE
fixtures проверяют digit/CJK boundaries, special tokens и ignore_merges=false.
При написании fixture исправлено ошибочное ожидаемое разбиение `中文abc`: в этом
тестовом словаре есть merge `abc`, поэтому правильный результат — `中文` + `abc`.
Native comparison до и после этого исправления кода tokenizer не требовал изменений.

**Отдельная регрессия и существующие ошибки:** расширенный запуск 71 tests дал
69 PASS / 2 FAIL. Обе ошибки находятся в `serve.test_glm5next_profile`:
`test_profile_identity_and_no_overwrite` и `test_memory_targets_reach_engine_arguments`.
Ожидания тестов не учитывают уже добавленные в HEAD pipeline/MTP arguments.
Повторный Python запуск с обоими изменёнными shared modules, загруженными через
`git show HEAD:...` в память, воспроизвёл те же 2 FAIL из 4 tests (exit1).
Файлы приложения при этой проверке не откатывались. Тесты и GLM runtime не правились;
эти старые ошибки нельзя считать новой Step-регрессией или скрывать в общем PASS.

**Числа и provenance:** суммарные files/payload/routed/non-routed совпали с PREP-01:
114163192448 / 114157912576 / 107017666560 / 7140246016 байт.
Structural fingerprint — `37305e11f45387f044806d639588fe6ef9d4b10eeaad1dc8ca0f02f9cdae6502`.
Oracle SHA-256 — `a7aa6143d5cbc301ecfda2ff671f51caeec7d767a3bcdcb61d933cb4ffb6a2c7`.
Compiler: MSVC 19.44.35222.0, CUDA OFF. Manifest фиксирует hashes исходников.

**Ограничения:** payload не хешировался/не загружался, inference/HTTP/GPU не
запускались, независимая Jinja rendering parity отсутствует. Измерений tok/s и
RAM/VRAM inference нет. Текущие профили, default settings, server routing и
старые backend builds не изменялись. Это соответствует приоритету пользователя
сохранить существующее приложение работающим.

**Далее:** P0.4 по точке продолжения выше; сначала CUDA numerical fixtures,
затем synthetic Step graph и только после них P1 full-model selected-copy baseline.

### STEP-02 / P0.4 — 2026-10-05 — CUDA build, kernels и исправления зависимости

**Исполнитель:** Codex. **Статус:** DONE для P0.4, P0 в целом IN_PROGRESS.
**База:** `97e016de11754cccd1b5550232c6c198cdbe0723` с изменениями STEP-01.
**Cwd:** `C:\work\git\my-repos\Strata`.

**Реализовано:**

- Изолированная Step CUDA-сборка: CUDA 13.0.48, MSVC 19.44.35222.0, actual SM120a;
  GPU RTX 5090, driver 581.80, total 32607 MiB. Использован локальный проверенный
  архив, загрузок сети/моделей не было.
- `check_kernels.cpp`: 72 matrix/layout + 6 router/layout cases, с direct CUDA
  backend, CPU и scalar references. Типы F32/Q8_0/Q6_K/Q4_K, batch1/4/17,
  288 experts/top8, broadcast/per-route inputs, padding/offsets/guards.
- `StrictF32.cmake`: custom MMF соблюдает `NVIDIA_TF32_OVERRIDE=0`.
- `RoutedStrides.cmake`: fallback MUL_MAT_ID gather учитывает `nb12/nb11`
  вместо `ne11`, не читает padding как вход следующего token.
- Padded router перед sigmoid упаковывается через GPU `ggml_cont`.
  CUDA unary kernel требует непрерывного layout; CPU fallback не добавлялся.
- Tokenizer `--version` сообщает фактический patch set. Python checker принимает
  только исходный `none` или проверенный комплект обоих patches; неизвестные и
  частичные наборы отвергаются. Новый provenance test добавлен в Step suite.

**Команды:**

```powershell
cmd /c build-local\build-step35-cuda.cmd
# exit0; воспроизводимые configure/build flags — backends/step35/README.md

$env:PATH="$PWD\build-local\cuda-13.0\bin;$PWD\build-local\cuda-13.0\bin\x64;$env:PATH"
$env:NVIDIA_TF32_OVERRIDE='0'
build-local/step35-cuda/bin/strata-step35-kernels-check --output docs/Step-3.7-Flash/STEP37_FLASH_CUDA_KERNELS.json
# exit0, 78/78 PASS с обоими patches

ctest --test-dir build-local/step35-cuda --output-on-failure --no-tests=error
# exit0, 2/2 PASS
cmd /c build-local\build-step35-oracle.cmd
# exit0, CPU rebuild + CTest 1/1 PASS

python -m unittest tools.test_step35_tokenizer tools.test_setup_step35
# exit0, 21/21 PASS
python tools/check_step35_tokenizer.py --gguf H:/models/Step-3.7-Flash/UD-Q4_K_S --oracle build-local/step35-cuda/bin/strata-step35-tokenizer.exe --output docs/Step-3.7-Flash/STEP37_FLASH_TOKENIZER_CUDA_PARITY.json
# exit0, 2190/2190 PASS
python tools/check_step35_tokenizer.py --gguf H:/models/Step-3.7-Flash/UD-Q4_K_S --oracle build-local/step35-oracle/bin/strata-step35-tokenizer.exe --output docs/Step-3.7-Flash/STEP37_FLASH_TOKENIZER_CPU_RECHECK.json
# exit0, 2190/2190 PASS
```

**Ошибки и их устранение:** unpatched запуск остановился после 75 случаев на
неподдержанном padded SIGMOID, при этом 14 F32 cases уже нарушили допуски.
После GPU packing router прошёл, осталось 14 F32 FAIL из 78. Только strict-F32
patch устранил TF32 error, но выявил четыре routed-stride ошибки с max abs до
317,51. Второй patch исправил индекс строки, после чего все 78 PASS.
Каждый промежуточный отчёт сохранён, допуски не расширялись.

**Результаты и evidence:**

- [Описание проверки и все промежуточные отчёты](STEP37_FLASH_CUDA_VALIDATION.md).
- [Итоговые 78 cases](STEP37_FLASH_CUDA_KERNELS.json).
- [Source/patch/compiler/binary hashes](STEP37_FLASH_CUDA_BUILD_MANIFEST.json).
- [CUDA tokenizer](STEP37_FLASH_TOKENIZER_CUDA_PARITY.json) и
  [CPU tokenizer recheck](STEP37_FLASH_TOKENIZER_CPU_RECHECK.json).

F32 max abs CUDA/scalar = 1,192e-7, NMSE = 1,866e-14. Quantized max abs ≤ 0,006609,
NMSE ≤ 5,510e-5. Все padded/compact matrix outputs bit-exact; inputs/guards
сохранены; router IDs exact. Это результаты synthetic fixtures, не полной модели.

**Совместимость:** patches генерируют `.cu` только в Step build directory.
Архив, распакованные source files и сборки/профили Qwen/GLM/DeepSeek не изменены.
Hashes `gguf_reader.py` и `strata_tokenizer.py` совпали с STEP-01 manifest;
общие Python-модули в этом шаге не менялись, старый suite не перезапускался без
причины. Два известных GLM profile test failures остаются отдельным baseline.
STEP-01 manifest хранится как исторический снимок; текущие hashes — в CUDA manifest.

**Ограничения:** F32 CUDA fallback содержит host bookkeeping/synchronization,
но matrix math идёт на GPU. Kernel checker не проверяет CUDA graph capture/replay,
полный Step scheduler, SWA state, загрузку 106 ГиБ весов, pipeline, HTTP, MTP или
скорость. Shared/routed SwiGLU clamps и attention проверять в P0.5.

**Далее:** P0.5 по точке продолжения выше.

### STEP-03 / P0.5 — 2026-10-05 15:44, Asia/Yekaterinburg — Native graph, SWA checkpoints и template

**Исполнитель:** Codex. **Статус:** DONE для P0.5; P1/P4 и запуск полной модели
этой записью не закрываются. База Strata и dependency pin прежние. Точные source,
binary и report hashes:
[STEP37_FLASH_GRAPH_BUILD_MANIFEST.json](STEP37_FLASH_GRAPH_BUILD_MANIFEST.json).

**Добавлено:** `synthetic_step.hpp`, `check_graph.cpp`, `session_snapshot.hpp`,
`template_oracle.cpp`, `tools/check_step35_template.py`; отдельные CMake targets
и проверки. Shared source, setup profiles, Qwen/DeepSeek/GLM backends и их binaries
не изменялись. Хэши `gguf_reader.py` и `strata_tokenizer.py` совпадают с STEP-01.

**Граф:** настоящий Step loader и graph builder, synthetic F32 веса с seed
`0x37d001`, три слоя full/SWA/full, width256, heads2/3/2, head_dim128, expert16/top8.
Есть dense FFN, routed/shared FFN, Q/K RMSNorm, head gate, full64/SWA128 RoPE,
общий rotary-factors tensor и активируемые routed7/shared16 clamps.
KV F32, context2048, FA off, `NVIDIA_TF32_OVERRIDE=0`.

В [STEP37_FLASH_GRAPH_VALIDATION.json](STEP37_FLASH_GRAPH_VALIDATION.json)
**75/75 PASS**, включая scalar intermediate references, full-prefill1024,
microbatch256/32/64, chunks127/128 и serial decode. GPU-vs-CPU logits:
max abs **3,7253e-6**, NMSE **5,4662e-12**; заранее выбранные пределы
**5e-4 / 1e-7** не расширялись. GPU compute nodes проверены callback-ом;
CPU tensor math в GPU runs отсутствует. CPU здесь только независимый oracle.

После1024: full cache2048 cells, SWA768 cells, retained SWA positions256..1023.
Step checkpoint содержит **2 905 192 байт native KV payload** плюс индексы/metadata
wrapper-а; это synthetic fixture, не RAM/VRAM всей модели. Сохраняются физические
индексы и masked retained rows, восстанавливается append cursor. A→B→A и
продолжения после snapshots 0/511/512/513/767/768/769/1024/1530/1536 — **bit-exact**.
Откат к checkpoint до tentative suffix и к старому checkpoint после eviction
также bit-exact. Snapshot другого context отклоняется.

**Ошибки и контрольные запуски сохранены:**

- [Первичный diagnostic](STEP37_FLASH_GRAPH_INITIAL_DIAGNOSTIC.json): harness
  ошибочно напечатал PASS/0 cases из-за invalidated reference внутри
  `ordered_json`. В массиве были 55 проверок и два FAIL. Этот запуск не засчитан.
  Результаты перенесены в отдельный JSON-массив; итог проверяет число cases и
  каждый `pass`. Исправленный harness воспроизвёл exit1 и два FAIL в
  [native state control](STEP37_FLASH_GRAPH_NATIVE_STATE_CONTROL.json).
- Native sequence save/restore compacted SWA; A→B→A отличался на **2,5332e-7**.
  Step checkpoint исправляет размещение, не меняя native dependency files.
- [Tail rollback control](STEP37_FLASH_GRAPH_TAIL_ROLLBACK_CONTROL.json):
  74/75 PASS; простой tail removal через оборот кольца отличался на **2,0862e-7**.
  Попытка восстановить только cursor не помогла: изменяется occupancy и поиск
  свободных ячеек. Этот путь не принят для exact rollback. Финальная проверка
  использует checkpoint до tentative suffix, допуск остаётся нулевым.
- Native Jinja не знает `fromjson`: в
  [template control](STEP37_FLASH_TEMPLATE_NATIVE_CONTROL.json) 24/209 mismatches.
  В Step oracle добавлен native JSON parsing string tool arguments в объекты;
  Python независимо рендерит исходные строки через `fromjson`. Сам GGUF template
  не менялся. Это адаптация oracle, ещё не подключённый API renderer.
- При сборке template target исправлены include/link dependencies (`common/json`,
  `common/unicode`, `ggml-base`). Первый CTest без CUDA DLL в PATH был остановлен
  на version check; в CMake добавлены его PATH и timeout15. Повтор успешен.

**Template:**
[STEP37_FLASH_TEMPLATE_PARITY.json](STEP37_FLASH_TEMPLATE_PARITY.json),
**215/215 PASS**: 208 byte/native token-ID comparisons, один ручной BOS/prefix
reference и шесть reject checks для malformed/non-object tool JSON.
Проверены low/medium/high, history, reasoning replay, system/observation,
text parts, object/string tool arguments, несколько вызовов/ответов инструментов,
generation prompt on/off. В каждом prompt ровно один BOS; native decoded bytes
совпадают с Python. Jinja2 3.1.4; полный шаблон имеет прежний SHA-256.

**Команды / cwd:** корень Strata; воспроизводимые configure/build/run команды
в [README](../../backends/step35/README.md). Локальный MSVC wrapper
`cmd /c build-local\build-step35-cuda.cmd` — exit0;
`ctest --test-dir build-local/step35-cuda --output-on-failure --no-tests=error`
— exit0, **4/4**, включая прежние **78/78 CUDA kernel cases**.
CPU build template/tokenizer и CTest — exit0, **2/2 version checks**.
`python tools/check_step35_template.py` с аргументами из README — exit0.
`python -m unittest tools.test_setup_step35 tools.test_step35_tokenizer`
— exit0, **21/21**. `git diff --check` — exit0. Старые два GLM profile failures
из STEP-01 здесь не исправлялись и не объявляются PASS.

**Ограничения:** synthetic graph не подтверждает logits или качество реальных
UD-Q4_K_S весов; F16 KV, FA, full-model sessions, sampler/output restore, HTTP,
memory budgets и транспорт экспертов ещё требуют проверки. KV helper ограничен
одним append-only sequence и тем же живым context. Контекст должен жить дольше
snapshot. Он не является файловым session format и не завершает P4.
Скорость Step в tokens/s **по-прежнему не измерена**.

**Далее:** P1.1–P1.2 по точке продолжения выше.

### STEP-04 / P1.1–P1.3 — 2026-10-05, Asia/Yekaterinburg — Native engine и полный GGUF

**Исполнитель:** Codex. **Статус:** DONE для P1.1–P1.3. P1.4, HTTP, cache,
pipeline и MTP остаются открытыми. База Strata и dependency pin прежние.
[Build manifest](STEP37_FLASH_RUNTIME_BUILD_MANIFEST.json) хранит hashes исходников,
generated patches, binaries, отчётов и raw logits, команды и границы проверки.

**Реализация:** отдельный `strata-step35` с одним владельцем model/context,
reader thread, ограниченной очередью и pipe GEN/STOP/QUIT/ENC. После запроса
очищаются KV и sampler; session hash пока только разделяет запросы. Native loader
загружает все четыре shards, включая первый с нулём tensors. Non-routed weights
на GPU, routed weights в обычном mmap, без whole-file prefetch. GPU audit до
compute/copy запрещает CPU tensor math; полный expert tensor копировать нельзя.

Selected ranges передаются через один pinned staging **16 МиБ** с синхронизацией
перед повторным использованием. Диапазоны крупнее staging разбиваются на части.
Планировщик и GPU scratch сохраняют native layout. Persistent expert cache и
overlap отсутствуют. Новый patch `step-sync-selected-copy-gpu-audit-mmap-demand`
добавлен к двум CUDA correctness patches; он генерирует backend/loader translation
units только в Step build directory. Извлечённые dependency sources не изменены.

Добавлены `RuntimePatches.cmake`, `runtime.hpp`, `sync_runtime.h/.inc`,
`gpu_only_audit.inc`, `main.cpp`, Step `protocol.hpp`/`tokenizer_protocol.hpp`,
`check_runtime.cpp`, `tools/check_step35_engine.py` и `tools/check_step35_model.py`.
Обновлены Step CMake, fixtures, допустимый patch set tokenizer checker и документы.

**Проверки:**

- [Transport](STEP37_FLASH_RUNTIME_VALIDATION.json): **27/27 PASS**. Resident GPU,
  native selected-copy и pinned selected-copy дают exact F32 logits. Наборы F32
  и mixed Q4_K experts/Q8_0/Q6_K weights, KV F32/F16, batches1/4/17, prefix513 через
  границу SWA и serial decode. Observer сверяет каждый переданный GPU byte с mmap.
  Диапазон **18 МиБ** проходит через staging16 двумя chunks. CPU embedding
  намеренно отклоняется до compute и H2D.
- [Pipe](STEP37_FLASH_PIPE_VALIDATION.json): **22/22 PASS**. Greedy, seeded repeat,
  A→B→A, malformed requests, STOP при prefill, следующий запрос после cancel/error,
  session isolation и QUIT во время работы.
- [Full-model request checks](STEP37_FLASH_MODEL_REQUESTS.json): **5/5 PASS**.
  ENC с parse_special off/on совпадает с Python; seeded repeat, STOP и свежий
  greedy после отмены совпадают с контрольным prefix. Процесс завершился с exit0.
- CUDA CTest **6/6 PASS**, включая прежние **78/78 kernels**, **75/75 graph/state**,
  новые transport/pipe и version checks. Default P0 fixture hash не изменился.
- [Tokenizer](STEP37_FLASH_RUNTIME_TOKENIZER.json) **2190/2190** и
  [template](STEP37_FLASH_RUNTIME_TEMPLATE.json) **215/215 PASS** на новой сборке.
  Python Step suite **21/21 PASS**. Старые GLM profile failures не объявляются PASS.

**Первый неуспешный запуск сохранён:**
[runtime initial](STEP37_FLASH_RUNTIME_INITIAL.json), exit1/0 cases: harness
ожидал неправильное число expert tensors. В этом Step pin dense/MoE выбираются
по наличию tensors, а GLM поле `n_layer_dense_lead` не заполняется. Проверка теперь
использует фактические layer tensors и проверяет полные gate/up/down/router группы.

**Полная модель:** структурный fingerprint прежний
`37305e11f45387f044806d639588fe6ef9d4b10eeaad1dc8ca0f02f9cdae6502`;
реально открыты 106,323 ГиБ / 754 tensors. Стенд: Windows, Ryzen9950X,
128 ГиБ RAM, RTX5090 32607 МиБ, driver581.80, CUDA13.0.48. Binary SHA-256:
`e844983bdfbb0754d39ec91bee828129a766a765ffae4f5f5882d3161d77354f`.
Параметры: context2048, batch17, F32 KV, FA off, TF32 off, greedy, reasoning low,
MTP/cache/pipeline off; 4 CPU threads для bookkeeping, tensor math на GPU.

[Короткий запуск](STEP37_FLASH_MODEL_SHORT.json) с max_new32 прошёл exact
сравнения, но обрывал reasoning. [Полный запуск](STEP37_FLASH_MODEL_VALIDATION.json)
с max_new128 завершил оба ответа native EOG128007: **54** generated tokens для
«2 + 2», итог **4**; **88** tokens для первых трёх простых чисел, итог **2, 3, 5**.
В обоих copy modes A→B→A exact. Native/pinned token IDs и полные F32 logits
побитово одинаковы. Это transport parity, не независимый CPU oracle полной модели.

| Запрос длинного запуска | Native decode, токен/с | Pinned decode, токен/с | Pinned prefill, с |
|---|---:|---:|---:|
| A, 38 prompt / 54 generated | 3,005 | 3,698 | 9,942 |
| B, 34 prompt / 88 generated | 3,220 | 4,054 | 4,513 |
| A повтор, 38 prompt / 54 generated | 4,630 | 4,569 | 3,485 |

Decode rate = forward steps / forward time: 53/87/53 steps, первый токен из
prefill не учитывается. В pinned run TTFT первых A/B — 9,956/4,516 с.
Порядок modes последовательный; состояние OS file cache различается. Это
наблюдения baseline, **не доказательство speedup** нового транспорта.

Пик process working set длинного pinned run **69,385 ГиБ**, sampled global
VRAM **12 560 МиБ**, sampled global occupied RAM **93,719 ГиБ**. Замеры глобальные
включают другие процессы; GPU polling раз в2 с может пропускать краткие пики.
Test monitor не зарегистрировал превышения95%; он завершает только свой engine
при превышении и не заменяет runtime admission/controller. Память ещё не
заполняется кэшем до разрешённого95%.

Pinned A передал **205 938 232 320 байт** за prefill+decode; повтор A столько же.
Source-copy time снизился с17,918 до8,874 с, H2D около5,05 с. Source bytes — чтение
mmap/RAM, **не измеренный SSD traffic**. У native control source/H2D counters
нулевые, поскольку этот путь не инструментирован.

**Команды / cwd / exit:** корень Strata, все финальные команды exit0;
CUDA DLL directories добавлены в PATH. Configure flags — в backend README.

```powershell
cmd /c build-local\build-step35-cuda.cmd
ctest --test-dir build-local/step35-cuda --output-on-failure --no-tests=error
python tools/check_step35_model.py --engine build-local/step35-cuda/bin/strata-step35.exe --gguf H:/models/Step-3.7-Flash/UD-Q4_K_S --output-dir build-local/step35-cuda/full-model-complete --predict 128
python tools/check_step35_model.py --engine build-local/step35-cuda/bin/strata-step35.exe --gguf H:/models/Step-3.7-Flash/UD-Q4_K_S --output-dir build-local/step35-cuda/full-request-check --request-reference build-local/step35-cuda/full-model-complete/model-report.json
python tools/check_step35_tokenizer.py --gguf H:/models/Step-3.7-Flash/UD-Q4_K_S --oracle build-local/step35-cuda/bin/strata-step35-tokenizer.exe --output docs/Step-3.7-Flash/STEP37_FLASH_RUNTIME_TOKENIZER.json
python tools/check_step35_template.py --gguf H:/models/Step-3.7-Flash/UD-Q4_K_S --oracle build-local/step35-cuda/bin/strata-step35-template.exe --tokenizer-oracle build-local/step35-cuda/bin/strata-step35-tokenizer.exe --output docs/Step-3.7-Flash/STEP37_FLASH_RUNTIME_TEMPLATE.json
python -m unittest tools.test_setup_step35 tools.test_step35_tokenizer
```

Raw logits и stderr остаются в ignored `build-local/step35-cuda/full-model-complete/`;
durable JSON reports и их hashes находятся в docs. Финальная пересборка обновила
описание scope в build record, engine binary SHA остался прежним.

**Совместимость и ограничения:** Qwen/DeepSeek/GLM sources, profiles и builds
не менялись. Shared reader/tokenizer hashes совпали с STEP-01. HTTP, tools,
runtime sessions, MTP, cache/pipeline, budget controller, OOM admission и длинные
full-model contexts не проверены. F16 KV проверен только на fixture. Нынешние
параметры не объявляются оптимальными defaults. Следующий шаг — P1.4/P3 по точке
продолжения: budget95 и bounded cache с exact bytes/logits до async delivery.

### STEP-05 / P1.4, P3 — 2026-10-05 17:04, Asia/Yekaterinburg — Bounded GPU cache

**Исполнитель:** Codex. **Статус:** DONE для sync cache, registry/P3.1 и
описанных memory checks. P1.4 и P3 целиком не закрыты: async pipeline, startup
OOM и реальные внешние pressure tests остаются. Pin и HEAD прежние.
[Build manifest](STEP37_FLASH_CACHE_BUILD_MANIFEST.json) содержит source/binary/
report hashes и команды. Финальный engine SHA-256:
`93a2d112db36e2528346b2908c3f622e65eb7896db8fbee8dd3af0545080b8ea`.

**Реализация:** Step `expert_cache.hpp`, `cache_registry.hpp`, `check_cache.cpp`,
новый `tools/check_step35_cache.py`; расширены Step sync runtime, engine metrics,
fixture checker, CMake и Python harness. Добавлен patch ID
`step-bounded-expert-cache`. Все generated units остаются внутри Step build.

Registry читает metadata всех четырёх shards, связывает каждый expert tensor
с фактическими shard/offset, type/shape и generation живой модели. Native
scheduler deduplicates выбранные IDs; cache хранит отдельные матрицы с MMQ tail.
Hit → D2D в native scratch; miss → прежний pinned16 МиБ путь → возможное
заполнение cache. Частота обращений с decay использует общий
`StrataExpertFrequencyHistory`; выбор eviction ограничен просмотром32 старых entries.
Каждый D2D завершён до reuse/eviction. Это **синхронная** реализация без leases
и без заявленного перекрытия compute/copy.

`--expert-cache-mib auto` ограничивает cache по глобальной VRAM; число задаёт
дополнительный cap в МиБ. Default остаётся **0**: первое заполнение увеличивает
prefill, оптимальный профиль по двум prompts не определяется. Native reference
mode работает с cache0. Состояние KV очищается после запроса, cache весов сохраняется.

Контроллер использует общие `StrataGlobalMemory`/`StrataVramPolicy`: на Windows
NVML по PCI, не per-process CUDA memory view. После fixed weights/context и
перед decode, а также каждых64 МиБ новых admissions, пересчитывается limit.
VRAM reserve =5%+256 МиБ; RAM admission оставляет5%+64 МиБ. Cache уменьшается
при давлении, allocation OOM даёт bypass в обычный selected-copy. Неверные или
недоступные samples и невозможное resident accounting запрещают дальнейшую работу.
Сэмплирование не резервирует память против других процессов и не гарантирует
startup/native CUDA OOM recovery; эти проверки остаются открытыми.

**Проверки финальной сборки:**

- [Transport/cache](STEP37_FLASH_CACHE_RUNTIME.json): **41/41 PASS**. Прежние27
  cases, плюс cold/warm cache и eviction для F32/mixed weights × F32/F16 KV через
  SWA boundary, отказ для незарегистрированных weights и восстановление после
  этой ошибки. Все GPU bytes проверяются observer-ом, logits bit-exact. Warm
  synthetic fixture помещается целиком: H2D/source bytes равны0, D2D ненулевой.
- [Cache/budget](STEP37_FLASH_CACHE_BUDGET.json): **11/11 PASS**. Реальные CUDA
  allocations, deterministic probes для pressure/invalid readings и OOM seam;
  trim, frequency admission, generation и accounting. Внешняя нагрузка здесь
  симулируется, не запускается конкурирующий процесс, занимающий всю VRAM.
- [Pipe](STEP37_FLASH_CACHE_PIPE.json): **22/22 PASS**. CUDA CTest **7/7 PASS**,
  включая [78 kernels](STEP37_FLASH_CACHE_KERNELS.json) и
  [75 graph/state cases](STEP37_FLASH_CACHE_GRAPH.json).
- [Tokenizer](STEP37_FLASH_CACHE_TOKENIZER.json) **2190/2190**,
  [template](STEP37_FLASH_CACHE_TEMPLATE.json) **215/215**, Python Step suite
  **21/21 PASS**. `py_compile` и `git diff --check` — exit0.

**Полная модель и скорость:**
[финальный raw report](STEP37_FLASH_CACHE_MODEL.json),
[вычисленная сводка](STEP37_FLASH_CACHE_SUMMARY.json). Стенд прежний:
Windows, Ryzen9950X, 128 ГиБ RAM, RTX5090 32607 МиБ, driver581.80, CUDA13.0.48.
UD-Q4_K_S106,323 ГиБ, context2048, batch17, F32 KV, FA/TF32/MTP off, greedy.
Те же A/B prompts и max_new128:38/34 входных,54/88 выходных tokens до EOG128007.

Четыре отдельных процесса в порядке **cache0 → auto → auto → cache0**;
в каждом A→B→A→B, затем STOP и свежий greedy prefix16. Все **16** полных ответов
и все F32 logits совпали с P1 reference; **4/4** cancel/recovery pairs прошли,
каждый процесс exit0. File cache не очищался, это не cold-SSD benchmark.

| Запуск | A первый, токен/с | B первый, токен/с | A повтор, токен/с | B повтор, токен/с |
|---|---:|---:|---:|---:|
| cache0, первый контроль | 3,438 | 3,880 | 4,327 | 4,350 |
| auto, первый запуск | 4,685 | 5,146 | 7,148 | 6,774 |
| auto, второй запуск | 4,324 | 4,912 | 6,753 | 6,674 |
| cache0, последний контроль | 3,286 | 3,753 | 4,279 | 4,296 |

Агрегат repeated decode = сумма forward steps / сумма forward time:
**4,315 → 6,806 токена/с**, отношение **1,577**, прирост **57,7%**. Суммарное
полное время четырёх repeated requests **80,507 → 55,666 с**: на **30,9%** меньше.
Первый sampled token исключён из decode throughput. Результат относится к этим
двум коротким prompts; не является оценкой скорости на всех задачах.

Первый A prefill ухудшился: **10,62/11,15 с** без cache против **15,58/16,68 с**
с cache. Это цена заполнения. У repeated A/B prefill с cache **3,25–3,87 с**.
Decode hit fraction repeated запросов **55,0–64,1%**, H2D для A снизился со
157,58 GB до56,64–58,97 GB за53 forward steps, для B с258,67 GB до113,50–116,42 GB
за87 steps. GB здесь десятичные bytes/1e9. Source bytes — mmap/RAM reads,
не измеренный SSD traffic. Новые counters разделяют prefill и decode.

В auto runs resident cache accounting **12,274–12,510 ГиБ**, sampled global
`nvidia-smi memory.used` до **30 229 МиБ**, global occupied RAM до **93,252 ГиБ**,
process peak working set до **69,413 ГиБ**. Controller `total-free` samples
до **30 736,36 МиБ** включают также reserved VRAM, поэтому отличаются от
`memory.used`. Target95 соблюдался по наблюдениям, guard не сработал; краткие пики
между samples не исключены. Driver/allocation overhead включён в global budget,
но не в logical entry payload; его уменьшение требует отдельного измерения.

**Контрольные итерации:**
[первый ABBA](STEP37_FLASH_CACHE_MODEL_INITIAL.json) и его
[provenance](STEP37_FLASH_CACHE_INITIAL_PROVENANCE.json) сохранены. Он уже проходил
exact checks; затем добавлены проверки inconsistent residency и destination shape,
recovery fixture и phase counters. Весь ABBA повторён на финальном binary;
числа в таблице относятся именно к нему. При сборке исправлены область видимости
импортированного CUDAToolkit target и ошибочное размещение fixture loop; итоговая
сборка и CTest успешны, неуспешная компиляция не считалась runtime validation.

**Команды / cwd:** корень Strata, финальные команды exit0. Build wrapper прежний,
`ctest --test-dir build-local/step35-cuda --output-on-failure --no-tests=error`.
Полный benchmark:

```powershell
python tools/check_step35_cache.py --engine build-local/step35-cuda/bin/strata-step35.exe --gguf H:/models/Step-3.7-Flash/UD-Q4_K_S --reference docs/Step-3.7-Flash/STEP37_FLASH_MODEL_VALIDATION.json --output-dir build-local/step35-cuda/cache-final-abba
```

Остальные команды, binary/source hashes и ограничения — в build manifest;
параметры запуска engine — в backend README. Raw stderr и последние logits каждой
сессии остаются в ignored `cache-final-abba/`, полные reference logits — в P1 directory.

**Совместимость:** общие primitives не менялись (сверены с HEAD), shared
`gguf_reader.py`/`strata_tokenizer.py` совпадают с STEP-04. Qwen/GLM/DeepSeek
profiles, sources и builds не менялись. Старые GLM profile failures не исправлены
и не объявляются PASS. HTTP, tools, MTP, long context, async leases и timeline
не входят в этот шаг. Следующий шаг — async delivery по точке продолжения выше.

### STEP-06 / P3.2–P3.6 — 2026-10-05 — Асинхронная доставка экспертов

**Исполнитель:** Codex. **Статус:** DONE для router lookahead и bounded async
delivery; P3.2 закрыт. P3.3/P3.4/P3.6 частично выполнены, P1.4/P3.5 остаются открыты.
**Cwd/база:** корень Strata, `97e016de11754cccd1b5550232c6c198cdbe0723`.
[Build manifest](STEP37_FLASH_PIPELINE_BUILD_MANIFEST.json) содержит source/binary
SHA, неизменённый pin, generated patches, команды и checks. Финальный engine SHA:
`0187b423b9ebe10832696403cff12e1561dd9dab750845080e744643ae724b00`.

**Реализация:** Step adapter использует общий `StrataExpertPipeline`, не меняя его.
После фактических router IDs строится план gate/up/down в порядке native scheduler.
Будущие cache hits закрепляются residency pins; misses заранее читаются в четыре
pinned slots и загружаются в отдельное GPU ring на независимом stream. Ready/used
events защищают reuse. Scheduler пишет scratch только после предыдущего consumer.
D2D из ring/cache в scratch и cache-fill пока синхронные; async cache leases
не заявляются. Отмена завершает workers/source reads до освобождения mappings.

Добавлены `pipeline_runtime.inc`, `pipeline_sched.inc`, `gpu_trace.hpp` и ring
checker; обновлены Step cache/runtime, CLI, CMake patches и Step checkers.
CLI: `--expert-pipeline-readers 0|1|2`, `--expert-pipeline-chunk-mib 4|8|16`.
Defaults cache0/pipeline0 сохранены. INFO сообщает настоящий размер ring и режим.
Ring резервируется до cache admissions; global memory controller учитывает его.

**Финальная проверка:** CTest **10/10 PASS**, [runtime](STEP37_FLASH_PIPELINE_RUNTIME.json)
**92/92**, [ring](STEP37_FLASH_PIPELINE_RING.json) **18/18**,
[cache/budget](STEP37_FLASH_PIPELINE_BUDGET.json) **14/14**.
Pipe **22/22** в каждом режиме: [off](STEP37_FLASH_PIPELINE_PIPE_OFF.json),
[1 reader](STEP37_FLASH_PIPELINE_PIPE_1.json), [2 readers](STEP37_FLASH_PIPELINE_PIPE_2.json).
[Kernels](STEP37_FLASH_PIPELINE_KERNELS.json) **78/78**, [graph/state](STEP37_FLASH_PIPELINE_GRAPH.json)
**75/75**, [tokenizer](STEP37_FLASH_PIPELINE_TOKENIZER.json) **2190/2190**,
[template](STEP37_FLASH_PIPELINE_TEMPLATE.json) **215/215**, Python Step suite **21/21**.

Fixture покрывает readers 1/2 × chunks 4/8/16 × cache 0/1/64 МиБ, mixed/F32 weights,
F32/F16 KV, SWA boundary, exact GPU bytes/logits, all-hit/no-H2D, eviction,
ошибочный registry и восстановление. Ring test проверяет 18 МиБ + 513 bytes,
guard bytes, wrap без host sync на каждый chunk, неверный source, cancellation
загруженного suffix и restart с другим содержимым. Это не raw-byte тест реальных
GGUF shards; отдельный sampler остаётся следующим шагом.

**Полная модель:** исходный [reader sweep](STEP37_FLASH_PIPELINE_MODEL_INITIAL.json)
`0→1→2→0`, затем [финальный ABBA](STEP37_FLASH_PIPELINE_MODEL.json) `0→1→1→0`.
В каждом процессе A→B→A→B, STOP и свежий greedy prefix16. Все **32** завершённых
ответа и полные F32 logits совпали с P1; **8/8** cancel/recovery, процессы exit0.
Модель та же UD-Q4_K_S 106,323 ГиБ, четыре shards; context 2048, batch 17, F32 KV, FA/TF32/MTP off.
Вход A/B 38/34 tokens, выход 54/88 до EOG 128007. Cache auto во всех сравнениях.
RTX 5090 / 128 ГиБ / Ryzen 9950X, Windows, driver 581.80, CUDA 13.0.48.
Файловый кэш не очищался; source bytes не равны SSD traffic.

Финальная скорость decode в токенах/с, первый sampled token исключён:

| Режим / процесс | A первый | B первый | A повтор | B повтор |
|---|---:|---:|---:|---:|
| sync, 1 | 4,454 | 4,813 | 6,362 | 5,923 |
| pipeline1, 2 | 5,714 | 6,217 | 8,289 | 6,826 |
| pipeline1, 3 | 4,724 | 4,958 | 6,450 | 6,074 |
| sync, 4 | 3,176 | 3,826 | 6,386 | 6,427 |

Агрегат сумма decode steps / сумма forward time: **6,242 → 6,718 токена/с (+7,6%)**.
Полное время четырёх repeated requests **60,356 →
54,347 с**, меньше на **10,0%**.
Разброс между процессами существенный; это измерение двух коротких prompts,
не универсальная оценка скорости. Первый A prefill: sync 16,42/20,36 с,
pipeline 10,74/11,83 с. Исходный sweep дал 6,671 (sync), 7,473 (reader 1), 6,720 (reader 2)
токена/с на repeated decode; финальный ABBA подтверждает меньший средний выигрыш.
Настройка 1 reader / 8 МиБ остаётся экспериментальным opt-in, chunks 4/16 требуют
отдельного full-model tuning. [Сводка](STEP37_FLASH_PIPELINE_SUMMARY.json).

В финальных runs sampled global GPU memory.used до **30314 МиБ**, global
occupied RAM до **107,297 ГиБ**, process peak working set до **69,426 ГиБ**.
Logical cache **9,809–9,978 ГиБ**; ring 32 МиБ pinned RAM
+32 МиБ VRAM при chunk 8. Controller total-free также включает reserved VRAM,
поэтому отличается от memory.used. Guard 95 не сработал. Это samples, а не
гарантия отсутствия коротких пиков или внешних concurrent allocations.

**CUDA timeline:** отдельный [отчёт](STEP37_FLASH_PIPELINE_TRACE_SUMMARY.json) и
[интервалы](STEP37_FLASH_PIPELINE_TRACE.json), 8 graphs, тот же полный ответ/logits exact.
События стоят на фактических H2D и backend compute streams. Пересечение:
prefill **0,261 мс** за 3 graphs,
decode **0,113 мс** за 5 graphs.
Это всего 0,14%/0,18% измеренных compute intervals: существенного GPU overlap
пока нет. Выигрыш нельзя объяснять большим перекрытием H2D с вычислениями.
Синхронный D2D/scratch bridge остаётся ограничением; его снятие требует
отдельных consumer leases/events и проверки aliasing. События ограничивают
stream intervals, а не заменяют профилирование времени каждого CUDA kernel.
Trace синхронизирует boundaries и не входит в throughput numbers.
CPU read/slot/consumer counters не выдаются за GPU execution duration.
Pipeline h2d_ms=0 означает отсутствие legacy timer; d2d_ms не включает ring D2D.

**Промежуточные исправления:** первый compile потребовал vendor include для JSON;
финальные build/CTest exit0. В initial engine INFO оставался sync label/16 МиБ,
это исправлено; transport не менялся, финальный ABBA выполнен заново.
[Initial provenance](STEP37_FLASH_PIPELINE_INITIAL_PROVENANCE.json) сохраняет связь
первого отчёта с его binary. Новые kernel numerical tolerances не вводились.

**Осталось:** raw bytes из реальных shards, реальные external pressure/startup OOM,
long-context/KV growth, early refill и другие chunk sizes в full-model timing.
Непредвиденная CUDA worker failure требует пересоздать pipeline; обычная отмена
и ошибка plan проверены. HTTP/tools/sessions/MTP пока не подключены.
Общие primitives совпадают с HEAD, reader/tokenizer — с STEP-05. Qwen/GLM/DeepSeek
sources/profiles/builds не менялись; старые GLM profile failures не объявляются PASS.

### STEP-07 / P3.4–P3.5 — 2026-10-05 — Реальные веса, контекст и внешняя нагрузка

**Исполнитель:** Codex. **Статус:** DONE для P3.4 и ограниченного pressure/context
сценария. P1.4/P3.5 остаются IN_PROGRESS: startup OOM и предельные режимы не проверены.
**База/cwd:** `97e016de11754cccd1b5550232c6c198cdbe0723`, корень Strata.
[Manifest](STEP37_FLASH_STEP07_BUILD_MANIFEST.json): source/binary/report hashes,
команды, exit codes и связь с STEP-06. Engine SHA остался
`0187b423b9ebe10832696403cff12e1561dd9dab750845080e744643ae724b00` — основной engine не пересобирался.
Добавлены отдельные `check_shards.cpp`, `pressure_holder.cpp`, Python runners
и output-guard tests, CMake targets; существующий transport/compute code не менялся.

**Реальные shards:** [973/973 byte checks](STEP37_FLASH_STEP07_SHARDS.json) PASS,
[81 samples и payload SHA](STEP37_FLASH_STEP07_SAMPLES.json), всего 238 906 368 байт.
Для каждого из трёх payload shards взяты gate/up/down из первых/средних/последних
доступных слоёв, эксперты 0/144/287. Покрытие слоёв: shard1 — 3/12/20;
shard2 — 21/30/38/39; shard3 — 39/40/42/44. Нумерация здесь GGUF split.no,
то есть файлы 00002/00003/00004; metadata-only shard0 не содержит весов.

Python и native GGUF reader независимо проверяют offsets/strides. Native checker
сопоставляет mmap bytes, отдельный file read и GPU результат. Chunks 4/8/16 МиБ,
readers 1/2, два consumer streams, cold и pinned cache hits, 512-byte tails,
256-byte guards, trim после unpin и reload mappings. Все 81 samples прошли 12
сравнений, плюс reload. Общий pipeline и production cache используются без копии
их реализации. Полных checksums всех 106,323 ГиБ эти samples не заменяют.

**Контекст/pressure:** [полный report](STEP37_FLASH_STEP07_CONTEXT.json),
[сводка](STEP37_FLASH_STEP07_SUMMARY.json). Два последовательных процесса:
sync и pipeline1/chunk8, cache auto, context4096, batch17, F32 KV, FA/TF32/MTP off.
Стенд прежний: Windows, Ryzen 9950X, 128 ГиБ, RTX 5090 32607 МиБ, driver581.80,
CUDA13.0.48. Один и тот же запрос на 2591 входной токен пересекает прежний лимит
2048 и несколько границ SWA512. Сохранены 16 output tokens и все их F32 logits;
pipeline совпал с sync побайтно. Это численная проверка длинного запроса,
а не проверка законченного ответа или качества модели.

В каждом процессе: короткий P1 ответ → длинный запрос → короткий перед нагрузкой
→ под нагрузкой → после освобождения → STOP длинного prefill → свежий короткий.
Все **12** request checks PASS, включая 10 законченных P1 ответов и 2 длинных
prefixes. Все короткие logits совпали с исходным P1 reference при context2048.
STOP/recovery **2/2**, engine и holder exit0. STOP отправлялся после первого PP,
поэтому cancellation после rollover длинного контекста отдельно не заявляется.

| Режим | Prefill 2591 токенов, с | Выход | Уменьшение кэша под нагрузкой, МиБ |
|---|---:|---:|---:|
| sync | 545,66 | 16 | 91,81 |
| pipeline1 | 408,39 | 16 | 117,75 |

Это последовательные времена выполнения, не paired throughput benchmark.
Файловый кэш не очищался, длинный текст состоит из повторяющихся records.
Новый выигрыш скорости или оптимальные настройки по этим двум временам не заявляются.

Отдельный holder до старта модели создаёт CUDA context; после её прогрева
выделяет/заполняет **128 МиБ VRAM + 2 ГиБ RAM**. До/после allocations проверяется
запас относительно 95%; при недостатке памяти тест отказывается от нагрузки.
Кэш реально уменьшился в обоих modes, численный результат сохранился. После
FREE память helper освобождена, следующий запрос exact. EOF/error также снимает
нагрузку; runner завершает только свои дочерние процессы.

Максимальные sampled global memory.used по двум запускам:
GPU **30260 МиБ**,
occupied RAM **115,449 ГиБ**;
process peak working set **89,937 ГиБ**.
Guard 95 не сработал. NVML total-free включает также reserved VRAM; sampling не
гарантирует отсутствие кратких пиков. Это умеренная реальная внешняя нагрузка,
не исчерпание ресурсов и не startup OOM.

**Найденный кандидат оптимизации:** в sync long request доля prefill cache hits
**4,8%**, суммарный cache-fill D2D **1,36 ТБ**
(накопленный traffic за запрос, не размер VRAM). Source reads составили
**3,73 ТБ**; legacy source_ms **212,59 с**,
H2D wall timer **98,28 с**, cache D2D wall timer
**73,95 с**. Эти counters включают 16-token output и
не равны времени исполнения GPU kernels. Следующая гипотеза — ограничить
admission во время prefill, сохранив cache reads и обычное decode admission.
Нужно измерить полный запрос: ускорение prefill может изменить первый decode.

**Проверки:** CUDA CTest **10/10 PASS**, прежний Python Step suite **21/21**,
output-guard test **1/1** с 4 негативными subcases: alias GGUF, alias manifest,
неверное расширение и безопасный error report. Исходный input остаётся неизменным.
Первый вариант длинного prompt был отклонён preflight до загрузки модели,
поскольку не превышал 2048 tokens; после увеличения до 128 records проверены 2591.
Финальные команды exit0; py_compile, ссылки docs и hashes проверены.

**Совместимость/границы:** Step engine binary и прежние runtime sources совпадают
с STEP-06, общие primitives и reader/tokenizer не менялись. Профили Qwen/GLM/DeepSeek
не затрагивались. Не проверены startup OOM, максимум 32768, принудительное
исчерпание RAM/VRAM, stop после длинного rollover. HTTP/tools/sessions/MTP остаются
следующими этапами. Предыдущий measured decode 6,718 токена/с относится к STEP-06;
в этом шаге скорость engine не изменялась.

### STEP-08 — 2026-10-05 — Ограничение заполнения кэша во время prefill

**Исполнитель / статус:** Codex / DONE в границах отдельной опции и проверок;
P3 остаётся IN_PROGRESS. Полная поддержка HTTP/tools/sessions/MTP не заявляется.
HEAD `97e016de11754cccd1b5550232c6c198cdbe0723`; pin прежний, новый patch `step-prefill-cache-admission`.
Engine SHA-256 `32a88bd85855395dbc79df2da323d971a9543c2847c618e7e34245a771bf9df3`.
Модель — прежние четыре UD-Q4_K_S shards; структурный fingerprint не менялся.

**Реализация:** `--expert-cache-prefill on|off`, default `on`. Режим `off`
сохраняет cache hits и frequency learning, но пропускает admission новых entries
во время prefill. Decode возвращает обычное заполнение. Одна функция используется
sync/pipeline путями. Фаза задаётся владельцем запроса, включая one-token tail;
RAII восстанавливает её при завершении, STOP и исключении. Memory checks и
CUDA/scratch synchronization сохранены. Общие primitives и другие backend не менялись.

Добавлены split prefill/decode fill counters и skipped-admission counters.
Skipped bytes — объём кандидатов, а не доказанная экономия: обычный admission
также мог бы их отклонить. Новый checker требует новый output directory,
сохраняет raw F32/hash/metrics и ограничивает только свои процессы guard95.

**Измерение:** Windows, Ryzen9950X, 128 ГиБ RAM, RTX5090 32 ГиБ, CUDA13.0.48.
Context4096, batch17, F32 KV, FA/TF32/MTP off, auto cache, pipeline1/8 МиБ.
Порядок admission **on/off/off/on**, по два процесса на вариант; внутри каждого
A → B → medium511/16 → A → A, затем STOP и отдельное recovery A.
File cache не очищался; фоновые приложения не выключались. Таблица — среднее
время двух запусков, decode — сумма forward steps / сумма forward time.

| Запрос | Prefill on → off, с | Полный запрос on → off, с | Decode on → off, токена/с |
|---|---|---|---|
| Первый A | 10,619 → 9,272 | 20,281 → 18,523 | 5,504 → 5,750 |
| B после A | 3,869 → 3,417 | 18,710 → 17,292 | 5,883 → 6,294 |
| 511-token prompt + 16 output | 57,376 → 40,281 | 60,137 → 43,550 | 5,459 → 4,608 |
| A после medium | 3,020 → 2,646 | 10,393 → 10,017 | 7,220 → 7,223 |
| Повтор A | 2,531 → 2,234 | 8,606 → 8,514 | 8,773 → 8,505 |

Для medium prefill time уменьшилось на **29,8%**,
полный request wall — на **27,6%**.
Средний prefill cache-fill D2D: **98,128 → 0 ГиБ**
накопленного traffic. Общая последовательность пяти запросов:
**118,127 → 97,895 с**
(−17,1%).
После medium decode замедлился 5,459 →
4,608 токена/с: заполнение кэша
переносится на генерацию. Поэтому это не универсальный выигрыш decode; default
сохранён. Для этого измеренного workload рекомендуется opt-in `--expert-cache-prefill off`.
Два запуска на режим не дают статистической гарантии для всех промптов.

**Точность и длинный запрос:** 24 законченных P1 ответа совпали по всем F32
logits/IDs; 3 medium prefixes совпали с новым контрольным prefix (первый control
не считается независимым сравнением). Отдельный long2591/16 совпал с STEP-07.
Итого **28 reference comparisons PASS**, плюс один новый control; **5/5**
STOP/recovery pairs. Long prefill **215,436 с**, request wall
**219,155 с** — отдельное наблюдение, без paired speed claim.
16 output tokens у medium/long являются prefixes, не законченными ответами.

Пики samples по пяти процессам: global GPU used **30285 МиБ**,
global occupied RAM **110,507 ГиБ**,
process peak working set **89,937 ГиБ**.
Guard95 не сработал; все процессы exit0. Это sampled значения с учётом прочих
приложений, без гарантии отсутствия кратких пиков и без forced OOM.

**Проверки:** CTest **13/13**: runtime **111**, kernels78, graph75, ring18,
budget14, **6×22** pipe checks (sync/readers1/readers2 × default/prefill-off).
Новые runtime cases: cold/warm, prompts1/18/513, one-token tail, SWA crossing,
prefill fill=0, decode fill>0, cache hits при запрете admission, реальные GPU
bytes и exception unwind. Python **22/22**, tokenizer **2190/2190**, template
**215/215**, py_compile PASS. Полные команды, hashes и exit codes — в manifest.

Артефакты: [summary](STEP37_FLASH_ADMISSION_SUMMARY.json),
[paired requests](STEP37_FLASH_ADMISSION_MODEL.json),
[long request](STEP37_FLASH_ADMISSION_LONG.json),
[runtime](STEP37_FLASH_ADMISSION_RUNTIME.json),
[build manifest](STEP37_FLASH_ADMISSION_BUILD_MANIFEST.json).
Прежние STEP-06/07 отчёты сохранены. Defaults cache0/readers0/prefill-on сохранены.

**Следующий шаг:** STEP-09 / P2.2–P2.3 — встроенный template и отдельный streaming
parser reasoning/XML tools с boundary fixtures. Затем P2.4–P2.6 HTTP/cancel gates.
Performance вопросы: chunk4/16 и ранний refill; startup OOM, расширенная pressure
и asynchronous cache leases остаются незакрытыми пунктами P1/P3.

### STEP-09 — 2026-10-05 — Step template и потоковый reasoning/tool parser

**Исполнитель / статус:** Codex / DONE для P2.2–P2.3 в отдельном adapter/native pipe.
P2 остаётся IN_PROGRESS: HTTP-профиля ещё нет. HEAD и native engine прежние;
engine SHA `32a88bd85855395dbc79df2da323d971a9543c2847c618e7e34245a771bf9df3`.
Общие frontend/server, GLM/DeepSeek adapters, CUDA/cache/pipeline не изменялись.

**Код:** [serve/step35.py](../../serve/step35.py),
[unit tests](../../serve/test_step35.py), точный 5723-byte
[GGUF template fixture](../../serve/fixtures/step37_chat_template.jinja),
[real-model checker](../../tools/check_step35_frontend.py).
`check_step35_template.py --runtime-adapter` сравнивает production renderer
с прежним независимым native Jinja/tokenizer oracle. Fixture не является fallback:
адаптер принимает извлечённый template path и BOS/EOS выбранной модели,
проверяет SHA, не подставляет Qwen/GLM template.

OpenAI/Anthropic normalization сохраняет IDs и reasoning, low/medium/high
без преобразования high→xhigh. При отсутствии effort не добавляется новая
инструкция; шаблон всё равно открывает `<think>`. Text blocks сохраняют native
пробел-разделитель; media отвергаются. String JSON arguments нормализуются
в object; history не изменяется. `fromjson` добавлен только Step environment.
Reasoning прошлого user turn удаляет сам GGUF template, текущий tool turn
сохраняется. Tool results следуют порядку прихода, без GLM-style сортировки по IDs.

Парсер разделяет reasoning/content, XML function/parameter calls и несколько
вызовов. Использует объявленные базовые типы schema, сохраняет строки, понимает
native scalar `True`/`False`/`None` и JSON. Calls появляются только после полного
закрытия и проверки; truncated/malformed остаются текстом. Проверены framing
newlines, literal tags, UTF-8, повторный `finish`, дубликаты XML аргументов,
нечисловые NaN/overflow и несовпадение declared type. Полная JSON Schema validation
(required/ranges/refs) и ранние tool_start/tool_args не реализованы. Nullable
string union сохраняет literal; raw delimiter sequence внутри строки неоднозначна.
Подробности и ограничения — [fixture README](../../serve/fixtures/README.md).

**Проверки:** production adapter **215/215** native render/token-ID checks;
208 ранее сохранённых render hashes также совпали. Общий Python suite
**148/148**, без failures/errors/skips: **22** новых Step tests, Qwen frontend
classes, GLM/DeepSeek adapters/Service/handlers/MCP и прежние Step setup/tokenizer
checks. Parser fixtures выполнили **3415** chunk sequences: все двухчастные
границы выбранных текстов, widths1/2/3/7/19, оборванные calls и byte-split UTF-8.
Это тесты конечного набора форматов, а не доказательство для всех XML/JSON.

**Полная UD-Q4_K_S:** тот же Windows/9950X/128 ГиБ/RTX5090. Context4096,
batch17, F32 KV, cacheauto, pipeline1/8 МиБ, prefill admission off, MTP/FA/TF32 off.
1. P1 A: **38 input /54 output**, прежние IDs и полные F32 logits bit-exact;
   парсер отделил reasoning и ответ `4`.
2. Tool request: **299 input /146 output**, модель завершила один
   `lookup_weather(city="Yekaterinburg", days=2, include_wind=true)`;
   типы string/integer/boolean сохранены.
3. Continuation: **496 input /140 output**, локальная заглушка результата
   инструмента привела к законченному обычному ответу о 7–8°C, без нового вызова.

Все три prompt bytes/IDs совпали с native oracles до генерации. Parsing совпал
на реальных token chunks, whole text и single-character chunks. После небольшого
ужесточения JSON duplicates/error-result normalization финальный adapter повторно
воспроизвёл все сохранённые prompts/IDs/output semantics; новый GPU-прогон для
этого replay не выполнялся. Engine exit0; guard95 не сработал. Sampled global
GPU peak **30244 МиБ**, occupied RAM **114.287 ГиБ**. Замеров ускорения
в этом этапе нет; native binary и STEP-08 настройки не менялись.

Это native pipe dialogue с локальной строкой tool result. Внешний weather/MCP
инструмент не вызывался. HTTP/SSE, Service stop-token resolver, capabilities,
profile/setup и disconnect ещё не проверены для Step. Нельзя использовать Qwen
stop defaults: native EOG **1/128007**, PAD2 исключён, resolve нужен отдельно.

Артефакты: [native template](STEP37_FLASH_FRONTEND_TEMPLATE.json),
[tests](STEP37_FLASH_FRONTEND_TESTS.json),
[real model](STEP37_FLASH_FRONTEND_MODEL.json),
[final-source replay](STEP37_FLASH_FRONTEND_REPLAY.json),
[manifest](STEP37_FLASH_FRONTEND_MANIFEST.json).

**Следующий конкретный шаг:** STEP-10 / P2.4–P2.6: явный EOG resolver и capabilities,
выбор Step template только для architecture=step35, JSON/SSE обоих API,
stop/length/tool_calls и HTTP cancel/disconnect с чистым следующим запросом.
Сначала mock/handler regression, затем отдельный loopback HTTP full-model smoke;
активные профили существующих моделей сохраняются.

## Шаблон следующей записи

```text
### Pn.m-XX — YYYY-MM-DD HH:MM, Asia/Yekaterinburg — Название
Исполнитель / статус:
Ревизия Strata и локальные изменения:
Backend pin / patch manifest / binary SHA:
Модель и fingerprints:
Изменённые файлы:
Команды, cwd, exit codes:
Проверки: fixture / полная модель / HTTP / performance:
Фактические результаты и существующие артефакты:
Ограничения, оставшиеся вопросы:
Следующий конкретный шаг и его критерий готовности:
```


## STEP-10 — HTTP API, EOG и отдельный экспериментальный профиль

**Изменено:** явный `architecture=step35` в configured_template; проверка native
control tokens и metadata до запуска. EOG **[1,128007]**, PAD2 исключён. Отдельный
Step parser factory обрабатывает stop strings до reasoning/tool parsing, включая
границы токенов и перекрывающиеся строки. OpenAI возвращает `stop`, Anthropic —
`stop_sequence` с совпавшей строкой; length/tool_calls проверены в обоих API.
Low/medium/high effort и replay reasoning сообщаются через capabilities;
отключение thinking, clear-thinking и hard budget не поддерживаются и отклоняются.
При отсутствии effort embedded prompt сохраняется; default high относится к UI.

`tools/prepare_step35_profile.py` создаёт только новый каталог и проверяет
structural fingerprint модели, source/patch identity engine, экспорт tokenizer
и template hash. Основной setup и существующие профили не переключены.
Рабочий experimental config:
`build-local/step35-cuda/step10-http-profile-8g/step37.json`.
Context4096, batch17, F32 KV, cache cap8192 MiB, pipeline1, chunk8 MiB,
prefill admission off, greedy, MTP off. Это ограниченный проверенный профиль,
не подбор максимальной скорости и не готовая поддержка всех Step-вариантов.

**Проверки:** 232 tests PASS без пропусков: полный `serve.test_server`, Step,
DeepSeek/GLM, HTTP loopback и profile refusal. После final stop-overlap/cap edits
повторены 34 Step/profile tests PASS. Два старых GLM profile failures не включены;
их baseline описан в предыдущих шагах. Exported tokenizer: **2190/2190** exact
native IDs/bytes; новый pack побайтно равен проверенному. Final-source replay:
3 prompts и 9 parser comparisons PASS. CLI `serve.server --lazy` проверен с
готовым профилем: health/models/settings и web HTML (24692 bytes); browser UI
не управлялся. Local in-memory MCP continuation проверен, внешних вызовов нет.

**Полная модель:** Windows/9950X/128 GiB/RTX5090, прежний engine SHA
`32a88bd85855395dbc79df2da323d971a9543c2847c618e7e34245a771bf9df3`. Четыре P1 HTTP запроса покрывают OpenAI/Anthropic
JSON/SSE. Затем OpenAI tool call → Anthropic streamed continuation с локальным
результатом. Все входные/выходные IDs совпали с STEP-09. После каждого из трёх
socket disconnect следующий P1 также exact. Всего **9 полных exact responses**.

| Запрос | Input / output с EOG | Prefill, с | Decode, токена/с | HTTP wall, с |
|---|---:|---:|---:|---:|
| P1_openai_False | 38 / 54 | 25.806 | 4.513 | 37.557 |
| P1_openai_True | 38 / 54 | 2.228 | 8.397 | 8.544 |
| P1_anthropic_False | 38 / 54 | 2.228 | 8.969 | 8.141 |
| P1_anthropic_True | 38 / 54 | 2.244 | 9.242 | 7.983 |
| tool_call_openai | 299 / 146 | 28.938 | 6.270 | 52.070 |
| tool_result_anthropic | 496 / 140 | 34.040 | 6.648 | 54.955 |
| after_prefill | 38 / 54 | 2.456 | 7.425 | 9.598 |
| after_decode | 38 / 54 | 2.201 | 8.514 | 8.429 |
| after_partial_tool | 38 / 54 | 2.467 | 7.707 | 9.348 |

Decode = (output−1)/engine decode wall; это последовательный functional smoke,
не paired benchmark и не доказательство ускорения относительно STEP-08.

Отмена до чистого DONE/FIFO release: prefill: 1.762 с, 0 output tokens до STOP; decode: 0.302 с, 5 output tokens до STOP; partial_tool: 0.504 с, 109 output tokens до STOP. Native finish всех отмен —
`cancel`. Engine завершился **exit0**; memory guard не сработал. Пики:
**RAM 118.408 GiB (94.31% от 125.555 GiB visible)**,
**VRAM 28931 MiB (88.73%)**.

**Почему cap8 GiB:** прежняя серия с auto cache14427 MiB завершила 9 запросов
и 3 отмены, но сам checker ошибся при чтении process после close(). Ошибка
checker исправлена. Повтор с auto17121 MiB пересёк guard95 по RAM; ещё один
с cap14336 MiB также пересёк его после серии отмен. Монитор остановил только
свой engine. Эти попытки сохранены в HTTP_ATTEMPTS, не засчитаны как PASS.
Cap8192 MiB выбран для запаса RAM и прошёл весь текущий сценарий; оптимальность
не доказана. Больший cache требует отдельной работы над RAM margin/pressure.

**Артефакты:** `STEP37_FLASH_HTTP_{TOKENIZER,TESTS,STARTUP,REPLAY,MODEL,ATTEMPTS,MANIFEST}.json`.
Raw финальный прогон: `build-local/step35-cuda/step10-http-model-r5/`.
Checker: `tools/check_step35_http.py`; воспроизведение — в backend README.

**Остаётся:** интерактивный web chat, HTTP unload/reload, отмена выполняющегося
внешнего MCP и длинного HTTP rollover; persistent session cache/SWA rollback,
MTP/vision и расширенные RAM-pressure сценарии. READY `session-id` означает
приём ID для изоляции запросов; INFO `conversation_cache=0` не обещает KV reuse.

## STEP-11 — native MTP trials (2026-10-05)

По запросу пользователя проверен готовый Step MTP driver из закреплённой
зависимости и официальный Q8_0 sidecar. Реализация изолирована в
`backends/step35/check_mtp.cpp`, harness — `tools/check_step35_mtp.py`.
Опция сборки `STRATA_STEP_MTP_PROBE` по умолчанию OFF. Рабочий pipe engine,
экспериментальный HTTP-профиль STEP-10 и остальные backends не изменялись.

Подробности, команды, ограничения:
[STEP37_FLASH_MTP_TRIALS.md](STEP37_FLASH_MTP_TRIALS.md).
IDs, timings, source/binary/sidecar hashes, memory peaks и неудачные попытки:
[STEP37_FLASH_MTP_TRIALS.json](STEP37_FLASH_MTP_TRIALS.json).

**Итог:** 100 завершённых запросов на двух P1 prompts, из них60 с MTP;
все IDs совпали с native reference. Проверены off/1/2/3, common embedding,
неактивные головы в RAM и ограниченный prefault экспертных страниц.
Финальная серия, context2048/batch17/F32/pipeline1/cap16384MiB:

| Режим | Decode, токена/с | Средняя пара запросов вместе с prefill |
|---|---:|---:|
| Без загрузки draft | 9,146 | 19,972с |
| Резидентный Q8, MTP2 | 10,274 | 18,453с |
| Q8, MTP2, shared embedding | **10,602** | **17,859с** |

Финальный кандидат: depth2, `p_min=0.6`, shared embedding, demand paging.
Это **+15,9% decode**, полная пара запросов **на10,6% быстрее**, acceptance92,55%.
Замеры после прогрева; переносить эти проценты на другие prompts пока нельзя.
На RTX5090 наблюдаемый power limit400W, driver581.80; параметры питания не менялись.

RAM удалось загрузить до **117,607ГиБ /93,67%**, VRAM до30216MiB.
Прогрев RAM не установил дополнительного выигрыша по throughput, увеличил startup
и остаётся opt-in. Запасы512/8192MiB при prefault привели к RAM>95%; monitor
остановил только свой child. Запас24576MiB прошёл короткий sweep. Матрицы
неиспользуемых голов в RAM дали9,33–9,43 токена/с; optional catch-up patch
по умолчанию OFF, поскольку ускорение не подтверждено.

**Проверки:** 24 существующих Step Python tests PASS, сборки upstream и optional
active-head catch-up PASS, guards CLI и `git diff --check` PASS. Исходный
`strata-step35.exe` сохранил SHA-256
`32a88bd85855395dbc79df2da323d971a9543c2847c618e7e34245a771bf9df3`.

**Следующий шаг:** P5.2–P5.5 ещё не закрыты целиком. Checker ограничен480
суммарными позициями и greedy. Нужны SWA512/full KV rollback обеих contexts,
длинные prompts, stop/cancel, stochastic sampling и интеграция в pipe/HTTP.
В рабочем приложении MTP остаётся выключенным.

## STEP-12 — GPU allocation reuse (2026-10-05)

Реализован opt-in reuse GPU-блока вытесняемой записи экспертного кэша. Размер
allocation должен совпадать, запись не должна быть закреплена plan, предыдущее
чтение завершено. Frequency admission, trim, growth checks и budget95 сохранены;
отдельного свободного пула нет. В обычном runtime опция по умолчанию выключена,
управление добавлено в offline checker через `--cache-reuse off,on`.

На Windows/9950X/128GiB/RTX5090, context2048/batch17/F32/pipeline1/cachecap16384,
два P1 prompts, warmup и четыре раунда с обратным порядком режимов:

| Режим | Reuse off → on, токена/с | Прирост | Пара запросов с prefill |
|---|---:|---:|---:|
| Без загрузки draft | 9,363 →9,873 | +5,45% | 19,312 →18,402с |
| Q8 MTP2/shared embedding/p_min0.6 | 10,355 →11,560 | +11,64% | 18,176 →16,742с |

**40/40** responses exact, включая warmup;20 с MTP. Все четыре парных раунда
показывают выигрыш. MTP acceptance92,55% сохранился. Новых allocations на decode
в среднем622 →24,625 без MTP и1385,625 →51,750 с MTP. H2D-трафик практически
тот же. В MTP серии контроль прогревался; последние два раунда дают +10,3%.
Проценты не являются обещанием для произвольного диалога.

Пики: RAM88,900/93,370GiB, VRAM30220/30234MiB для no-MTP/MTP2; guard95 не
сработал. Дополнительное заполнение RAM не использовалось.23 CUDA cache checks,
117 runtime cases (включая6 новых byte/bit-exact logits reuse вариантов) и24
существующих Step Python tests PASS. Native `strata-step35.exe` имеет прежний
SHA-256 `32a88bd85855395dbc79df2da323d971a9543c2847c618e7e34245a771bf9df3`.

Отчёт и команды: [STEP37_FLASH_CACHE_REUSE.md](STEP37_FLASH_CACHE_REUSE.md).
Все IDs/timings, hashes и checks: [STEP37_FLASH_CACHE_REUSE.json](STEP37_FLASH_CACHE_REUSE.json).
Raw: `build-local/step35-cuda/step12-speed/`. Проверенный probe сохранён как
`strata-step35-mtp-check-cache-reuse-tested.exe` в `build-local/step35-cuda/bin/`.

Следующий шаг — admission opt-in reuse в pipe/HTTP: длинные запросы,
cancel/recovery и pressure. Затем выбирать default. Следующие performance
кандидаты: объединение D2D-синхронизаций с event/lease защитой и early host refill;
их дополнительный выигрыш пока не измерен. P5 state/SWA rollback и production
MTP integration остаются отдельными обязательными этапами.
