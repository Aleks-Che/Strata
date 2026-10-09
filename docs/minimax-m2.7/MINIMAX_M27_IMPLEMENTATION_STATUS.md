# Статус внедрения MiniMax-M2.7

Обновлено: **2026-10-10**, `Asia/Yekaterinburg`.
План: [MINIMAX_M27_IMPLEMENTATION_PLAN.md](MINIMAX_M27_IMPLEMENTATION_PLAN.md).

Файл хранит проверенный прогресс и точку продолжения.
**Обновление весов:** target перенесён в `H:\models\MiniMax-M2.7`;
найдены DFlash Q3/Q4/Q5, ranges и полные draft SHA-256 проверены.
[Инспекция и этапы подключения DFlash](MINIMAX_M27_DFLASH.md).
**P5.DF-03:** реализован offline greedy DFlash driver и проведены
[измерения полезной скорости](MINIMAX_M27_DFLASH_BENCHMARK.md).
Q3/Q4/Q5 depths1/2/4/7 прошли full-logit corpus parity с opt-in tokenwise
verification. Исходный batched path менял routing на близких scores;
исправление и неудачные артефакты сохранены. Serving/defaults остаются off,
live EOS/cancel/pressure/recovery ещё TODO; native MTP weights отсутствуют.
Медианы трёх64-token повторов: off/cap18 **3.550**, Q4/depth1 **3.675**,
Q4/depth2 **3.497 токена/с**. Номинальные+3.54% не подтверждают устойчивого
выигрыша на фоне разброса off; все logits побитно совпали. Default off сохранён.
Исторические отчёты и команды ниже содержат пути прежних запусков.

**MM27-15: реализован и проверен запас для роста GPU arena.**
Новый `--arena-growth-reserve-mib` устраняет повторные allocations/frees
в управляемом тесте колебаний бюджета. На полной модели выигрыш мал;
устойчивое преимущество32 МиБ над64 не установлено. Defaults64/reserve0 сохранены.
**MM27-16: реализованы и проверены группы gate/up/down.**
Один слот на эксперта, совместная замена, readiness после завершения copy,
plan pins и admission один раз за router plan. **488 checks PASS**;
шесть full-model запусков побитно совпали со старым EXE по всем токенам/logits.
Три пары: aggregate decode **3,912→3,930 токена/с (+0,47%)**, полное время
**86,935→86,154 с**. Устойчивого выигрыша нет, `--cache-group-experts 0` сохранён.
[Реализация, измерения и артефакты](MINIMAX_M27_GROUP_CACHE.md).
**MM27-17: реализован ограниченный RAM LRU и partition с GPU-кешем.**
Read-only GGUF views удаляются после GPU fill; отложенное admission уменьшает
лишние отображения. **816 checks PASS**,21 ответ/1344 IDs/268886016 logits
совпали с сохранёнными эталонами побитно. Три короткие пары: RAM64 **3,485**
против off **3,941 токена/с**; workload89,556 против85,944 с. RAM cache default0.
Long256 достиг55,859 ГиБ views, global RAM69,88%; natural EOS не достигнут.
[Реализация RAM-кеша, отрицательный speed result и артефакты](MINIMAX_M27_RAM_CACHE.md).
**MM27-18: bounded output budget, английский EOS и live JSONL recovery проверены.**
Предел256 заменён свободным местом в context. Английский ответ и повтор
естественно завершились на414 tokens;17 JSONL boundary/EOS/cancel/recovery
scenarios PASS. Добавлен отдельный reasoning parser:13 parser/tokenizer tests
и18 real-output replay checks PASS. **Полный completion gate FAIL:** русский
список1..96 повторяется в reasoning и не даёт final за1536 tokens.
Serial ReadFile/H2D с cache/pipeline off повторил первые1024 IDs и204865536
logits побитно, включая14 повторов фразы: pipeline/cache не объясняют этот цикл.
Sampler и defaults не менялись; parser в API не зарегистрирован.
[Изменения, результаты и границы MM27-18](MINIMAX_M27_COMPLETION.md).
**MM27-18b: optional seeded sampling реализован и проверен.**
T1/top_p0,95/top_k40 завершил русский список на398/567/1092 tokens при трёх seed;
все семь полных RU/EN/ZH запросов дошли до EOS. Greedy96 IDs/19206144 logits
и повтор seed42 по398 IDs/79625472 logits побитные.48 CPU,14 JSONL и48 parser
checks PASS; sampling0,73–0,91 мс/токен. **Ручной quality gate FAIL:**
английский final указал отношение рассеяния10 вместо≈4,35 для450/650 нм.
Default greedy сохранён; API не подключён. [Результаты MM27-18b](MINIMAX_M27_SAMPLING.md).
**MM27-19: P2.3 и canonical input tool history реализованы.**
Нормализуются начальные system/developer, metadata и reasoning; ID инструментов
сохраняются, ответы располагаются в порядке calls.33 unit methods,293 prompts /
71189 IDs и96 legacy comparisons PASS. Новый CPU template oracle сохраняет
точность JSON float; raw oracle оставлен. Late instructions и неоднозначные
tool histories отклоняются. [Контракт и проверки MM27-19](MINIMAX_M27_HISTORY.md).
**MM27-20: P2.4 streaming tool-output parser реализован отдельно.**
Schema types/validation, atomic multi-calls, IDs, literal strings, UTF-8,
stop/truncation и bounded buffer проверены.51 unit methods,281 native groups
+281 histories /22079 token IDs,8 real completion replays /3966 checks PASS.
Tool corpus создан native template oracle; живой model tool cycle ещё не проверен.
[Контракт и ограничения MM27-20](MINIMAX_M27_TOOLS.md).
**MM27-21: OpenAI/Anthropic API adapters проверены через real loopback HTTP.**
191 tests,16 scripted tool cycles,32 HTTP replays прежних completions и272
report checks PASS.32 API prompts /14284 IDs совпали с native oracle.
Проверены JSON/SSE, ID/results, usage, finish_reason, strict UTF-8 и disconnect/recovery.
[Контракт и ограничения MM27-21](MINIMAX_M27_API.md).
**MM27-22: native JSONL Engine и отдельный экспериментальный сервер реализованы.**
31 CPU methods,11 live GPU scenarios и6 standalone startup checks PASS: полный ответ414 IDs совпал с
эталоном, OpenAI toolcall/result/answer, Anthropic SSE prefix, stop/disconnect,
повторная отмена и recovery/reload. Исправлен одноразовый SIGBREAK handler MSVC,
из-за которого вторая отмена завершала CUDA-процесс; исходный FAIL сохранён.
Полный EN decode **3,978 токена/с**; это транспортная проверка, не speedup A/B.
[Запуск, измерения и ограничения MM27-22](MINIMAX_M27_NATIVE_API.md).
**MM27-23: live multi-call/error matrix обоих API и web chat проверены.**
Четыре JSON/SSE цикла /8 model requests PASS: два calls, typed arguments,
кириллица, результаты в обратном порядке и ошибка инструмента. Prompt/generated
IDs совпали между API. Исправлен порядок полей tool definitions; web chat
не отправляет session header неподдерживающему движку.213 различных CPU методов,
272 regression checks и32 native prompts /14284 IDs PASS.
[Живые проверки, browser evidence и ограничения](MINIMAX_M27_LIVE_TOOLS_WEB.md).
**MM27-24: HTTP context4K/pressure/recovery проверены.**
17 scenarios и33 CPU methods PASS:8 ранних HTTP400, одинаковые4080/8 ответы
обоих API, два disconnect под реальной нагрузкой и recovery в том же CUDA PID.
Все пять32-token controls совпали с прежним эталоном. RAM достигла87,025%,
VRAM94,479%; кеш освободил128МиБ. Native EXE и defaults не менялись.
[Проверки, память и ограничения MM27-24](MINIMAX_M27_HTTP_CONTEXT.md).
**MM27-25: optional resident prefix reuse реализован.**
16 full-model scenarios PASS:128 IDs и25 608 192 logits побитно совпали с fresh.
21 live HTTP/lifecycle scenarios,14 430 C++ checks и29 Python methods PASS.
Повтор512 в одном измерении: prefill51,236→1,579с; полное время8 outputs
53,060→3,415с. Это повтор prompt, не повышение decode tokens/s.
Явный session ID, cached usage обоих API, cancel/error/restart invalidation;
default off. [Контракт, команды и границы MM27-25](MINIMAX_M27_PREFIX_CACHE.md).
**MM27-26: reuse2K/4K, EOS и pressure проверены.**
11 native requests /6 сравнений:567 IDs и113 436 288 logits побитно,
включая EOS414, sampling128 и один output.11 live pressure scenarios
(9 HTTP ответов /72 IDs) и6 CPU methods PASS. RAM86,604%/VRAM94,425%,
trim128МиБ, cancel/fresh/reuse recovery в том же native PID. Runtime и pin
не менялись. Повтор4K: prefill304,798→2,042с в одном offline измерении.
[Контракт, timings и границы MM27-26](MINIMAX_M27_PREFIX_CONTEXT.md).
**MM27-27: bounded RAM snapshot/restore нескольких сессий реализован.**
27 сравнений /188 IDs /37 612 032 logits побитно,14 live HTTP/lifecycle
scenarios,531 archive checks и54 Python methods PASS. LRU/byte/count caps,
cancel/error isolation, anonymous/return и restart проверены. Возврат512
в одном проходе: полное время55,864→6,960с с учётом save/restore.
Default archive cap0. [Контракт и границы MM27-27](MINIMAX_M27_SESSION_ARCHIVE.md).
**MM27-28: архив2K/4K/EOS/sampling и real pressure проверены.**
14 offline requests /1150 IDs /230 073 600 logits совпали с прежним reference;
шесть fresh/restore сравнений /567 IDs /113 436 288 logits побитно.
19 live scenarios /662 полных HTTP IDs и14 CPU methods PASS.
RAM94,191%/VRAM94,430%: отказ сохранить4K, защищённый restore2K,
eviction, trim128МиБ и обе отмены/recovery прошли. Engine EXE/defaults прежние.
[Измерения и границы MM27-28](MINIMAX_M27_SESSION_CONTEXT.md).
**MM27-29: пакетное копирование KV реализовано и проверено.**
Scratch до16МиБ, прежний формат snapshot;1200 CPU cases/12 CUDA checks,
192 full-model IDs /38 412 288 logits побитно,10 HTTP/lifecycle scenarios PASS.
Три тёплых переключения на target: полное время2K **7,941→5,277с**,
4K **8,024→3,608с**; transfer4K **4,842→0,551с**. Это session latency,
не подтверждение роста decode tokens/s. Новый admission EXE `98e85e80…`.
[Измерения, hashes и ограничения](MINIMAX_M27_STATE_BULK.md).
**MM27-30: bulk EXE проверен на полном корпусе и real pressure.**
14 offline requests /1150 IDs /230 073 600 logits побитно совпали с эталоном;
19 live scenarios /662 полных HTTP IDs и18 CPU methods PASS.
RAM94,197%/VRAM94,403%, trim64МиБ, отказ сохранить4K с защищённым restore2K,
обе отмены/recovery прошли. Независимый observer записал2093 отсчёта;
все измерения в пределах95%. EXE/defaults не менялись.
[Измерения, hashes и ограничения](MINIMAX_M27_STATE_BULK_PRESSURE.md).
**MM27-31: границы prefix/archive при GPU batch1/8 проверены.**
Context1024, prompts1/8…513: по25 session-сценариев на batch, всего568 IDs /
113 636 352 logits побитно совпали с fresh при том же batch. Ветвление,
укорочение, single-output и seeded sampling64, восемь RAM restores на batch PASS.
14 CPU methods и59 итоговых условий PASS. Runtime/defaults прежние.
[Корпус, измерения и ограничения](MINIMAX_M27_SESSION_BATCHES.md).
**MM27-32: проверен перенос настройки frequency decay из GLM.**
Периоды65536/131072/262144: медианы **4,120/4,171/4,183 токена/с** в трёх
процессах на вариант. На новой теме длинные периоды медленнее на2,79/6,87%;
default65536 сохранён.4608 IDs /921894912 logits побитно,19 history checks,
216 cache regression checks и285 итоговых gates PASS. Новый native flag
`--cache-decay-period` доступен в отдельном candidate EXE36fbac11…;
сервер использует прежний98e85e80…. [Результаты и границы](MINIMAX_M27_CACHE_DECAY.md).
**MM27-33: CUDA matrix events и async split compute реализованы и измерены.**
Native `--pipeline-events 2`: медиана трёх пар **4,086→4,266 токена/с
(+4,40%)**, полное время четырёх запросов164,030→157,808с. Все три пары
быстрее на4,07–4,40%; cache decisions прежние. Режим1 с одним переносом
ожиданий на events не ускорил screen.4608 IDs /921894912 logits побитно,
1256 CUDA fixture checks,6 CPU methods,8 CLI cases и357 evidence gates PASS.
Candidate EXE13a885cc… сохранён отдельно; native default0/server98e85e80…
сохранены до long-context и live pressure/cancel проверки нового EXE.
[Реализация, измерения и ограничения](MINIMAX_M27_COPY_EVENTS.md).
**MM27-34: events2 проверен на длинном корпусе и подключён к серверу.**
При ctx4K/batch16:14 requests /1150 IDs /230073600 logits побитно,
ещё113436288 logits в шести fresh/restore парах.19 live pressure scenarios
PASS:17 полных HTTP-ответов /662 IDs, EOS414, sampling128, RAM admission,
trim128МиБ, обе отмены и recovery в том же PID. Пики RAM94,199%/VRAM94,451%.
6 CLI/auth/unload/reload checks,51 CPU methods и111 итоговых gates PASS.
Server opt-in `--pipeline-events 2` требует candidate EXE13a885cc…;
legacy98e85e80… и default0 сохранены. Это проверка надёжности, не новый speed A/B.
[Проверенный запуск, результаты и границы](MINIMAX_M27_COPY_EVENTS_CONTEXT.md).
**MM27-35: host router-ID reuse реализован и измерен.**
Strict F32 Q4_K/Q6_K использует snapshot scheduler с проверкой node/storage/
layout и lifetime одного split; mismatch/callback возвращает прежний D2H.
Три пары поверх events2: **4,292→4,458 токена/с (+3,87%)**, все пары+2,56–6,74%.
24 requests /3072 IDs /614596608 logits совпали с legacy reference побитно.
Cache/H2D decisions прежние;97464 повторных чтений/ожиданий устранены на corpus.
97 router-ID и230 cache/fault/reload checks,12 Python methods и4 CLI cases PASS.
Short full-model batch1/8:32 IDs/6402048 logits в off/on парах побитно.
695 итоговых gates/163 SHA PASS,2144 memory samples, RAM22,213%/VRAM84,406%.
Native `--router-host-ids 1` требует candidate612354a1…; default0 и server
admission сохранены. [Измерения и границы](MINIMAX_M27_ROUTER_IDS.md).
Следующий шаг P3 — long-context/session/real-pressure проверка этого EXE
перед подключением к серверу. H2D fence таблицы перестановки ещё остаётся.
В P4 остаются GPU batch1/8 на2K/4K с fresh при том же batch и shift;
installer по-прежнему требует quality/model-oracle gates.
Экспериментальный набор **18 ГиБ /2 readers /chunk4 МиБ** сохранён.
Проверенные флаги: `--gpu-cache-mib 18432 --gpu-cache-allocator arena --pipeline-readers 2 --pipeline-chunk-mib 4 --pipeline-lookahead 1 --pipeline-d2d-batch 1`.
Defaults: GPU cache0, RAM expert cache0, session archive0, allocator `cuda`, reader `file`, pipeline0, lookahead0, D2D batch0, pipeline events0, router host IDs0, prefix cache off.
Strict F32 activations/KV; FA/graphs/MTP off. Есть изолированный API entry point;
общий installer и утверждённый P6 профиль ещё не готовы.
**OPEN:** причина старого MM27-06 tiny native-after-cache reload расхождения
не установлена. Новые PASS не закрывают его причину.

## Текущее состояние

| Область | Статус |
|---|---|
| PREP-01 | DONE: инспекция и два документа |
| Ревизия Strata при подготовке | `e111f63f89aa77d69aed46b2150f7e61aaa5b05d`, рабочее дерево уже содержит изменения |
| Основной GGUF | `H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf` |
| Файл | 138 342 384 352 байта / 128,841 ГиБ, один GGUF |
| Архитектура | `minimax-m2`; 62 MoE-блока, full attention |
| Кванты | Q4_K375, Q6_K61, F32 373; BF16 tensors нет |
| Header/ranges | PASS постоянного inspector/contract, все809 tensors |
| Candidate dependency | Unsloth `86ebfef2`; отдельные CPU/CUDA builds, source/patch hashes сохранены |
| MTP / DFlash | Native MTP weights отсутствуют; DFlash offline greedy/corpus parity PASS, Q3/Q4/Q5 depth screen и повторные timings измерены; serving off |
| Tokenizer/template/API | MM27-23/24: native Engine/entry point, live multi-call/error JSON/SSE, web chat и HTTP4K/pressure/recovery PASS; широкий tools/quality corpus и server-side MCP не проверены |
| CUDA kernels / tiny graph | PASS:83 kernel cases в обоих режимах;217 graph cases с F32 активациями |
| Fast quant graph | 208/217 PASS;9 logit FAIL, greedy fixture совпал; runtime default не утверждён |
| Engine/cache/pipeline/sessions | Optional GPU cache/groups и RAM view LRU с GPU partition; file pipeline с lookahead/D2D batch; RAM64 short speed хуже off; prefix/RAM snapshots и bulk KV transfer2K/4K PASS. Pressure94%/admission/cancel на bulk EXE прошли MM27-30; batch1/8 до513 tokens — MM27-31. Batch1/8 на2K/4K и shift ещё TODO |
| Reload validation | MM27-11: 55 stress и 18 tiny lifecycle checks PASS; старый native discrepancy остаётся OPEN |
| Скорость, токенов/с | P5.DF-03, медиана workload64: off/cap18 3.550; DFlash Q4/depth1 3.675, depth2 3.497; три повтора, устойчивый speedup не подтверждён |
| Рекомендуемые defaults | Для correctness: strict F32, FA/graphs/MTP off, context512/batch8; быстрый профиль не выбран |
| Следующая задача | P3: long-context/session/real-pressure для router-ID EXE612354a1… перед server admission; MM27-35 short A/B +3,87%, MM27-34 events EXE13a885cc… уже принят при batch16. Frequency decay default65536 сохранён. P4: batch1/8 prefix/archive на2K/4K и shift; sparse snapshots требуют отдельного A/B. Installer, answer quality/model oracle остаются gates P6 |

В PREP-01 созданы только план и статус в `docs/minimax-m2.7`.
Код, действующие профили и GGUF не изменялись. Существующая работа
над MiMo и другими моделями не является результатом внедрения MiniMax.

## Подтверждено при подготовке

- GGUFv3, alignment32, metadata43, tensors809;
  header_end **8 287 960**, data_start **8 287 968**.
- Имена уникальны, размеры типов известны, ranges выровнены, не пересекаются,
  не выходят за файл; конец последнего payload совпадает с EOF.
- Payload **138 334 096 384 байта (128,834 ГиБ)**:
  routed **135 725 580 288 (126,404 ГиБ)**,186 tensors;
  остальные **2 608 516 096 (2,429 ГиБ)**,623 tensors.
- Все62 блока содержат routed gate/up/down, router и correction bias;
  dense/shared/NextN tensors не обнаружены. Нет блоков за пределами0..61.
- Head geometry128, Q48/KV8; Q/K norm weights6144/1024.
  По local source нормы вычисляются до reshape на головы;
  partial NeoX RoPE64, full KV, sigmoid top-8/256 с normalized weights.
- GGUF nextn/mtp keys отсутствуют. Не приписывать metadata значение0:
  факт отсутствия MTP подтверждён также составом tensor directory.
- Header SHA-256 без padding/payloads:
  `9a011b2ad6fb20db3bb0cdfe9af08665939ee1f0cb0609ffe2bfee7f31c825db`.
- Template6512 символов; SHA-256 UTF-8 строки:
  `893d908f7b5cc65fdde270dcae5ea1a99647c6a7ce572ae874a57b7160069566`.
- GPT-2 BPE, pre-tokenizer `minimax-m2`, vocab200064, merges199744.
  BOS200034=`]~!b[`, EOS/PAD200020=`[e~[`, UNK200021=`]!d~[`;
  role delimiter200019=`]~b]`. MM27-01 подтвердил `add_bos=false`, `add_eos=false`;
  native EOG IDs:200004 `<fim_pad>`,200005 `<reponame>`,200020 `[e~[`.
  Это поведение dependency; stop policy runtime ещё предстоит проверить.
- Встроенный template всегда завершает generation prompt открытым `<think>`;
  переменной `enable_thinking` в нём нет. Отдельный no-thinking режим не проверен.
- `minimax-m2.cpp` в `third_party`, распакованном GLM candidate и архиве совпал:
  6362 байта, SHA-256 `6574a8618655d8164588627c1a4d2fc4fa8953b65fd783b8c1849f1c5c93638e`.
- Архив `build-local/llama-glm-86ebfef2.tar.gz` — 37 493 950 байт,
  SHA-256 `f8e524b635b726bae74fd8f84bb9249e5b09384c63207f6707c5a3f921acad99`.
  Candidate SHA — `86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9`.
- Windows RAM **125,555 ГиБ**, RTX 5090 VRAM **32 607 МиБ**, driver 581.80;
  95%-лимиты **119,277 ГиБ / 30 976,65 МиБ** глобально.
  Снимок до модельных тестов: available RAM≈95,62 ГиБ, GPU used4721 МиБ.
  Это внешняя нагрузка, не память MiniMax; перед тестом измерить заново.

В [плане](MINIMAX_M27_IMPLEMENTATION_PLAN.md) приведены расчёты KV/H2D,
ссылки на опубликованный одноимённый GGUF, официальный config и loader.
Публичный MTP config описывает архитектуру, а не наличие draft в локальном файле.

## Не проверено

- Полный checksum локальных весов и совпадение с опубликованным llmfan46 GGUF.
  Header hash не является checksum файла. Source/converter revision не записаны.
- Широкая full-model quality parity и режимы CUDA Graphs/graph reuse.
  Первый корпус с production shapes и file/native parity — MM27-03.
  Быстрый MMVQ/MMQ путь не прошёл строгие logit tolerances малого графа; нужно
  отдельно решить policy точности/скорости перед выбором defaults.
- Default temperature finetune и stop policy в HTTP/API. Experimental runtime
  останавливается по EOS/PAD200020; FIM/reponame aliases не являются stop IDs.
  Содержимое конфигурации самого finetune не получено; официальный config
  базовой модели не заменяет проверку локального файла. MM27-18b проверил optional
  sampling на фиксированном корпусе: natural EOS PASS, ручной answer-quality FAIL
  английского final. Причина фактической ошибки не установлена; default greedy.
- Широкий live tools/quality corpus и no-thinking профиль. MM27-19 проверил canonical history;
  MM27-20 — generated-tool parser. MM27-21 проверил оба HTTP API и ID/results
  через scripted engine, включая native tokenizer и сохранённые outputs.
  MM27-22 подключил native Engine к Service, проверил один настоящий OpenAI
  tool cycle, Anthropic SSE prefix и repeated cancel/drain/reload через API.
  MM27-23 добавил четыре live multi-call/error цикла обоих API и web chat.
  Это одна инструментальная задача и короткий текстовый диалог, не широкий
  quality corpus. Server-side MCP и no-thinking ещё не проверены.
- Причина двух tiny long-context native-after-cache reload расхождений MM27-06.
  Поздние PASS с диагностическим replay не доказывают устранение причины.
  Full-model cache-on pressure/cancel/unload и context2K/4K прошли в MM27-06;
  cache-off — MM27-04. MM27-18: английский EOS/полный повтор/live signal recovery PASS,
  русский greedy repetition остаётся OPEN. Расширенная CUDA kernel timeline,
  continuous memory peaks/physical SSD traffic и многошаговые сессии ещё не проверены.
- Native MTP weights по-прежнему не найдены. DFlash Q3/Q4/Q5 прошли loader,
  head/MASK/feature checks, полный offline greedy цикл и корпусное сравнение
  с serial target; acceptance и скорость измерены. Открыты numerical parity
  с NVIDIA reference, live GPU EOS/cancel/pressure/recovery, длинный контекст,
  sessions и устойчивый полезный выигрыш для serving.

## Таблица этапов

| Этап | Статус | Условие готовности |
|---|---|---|
| PREP-01 | DONE | Исследование и документация сохранены |
| P0 — contract/oracles | DONE для F32 activation baseline; fast-quant logit issue открыт | Inspector, полный contract, изолированная сборка и CUDA fixtures |
| P1 — GPU baseline | Functional gates PASS: corpus, 2K/4K, cancel/recovery, pressure/unload; MM27-18b sampled RU/EN/ZH EOS PASS, ручной answer-quality FAIL | Greedy repetition, independent oracle, quality/SSD measurements остаются |
| P2 — tokenizer/template/API | PARTIAL: adapters, native Engine/entry point, live multi-call/error JSON/SSE, web chat и HTTP4K/pressure/recovery PASS | Более широкий tools/quality corpus; server-side MCP вне scope |
| P3 — cache/pipeline | PARTIAL: cache, mmap, lookahead/D2D, overlap, groups и RAM LRU/partition; MM27-33 matrix events/async splits +4,40% в трёх коротких A/B, MM27-34 long corpus/live pressure/cancel и server opt-in PASS при batch16; RAM64 short speed хуже off; tiny reload OPEN | MM27-35 router-ID reuse +3,87% short A/B; для нового EXE нужны long-context/session/real-pressure. Полезный RAM speedup, residency/SSD measurements и широкий corpus остаются |
| P4 — sessions/context | PARTIAL: prefix и bounded RAM snapshot/restore до4080 /ctx4K/batch16, EOS414/sampling128/single-output exact logits; bulk copies и HTTP94% pressure/admission/cancel/recovery PASS. GPU batch1/8 до513 /ctx1024 с full logits PASS, MM27-31 | Shift, GPU batch1/8 на2K/4K, context>4K; sparse-copy A/B отдельно |
| P5 — MTP / DFlash | Native MTP ждёт weights/backend; DFlash OFFLINE_GREEDY_AND_CORPUS_PASS, timings измерены | Live GPU EOS/cancel/pressure/recovery, context/sessions и полезный speedup |
| P6 — profile/regressions | PARTIAL: изолированный experimental entry point MM27-22 | Installer, quality/oracle gates и утверждённые defaults |

Отсутствие P5 не блокирует P0–P4/P6. `DONE` относится только к указанной
части; чтение header и найденный upstream loader не означают working inference.

## Точка продолжения: P3/P4, качество ответов и оставшиеся DFlash lifecycle gates

MM27-33 завершил short A/B CUDA matrix events/async splits. MM27-34 проверил
prompts2K/4K при ctx4096/batch16, prefix/archive, EOS/sampling, real pressure,
обе отмены/recovery и CLI reload на candidate13a885cc…. Server opt-in принят,
default0 сохранён. MM27-35 реализовал reuse уже прочитанных router IDs,
проверил identity/lifetime и полные logits; три короткие пары дали+3,87%.
Следующий шаг — long-context/session/real-pressure на EXE612354a1… перед
server admission. Затем можно исследовать H2D fence таблицы перестановки
с отдельным bounded host-buffer lifetime. Частотный default65536 прежний.

1. P5.DF-03a выполнен для offline greedy и короткого корпуса. DFlash оставлен
   off; перед serving нужны live GPU EOS/cancel/pressure/recovery, длинный
   контекст и session restore. Depth1/2/4/7 и кванты уже измерены — повторять
   sweep после изменения реализации, а не вместо закрытия этих gates.
2. MM27-16 завершён в проверенном scope: группа `(generation, layer, expert)`,
   плотная упаковка, совместная admission/replacement, readiness и group pins.
   Density/bytes/logits/cancel/pressure и три A/B относительно64/reserve0 PASS.
   Speedup не подтверждён; grouping остаётся optional/off. Повторять эти
   timings после изменения реализации или корпуса, а не вместо новой работы.
3. MM27-17 реализовал RAM view LRU/partition и отложенное admission.816 checks
   и long256 parity PASS; три короткие пары показали замедление. RAM0 сохранён.
   Не включать эксперимент по одному удачному timing; full RAM-cap LRU,
   фактическая residency/SSD traffic и speedup долгих сессий остаются проверками.
4. MM27-18 проверил свободный context budget, английский EOS на414 tokens,
   полный повтор и реальную адресную отмену после257 tokens с восстановлением.
   Русский1..96 зациклился в reasoning: исторический complete-corpus gate FAIL.
   MM27-18b реализовал optional sampling; семь полных ответов с EOS, seeded
   repeat/greedy/cancel/recovery PASS. Ручная проверка английского final FAIL:
   неверное численное отношение рассеяния. Независимый reference и широкое
   качество остаются gates P6; не подбирать успешный seed вместо этой проверки.
   MM27-19 реализовал P2.3 и canonical tool inputs: instructions/reasoning,
   typed calls, ID/result correlation и порядок results, native prompt/ID parity.
   MM27-20 проверил standalone P2.4 tools parser, native fixtures и byte replay.
   MM27-21 проверил API adapters, real loopback JSON/SSE и scripted tool cycle.
   MM27-22 реализовал resident JSONL Engine, startup entry point и dependency
   preflight. Live EOS, repeated cancel/recovery, один OpenAI tool cycle и
   Anthropic SSE prefix PASS. MM27-23 проверил live tools JSON/SSE обоих API,
   multi-call/error corpus и web chat; исправлены tool-definition order и web
   session header. MM27-24 проверил HTTP4080/8 boundary, ранние overflow400,
   реальную RAM/VRAM pressure, два disconnect и точный следующий ответ в том же
   процессе. MM27-25 реализовал P4 resident prefix reuse: явная session identity,
   фактически вычисленные KV positions, short fresh token/logit parity, isolation
   и cancel/error/restart invalidation. MM27-26 проверил reuse2032/4080
   /ctx4K/batch16, EOS414, sampling128 и single-output exact logits, live pressure
   обоих API, trim и cancel/fresh/reuse recovery. MM27-27 добавил bounded
   RAM snapshot/restore: short full-logit parity, LRU/caps и HTTP lifecycle.
   MM27-28 проверил archive2K/4K/EOS/sampling и реальное RAM94,191%/VRAM94,430%
   давление, защищённый restore при отказе admission и repeated cancel/recovery.
   MM27-29 добавил bounded bulk IO: state bytes на CUDA fixture и полные logits
   на2K/4K совпали, три тёплых повтора и HTTP lifecycle прошли. MM27-30 повторил
   полный offline corpus и real pressure/admission/cancel на bulk EXE:
   RAM94,197%/VRAM94,403%,19 scenarios PASS. MM27-31 проверил GPU batch1/8
   до513 tokens при context1024: по25 session cases, single-output/sampling64
   и full-logit fresh parity. Следующий шаг — GPU batch1/8 prefix/archive
   на2K/4K относительно fresh с тем же batch, затем shift. Installer
   registration требует выполнения quality/model-oracle gates.
   Default sampler greedy; RAM cache, grouping и prefix cache выключены.
   Cache/pipeline off уже воспроизвёл1024 IDs/logits побитно; повторять этот
   контроль без изменения реализации или корпуса не требуется.
5. Старый MM27-06 discrepancy OPEN: при повторении сохранить EXE/logits и
   трассировать первый расходящийся node. Native MTP требует отсутствующих local weights.

## Подтверждённый журнал

### MM27-35 / часть P3.2 — 2026-10-10 — Host router IDs в strict F32

Добавлен native `--router-host-ids 0|1`, default0. Snapshot до64КиБ хранит
owned bytes scheduler на одном thread, используется один раз и очищается
на границе split/reset/release. Exact node/weight/IDs identity, shape/stride/
view/storage checks; callback или mismatch возвращают прежний CUDA D2H.
Арифметика, CUDA graphs policy, таблица перестановки/H2D fence и graph drain
сохранены. Bridge действует только в private MiniMax runtime build.

Candidate612354a1…/dependency86ebfef2…, RTX5090/RAM125,555ГиБ. Context2048,
batch16/cache18ГиБ/arena64/reserve0/decay65536/readers2/chunk4/lookahead/D2D/
events2, strict F32, RAM/prefix/archive/MTP off. A,A,B,A по128 выходных токенов.
Три пары off/on, on/off, off/on:4,176→4,458;4,295→4,405;4,292→4,470 токена/с.
Медианы4,292→4,458 (+3,87%), полное время157,033→152,176с.
Все3072 IDs и614596608 logits побитно; cache/H2D decisions одинаковы.
Per-case медианы быстрее на3,19–5,29%. На workload97464 reuse hits,0 misses;
94488 приходится на decode. Матрицы весов передаются в прежнем объёме.

97 identity/CUDA batch1/8/16 checks и230 cache/fault/pressure/reload checks
PASS;12 Python methods и4 early CLI cases PASS. Short full-model batch1/8
также побитный:32 IDs/6402048 logits в двух парах.695 evidence gates/163 SHA
PASS;2144 независимых samples, RAM22,213%/VRAM84,406%,10 native PID закрыты. Новый EXE сохранён отдельно,
legacy98e85e80… восстановлен, events13a885cc… сохранён. Server admission не
расширен до проверки full-model long-context/sessions/real pressure нового EXE.
[Полный отчёт, команды и артефакты](MINIMAX_M27_ROUTER_IDS.md).

### MM27-34 / часть P3.2/P4 — 2026-10-09–10 — Events2: context, pressure и server opt-in

Native candidate13a885cc…/dependency86ebfef2… не изменены. Adapter/CLI получили
опцию `--pipeline-events 2` с exact EXE admission, проверкой ready mode/decay
и отказом при events без readers или со старой сборкой. Legacy98e85e80…
с режимом0 сохраняется. Диагностические drivers принимают явный engine/mode;
offline теперь также имеет независимый memory observer. Учет events допускает
нулевой decode graph для ответа из одного токена.

RTX5090/RAM125,555ГиБ, Q4_K_M target, ctx4096/batch16/F32, cache18ГиБ,
arena64/reserve0/decay65536, readers2/chunk4/lookahead/D2D/events2,
RAM expert cache0, prefix on/archive6144МиБ/4, bulk KV on, MTP/DFlash off.
14 offline requests /1150 IDs /230073600 logits совпали с MM27-26 побитно;
шесть fresh/restore пар /113436288 logits тоже побитно. EOS414, seed42
sampling128 и single-output прошли.1696 independent memory samples без ошибок.

19 live scenarios в одном native PID:17 HTTP answers /662 IDs, оба API,
cached usage, EOS/sampling, physical RAM admission rejection с защитой2K,
GPU trim128МиБ, prefill disconnect1,702с и decode cancel3,561мс, fresh/reuse
recovery. Пики RAM94,199%/VRAM94,451%,2110 independent samples; все≤95%.
Ближайший к RAM94 независимый sample выбран по времени: за46мс до marker,
94,076%. Первый последующий уже видит освобождение архива,93,363%; он всё
ещё подтверждает нехватку137594440байт для4K snapshot с резервом. Исходное
ожидание93,5–95% именно в последующем sample не выполнено и сохранено
отдельным false diagnostic; native limits/admission не менялись.

6 реальных CLI/auth/unload/reload checks,51 CPU methods и111 итоговых gates /
165 source checks /231 artifact hashes PASS, exit0. Четыре новых CPU methods
проверяют временную привязку RAM evidence. Все owned процессы завершены,
buffers holder освобождены. [Команды, timings и scope](MINIMAX_M27_COPY_EVENTS_CONTEXT.md),
[машинный отчёт](MINIMAX_M27_COPY_EVENTS_CONTEXT_CHECK.json).

Mode2 теперь server opt-in; default0 и основной EXE98e85e80… сохранены.
Повторный speed A/B не выполнялся, результат+4,40% относится к MM27-33.
Следующий кандидат — host router-ID reuse в strict F32. Batch1/8 на2K/4K,
shift/context>4K, большой quality corpus, model oracle и MM27-06 ещё OPEN.

### MM27-33 / часть P3.2 — 2026-10-09 — CUDA matrix events и async split compute

Перенесена идея matrix-level event dependencies из GLM. Флаг
`--pipeline-events 0|1|2`: прежние fences /events /events+async split API.
Scratch защищён backend→delivery event, compute ждёт delivery→backend event;
pending fills закреплены до retirement перед cache decisions следующей матрицы.
Public graph boundary синхронизирует CUDA backend на успехе, ошибке и отмене.
Shared transport и математика не изменены; strict F32 internal waits остаются.

Strata HEAD54467693… с прежними staged/unstaged changes, dependency86ebfef2…;
candidate SHA `13a885cc75352bb39cc4231174da3d9f03705d5f9c150117935d6f05ea9d2401`.
Основные изменения: `pipeline_state.inc`, `pipeline_runtime.inc`,
`sync_runtime.*`, `RuntimePatches.cmake`, `main.cpp`, `check_cache.cpp`.
Build manifest/source snapshots и команды сохранены в
[отчёте](MINIMAX_M27_COPY_EVENTS.md); build и tests exit0.

RTX5090/RAM128GB, target Q4_K_M, ctx2048/batch16/F32, cache18ГиБ/arena64,
decay65536, readers2/chunk4/lookahead/D2D; RAM/groups/prefix/MTP off.
Screen0/1/2:4,046/4,046/4,234 токена/с. Отдельные три пары0↔2:
**4,086→4,266 токена/с (+4,40%)**, request164,030→157,808с (−3,79%).
Каждая пара быстрее на4,07–4,40%; все четыре сценария A,A,B,A быстрее
в медианах на3,78–4,79%. H2D/cache decisions во всех вариантах одинаковы.

Всего36 requests/4608 IDs/921894912 logits побитно совпали с old EXE98e85e80….
1256 CUDA fixture checks включают ошибки, pending fills, observer-free
cancel/pressure, группы, RAM LRU, no-lookahead и recovery/reload/unload.
6 CPU methods,8 CLI cases и357 итоговых gates PASS.2917 memory samples:
RAM≤23,710%,VRAM≤85,085%; это не реальное давление около95%.
[Машинные проверки](MINIMAX_M27_COPY_EVENTS_CHECK.json).

Native default0 и admitted server EXE98e85e80… сохранены. Short A/B принят
как ускорение в этом корпусе; full-model long context/pressure/cancel и
prefix/archive на candidate ещё TODO. `compute_ms` режима2 отражает CPU
submission, не GPU wall time. Следующий кандидат — убрать повторное чтение
router IDs в strict F32 без изменения арифметики; speedup не измерен.

### MM27-32 / часть P3 — 2026-10-09 — Frequency decay

Добавлен native `--cache-decay-period N`,1..4294967295, default65536;
параметр передаётся в конструктор cache, отражается в JSON header.
Общий алгоритм history и defaults других backends прежние.
Candidate EXE `36fbac1185a69a31b627eb818feb9f4193741937d42c391bcb44ebd64b4d9efa`
сохранён отдельно; admitted server EXE98e85e80… не заменён.

RTX5090/RAM128GB, target Q4_K_M, ctx2048/batch16/F32, cache18ГиБ,
arena64/reserve0/file/readers2/chunk4/lookahead/D2D; RAM/groups/prefix/MTP off.
Один старый EXE для numerical control, затем9 candidate процессов в трёх
латинских порядках. Каждый: A, повтор A, новая тема B, возврат A, по128 tokens.
40 requests/5120 outputs; все4608 candidate IDs и921894912 logits побитно.
Все три65536 процесса повторили cache decisions старого EXE.

Медианы65536/131072/262144: **4,120/4,171/4,183 токена/с**,
полное время корпуса **163,804/161,512/160,484с**.
H2D **984,085/987,351/988,071ГиБ**, вытеснения40711/22633/17602.
На возврате A длинные периоды быстрее на7,08/12,81%, но новая тема медленнее
на2,79/6,87%. Малый агрегатный выигрыш не оправдывает общий default:
корпус содержит три A и один B. **Default65536 сохранён.**

19 history/admission checks,10 CLI boundaries,5 CPU methods,216 cache regression
checks и285 итоговых gates PASS.3249 независимых samples: RAM≤23,608%,
VRAM≤85,654%; реального давления около95% здесь не было. Процессы завершены.
Следующий кандидат — CUDA events с отдельным lifetime/cancel/pressure A/B.
Открытые P4/quality/MM27-06 gates этот эксперимент не закрывает.
[Полный отчёт](MINIMAX_M27_CACHE_DECAY.md),
[машинные проверки](MINIMAX_M27_CACHE_DECAY_CHECK.json).

### MM27-31 / часть P4 — 2026-10-09 — Границы GPU batch1/8

Ревизия `54467693c8354d8c64a72ec1cb083473a529b4ba`, dirty tree.
Добавлены отдельный corpus runner, четыре CPU methods и итоговый validator.
Native EXE `98e85e80…`, target/dependency и defaults прежние, bulk1/tokenwise0.
Context1024, F32, cache18ГиБ, file/readers2/chunk4/lookahead/D2D.
На batch: один fresh-процесс с prefix/archive off, затем session-процесс с
archive768МиБ/4 записи; процессы выполнялись последовательно.

68 native requests /810 output tokens.50 session-сценариев /568 IDs /
113 636 352 logits побитно совпали с fresh при том же batch, NaN/Inf нет.
Проверены prompt ровно из одного пакета, repeat/branch/shorten/extend,
anonymous/return, RAM restore,512→513, single-output и sampling64/seed42.
При repeat512 reused511 для batch1 и504 для batch8; extend512→513 reused512.
По восемь RAM restores на batch; все accounting/caps/IO-time/GPU checks PASS.

Глобальные sampled peaks: batch1 RAM24,063%/VRAM88,959%,
batch8 RAM23,507%/VRAM87,301%; pressure holder не запускался.
14 CPU methods;59 итоговых условий /166 source checks /194 artifact hashes PASS.
Четыре native-процесса завершены, независимые memory observers закрыты.
Полное время8 outputs при prompt512, одна наблюдаемая тройка на batch:
batch1 fresh159,425с /resident1,950с /RAM restore3,824с;
batch8 fresh88,182с /resident3,029с /RAM restore5,582с.
Это не repeated speed A/B и не основание менять defaults.

[MM27-31: команды и scope](MINIMAX_M27_SESSION_BATCHES.md),
[SESSION_BATCHES_CHECK](MINIMAX_M27_SESSION_BATCHES_CHECK.json).
Следующий пункт — batch1/8 на2K/4K; shift, context>4K, широкий quality/model
oracle и MM27-06 reload discrepancy остаются OPEN. EOS и HTTP pressure этого
этапа не проверялись: предыдущие длинные проверки относятся к batch16.

### MM27-30 / часть P4 — 2026-10-09 — Bulk KV при реальном давлении памяти

Ревизия `54467693c8354d8c64a72ec1cb083473a529b4ba`, dirty tree.
EXE `98e85e80…`, holder `50cf15e7…`, target/header/template/dependency прежние;
`STRATA_MM27_STATE_BULK=1`, context4096/batch16, F32, cache18ГиБ,
file/readers2/chunk4/lookahead/D2D, archive6144МиБ/4 записи.
Изменён диагностический код: независимый Windows/NVML observer RAM/commit/VRAM,
ошибки и95% guard, join/закрытие до hashes, environment/stage evidence и validator.
Новых runtime optimizations или defaults на этом этапе нет.

14 offline requests /1150 IDs /230 073 600 logits совпали с MM27-26;
6 fresh/restore сравнений /567 IDs /113 436 288 logits побитно.
EOS414, sampling128/seed42, one-output PASS.19 live scenarios /662 полных IDs
PASS в одном native PID; pressure проверяет IDs, не полные logits.
Пики по записанным отсчётам RAM94,196664%/VRAM94,403402%, все≤95%.
Отдельный observer:2093 отсчёта, максимум RAM94,140979%/VRAM94,403402%.
Holder private11,65625ГиБ и GPU1184МиБ; FREE обнулил выделения.
Arena trim64МиБ. Отказ сохранить4K по свободной физической RAM, eviction
и protected restore2K (1 035 642 236байт) прошли; commit и archive caps достаточны.
HTTP disconnect до первого output:1,644с, decode cancel3,519мс без лишних client tokens;
другая сессия сохранна, fresh2K recovery совпал, процессы завершены.

18 CPU methods;100 итоговых условий,160 source checks/211 artifact hashes PASS.
Один pressure pass, не повторный speed A/B. Команды и scope:
[MM27-30](MINIMAX_M27_STATE_BULK_PRESSURE.md),
[STATE_BULK_PRESSURE_CHECK](MINIMAX_M27_STATE_BULK_PRESSURE_CHECK.json).
Следующий шаг — GPU batch1/8 prefix/archive parity, затем shift.
Context>4K, широкий quality/model oracle и MM27-06 discrepancy остаются OPEN.

### MM27-29 / часть P4 — 2026-10-09 — Пакетные host KV copies

Ревизия `54467693c8354d8c64a72ec1cb083473a529b4ba`, dirty tree.
Private generated `llama-context.cpp`, source SHA pin и два точных anchors;
`state_bulk.hpp`, CUDA/CPU fixture, A/B/HTTP checkers и итоговый validator.
Исходный dependency и формат snapshots не изменены. Scratch16МиБ, gaps
сохраняются read-modify-write, при плотности ниже25% остаётся legacy path.
Default bulk on, диагностическое `STRATA_MM27_STATE_BULK=0`; archive cap0.

1200 CPU cases/12 CUDA checks,21 подтверждённая bulk group; state5 901 216байт
и continuation logits побитно. На полной модели24 запроса /192 IDs /
38 412 288 logits совпали с MM27-28, NaN/Inf нет. Два последовательных
процесса, по три тёплых переключения на каждый target, одинаковая cold setup:
2K request7,941→5,277с (1,51×),4K8,024→3,608с (2,22×),
short6,807→4,281с (1,59×). Transfer4K4,842→0,551с (8,79×).
Короткий decode по7 шагов не является устойчивым throughput benchmark.

10 live scenarios /312 полных HTTP IDs и2 cancelled IDs PASS:
оба API JSON/SSE, restore, cancel/isolation/fresh recovery, restart.
25 Python methods и отдельный пересекающийся набор14 methods PASS.
Build/fixture/A/B/live/unit exit0; итоговый validator159 source/29 artifact
hash checks PASS. Новый admission EXE `98e85e80…`; старый `275e89f8…`
сохранён до сборки. Точные команды, hashes и ограничения:
[MM27-29](MINIMAX_M27_STATE_BULK.md),
[STATE_BULK_CHECK](MINIMAX_M27_STATE_BULK_CHECK.json).

Следующий шаг — real RAM/VRAM94% pressure, отказ admission, protected restore
и обе отмены на новом EXE. MM27-28 pressure PASS относится к старому EXE.
GPU batch1/8, shift, широкий quality corpus и MM27-06 остаются открытыми.

### MM27-28 / часть P4 — 2026-10-09 — Archive2K/4K, EOS, sampling и pressure

Ревизия `54467693c8354d8c64a72ec1cb083473a529b4ba`, dirty tree,
прежние target/header/template/dependency86ebfef2 и native EXE `275e89f8…`.
Добавлены offline/live checkers, три новых CPU methods, независимый memory
observer и финальный validator. Только диагностический holder получил
bounded private RAM topup до94,2%; до16ГиБ chunks32МиБ, guard95%.
Native arithmetic, session runtime, admission pin и defaults не менялись.

**PASS, exit0:**14 CPU methods,14 native requests /1150 tokens и все
230 073 600 logits с MM27-26 reference; шесть дополнительных fresh/restore
сравнений /567 IDs /113 436 288 logits побитно. EOS414, seed42 sampling128,
один output, snapshot987,665/1979,689МиБ проверены. Повтор4K request
306,510→7,185с в одном corpus; это latency, не decode speedup/три A/B.

**19 live scenarios PASS:**17 ответов /662 IDs, оба API JSON/SSE, EOS414 и
sampling128 после restore под нагрузкой; RAM94,191%/VRAM94,430%,771 samples
оба выше85%, expert arena trim128МиБ. При недостатке physical RAM уходящий
snapshot4K не сохраняется, старая запись вытесняется, запрошенная2K восстанавливается.
Cap6ГиБ/4 slots и доступного commit хватало. Prefill/restore disconnect
3051,781мс /0 outputs; decode cancel3,187мс /0 лишних клиентских tokens;
другая сессия сохранна, отменённая2K прошла fresh recovery в том же PID.

Full pressure logits не экспортировались. GPU batch1/8, shift/context>4K,
другие KV types и широкое качество остаются TODO. Старые OPEN gates не закрыты.
Артефакты: `build-local/minimax-m2-sessions-context-01`,
`build-local/minimax-m2-sessions-pressure-01`; holder baseline сохранён отдельно.
Команды, hashes и scope: [MM27-28](MINIMAX_M27_SESSION_CONTEXT.md),
[SESSION_CONTEXT_CHECK](MINIMAX_M27_SESSION_CONTEXT_CHECK.json).
Следующий шаг — пакетные V copies с bounded scratch, bit-exact state/logits,
затем три A/B полной задержки переключения с учётом transfer.

### MM27-27 / часть P4 — 2026-10-09 — RAM snapshots нескольких сессий

Ревизия `54467693c8354d8c64a72ec1cb083473a529b4ba`, dirty tree,
dependency86ebfef2, прежние target/header/template. Перед сборкой сохранены
старый EXE/source/manifest; индекс Git не изменялся.
Добавлены `sessions.hpp`, `session_runtime.hpp`, CPU checker,
native/server flags `session-cache-mib`/`session-cache-slots`, protocol counters,
offline/live audit tools. Архив принадлежит одному model/context; disk import
нет. Default cap0; kernels/sampler/expert pipeline не менялись.

**PASS, exit0:**531 archive и14 430 prefix C++ checks,54 Python methods,
31 native requests /27 full-logit comparisons,14 HTTP/lifecycle scenarios.
188 сравниваемых IDs /37 612 032 logits побитно; оба API и cached usage,
LRU, недостаточный cap, branch/shortening, anonymous/return, cancel/error
isolation и restart проверены. Ещё6 CLI invalid-limit cases вернули ожидаемый
exit2 до загрузки. Первая сборка FAIL из-за missing windows.h исправлена;
первоначальный лог сохранён. GPU correctness FAIL не было.

Одна пара512: request55,864→6,960с, restored prefill1,606с,
save/restore/trim3,516с. Это latency reuse, не decode speedup и не3-repeat A/B.
Sampled global RAM22,718%, VRAM84,579%; реального pressure в этапе нет.
Новый EXE `275e89f8…` допущен после offline checks и проверен обычным Engine.

Артефакты: `build-local/minimax-m2-session-baseline`,
`build-local/minimax-m2-sessions-01`, `build-local/minimax-m2-sessions-http-01`;
полные команды, hashes и ограничения в
[MM27-27](MINIMAX_M27_SESSION_ARCHIVE.md) и
[SESSION_ARCHIVE_CHECK](MINIMAX_M27_SESSION_ARCHIVE_CHECK.json).
Следующий шаг — archive2K/4K/EOS/sampling и eviction/cancel под real pressure.
Оптимизация мелких KV copies, shift, GPU batch1/8 и P6 quality/oracle отдельно.

### MM27-26 / часть P4 — 2026-10-09 — Длинный prefix, EOS и pressure

Добавлены full-logit context checker, live pressure checker и6 CPU методов
для ограниченного по памяти сравнения logits/one-output accounting.
Native EXE `294ce711…`, CUDA arithmetic, weights и defaults не менялись.

**PASS:**11 native requests,6 сравнений /567 IDs /113 436 288 logits побитно.
2032/4080 prompt tokens при context4096/batch16, продолжение2K→4K, EOS414,
T1/top_p0,95/top_k40/seed42 с budget128, один output с нулевым decode graph.
Первый EOS414 также побитно совпал с историческим английским эталоном MM27-18.
6 CPU methods PASS; initial mock-test FAIL исправлен и сохранён, GPU FAIL нет.

11 live scenarios /9 HTTP ответов /72 IDs совпали с offline fresh reference:
OpenAI/Anthropic JSON/SSE cached usage, два repeated4K под pressure,
trim128МиБ (arena16384→16256МиБ), HTTP disconnect с3936 reusable positions,
полный4K recovery/repeat, native cancel после первого output, short recovery
и FREE. Native PID19336 не перезапускался; все owned процессы завершены.
Пики RAM108,736ГиБ /86,604%, VRAM30,068ГиБ /94,425%;17 monitor samples
одновременно выше85%. Pressure проверяет IDs, не full logits.

Один offline corpus: prefill2K221,605→2,096с;4K304,798→2,042с.
Live4K repeat под pressure:2,653/1,850с, после cancel full prefill341,153с,
следующий repeat1,900с. Это latency общего prompt, не рост decode tokens/s.
Во время cancel runtime очищает expert cache; GPU usage в fresh recovery
падает, затем arena растёт. Пик94,425% не описывает постоянную загрузку.

Артефакты: `build-local/minimax-m2-prefix-context-01`,
`build-local/minimax-m2-prefix-pressure-01`; команды и полный scope:
[MM27-26](MINIMAX_M27_PREFIX_CONTEXT.md),
[26 gates /318 hash checks](MINIMAX_M27_PREFIX_CONTEXT_CHECK.json).
Следующий шаг — bounded snapshot/restore нескольких сессий; shift и GPU
batch1/8 отдельно. Quality/model oracle, MM27-06 и DFlash serving остаются OPEN.

### MM27-25 / часть P4 — 2026-10-09 — Resident KV prefix reuse

Добавлены `prefix.hpp`, CPU policy fixture, native `--prefix-cache 1`,
серверный `--prefix-cache` и проверяемые reused/evaluated/KV counters.
Одна именованная сессия; совпадение полных prefill-пакетов, последний пакет
считается заново. Decode tail не объявляется готовым batched prefill.
Native ошибка/отмена, anonymous/ID switch и restart инвалидируют кеш.

**PASS:**14 430 policy checks,29 Python methods,16 full-model cases
(16–513 prompt tokens;128 IDs /25 608 192 logits) и21 live scenarios
(17 HTTP ответов /136 IDs, две отмены, already-cancelled, malformed key).
Новый fresh побитно совпал со старым EXE на16 IDs /3 201 024 logits.
После restart новый native PID начал с reused0; owned процессы завершены.
Native GPU-only/bytes/drain/global95% gates PASS; RAM max36,036ГиБ /28,701%,
VRAM27,170ГиБ /85,324%. External pressure здесь не создавался.

Один corpus fresh/reuse: prompt52 prefill8,235→0,689с, prompt512
51,236→1,579с. Request512/8:53,060→3,415с. Повторов A/B нет; это latency
общего prompt, не заявка на рост decode tokens/s. Default off сохранён.
Новый admission SHA `294ce7117876b7a112e5a29c199c8822d9f48bc333a274cbd7f6d8ff65ebbfcd`.

Артефакты: `build-local/minimax-m2-prefix-01`,
`build-local/minimax-m2-prefix-http-01`; полный контракт, команды и ограничения:
[MM27-25](MINIMAX_M27_PREFIX_CACHE.md),
[проверка артефактов](MINIMAX_M27_PREFIX_CACHE_CHECK.json).
Следующий шаг — reuse2K/4K/EOS/pressure; затем snapshot/restore/shift.
Quality/model oracle, MM27-06 discrepancy и DFlash serving gates остаются OPEN.

### MM27-24 / часть P2.6/P4 — 2026-10-09 — HTTP4K и реальное давление на память

Основание: commit `54467693c8354d8c64a72ec1cb083473a529b4ba`, dirty tree;
dependency86ebfef2, прежние model/header/template и EXE SHA
`f4ff449823beee3db1b260597cac1ed9276fb53922c5816f082f77ccbf52f619`.
Добавлены opt-in pressure-holder target, HTTP checker и CPU boundary tests.
CUDA/model/serving defaults не менялись; собран только диагностический helper.

**17 scenarios PASS**, exit0:8 overflow400 без native generation, OpenAI JSON
и Anthropic SSE4080 prompt /8 output с точными одинаковыми IDs, пять controls
52/32 с совпадением со старым greedy reference, два disconnect под давлением,
восстановление в том же PID59184 и cleanup всех owned processes.
**33 различных CPU methods PASS**,0 skips;109 source snapshots сверены.
GPU-only/bytes/drain/global95% PASS. Arena18432→18304МиБ,2 trims;
пики среди всех видов samples: RAM109,264ГиБ (87,025%), VRAM30,085ГиБ (94,479%).
Наблюдаемые cancel/drain141/156мс, опрос50мс.

Fresh4080-token prefill405,253с и328,856с. Decode8-token окон2,455/6,676
токена/с не является сравнением API или устойчивым speedup: кеши и порядок
различаются. Короткие controls3,186–4,088 токена/с. Defaults сохранены.
Нагрузка — отдельный read-only RAM mapping и1248МиБ CUDA allocations;
runtime RAM cache остаётся0. Native memory samples подтвердили≥85% обеих
памятей в каждом полном pressure/recovery запросе, но не на всей его длине.

Команды, timings и границы: [MM27-24](MINIMAX_M27_HTTP_CONTEXT.md).
Артефакты: `build-local/minimax-m2-http-context-01`,
[проверка evidence и источников](MINIMAX_M27_HTTP_CONTEXT_CHECK.json).
Следующий шаг — P4 resident prefix reuse с fresh parity/isolation/cancel.
Независимый oracle, широкое качество, старые FAIL, installer и DFlash serving открыты.

### MM27-23 / часть P2.5/P2.6 — 2026-10-09 — Live multi-tools и web chat

Основание: commit `54467693c8354d8c64a72ec1cb083473a529b4ba`, dirty tree;
dependency86ebfef2, те же model/header/template и EXE SHA
`f4ff449823beee3db1b260597cac1ed9276fb53922c5816f082f77ccbf52f619`.
Native/CUDA, sampler и cache/pipeline не менялись.

Исправлены порядок `name/description/parameters/strict` в tool definitions обоих
API и unconditional session header web chat. Health сообщает capability;
клиент опускает header при явном false. Добавлены live matrix, независимый SSE
reader, focused CPU tests и helper для браузерной проверки настоящей модели.

Результаты:4 live cycles /8 requests PASS. Два calls с integer revision и
кириллицей, reversed results, нормальный/error final, EOS/usage/GPU-only/bytes/
drain/memory95% PASS. Prompt/generated IDs совпали между всеми четырьмя первыми
запросами и обоими follow-up парами. Decode3,788–3,920 токена/с; CPU regressions
выполнялись одновременно, это не isolated speed A/B.
213 различных CPU методов,16 scripted cycles,32 HTTP replays,272 report checks,
32 native prompts /14284 IDs PASS,0 skips. Браузерный диалог сохраняет историю
и разделяет reasoning/final; два Stop дошли до native Engine в одном PID.

Исходный CPU field-order FAIL и прерванный диагностический live проход01
сохранены. Полные команды, browser results и ограничение истории после Stop:
[MM27-23](MINIMAX_M27_LIVE_TOOLS_WEB.md). Сводный проверяемый отчёт:
[CHECK](MINIMAX_M27_LIVE_TOOLS_WEB_CHECK.json).

Следующий шаг: HTTP context4K/pressure/recovery. Sessions, installer, independent
oracle, старые numerical/quality FAIL и DFlash serving остаются открытыми.

### MM27-22 / часть P2.5/P2.6 — 2026-10-09 — Native API Engine и повторная отмена

Основание: commit `54467693c8354d8c64a72ec1cb083473a529b4ba`, dirty tree;
dependency86ebfef2, прежний model/header/template. Новый EXE SHA
`f4ff449823beee3db1b260597cac1ed9276fb53922c5816f082f77ccbf52f619`.
Изменены Engine/worker/CLI, добавлены unit/live/startup checkers и optional
requirements. Native main.cpp использует постоянный Windows console handler.
GPU math, веса, cache/pipeline, sampler/defaults и остальные модели не менялись.

31 CPU methods PASS,0 skips, exit0; build exit0. Финальный live API checker:
11 scenarios PASS, exit0, `build-local/minimax-m2-live-api-02/report.json`.
Standalone CLI:6 auth/HTML/prefix/unload/reload checks PASS, exit0,
`build-local/minimax-m2-startup-01/report.json`. Sources/EXE/report hashes
проверены в `MINIMAX_M27_NATIVE_API_VALIDATION_CHECK.json`; sampled global
RAM30,68%, VRAM84,65%, обе ниже95%. Серверы после тестов остановлены.
Полный EN ответ414 IDs совпал с эталоном;3,978 decode tokens/s, prefill8,494с,
TTFT8,494с, HTTP wall112,656с. Три prefill disconnect после stop-string cancel
сохранили PID и дали восстановление за0,531–0,625с. Live get_code/Oslo вызван
самой моделью, ID результата связан, финальный ответ содержит OSLO-4179 и EOS.
Tools53/37 generated, decode3,332/3,751 tokens/s, sampling T1/p0,95/k40/seed42.

Исходный `minimax-m2-live-api-01` FAIL сохранён: первый cancel проходил,
второй завершал native process из-за reset SIGBREAK→SIG_DFL в MSVC.
Старые EXE/main/manifest — `minimax-m2-live-api-baseline-01`.
Новый handler проверен четырьмя отменами в одном native process и token recovery.
[Команды, отчёты и границы проверки](MINIMAX_M27_NATIVE_API.md).
Это один tool cycle, не общая model-quality приёмка. Anthropic live tools,
multi-call/error corpus, web chat, HTTP pressure/4K, sessions, installer,
independent oracle и прежние numerical/quality FAIL остаются открытыми.

### MM27-21 / часть P2.5/P2.6 — 2026-10-09 — OpenAI/Anthropic API adapters

Ревизия `d342001b4dca70860bcd585f28004591a83e5bf3`, dirty tree.
Добавлены `serve/minimax_m2_api.py`,15 tests и API checker. Canonical messages,
tools и fixed reasoning передаются в MiniMax template/parser через Service.
В общем сервере добавлены optional decoder hook и terminal UTF-8 flush;
MiniMax использует strict decoder. Profile/startup пока не зарегистрирован.

**PASS, exit0:**191 tests (15 API +51 MiniMax +125 общих/других моделей), без
skips;16 двухшаговых scripted tool cycles и32 HTTP replays восьми прежних
completions,272 checks. Использованы production handlers, реальные sockets
127.0.0.1, оба API, JSON/SSE и native tokenizer.32 prompts /14284 IDs совпали
с CPU native oracle. Проверены ID/result order, precise JSON values, usage,
EOS/length/stop, malformed UTF-8, disconnect и следующий успешный запрос.

Первый length fixture неверно учитывал special token; первый replay harness
оставлял старый single-script reference. Ошибки тестов исправлены, evidence
сохранён. Это не ошибки CUDA/model output. Native engine/math/cache/pipeline
не менялись, GPU inference и новые timings не запускались.

[Контракт и команды](MINIMAX_M27_API.md), [отчёт](MINIMAX_M27_API_CHECK.json),
[source/EXE hashes и проверка](MINIMAX_M27_API_VALIDATION_CHECK.json).
Финальные артефакты: `build-local/minimax-m2-api-check-02`.
Следующий пункт — native JSONL Engine bridge и startup profile; после них live
model toolcall → result → answer и web chat. Общий P2/P6 остаётся PARTIAL,
quality FAIL и independent model oracle по-прежнему открыты.

### MM27-20 / P2.4 — 2026-10-09 — streaming tool-output parser

Ревизия `d342001b4dca70860bcd585f28004591a83e5bf3`, dirty tree.
Добавлен `serve/minimax_m2_tools.py`,18 unit methods, ручные wire fixtures и
`tools/check_minimax_m2_tools.py`. Вся группа invokes проверяется до выдачи
calls/IDs, аргументы сохраняют raw string/JSON types и проходят schema validation.
Невалидные/неполные группы остаются текстом; malformed/oversized group отключает
дальнейшее распознавание calls. Stop применяется до тегов, group buffer ограничен.
Требуется optional jsonschema только при tools; поддержанный wire/schema contract
описан явно. Истории с полученными ID и переставленными results проверены.

**PASS, exit0:**51 unit methods,281 native tool groups +281 native history prompts,
22079 token IDs; whole/token/byte delivery и значения, включая signed zero,
int64 и finite double.8 прежних completions прошли по tokens/bytes с tools on/off.
Итого3966 report checks, отдельно от unit tests. Это template fixtures и replay,
не новые вызовы инструментов от полной модели. GPU/speed не запускались.

[Контракт, команды и ограничения](MINIMAX_M27_TOOLS.md),
[отчёт](MINIMAX_M27_TOOLS_CHECK.json),
[hashes и неизменность прежних файлов](MINIMAX_M27_TOOLS_VALIDATION_CHECK.json).
Артефакты: `build-local/minimax-m2-tools-check-02`, первый проход01 сохранён.
Следующий пункт — P2.5/P2.6 API adapters и live toolcall → result → answer.
Quality FAIL MM27-18b, independent full-model oracle, P4 sessions и DFlash live
lifecycle остаются открытыми; общий P2/P6 этим этапом не закрыт.

### MM27-19 / P2.3 — 2026-10-09 — history normalization и canonical tool inputs

Ревизия `d342001b4dca70860bcd585f28004591a83e5bf3`, dirty tree.
Добавлены `serve/minimax_m2_history.py`,20 unit methods, pinned GGUF template
fixture, ручной корпус и native checker. Начальные system/developer объединяются,
старый reasoning удаляется из копии, final/whitespace/буквальные теги сохраняются.
Calls сохраняют ID и typed JSON; все6 порядков ответов трёх одноимённых calls
дают правильный prompt. Late instructions, отсутствующие/чужие/повторные result
IDs, незавершённые группы и неэкранированные structural tags отклоняются явно.

В native raw oracle обнаружено округление float и переполнение unsigned integers.
Новый отдельный `strata-minimax-m2-template-json` использует точный JSON double
serializer; original/generated hashes и явный provenance записаны. Целые
входного контракта ограничены signed64. Исходный raw oracle и отрицательный
number probe сохранены. Первый build с конфликтующим include path исправлен;
финальные CPU build и checker — **exit0**.

**PASS:**33 unit methods (20 history +7 parser +6 tokenizer/input),23 вручную
заданных prompts,293 native prompts/2074 checks/71189 token IDs,268 numeric cases,
96 старых template cases.114 baseline hashes проверены. Existing Python helpers,
GPU math/cache/pipeline и bench EXE не изменились; inference/speed не запускались.

[Полный контракт и воспроизведение](MINIMAX_M27_HISTORY.md),
[результаты](MINIMAX_M27_HISTORY_CHECK.json),
[source/EXE/manifest hashes](MINIMAX_M27_HISTORY_VALIDATION_CHECK.json).
Артефакты: `build-local/minimax-m2-history-check-04`; прежние запуски сохранены.
Следующий пункт — P2.4 streaming tool-output parser. OpenAI/Anthropic adapters,
live tool cycle, no-thinking, sampling quality FAIL и independent model oracle
по-прежнему открыты; этот этап не закрывает общий P2/P6.

### MM27-18b — 2026-10-08–09 — request-local sampling, EOS корпуса и отдельный quality FAIL

На прежнем dirty tree добавлены `sampling.hpp`, `check_sampling.cpp`, optional
JSON request config в `main.cpp`, target/manifest hashes и два Python drivers.
Ревизия при фиксации `d342001b4dca70860bcd585f28004591a83e5bf3`;
baseline source/EXE snapshot сохранён до изменений.
Явные header dependencies исправили пропущенную пересборку локализованным MSVC.
188 hashes baseline snapshot проверены; shared/backend math и serving не менялись.

Build/48 CPU/greedy96/4894-token screen/14 pipe/48 parser checks: **exit0**.
Все979113216 corpus logits конечны, CPU sampler replay выбрал те же4894 IDs.
Полный seed42 repeat и32-token seed7 prefix побитно совпали. Адресная отмена
после8 tokens:2,254 мс в одном измерении; последующий64-token ответ совпал.
Русский список завершился на398/567/1092 tokens при T1, на1219 при T0,7.
Forward decode полных ответов4,268–4,544 токена/с; sampling0,73–0,91 мс/токен.
Это не A/B speedup: ответы и длины различаются. RAM/VRAM sampled peaks30,07/84,66%.

Natural-completion gate **PASS**; ручной quality review **6 PASS /1 FAIL**.
Английский final ошибочно указал≈10 для `(650/450)^4 ≈ 4,3531`.
Итоговый validation **pass=false** при `implementation_checks_pass=true`;
исторический greedy FAIL тоже сохранён. T1/top_p0,95/top_k40 — opt-in пример,
а не утверждённый P6 профиль. Default greedy/кеши/precision не изменены.

[Полный отчёт, команды, таблица и артефакты](MINIMAX_M27_SAMPLING.md).
[Сводные gates и source/EXE/report hashes](MINIMAX_M27_SAMPLING_VALIDATION_CHECK.json).
Bench SHA `07844c0345c12f78ded9f24d7f86154c9f360bfb574e96569e8694489268b0ef`.
Следующий пункт: P2.3 history normalization, затем streaming tool parser/API;
independent model oracle и качество остаются отдельными условиями P6.

### MM27-18 — 2026-10-08 — bounded output, английский EOS, JSONL отмена и reasoning parser

В `main.cpp` max_tokens ограничен свободным context budget вместо256,
default8 сохранён. Добавлены stop_token_id, явный token-boundary cancellation
и Windows SIGBREAK для адресной отмены созданной process group. Старый raw
text/token stream по-прежнему содержит EOS. Native математика и stop IDs не менялись.

На полном target, strict F32 ctx2048/batch16,18/2/4, budget1536:
английский запрос и повтор завершились на414 tokens, decode4,170/4,248 токена/с,
request112,252/107,592 с. Все414 IDs и82826496 logits повторились побитно.
Первые256 каждого из трёх запросов совпали с MM27-12 reference:768 IDs и153649152 logits.
Global sampled RAM24,89%, VRAM84,14%. Это correctness run, не speed A/B.

Русский запрос со списком1..96 дал1536 tokens без закрытия reasoning и final,
33 раза повторив одну фразу. **Полный natural-completion gate FAIL сохранён.**
17 live JSONL scenarios PASS:13 ошибочных запросов, default8, точный context
budget1996 с EOS414, реальный CTRL_BREAK после257 tokens и полный ответ после
отмены без перезапуска. Streaming IDs и raw text после восстановления совпали
с полным английским reference; pipe не экспортирует logits.

Отдельный `MiniMaxReasoningParser` сохраняет chunks/UTF-8/whitespace и partial
reasoning; tools не исполняет и в API не зарегистрирован.13 unit methods и18
проверок replay реальных output bytes PASS. API, history/tool correlation,
multilingual completion и рабочий sampling profile остаются открытыми.
Контрольный serial ReadFile/H2D без GPU/RAM cache и pipeline повторил1024
IDs и204865536 logits русского ответа побитно;14 повторов той же фразы,
reasoning не закрыт, stop length. Это numerical diagnostic PASS, не completion
PASS и не независимый model/framework oracle. Полный неудачный gate сохранён
в [validation report](MINIMAX_M27_COMPLETION_VALIDATION_CHECK.json).
[Подробности и воспроизведение](MINIMAX_M27_COMPLETION.md).

### MM27-17 — 2026-10-08 — RAM view LRU, GPU partition и отложенное admission

Добавлены `host_cache.hpp`, `check_host_cache.cpp`, `--ram-cache-mib` в bench
и checkers, telemetry и tuning support. File pipeline читает из ограниченных
read-only GGUF views. GPU fill удаляет пересекающиеся RAM views после fence;
активные readers удерживают адрес до конца memcpy. LRU учитывает alignment,
ограничен physical headroom, commit reserve,65536 entries и process target94%.
Global95% и source/lifecycle guards сохранены; shared backend файлы не менялись.

Первый eager prototype замедлил decode; cap84 вместо64 не помог. В final
варианте RAM admission ждёт предыдущего GPU refusal. Число созданий/удалений
на первом запросе уменьшилось19410/13137→3216/1755. Проверки повторены:
35 unit +228 matrix pipeline +120 grouped +180 sync +216 off regression
+18 fixture lifecycle +19 full pressure = **816 PASS**. Full pressure snapshot
89,92% RAM/94,10% VRAM; cache backing15,875→9,5625 ГиБ, logits bit-exact.

Три final off/RAM64 пары на корпусе52/52/343 input,32 output tokens:
median aggregate forward decode **3,941→3,485 токена/с (−11,57%)**,
workload **85,944→89,556 с (+4,20%)**. Default RAM0 сохранён. Это ordinary
forward timing, не useful throughput DFlash. Context2048/batch16, GPU cap18,
arena64/reserve0, file2/chunk4/lookahead/D2D, strict F32, FA/graphs/MTP off.

Дополнительный RAM64 long256 workload прошёл: все768 IDs/153649152 logits
совпали с сохранённым MM27-12 reference; peak views55,859 ГиБ, global RAM69,88%.
Лимит64 не достигнут; все ответы закончились по length. Скорость3,635/3,689/4,808
токена/с не является A/B с текущим off на длинном корпусе. Final total:
**21 ответ /1344 IDs /268886016 logits** побитно совпали с эталонами.

[Полный отчёт и артефакты](MINIMAX_M27_RAM_CACHE.md). RAM/GPU partition
относится к собственным views, не всему системному file cache. Утверждённого
быстрого RAM-профиля, natural EOS, API и session reuse по-прежнему нет;
MM27-06 reload discrepancy OPEN.

### MM27-16 — 2026-10-08 — Группы gate/up/down, readiness и full-model A/B

Добавлен MiniMax-only `group_cache.hpp` и флаг `--cache-group-experts 0|1`.
Три матрицы одного эксперта размещены в одном слоте; замена/eviction/pins
действуют на целую тройку. Готовность компонента публикуется после CUDA fence;
новые fills защищены до конца router plan. Admission и frequency training
выполняются один раз за plan, чтобы отказ на gate не приводил к частичной
группе при позднем admission на down. GGUF и shared Step/Hy3/common не менялись.
В CMake исправлены зависимости generated backend от included headers/`.inc`.

31 unit,114 grouped pipeline,216 matrix regression,90 synchronous D2D,
18 fixture lifecycle и19 full-target pressure/lifecycle checks — **488 PASS**.
Контекст4K проверен на fixture; full target pressure использовал context512.
При реальном давлении измерено89,95% RAM/94,07% VRAM, physical cache уменьшился
с15,875 до9,3125 ГиБ с побитно точными logits. Production-stride GPU unit
вместил16 Q4 или14 mixed Q4/Q6 троек в128 МиБ (готовые bytes94,94/95,76%).

Три чередующиеся пары matrix/group на полной модели: cap18,arena64/reserve0,
file2/chunk4/lookahead/D2D batch,context2048/batch16,F32 activations/KV,
FA/graphs/MTP/DFlash off. Prompts52/52/343, по32 output tokens.
Все18 ответов/576 IDs/115236864 logits совпали со старым EXE побитно.
Median decode первого/повторного/нового запроса:
**3,460/4,664/3,860 → 3,460/4,699/3,817 токена/с**.
Агрегат3,912→3,930 (+0,47%), workload86,935→86,154 с; диапазоны перекрываются.
Это forward decode harness, напрямую не сопоставимый с useful DFlash timing.
Default grouping0 и экспериментальный18/2/4 сохранены.

Подробности, commands, hashes, raw artifact paths и сохранённые ранние FAIL:
[MINIMAX_M27_GROUP_CACHE.md](MINIMAX_M27_GROUP_CACHE.md).
Прежнее MM27-06 reload расхождение остаётся OPEN. P3.4, API, sessions,
рабочий профиль и DFlash live lifecycle не закрыты этим этапом.

### P5.DF-03 — 2026-10-08 — Offline greedy DFlash, router tie и полезная скорость

Strata HEAD `ea56544abb2a52c8eeace467265cc1639ea172b0`, дерево dirty;
закреплённая dependency `86ebfef2`, CUDA13.0.48/MSVC19.44.35222.0.
Добавлены `dflash_decode.hpp`, `speculative_bench.cpp`, `minimax_tokenwise.hpp`,
`Tokenwise.cmake` и `tools/bench_minimax_m2_dflash.py`; обновлены CMake/manifest.
Общие cache/pipeline implementations и веса не менялись на этом этапе.
Patch применяется только к generated SHA-checked CUDA dispatch.

Полный offline greedy цикл: borrowed anchor/MASK, non-causal draft block,
batched target verification, prefix acceptance, correction/bonus, rollback
обеих KV histories и только committed feature injection. GPU/drain/95% guards
сохранены. Три реальных chat prompts имеют48/58/59 input tokens, template
открывает `<think>`; ответы32/64 токена заканчиваются по limit и ещё могут
содержать reasoning. Context512, batch8, F32 KV, FA/graphs off, strict targetF32,
pipeline2/chunk4/lookahead/D2D batch, arena64/reserve0; Q3_K/Q5_K draft arithmetic
остаётся штатной для dependency.

Первый Q4/depth2 прошёл greedy IDs, но провалил full-logit gate: max_abs1.566925.
Трасса нашла изменение восьмого MoE expert76→160 в позиции84/слое30:
batch округлил biased score76 на один ULP ниже равного serial score160.
Окончательный opt-in tokenwise mode3 считает ordinary/attention и routed
matmuls по токенам при target verification, сохраняя пакетную доставку experts.
Разделения только весовых matmul оказалось недостаточно. Mode0 по умолчанию
сохраняет прежний путь; tolerances max_abs≤5e-4/NMSE≤1e-7 не менялись.
Это отдельный установленный numerical issue, а не закрытие MM27-06 reload FAIL.

Измерены Q3/Q4/Q5 depths1/2/4/7 при cap18, затем cap24 для off/Q4-depth1/2.
На коротком screen лучший Q4/depth1 дал3.775 токена/с, off/cap18 —3.684/3.699
до/после серии. После возврата памяти draft off/cap24 дал3.867,
а Q4/depth1/cap24 —3.661. Поэтому одного screen недостаточно.
Три64-token повтора off/cap24 и Q4-depth1/2 чередовали порядок;
дополнительные три off/cap18 выполнены после основной серии.

| Режим | Три агрегата полезных токенов/с | Медиана | Медиана трёх запросов, с |
|---|---|---:|---:|
| off/cap18 | 3.801 / 3.550 / 3.118 | 3.550 | 91.482 |
| off/cap24 | 3.533 / 3.971 / 3.364 | 3.533 | 95.433 |
| Q4/depth1/cap18 | 3.675 / 3.686 / 3.607 | 3.675 | 87.191 |
| Q4/depth2/cap18 | 3.548 / 3.051 / 3.497 | 3.497 | 89.967 |

Acceptance Q4/depth1 —82.52%, depth2 —75.68%. Номинальный выигрыш depth1
против off/cap18 по медиане3.54%; он мал относительно разброса off.
Дополнительный off/cap18 не чередовался с on, внешняя нагрузка/Windows cache
не контролировались. Устойчивый speedup не доказан; defaults и18/2/4 сохранены.
Depth1 использовал около2.4% времени generation на draft compute; target
verification и доставка experts занимают основное время. H2D decode2.013 ГиБ
на полезный токен против1.801 у off/cap24. Это host phase timers, не CUDA profile.

Итоговые29 запусков/87 request comparisons:3936 output IDs и787451904 logits
побитно совпали. Все sampled global peaks ниже95%: VRAM≤94.212%, RAM≤39.345%.
Cap24 реально дал до21.5625 ГиБ arena, а не24 ГиБ. Процессные private/working-set
peaks benchmark отдельно не экспортировал; глобальные RAM нельзя приписать ему.
После замеров1404 host acceptance cases и49 GPU runtime fixtures PASS.
Старый обычный EXE `c13e1583…` и новый `d6c669ec…` независимо совпали с serial
режимом нового driver:96 output IDs и19206144 logits каждый, побитно.
Сборки/проверки exit0. P5.DF-02 JSON и исходные failed runs сохранены.

Команды, полные числа, build/source hashes и ссылки на четыре suite reports,
router trace и validation:
[MINIMAX_M27_DFLASH_BENCHMARK.md](MINIMAX_M27_DFLASH_BENCHMARK.md).
Артефакты: `build-local/minimax-m2-dflash-speed-screen-02`,
`minimax-m2-dflash-memory-screen-01`, `minimax-m2-dflash-speed-confirm-01`,
`minimax-m2-dflash-off18-confirm-01`, `minimax-m2-dflash-speed-validation-01`.
Live GPU EOS/cancel/pressure/recovery, long context, sessions и serving остаются
отдельными gates. Host EOS self-test не заменяет их.

### P5.DF-02 — 2026-10-08 — DFlash loader, borrowed head и GPU probe

Основание: `ea56544abb2a52c8eeace467265cc1639ea172b0`, dirty MM27-12…15
сохранены. До правок сохранены backend/EXE/manifest/build logs в
`build-local/minimax-m2-dflash-baseline-01`, исходный bench SHA-256
`9d075e2f914e53014e44dd9033f3264e26c7e75303ddcd92ae27db909dcc1888`.
Target header и полные draft hashes повторно совпали с P5.DF-01.

Изменено: C++ header admission всех58 draft tensors, tokenizer prefix/merges,
feature layers и ranges; no-allocation loader; generated `dflash.cpp` с
компактизацией каждой строки logits200064→200055 после shared projection.
Веса embedding/head остаются у target. Добавлен отдельный GPU probe и runner
с сохранением EXE/source/logits/stderr. Архив и extracted dependency не менялись.
Обычные bench/API defaults прежние; speculative serving не включён.

Сборки: CUDA13.0.48, MSVC19.44.35222.0, Ninja Release, arch120/effective120a;
dependency `86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9`.
Новый bench SHA-256 `c13e15830aa96734e6bef994aff3e2f544bfaaf310744e9a5c56492205fd579b`;
probe `aac1296badce971286c0857c626692485d707d9719ac23d33346f3dc05213e92`.
[CUDA manifest](MINIMAX_M27_DFLASH_BUILD_MANIFEST.json),
[CPU manifest](MINIMAX_M27_DFLASH_CPU_BUILD_MANIFEST.json).

Проверки, все exit0:

- CPU build/CTest4 version checks PASS; отдельный CPU loader3/3 и negative
  checks3/3 PASS. CUDA build loader3/3 и те же negative3/3 PASS.
  Weight allocation=0 в oracle. Неверный layer/tokenizer и truncated payload
  отклоняются до weight loading.
- RTX5090, driver581.80, RAM125.555 ГиБ: Q3/Q4/Q5 по30 GPU checks, всего90 PASS.
  Depths1/2/4/7, all output rows/full-head prefix, anchor/MASK, feature capture,
  noise rollback, committed-prefix injection и fresh replay.
  MASK/head/capture/replay побитны. Target batch-vs-serial greedy совпал,
  max_abs≤3.052e-5, NMSE≤8.864e-13; прежние tolerances5e-4/1e-7 сохранены.
- Global sampled peaks: VRAM92.835…93.694%, RAM26.341…26.438%; guard95%.
  Context512/batch8, F32 KV/target activations, FA/graphs off,
  cache18 ГиБ/arena64/reserve0, readers2/chunk4/lookahead1/D2D batch1.
  Q3_K/Q5_K draft kernels используют обычную quant арифметику dependency.
- Runtime fixture49/49 PASS. Генерация без DFlash: три запроса по24 токена,
  context2048/batch16, тот же pipeline18/2/4; все14 404 608 float32 logits и
  72 token IDs побитно равны `minimax-m2-reclaim-ab-01/01-legacy`.
  Один regression run дал3.349/4.637/4.114 токена/с; это не A/B и не speedup DFlash.

Ограничение: probe использует первые8 токенов технической строки, без chat
template. Единственный verification block принял0/7; tested committed prefix
содержит anchor. Это не corpus acceptance. Положительные prefixes/все reject
positions, EOS/limit/cancel/induced pressure, длинный context и полный loop TODO.
Numerical equivalence draft с NVIDIA reference и provenance локальных квантов
не доказаны. Старый MM27-06 native reload discrepancy остаётся OPEN.

Команды:

```powershell
.\build-local\build-minimax-m2-dflash.bat
.\build-local\build-minimax-m2-dflash-cpu.bat
$modelDir = 'H:\models\MiniMax-M2.7'
python -X utf8 tools/check_minimax_m2_dflash.py --model "$modelDir\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf" --draft "$modelDir\MiniMax-M2.7-DFlash-Q3_K_M.gguf" --draft "$modelDir\MiniMax-M2.7-DFlash-Q4_K_M.gguf" --draft "$modelDir\MiniMax-M2.7-DFlash-Q5_K_M.gguf" --gpu-draft "$modelDir\MiniMax-M2.7-DFlash-Q4_K_M.gguf" --gpu-draft "$modelDir\MiniMax-M2.7-DFlash-Q3_K_M.gguf" --gpu-draft "$modelDir\MiniMax-M2.7-DFlash-Q5_K_M.gguf" --out build-local/minimax-m2-dflash-probe-03
python -X utf8 tools/check_minimax_m2_dflash.py --model "$modelDir\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf" --draft "$modelDir\MiniMax-M2.7-DFlash-Q3_K_M.gguf" --draft "$modelDir\MiniMax-M2.7-DFlash-Q4_K_M.gguf" --draft "$modelDir\MiniMax-M2.7-DFlash-Q5_K_M.gguf" --build build-local/minimax-m2-oracles --out build-local/minimax-m2-dflash-cpu-01
python -X utf8 tools/run_minimax_m2.py --engine build-local/minimax-m2-cuda/bin/strata-minimax-m2-runtime-check.exe -- build-local/minimax-m2-dflash-fixtures-01
python -X utf8 tools/tune_minimax_m2_pipeline.py --model "$modelDir\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf" --configs build-local/minimax-m2-dflash-regression-config.json --repeats 1 --tokens 24 --reference-logits build-local/minimax-m2-reclaim-ab-01/01-legacy.f32 --reference-report build-local/minimax-m2-reclaim-ab-01/01-legacy.json --out build-local/minimax-m2-dflash-regression-01
```

Выходные каталоги уже существуют; для повторного запуска нужны новые имена.
Полные build logs: `build-local/minimax-m2-dflash-{configure,build}.log` и
`minimax-m2-dflash-cpu-{configure,build,ctest}.log`.
Результаты: [GPU/loader](MINIMAX_M27_DFLASH_PROBE_CHECK.json),
[CPU loader](MINIMAX_M27_DFLASH_CPU_CHECK.json),
[runtime fixture](MINIMAX_M27_DFLASH_RUNTIME_CHECK.json),
[обычная генерация](MINIMAX_M27_DFLASH_REGRESSION_CHECK.json).
Все EXE/source/logits сохранены в соответствующих каталогах `build-local`.
Следующий шаг: **P5.DF-03**, полный greedy driver и output parity на chat corpus.

### P5.DF-01 — 2026-10-08 — Перенос target и инспекция трёх DFlash

Target найден по новому пути в `H:\models\MiniMax-M2.7`; прежний путь отсутствует.
Размер и header hash совпали с MM27-01, основной loader contract/ranges PASS.
Добавлен `tools/inspect_minimax_m2_drafts.py`, три DFlash проверены без GPU:
GGUF/ranges, metadata, общая часть tokenizer и полные draft SHA-256, exit0.
[Отчёт](MINIMAX_M27_DFLASH_INSPECTION.json), [детали и продолжение](MINIMAX_M27_DFLASH.md).
Проанализированы DeepSeek DSpark и generic DFlash в закреплённой dependency.
Обнаружена необходимость адаптации head/logit width200064/200055.
Engine, его defaults и исходные GGUF не менялись; DFlash runtime/acceptance/speed TODO.
Исторические JSON/manifests с прежними путями сохранены без редактирования.

### MM27-15 — 2026-10-08 — Запас для роста GPU arena

Реализован `--arena-growth-reserve-mib N` (0..1024, default0) в MiniMax
bench/cache/lifecycle и optional `arena_growth_reserve_mib` в tuner JSON.
Запас вычитается из global free VRAM только при создании нового блока.
Свободные слоты существующих блоков можно заполнять и переиспользовать;
порог сокращения кэша, hard backing cap, host commit и global95 guards прежние.
Запас не вычитается повторно из cache cap. Allocator `cuda` его не использует.
Вычитание насыщаемое: свободная память меньше запаса не вызывает underflow.
Shared Hy3/Step headers в этом этапе не менялись; семантической группировки нет.

Основание: `ea56544abb2a52c8eeace467265cc1639ea172b0`, dirty MM27-12/13/14 сохранены.
До правок сохранены engine/source snapshot и build logs в
`build-local/minimax-m2-growth-baseline`. Новая сборка SHA-256
`9d075e2f914e53014e44dd9033f3264e26c7e75303ddcd92ae27db909dcc1888`. [Manifest](MINIMAX_M27_GROWTH_BUILD_MANIFEST.json)
содержит source/artifact hashes и toolchain/dependency. Commit не создавался.
Тот же локальный GGUF/header/template, RTX5090/32607 МиБ/driver581.80,
RAM125,555 ГиБ, CUDA13.0.48/MSVC19.44.35222.0/Ninja Release120a.

**Управляемое воспроизведение:** блок8 МиБ, матрицы4 МиБ, двадцать колебаний
бюджета255↔256 МиБ. Без запаса —20 освобождений и20 новых allocations;
с запасом32 МиБ —0/0, retained224 МиБ. Все surviving bytes совпали полностью.
Дополнительные проверки подтвердили заполнение слотов внутри резерва,
рост обратно до полного cap256 при возвращении global headroom и сокращение
до128 МиБ при сильном давлении. [25 arena](MINIMAX_M27_GROWTH_ARENA_CHECK.json)
и [25 reclaim checks](MINIMAX_M27_GROWTH_RECLAIM_CHECK.json) — PASS.
Это воспроизведение колебаний бюджета, не объяснение всех driver/NVML колебаний.

**Полная модель:** file reader,2 readers,chunk8,lookahead1,D2D batch1,
ctx2048/batch16, strict F32 activations/KV, FA/graphs/MTP off, greedy.
Корпус52/52/343 prompt tokens, по24 output tokens. Новый процесс и пустой
GPU cache для каждого workload; OS file cache и внешняя нагрузка не фиксированы.
Сначала [четыре screening runs](MINIMAX_M27_GROWTH_SCREEN_CHECK.json) при cap20:
64/reserve0,32/0,32/32,8/32. Все дошли до20 ГиБ backing без pressure trims,
пик общей VRAM89,65%. При достаточном запасе памяти новый флаг не влиял на рост.
Фон отличался от MM27-14, где cap20 уже ограничивался физической памятью.

Для проверки у лимита запрошен cap24 ГиБ: действующий guard ограничивал
фактический backing21,125–21,500 ГиБ. Три повторения каждого варианта,
порядок A/B/C, C/B/A, A/B/C. Медианы [confirmation](MINIMAX_M27_GROWTH_CONFIRM_CHECK.json):

| Блок, МиБ | Запас роста, МиБ | Decode первого / повтора / новой темы, ток/с | Workload, с |
|---:|---:|---:|---:|
| 64 | 0 | 3.270 / 4.605 / 4.130 | 80.241 |
| 32 | 0 | 3.279 / 4.588 / 4.114 | 80.881 |
| 32 | 32 | 3.305 / 4.665 / 4.140 | 79.476 |

Для32 МиБ запас улучшил медиану decode повтора на1.69%; относительно64
медиана выше на1.32%. Это небольшое различие: диапазон повтора64 составил
4.466–4.684 ток/с, а32/reserve32 —4.537–4.672. В первой тройке64 был быстрее
32/reserve32. Изменялись также sampled budgets и фактическая residency.
Устойчивое преимущество над64 МиБ не установлено; default **64 МиБ/reserve0**
и экспериментальные **18 ГиБ /2 readers /chunk4** сохранены.
Cap24 в этом тесте не выбран новым рекомендуемым cap.

Число pressure groups за весь workload (prefill+decode) по повторам:
64/0 —0/5/1,32/0 —2/2/1,32/32 —1/0/0. В этой серии не повторились сотни
освобождений MM27-14; сильный эффект гистерезиса доказан управляемым replay.
При примерно одинаковом backing у32 было меньше useful live cache:
в первом decode первого прохода20,487 ГиБ против20,763 у64 (выравненные slots).
Гистерезис не меняет упаковку size classes и не устраняет эту разницу.

Все13 workloads screen+confirm совпали с сохранённым reference побитно:
**187259904 пар float32 logits**, одинаковые72 token IDs на workload.
Memory95, GPU-only, copy accounting, drain и `arena_reserved <= cache_limit`
проверены tuner; этот physical-budget gate теперь включён в постоянный audit.
Пик обычных замеров: RAM26,745%, VRAM94,289%. Сохранены EXE, sources, requests,
JSON/stdout/stderr и полные logits каждого запуска.

**Другие проверки:** [1152 cache/pipeline checks](MINIMAX_M27_GROWTH_FIXTURE_CHECK.json)
(sync8/reserve32; pipeline8/32,32/32,64/0) — PASS. Worker/copy faults, cancel,
pins, OOM bypass и recovery проверены. [37 lifecycle checks](MINIMAX_M27_GROWTH_LIFECYCLE_CHECK.json)
(fixture18 + full_pressure19) — PASS при cap24/block32/reserve32.
Реальная нагрузка: read-only model mapping85,375 ГиБ и отдельный GPU holder12 ГиБ;
пики RAM90.019%, VRAM94.201%. Cache backing сократился16,031→10,688 ГиБ
под внешним давлением; logits recovery/unload/reload побитно совпали.
[41 CPU regression и четыре invalid-CLI checks](MINIMAX_M27_GROWTH_REGRESSION_CHECK.json) — PASS.
Все build/test commands завершились exit0; invalid CLI ожидаемо дал exit2.
Отдельные2K/4K и shared Hy3 GPU checks повторно не запускались: allocator Hy3
в этом этапе не менялся, прежние результаты остаются историческими.
MM27-06 tiny native-after-cache discrepancy остаётся **OPEN**; новые PASS его не объясняют.

**Воспроизведение:** из корня репозитория, новые output paths для каждого запуска.

```powershell
& .\build-local\build-minimax-m2-runtime.bat
python -X utf8 tools/run_minimax_m2.py --engine build-local/minimax-m2-cuda/bin/strata-minimax-m2-arena-check.exe -- build-local/minimax-m2-growth-arena-01.json
python -X utf8 tools/run_minimax_m2.py --engine build-local/minimax-m2-cuda/bin/strata-minimax-m2-reclaim-check.exe -- build-local/minimax-m2-growth-reclaim-01.json
python -X utf8 build-local/check-minimax-growth-fixtures.py
$modelPath = 'H:\models\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf'
python -X utf8 tools/tune_minimax_m2_pipeline.py --model $modelPath --configs build-local/minimax-m2-growth-screen-configs.json --out build-local/minimax-m2-growth-screen-01 --reference-logits build-local/minimax-m2-reclaim-ab-01/01-legacy.f32 --reference-report build-local/minimax-m2-reclaim-ab-01/01-legacy.json
python -X utf8 tools/tune_minimax_m2_pipeline.py --model $modelPath --configs build-local/minimax-m2-growth-confirm-configs.json --out build-local/minimax-m2-growth-confirm-01 --repeats 3 --reference-logits build-local/minimax-m2-reclaim-ab-01/01-legacy.f32 --reference-report build-local/minimax-m2-reclaim-ab-01/01-legacy.json
python -X utf8 tools/check_minimax_m2_lifecycle.py --model $modelPath --out build-local/minimax-m2-growth-lifecycle-01 --cases fixture full_pressure --gpu-cache-mib 24576 --gpu-cache-allocator arena --arena-block-mib 32 --arena-growth-reserve-mib 32 --pipeline-readers 2 --pipeline-chunk-mib 8 --pipeline-lookahead 1 --pipeline-d2d-batch 1
```

Build logs сохранены как `build-local/minimax-m2-growth-runtime-{configure,build}.log`.
Рекомендуемый CLI не меняется. Новый флаг доступен для последующих экспериментов.
Следующий шаг MM27-16: плотная упаковка gate/up/down одного эксперта,
совместная admission/replacement и group pins, затем correctness/pressure/A/B.
Законченные ответы/EOS, API/sessions и MTP остаются отдельными незавершёнными задачами.

### MM27-14 — 2026-10-08 — Размеры блоков 8/16/32/64 МиБ и дизайн группировки

По предложению пользователя проверено уменьшение64 МиБ. Добавлен CLI
`--arena-block-mib 8|16|32|64` в bench/cache/lifecycle, соответствующий аргумент
runtime и optional `arena_block_mib` в JSON конфигурации tuner.
Default **64 МиБ сохранён**. Это размер физической аллокации, независимый от
`--pipeline-chunk-mib`; выбранные матрицы копируются прежними отдельными slices.
Таргет может быть уменьшен меньшим cache cap или увеличен для oversized matrix,
округление allocator —2 МиБ. Guard глобальной RAM/VRAM95% и host commit сохранён.
Shared Hy3 allocator получил параметр с прежним default64; его политика прежняя.

Основание: `ea56544abb2a52c8eeace467265cc1639ea172b0`, dirty MM27-12/13 сохранены.
До правок сохранены engine/source snapshot в `build-local/minimax-m2-blocks-baseline`.
Новая сборка SHA-256 `dea0606f852f80346da02540f09b7d64381dd06c97058bfd45a4d9380a4ddd0a`;
source/artifact hashes и compiler/dependency — [manifest](MINIMAX_M27_BLOCKS_BUILD_MANIFEST.json).
Тот же GGUF/header/template, RTX5090/32607 МиБ/driver581.80, RAM125,555 ГиБ,
CUDA13.0.48/MSVC19.44.35222.0/Ninja Release120a. Commit не создавался.

**Размеры и освобождение.** На512 матрицах384×Q4 slot2,5625 МиБ +128×Q6 slot3,75 МиБ
все варианты arena заняли1536 МиБ backing при1464 МиБ charged live; наблюдаемый
прирост global NVML также1536 МиБ. Число CUDA-блоков:192/96/48/24 для8/16/32/64.
Отдельные cudaMalloc заняли2048 МиБ по NVML. Это один allocator-run на этом ПК;
NVML отражает и внешнюю активность, не только payload.
[20 arena checks](MINIMAX_M27_BLOCKS_ARENA_CHECK.json) прошли, включая oversized
10 МиБ при target8, alignment, reuse и освобождение всех allocations.

При снижении бюджета256→255 МиБ,64 матрицах по4 МиБ и перемежённом LRU
сохранилось **248/240/224/192 МиБ** при блоках8/16/32/64 соответственно.
[17 reclaim checks](MINIMAX_M27_BLOCKS_RECLAIM_CHECK.json) прошли; байты оставшихся
матриц проверены полностью. Значит меньший блок действительно сокращает
минимальную потерю кэша при единичном освобождении.

**Предварительный полный прогон, по одному на вариант:** cache20 ГиБ, arena,
file reader,2 readers, chunk8, lookahead1, D2D batch1, ctx2048/batch16, strict F32
activations/KV, FA/graphs/MTP off, greedy. Запросы52/52/343 prompt tokens,
по24 output tokens, новый процесс/пустой GPU cache для каждого workload.
OS file cache и внешняя нагрузка не фиксированы. Это отбор, не рейтинг defaults.

| Блок, МиБ | Decode первого / повтора / новой темы, ток/с | Workload, с |
|---:|---:|---:|
| 64 | 3.382 / 4.678 / 4.202 | 79.168 |
| 8 | 3.357 / 3.815 / 3.434 | 80.310 |
| 16 | 3.465 / 4.650 / 4.132 | 77.480 |
| 32 | 3.446 / 4.645 / 4.146 | 77.466 |

[Screen report](MINIMAX_M27_BLOCKS_SCREEN_CHECK.json). У8 МиБ decode-фазы вызвали
84/384/341 освобождений, у64 —0/0/0. Полезный кэш8 МиБ был18,16–18,28 ГиБ;
при64 —18,40 ГиБ. Уменьшение гранулярности сопровождалось частыми повторными
аллокациями/вытеснениями возле физического лимита и дополнительным H2D.
16 и32 МиБ дали почти одинаковое время;32 выбран для повторов из-за меньшего
числа аллокаций и pressure events в этом проходе.

**Подтверждение64↔32, три пары A/B,B/A,A/B, те же условия:**

| Запрос | 64 МиБ, ток/с | 32 МиБ, ток/с | Изменение |
|---|---:|---:|---:|
| Первый | 3.409 | 3.103 | -8.97% |
| Повторный | 4.638 | 4.255 | -8.25% |
| Новая тема | 4.208 | 3.912 | -7.04% |

Median workload **78.251→79.423 с**.
[Confirmation report](MINIMAX_M27_BLOCKS_CONFIRM_CHECK.json). В отдельных decode-фазах
32 МиБ было до334 освобождений, поэтому один хороший screening-run не подтвердился.
Вывод относится к cache20 у границы VRAM; меньшие блоки при cap18 здесь не сравнивались.
Все10 прогонов screen+confirm прошли token/logit parity с историческим MM27-13
reference:144046080 пар float32 logits, одинаковые72 token IDs в каждом прогоне.
Все memory/copy/drain gates PASS; отдельно проверено `arena_reserved <= cache_limit`.
64 остаётся default, экспериментальные18 ГиБ /2 readers /chunk4 сохранены.

**Другие проверки:** [1872 cache/pipeline checks](MINIMAX_M27_BLOCKS_FIXTURE_CHECK.json)
для sync8/64 и pipeline8/16/32/64 с lookahead/batching — PASS; все bytes/logits exact,
worker/copy faults, cancel, pins, OOM bypass, recovery/unload проверены.
[37 lifecycle checks](MINIMAX_M27_BLOCKS_LIFECYCLE_CHECK.json) при block32 включают
fixture и полную модель с реальным давлением на память. [Hy3 arena и CPU регрессии](MINIMAX_M27_BLOCKS_REGRESSION_CHECK.json).
2K/4K повторно не прогонялись: отдельная проверка MM27-13 остаётся исторической.

**Группировка — следующий эксперимент, пока не реализована.** Уже освобождается
целый физический блок. Семантическая группировка нужна, чтобы в нём находились
связанные по использованию матрицы. Первая естественная единица —
`(generation, layer, expert)` со всеми `gate/up/down`:

| Слои GGUF | Payload трёх матриц | Три выровненных слота | Расчёт размера отдельного блока |
|---|---:|---:|---:|
| 32 слоя: Q4/Q4/Q4 | 7,59375 МиБ | 7,6875 МиБ | 8 МиБ |
| 30 слоёв: Q4/Q4/Q6 | 8,75390625 МиБ | 8,875 МиБ | 10 МиБ |

Это [арифметика локального GGUF](MINIMAX_M27_BLOCKS_GROUPING_ESTIMATE.json),
не измерение скорости/VRAM такого allocator. Смежные ID экспертов сами по себе
не гарантируют совместный выбор router. Фиксированный16 МиБ блок для одного
смешанного expert bundle тратил бы много места; необходима плотная упаковка.

Нужны ключ группы и offsets матриц, совместная admission/replacement, явная
готовность каждой матрицы до cache hit и group pins для незавершённых copies.
Текущий `matching_reuse` заменяет одну матрицу: без изменения этого пути группы
перемешаются снова. Проверки: full bytes/logits, partial fills/cancel/pressure,
модельная плотность кэша и три A/B пары. Предварительно проверить запас между
порогами роста и сокращения, чтобы мелкие блоки не выделялись сразу после сброса.
Частые pressure events измерены; точная причина изменения внешнего free budget
не изолирована и требует отдельного контролируемого replay.

Команды:

```powershell
& .\build-local\build-minimax-m2-runtime.bat
python tools/run_minimax_m2.py --engine build-local/minimax-m2-cuda/bin/strata-minimax-m2-arena-check.exe -- build-local/minimax-m2-blocks-arena-01.json
python tools/run_minimax_m2.py --engine build-local/minimax-m2-cuda/bin/strata-minimax-m2-reclaim-check.exe -- build-local/minimax-m2-blocks-reclaim-01.json
python build-local/check-minimax-blocks-fixtures.py
python tools/tune_minimax_m2_pipeline.py --model "H:\models\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf" --configs build-local/minimax-m2-blocks-screen-configs.json --out build-local/minimax-m2-blocks-screen-01 --reference-logits build-local/minimax-m2-reclaim-ab-01/01-legacy.f32 --reference-report build-local/minimax-m2-reclaim-ab-01/01-legacy.json
python tools/tune_minimax_m2_pipeline.py --model "H:\models\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf" --configs build-local/minimax-m2-blocks-confirm-configs.json --out build-local/minimax-m2-blocks-confirm-01 --repeats 3 --reference-logits build-local/minimax-m2-reclaim-ab-01/01-legacy.f32 --reference-report build-local/minimax-m2-reclaim-ab-01/01-legacy.json
python tools/check_minimax_m2_lifecycle.py --model "H:\models\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf" --out build-local/minimax-m2-blocks-lifecycle-01 --cases fixture full_pressure --gpu-cache-mib 20480 --gpu-cache-allocator arena --arena-block-mib 32 --pipeline-readers 2 --pipeline-chunk-mib 8 --pipeline-lookahead 1 --pipeline-d2d-batch 1
```

Для повторения нужны новые output paths. Сохранены EXE, исходники, команды,
логи и raw logits. Math/precision не менялись, причина старого MM27-06 discrepancy
остаётся OPEN. API, длинные завершённые ответы и MTP — отдельные незакрытые задачи.

### MM27-13 — 2026-10-08 — Освобождение целых блоков GPU arena

Основание: `ea56544abb2a52c8eeace467265cc1639ea172b0`, dirty tree с предыдущими
MM27-12 docs/tools. Старая сборка и исходники сохранены до правок в
`build-local/minimax-m2-reclaim-baseline`. Commit не создавался.
Dependency `86ebfef2`, CUDA13.0.48/MSVC19.44.35222.0, Ninja Release120a,
RTX5090/32607 МиБ/driver581.80, RAM125,555 ГиБ, тот же локальный Q4_K_M GGUF.
Header/template hashes и все настройки точности совпадают с MM27-12.
[Build manifest](MINIMAX_M27_RECLAIM_BUILD_MANIFEST.json) содержит source/artifact hashes.

**Изменено.** `step35::ExpertCache` получил opt-in callback физической группы.
Его включает только MiniMax arena; Step/Hy3 сохраняют прежнюю policy.
`hy3::GpuArena::allocation_group()` только читает принадлежность живого slot.
При превышении физического бюджета выбираются целые блоки: сначала меньше
полезных байтов, затем наиболее давно использованный блок. Любой plan/fill pin
защищает весь блок; невозможность уложиться в бюджет вызывает ошибку и drain.
Нормальная частотная замена отдельных экспертов не изменилась. Политика работает
и в явном `refresh()`, и во внутреннем refresh при `admit()`; sampled budget
больше не уменьшается повторными logical trim. Добавлены четыре pressure-счётчика
в JSON: число вызовов, блоков, вытесненные live bytes и освобождённые backing bytes.

**Контролируемое воспроизведение:** четыре CUDA slab по64 МиБ, 64 матрицы по4 МиБ,
LRU перемежает блоки. Снижение бюджета256→255 МиБ заставило прежний цикл
вытеснить **256 МиБ /64 матрицы**, новый — **64 МиБ /16 матриц**, сохранив192 МиБ.
Все сохранённые байты проверены. Закреплённый блок, все блоки с pins, recovery,
частично заполненный блок, refresh внутри admission и нулевой бюджет проверены:
[13 checks PASS](MINIMAX_M27_RECLAIM_CHECK.json). Это измерение эффективности
освобождения памяти, а не показатель tokens/s.
Первый вариант теста ошибочно ожидал60 МиБ остатка у старого цикла; измерено0.
Исправлено ожидаемое поведение старого алгоритма, без изменения runtime/math
или numerical tolerances; исходный FAIL сохранён и указан в manifest.

**Проверки, все exit0:**

- [4920 cache/pipeline checks,25 конфигураций](MINIMAX_M27_RECLAIM_PIPELINE_CHECK.json), включая pending fills, pinned pressure, worker/copy faults и cancel.
- [47 lifecycle checks](MINIMAX_M27_RECLAIM_LIFECYCLE_CHECK.json): fixture, полная модель под реальным давлением,2K/4K, cancel/recovery/unload, побитные logits.
- [55 reload checks](MINIMAX_M27_RECLAIM_RELOAD_CHECK.json): девять lifetimes, десять одинаковых native outputs, исторические fixture/continuation hashes.
- [Регрессии](MINIMAX_M27_RECLAIM_REGRESSION_CHECK.json): MiniMax arena14, runtime49, shared Step cache29 (включая131200 решений fast-scan parity), Hy3 arena18, CPU41.
- Под реальным давлением RAM **89.967%**, VRAM **93.975%**; cache live **15.272→7.506 ГиБ**, backing **15.875→7.688 ГиБ**. Держатель давления выделил12 ГиБ GPU и read-only mmap pages до90% RAM. Это отдельный correctness run.

Контексты2K/4K используют repeated prefix и33 разных последних токена; это
численный stress, не широкая quality-проверка.

**Full-model A/B:** один и тот же cap20 ГиБ, arena, reader=file, readers2,
chunk8 МиБ, lookahead1, D2D batch1, ctx2048/batch16, strict F32 activations/KV,
FA/graphs/MTP off, greedy. Три пары A/B,B/A,A/B; новый процесс и пустой GPU cache
для каждого workload. Три запроса:52/52/343 prompt tokens и24 output tokens
каждый; OS cache и внешняя нагрузка не контролируются. Загрузка модели не входит
в request latency. Медианы по трём запускам:

| Запрос | Старая сборка, ток/с | Новая, ток/с | Изменение | TTFT, с: старая → новая |
|---|---:|---:|---:|---:|
| Первый запрос | 3.392 | 3.421 | +0.86% | 11.886 → 11.805 |
| Повтор запроса | 4.382 | 4.663 | +6.41% | 7.953 → 7.880 |
| Новая тема | 4.148 | 4.177 | +0.70% | 42.352 → 40.828 |

Median полного workload: **80.513→77.432 с**.
Это сокращение времени на **3,83%** в этом корпусе. Между отдельными прогонами
decode повтора менялся в пределах4,195–4,682 ток/с у старой сборки и
4,651–4,670 у новой. Ускорение первого запроса и новой темы меньше1%;
универсальный прирост по этим измерениям не заявляется.

В старых прогонах02 и03 на decode повтора счётчик
`cache_evictions - cache_reuses` составил3454 и4238 при одном `arena_frees`.
Размер кэша к концу фазы восстановился до18,34/18,40 ГиБ, скрывая промежуточную
потерю резидентности. В каждой из четырёх pressure-фаз новой сборки освобождён
один64 МиБ блок с вытеснением24 матриц /61,5 МиБ live. Внешнее давление в
разных процессах не идентично, поэтому controlled replay остаётся отдельным
доказательством механизма. Пики обычного A/B: RAM24,21% /VRAM94,20%; реальное
давление до89,97% /93,97% проверено отдельным lifecycle-тестом выше.
[Полный A/B report](MINIMAX_M27_RECLAIM_AB_CHECK.json) содержит individual timings,
cache residency, hit bytes, H2D, memory peaks, pressure counters и команды.
Все6 прогонов имеют одинаковые72 token IDs и SHA-256 logits
`35d2d4995390c2ac58739e343d3f05e85f412eeba2d6e917108888c96d97023e`;
5 сравнений / **72,023,040** float32 logits побитно совпали.
Новая сборка также прошла отдельный gate `arena_reserved <= cache_limit`.

**Вывод:** чрезмерное вытеснение воспроизведено и исправлено. Влияние на скорость
зависит от давления на VRAM; кэш без превышения бюджета использует прежний путь.
Эти пары сравнивают версии алгоритма при20/2/8, не выбирают оптимальный cap.
Экспериментальный18/2/4 и generic defaults cache0/allocator=cuda/pipeline0
сохранены. Более высокий cap пока не объявлен общим default.

Команды (GPU workloads выполнялись последовательно):

```powershell
& .\build-local\build-minimax-m2-runtime.bat
& .\build-local\build-minimax-m2-reclaim-shared.bat
python tools/run_minimax_m2.py --engine build-local/minimax-m2-cuda/bin/strata-minimax-m2-reclaim-check.exe -- build-local/minimax-m2-reclaim-check-02.json
python tools/check_minimax_m2_pipeline.py --out build-local/minimax-m2-reclaim-pipeline-01 --lookahead --d2d-batch
python build-local/check-minimax-m2-reclaim-validation.py
python tools/check_minimax_m2_reload.py --out build-local/minimax-m2-reclaim-reload-01 --check-mm27-06-baseline --pipeline-readers 2 --pipeline-chunk-mib 8 --pipeline-lookahead 1 --pipeline-d2d-batch 1
python tools/check_minimax_m2_lifecycle.py --model "H:\models\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf" --out build-local/minimax-m2-reclaim-lifecycle-01 --gpu-cache-mib 20480 --gpu-cache-allocator arena --pipeline-readers 2 --pipeline-chunk-mib 8 --pipeline-lookahead 1 --pipeline-d2d-batch 1
python tools/check_minimax_m2_reclaim.py --model "H:\models\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf" --baseline-dir build-local/minimax-m2-reclaim-baseline --out build-local/minimax-m2-reclaim-ab-01
```

Пути отчётов уже существуют; для повторения нужны новые output paths.
Сохранены old/new EXE, исходники, raw logits, stdout/stderr, commands и hashes.
Численные kernels не менялись. Старый MM27-06 tiny native-after-cache
**остаётся OPEN**: новые PASS не устанавливают причину прежнего расхождения.
Следующий шаг — MM27-14: generation budget выше256 в рамках допустимого
context, законченные reasoning/final ответы и natural EOS; затем API/session.

### MM27-12 — 2026-10-08 — Подбор кэша и конвейера без изменения kernels

**Результат:** прежние 18 ГиБ / 2 readers / chunk4 сохраняются как
экспериментальный вариант. Единственный проход отбирал кандидата, но повторный
A/B не подтвердил полезный выигрыш полного запроса от 20 ГиБ / chunk8.
Ни параметры по умолчанию, ни математический код, ни общие helpers не менялись.
Добавлен `tools/tune_minimax_m2_pipeline.py`: конфигурации из JSON, чередование
порядка, strict F32, сохранённые EXE/исходники/logits/команды, проверки 95%,
bytes/drain и побитного совпадения. Лимит cache и его фактическое заполнение
пишутся отдельно. Кандидат использует 32 + 32 МиБ колец, baseline 16 + 16 МиБ.

**Условия:** локальный MiniMax-M2.7 Q4_K_M, 128,841 ГиБ, RTX 5090 32607 МиБ,
RAM 125,555 ГиБ, Windows, driver 581.80, та же сборка MM27-11
`6e8e465dd2f653a2c282d22dc4dda41afc71f4f371b206f1659ff8fd40710f6a`.
Strict F32 activations/KV, FA/graphs/MTP off, greedy, ctx2048/batch16,
arena, file pipeline, router lookahead и D2D batch. Процессы запускаются
последовательно; каждый начинает с пустым GPU cache. OS file cache и внешняя
нагрузка не фиксировались. Счётчик ReadFile не является физическим SSD traffic.

**Предварительный перебор, по одному проходу.** Входы 52/52/343 токена,
выходы по 24 токена. Контроль 18 ГиБ запущен до/после перебора кэша;
контроль 2 readers/chunk4 — до/после перебора конвейера. Это не изолированный
рейтинг: фактические бюджеты VRAM различались, а контрольные замеры колебались.
Cap 20 и 24 в первом переборе дали одинаковые 18,83 ГиБ live при 19,45 ГиБ budget.

| Конфигурация | Первый / повтор / новая тема, ток/с | Всё, с | Бюджет кэша в конце decode, ГиБ |
|---|---|---:|---|
| `c18-r2-s4-start` | 3,41 / 4,62 / 4,12 | 78,79 | 18,00 / 18,00 / 18,00 |
| `c12-r2-s4` | 3,23 / 4,13 / 3,80 | 82,41 | 12,00 / 12,00 / 12,00 |
| `c24-r2-s4` | 3,48 / 4,73 / 4,26 | 75,74 | 19,45 / 19,45 / 19,45 |
| `c16-r2-s4` | 3,38 / 4,48 / 3,99 | 78,81 | 16,00 / 16,00 / 16,00 |
| `c20-r2-s4` | 3,49 / 4,73 / 4,27 | 75,85 | 19,46 / 19,46 / 19,46 |
| `c18-r2-s4-end` | 3,45 / 4,61 / 3,68 | 77,95 | 18,00 / 18,00 / 18,00 |

| Конфигурация | Первый / повтор / новая тема, ток/с | Всё, с | Бюджет кэша в конце decode, ГиБ |
|---|---|---:|---|
| `c20-r2-s4-start` | 3,06 / 4,26 / 4,08 | 82,91 | 18,82 / 18,86 / 18,87 |
| `c20-r1-s16` | 2,65 / 3,88 / 3,40 | 114,74 | 18,82 / 18,83 / 19,08 |
| `c20-r2-s8` | 3,46 / 4,67 / 4,18 | 77,37 | 19,03 / 19,12 / 19,12 |
| `c20-r1-s4` | 2,66 / 3,95 / 3,43 | 106,67 | 19,10 / 19,13 / 19,13 |
| `c20-r2-s16` | 3,47 / 4,31 / 4,06 | 78,74 | 19,09 / 19,00 / 18,98 |
| `c20-r1-s8` | 2,72 / 3,73 / 3,31 | 105,95 | 19,05 / 18,98 / 19,00 |
| `c20-r2-s4-end` | 3,36 / 4,55 / 4,17 | 83,00 | 19,02 / 19,02 / 19,05 |

Один reader проиграл при всех трёх размерах чанка. Для повторной проверки
выбран кандидат 20 ГиБ / 2 readers / chunk8 с наименьшим временем предварительного
прохода. Такой выбор ещё не означает оптимальный профиль.

**Подтверждение: три пары, A/B, B/A, A/B.** Здесь одновременно меняются cap
и chunk, поэтому эффект нельзя приписывать одному из них.

| Запрос | 18 ГиБ / 2 / 4 МиБ | 20 ГиБ / 2 / 8 МиБ | Изменение decode |
|---|---:|---:|---:|
| Первый короткий | 3,41 | 3,44 | +0,89% |
| Повтор | 4,61 | 4,67 | +1,41% |
| Новая тема | 4,13 | 4,16 | +0,74% |

Медиана суммарного времени: **77,70 → 81,02 с**,
то есть **+4,27% времени**.
Во второй паре повтор кандидата замедлился до 4,05 против 4,62 ток/с.
Live cache кандидата составил 13,59 / 16,01 / 18,41 ГиБ по трём запросам;
физический reserved в первом/втором оставался 18,94 ГиБ. У baseline live cache
во всех повторах 17,43 ГиБ. Рост cap не дал устойчивого преимущества.

Счётчики показывают освобождение 9 slab и 13 387 evictions в первом запросе
второй пары кандидата; OOM/reject здесь может означать отказ arena budget,
а не аппаратное исчерпание VRAM. По коду `mm27_cache_refresh` общий trim
продолжается до освобождения физических slab; отдельные holes ещё не уменьшают
reserved. **Рабочая гипотеза:** при небольшом изменении бюджета освобождается
избыточный объём полезного кэша. Точной трассы каждого trim здесь нет;
причину и исправление нужно проверить контролируемым replay в MM27-13.

**Более длинная генерация: одна пара с max_tokens=256.** Это проверка
точности и устойчивости; единичные скорости не подтверждают новый speedup.

| Запрос | Выходных токенов | Stop | Baseline / candidate, ток/с |
|---|---:|---|---|
| Первый короткий | 256 | length | 3,88 / 4,17 |
| Повтор | 256 | length | 4,21 / 4,29 |
| Новая тема | 256 | length | 5,13 / 5,19 |

Все logits и IDs пары совпали. Выходы с `length` ограничены лимитом и не
считаются завершёнными ответами. [Необработанные примеры](MINIMAX_M27_LONG_GENERATION_SAMPLES.md)
сохранены для просмотра. Во всех трёх запросах 256 токенов израсходованы на
reasoning; закрывающий `</think>` и конечный ответ не получены.
Это runtime/token-limit наблюдение, не подтверждение качества готовых ответов.
Широкий quality benchmark не выполнен.

**Проверки:** 21 запуск модели; **427 336 704 пары logits**
и **2136 пар token IDs** совпали побитно.
Все 19 коротких прогонов сохранили исторический SHA logits
`35d2d4995390c2ac58739e343d3f05e85f412eeba2d6e917108888c96d97023e`.
Все memory/byte/drain gates PASS. На кандидате дополнительно прошли
**24 lifecycle checks**: real pressure, cancel/recovery/unload и ctx4096
с повторяемым префиксом и 33 различающимися токенами; native/file logits точные.
Реальная нагрузка достигла **89,95% RAM / 93,98% VRAM**.
Это выборочные измерения глобальной памяти, а не непрерывные peaks.
Полную матрицу 4920 fixtures MM27-11 заново не запускали: численный engine,
все C++ исходники и shared helpers имеют прежние хэши.

Raw logs, inputs, EXE и logits: `build-local/minimax-m2-tune-*`.
Архивы: [cache](MINIMAX_M27_TUNING_CACHE_CHECK.json),
[transport](MINIMAX_M27_TUNING_TRANSPORT_CHECK.json),
[confirmation](MINIMAX_M27_TUNING_CONFIRM_CHECK.json),
[long generation](MINIMAX_M27_TUNING_LONG_CHECK.json),
[lifecycle](MINIMAX_M27_TUNING_LIFECYCLE_CHECK.json),
[telemetry](MINIMAX_M27_TUNING_TELEMETRY_CHECK.json),
[manifest](MINIMAX_M27_TUNING_BUILD_MANIFEST.json).
P3.5 выполнен частично: prefill admission/frequency decay не менялись.
Старый MM27-06 native-after-cache discrepancy остаётся OPEN.

### MM27-11 — 2026-10-08 — Одно ожидание доставки на тензор

**Результат:** opt-in `--pipeline-d2d-batch 1`, работает и с tensor plan, и с
lookahead. Внутри одной матрицы cache hits, доставка из кольца, защитные
байты и cache fills отправляются в один CUDA-поток. Новые cache entries
удерживаются отдельными `PlanPins` до завершения потока; все будущие hits
по-прежнему защищены на время плана. Перед перезаписью scratch остаётся
синхронизация предыдущего compute consumer. При исключении поток завершается
до снятия pins и очистки кэша. Общие helpers Step/Hy3/GLM/common не менялись.

Изменены MiniMax runtime, counters и CLI; добавлены A/B `--comparison d2d`,
матрица `--lookahead --d2d-batch` и faults после двух отправленных экспертов.
Observers читают готовые данные после tensor fence; отдельные checks выполняют
decode и faults без observer, чтобы его чтение не скрывало ошибку синхронизации.
Проверяются partial enqueue failure, cancellation, RAM pressure, worker/compute
failure, cache eviction/OOM, mixed Q4_K/Q6_K и отсутствие незавершённых copies.
При маленьком кэше защита записей может изменить admission: незавершённая
запись не может быть вытеснена. Дополнительный массив pins ограничен 256 записями.

**Измерение:** RTX 5090, 32607 МиБ, RAM 125,555 ГиБ, Windows, driver 581.80;
локальный MiniMax-M2.7 Q4_K_M, 128,841 ГиБ. Strict F32 activations/KV,
FA/graphs/MTP off, greedy, ctx2048/batch16, GPU arena cap 18 ГиБ,
2 readers, chunk 4 МиБ, router lookahead в обоих вариантах, trace off.
Три пары в порядке A/B, B/A, A/B; каждый процесс начинает с пустым GPU cache.
Входы по 52/52/343 токена, по 24 выходных токена; decode измеряет 23 шага.
Состояние file cache Windows и внешняя нагрузка не фиксировались.

| Запрос | Lookahead, ток/с | + D2D batch, ток/с | Изменение |
|---|---:|---:|---:|
| Первый короткий, пустой GPU cache | 2,90 | 3,31 | +14,24% |
| Повтор короткого | 3,94 | 4,20 | +6,61% |
| Новая тема, более длинный prompt | 3,61 | 3,98 | +10,40% |

Медиана суммарного времени трёх запросов: **85,47 → 80,24 с**,
сокращение **6,13%**. В отчёте отдельно сохранены
prefill/TTFT и диапазоны decode по всем повторам; выигрыш decode не заменяет
оценку общего времени. Во всех трёх парах совпали объём H2D/D2D, hit bytes,
cache fills, live cache и его бюджет. Здесь выигрыш не сопровождается
изменением cache residency.

Число delivery/cache host fences за decode: **37181–45360 → 4278**
(186 тензоров × 23 шага). Это счётчик явно объединённых ожиданий;
plan-completion, setup, cleanup и ring events в него не входят.
Размер колец прежний: 16 МиБ pinned + 16 МиБ GPU. Sampled global peaks
RAM **29,63 ГиБ**,
VRAM **29,23 ГиБ**;
все global95 gates прошли. Это выборочные, а не непрерывные peaks.

**Проверки:**

- 25 конфигураций, **4920 fixtures PASS**: sync, tensor/lookahead,
  readers1/2, chunks4/8/16, batch off/on; CUDA/arena, F32/mixed fixtures.
- **43 213 824 пары logits и 216 пар token IDs** совпали побитно.
  Все 6 дампов logits сохранили исторический MM27-06/MM27-09 SHA-256
  `35d2d4995390c2ac58739e343d3f05e85f412eeba2d6e917108888c96d97023e`.
- **47 lifecycle**, **55 reload**, **49 runtime** и **41 CPU tests PASS**;
  реальная нагрузка достигла **89,96% RAM / 92,25% VRAM**.
  Кэш уменьшился с **15,27 до 6,21 ГиБ**.
  Прошли cancel/recovery/unload и ctx2K/4K с повторяемым префиксом
  и 33 различающимися токенами.
  Проверки native/file/cache сравнивают logits без ослабления tolerances.
- CUDA event trace первых 4 plans: interval accounting, drain и первые 2
  vocabulary rows совпали с сохранённым reference. Интервалы CUDA events
  включают промежутки между командами; это не профиль активности kernels.

EXE, исходники/хэши, команды и raw logs сохранены в
`build-local/minimax-m2-d2d-*`; предыдущий backend и EXE —
`build-local/minimax-m2-before-d2d-01`. Архивы результатов:
[fixtures](MINIMAX_M27_D2D_FIXTURES_CHECK.json),
[A/B](MINIMAX_M27_D2D_AB_CHECK.json),
[lifecycle](MINIMAX_M27_D2D_LIFECYCLE_CHECK.json),
[reload](MINIMAX_M27_D2D_RELOAD_CHECK.json),
[runtime](MINIMAX_M27_D2D_RUNTIME_CHECK.json),
[trace](MINIMAX_M27_D2D_TRACE_CHECK.json),
[telemetry](MINIMAX_M27_D2D_TELEMETRY_CHECK.json),
[build manifest](MINIMAX_M27_D2D_BUILD_MANIFEST.json).

Defaults не изменены. P3 остаётся PARTIAL; старый MM27-06 native reload
discrepancy остаётся OPEN, причина не установлена. API, sessions и MTP
не добавлялись в этом этапе.

### MM27-10 — 2026-10-08 — Загрузка следующих матриц по известным ID экспертов

**Результат:** `--pipeline-lookahead 1` объединяет до трёх матриц текущего
MoE-блока в план чтения. В трёх парах A/B прирост медианы decode составил
**17,74–31,44%** относительно конвейера MM27-09 для одной матрицы.
Входные данные и математические kernels не менялись, все logits совпали побитно.
Это дополнительный экспериментальный режим; defaults прежние.

**Реализация:** после получения реальных router IDs scheduler просматривает
не более 32 следующих splits. В план попадают только матрицы того же блока,
backend и router tensor, в фактическом порядке исполнения, максимум три.
Для каждой сохраняется до 256 диапазонов. Перед доставкой проверяются источник
и диапазоны; изменение порядка или незавершённый план даёт ошибку.
Предсказания экспертов следующего слоя нет.

Общий транспорт `common/expert_pipeline.hpp`, Step cache и Hy3 arena не менялись.
Кольца сохраняют четыре слота: при chunk 4 МиБ это **16 МиБ pinned RAM + 16 МиБ VRAM**.
GPU cache cap — 18 ГиБ. Работники читают только пропуски кэша и пишут только
в собственное кольцо; будущий scratch scheduler не заполняется заранее.
Копирование в scratch завершается до его compute. Будущие попадания в кэш
защищены до завершения плана. Scope guard завершает операции и снимает защиту
при любом выходе scheduler, включая сбой compute с незавершённым lookahead.
Проверка такого сбоя и последующего восстановления добавлена в fixtures.

**Условия A/B:** тот же локальный Q4_K_M GGUF, RTX 5090 (32 607 МиБ VRAM),
125,555 ГиБ RAM, Windows/driver 581.80, CUDA 13.0.48 / MSVC 19.44.35222.0,
Release sm120a, dependency `86ebfef2`. Strict F32 activations/KV, FA/graphs/MTP off,
greedy, ctx 2048 / batch 16, два reader threads, chunk 4 МиБ, arena cap 18 ГиБ.
Оба варианта используют один новый executable; отличается только lookahead 0/1.
Три пары в порядке A/B, B/A, A/B; новый процесс на вариант, 24 выходных токена
на запрос, 23 измеренных decode forwards. Сборки и другие наши GPU-тесты
во время A/B не выполнялись. Состояние OS file cache не контролировалось.

В третьем запуске `tensor` на новой теме глобальное давление VRAM уменьшило
live cache budget до 17,793 ГиБ, а объём занятых записей arena к концу decode —
до 9,526 ГиБ. В третьем `lookahead` budget составлял 17,863–17,892 ГиБ,
занятые записи — 17,253–17,313 ГиБ. В первых двух парах budget был 18 ГиБ,
занятые записи — 17,433 ГиБ. Sampled global VRAM в третьей паре достигла
примерно 30 ГиБ. Лимит 95% не превышен, но фоновая нагрузка не была постоянной;
замеры нельзя считать сравнением при полностью одинаковой доступной памяти.

| Запрос | Одна матрица, ток/с | Lookahead, ток/с | Прирост медианы |
|---|---:|---:|---:|
| Первый, 52 входных токена | 2,33 | 2,82 | 20,69% |
| Повтор, 52 входных токена | 3,14 | 3,70 | 17,74% |
| Новая тема, 343 входных токена | 2,61 | 3,43 | 31,44% |

| Запрос | Медиана TTFT, одна матрица / lookahead, с | Медиана запроса, одна матрица / lookahead, с |
|---|---:|---:|
| Первый, 52 входных токена | 12,81 / 13,43 | 22,20 / 21,61 |
| Повтор, 52 входных токена | 8,87 / 8,92 | 16,12 / 15,00 |
| Новая тема, 343 входных токена | 48,10 / 47,14 | 56,94 / 54,12 |

Медиана суммы трёх запросов без загрузки модели: **94,35 → 91,69 с**,
сокращение **2,82%**. Диапазоны скоростей и hit rate сохранены в A/B
и telemetry. Защита записей на весь план может менять вытеснение; это измерение
всего режима, а не изолированной стоимости перекрытия H2D.
Устойчивого ускорения prefill не получено: TTFT первого запроса даже вырос.
В коротких ответах выигрыш decode мало влияет на полное время, поскольку
обработка входа занимает большую часть запроса.

| Режим | Максимум sampled global RAM, ГиБ (%) | Максимум sampled global VRAM, ГиБ (%) |
|---|---:|---:|
| tensor | 34,58 (27,54%) | 30,00 (94,20%) |
| lookahead | 34,39 (27,39%) | 30,00 (94,21%) |

В отдельном тесте давления памяти достигнуты **89,96% RAM / 89,95% VRAM**.
Кэш сократился,
проверки logits и восстановления прошли. Global guard 95% сохранён;
pagefile и модель не менялись. Пики измеряются выборками, не непрерывно.

**CUDA timeline:** отдельный запуск с диагностикой первых четырёх планов.
Интервалы на фактических CUDA streams: H2D **89,99 мс**,
compute **31,59 мс**, их пересечение **0,77 мс**.
Trace подтверждает перекрытие интервалов, но его малая длительность сама
по себе не объясняет весь прирост. Event intervals включают промежутки между
GPU-командами; это не профилирование активности отдельных kernels.
Трассировка была выключена в A/B; её logits совпали с сохранённым sync-эталоном.

**Проверки:** 2424 cache/transport cases: 504 sync, 6 × 152 tensor и 6 × 168
lookahead (readers 1/2, chunks 4/8/16 МиБ); 49 runtime, 55 reload, 18 tiny
lifecycle, 29 full pressure/2K/4K, 41 CPU/tokenizer regressions. Все PASS.
43 213 824 пары logits и 216 пар token IDs в A/B совпали побитно.
Шесть файлов logits имеют прежний MM27-06/MM27-09 SHA-256
`35d2d4995390c2ac58739e343d3f05e85f412eeba2d6e917108888c96d97023e`.
Численные tolerances сохранены. Context stress использует повторяющийся
префикс и разнообразный суффикс, а не длинные естественные ответы.
Причина старого MM27-06 native-after-cache reload расхождения остаётся **OPEN**.

**Счётчики:** `pipeline_plans` включает планы без пропусков кэша,
`pipeline_matrices` — число матриц, `pipeline_plan_peak` — максимум в плане,
`pipeline_lookahead_plans` — планы из нескольких матриц. Transport groups
считают только планы с чтением. В A/B lookahead каждый план содержит три матрицы;
после запроса нет queued/reader-owned/unused payload. Byte equations и
ограничения памяти повторно проверены в telemetry.

**Отчёты:** [fixtures](MINIMAX_M27_LOOKAHEAD_FIXTURES_CHECK.json),
[A/B](MINIMAX_M27_LOOKAHEAD_AB_CHECK.json), [CUDA trace](MINIMAX_M27_LOOKAHEAD_TRACE_CHECK.json),
[lifecycle](MINIMAX_M27_LOOKAHEAD_LIFECYCLE_CHECK.json), [reload](MINIMAX_M27_LOOKAHEAD_RELOAD_CHECK.json),
[runtime](MINIMAX_M27_LOOKAHEAD_RUNTIME_CHECK.json), [telemetry](MINIMAX_M27_LOOKAHEAD_TELEMETRY_CHECK.json),
[manifest](MINIMAX_M27_LOOKAHEAD_BUILD_MANIFEST.json).
Исходники, EXE и сырые выводы: `build-local/minimax-m2-lookahead-*`;
предыдущий backend и EXE сохранены в `build-local/minimax-m2-before-lookahead-01`.
Текущие hashes и команды — в manifest, воспроизведение — в backend README.

Проверенные флаги: `--gpu-cache-mib 18432 --gpu-cache-allocator arena --pipeline-readers 2 --pipeline-chunk-mib 4 --pipeline-lookahead 1`.

Следующий шаг — MM27-11: сократить число отдельных ожиданий D2D при доставке
и заполнении кэша, сохранив время жизни scratch/entries. Сравнить с MM27-10
по byte/logit parity, fault/pressure/reload, timeline и трём парам A/B.

### MM27-09 — 2026-10-08 — Асинхронное чтение текущей матрицы и завершение D2D

**Результат:** добавлен включаемый явно файловый конвейер для выбранных экспертов
текущей матрицы. В трёх парах A/B decode ускорился на 22,9–32,2%, обработка
входных запросов также ускорилась. Defaults сохранены: pipeline 0, cache 0,
allocator `cuda`, reader `file`. Проверенный экспериментальный вариант:
`--gpu-cache-mib 18432 --gpu-cache-allocator arena --pipeline-readers 2 --pipeline-chunk-mib 4`.

**Изменено:** `pipeline_state.inc`, `pipeline_runtime.inc`, `pipeline_trace.hpp`,
`sync_runtime.h/.inc`, `cache_runtime.inc`, private scheduler patch, CLI/checks
и Python drivers. `common/expert_pipeline.hpp`, `expert_file.hpp`, Step cache
и Hy3 arena переиспользованы без изменения. Модель и математические kernels не менялись.

- Четыре слота в pinned RAM и четыре в VRAM, каждый по 4/8/16 МиБ;
  один или два потока чтения. Выбранные два потока со слотами по 4 МиБ используют
  16 МиБ pinned RAM и 16 МиБ VRAM, заменяя старый staging buffer на 16 МиБ.
  Отдельные H2D/delivery streams и CUDA events защищают повторное использование
  слотов. Память выделяется при первом обращении с проверкой глобального лимита
  95% и запаса host commit. Полная копия весов в heap не создаётся.
- План содержит до 256 диапазонов текущего тензора. Читаются только отсутствующие
  в кэше данные; будущие попадания в кэш защищены от вытеснения до окончания плана.
  Если защищённые записи не позволяют уменьшить arena под давлением памяти,
  операция возвращает ошибку и дожидается завершения конвейера. Затем снимается
  защита записей и очищается кэш. Бесконечный цикл сокращения arena исключён.
- Перед вычислением и повторным использованием scratch дожидаемся завершения D2D.
  При ошибке чтения, копирования или отмене завершаем операции I/O и GPU до
  освобождения отображений, кэша и буферов. Отмена проверяется на границах transfer
  и перед H2D; текущий файловый read может завершиться до начала очистки.
  Мгновенная отмена I/O не гарантируется.
- Устранено отсутствие явного ожидания cache D2D **в обоих путях**: CUDA API
  допускает возврат host-вызова `cudaMemcpy` D2D до завершения GPU-копии;
  ggml compute stream создан с `cudaStreamNonBlocking`. Теперь cache hit/fill
  используют `cudaMemcpyAsync` и `cudaStreamSynchronize` соответствующего stream.
  Это устраняет потенциальную гонку. Причина старого MM27-06 reload FAIL
  всё ещё **OPEN**: исходный сбой не воспроизведён, его устранение не доказано.

**Условия:** RTX 5090, 32 607 МиБ VRAM, 125,555 ГиБ RAM, Windows/driver 581.80,
MSVC 19.44.35222.0 / CUDA 13.0.48, Release/Ninja sm120a, dependency `86ebfef2`.
HEAD `295cac22846e6aa2cea7343cd4b6c13ab32258a3`, dirty tree, commit не создавался.
Тот же Q4_K_M GGUF/header/template; source/binary hashes и команды — в manifest.
Strict F32/KV, FA/graphs/MTP off, greedy, ctx 2048 / batch 16, arena cap 18 ГиБ.
Три пары с порядком A/B, B/A, A/B, новый процесс на вариант; 24 выходных токена
на запрос, 23 измеренных decode forwards, reasoning/EOS входят в подсчёт.
OS file cache не контролируется; empty GPU cache не означает cold SSD.
Оба варианта имеют новое ожидание cache D2D. Это не A/B со старым binary MM27-08.

| Запрос | Sync, ток/с | Pipeline, 2 потока / слоты 4 МиБ, ток/с | Прирост медианы decode |
|---|---:|---:|---:|
| Первый, 52 входных токена | 1,82 | 2,41 | +32,20% |
| Повтор, 52 входных токена | 2,59 | 3,18 | +22,88% |
| Новая тема, 343 входных токена | 2,24 | 2,85 | +26,89% |

| Запрос | Медиана TTFT sync / pipeline, с | Медиана времени запроса sync / pipeline, с |
|---|---:|---:|
| Первый, 52 входных токена | 20,78 / 12,85 | 33,38 / 22,35 |
| Повтор, 52 входных токена | 14,44 / 8,91 | 23,32 / 16,18 |
| Новая тема, 343 входных токена | 74,50 / 46,28 | 84,71 / 54,51 |

Медиана суммы трёх запросов без загрузки модели: **141,42 → 93,04 с**,
сокращение времени **34,21%**. Диапазоны decode/load, counters и
полные timings сохранены в telemetry/AB. Pins слегка меняют eviction/hit rate;
измеряется интегрированный pipeline, не изолированная стоимость memcpy.

| Режим | Sampled global RAM, ГиБ (%) | Sampled global VRAM, ГиБ (%) | Process WS / private peaks, ГиБ |
|---|---:|---:|---:|
| sync | 30,89 (24,60%) | 29,14 (91,51%) | 2,97 / 26,22 |
| pipeline | 31,90 (25,41%) | 29,15 (91,55%) | 2,97 / 26,23 |

В отдельном тесте давления памяти: **89,91% RAM / 93,37% VRAM**,
с дополнительными 12 ГиБ GPU allocations и readonly RAM mapping.
Кэш уменьшился, logits/recovery сохранились.
Это sampled peaks, не continuous memory trace; pagefile не менялся.

**Проверки, exit 0:** 1416 cache/transport cases (504 sync и 6 × 152 pipeline),
49 runtime checks, 55 reload checks, 18 tiny lifecycle, 29 full pressure/2K/4K,
41 CPU/tokenizer regressions. Добавлены worker read/copy failures, recovery,
drain/capacity и принудительное давление VRAM при plan pins. Один ранний тест
ожидал тёплый cache после abort; исправлена предпосылка — явный rewarm перед
проверкой eviction. Численные tolerances не менялись.
Три пары A/B: 43 213 824 пары logits и 216 пар token IDs совпали побитно;
все шесть output files имеют прежний MM27-06 SHA-256
`35d2d4995390c2ac58739e343d3f05e85f412eeba2d6e917108888c96d97023e`.
Context stress использует повторяющийся префикс, разнообразный суффикс
из 33 токенов и 8 токенов продолжения; качество длинных ответов здесь не проверяется.

**CUDA trace:** отдельный запуск с `--pipeline-trace` на первых четырёх tensor
plans; H2D **22,42 мс**, ring delivery **1,46 мс**,
пересечение интервалов **0,06 мс**. Оно мало и само по себе не объясняет весь
прирост скорости. Logits совпали с untraced sync.
Это интервалы CUDA events на фактических transfer streams, не сумма CPU timers.
Trace выключен в A/B. **Перекрытие с вычислением модели не реализовано:**
текущий tensor полностью доставляется до его compute; следующий шаг — lookahead
на следующие матрицы того же MoE-блока с сохранением source/consumer lifetime.

В успешных запросах `source_bytes=file_bytes=h2d_bytes`,
`pipeline_h2d_bytes=pipeline_d2d_bytes=h2d_bytes`; unused/queued/reader-owned равны 0
после plan. При abort raw pipeline counters учитывают и abandoned uploads.
CPU read/wait/submit sums могут перекрываться, не являются GPU длительностями
или physical SSD traffic. Фиксированные ring capacities не складывать с
логическим объёмом jobs как с отдельной allocated памятью.

**Отчёты:** [fixtures](MINIMAX_M27_PIPELINE_FIXTURES_CHECK.json),
[pilot](MINIMAX_M27_PIPELINE_PILOT_CHECK.json), [A/B](MINIMAX_M27_PIPELINE_AB_CHECK.json),
[runtime](MINIMAX_M27_PIPELINE_RUNTIME_CHECK.json), [reload](MINIMAX_M27_PIPELINE_RELOAD_CHECK.json),
[tiny lifecycle](MINIMAX_M27_PIPELINE_FIXTURE_LIFECYCLE_CHECK.json),
[full lifecycle](MINIMAX_M27_PIPELINE_LIFECYCLE_CHECK.json), [CUDA trace](MINIMAX_M27_PIPELINE_TRACE_CHECK.json),
[telemetry](MINIMAX_M27_PIPELINE_TELEMETRY_CHECK.json), [manifest](MINIMAX_M27_PIPELINE_BUILD_MANIFEST.json).
Raw outputs/retained EXE: `build-local/minimax-m2-pipeline-*`.
Pilot — один проход 0/1/2 readers, 12 выходных токенов, не итоговое сравнение.
Во время части первого A/B pipeline prefill собирался только lifecycle test;
benchmark EXE не менялся. Остальные пары выполнены без сборки, диапазоны сохранены.

**Повторение**, после сборки по README, новые output directories:

```powershell
$modelPath = 'H:\models\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf'
python -X utf8 tools/check_minimax_m2_pipeline.py --out build-local/mm27-new-pipeline-fixtures
python -X utf8 tools/check_minimax_m2_cache.py --model $modelPath --out build-local/mm27-new-pipeline-ab --comparison pipeline --pipeline-readers 2 --pipeline-chunk-mib 4 --cache-mib 18432 --repeats 3 --tokens 24
python -X utf8 tools/check_minimax_m2_reload.py --out build-local/mm27-new-pipeline-reload --check-mm27-06-baseline --pipeline-readers 2 --pipeline-chunk-mib 4
python -X utf8 tools/check_minimax_m2_lifecycle.py --model $modelPath --out build-local/mm27-new-pipeline-life --cases full_pressure context2048 context4096 --gpu-cache-mib 18432 --gpu-cache-allocator arena --pipeline-readers 2 --pipeline-chunk-mib 4
```

Следующий шаг — MM27-10/P3.2: lookahead следующей матрицы, CUDA timeline
H2D/compute, затем parity/pressure/reload и новое трёхпарное A/B.

### MM27-08 — 2026-10-08 — Ограниченные mmap readers и full-model A/B

**Результат:** P3.4 выполнен частично. Реализованы три альтернативы ReadFile
без второй heap-копии экспертов: mapped→pinned, direct mapped→CUDA и
`mmap-decode` (ReadFile в prefill/первом serial decode, далее mapped→pinned).
Последний вариант ускоряет decode повторного запроса, но замедляет decode
первого запроса и запроса на новую тему. Он остаётся экспериментальным; defaults не менялись.

**Изменено:** `sync_runtime.h/.inc`, CLI/JSON в `main.cpp`, cache/lifecycle/reload
checks, Python drivers и backend README. Использован существующий
`glm5next/host_pages.hpp` без изменения GLM или других backends.
Process working-set target94% общей RAM учитывает другую нагрузку, обновляется
раз в250мс либо раньше при давлении; исходные Windows limits восстанавливаются
на release. Hard guard95% общей RAM/VRAM остаётся отдельной проверкой.
Размер transfer chunk/pinned buffer16МиБ, sync fences перед повторным использованием,
range validation, cancel между chunks. `mmap-direct` не выделяет application
staging buffer, но драйвер может копировать pageable memory внутри себя.
Ни pagefile, ни GGUF, ни численный CUDA путь не менялись.

**Условия:** Windows, RTX5090 32607МиБ/driver581.80, RAM125,555ГиБ,
MSVC19.44.35222.0/CUDA13.0.48, Release/Ninja sm120a. HEAD
`295cac22846e6aa2cea7343cd4b6c13ab32258a3`, dirty tree; commit не создавался.
Dependency `86ebfef2`, прежние header/template hashes; точные source/patch/binary
hashes — в новом build manifest. Strict F32/KV, FA/graphs/MTP/pipeline off,
greedy, ctx2048/batch16, GPU arena cap18ГиБ (17,433ГиБ полезных матриц).
Три пары A/B с порядком A/B,B/A,A/B; новый процесс для каждого варианта,
52/52/343 prompt tokens,24 generated на запрос,23 timed decode forwards.
В counts входят reasoning и возможный EOS; это не скорость только final answer.
OS file-cache state не контролируется, empty GPU cache не означает cold SSD.

| Запрос | file, ток/с | mmap-decode, ток/с | Изменение median decode |
|---|---:|---:|---:|
| Первый запрос, 52 prompt tokens | 2,05 | 1,05 | -48,84% |
| Повтор, 52 prompt tokens | 2,78 | 4,17 | 49,72% |
| Другая тема, 343 prompt tokens | 2,43 | 1,91 | -21,47% |

| Запрос | Median TTFT file / mmap-decode, с | Median request file / mmap-decode, с |
|---|---:|---:|
| Первый запрос, 52 prompt tokens | 21,57 / 21,38 | 32,90 / 42,67 |
| Повтор, 52 prompt tokens | 14,06 / 13,19 | 22,17 / 18,72 |
| Другая тема, 343 prompt tokens | 72,97 / 70,26 | 82,58 / 82,32 |

Median суммы трёх request times: **136,83 / 143,21 с** (file / mmap-decode),
без model load. Диапазоны decode, load и отдельные prefill/source/H2D timings
сохранены в telemetry report. Ускорение повторного decode не означает такого же
ускорения всего запроса. Для первого mmap-доступа source time вырос; page-fault
trace пока нет, поэтому точная причина задержки не установлена.

| Reader | Sampled global RAM, ГиБ (%) | Sampled global VRAM, ГиБ (%) | Process WS / private peaks, ГиБ |
|---|---:|---:|---:|
| file | 35,45 (28,23%) | 29,69 (93,24%) | 2,97 / 26,22 |
| mmap-decode | 84,50 (67,30%) | 29,67 (93,17%) | 57,38 / 26,22 |

Это sampled peaks, не непрерывное наблюдение. `source_bytes=file_bytes+mmap_bytes=h2d_bytes`,
`h2d_bytes+cache_hit_bytes=selected_bytes+cache_guard_bytes`; обе формулы проверены.
Эти bytes не равны физическому SSD traffic. У direct reader faults/driver staging
включены в H2D wall time, source time0; нельзя напрямую сравнивать только source_ms.

**Отбор кандидатов:** короткий pilot с двумя одинаковыми запросами давал
file1,94/2,92, mmap2,86/4,56, direct2,73/4,30 ток/с, все logits совпали.
Это один прогон без контроля порядка, отдельная ранняя сборка, не итоговый A/B.
На корпусе со сменой темы all-phase mmap дал1,41/4,52/1,30 против
file2,29/2,98/2,49 ток/с в первой паре. Второй mmap child был намеренно остановлен:
exit4294967295 в screening report — остановка эксперимента, не наблюдавшийся crash.
Screening сохранён как `NOT_SELECTED`; затем выполнены все три пары гибридного A/B.

**Проверки, exit0:**

| Проверка | Результат |
|---|---|
| Cache fixtures:4 readers×2 allocators×2 quants, faults/pressure/restoration | 504/504 PASS |
| Uncached runtime regression | 49/49 PASS |
| Reload stress mmap-decode, сохранённый EXE и historical golden hash | 55/55 PASS |
| Tiny lifecycle mmap-decode/arena256МиБ | 18/18 PASS |
| Full GGUF: pressure/cancel/recovery/unload и context2048/4096 | 29/29 PASS |
| Python/tokenizer CPU regressions | 41/41 PASS |
| Три full-model A/B пары | 43 213 824 logit pairs и216 paired token IDs: bit-exact |
| Аудит telemetry: memory95/bytes/backing/counts/historical hash | 153/153 PASS |

Все шесть A/B logit files совпали также со старым MM27-06 hash
`35d2d4995390c2ac58739e343d3f05e85f412eeba2d6e917108888c96d97023e`.
В отдельном pressure run sampled global peaks составили **89,95% RAM / 93,80% VRAM**.
Проверено вытеснение warm GPU cache при дополнительной нагрузке12ГиБ,
сохранение logits и восстановление после снятия нагрузки.
Context2K/4K — stress с repeated prefix и varied33-token suffix,8 continuation
tokens, не long-answer quality corpus. Старый native-after-cache FAIL остаётся OPEN.

**Отчёты:** [cache](MINIMAX_M27_READERS_CACHE_CHECK.json),
[runtime](MINIMAX_M27_READERS_RUNTIME_CHECK.json), [reload](MINIMAX_M27_READERS_RELOAD_CHECK.json),
[tiny lifecycle](MINIMAX_M27_READERS_FIXTURE_CHECK.json), [pilot](MINIMAX_M27_READERS_PILOT_CHECK.json),
[отклонённый mmap](MINIMAX_M27_READERS_MMAP_SCREENING.json), [A/B](MINIMAX_M27_READERS_AB_CHECK.json),
[full lifecycle](MINIMAX_M27_READERS_LIFECYCLE_CHECK.json),
[telemetry](MINIMAX_M27_READERS_TELEMETRY_CHECK.json), [manifest](MINIMAX_M27_READERS_BUILD_MANIFEST.json).
Raw outputs и сохранённые test EXE находятся в `build-local/minimax-m2-readers-*`.
Точные выполненные команды перечислены в manifest; команды повторения с новыми каталогами:

```powershell
$modelPath = 'H:\models\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf'
python -X utf8 tools/check_minimax_m2_cache.py --model $modelPath --out build-local/mm27-new-reader-ab --comparison reader --candidate-reader mmap-decode --cache-mib 18432 --repeats 3 --tokens 24
python -X utf8 tools/check_minimax_m2_reload.py --out build-local/mm27-new-reader-reload --check-mm27-06-baseline --expert-reader mmap-decode
python -X utf8 tools/check_minimax_m2_lifecycle.py --model $modelPath --out build-local/mm27-new-reader-life --cases full_pressure context2048 context4096 --gpu-cache-mib 18432 --gpu-cache-allocator arena --expert-reader mmap-decode
```

Сборка и остальные команды — в backend README. Следующий шаг: MM27-09/P3.2,
bounded async pipeline с обязательными byte/logit/lifecycle gates и timeline.

### MM27-07 — 2026-10-08 — Reload stress, обязательный replay gate и сохранение EXE

**Результат:** расширена проверка lifecycle. Первоначальная numerical discrepancy
MM27-06 остаётся **OPEN**; engine computation и tolerances не изменялись.
Benchmark EXE побитно совпадает с MM27-06, SHA-256
`42856cfe85bcba6b4251e2d6b23121828dc35c7544e688b5bb2b28de604695e4`.
Скорость в этом этапе повторно не измерялась; действуют числа MM27-06.
Те же HEAD/dirty tree, dependency, Windows/RTX5090/CUDA13.0.48; commit не создавался.

**Изменения:**

- `check_lifecycle.cpp`: `same_context_vs_initial_file` теперь входит в список
  обязательных tests и влияет на общий PASS. Раньше сравнение сохранялось как
  информационное поле. Это реальный пробел проверки; он не объясняет старые FAIL.
- `check_reload.cpp`, новый CMake target `strata-minimax-m2-reload-check`:
  свежий native oracle и дополнительный resident-weight oracle;9 циклов,
  чередующих cache off/cuda/arena. После4096-контекста меняется рабочий набор
  короткими запросами, model/cache выгружаются. Между загрузками выделяются,
  заполняются и освобождаются GPU buffers13/17/64/96МиБ; в нечётных циклах13МиБ
  остаются занятыми до конца цикла. Это вариация истории allocations,
  не доказательство покрытия всех вариантов повторного использования памяти.
- Сравниваются logits после каждого prefill batch и decode step, включая
  replay с исходным oracle. Context4096, prompt4079, batch16,8 continuation
  tokens, strict F32/KV, FA/graphs off; синтетический repeated-prefix corpus.
- `tools/check_minimax_m2_reload.py` и lifecycle driver сохраняют копию
  запускаемого `engine.exe`, backend sources, hashes, команды и отчёты.
  Один SHA не позволяет повторить старую сборку, если её EXE уже заменён.

**Проверки, exit0:**

| Проверка | Результат |
|---|---|
| Текущий instrumented tiny lifecycle до изменений | 17/17 PASS |
| Реконструированный сценарий без replay instrumentation | 16/16 arena и16/16 cuda PASS |
| Новый reload stress | **55/55 PASS**;603648 exact-required logit pairs совпали побитно |
| Дополнительный resident oracle | 16768 logits: bit-exact, внутри прежних5e-4/1e-7 tolerances |
| Native continuation после перезагрузок | Все10 files совпали со старым правильным MM27-06 continuation hash |
| Lifecycle с обязательным replay-vs-initial gate | **18/18 arena и18/18 cuda PASS** |
| Python drivers | `py_compile` PASS |

Новый stress сначала запускался напрямую (55/55), затем через сохраняющий
EXE driver (55/55); в таблице это одна проверяемая конфигурация, не110 разных cases.
Все10 continuation files имеют SHA-256
`662a8b04a40790b72cff94d4106f368fcfea7c77ec5a6816bc46380153648589`.
Synthetic GGUF SHA-256 совпал с историческим:
`0fe45556a1d92c4c8783716ae1118e2278a84fc0693b98d85f45edf03e4981e9`.
Golden-hash gate включается явно для этой зафиксированной платформы/fixture.

**Диагностика точности:** отдельный режим намеренно использовал F16, BF16 и
explicit TF32 после strict baseline. Ни один результат не совпал с сохранённым
ошибочным native output. Отчёт этого режима является `DIAGNOSTIC`, а не
проверкой допустимой точности. Он исключает лишь точное совпадение с этими
тремя численными результатами, а не все возможные проблемы precision dispatch.

**Ограничение расследования:** исходный failing EXE с SHA
`456b9d967b1468d9de32925138b7233791585364b8d4a15e46ba65a19198060b`
не был сохранён. Реконструкция без replay имеет другой hash и проходит;
это не доказательство поведения прежней сборки. Старые два FAIL не удалены,
причина не объявлена исправленной. Production benchmark binary не менялся,
поэтому полный128,8ГиБ GGUF в этом этапе повторно не запускался.

**Воспроизведение**, после сборки по backend README; output directories новые:

```powershell
python -X utf8 tools/check_minimax_m2_reload.py --out build-local/mm27-new-reload --check-mm27-06-baseline
python -X utf8 tools/check_minimax_m2_lifecycle.py --out build-local/mm27-new-reload-arena --cases fixture --gpu-cache-mib 256 --gpu-cache-allocator arena
python -X utf8 tools/check_minimax_m2_lifecycle.py --out build-local/mm27-new-reload-cuda --cases fixture --gpu-cache-mib 256 --gpu-cache-allocator cuda
```

Артефакты: [reload stress](MINIMAX_M27_RELOAD_STRESS_CHECK.json),
[arena lifecycle](MINIMAX_M27_RELOAD_ARENA_LIFECYCLE_CHECK.json),
[cuda lifecycle](MINIMAX_M27_RELOAD_CUDA_LIFECYCLE_CHECK.json),
[OPEN investigation](MINIMAX_M27_RELOAD_INVESTIGATION.json),
[build manifest](MINIMAX_M27_RELOAD_BUILD_MANIFEST.json).

**Следующий шаг:** MM27-08, P3.4 bounded RAM/read-path experiment с этими
regression gates. Defaults остаются прежними; старый reload issue открыт.

### MM27-06 / P3.3 — 2026-10-08 — Плотный GPU cache и полная проверка под давлением памяти

**Основание:** HEAD `295cac22846e6aa2cea7343cd4b6c13ab32258a3`, dirty tree,
без commit; предыдущая работа сохранена. Тот же локальный GGUF, header/template
hashes, dependency `86ebfef2`, RTX5090 32607МиБ, RAM125,555ГиБ, driver581.80,
MSVC19.44.35222.0, CUDA13.0.48, Release/Ninja, requested120/effective120a.

**Реализовано:** MiniMax `gpu_arena.hpp` переиспользует неизменённый Hy3 slot
allocator: size classes,64КиБ alignment, обычно64МиБ backing blocks, возврат
пустого блока. MiniMax ограничивает сумму всех backing allocations, включая
пустые слоты и padding, значением `--gpu-cache-mib`. Перед новым блоком учитывает
global VRAM95% с резервом256МиБ и host commit1ГиБ сверх всего блока.
`cache_runtime.inc` учитывает физический backing при бюджете и продолжает eviction,
пока освобождённые слоты действительно не вернут нужные blocks CUDA.
Добавлены telemetry и `--gpu-cache-allocator cuda|arena`, default `cuda`.
Синхронные copies, admission policy и математика модели сохранены; Step/Hy3
и остальные backend не изменялись. Cache default остаётся0.

**Проверки:**

| Проверка | Результат |
|---|---|
| Arena allocation/guards | **14/14 PASS**: alignment, unique slots, byte reuse, mixed sizes, holes/backing cap, invalid/double release, VRAM/commit refusal, release |
| F32 и mixed Q4_K/Q6_K cache, оба allocator | **112/112 PASS**:2/256МиБ caps, bytes/logits, faults, identity, unload/reload |
| Uncached runtime regression | **49/49 PASS** |
| CPU contract/tokenizer regression | **41/41 PASS** |
| Full-model A/B | **3/3 пары PASS**,6 свежих процессов; cuda/arena, arena/cuda, cuda/arena |
| Full-model pressure/cancel/unload и context2K/4K | **29/29 PASS**,8 continuation steps в каждом long-context case |
| Instrumented tiny lifecycle | **17/17 PASS**; ранние FAIL сохранены отдельно, причина OPEN |
| Telemetry audit | **156/156 PASS**: global95%, backing cap, accounting, counts, logit hashes |

Все перечисленные PASS-запуски завершились с exit0. Два более ранних tiny
запуска завершились с FAIL/exit1; они не включены в PASS totals.
На полной модели побитно сравнены **43 213 824 logits**, совпали216 пар
greedy IDs (432 generated IDs суммарно). Все6 logit files также совпали
с A/B предыдущего MM27-05, SHA-256:
`35d2d4995390c2ac58739e343d3f05e85f412eeba2d6e917108888c96d97023e`.

**Медианы трёх запусков каждого allocator, cap18432МиБ:**

| Запрос | CUDA allocations, токенов/с | Arena, токенов/с | Прирост | Hit bytes CUDA /arena |
|---|---:|---:|---:|---:|
| Пустой GPU cache,52 tokens | 1,781 | 1,991 | +11,8% | 38,62% /43,63% |
| Повторный prompt,52 tokens | 2,425 | 2,829 | +16,7% | 50,99% /58,03% |
| Новая тема,343 tokens | 2,062 | 2,401 | +16,4% | 44,27% /50,50% |

Context2048, batch/ubatch16, F32 activations/KV, FA/graphs/MTP/pipeline off,
greedy. По24 generated tokens на запрос,23 timed decode forwards; первый
токен из prefill. Clear KV перед каждым запросом, без prefix reuse. Prefill
использует cache hits без fills. Его медианы CUDA/arena:22,139/22,218с,
15,216/13,864с и81,852/73,165с соответственно. OS file cache не контролировался,
ReadFile bytes не означают физический SSD traffic. Сравнивать allocator следует
внутри этого A/B; прежние абсолютные скорости MM27-05 получены в другой серии.

**Измерение размещения:** для512 матриц (384 Q4_K и128 Q6_K), requested1464МиБ,
отдельные CUDA allocations увеличили NVML used на2048МиБ; arena — на1536МиБ,
совпавшие с её backing. **Экономия25% VRAM в этом отдельном microbenchmark.**
На полной модели максимум полезных entries вырос с **12,820 до17,433ГиБ**;
arena backing18ГиБ, разница≈0,567ГиБ — незанятые слоты/padding.
Sampled global VRAM peaks CUDA/arena:30,011/29,724ГиБ (94,248%/93,347%);
RAM36,205/35,579ГиБ, process private27,215/26,218ГиБ.
`cache_oom` arena123744 совпал с `arena_rejects`: это намеренные отказы
новым backing blocks при заполненном cap с дальнейшим reuse подходящих slots,
а не ошибки запросов или реальные CUDA allocation OOM. CUDA variant:0.

**Реальное давление:** после прогрева создан disposable GPU holder12ГиБ
и прочитано83,5ГиБ readonly GGUF mapping для нагрузки на RAM. Backing cache
сократился с15,875 до7ГиБ; при decode свободные слоты заполнялись без роста
backing. Все600192 pressure logits совпали побитно. Sampled inference peaks:
RAM≈112,93ГиБ (89,9%), VRAM≈29,896ГиБ (93,9%); реальный pressure run без
budget rejection. Holder/mapping освобождены. Эти samples не являются
непрерывной трассой пиков во время создания нагрузки. Cancel, injected
RAM/VRAM refusal, recovery и unload/reload также прошли. Context2048/4096
проверен с2031/4079 input tokens, повторным префиксом и33 varied final tokens;
это проверка поздних позиций, не качество длинных естественных ответов.

**Открытая диагностика:** в первых tiny runs с arena и cuda единственным
FAIL был `file_vs_native/long_context` после cached model unload/reload:
max_abs `0.00039499998092651367`, NMSE `3.1290744009193525e-7`.
Greedy IDs совпали, но все512 logits различались. File/cache outputs
побитно совпали с historical MM27-04, расходился последующий native reference.
Без cache новый tiny lifecycle прошёл15 checks и historical parity.
После добавления same-context file replay в harness два arena и один cuda
запуск прошли17/17 каждый, без изменения engine computation или tolerances.
Это **не установило причину первых FAIL**. Исходные FAIL и поздние PASS
сохранены вместе; issue остаётся OPEN перед выбором defaults.

**Воспроизведение:** сборка по backend README, новые output directories.

```powershell
python tools/run_minimax_m2.py --engine build-local/minimax-m2-cuda/bin/strata-minimax-m2-arena-check.exe -- build-local/mm27-new-arena-unit.json
python tools/run_minimax_m2.py --engine build-local/minimax-m2-cuda/bin/strata-minimax-m2-cache-check.exe -- build-local/mm27-new-arena-cache
python -X utf8 tools/check_minimax_m2_cache.py --model H:\models\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --out build-local/mm27-new-arena-ab --comparison allocator --cache-mib 18432 --repeats 3 --tokens 24
python -X utf8 tools/check_minimax_m2_lifecycle.py --model H:\models\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --out build-local/mm27-new-arena-lifecycle --cases full_pressure context2048 context4096 --gpu-cache-mib 18432 --gpu-cache-allocator arena
```

Одиночный benchmark использует `--gpu-cache-mib 18432 --gpu-cache-allocator arena`.
Source/binary/report hashes и выполненные команды — в manifest; raw logs/logits —
в указанных там `build-local` directories. Benchmark exe SHA-256:
`42856cfe85bcba6b4251e2d6b23121828dc35c7544e688b5bb2b28de604695e4`.

Отчёты: [arena unit](MINIMAX_M27_ARENA_UNIT_CHECK.json),
[cache](MINIMAX_M27_ARENA_CACHE_CHECK.json),
[runtime](MINIMAX_M27_ARENA_RUNTIME_CHECK.json),
[A/B](MINIMAX_M27_ARENA_AB_CHECK.json),
[full lifecycle](MINIMAX_M27_ARENA_LIFECYCLE_CHECK.json),
[tiny lifecycle](MINIMAX_M27_ARENA_FIXTURE_CHECK.json),
[OPEN reload diagnostic](MINIMAX_M27_ARENA_RELOAD_DIAGNOSTIC.json),
[telemetry](MINIMAX_M27_ARENA_TELEMETRY_CHECK.json),
[build manifest](MINIMAX_M27_ARENA_BUILD_MANIFEST.json).

**Следующий шаг:** MM27-07, локализовать tiny reload discrepancy. Managed RAM,
async pipeline, HTTP/API, sessions, fast profile и MTP остаются отдельными задачами.

### MM27-05 / P3.1 — 2026-10-08 — Синхронный GPU matrix cache

**Основание:** HEAD `295cac22846e6aa2cea7343cd4b6c13ab32258a3`, dirty tree;
предыдущие изменения MiniMax сохранены, commit не создавался. Тот же GGUF,
header/template hashes и pinned dependency `86ebfef2`. RTX5090, driver581.80,
Windows, MSVC19.44.35222.0, CUDA13.0.48, Release/Ninja,120a.

**Реализовано:**

- `cache_runtime.inc`: собственные MiniMax generation/tensor/expert identities;
  проверка name/type/shape/strides, live mapping и offset. Cache policy переиспользует
  `backends/step35/expert_cache.hpp` без изменения Step или других backend.
- `--gpu-cache-mib N`, default0; только mode2. Общий VRAM budget95% минус256МиБ;
  live eviction, частотная admission/LRU, reuse подходящих allocations. Размер
  cudaMalloc округляется до64КиБ; это requested allocation bytes, а overhead
  драйвера дополнительно учитывается через global NVML. Host commit growth
  требует не менее1ГиБ доступного commit сверх очередного allocation.
- Синхронная доставка hits через D2D; entry хранит матрицу и реальные guard bytes
  следующего эксперта, максимум512. Последний expert не читается за границу tensor.
  Fill failure очищает cache; allocation OOM оставляет корректный scratch и
  пропускает admission. Все consumers завершаются до reuse/eviction.
- Context creation освобождает cache перед KV/workspace allocation. Prefill
  использует hits без admission; первый serial decode прогревает workspace,
  затем cache заполняется. Clear KV между запросами сохраняет weights cache.
  Model unload инвалидирует entries и identities до unmap.
- `check_cache.cpp`, отдельный target, и `tools/check_minimax_m2_cache.py`:
  fixtures/faults, sequential A/B и строгие bytes/logits checks. В JSON добавлены
  hit/fill/guard/resident/limit/eviction/reuse/OOM counters.

**Проверки, exit0:**

| Проверка | Результат |
|---|---|
| Tiny F32 и mixed Q4_K/Q6_K cache | **56/56 PASS**: cold/warm bytes/logits,2/128МиБ caps, eviction/reuse, new topic, prefill hits без fills, cancel/recovery, pressure, OOM bypass, failed fill, changed identity, unload/reload/context recreation |
| Uncached runtime regression | **49/49 PASS** |
| CPU tokenizer/contract regressions | **41/41 PASS** |
| Full-model A/B | **3/3 пары PASS**,6 отдельных процессов; off/on, on/off, off/on |
| Full-model logits и IDs | **43 213 824 logits** сравнены побитно; все6 полных logit files имеют один SHA-256;216 пар greedy IDs совпали (всего432 generated IDs) |
| Telemetry/accounting audit | **83/83 PASS**; GPU-only, budget95, caps, prefill без admission, byte accounting |

В каждом процессе: короткий English prompt52 токена, точный повтор и новый
Russian/list prompt343 токена. Context2048, batch/ubatch16, F32 activations/KV,
FA/graphs/MTP/pipeline off, greedy; каждый запрос генерировал24 токена,
включая reasoning, без EOS. Decode speed считает23 forward steps; первый
output token приходит из prefill. Каждый запрос очищает KV, prefix reuse нет.
GPU cache в начале процесса пуст; ОС file cache не сбрасывался. Это сравнение
GPU cache off/on, не cold-SSD benchmark. Cap18432МиБ; полезные entries в конце
decode занимали примерно12–13ГиБ после live clamp.

**Медианы трёх запусков каждого варианта:**

| Запрос | Decode off, токенов/с | Decode on, токенов/с | Прирост | Hit по байтам decode |
|---|---:|---:|---:|---:|
| Пустой GPU cache,52 tokens | 1,220 | 1,669 | +36,8% | 38,0% |
| Повторный prompt,52 tokens | 1,328 | 2,250 | +69,4% | 50,2% |
| Новая тема,343 tokens | 1,277 | 1,938 | +51,8% | 44,8% |

| Запрос | Prefill off/on, с | TTFT off/on, с | Request off/on, с |
|---|---:|---:|---:|
| Пустой GPU cache | 24,140 /22,862 | 24,141 /22,863 | 42,165 /36,335 |
| Повторный | 19,505 /16,608 | 19,505 /16,609 | 36,835 /26,888 |
| Новая тема | 103,257 /85,021 | 103,258 /85,021 | 121,697 /96,901 |

Load1,188–2,476с, отдельно от request. Для исходного file path decode переносит
около90,9ГиБ за23 шага; cache сокращает этот объём примерно на указанный hit ratio.
`source_bytes` — completed ReadFile bytes, включая файловый кэш ОС, не SSD traffic.
Для успешного запроса проверено:
`H2D + cache_hit_bytes = selected_bytes + cache_guard_bytes`.
`cache_fill_bytes` — отдельная scratch→cache D2D передача.

**Память с cache on, sampled maxima всех трёх запусков:** VRAM **30,001ГиБ /
31,843ГиБ =94,215%**, RAM used **34,614ГиБ /125,555ГиБ =27,569%**;
process private **27,381ГиБ**, working set **2,982ГиБ**. Standby/file-cache pages
могут учитываться Windows как available; эти числа не означают отсутствие
файлового кэша в RAM. Maximum requested cache allocations **12,988ГиБ**;
reuse36 298, OOM0, CPU/full-expert fallback0. Отдельный managed RAM cache пока
не создаётся. Это sampled global peaks, не непрерывная трасса.

**Воспроизведение**, из корня репозитория после сборки по backend README:

```powershell
python tools/run_minimax_m2.py --engine build-local/minimax-m2-cuda/bin/strata-minimax-m2-cache-check.exe -- build-local/mm27-cache-new-fixture
python tools/run_minimax_m2.py --engine build-local/minimax-m2-cuda/bin/strata-minimax-m2-runtime-check.exe -- build-local/mm27-cache-new-runtime
python -X utf8 tools/check_minimax_m2_cache.py --model H:\models\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --out build-local/mm27-cache-new-ab --cache-mib 18432 --repeats 3 --tokens 24
```

Каждый output directory должен быть новым. Выполненные команды и SHA source,
dependency, binary и reports сохранены в manifest. Локальные raw logs/logits:
`build-local/minimax-m2-cache-fixture-01`, `minimax-m2-runtime-07-cache`,
`minimax-m2-cache-ab-01`. Benchmark exe SHA-256:
`25a8d267598b2ffa85594fc5d7f89ab0f87246f5d2c828438e568f5464ffbf10`.

Отчёты: [tiny cache](MINIMAX_M27_GPU_CACHE_CHECK.json),
[runtime regression](MINIMAX_M27_GPU_CACHE_RUNTIME_CHECK.json),
[полный A/B](MINIMAX_M27_GPU_CACHE_AB_CHECK.json),
[telemetry audit](MINIMAX_M27_GPU_CACHE_TELEMETRY_CHECK.json),
[build manifest](MINIMAX_M27_GPU_CACHE_BUILD_MANIFEST.json).

**Не закрыто:** cache-on full-model fault/pressure и заполненные окна2K/4K;
tiny fault tests не заменяют эти проверки. Длинные ответы/quality, server/profile,
managed RAM, pipeline и MTP не добавлялись. Cache default остаётся0;
проверенный opt-in для текущего стенда — `--gpu-cache-mib 18432`.
Следующий шаг — MM27-06, сравнение плотного allocator с текущим и cache-on pressure.

### MM27-04 / P1.4 и context stress — 2026-10-07/08 — Память, отмена и окна2K/4K

**База:** `295cac22846e6aa2cea7343cd4b6c13ab32258a3`, dirty MM27 work сохранён;
тот же GGUF/header hash и pin `86ebfef2`. RTX5090, driver581.80,
CUDA13.0.48, MSVC19.44.35222.0, Release/Ninja, effective CUDA120a.
Другие backends, веса и системные параметры памяти не изменялись.

**Изменено:**

- `sync_runtime.inc/.h`: live guard на границах copy/compute, throttling25мс.
  Budget проверяется и внутри долгого decode, а не только до/после него.
  Ошибка сохраняет причину RAM/VRAM/cancel; `runtime.hpp` передаёт её caller.
  Добавлены memory_checks/pressure_rejections и sampled RAM/VRAM/private/WS peaks.
- `sync_test.h`: native fault-injection ceilings только уменьшают реальную
  available memory; через bench/pipe не доступны. Их сбрасывают release/recovery.
- `check_lifecycle.cpp`: full-model/fixture tests, cancellation после копирования
  выбранного эксперта середины модели в prefill/decode, восстановление и reload.
  `Pressure` создаёт собственные CUDA buffers и отдельный readonly GGUF mapping;
  затрагивает страницы до90% global memory и освобождает только свои ресурсы.
- `tools/check_minimax_m2_lifecycle.py`: последовательный launcher четырёх cases,
  отдельные stdout/stderr/logits, SHA бинарника и logit files. Новый CMake target:
  `strata-minimax-m2-lifecycle-check`. README содержит команды и scope тестов.

**Результаты** — [основной отчёт](MINIMAX_M27_LIFECYCLE_CHECK.json),
[дополнительная проверка телеметрии](MINIMAX_M27_LIFECYCLE_TELEMETRY_CHECK.json):

| Case | Проверки | Контекст / prompt / batch | Результат |
|---|---:|---|---|
| Mixed tiny fixture | 15 | 4096 /4079 /16 | PASS |
| Full-model lifecycle + real pressure | 17 | 512 /31 /8; recovery input17 | PASS |
| Full-model context2048 | 4 | 2048 /2031 /16 | PASS |
| Full-model context4096 | 4 | 4096 /4079 /16 | PASS |

**40/40 PASS**, suite exit0. Во всех случаях восемь continuation steps,
в recovery/pressure comparisons — три. Long stress input: повтор одного token,
последние33 tokens различаются; это проверяет attention по разным values на
поздних позициях, но не качество естественного длинного ответа. Это также
уменьшает union выбранных экспертов по сравнению с произвольным длинным текстом.

File/native logits и greedy continuation IDs совпали побитно в обоих длинных
окнах. В основных трёх полных file/native comparisons проверены4 801 536 logits,
max_abs/NMSE0. Все clear/cancel/pressure/reload comparisons также bit-exact.
GPU audit: CPU compute/full-expert fallback0. Memory checks внутри long runs:
file2048 —4627, file4096 —8541, включая явные проверки на границах decode.

**Отмена и budget failure:** полные вызовы prefill и decode прерваны после
selected-copy в блоке31. Принудительный RAM limit остановился на тех же
2 123 735 552 H2D bytes, что prefill cancellation; VRAM limit — на тех же
14 850 220 544 bytes, что decode cancellation (включая завершённый prefill).
Это подтверждает остановку доставки до конца текущего decode. После каждого
случая fresh request восстановил logits побитно. После unload registry пуст,
pinned buffer освобождён; повторная загрузка дала прежние logits.
Тест устанавливает native cancellation flag; доставку клавиатурного Ctrl+C ОС
этот harness не проверяет.

**Реальная нагрузка:** отдельный readonly mapping затронул85,5ГиБ модельного
файла, GPU holder выделил16,75ГиБ. Непосредственно перед вызовом GPU/RAM usage
был близок к90%; во время inference sampled peaks составили:

- RAM used **112,797ГиБ /125,555ГиБ =89,839%**;
- VRAM used **28,540ГиБ /31,843ГиБ =89,628%**;
- process private **24,494ГиБ**, working set **88,475ГиБ**;
- pressure rejections0, source_bytes=H2D bytes,600 192 logits bit-exact.

После освобождения holder и после reload проверка снова PASS. Это реальная
нагрузка с запасом до95%; crossing95% branch проверен отдельно injected ceilings,
без намеренного превышения лимита пользователя. Пиковые значения — sampled,
не continuous; текущий guard не прерывает уже исполняемый kernel или native read.

**Времена long stress** (file mode2, strict F32 activations/KV, FA/graphs/cache/
pipeline/MTP off; измерения correctness, один прогон каждого режима):

| Context / input | File prefill, с | File decode, ток/с | Native prefill, с | Native decode, ток/с |
|---|---:|---:|---:|---:|
| 2048 /2031 | 105,055 | 1,285 | 75,302 | 1,803 |
| 4096 /4079 | 197,578 | 1,269 | 135,515 | 1,556 |

Decode содержит7 forwards после первого logit. Повторяющийся prefix и warm/cold
Windows cache делают эти числа несопоставимыми с MM27-03 как speed A/B.
Ускорение на этом этапе не заявляется. Source API bytes не равны physical SSD
reads; чтение pressure mapping не включено в экспертные H2D/source counters.

**Регрессии:** [runtime49/49 PASS](MINIMAX_M27_LIVE_GUARD_RUNTIME_CHECK.json),
[graph217/217 PASS](MINIMAX_M27_LIVE_GUARD_GRAPH_CHECK.json),
[pipe error→fresh PASS](MINIMAX_M27_LIVE_GUARD_PIPE_CHECK.json), CPU41/41 PASS;
все exit0. CUDA numerical kernels не менялись, предыдущие83 checks — MM27-03.

**Команды** после сборки target из README:

```powershell
python tools/check_minimax_m2_lifecycle.py --model H:\models\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --out build-local/minimax-m2-lifecycle-suite-01
python tools/run_minimax_m2.py --engine build-local/minimax-m2-cuda/bin/strata-minimax-m2-runtime-check.exe -- build-local/minimax-m2-runtime-06-live-guard
python tools/check_minimax_m2_runtime.py --model H:\models\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --out build-local/minimax-m2-pipe-02-live-guard --pipe-only
python tools/run_minimax_m2.py --engine build-local/minimax-m2-cuda/bin/strata-minimax-m2-graph-check.exe -- build-local/minimax-m2-graph-06-live-guard
python -X utf8 -m unittest tools.test_minimax_m2_gguf tools.test_minimax_m2_tokenizer tools.test_mimo2_tokenizer tools.test_hy3_tokenizer tools.test_step35_tokenizer tools.test_glm5next_tokenizer
```

Все каталоги уже существуют; для повторения задать новые. Raw logs/logits
остались в указанных build-local directories. Ранний fixture-01 выполнил15 cases,
но summary stdout показал0 из-за инвалидированного reference на поле
`ordered_json`; harness исправлен до финального suite и теперь проверяет
ожидаемое число cases. Tolerances/вычислительный граф не менялись.

Source/binary/generated/report SHA: [manifest](MINIMAX_M27_LIFECYCLE_BUILD_MANIFEST.json).
**Следующий шаг:** MM27-05, bounded GPU cache с off/on bytes/logits parity;
RAM working set и async pipeline — после него. API и быстрый профиль пока не подключены.


### MM27-03 / P1.1–P1.3, часть P1.4 — 2026-10-07 — Синхронный runtime и полный GGUF

**База Strata:** `295cac22846e6aa2cea7343cd4b6c13ab32258a3`, предыдущие MM27
изменения и этот этап в рабочем дереве. Candidate/архив, loader, модель и header
hash прежние. RTX5090 32 607 МиБ, driver581.80, CUDA13.0.48, MSVC19.44.35222.0,
Release/Ninja, requested120/effective120a. Другие backends не изменялись.

**Реализация:**

- `contract.hpp`: native admission до выделения весов, topology/types/ranges,
  whole-projection Q/K norms и отдельный opt-in для synthetic fixture.
- `RuntimePatches.cmake`, `sync_runtime.*`, `gpu_only_audit.inc`: отдельные
  generated scheduler/loader/mmap sources с SHA guards. Demand mapping, Windows
  file reads через общий `expert_file.hpp`, один pinned buffer16МиБ, fences перед
  reuse, cancellation, GPU-only audit и запрет full-expert fallback.
- `runtime.hpp`, `main.cpp`: фиксированные веса/KV на GPU, greedy generation,
  benchmark JSON и experimental JSONL pipe с fresh KV на каждом запросе.
  EOS/PAD200020 — stop; native FIM/reponame aliases не останавливают этот loop.
  Никаких API/profile/cache/pipeline/MTP/prefix reuse пока нет.
- NVML global VRAM и GlobalMemoryStatusEx RAM: admission и повторные checks95%
  с учётом чужих процессов. На старте резервируется workspace, staging ограничен.
  Это проверка бюджета, не жёсткий системный лимит для внешних приложений.
- `tools/run_minimax_m2.py`: CUDA DLL PATH и strict environment до старта процесса.
  `tools/check_minimax_m2_runtime.py`: воспроизводимый full-model corpus/pipe check.
  Инструкции: [backend README](../../backends/minimax_m2/README.md).

**Обнаружено и исправлено:** установка environment через `_putenv_s` внутри
`main()` на этом ПК была недостаточна: расширенный tiny test дал12 cross-batch
FAIL, max_abs до0,000645. Отключение FA само по себе не устранило FAIL. Передача
настроек из родительского процесса устранила все расхождения в пределах исходного
порога; runtime теперь требует эти значения до CUDA initialization. Конкретный
момент раннего чтения настроек библиотеками отдельно не локализован.
[Диагностика FAIL](MINIMAX_M27_RUNTIME_ENV_DIAGNOSTIC.json) сохранена; tolerances
не расширялись. Ранние full smoke01/02 с in-process settings не считаются строгим
numerical baseline и не используются в таблице ниже. Final baseline: FA off,
F32 activations/KV, graphs/reuse/cache/pipeline/MTP off.

**Проверки:**

- [Tiny runtime](MINIMAX_M27_RUNTIME_CHECK.json): **49/49 PASS**, exit0.
  F32 и mixed Q4_K/Q6_K, resident/native/file,65 teacher-forced tokens,
  batch1/8/16, повтор/clear, cancel→fresh, unregistered file→reload recovery,
  truncated range rejection, fixture admission rejection, mapping release,
  over-budget reservation rejection. Resident/native/file logits побитно совпали
  при одинаковом batch; cross-batch max_abs≤2,980233e-7. Побайтно проверены
  390 445 056 скопированных байтов в основных file runs.
- [Environment guard](MINIMAX_M27_ENVIRONMENT_CHECK.json): PASS, ожидаемый exit2
  без унаследованных настроек, до загрузки модели.
- [Полный GGUF](MINIMAX_M27_FULL_MODEL_CHECK.json): **5/5 PASS**, exit0.
  English/Russian/Chinese/code/числа; prompts44/46/46/49/190 токенов.
  **8 002 560 logits побитно совпали**, max_abs/NMSE0; совпали40 greedy tokens.
  Graph выбранного pin и исходные Q4_K/Q6_K веса одинаковы в обеих доставках.
  Все model operations на GPU; rejected CPU nodes/full copies0.
- [Native pipe](MINIMAX_M27_PIPE_CHECK.json): PASS, exit0. В одном загруженном
  процессе valid→invalid token→valid; порядок событий ожидаемый, оба fresh
  запроса дали одинаковые2 tokens, GPU audit PASS и source_bytes=H2D bytes.
- [Graph regression](MINIMAX_M27_RUNTIME_GRAPH_REGRESSION.json): **217/217 PASS**,
  exit0 после подключения scheduler hooks; CPU/dequant reference, scalar
  components и FA comparison сохранили прежние gates.
- [Kernel regression](MINIMAX_M27_RUNTIME_KERNELS_REGRESSION.json): **83/83 PASS**,
  exit0 в strict F32 configuration после новой сборки.
- CPU regression: **41/41 PASS**, exit0 —17 MiniMax и24 существующих tokenizer tests.

**Измерения полного GGUF**, file mode2, context512/batch=ubatch8, F32 KV,
FA off, greedy,8 generated tokens на запрос. Decode timing содержит7 forwards:
первый token получается из prefill. Reasoning включён в счётчик; за8 токенов
модель обычно только начинает рассуждение, это не оценка качества готового ответа.

| Prompt | Tokens | Load, с | Prefill, с | TTFT, с | Decode, ток/с |
|---|---:|---:|---:|---:|---:|
| English | 46 | 2,035 | 25,123 | 25,124 | 1,424 |
| Russian | 46 | 1,211 | 22,108 | 22,109 | 1,323 |
| Chinese | 44 | 1,171 | 19,250 | 19,251 | 1,421 |
| Code | 49 | 1,157 | 22,422 | 22,423 | 1,377 |
| Numeric list | 190 | 1,244 | 72,681 | 72,682 | 1,403 |

TTFT указан без загрузки. Порядок file→native, один fresh process на case/mode;
состояние Windows cache не сбрасывалось. Это короткие correctness measurements,
не controlled cold/warm A/B и не основание выбирать быстрый default. Reference
native mmap mode1 показал1,796–2,026 ток/с, но его working set гораздо больше;
разницу нельзя приписывать только алгоритму копирования без повторных A/B.

Memory samples на границах decode: file process working set≤2,966ГиБ,
private commit≤7,451ГиБ, global RAM used≤46,787ГиБ, global VRAM used≤11,650ГиБ.
Native reference: working set≤89,409ГиБ, global RAM used≤114,448ГиБ,
global VRAM used≤11,654ГиБ. Это не continuous peaks; RAM used=total−available,
VRAM включает фоновые приложения. Лимит95% не превышен в отсчётах; mmap138ГБ
не считается RAM allocation. До заполнения оставшихся RAM/VRAM потребуется P3.

Все35 file decode forwards: read15,252с, H2D6,167с, GPU compute3,588с суммарно.
Передача около3,95ГиБ/token плюс scheduler padding остаётся основной стоимостью.
`source_bytes` — успешно завершённые file API reads, включая Windows cache hits;
**physical SSD bytes не измерены**. Adaptive cache и overlap здесь ещё отсутствуют.

**Воспроизведение** из корня после сборки по backend README:

```powershell
python tools/run_minimax_m2.py --engine build-local/minimax-m2-cuda/bin/strata-minimax-m2-runtime-check.exe -- build-local/mm27-new-runtime
python tools/check_minimax_m2_runtime.py --model H:\models\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --out build-local/mm27-new-full --tokens 8
python tools/check_minimax_m2_runtime.py --model H:\models\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --out build-local/mm27-new-pipe --pipe-only
python -X utf8 -m unittest tools.test_minimax_m2_gguf tools.test_minimax_m2_tokenizer tools.test_mimo2_tokenizer tools.test_hy3_tokenizer tools.test_step35_tokenizer tools.test_glm5next_tokenizer
```

Исполненные каталоги: `build-local/minimax-m2-runtime-05`,
`build-local/minimax-m2-full-03-strict`, `build-local/minimax-m2-pipe-01`; stdout/stderr и F32 logits сохранены
рядом. Новый запуск требует новый output directory. Logit/report SHA каждого
полного прогона записан в aggregate JSON. Сборка и SHA binary/source/generated
patch/report files: [runtime manifest](MINIMAX_M27_RUNTIME_BUILD_MANIFEST.json).
Исполненные дополнительные команды:

```powershell
python tools/run_minimax_m2.py --engine build-local/minimax-m2-cuda/bin/strata-minimax-m2-graph-check.exe -- build-local/minimax-m2-graph-05-runtime
python tools/run_minimax_m2.py --engine build-local/minimax-m2-cuda/bin/strata-minimax-m2-kernels-check.exe -- --output build-local/minimax-m2-kernels-05-runtime.json
```


**Не закрыто:** contexts2048/4096, full-model pressure/cancel во время вычислений,
длительная генерация/память, sessions/API/profile, fast quant precision policy,
cache/pipeline и controlled speed A/B. P1 помечен PARTIAL по этим причинам.
**Следующий шаг:** MM27-04 — расширить P1 проверки, затем bounded expert cache.


### MM27-02 / P0.4–P0.5 — 2026-10-07 — CUDA kernels и native graph

**База Strata:** `295cac22846e6aa2cea7343cd4b6c13ab32258a3`, изменения MM27-01
и MM27-02 в рабочем дереве. Основной GGUF и другие backends не изменялись.

**Реализация:**

- `STRATA_MM27_CUDA`, отдельный `build-local/minimax-m2-cuda`, CUDA13.0.48,
  MSVC19.44.35222.0, Ninja/Release. Requested architecture120; ggml-cuda выбрал120a.
- `StrictF32.cmake` и `RoutedStrides.cmake` перенесены в отдельный backend
  с SHA checks исходников. Generated files не меняют архив, `_deps` или другие builds.
- `check_kernels.cpp`: F32/Q4_K/Q6_K, CPU dequant + scalar double matmul,
  GET_ROWS, MUL_MAT и MUL_MAT_ID; 256 experts/top8, compact/padded tensors,
  boundary/repeated experts, batch1/2/8/17, проверка неизменности inputs.
- `synthetic_minimax_m2.hpp`, `check_graph.cpp`:3 MoE layers, hidden256/FFN512,
  experts16/top8, vocab64. Настоящая attention geometry48Q/8KV, head128,
  flattened Q/K norms6144/1024 и partial NeoX RoPE64. Варианты F32 и Q4_K/Q6_K/F32;
  CPU reference хранит **те же деквантованные** quant weights, не исходные float weights.
- `QuantF32.cmake`: opt-in `STRATA_MM27_QUANT_F32=1` для Q4_K/Q6_K — GPU dequant
  и cuBLAS F32 без дополнительного Q8-квантования входа. Guard охватывает fusion,
  обычный/routed dispatch и sync planner. При unset/0 старый быстрый путь сохранён.
  CUDA Graphs пока выключены и не валидированы; изменение planner не заменяет их тест.
- Manifest содержит source/archive/loader/generated SHA, binaries, compiler, flags
  и фактическую backend architecture. Source hash refresh привязан к CMake dependencies.

**Результаты на RTX5090, driver581.80, Windows:**

| Проверка | Результат |
|---|---|
| CUDA kernels, обычные quant activations | **83/83 PASS** |
| CUDA kernels, F32 activations | **83/83 PASS** |
| Native graph, F32 activations | **217/217 PASS**, без CPU tensor math в GPU runs |
| Native graph, обычные quant activations | **208/217 PASS**,9 logit FAIL; отдельный diagnostic report |
| CPU CMake regression | Сборка oracles и3/3 CTest PASS |

Graph checks включают scalar Q/K RMSNorm и отрицательный контроль per-head norm,
RoPE64 с неизменной второй половиной головы, causal masks/softmax, sigmoid router,
точные biased top8 IDs и нормированные **unbiased** weights, оба residuals.
Проверены128 teacher-forced tokens, ubatch8/32/64, serial/prefill, FA off/on,
первый output,12 generated tokens, append, snapshot/restore и clear/replay.
Это маленький синтетический словарь/контекст; качество настоящей модели не оценивалось.

**Погрешности и ограничение быстрого пути:**

- Mixed GPU vs деквантованный CPU/F32: max abs **6,2585e-7**, NMSE **5,0454e-13**
  с F32 активациями. При normal MMVQ/MMQ: **0,0147121 / 2,93987e-4**.
- Общий mixed logit gate остался abs≤0,002 и NMSE≤1e-5; допуски не расширялись,
  чтобы засчитать быстрый путь. Его FAIL сохранён и блокирует такой correctness default.
  Greedy fixture совпал, но совпадение токенов не отменяет расхождение логитов.
- FA приводит KV к F16; mixed FA vs off max abs **0,000344321**, NMSE **1,83437e-7**,
  в отдельном допуске abs≤0,001/NMSE≤1e-6.
- Для нормированных Q/K при pos127 RoPE max abs **5,49555e-5**.
  Первоначальный component abs limit4e-5 был слишком узким для F32 powf/sincos
  при unit-RMS входах. Финальный документированный component limit1e-4,
  NMSE≤1e-10; это не изменение logit gate. Сам RMSNorm max abs **4,76837e-7**.
- Первые residual comparisons читали operand storage после его повторного использования
  allocator и ошибочно давали FAIL. Исправлена только instrumentation: operands
  сохраняются при вычислении producer. Реальный model graph не менялся.
- Все217 strict cases и83+83 kernel cases прошли на финальной версии checks.
  Первый diagnostic прогон и промежуточные логи оставлены в `build-local`.

**Не измерялось:** скорость/TTFT/пиковая память полной модели, изменение качества
реальных ответов, выгода от pipeline/MTP. F32-control нужен для correctness;
он может быть медленнее и требовать больше workspace. Его стоимость ещё не измерена.

**Отчёты:**

- [CUDA kernels, fast](MINIMAX_M27_CUDA_KERNELS_FAST_CHECK.json)
- [CUDA kernels, F32](MINIMAX_M27_CUDA_KERNELS_F32_CHECK.json)
- [Native graph, F32 — PASS](MINIMAX_M27_CUDA_GRAPH_F32_CHECK.json)
- [Native graph, fast — diagnostic FAIL](MINIMAX_M27_CUDA_GRAPH_FAST_DIAGNOSTIC.json)
- [CUDA build manifest](MINIMAX_M27_CUDA_BUILD_MANIFEST.json)

**Воспроизведение**, Developer PowerShell с MSVC x64:

```powershell
$cudaRoot = "$PWD/build-local/cuda-13.0"
$env:PATH = "$cudaRoot/bin;$cudaRoot/bin/x64;$env:PATH"
cmake -S backends/minimax_m2 -B build-local/minimax-m2-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_MM27_CUDA=ON -DSTRATA_MM27_STRICT_F32=ON -DSTRATA_MM27_ROUTED_STRIDES=ON -DSTRATA_MM27_QUANT_F32_CONTROL=ON -DCMAKE_CUDA_ARCHITECTURES=120 "-DCMAKE_CUDA_COMPILER=$cudaRoot/bin/nvcc.exe" "-DSTRATA_MM27_ARCHIVE=$PWD/build-local/llama-glm-86ebfef2.tar.gz"
cmake --build build-local/minimax-m2-cuda --target strata-minimax-m2-kernels-check strata-minimax-m2-graph-check -j 6
$env:NVIDIA_TF32_OVERRIDE = '0'
$env:GGML_CUDA_CUBLAS_COMPUTE_TYPE = 'f32'
$env:GGML_CUDA_DISABLE_GRAPHS = '1'
$env:LLAMA_GRAPH_REUSE_DISABLE = '1'
$env:STRATA_MM27_QUANT_F32 = '1'
$runDirectory = "build-local/minimax-m2-graph-$([guid]::NewGuid().ToString('N'))"
build-local/minimax-m2-cuda/bin/strata-minimax-m2-kernels-check.exe --output build-local/minimax-m2-kernels-recheck.json
build-local/minimax-m2-cuda/bin/strata-minimax-m2-graph-check.exe $runDirectory
```

`$runDirectory` обязан быть новым. CUDA bin/x64 в PATH нужен для cuBLAS DLL.
Для воспроизведения fast diagnostic задать `STRATA_MM27_QUANT_F32=0` и новый
run directory: на этом стенде graph-check возвращает exit1 с9 mixed logit FAIL.
Kernel-check в этом режиме возвращает exit0; разные допуски/операции указаны в reports.
Локальные build helpers: `build-local/build-minimax-m2-cuda.bat` (первичная сборка)
и `build-local/build-minimax-m2-checks.bat` (оба numerical targets).

**Следующий шаг:** MM27-03 / P1 — синхронная доставка экспертов и full-model baseline.


### MM27-01 / P0.1–P0.3 — 2026-10-07 — Contract и CPU oracles

**База Strata:** `295cac22846e6aa2cea7343cd4b6c13ab32258a3`; результат находится
в рабочем дереве. До этапа незакоммиченными были только эти MiniMax docs.
Изменение общего кода — добавлена ветка `minimax-m2` в `tools/strata_tokenizer.py`;
существующие pre-tokenizer branches сохранены, их CPU regressions прошли.

**Создано:**

- `tools/inspect_minimax_m2_gguf.py`, `minimax_m2_loader_contract.py`,
  `test_minimax_m2_gguf.py`: строгая topology/type/range admission и negative fixtures.
- `backends/minimax_m2`: отдельный CMake, no-allocation loader, vocabulary-only
  tokenizer, native Jinja renderer и build manifest. Private `_deps`, pinned archive,
  loader hash check, patches=none; CPU-only, без dependency от существующих build dirs.
- `tools/check_minimax_m2_oracles.py`, `minimax_m2_template.py`,
  `test_minimax_m2_tokenizer.py`: сравнение C++/Python, text/function normalization.
  Шаблон читается непосредственно из GGUF, его bytes не менялись.

**Проверено на локальном GGUF, Windows, MSVC19.44.35222.0, Release/Ninja:**

| Проверка | Результат |
|---|---|
| Inspector/contract | PASS:809 tensors,138334096384 payload bytes; header hash совпал с PREP-01 |
| Реальный loader | PASS:809 names/shapes/types/bytes и62 layer geometries совпали; allocated weight bytes=0 |
| `load_mtp=false/true` | Оба варианта:62 main/all blocks,0 NextN; флаг не создаёт draft weights |
| Tokenizer | **1316/1316 PASS**, точные IDs и decoded UTF-8 bytes, обе комбинации parse_special/add_special |
| Template | **103/103 PASS**:96 Python/native renders+prompt IDs и7 независимых semantic checks |
| CPU unittest | **41/41 PASS**:17 новых и24 существующих tokenizer regression tests |
| CTest | **3/3 PASS**, version/provenance smoke |

Генерация, logits, CUDA kernels, скорость и пиковая память inference в этом этапе
**не проверялись**. Registration сравнивает только headers, без чтения payload;
payload bytes в отчёте — логический размер весов, не выделенная RAM.

**Уточнения контракта:**

- Q/K norm shapes проверяются как полные проекции6144/1024; per-head128 rejected.
  Все62 блока — MoE. Shared/dense/fused-QKV/NextN extras отклоняются.
- Проверены mixed Q4_K/Q6_K, F32 router/norms, `.bias`, expert axes, row geometry,
  duplicate/missing/extra, overlap/truncation, offsets>4ГиБ и output/input alias protection.
- Native BOS200034, EOS200020, `add_bos=false`, `add_eos=false` даже при
  `add_special=true`. Template сам печатает BOS. Native EOG также содержит
  `<fim_pad>` и `<reponame>`; это пока не утверждённая serving stop policy.
- MiniMax pre-tokenizer взят из **активного** выражения pinned llama-vocab.cpp,
  включая ASCII case boundaries и поведение combining marks. Qwen2/Qwen35 не подходят.
- OpenAI nested function calls приводятся к flat name/arguments, JSON-string arguments
  разбираются в object. Проверены Unicode/nested values, несколько calls/results,
  grouping tool role и reasoning до/после последнего user. Helper не проверяет API
  call ID correlation и не парсит сгенерированные tool calls — это P2.
- `enable_thinking=false/true` даёт тот же `<think>` generation prefix.
  No-thinking режим не реализован. Media parts, orphan tool response и system message
  после первого отвергаются helper, чтобы template не потерял их молча.

**Сохранённые отчёты:**

- [Inventory](MINIMAX_M27_INSPECTION.json)
- [Loader](MINIMAX_M27_LOADER_CHECK.json)
- [Tokenizer](MINIMAX_M27_TOKENIZER_CHECK.json)
- [Template](MINIMAX_M27_TEMPLATE_CHECK.json)
- [Build manifest](MINIMAX_M27_BUILD_MANIFEST.json)

**Воспроизведение:** команды CMake выполнять из Developer PowerShell/Command Prompt
с MSVC x64. Архив уже находится в `build-local`; новые загрузки не требуются.

```powershell
cmake -S backends/minimax_m2 -B build-local/minimax-m2-oracles -G Ninja -DCMAKE_BUILD_TYPE=Release "-DSTRATA_MM27_ARCHIVE=C:/work/git/my-repos/Strata/build-local/llama-glm-86ebfef2.tar.gz"
cmake --build build-local/minimax-m2-oracles --target strata-minimax-m2-loader strata-minimax-m2-tokenizer strata-minimax-m2-template -j 6
ctest --test-dir build-local/minimax-m2-oracles --output-on-failure
python -X utf8 -m unittest tools.test_minimax_m2_gguf tools.test_minimax_m2_tokenizer tools.test_mimo2_tokenizer tools.test_hy3_tokenizer tools.test_step35_tokenizer tools.test_glm5next_tokenizer
$modelPath = 'H:\models\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf'
python -X utf8 tools/inspect_minimax_m2_gguf.py --gguf $modelPath --output docs/minimax-m2.7/MINIMAX_M27_INSPECTION.json
foreach ($checkKind in @('loader', 'tokenizer', 'template')) {
    $reportPath = "docs/minimax-m2.7/MINIMAX_M27_$($checkKind.ToUpper())_CHECK.json"
    python -X utf8 tools/check_minimax_m2_oracles.py --gguf $modelPath --bin-dir build-local/minimax-m2-oracles/bin --kind $checkKind --output $reportPath
    if ($LASTEXITCODE -ne 0) { throw "MiniMax $checkKind failed" }
}
```

Локальный helper вызова vcvars64 и этих CMake команд: `build-local/build-minimax-m2-oracles.bat`.
Build logs: `build-local/minimax-m2-configure.log`, `build-local/minimax-m2-build.log`.
Сохранённый manifest относится к этой CPU сборке; после изменений его нужно обновить.

**Следующий шаг:** MM27-02, CUDA numerical fixtures P0.4/P0.5.


### PREP-01 — 2026-10-07 — Инспекция локального GGUF и план

**Действия:** прочитаны инструкции репозитория, образцы GLM, актуальный статус
MiMo, локальный reader/loader/tokenizer и общий graph code. Проверены header,
ranges, hashes dependency, available RAM/VRAM. Изучены первоисточники:
публикация одноимённого GGUF, config/generation config MiniMaxAI и upstream loader.

**Результат:** подтверждены состав, архитектура и отсутствие MTP;
размеры RAM/VRAM и этапы интеграции записаны в плане. Payload модели не читался
и не хэшировался. GPU kernels/inference не запускались, новые веса не скачивались.

**Воспроизведение инспекции**, PowerShell из корня репозитория:

```powershell
@'
from pathlib import Path
from collections import Counter
import hashlib
import re
from tools.gguf_reader import GGUFFile

p = Path(r'H:\models\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf')
g = GGUFFile(p)
assert g.version == 3 and g.alignment == 32
assert g.metadata['general.architecture'] == 'minimax-m2'
assert g.metadata['minimax-m2.block_count'] == 62
assert len(g.metadata) == 43
assert not any('nextn' in k or 'mtp' in k for k in g.metadata)
assert len(g.tensors) == len({t.name for t in g.tensors}) == 809
size = p.stat().st_size
end = 0
for t in sorted(g.tensors, key=lambda t: t.offset):
    n = t.expected_bytes()
    assert n is not None and n > 0, t.name
    assert t.offset % g.alignment == 0, t.name
    assert t.offset >= end, t.name
    assert g.data_start + t.offset + n <= size, t.name
    end = t.offset + n
assert g.data_start + end == size == 138342384352
assert (g.header_end, g.data_start) == (8287960, 8287968)
assert not any('nextn' in t.name or 'mtp' in t.name for t in g.tensors)
layers = {int(m[1]) for t in g.tensors if (m := re.match(r'blk\.(\d+)\.', t.name))}
assert layers == set(range(62))
by_name = {t.name: t for t in g.tensors}
for i in range(62):
    assert by_name[f'blk.{i}.attn_q_norm.weight'].shape == [6144]
    assert by_name[f'blk.{i}.attn_k_norm.weight'].shape == [1024]
routed = [t for t in g.tensors if '_exps.weight' in t.name]
routed_bytes = sum(t.expected_bytes() for t in routed)
total_bytes = sum(t.expected_bytes() for t in g.tensors)
assert len(routed) == 186 and routed_bytes == 135725580288
assert total_bytes == 138334096384
assert total_bytes - routed_bytes == 2608516096
types = Counter(t.type_name for t in g.tensors)
assert dict(types) == dict(Q4_K=375, Q6_K=61, F32=373)
assert sum(t.expected_bytes() // 256 * 8 for t in routed) == 4241424384
with p.open('rb') as stream:
    header_sha = hashlib.sha256(stream.read(g.header_end)).hexdigest()
assert header_sha == '9a011b2ad6fb20db3bb0cdfe9af08665939ee1f0cb0609ffe2bfee7f31c825db'
template_sha = hashlib.sha256(g.metadata['tokenizer.chat_template'].encode('utf-8')).hexdigest()
assert template_sha == '893d908f7b5cc65fdde270dcae5ea1a99647c6a7ce572ae874a57b7160069566'
print('PASS header/ranges/norm shapes; payload bytes:', total_bytes)
print('Routed/other bytes:', routed_bytes, total_bytes-routed_bytes)
print('Types:', dict(types))
print('Header SHA-256:', header_sha)
print('Template SHA-256:', template_sha)
'@ | python -X utf8 -
```

Это проверка зафиксированного файла, не универсальный contract и не
числовая проверка Q/K norm. Она не подтверждает содержимое матриц или inference.

**Проверка документации:** приведённый Python-код выполнен из этого Markdown,
exit0, `PASS header/ranges/norm shapes`; размеры, типы и оба хэша совпали.
Локальные ссылки, code fences и отсутствие trailing whitespace — PASS.
Арифметика payload, H2D, KV и примера GPU cache budget — PASS.

## Правила обновления

- Новые записи `MM27-01`, `MM27-02` и далее: дата, commit/dirty state,
  patch hashes, файлы, точные build/test команды, exit codes и PASS/FAIL/SKIPPED.
- Ссылаться на существующие JSON/logs; будущие пути явно отмечать.
  Результаты базового MiniMax или других моделей не выдавать за этот finetune.
- Фиксировать model/header/template hashes, dependency SHA, GPU/driver/compiler,
  context/KV, FA/graphs, batch/ubatch, sampler/seed, cache/readers/chunks.
- Load/prefill/TTFT/decode/request latency измерять отдельно, generated counts
  и учёт reasoning/EOS указывать явно; cold/warm/prefix-hit не смешивать.
- RAM: global used/available, process commit/working set и pinned buffers.
  VRAM: global used/total, KV/cache/workspace и peaks. mmap size не считать RSS.
- Для оптимизаций сохранять correctness и минимум три повторения A/B;
  пройденный короткий smoke не закрывает context/pressure/session проверки.
- MTP: раздельно доказать наличие совместимых weights, graph correctness,
  acceptance и полезный speedup. Если чего-то нет — оставить capability off.

## Шаблон следующей записи

```text
### MM27-NN / Pn.m — YYYY-MM-DD HH:MM, Asia/Yekaterinburg — Название
Основание: commit/dirty state, dependency/patch SHA, model/template hashes.
Изменено: файлы и поведение.
Команды: точные build/test/benchmark команды.
Проверки: PASS/FAIL/SKIPPED, exit codes, fixtures/corpus.
Результаты: logits/token parity, TTFT/tokens/s, память, cache/H2D/SSD, повторы.
Артефакты: ссылки на созданные отчёты и логи.
Не закрыто: ошибки, отсутствующие веса, непроверенные режимы.
Следующий шаг: одна конкретная задача с критерием приёмки.
```
