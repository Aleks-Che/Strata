# Статус внедрения MiniMax-M2.7

Обновлено: **2026-10-08**, `Asia/Yekaterinburg`.
План: [MINIMAX_M27_IMPLEMENTATION_PLAN.md](MINIMAX_M27_IMPLEMENTATION_PLAN.md).

Файл хранит проверенный прогресс и точку продолжения.
**MM27-11: реализовано объединение D2D-ожиданий в пределах тензора.**
Опция `--pipeline-d2d-batch 1` дополняет router lookahead. Новые cache fills
защищены от eviction/reuse до завершения копий; отмена и ошибки сначала
завершают операции, затем освобождают записи. Три пары A/B дали **+6,61–14,24% decode**
относительно MM27-10. Все logits совпали побитно; 4920 fixture checks,
давление памяти, перезагрузки и контексты 2K/4K прошли.
Проверенные флаги: `--gpu-cache-mib 18432 --gpu-cache-allocator arena --pipeline-readers 2 --pipeline-chunk-mib 4 --pipeline-lookahead 1 --pipeline-d2d-batch 1`.
Defaults: cache 0, allocator `cuda`, reader `file`, pipeline 0, lookahead 0, D2D batch 0.
Strict F32 activations/KV; FA/graphs/MTP off. API и рабочий профиль ещё не готовы.
**OPEN:** причина старого MM27-06 tiny native-after-cache reload расхождения
не установлена. Новые PASS не закрывают его причину.

## Текущее состояние

| Область | Статус |
|---|---|
| PREP-01 | DONE: инспекция и два документа |
| Ревизия Strata при подготовке | `e111f63f89aa77d69aed46b2150f7e61aaa5b05d`, рабочее дерево уже содержит изменения |
| Основной GGUF | `H:\models\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf` |
| Файл | 138 342 384 352 байта / 128,841 ГиБ, один GGUF |
| Архитектура | `minimax-m2`; 62 MoE-блока, full attention |
| Кванты | Q4_K375, Q6_K61, F32 373; BF16 tensors нет |
| Header/ranges | PASS постоянного inspector/contract, все809 tensors |
| Candidate dependency | Unsloth `86ebfef2`; отдельные CPU/CUDA builds, source/patch hashes сохранены |
| MTP | Нет local weights/NextN metadata; в candidate нет MiniMax MTP graph/loader |
| Tokenizer/template | PASS: 1316 tokenizer cases и103 template checks; API TODO |
| CUDA kernels / tiny graph | PASS:83 kernel cases в обоих режимах;217 graph cases с F32 активациями |
| Fast quant graph | 208/217 PASS;9 logit FAIL, greedy fixture совпал; runtime default не утверждён |
| Engine/cache/pipeline/sessions | Optional packed GPU cache, bounded mmap readers и tensor file pipeline с router lookahead и D2D batch; API/session reuse TODO |
| Reload validation | MM27-11: 55 stress и 18 tiny lifecycle checks PASS; старый native discrepancy остаётся OPEN |
| Скорость, токенов/с | D2D batch: 3,31 / 4,20 / 3,98; три пары A/B, подробности в MM27-11 |
| Рекомендуемые defaults | Для correctness: strict F32, FA/graphs/MTP off, context512/batch8; быстрый профиль не выбран |
| Следующая задача | **MM27-12: подобрать cache/readers/chunks и проверить длинные ответы; при numerical FAIL — трасса первого расхождения** |

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
- Windows RAM **125,555 ГиБ**, RTX5090 VRAM **32 607 МиБ**, driver581.80;
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
- Default temperature finetune и stop policy в HTTP/API. Experimental greedy runtime
  останавливается по EOS/PAD200020; FIM/reponame aliases не являются stop IDs.
  Содержимое конфигурации самого finetune не получено; официальный config
  базовой модели не заменяет проверку локального файла.
- API call ID/result correlation, tool-output parser, reasoning/content streaming
  и no-thinking профиль. Template-level nested `function`, JSON-string arguments
  и несколько calls/results проверены после явной нормализации.
- Причина двух tiny long-context native-after-cache reload расхождений MM27-06.
  Поздние PASS с диагностическим replay не доказывают устранение причины.
  Full-model cache-on pressure/cancel/unload и context2K/4K прошли в MM27-06;
  cache-off — MM27-04. Расширенная CUDA kernel timeline, continuous memory peaks/physical SSD traffic, длинные ответы
  и многошаговые сессии остаются отдельными проверками.
- Наличие пригодных native MTP weights, совместимость с heretic target,
  реализация graph/hidden/KV contract, acceptance и ускорение.

## Таблица этапов

| Этап | Статус | Условие готовности |
|---|---|---|
| PREP-01 | DONE | Исследование и документация сохранены |
| P0 — contract/oracles | DONE для F32 activation baseline; fast-quant logit issue открыт | Inspector, полный contract, изолированная сборка и CUDA fixtures |
| P1 — GPU baseline | Functional gates PASS: corpus, 2K/4K, cancel/recovery, pressure/unload | Остались расширенные quality/long-generation/SSD measurements |
| P2 — tokenizer/template/API | PARTIAL: offline tokenizer/template PASS; API TODO | Oracle fixtures, tools/reasoning, оба API и web chat |
| P3 — cache/pipeline | PARTIAL: synchronous packed GPU cache, full-model A/B/pressure PASS; tiny reload issue OPEN; bounded mmap, router lookahead, D2D batch и CUDA event overlap проверены | Bounded memory, bytes/logits parity, timeline и A/B |
| P4 — sessions/context | TODO | Fresh/restore/shift/cancel parity |
| P5 — MTP | WAITING_WEIGHTS_AND_BACKEND, дополнительный этап | Совместимые weights, native graph/loader, correctness и speedup |
| P6 — profile/regressions | TODO | Измеренные defaults и отдельный рабочий профиль |

Отсутствие P5 не блокирует P0–P4/P6. `DONE` относится только к указанной
части; чтение header и найденный upstream loader не означают working inference.

## Точка продолжения: MM27-12

1. P3.5: сравнить размеры GPU cache, chunks4/8/16 и readers1/2 с новым D2D batch.
   Учитывать фактическую residency после global clamp; измерять prefill/TTFT,
   decode и полное время запроса, включая более длинные ответы.
2. MM27-11 сократил host waits до одного на выбранный тензор. Compute fences,
   plan completion и ring reuse events сохранены; defaults пока не утверждены.
3. Старый MM27-06 discrepancy остаётся OPEN. При повторении сохранить EXE,
   inputs/logits и трассировать первый расходящийся node без ослабления tolerances.
4. P3.4 и P2/P4/P6: управляемый RAM LRU/GPU-resident partition, API/tools/reasoning,
   long-answer quality, snapshots/prefix/shift и рабочий профиль ещё предстоят.

## Подтверждённый журнал

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
