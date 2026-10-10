# MM27-36 — host router IDs: контекст, сессии и давление памяти

**MM27-36 завершён: offline, live pressure и CLI PASS.** Режим доступен
в экспериментальном сервере как opt-in; defaults сохранены.

Дата:2026-10-10. Продолжение [MM27-35](MINIMAX_M27_ROUTER_IDS.md).
Native EXE612354a1… не пересобирается; dependency86ebfef2… и арифметика
сохранены. Этот этап проверяет надёжность прежнего ускорения, а не новый A/B.
Ускорение3,87% относится к короткому корпусу и трём парам MM27-35.

## Изменения адаптера

Экспериментальный сервер получил boolean flag `--router-host-ids`.
Он требует `--pipeline-events 2` и точный SHA-256 бинарника:
`612354a12513eb1ce616ac2cc370bdd314a5c4725760bb586f6b9dd0ee1e6177`.
Аргумент native CLI имеет вид `--router-host-ids 1`; сервер переводит flag
в этот аргумент. Для нового EXE сервер передаёт0, если flag не задан.
Ready header обязан подтвердить boolean mode, events2 и decay65536.
Другой SHA либо несовместимая старая сборка отвергаются до загрузки весов.

Legacy98e85e80… и events13a885cc… по-прежнему принимаются в прежних режимах.
Основной bench EXE98e85e80… сохранён. Defaults router off/events0/cache0/
readers0/prefix off/archive0 не изменены. CUDA graph policy, dtype, sampler,
MTP/DFlash и shared transport не менялись.

Offline/live/startup drivers получили явный flag и запись режима в отчёт.
Они проверяют reuse counters отдельно от event/drain counters. Single-output
допускает отсутствие decode graph. Итоговый валидатор сверяет новый режим,
связывает этот EXE с коротким A/B MM27-35 и повторно читает полные logits,
memory logs и SHA исходников/артефактов.

60 CPU methods PASS: старый supervisor/pipe/cancel corpus, exact EXE admission,
boolean ready mode, несовместимые flags, default off, forwarding CLI flag,
memory/event/router evidence. Лог:
`build-local/minimax-m2-router-ids-context-unit.log`.

## Корпус и условия

RTX5090 32607МиБ, Ryzen9 9950X, RAM125,555ГиБ, Windows, driver581.80,
CUDA13.0. Target Q4_K_M138342384352 bytes в `H:\models\MiniMax-M2.7`.
Context4096/batch16, F32 activations/KV, FA/TF32/graphs/MTP/DFlash off,
GPU cache18ГиБ/arena64МиБ/reserve0/decay65536, readers2/chunk4/lookahead1/
D2D1/events2/router-host-ids1. RAM expert cache0, prefix on,
session archive6144МиБ/4 snapshots, `STRATA_MM27_STATE_BULK=1`.

Offline: прежний корпус14 requests из MM27-28/MM27-34. Cold/restore2K/4K,
EOS414, sampling128/seed42 и single-output. Каждый output ID и полный F32
logit vector сравниваются с сохранённым MM27-26 reference; ещё шесть пар
fresh/restore сравниваются внутри текущего процесса.

Live:19 сценариев в одном native PID, OpenAI/Anthropic JSON/SSE. Restore,
EOS/sampling при RAM/VRAM pressure, GPU arena trim, отказ сохранения4K при
дефиците физической RAM с защитой запрошенного2K, disconnect во время prefill,
отмена decode и восстановление работы. Live сравнивает IDs, не полные logits.
Затем отдельный реальный server CLI: auth401, HTML200, два одинаковых коротких
ответа с unload и автоматической повторной загрузкой.

Pressure holder использует read-only GGUF pages, до16ГиБ private RAM и до12ГиБ
GPU buffers. Цели RAM94,2%/VRAM94,5%, предел95%; это диагностическая нагрузка,
а не кеш движка. GPU-прогоны последовательные; owned processes ограничены
Job/timeout. Native guards и независимый Windows/NVML observer сохранены.

## Команды

При повторении требуются новые output directories.

```powershell
python -X utf8 -m unittest serve.test_minimax_m2_engine serve.test_minimax_m2_server tools.test_minimax_m2_memory_observer tools.test_minimax_m2_sessions_context tools.test_minimax_m2_sessions tools.test_minimax_m2_prefix_context tools.test_minimax_m2_http_context tools.test_minimax_m2_copy_events tools.test_minimax_m2_state_bulk_pressure tools.test_minimax_m2_router_ids
python -X utf8 tools/check_minimax_m2_sessions_context.py --gguf H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --engine build-local/minimax-m2-router-ids-candidate/engine.exe --pipeline-events 2 --router-host-ids --out build-local/minimax-m2-router-ids-context-01 --stage 'MM27-36 offline'
$env:STRATA_MM27_STATE_BULK='1'
python -X utf8 tools/check_minimax_m2_sessions_pressure.py --gguf H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --engine build-local/minimax-m2-router-ids-candidate/engine.exe --pipeline-events 2 --router-host-ids --offline build-local/minimax-m2-router-ids-context-01 --out build-local/minimax-m2-router-ids-pressure-01 --stage 'MM27-36 pressure'
python -X utf8 tools/check_minimax_m2_startup.py --gguf H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --engine build-local/minimax-m2-router-ids-candidate/engine.exe --pipeline-events 2 --router-host-ids --out build-local/minimax-m2-router-ids-startup-01
python -X utf8 tools/validate_minimax_m2_state_bulk_pressure.py --offline build-local/minimax-m2-router-ids-context-01 --pressure build-local/minimax-m2-router-ids-pressure-01 --engine build-local/minimax-m2-router-ids-candidate/engine.exe --pipeline-events 2 --router-host-ids --stage MM27-36 --unit-log build-local/minimax-m2-router-ids-context-unit.log --python-methods 60 --startup build-local/minimax-m2-router-ids-startup-01 --out docs/minimax-m2.7/MINIMAX_M27_ROUTER_IDS_CONTEXT_CHECK.json
```

## Offline: PASS

14 requests /1150 IDs /230073600 F32 logits побитно совпали с reference.
Шесть fresh/restore пар: ещё113436288 сравнений logits, также побитно.
Оба длинных ответа завершились по EOS на414 tokens; sampling128/seed42 и
single-output прошли. NaN/Inf нет.

286812 reuse hits,0 misses;1130156928 bytes повторных D2H чтений устранены.
Число copy events также286812, graph-exit fences1542. Pending copy и retirement
waits на границах0. Независимый monitor:1604 samples, RAM24,563%/VRAM87,506%,
ошибок и превышений95% нет; observer завершён до hashing.

| Запрос | С нуля, с | Восстановление, с |
|---|---:|---:|
| 2K,8 outputs | 215,507 | 5,190 |
| 4K,8 outputs | 288,314 | 5,300 |
| EOS414 | 101,501 | 91,810 |
| Sampling128 | 39,015 | 28,128 |
| Один output | 10,009 | 3,008 |

Это полное request time одного прохода. Состояние expert cache и исходящих
сессий различается; таблица не является новым speed A/B. Повторное restore4K
после других запросов заняло7,103с. Артефакты:
`build-local/minimax-m2-router-ids-context-01`.

## HTTP/pressure и CLI: PASS

Все19 scenarios в одном native PID прошли.17 полных HTTP-ответов /662 IDs
совпали с offline, EOS414 и sampling128/seed42 завершены. OpenAI/Anthropic
JSON/SSE, cached usage и сохранение другой сессии после обеих отмен прошли.
218364 reuse hits /0 misses, столько же copy events,1174 graph-exit fences.

| Метрика | Результат |
|---|---:|
| Пик global RAM по всем записанным samples | **94,179683%** |
| Пик global VRAM | **94,430248%** |
| Независимый observer: RAM /VRAM | 94,109797% /94,430248% |
| Независимые samples | 1967, без ошибок |
| Private RAM buffers holder | 11,625ГиБ из лимита16ГиБ |
| GPU buffers holder | 1856МиБ из лимита12ГиБ |
| Прочитано через read-only GGUF mapping | 138342384352байт |
| GPU arena trim | 128МиБ, один trim |
| Отказ сохранения4K /eviction archive | 1 /1 |
| Восстановлено из защищённого2K | 1035642236байт |
| Prefill disconnect | 1361,304мс,0 output tokens |
| Decode cancel | 3,232мс,0 дополнительных tokens |

Все записанные samples укладываются в95%. Target RAM94 зафиксировал94,175%;
ближайший независимый sample через110мс подтвердил диапазон93,5–95%.
Первый последующий sample также подтверждает недостаток физической RAM
для4K snapshot плюс5% и256МиБ резерва (нужно9084974664байт), при достаточном
commit и свободных caps архива. Пороги и правила admission не менялись.

В одном проходе restore4K под нагрузкой занял8,829с, restore2K —3,496с,
protected2K —5,123с, EOS414 —102,334с, sampling128 —32,759с.
Fresh2K после prefill cancel:161,017с, прежние8 IDs, reuse0.
Это HTTP wall times с разными состояниями кэша, не новый speed A/B.
После FREE все GPU/private/touched counters holder стали0; следующий запрос
тоже прошёл. Engine и holder закрыты, cleanup errors отсутствуют.

Отдельный настоящий CLI:6 checks PASS —401 без ключа, HTML200, два одинаковых
8-token ответа с unload и автоматической повторной загрузкой. Server остановлен.
CLI scope — context2048/batch16 без prefix/archive; long/session scope проверен
отдельным offline и live pressure при context4096/batch16.

Итоговый [машинный отчёт](MINIMAX_M27_ROUTER_IDS_CONTEXT_CHECK.json):
**116 gates,173 source checks,239 artifact SHA-256 PASS**. Полные logits
повторно сравнены, router/event accounting пересчитан, exact EXE связан с
MM27-35, source snapshots/memory logs/закрытие четырёх owned PID проверены.
После всех проверок GPU1202МиБ/0%. Артефакты:
`build-local/minimax-m2-router-ids-pressure-01`,
`build-local/minimax-m2-router-ids-startup-01`,
`build-local/minimax-m2-router-ids-context-validation.log`.

## Запуск сервера

Для профиля context4K/batch16 с prefix/session archive:

```powershell
$env:STRATA_MM27_STATE_BULK='1'
python -X utf8 -m serve.minimax_m2_server --gguf H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --engine build-local/minimax-m2-router-ids-candidate/engine.exe --ctx 4096 --batch 16 --gpu-cache-mib 18432 --pipeline-readers 2 --pipeline-chunk-mib 4 --pipeline-events 2 --router-host-ids --prefix-cache --session-cache-mib 6144 --session-cache-slots 4 --port 8080
```

Адрес по умолчанию `127.0.0.1`; внешний bind требует `--api-key`.
Новый flag не меняет публичные OpenAI/Anthropic request bodies.
Для короткого context2048 без сессий можно убрать три prefix/session flags
и заменить `--ctx 4096` на `--ctx 2048`; этот вариант проверяет CLI smoke.

## Оставшиеся границы

Этот этап относится к batch16/context4096. Batch1/8 prefix/archive на2K/4K,
shift/context>4K, широкое answer quality, independent model oracle и причина
старого MM27-06 reload discrepancy остаются отдельными gates. Один проход
надёжности не заменяет повторный speed A/B. H2D fence таблицы перестановки
в strict F32 ещё остаётся; оптимизация его lifetime требует отдельной проверки.
