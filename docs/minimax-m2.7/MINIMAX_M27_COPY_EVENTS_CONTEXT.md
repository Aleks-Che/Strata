# MM27-34 — CUDA events: длинный контекст, сессии и давление памяти

Дата:2026-10-09–10, `Asia/Yekaterinburg`. Strata HEAD54467693… с ранее
существовавшими staged/unstaged changes; dependency86ebfef2….

Продолжение [MM27-33](MINIMAX_M27_COPY_EVENTS.md). Offline, live pressure и CLI
проверки **PASS**. Режим2 доступен в экспериментальном сервере с конкретным
EXE13a885cc…; defaults сохранены. Это проверка надёжности ранее измеренного
ускорения4,40%, нового speed A/B на этом этапе нет.

## Изменения

Native runtime и кандидат13a885cc… не пересобираются. Серверный адаптер
получил opt-in `--pipeline-events 2` и admission конкретного events EXE.
Legacy98e85e80… продолжает работать с режимом0; запрос events2 к этому EXE
отклоняется до загрузки модели. Неизвестный SHA также отклоняется.
Ready header должен подтвердить выбранный режим и decay65536.
Флаг требует readers; сервер включает lookahead/D2D при включённом pipeline.
Default events0, cache0, readers0, prefix/archive off сохранён.

Диагностические drivers теперь принимают явные `--engine` и
`--pipeline-events`; прежние defaults сохраняются. Проверка event accounting
учитывает ответ из одного токена: он использует prefill logits и не запускает
decode graph. К offline-проходу добавлен независимый Windows/NVML observer;
при ошибке измерения или превышении95% он завершает свой native child.
GPU-проходы последовательные, процессы ограничены Job/timeout.

## Корпус и границы

Используется прежний MM27-28/MM27-30 корпус:14 offline requests, cold/restore
2K/4K, полный английский EOS, sampling128 с seed42, single-output. Полные
F32 logits и IDs сравниваются с сохранённым reference MM27-26; дополнительно
сравниваются шесть fresh/restore пар внутри текущего процесса.

Затем19 live scenarios через оба API: JSON/SSE, cached usage, restore2K/4K,
EOS/sampling при давлении RAM/VRAM, GPU cache trim, отказ сохранить исходящий
4K snapshot с защитой запрошенного2K, prefill HTTP disconnect, decode cancel
и fresh recovery в том же native PID. Live JSONL сравнивает IDs, не полные logits.
После нагрузки отдельный CLI smoke проверяет запуск через реальный server
entry point,401 без ключа, доступ к HTML, два одинаковых коротких ответа
с HTTP unload между ними и автоматическую повторную загрузку.

Context4096/batch16, F32 activations/KV, TF32/FA/graphs/MTP/DFlash off,
cache18ГиБ/arena64/reserve0/decay65536, readers2/chunk4/lookahead/D2D,
events2, RAM expert cache0, prefix on, archive6144МиБ/4 snapshots,
`STRATA_MM27_STATE_BULK=1`. RTX5090 32607МиБ, RAM125,555ГиБ,
Windows/CUDA13.0/driver581.80. Локальный target Q4_K_M и dependency86ebfef2 прежние.

Pressure holder не изменён: read-only GGUF pages, private buffers до16ГиБ
и GPU buffers до12ГиБ, цели RAM94,2%/VRAM94,5%, guard95%.
Это дополнительная диагностическая нагрузка, не кеш движка.
Проверка не является повторным speed A/B; измеренное ускорение остаётся
результатом MM27-33. Batch1/8 на2K/4K, shift/context>4K, большой quality corpus,
independent oracle и старый MM27-06 discrepancy остаются отдельными gates.

## Команды

Из корня Strata; новые output directories обязательны при повторении.

```powershell
python -X utf8 -m unittest serve.test_minimax_m2_engine serve.test_minimax_m2_server tools.test_minimax_m2_memory_observer tools.test_minimax_m2_sessions_context tools.test_minimax_m2_sessions tools.test_minimax_m2_prefix_context tools.test_minimax_m2_http_context tools.test_minimax_m2_copy_events tools.test_minimax_m2_state_bulk_pressure
python -X utf8 tools/check_minimax_m2_sessions_context.py --gguf H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --engine build-local/minimax-m2-copy-events-candidate/engine.exe --pipeline-events 2 --out build-local/minimax-m2-copy-events-context-01 --stage 'MM27-34 offline'
$env:STRATA_MM27_STATE_BULK='1'
python -X utf8 tools/check_minimax_m2_sessions_pressure.py --gguf H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --engine build-local/minimax-m2-copy-events-candidate/engine.exe --pipeline-events 2 --offline build-local/minimax-m2-copy-events-context-01 --out build-local/minimax-m2-copy-events-pressure-01 --stage 'MM27-34 pressure'
python -X utf8 tools/check_minimax_m2_startup.py --gguf H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --engine build-local/minimax-m2-copy-events-candidate/engine.exe --pipeline-events 2 --out build-local/minimax-m2-copy-events-startup-01
python -X utf8 tools/validate_minimax_m2_state_bulk_pressure.py --offline build-local/minimax-m2-copy-events-context-01 --pressure build-local/minimax-m2-copy-events-pressure-01 --engine build-local/minimax-m2-copy-events-candidate/engine.exe --pipeline-events 2 --stage MM27-34 --unit-log build-local/minimax-m2-copy-events-context-unit-02.log --python-methods 51 --startup build-local/minimax-m2-copy-events-startup-01 --out docs/minimax-m2.7/MINIMAX_M27_COPY_EVENTS_CONTEXT_CHECK.json
```

51 CPU test methods PASS, exit0; лог
`build-local/minimax-m2-copy-events-context-unit-02.log`. Первый проход47 методов
также PASS, сохранён отдельно в `build-local/minimax-m2-copy-events-context-unit.log`. Admission tests проверяют
подменённый EXE, несовместимый старый EXE, неверный ready mode/decay и ранний
отказ при events без readers. Прежний subprocess lifecycle корпус также прошёл.

## Offline: PASS

14 requests /1150 IDs /230073600 F32 logits побитно совпали с reference;
шесть fresh/restore пар — ещё113436288 сравнений logits, также побитно.
NaN/Inf нет. Оба полных ответа дошли до EOS на414 tokens; sampling128/seed42
и single-output прошли. Счётчики matrix dependencies, pending copy и graph
drain проверены на prefill/decode каждого запроса.1696 independent samples,
пики RAM26,428%/VRAM88,175%; ошибок измерения и превышений95% нет.
Observer завершён до hashing.
В сумме286812 matrix copy events и столько же retirement checks;
1542 graph-exit fences. Pending copies и observer fences на границах0.
Retirement ожиданий0; внутренние strict F32 CUDA waits этим не измеряются.

| Запрос | С нуля, с | Восстановление, с |
|---|---:|---:|
| 2K,8 outputs | 228,184 | 5,435 |
| 4K,8 outputs | 299,716 | 5,439 |
| EOS414 | 108,853 | 100,946 |
| Sampling128 | 41,461 | 29,853 |
| Один output | 10,142 | 3,066 |

Это полное request time одного прохода с разными состояниями expert cache
и исходящих сессий. Таблица не является A/B скорости режима0 против2.
Сравнение sampling выполнено с fresh без session key; greedy single-output
и EOS имеют отдельные сессии. Второй restore4K после других запросов занял7,358с.
Артефакты: `build-local/minimax-m2-copy-events-context-01`, включая raw logits,
EXE/source snapshots, JSON report и независимый memory log.

## HTTP/pressure и запуск: PASS

Все19 scenarios в одном native PID прошли:17 полных HTTP-ответов /662 IDs
совпали с offline, ещё два сценария проверили отмену. Оба API, JSON/SSE,
cached usage, EOS414 и sampling128/seed42 прошли. Mode2 подтверждён ready
header; в полных ответах218364 copy events/retirement checks,1174 graph-exit
fences, pending copies0 и observer fences0. Live проверка сравнивает IDs;
полные logits проверены отдельным offline-проходом.

| Метрика | Результат |
|---|---:|
| Пик global RAM по всем записанным отсчётам | **94,199274%** |
| Пик global VRAM | **94,450638%** |
| Независимый observer: RAM / VRAM | 94,076136% /94,450638% |
| Отсчёты независимого observer | 2110, без ошибок |
| Дополнительные private RAM buffers holder | 12,125ГиБ из лимита16ГиБ |
| GPU buffers holder | 1792МиБ из лимита12ГиБ |
| Сокращение GPU arena | **128МиБ**, один trim |
| Отказ сохранить4K / eviction archive | 1 /1 |
| Восстановлено из защищённого2K | 1035642236байт |

Все записанные RAM/VRAM samples укладываются в95%. Holder прочитал весь GGUF
через read-only mapping; это не дополнительный полный heap copy модели.
HTTP disconnect во время restored prefill обработан за1,702с; отмена после
первого decode token — за3,561мс, дополнительных токенов клиент не получил.
Другая сессия пережила обе отмены. Fresh2K recovery: прежние8 IDs, reuse0.
После FREE у holder нулевые GPU/private/touched counters; последний resident
reuse также прошёл. Owned native/holder/CLI процессы завершены, cleanup без ошибок.

В одном проходе: restore4K при первой нагрузке6,499с, restore2K3,733с;
защищённый restore2K —5,444с; EOS (414 токенов) —111,923с;
sampling (128 токенов) —35,245с.
Это HTTP wall time с разными expert-cache состояниями, не новый speed A/B.

**Уточнение измерения RAM.** Holder зафиксировал94,196% при RAM94. Ближайший
независимый sample за46мс до этого показал94,076%; первый последующий через
469мс уже застал освобождение архива и показал93,363%. Старое ожидание,
что именно последующий sample обязательно останется в диапазоне93,5–95%,
здесь не выполнилось (`post_percent_target_met=false` сохранён в JSON).
Validator выбирает ближайший к timestamp отсчёт по времени, не по проценту,
и требует расстояние не более одного периода опроса плюс50мс.
Отдельно проверяет нехватку физических bytes по первому последующему sample:
свободно8947380224байт против9084974664 для4K snapshot с резервом; нехватка
137594440байт. Commit свободен54,783ГБ, caps архива тоже не исчерпаны.
Поэтому причина отказа подтверждена физической RAM. Порог95%, размер
резерва и native admission не менялись. Четыре новые CPU проверки покрывают
раннее освобождение, отсутствие независимого target, слишком редкие samples
и ложную подмену физического дефицита недостатком commit.

Отдельный CLI smoke:6 checks PASS —401 без ключа, HTML200, два одинаковых
8-token ответа, HTTP unload и автоматический reload между ними.51 CPU methods,
**111 итоговых gates /165 source checks /231 artifact hashes PASS**, exit0.
[Машинный отчёт](MINIMAX_M27_COPY_EVENTS_CONTEXT_CHECK.json).
Pressure и CLI артефакты:
`build-local/minimax-m2-copy-events-pressure-01` и
`build-local/minimax-m2-copy-events-startup-01`.

## Проверенный серверный профиль

Из корня репозитория:

```powershell
python -X utf8 -m serve.minimax_m2_server --gguf H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --engine build-local/minimax-m2-copy-events-candidate/engine.exe --ctx 2048 --batch 16 --gpu-cache-mib 18432 --pipeline-readers 2 --pipeline-chunk-mib 4 --pipeline-events 2 --port 8080
```

Для проверенного4K/session профиля заменить context на `--ctx 4096` и добавить
`--prefix-cache --session-cache-mib 6144 --session-cache-slots 4`.
Клиент указывает `X-Strata-Session-Id`. Адрес по умолчанию127.0.0.1;
внешний bind требует `--api-key`. Тестовые серверы после проверки закрыты.

Допущенный events EXE SHA-256:
`13a885cc75352bb39cc4231174da3d9f03705d5f9c150117935d6f05ea9d2401`.
Legacy98e85e80… также допущен для прежнего режима0; основной bench EXE
не заменён. `--pipeline-events 2` требует events EXE и readers, mode1 остаётся
только native диагностикой. Full-model scope нового server opt-in: batch16,
ctx2048 на коротком корпусе MM27-33/CLI и ctx4096 на2K/4K/EOS/session корпусе.
Общий installer и P6 quality/oracle gates остаются.
