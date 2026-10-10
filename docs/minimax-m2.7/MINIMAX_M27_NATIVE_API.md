# MM27-22: resident JSONL Engine и экспериментальный сервер

Дата: **2026-10-09**, `Asia/Yekaterinburg`; ревизия
`54467693c8354d8c64a72ec1cb083473a529b4ba`, dirty tree.
[План](MINIMAX_M27_IMPLEMENTATION_PLAN.md),
[статус](MINIMAX_M27_IMPLEMENTATION_STATUS.md).

`serve/minimax_m2_engine.py` подключает настоящий CUDA JSONL-процесс к
`Service`. Оба API используют прежние history/parser/JSON/SSE adapters.
`serve/minimax_m2_server.py` — отдельный startup entry point. Это часть
P2.5/P2.6 и экспериментальный профиль; общий installer/config registry и
приёмка web chat остаются отдельной работой.

## Запуск на этом ПК

Из корня репозитория, в существующем Python environment Strata:

```powershell
python -m pip install -r serve/requirements-minimax.txt
python -X utf8 -m serve.minimax_m2_server --gguf H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --ctx 2048 --batch 16 --gpu-cache-mib 18432 --pipeline-readers 2 --pipeline-chunk-mib 4 --port 8080
```

Дополнительная зависимость tools — jsonschema4.26.0; базовые зависимости
из корневого requirements.txt должны быть установлены. На этом ПК пакет
уже был установлен; новых пакетов при проверках не скачивали. Startup
проверяет validator до загрузки весов. API model name: `minimax-m2.7`.
Адрес по умолчанию127.0.0.1; иной `--host` требует `--api-key`.

Default CLI: context512, batch8, cache0, readers0, chunk8. Команда выше
явно выбирает ранее измеренный экспериментальный набор18ГиБ/2/4 и context2K.
Адаптер использует arena64/reserve0, file reader, RAM0/grouping off,
lookahead/D2D batch при включённом pipeline. Native CLI defaults сохранены.
Precision задаётся до запуска процесса: F32 activations/KV, TF32/FA/graphs off.
Sampling default greedy; MTP/DFlash и session reuse выключены.
Для opt-in RAM-снимков неактивных сессий добавлены
`--prefix-cache --session-cache-mib 600 --session-cache-slots 2`;
клиент передаёт `X-Strata-Session-Id`. Контракт и границы проверок:
[MM27-27](MINIMAX_M27_SESSION_ARCHIVE.md). Default cap0.

MM27-34 добавил проверенный opt-in `--pipeline-events 2` для CUDA matrix
events/async splits. К команде запуска выше добавить
`--engine build-local/minimax-m2-copy-events-candidate/engine.exe --pipeline-events 2`.
Server принимает конкретный SHA13a885cc…; старый98e85e80… поддерживает режим0.
Batch16/context2K/4K, prefix/archive, EOS, real RAM94,20%/VRAM94,45%, обе
отмены/recovery и CLI reload проверены. [Команды и область MM27-34](MINIMAX_M27_COPY_EVENTS_CONTEXT.md).
Default events0 сохранён; mode1 доступен только в native diagnostics.
[MM27-35](MINIMAX_M27_ROUTER_IDS.md) добавляет native `--router-host-ids 1`.
[MM27-36](MINIMAX_M27_ROUTER_IDS_CONTEXT.md) проверил этот EXE на2K/4K,
sessions/EOS/sampling, real RAM94,180%/VRAM94,430% и обеих отменах/recovery.
Server CLI принимает boolean `--router-host-ids` с
`--engine build-local/minimax-m2-router-ids-candidate/engine.exe --pipeline-events 2`.
Нужны readers; ready header подтверждает режим, несовместимый SHA отвергается.
Default off сохранён. Драйвер engine.py принимает `router_host_ids=True`.

Используется только проверенный GGUF/header и конкретный EXE. Изменившаяся
сборка отклоняется до загрузки, пока её SHA не проверен и явно не принят.
EXE SHA этого исторического этапа (новый admission указан в
[MM27-29](MINIMAX_M27_STATE_BULK.md)):
`f4ff449823beee3db1b260597cac1ed9276fb53922c5816f082f77ccbf52f619`.
Header SHA:
`9a011b2ad6fb20db3bb0cdfe9af08665939ee1f0cb0609ffe2bfee7f31c825db`.
Dependency86ebfef2; template и веса не менялись. Неизвестная модель или
непроверенный бинарник не допускаются переключателем обхода проверки.

## Транспорт и повторная отмена

Native JSONL не содержит request IDs и stdin-команды cancel. Отдельный
`serve/minimax_m2_worker.py` владеет ровно одним дочерним процессом,
добавляет IDs и принимает отмену только активного запроса. На Windows
worker имеет скрытую отдельную консоль; CTRL_BREAK адресуется группе
своего native child. Windows Job Object завершает child при аварии worker.

Adapter проверяет ready config/provenance/95% memory admission, token IDs,
порядок EOS, terminal counts, timings и cache counters. Закрытие генератора
дочитывает terminal под lock; ранняя остановка сначала отменяет native work.
Так следующий запрос не получает хвост предыдущего. Ошибка native request
остаётся recoverable, ошибка протокола/EOF/timeout завершает owned process;
следующий запрос Service запускает его заново. Очередь событий ограничена64
записями, строка16МиБ; request timeout900с, cancel/drain30с. Pre-cancelled
request не отправляется в CUDA. Concurrent access вне FIFO отклоняется.

Первый live проход нашёл native bug: stop-string отмена проходила, а
следующая отмена во время prefill завершала процесс. MSVC сбрасывает
`signal(SIGBREAK, ...)` в SIG_DFL перед вызовом обработчика —
[документация Microsoft](https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/signal?view=msvc-170).
В main.cpp установлен постоянный
[SetConsoleCtrlHandler](https://learn.microsoft.com/en-us/windows/console/setconsolectrlhandler).
CUDA kernels, arithmetic, sampler и delivery pipeline не изменены.
Старый EXE/main/manifest сохранены в
`build-local/minimax-m2-live-api-baseline-01`, исходный FAIL —
`build-local/minimax-m2-live-api-01/report.json`; успешные ранние checks
не превращают этот первый проход в PASS.

## Воспроизведение

**31 CPU methods PASS, exit0**, без skips:14 subprocess/Engine,2 startup
admission и15 прежних API tests. **11 live GPU scenarios PASS, exit0** в
`build-local/minimax-m2-live-api-02`: OpenAI JSON/Anthropic SSE prefix,
полный OpenAI EOS, stop string, три HTTP prefill disconnect в одном PID,
recovery, настоящий OpenAI toolcall/result/answer и HTTP-triggered reload.
В каждом полноценном native result проверены GPU-only/bytes/drain и95% cap.
Три disconnect заняли0,625/0,578/0,531с до завершения native cancel/drain.
Отдельный CLI startup checker: **6 checks PASS, exit0**,401 без API key,
доступ к HTML, два8-token запроса с одинаковым ответом и две HTTP выгрузки,
между которыми Service автоматически загрузил модель снова. Артефакты:
`build-local/minimax-m2-startup-01`. Тестовые серверы завершены.
Сводка: [NATIVE_API_CHECK](MINIMAX_M27_NATIVE_API_CHECK.json),
проверка хэшей и артефактов:
[NATIVE_API_VALIDATION_CHECK](MINIMAX_M27_NATIVE_API_VALIDATION_CHECK.json).
Snapshots/logs/manifest проверки сохранены в
`build-local/minimax-m2-native-api-validation-01`.

RTX5090,32607МиБ, driver581.80, system RAM125,555ГиБ;
MSVC19.44.35222.0, CUDA13.0.48, SM120a. Сборка:
`build-local/build-minimax-m2-live-api.bat`, exit0; configure/build logs
`build-local/minimax-m2-live-api-{configure,build}.log`.

| Запрос | Prompt / generated | Prefill, с | TTFT, с | Decode, с | Decode, токенов/с |
|---|---:|---:|---:|---:|---:|
| Полный EN ответ, greedy | 52 /414 | 8,494 | 8,494 | 103,818 | 3,978 |
| get_code, T1/p0,95/k40/seed42 | 220 /53 | 38,838 | 38,840 | 15,608 | 3,332 |
| Ответ после результата tool | 295 /37 | 46,080 | 46,081 | 9,599 | 3,751 |

Generated включает reasoning, tool wire и EOS; decode rate считает N−1
forward steps. Полный EN ответ совпал с MM27-18 по всем414 IDs, HTTP wall112,656с.
Это один проход без независимого logit oracle и без A/B speedup утверждения.
Короткий прогретый8-token SSE prefix дал8,047 токена/с; он не сопоставим
с полным ответом. Tools corpus содержит один get_code/city=Oslo и результат
OSLO-4179; модель вернула этот код и EOS. Никаких внешних API tool calls не было.
Максимальные sampled global peaks полного live прохода: RAM41,359ГБ
(30,68% из125,555ГиБ), VRAM28,943ГБ (84,65% из32607МиБ), ниже95%.
Это периодические snapshots/runtime samples, не continuous hardware trace.

```powershell
python -X utf8 -m unittest -v serve.test_minimax_m2_engine serve.test_minimax_m2_server serve.test_minimax_m2_api
python -X utf8 tools/check_minimax_m2_live_api.py --gguf H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --out build-local/mm27-new-live-api
python -X utf8 tools/check_minimax_m2_startup.py --gguf H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --out build-local/mm27-new-startup
```

Оба GPU checker требуют нового `--out` и запускаются последовательно.
Live checker сохраняет raw HTTP replies, requests, native results, source
snapshots и EXE. Startup checker запускает настоящий CLI процесс, проверяет
auth и HTTP unload/reload. Получение HTML не считается приёмкой web chat.
CPU suite запускает настоящий supervisor с небольшой Python fixture вместо
CUDA: EOF, malformed/stale records, terminal drain, error recovery, repeated
requests, targeted cancellation, timeout, shutdown и отсутствие orphan child.

## Что остаётся открытым

Следующий [MM27-23](MINIMAX_M27_LIVE_TOOLS_WEB.md) проверил ограниченный
multi-call/error corpus обоих API и настоящий browser chat. Ниже зафиксирован
scope на момент завершения MM27-22.

Широкий model tool corpus, несколько инструментов/ошибок в живой сессии,
браузерный chat и общий installer ещё не приняты. Тестовый local get_code
проверяет только один цикл с данными fixture; внешние инструменты не вызываются.
Входные ограничения из [MM27-21](MINIMAX_M27_API.md) сохраняются: text-only,
always-on reasoning, auto/none tools, без forced tools/structured output/MCP.
HTTP pressure/4K, sessions, независимый full-model oracle, общее качество
ответов и старое MM27-06 reload discrepancy не закрываются этим этапом.
Транспортные token matches не являются новым logit oracle или speedup A/B.
