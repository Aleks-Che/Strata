# MM27-21: адаптеры OpenAI и Anthropic

Продолжение: [MM27-22 — подключение native CUDA Engine и отдельный сервер](MINIMAX_M27_NATIVE_API.md).
Ниже сохранён scope проверки MM27-21 со scripted engine.

Дата: **2026-10-09**, `Asia/Yekaterinburg`; ревизия
`d342001b4dca70860bcd585f28004591a83e5bf3`, dirty tree.
[План](MINIMAX_M27_IMPLEMENTATION_PLAN.md),
[статус](MINIMAX_M27_IMPLEMENTATION_STATUS.md).

Добавлен `serve/minimax_m2_api.py`: `MiniMaxAPITemplate` подключает history,
parser, API normalization и строгий UTF-8 decoder к существующему `Service`.
Используются общие HTTP handlers и JSON/SSE serializers. В `serve/server.py`
добавлены выбор decoder по template и финальная проверка его хвоста.
Остальные templates сохраняют прежний decoder и его поведение.

Это часть P2.5/P2.6. HTTP-проверки используют scripted engine, настоящий
loopback socket и production handlers. Native CUDA-процесс к Service ещё
не подключён; startup/config registration, рабочий профиль и web chat ещё TODO.
Готовность модели к реальным tool calls этими проверками не установлена.

## Входной контракт

- OpenAI: `messages`, начальные system/developer, text/input_text parts,
  отдельный reasoning и nested/flat function calls нормализуются прежним
  canonical history module. Входной request не мутирует. Call IDs и JSON-string
  arguments сохраняются; результаты переставляются в порядок исходных calls.
- Anthropic: верхний `system`, user/assistant messages, thinking/text/tool_use
  и tool_result преобразуются в тот же canonical context. Thinking signature
  принимается как текстовая annotation; криптографическая проверка отсутствует.
  `is_error=true` у результата даёт строку `Error: ` перед его содержимым.
- Порядок блоков, который template представить не может, отклоняется явно:
  thinking после text/tools, text после tool_use, новый user text до всех
  ожидаемых tool results. Redacted thinking, media и неизвестные blocks
  не пропускаются. Поддержка произвольного interleaved assistant output/history
  не заявляется: после calls native template ожидает результаты инструментов.
- Tool definitions и optional dependency jsonschema проверяются до generation.
  Сохраняются [ограничения wire/schema parser](MINIMAX_M27_TOOLS.md).
  `tool_choice=auto` и `none` поддержаны; `none` убирает доступные tools из
  нового prompt/parser, не удаляя historical calls/results. Forced/named choice
  и запрет parallel calls пока отклоняются. Tools исполняет внешний API client.
- Встроенный template всегда начинает reasoning. Поддержан только фиксированный
  профиль `high`/always-on; это обозначение существующего режима, не новый
  уровень затрат. Anthropic `thinking.type=enabled` принимается без budget.
  No-thinking/adaptive, low/medium и hard reasoning budgets не эмулируются.
- Неизвестные request/template controls отклоняются. Проверяются типы stream,
  output cap, stop sequences, диапазоны temperature/top_p/top_k/seed из native
  sampler. `max_tokens`/`max_completion_tokens` допускают0/-1 как остаток context;
  разные явно переданные лимиты дают400. Превышение context проверяет Service.
  Default sampler этой работой не меняется.
- Structured response_format, server-side MCP execution и дополнительные
  sampling controls не подключены. Общий installer/requirements не изменён;
  jsonschema capability/install необходимо включить при добавлении startup profile.

## Выход, EOS и отмена

Оба API используют проверенные reasoning/content/tool events. OpenAI выдаёт
`tool_calls`, Anthropic — `tool_use`; полные tool arguments передаются атомарно.
Usage включает reasoning и EOS. При достижении output cap приоритет остаётся
у `length`/`max_tokens`, даже если целая группа tools уже получена.
Незавершённая группа остаётся текстом. Reasoning-only OpenAI reply имеет
`content=null`; это существующий serializer contract.

EOS определяется по metadata и проверенным token spellings/types: только
`[e~[` (в локальной модели200020). PAD совпадает с EOS. `<fim_pad>`/`<reponame>`
не добавляются в stop set. Для тестового tokenizer IDs перемещены; для
реального GGUF дополнительно проверено соответствие ID200020.

Строгий decoder сохраняет UTF-8 между token bytes, отказывается от invalid
bytes и проверяет незаконченный символ в конце генерации. Ошибка не маскируется
символом замены или успешным finish_reason. Для JSON используется существующий
error response сервера, для SSE — error event/chunk; следующий запрос работает.
При disconnect соединение уже закрыто, decoder не пытается завершить его ответ.

Socket watcher отменяет запрос в очереди, prefill, reasoning и внутри tool group.
Генератор закрывается под FIFO, после чего следующий запрос получает свой
parser/decoder и успешно завершается. Это проверка Service и scripted engine;
отмена и drain настоящего JSONL CUDA-процесса остаются отдельным этапом.

## Проверки и артефакты

Команда checker сохраняет копии sources/CPU EXE, unit logs, полные native prompts,
token IDs, JSON/SSE replies и report. Подробные результаты находятся в
[API_CHECK](MINIMAX_M27_API_CHECK.json) и
[API_VALIDATION_CHECK](MINIMAX_M27_API_VALIDATION_CHECK.json).

Финальный результат **PASS, exit0**:191 methods без failures/errors/skips
(15 новых adapter/HTTP,51 прежний MiniMax parser/history/tokenizer и125
server/GLM/Step/Hy3 regressions);272 дополнительных report checks.
Реальный tokenizer используется для16 scripted двухшаговых tool cycles
(два API × JSON/SSE × четыре double-значения) и32 HTTP replays восьми
сохранённых completions MM27-18b. Tool-output fixtures создаёт native Jinja,
а не LLM. Native template/tokenizer совпали с32 API prompts /14284 token IDs.
Финальные artifacts: `build-local/minimax-m2-api-check-02`; первый проход сохранён.

Две первоначальные ошибки fixtures сохранены:

1. `build-local/minimax-m2-api-initial-fixture-failure-01`: четыре length subcases
   считали budget по частично написанному closing token, затем применяли его
   к stream с целым special token. Тест исправлен на настоящую token boundary.
2. `build-local/minimax-m2-api-check-01`:191 unit/regression methods прошли,
   но replay менял `scripts[0]` без активного single-script reference `script`.
   Из-за этого возвращался прежний тестовый текст. Исправлен harness; parser,
   native engine и model output не подменялись для получения PASS.

Target: `H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf`.
Header SHA `9a011b2ad6fb20db3bb0cdfe9af08665939ee1f0cb0609ffe2bfee7f31c825db`,
template SHA `893d908f7b5cc65fdde270dcae5ea1a99647c6a7ce572ae874a57b7160069566`.
Использованы прежние CPU oracles dependency86ebfef2, MSVC19.44.35222.0 Release,
с отдельным JSON float-roundtrip patch MM27-19. GPU kernels, weights, cache,
pipeline и sampling defaults не менялись. Новых измерений tokens/s нет.

```powershell
python -X utf8 -m unittest serve.test_minimax_m2_api -v
python -X utf8 tools/check_minimax_m2_api.py --gguf H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --out build-local/mm27-new-api --replay build-local/minimax-m2-sampling-output-input-01.json
```

`--out` требует нового каталога. Требуются CPU oracles, созданные в MM27-19,
и jsonschema (на данном ПК4.26.0). Listener тестов привязан только к127.0.0.1
с автоматически выбранным свободным портом и закрывается после проверки.

Следующая задача: адаптировать `strata-minimax-m2-bench --pipe` к Engine protocol —
ready/token/result/error, admission/provenance, sampling, закрытие/drain при EOS,
cancel и restart. После этого подключить изолированный startup profile и проверить
на GPU оба API, настоящий model toolcall → result → answer и web chat. P2 в целом
остаётся PARTIAL; quality FAIL MM27-18b, independent model oracle и P4/P6 не закрыты.
