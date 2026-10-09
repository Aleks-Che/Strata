# MM27-23: живые tools обоих API и web chat

Дата: **2026-10-09**, `Asia/Yekaterinburg`; commit
`54467693c8354d8c64a72ec1cb083473a529b4ba`, dirty tree.
[План](MINIMAX_M27_IMPLEMENTATION_PLAN.md),
[статус](MINIMAX_M27_IMPLEMENTATION_STATUS.md).

Этот этап продолжает [native API MM27-22](MINIMAX_M27_NATIVE_API.md).
Модель, EXE, CUDA kernels, sampler и cache/pipeline не меняются. Проверяется
полный путь от JSON/SSE и браузера до настоящей генерации модели.

## Исправления

1. Одинаковые определения tools в OpenAI и Anthropic имели разный порядок
   полей `description`/`parameters`. Native template сохраняет insertion order,
   поэтому различались prompt и token IDs. В MiniMax API adapter добавлен
   единый порядок `name`, `description`, `parameters`, `strict` после проверки
   входных полей. Порядок tools и properties внутри schema сохраняется;
   неизвестные поля по-прежнему отклоняются. Входной request не мутирует.
2. Web chat всегда отправлял `X-Strata-Session-Id`. Native MiniMax не
   поддерживает session cache и отклоняет этот header до generation.
   `/health` и `/api/health` теперь сообщают `session_id`; браузер не отправляет
   header при явном `false`. Для старых серверов без поля поведение сохранено.
   API продолжает отклонять явно переданный header неподдерживающему движку;
   фиктивные sessions/prefix reuse не включаются.

Первый CPU cross-API prompt test обнаружил порядок полей, исходный FAIL
сохранён в `build-local/minimax-m2-live-tools-client-tests.log`.
Диагностический `build-local/minimax-m2-live-tools-01` завершил первый
Anthropic SSE error cycle; затем принадлежащий проверке процесс остановлен,
чтобы не продолжать сравнение на старом adapter. Причина и вышедшие дочерние
процессы записаны в `INTERRUPTED.json`. Этот проход не считается полным PASS.

## Live corpus

`tools/check_minimax_m2_live_tools.py` использует одну задачу во всех четырёх
комбинациях OpenAI/Anthropic × JSON/SSE. Модель должна сама вызвать
`get_code(city="Oslo", revision=2)` и `get_code(city="Уфа", revision=3)`
до ответа. Аргументы проверяются на точные значения и integer type;
IDs двух calls должны быть уникальными.

Клиент возвращает результаты в обратном порядке. JSON получает оба кода:
OSLO-4179 и UFA-9264. SSE получает первый код и ошибку CODE_UNAVAILABLE для
Уфы: Anthropic `is_error=true`, OpenAI content `Error: CODE_UNAVAILABLE`.
Prompt заранее требует сообщить ошибку без retry/выдуманного кода. Проверяются
финальный текст, stop/tool finish, EOS, input/output usage, GPU-only, bytes,
drain и global memory95%. Внешние инструменты или сервисы не вызываются.

Отдельный SSE reader собирает chunks/blocks и проверяет порядок, завершение,
call IDs и JSON arguments. Он не использует server-side collectors. Полные
requests, prompts/IDs, wire responses, collected outputs, native results,
EXE и source snapshots сохраняются. Для одинаковых первых запросов сравниваются
все prompt и generated IDs между четырьмя режимами, с T1/p0,95/k40/seed42.

Это ограниченный corpus: одна двухинструментальная задача, два вида результатов,
кириллица и integer argument. Он не доказывает общую model answer quality,
поддержку произвольных schemas, инструментов или interleaved tool/text history.

## Результаты API и CPU

Завершённый matrix: `build-local/minimax-m2-live-tools-02/report.json`, exit0,
**4 cycles /8 requests PASS**. Во всех четырёх первых запросах совпали278 prompt
и110 generated IDs. Между API также совпали полные follow-up:421/98 IDs для
двух успешных результатов и422/93 IDs для результата с ошибкой. Все восемь
генераций завершились по EOS; usage, GPU-only, bytes, drain и memory95% PASS.

| API / режим | Prefill calls / answer, с | Decode calls / answer, токенов/с |
|---|---:|---:|
| Anthropic SSE, ошибка | 55,692 /62,268 | 3,850 /3,788 |
| OpenAI JSON, оба кода | 44,833 /60,646 | 3,920 /3,833 |
| Anthropic JSON, оба кода | 45,384 /60,800 | 3,891 /3,836 |
| OpenAI SSE, ошибка | 44,633 /60,950 | 3,902 /3,867 |

Это последовательные запросы одного процесса; cache state меняется. Decode
учитывает generated reasoning/tool syntax/EOS, а не только видимый final.
Разность строк не является сравнением производительности API.

CPU regressions:191 методов полного API checker,58 focused методов и17
session/lifecycle методов; с учётом пересечения **213 различных методов PASS**,
без skipped. Дополнительно16 scripted cycles,32 HTTP completion replays,
272 report checks и32 native prompts /14284 IDs PASS. Логи:
`build-local/minimax-m2-api-regression-23/unit.log`,
`build-local/minimax-m2-live-tools-regression.log`,
`build-local/minimax-m2-web-capability-tests.log`.

## Результаты браузера

Отдельный live server: `build-local/minimax-m2-web-check-01`. В обычном web UI
выполнены пять запросов. Первые два завершились ответами `SAVED` и `TEST-4821`:
история передана без session cache, reasoning отделён от final. Третья генерация
остановлена кнопкой Stop во время reasoning. Четвёртая начала новый поток,
затем также остановлена. Пятая завершилась ответом `4`. Обе отмены подтверждены
native cancellation event; все пять запросов использовали CUDA PID58372,
процесс остался жив после каждой генерации/отмены. JS warnings/errors:0.

Оговорка: Stop оставляет незавершённый user request в истории. На короткий
четвёртый вопрос `What is 2+2? Reply with the number only.` модель начала
рассуждать об исполнении обоих заданий, включая прежний список100 применений
карандаша. Он остановлен без final/EOS и **не считается успешным ответом4**.
Пятый запрос явно отменял старое задание и получил4. Поведение истории не
менялось; эта проверка подтверждает transport recovery, а не игнорирование
старого вопроса после Stop или общую answer quality.

| UI запрос | Prompt / generated IDs | Prefill, с | Native decode, токенов/с |
|---|---:|---:|---:|
| Запомнить код, ответ SAVED | 51 /56 | 12,353 | 4,013 |
| Вернуть код из истории | 76 /41 | 11,341 | 4,112 |
| Ответ4 после явной отмены старого задания | 168 /155 | 27,258 | 4,322 |

UI default sampler: T0,6/p0,95/k20, seed не передан (native42), reasoning High.
Это короткие последовательные запросы, не performance A/B. UI показывает
собственный округлённый stream timing; таблица использует native decode.
Три полных ответа прошли EOS/GPU-only/bytes/drain/memory95%. У cancelled
запросов нет terminal result stats; их peak/drain здесь не измерены отдельно.
После STOP-файла helper и native worker завершились, exit0.

В memory samples восьми tool requests и трёх полных UI requests максимум
global RAM32,436ГиБ (25,834%), VRAM27720,926МиБ (85,015%); process private
26,236ГиБ и working set2,972ГиБ. Это sampled значения, не непрерывный OS trace.
95% — верхний предел, этот этап не подбирает увеличенный RAM/VRAM cache budget.

Сводный [CHECK](MINIMAX_M27_LIVE_TOOLS_WEB_CHECK.json) проверяет источник
каждого snapshot, parity, сохранённые результаты и logs. Raw evidence:
`build-local/minimax-m2-web-check-01/evidence.json`, `ui-observations.json`,
`final-ui.txt`, `chat-history.jpg`, `chat-recovery.jpg`.
Повторная проверка сохранённых данных, без загрузки модели:
`python -X utf8 build-local/minimax-m2-live-tools-web-validation.py` (exit0).

## Воспроизведение

```powershell
python -X utf8 -m unittest -v tools.test_minimax_m2_live_tools serve.test_minimax_m2_api serve.test_minimax_m2_tools serve.test_minimax_m2_history
python -X utf8 -m unittest -v serve.test_session_cache serve.test_lifecycle
python -X utf8 tools/check_minimax_m2_live_tools.py --gguf H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --out build-local/mm27-new-live-tools
python -X utf8 tools/check_minimax_m2_api.py --gguf H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --out build-local/mm27-new-api-regression --replay build-local/minimax-m2-sampling-output-input-01.json
```

Для browser проверки запускать отдельно, после завершения другого GPU checker:

```powershell
python -X utf8 tools/serve_minimax_m2_web_check.py --gguf H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --out build-local/mm27-new-web-check
```

Открыть URL из `ready.json` в браузере и использовать обычный UI. Helper
сохраняет запросы, наличие session header и native records в `evidence.json`.
Он слушает только 127.0.0.1, не запускает браузер и не подменяет engine output.
Создание пустого файла `build-local/mm27-new-web-check/STOP` отменяет активную
работу и закрывает owned процессы. Дополнительный предел жизни —30 минут.
Настоящий пользовательский запуск остаётся `python -m serve.minimax_m2_server`.

Контекст2048/batch16, cache18ГиБ, arena64/reserve0, file reader, readers2/chunk4,
lookahead/D2D batch on, RAM cache0, strict F32, FA/graphs/MTP/DFlash off.
RTX5090/32607МиБ, driver581.80, RAM125,555ГиБ. Веса и EXE закреплены теми же
hashes, что в MM27-22. Во время live проверок выполнялись CPU regressions;
эти времена не являются изолированным performance A/B или основанием менять defaults.

## Оставшиеся gates

Следующий этап — HTTP context4K/pressure/recovery в native Engine. Проверить
запрос у границы контекста, адресную отмену при внешней нагрузке и успешный
следующий запрос в том же процессе с global RAM/VRAM≤95%.

Installer/общий config registry, независимый full-model oracle, старые numerical
и answer-quality FAIL, HTTP pressure/4K, sessions, DFlash serving и полезное
ускорение остаются отдельными задачами. Server-side MCP и no-thinking не
подключены; web chat проверяется как текстовый клиент с обычной историей.
