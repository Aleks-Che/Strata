# MM27-20: потоковый разбор сгенерированных tool calls

Дата: **2026-10-09**, `Asia/Yekaterinburg`; ревизия
`d342001b4dca70860bcd585f28004591a83e5bf3`, dirty tree.
[План](MINIMAX_M27_IMPLEMENTATION_PLAN.md),
[статус](MINIMAX_M27_IMPLEMENTATION_STATUS.md).

Добавлен отдельный `serve/minimax_m2_tools.py`: `MiniMaxOutputParser` объединяет
прежний reasoning parser и разбор `<minimax:tool_call>` / `<invoke>` /
`<parameter>`. Это реализация P2.4 для проверенного формата. Регистрации в
HTTP server и исполнения инструментов пока нет. GPU engine и defaults не менялись.

## Контракт

- Template уже содержит открывающий `<think>`. До первого `</think>` весь
  вывод остаётся reasoning; tool-теги внутри него не создают calls. Поздние
  think-теги сохраняются буквально. Пробелы и переводы строк не удаляются.
- При переданном списке tools parser распознаёт точную форму тегов native
  template с двойными кавычками в атрибутах. При пустом списке tool-теги
  остаются обычным текстом.
- Группа выдаётся как последовательность `Event('tool_call', call=ToolCall(...))`
  только после закрывающего тега и проверки **всех** invokes. Каждый получает
  собственный ID. Частичных arguments/tool deltas нет. Parser сам tools не запускает.
- Имена вызовов должны быть объявлены, имена аргументов — присутствовать в
  `parameters.properties`. Повторный параметр, неизвестное имя, лишний текст
  внутри framing, незакрытая или невалидная группа не создают вызовов.
- Строки идут буквально: `42`, `null`, кавычки, пробелы, `&amp;`, `&lt;`,
  обычный `<xml>` не преобразуются. Остальные типы разбираются как JSON,
  без повторяющихся ключей, NaN/Infinity, переполнения double и целых вне signed64.
  Вложенные значения проходят те же проверки UTF-8/JSON, что входная history.
- После разбора проверяется JSON Schema draft2020-12: required, вложенные
  types/items/properties, additionalProperties, enum/const, численные ограничения
  и другие assertion keywords валидатора. `format` остаётся annotation:
  FormatChecker не включён. Внешние references запрещены, registry не загружает
  сеть; локальные references проверены. Defaults из schema не подставляются.
- Для wire-параметра нужен явный `type` непосредственно в `properties`.
  Описание типа только через `$ref`/`anyOf`, динамические имена параметров
  через additionalProperties и другие schema dialects пока не поддерживаются.
  Union со string, допускающий одновременно raw string и JSON-значение
  (например, `null` при string|null), оставляется текстом без угадывания типа.
- Структурные tool-теги в значениях неоднозначны: native template их не
  экранирует. Такой payload оставляется текстом. XML entity decoding не
  добавлен, иначе значение `&amp;` перестало бы совпадать с исходной строкой.
  JSON escape, который превращается в структурный тег после декодирования,
  также не допускается; это соответствует контракту canonical history.
- При ошибке группа выводится буквально и до конца данного ответа parser
  больше не распознаёт tools. Это исключает повторное распознавание вложенной
  части ошибочной группы как самостоятельного вызова. Причина доступна в
  `tool_error`; уже завершённые предыдущие группы не отзываются.
- `<response>` относится к tool-result history. В сгенерированном assistant
  output это буквальный текст, а не результат инструмента. ID/result correlation
  и порядок `<response>` выполняет [history normalizer](MINIMAX_M27_HISTORY.md).

## Streaming и завершение

`feed(str)` принимает текст после строгого incremental UTF-8 decoder. Разрывы
байтов внутри Unicode не заменяются символом замены. EOS/PAD200020 должен
обрабатывать caller по ID; parser не удаляет буквальные `[e~[`/FIM/reponame.

`stops` применяется к raw text до разбора reasoning/tools. Выигрывает первая
полностью полученная последовательность; при совпадении позиции её конца —
порядок в списке. Stop и последующий текст не выдаются. Stop внутри группы
не создаёт частичный вызов. `finish()` идемпотентен, сохраняет незавершённые
reasoning/tag/group tails; он не добавляет закрывающих тегов. После finish
новые feed запрещены. Для cancel/length/EOS caller вызывает этот же flush;
API finish_reason и политика отправки уже завершённых calls при отмене — P2.6.

Буфер одной группы ограничен **1 048 576 символами**, параметр `max_group_chars`.
При превышении весь остаток становится текстом, без дальнейшего tool parsing.
Группа накапливается в StringIO; при новом delta сканируются новые символы и
возможный хвост тега. Большой delta обрабатывается частями по4096 символов.
Stop buffer отдельно хранит возможный префикс stop-последовательности.

Для tool schemas требуется установленный `jsonschema>=4.23,<5`; отсутствие
даёт явную ошибку до generation. Без tools зависимости нет. На этом ПК
проверено с jsonschema4.26.0. Общий requirements/installer пока не менялся:
проверку capability/установку зависимости нужно подключить вместе с API profile.

## Измеренные проверки

Windows, CPU; прежние Release CPU oracles MSVC19.44.35222.0,
dependency `86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9`.

- **51 unittest methods PASS:**18 новых parser,20 history,7 reasoning,
  6 tokenizer/input. Проверены все символьные/UTF-8 разрывы выбранных fixtures,
  все префиксы незавершённой группы,29 malformed groups, schema constraints,
  atomicity, stop ordering, размер буфера и IDs/result history.
- **281 native tool groups +281 native history prompts PASS**: типизированный
  multi-call,12 строковых примеров и268 числовых случаев, включая крайние
  значения и воспроизводимые конечные IEEE-754 bit patterns. Replay выполнялся
  целиком, по token bytes и по одному байту. Сравнение JSON-значений различает
  `-0.0`, `0.0`, int и double; **22 079 token IDs** совпали с native tokenizer.
- **8 сохранённых реальных completions** MM27-18b повторно разобраны по tokens
  и по bytes, с tools и без tools: reasoning/content сохранены точно, случайных
  calls нет. Исходные reports сверены по hashes. Это replay, не новая генерация.
- Итого **3966 report checks PASS**, отдельно от51 unit methods, exit0.
  Истории с полученными parser IDs, JSON-string arguments и обратным порядком
  результатов дали ожидаемый порядок `<response>` и native/Python prompt parity.

Tool-output corpus создан native template oracle из заданных calls, **не LLM**.
Живую способность модели выбирать tools и полный API cycle этот результат не
доказывает. Model quality FAIL MM27-18b и independent model oracle остаются gates.

Использован локальный target `H:\models\MiniMax-M2.7`:
header SHA `9a011b2ad6fb20db3bb0cdfe9af08665939ee1f0cb0609ffe2bfee7f31c825db`,
template SHA `893d908f7b5cc65fdde270dcae5ea1a99647c6a7ce572ae874a57b7160069566`.
Weights payload не читался, CUDA inference и новый speed benchmark не запускались.

## Артефакты и повторение

[Полный отчёт](MINIMAX_M27_TOOLS_CHECK.json),
[hashes и проверка неизменности прежних файлов](MINIMAX_M27_TOOLS_VALIDATION_CHECK.json).
Финальная копия sources/EXE, native fixtures/token IDs/history и unit log:
`build-local/minimax-m2-tools-check-02`; первоначальный проход01 сохранён.
Native template-json oracle использует ранее проверенный float-roundtrip patch
MM27-19; raw oracle не является эталоном точности этих numeric fixtures.

```powershell
python -X utf8 -m unittest serve.test_minimax_m2_tools serve.test_minimax_m2 serve.test_minimax_m2_history tools.test_minimax_m2_tokenizer -v
python -X utf8 tools/check_minimax_m2_tools.py --gguf H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --out build-local/mm27-new-tools --replay build-local/minimax-m2-sampling-output-input-01.json
```

`--out` требует нового каталога; `--replay` опционален, без него real-output
regression пропускается. Native EXE должны быть собраны по MM27-19.
Следующий пункт — P2.5/P2.6: изолированный API adapter, живой toolcall → result →
answer, JSON/SSE OpenAI и Anthropic, usage/finish_reason, cancel/recovery.
