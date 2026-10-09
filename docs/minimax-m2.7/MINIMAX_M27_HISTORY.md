# MM27-19: история диалога и соответствие результатов инструментам

Дата: **2026-10-09**, `Asia/Yekaterinburg`; ревизия при фиксации
`d342001b4dca70860bcd585f28004591a83e5bf3`, dirty tree.
[План](MINIMAX_M27_IMPLEMENTATION_PLAN.md),
[статус](MINIMAX_M27_IMPLEMENTATION_STATUS.md).

Добавлен `serve/minimax_m2_history.py`: `prepare_minimax_context(request)`
создаёт отдельную нормализованную копию истории, `MiniMaxChatTemplate(source)`
проверяет hash и рендерит исходный GGUF template. Это P2.3 и входная часть P2.5.
Модуль пока не зарегистрирован в сервере и не разбирает сгенерированные calls.
Существующий oracle helper `tools/minimax_m2_template.py` сохраняет свой
прежний контракт для исторических fixtures и benchmark drivers.

## Контракт истории

- Несколько начальных `system`/`developer` сообщений объединяются в порядке
  поступления через два перевода строки в единственный system slot шаблона.
  Отдельная иерархия двух ролей в формате модели не представлена. Поздние
  инструкции отклоняются явно; они не переносятся к началу и не превращаются
  в user text. Неизвестные поля сообщения тоже дают ошибку.
- `current_date`/`current_location` принимаются как строки в начальных
  инструкциях или в контексте. Одинаковые значения допустимы, конфликтующие
  отклоняются. Часы/местоположение машины автоматически не подставляются.
  При отсутствии непустого system content сохраняется встроенная identity;
  её можно явно заменить строкой `model_identity`.
- Сохраняются текстовые части, Unicode, пробелы и переводы строк. Media и
  непонятные части не пропускаются молча. `content=null` допускается только
  у assistant и становится пустым текстом.
- Inline reasoning извлекается только из начального `<think>…</think>`,
  при отсутствии явного `reasoning_content`. Разделение идёт по первому
  закрывающему тегу без удаления whitespace; остаток final сохраняется целиком.
  Явное поле имеет приоритет, а content при нём трактуется буквально.
  Незакрытый inline block отклоняется. Reasoning до последнего user удаляется
  из нормализованной копии; последующий остаётся, в том числе вокруг tools.
- Поле `reasoning_content` присутствует у каждого нормализованного assistant,
  даже когда пусто. Это отключает fallback исходного Jinja, который делит
  content по последнему `</think>` и может потерять текст между тегами.
  Буквальные теги в final не становятся управляющими.
- `enable_thinking=true` допустим; false отклоняется: встроенный template
  всегда печатает thinking prefix. No-thinking profile не реализован.

## Tool history

Принимаются nested OpenAI function calls и flat calls. Каждый требует
непустой уникальный ID, имя и arguments-object; JSON-string arguments
разбираются без допуска повторяющихся ключей. Canonical context сохраняет ID
и типы JSON, включая вложенные массивы/объекты, bool и null. Отрицательный
ноль и точность конечных double сохраняются. Целые ограничены signed64;
большие значения, NaN/Infinity и циклы отклоняются до рендера.

Результат обязательно содержит `tool_call_id`. Неизвестные, повторные,
устаревшие ID и несовпадение явно переданного имени отклоняются. Несколько
результатов буферизуются и располагаются в порядке исходных calls: GGUF
template не печатает ID/имя в `<response>`, поэтому порядок — единственная
связь в serialized prompt. Проверены все6 перестановок трёх результатов
вызовов одной и той же функции и несколько последовательных tool rounds.

До получения всех результатов нельзя продолжить разговор или добавить
generation prefix. При `add_generation_prompt=false` разрешён transcript,
который заканчивается calls без единого результата. Частично полученная
группа не рендерится: это исключает тихую потерю отложенных результатов.

Function names ограничены `[A-Za-z_][A-Za-z0-9_.-]*`; имена параметров не
содержат кавычки, `<>&` или управляющие символы. Структурные tool-теги в
arguments/results отвергаются, поскольку исходный шаблон не экранирует их.
Обычный текст вроде `<xml>` допустим. Формат экранирования и parser таких
payloads остаются задачей P2.4. Текущие definitions валидируются как function
objects с object schema; выполнение JSON Schema для arguments не реализовано.
Исторические calls могут ссылаться на инструменты, отсутствующие в текущем
списке доступных definitions.

## Исправление отдельного native-эталона

Raw native Jinja ревизии86ebfef2 сериализует float через `ostringstream`
с default precision. Probe выявил `0.12345678901234566 → 0.123457` и
`-0.0 → -0`. Целые вне signed64 также переполняются/теряют точность.
Исходный отрицательный probe сохранён; это дефект CPU template oracle,
не расхождение logits или inference.

Добавлен отдельный target `strata-minimax-m2-template-json`. Он использует
сгенерированную копию `jinja/value.cpp`: только ветка JSON float заменена
на `common_json(double).dump()`, а два quoted include квалифицированы путём
`jinja/…`. Перед генерацией проверяется SHA исходного файла. Исходники
dependency и прежний target `strata-minimax-m2-template` остаются без этой
поправки. Новый `--version` явно отмечает `jinja-json-float-roundtrip`;
build manifest содержит original/generated/patch hashes.

Первый build не прошёл: добавление `common/jinja` в include search path
подменило системный `string.h`. Исправлено квалифицированными includes
без изменения общего search path. Неудачные CMake/log сохранены. Финальная
сборка обоих CPU template targets и проверка завершились с **exit0**.

## Проверки

На Windows, MSVC19.44.35222.0, Release, pinned dependency86ebfef2:

- **33 unittest methods PASS:**20 новых history,7 reasoning parser,6 прежних
  tokenizer/input tests. Проверяются также отсутствие мутации и идемпотентность.
- **23 вручную заданных prompts** совпали побайтно.
- **293 prompts /2074 checks /71189 token IDs PASS** против исправленного
  native Jinja и native tokenizer. Среди них268 числовых случаев: крайние
  значения и256 воспроизводимых finite IEEE-754 bit patterns, одновременно
  в scalar/array/object arguments и JSON Schema defaults.
- **96 legacy template cases PASS:** исходный raw native, новый native и
  прежний Python renderer выдали одинаковые prompts на старом корпусе.
- **114 baseline files проверены.** Из прежних файлов менялись только
  `.gitattributes`, MiniMax CMake/manifest/template oracle. Существующие
  serving/normalization helpers, GPU math, cache/pipeline и runtime не менялись.
  SHA bench остаётся `07844c0345c12f78ded9f24d7f86154c9f360bfb574e96569e8694489268b0ef`.

Fixture `serve/fixtures/minimax_m27_chat_template.jinja` побайтно извлечён
из локального GGUF и защищён от Git newline conversion. SHA:
`893d908f7b5cc65fdde270dcae5ea1a99647c6a7ce572ae874a57b7160069566`.
Использован target из `H:\models\MiniMax-M2.7`; header SHA:
`9a011b2ad6fb20db3bb0cdfe9af08665939ee1f0cb0609ffe2bfee7f31c825db`.
Читались metadata/vocabulary, GPU inference и speed benchmark не запускались.

## Артефакты и воспроизведение

- [Unit/golden/native/tokenizer checks](MINIMAX_M27_HISTORY_CHECK.json).
- [Исходный raw numeric probe с расхождениями](MINIMAX_M27_HISTORY_RAW_NUMBER_PROBE.json).
- [CPU build manifest](MINIMAX_M27_HISTORY_BUILD_MANIFEST.json),
  [сводная проверка и hashes](MINIMAX_M27_HISTORY_VALIDATION_CHECK.json).

Финальные EXE, исходники, контексты, полные native prompts/token IDs и unit log:
`build-local/minimax-m2-history-check-04`. Предыдущие запуски01–03 сохранены;
baseline — `build-local/minimax-m2-history-baseline-01`. Build logs находятся
в `build-local/minimax-m2-history-*.log`, helper — `build-minimax-m2-history.bat`.

Из настроенной MSVC среды, новый output directory для каждого запуска:

```powershell
cmake -S backends/minimax_m2 -B build-local/minimax-m2-oracles -G Ninja -DCMAKE_BUILD_TYPE=Release "-DSTRATA_MM27_ARCHIVE=C:/work/git/my-repos/Strata/build-local/llama-glm-86ebfef2.tar.gz"
cmake --build build-local/minimax-m2-oracles --target strata-minimax-m2-template strata-minimax-m2-template-json strata-minimax-m2-tokenizer -j 6
python -X utf8 tools/check_minimax_m2_history.py --gguf H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --out build-local/mm27-new-history
```

Следующий пункт — P2.4: streaming parser сгенерированных invoke/parameter,
schema types, порванные теги/UTF-8, несколько calls и truncated output.
Затем нужны API IDs/result correlation и OpenAI/Anthropic JSON/SSE adapters.
MM27-18b answer-quality FAIL, independent model oracle, P4 sessions и
DFlash live lifecycle остаются открытыми. Этот PASS не закрывает их.
