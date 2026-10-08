# MM27-18: длинные ответы, EOS и потоковый reasoning

Дата: **2026-10-08**, Windows, RTX 5090 32 ГиБ, RAM125,555 ГиБ.
[План](MINIMAX_M27_IMPLEMENTATION_PLAN.md),
[статус](MINIMAX_M27_IMPLEMENTATION_STATUS.md).

В экспериментальном benchmark/JSONL движке снят жёсткий предел256 output
tokens. `max_tokens` теперь принимает целое число от1 до свободного места
после prompt в текущем контексте. Default8 и предел context4096 сохранены.
Переполнение контекста, bool/float/string/null и недопустимые token IDs
отклоняются до prefill. Бюджет включает reasoning и EOS.

`result` содержит запрошенный `max_tokens` и `stop_token_id`:200020 при EOS,
null при остановке по длине. Stop policy не менялась: FIM/reponame aliases
200004/200005 не завершают разговор. Raw text и token events сохраняют EOS;
будущий adapter должен поглощать его по ID, не удалять похожую строку из текста.

Добавлен Windows SIGBREAK для адресной отмены созданной process group.
Cancellation проверяется перед выдачей токена и внутри существующей доставки
экспертов. После ошибки JSONL loop очищает KV и принимает следующий запрос.
Математика, веса, sampler, настройки кешей и глобальный95%-лимит не изменены.

## Потоковое разделение текста

`serve/minimax_m2.py` содержит `MiniMaxReasoningParser`. Он начинает в reasoning:
открывающий `<think>` уже напечатан встроенным template. Первый `</think>`
переключает поток в content. На границе порций удерживается не более7 символов;
пробелы и переводы строк сохраняются. `finish()` возвращает незавершённый
суффикс как reasoning, `reasoning_complete` остаётся false при отсутствии
закрывающего тега. Искусственный финальный ответ не добавляется.

Caller должен использовать stateful UTF-8 decoder. Тесты проверяют каждую
позицию разрыва тега, UTF-8, доставку по одному байту, незавершённые теги,
похожие литералы и повторный finish. Tool-like текст остаётся текстом;
модуль **не зарегистрирован в API**, не является tool parser или переключателем
no-thinking. P2 history/tools/OpenAI/Anthropic integration остаётся отдельно.

13 unittest methods PASS:7 новых parser tests и6 существующих tokenizer/input
tests. Replay реальных2364 token IDs против native text и через token-sized /
single-byte UTF-8 chunks:18 checks PASS. У незавершённого русского ответа
сохраняется reasoning и пустой content.

## Полная модель: исходный корпус с budget1536

Target: `H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf`,
138342384352 байта. Header SHA-256
`9a011b2ad6fb20db3bb0cdfe9af08665939ee1f0cb0609ffe2bfee7f31c825db`;
полный checksum весов не вычислялся. Dependency86ebfef2, CUDA13.0.48,
MSVC19.44.35222.0, driver581.80, architecture120/120a.

Strict F32 activations/KV, greedy, ctx2048/batch16, GPU cache18 ГиБ,
arena64/reserve0, file2/chunk4/lookahead/D2D batch. RAM cache, grouping,
FA/graphs/MTP/DFlash выключены, `STRATA_MM27_TOKENWISE=0`.
KV очищается между запросами, GPU cache остаётся. Один процесс и один прогон;
OS file cache и внешняя нагрузка не контролировались. GPU tests шли последовательно.

| Запрос | Prompt / output tokens | Остановка | Decode, токенов/с | TTFT, с | Request, с |
|---|---:|---|---:|---:|---:|
| Почему небо голубое, английский | 52 /414 | EOS, reasoning закрыт, final есть | 4,170 | 12,973 | 112,252 |
| Точный повтор | 52 /414 | EOS, тот же полный ответ | 4,248 | 10,138 | 107,592 |
| Сумма списка1..96, русский | 343 /1536 | length, reasoning не закрыт | 4,757 | 46,695 | 370,233 |

Decode измеряет только forward для N−1 токенов; первая выдача входит в
prefill. Sampling, запись logits и внешние memory samples исключены из
decode interval, но включены в request time. Load отдельно:1,183 с.
Это **не A/B ускорения**; метод измерения отличается от useful throughput DFlash.
Глобальные sampled peaks: RAM24,89%, VRAM84,14%; guards не сработали.

Все первые256 токенов каждого запроса и153649152 logits побитно совпали
с сохранённым MM27-12 reference. Повтор английского ответа совпал по всем414
IDs и82826496 logits. Все472951296 logits полного прогона конечны;
это не означает сравнения всех них с независимым эталоном.

**Общий natural-completion gate FAIL.** Английский запрос завершился естественно;
русский33 раза повторил фразу `We can also show that the sum of numbers from 1
to 96 is 4656.` внутри reasoning и исчерпал бюджет. Число4656 вычислено в
reasoning, но пользовательского финального ответа нет. Исходные запросы,
неудачный отчёт, полный текст и logits сохранены. Этот результат нельзя
объявлять PASS на основании исправного EOS в другом запросе.

## JSONL: границы бюджета и отмена

Отдельный загруженный процесс прошёл17 сценариев:13 некорректных запросов,
default8 после ошибок, точный context budget1996 (prompt52 + output budget1996),
адресная отмена после257 выданных токенов и полный запрос после отмены.
При budget1996 ответ сам завершился на414-м токене; EOS встретился ровно один
раз и совпал в token events, result IDs, raw text и stop_token_id.

CTRL_BREAK был отправлен только process group тестового engine. Получена
ошибка `MiniMax decode failed: 2 MiniMax request cancelled`; следующий запрос
без перезапуска дал те же414 IDs и raw text, что reference. Pipeline/bytes/
memory/drain checks прошли. Pipe не экспортирует logits: численное сравнение
здесь ограничено token IDs и текстом. Сигнал и ответ попали в один tick
`time.monotonic` (отчёт0 мс); точная задержка отмены этим замером не установлена.
Это проверка адресного OS signal, не физического нажатия Ctrl+C в терминале.

## Границы результата

Отдельный serial ReadFile/H2D прогон русского запроса, GPU/RAM caches и pipeline
выключены, budget1024, те же ctx2048/batch16 и greedy/F32: все1024 IDs и
204865536 logits **побитно совпали** с префиксом кешированного запуска.
Та же фраза повторилась14 раз; reasoning не закрылся, остановка length.
GPU-only, bytes, memory95 и serial-path gates PASS. Это исключает добавленные
GPU cache/file pipeline как объяснение повтора в проверенном префиксе.
Это не независимый model/framework oracle: loader, CUDA math и sampler общие.
Низкая скорость этого диагностического режима не является A/B рабочего профиля.

Полный корпус ещё не прошёл проверку завершённых ответов. Больший output
budget не является исправлением повторов; смена sampler требует отдельного
сравнения качества и воспроизводимости. Профиль inference по умолчанию
не утверждён. Старый MM27-06 tiny reload discrepancy, P2 tools/API,
P4 sessions и DFlash live lifecycle остаются открытыми.

## Артефакты

- [Исходный completion gate — FAIL](MINIMAX_M27_COMPLETION_CHECK.json):
  английские EOS PASS, русский length/repetition FAIL; численные gates PASS.
- [17 live JSONL scenarios](MINIMAX_M27_COMPLETION_PIPE_CHECK.json),
  [13 unit methods](MINIMAX_M27_COMPLETION_UNIT_CHECK.json),
  [18 replay checks реальных token bytes](MINIMAX_M27_COMPLETION_OUTPUT_CHECK.json).
- [Serial reference — numerical PASS, repetition сохранён](MINIMAX_M27_COMPLETION_REFERENCE_CHECK.json).
- [Build manifest](MINIMAX_M27_COMPLETION_BUILD_MANIFEST.json),
  [source/EXE/report hashes и сводные gates](MINIMAX_M27_COMPLETION_VALIDATION_CHECK.json).

Новый bench SHA-256
`6da69ae31c6ed936b1bcbbee07b8beb038f1faf170f262b9a0717e523f474680`.
Снимок до изменений: `build-local/minimax-m2-completion-baseline-01`,188 hashes
проверены. Новый EXE и исходники дополнительно сохранены в
`build-local/minimax-m2-completion-evidence-01`. Shared common/Step/Hy3/GLM и
существующие MiniMax support sources не менялись; из runtime изменён только
`main.cpp`. Logits полного corpus и serial reference, JSONL token events,
команды и stderr сохранены в `build-local/minimax-m2-completion-*`.

Validation JSON намеренно сохраняет `pass=false` и
`natural_completion_gate_pass=false` при `implementation_checks_pass=true`.
Отрицательный quality result не скрыт успешными infrastructure checks.

## Воспроизведение

Из настроенной CUDA/MSVC среды:

```powershell
cmake --build build-local/minimax-m2-cuda --target strata-minimax-m2-bench -j 6
python -m unittest serve.test_minimax_m2 tools.test_minimax_m2_tokenizer
python tools/check_minimax_m2_completion.py --model H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --out build-local/mm27-new-completion --reference-report build-local/minimax-m2-tune-long-01/01-baseline-c18-r2-s4.json --reference-logits build-local/minimax-m2-tune-long-01/01-baseline-c18-r2-s4.f32
python tools/check_minimax_m2_output.py --report build-local/mm27-new-completion/generation.json --out build-local/mm27-new-output
python tools/check_minimax_m2_completion_pipe.py --source build-local/mm27-new-completion --out build-local/mm27-new-pipe
python tools/check_minimax_m2_completion_reference.py --source build-local/mm27-new-completion --out build-local/mm27-new-serial-reference --tokens 1024
```

Каждый output directory должен быть новым. Completion driver возвращает1,
если хотя бы один запрос не завершился EOS с закрытым reasoning и final.
Pipe checker использует только проверенный английский ответ, сохраняет
отрицательный статус русского корпуса и не посылает сигналы посторонним
процессам. Его Windows supervisor запускается со скрытым console window.
