# MM27-27: снимки KV неактивных сессий в RAM

Дата: **2026-10-09**, `Asia/Yekaterinburg`.
[План](MINIMAX_M27_IMPLEMENTATION_PLAN.md),
[статус](MINIMAX_M27_IMPLEMENTATION_STATUS.md).

## Контракт

Native `--prefix-cache 1 --session-cache-mib N --session-cache-slots K`
сохраняет завершённые сессии в RAM при переключении. Сервер использует
`--prefix-cache --session-cache-mib N --session-cache-slots K`.
Default: архив0МиБ, максимум4 неактивные записи, prefix cache off.
Лимиты проверяются до загрузки модели:0..131072МиБ,1..64 записи;
ненулевой архив требует prefix cache. HTTP ID — прежний
`X-Strata-Session-Id`, передаваемый в native как SHA-256.

Один GPU KV остаётся активным. Архив содержит CPU blobs
`llama_state_seq_get_data`, метаданные полного prefill и реально вычисленное
число KV positions. При возврате snapshot загружается через
`llama_state_seq_set_data`; min/max positions проверяются перед reuse.
Последний пакет prompt считается заново, serial generated tail не используется
как batched prefill. Архивируемые blobs включают хвост KV, который при reuse
удаляется. Числовые kernels, sampler и expert pipeline не менялись.

Архив создаётся для одного неизменного model/context и не может копироваться.
Он не экспортируется на диск, не импортируется и не переносится между engines.
Так model/quant, context, batch и F32 KV не могут измениться у живого snapshot.
Точное совпадение token IDs проверяется даже при том же session ID: изменение
template/history повторно использует только совпавшую часть prompt.
Restart/unload уничтожает весь архив; cache-admin API пока не реализован.

Перед сохранением освобождаются старейшие неактивные записи. Запрашиваемый
подходящий snapshot защищён от вытеснения; если вместе с ним уходящий KV не
помещается, уходящая сессия не сохраняется. Это относится и к лимиту1 запись.
CPU blob выделяется один раз, без промежуточной полной копии. Бюджет учитывает
blob, token metadata и запас256байт плюс sizeof записи для служебных данных;
это счётчик выделений архива, не RSS всего процесса.

RAM admission оставляет5% физической памяти плюс256МиБ и также проверяет
доступный commit. После каждого вытеснения доступная память считывается заново:
освобождение paged-out blob не означает такое же увеличение physical available.
Проверки и освобождение архива выполняются при переключении и перед каждым
prefill/decode batch. Существующий runtime guard продолжает проверять95% RAM
и VRAM внутри вычислений. Резкая внешняя нагрузка во время kernel/copy может
завершить запрос через guard; фонового eviction thread нет.

Восстановленный snapshot потребляется до вычисления нового запроса. После
cancel/native error активный KV очищается и новый checkpoint не создаётся;
завершённые snapshots других сессий остаются. Anonymous запрос не получает
reuse и не архивируется; перед ним можно сохранить прежнюю именованную сессию.

Native result содержит `session_restore`, `session_saved_bytes`,
`session_restored_bytes`, `session_archive_bytes`, `session_archive_entries`,
накопительные `session_archive_evictions`/`session_archive_rejected`.
Python проверяет типы, cap, число записей и согласованность restore/reuse.
`session_ms` включает snapshot/restore/clear/trim перед prefill; `prefill_ms`
измеряет вычисление prompt отдельно. `ttft_ms` и `request_ms` теперь включают
`session_ms`. Копирование KV не скрывается из полной задержки запроса.

## Проверки

RTX5090,32607МиБ, driver581.80, RAM125,555ГиБ, Windows;
ревизия `54467693c8354d8c64a72ec1cb083473a529b4ba`, dirty tree.
Полный target Q4_K_M из плана, dependency `86ebfef2`, template и веса прежние.
Context2048/batch16, F32 activations/KV, TF32/FA/graphs/MTP/DFlash off,
greedy, arena64/reserve0, GPU cache18432МиБ, file, readers2/chunk4,
lookahead/D2D on, RAM expert cache0/grouping off. Архив600МиБ/2 записи
offline и128МиБ/2 записи в HTTP; отдельный проход с cap1МиБ.

- **531 C++ checks PASS:** byte/count caps, LRU, защита запрашиваемой записи,
  pressure с повторным чтением available, overflow, allocation/write failure,
  token/batch mismatch. Pressure здесь синтетический, без внешнего holder.
  Прежние **14 430 prefix checks PASS**.
- **54 Python methods PASS:** Engine/ready/protocol accounting, API/session
  headers, invalid caps/types, supervisor/cancel, startup, context helpers и
  обнаружение одного изменённого бита/NaN/truncated logits в audit tool.
  **6 native CLI rejections PASS**, exit2 до чтения model/request files.
- **27 full-logit comparisons PASS:** четыре old/new fresh,20 сценариев
  архива, три отказа admission при cap1МиБ. Всего188 сравниваемых IDs и
  **37 612 032 F32 logits**, nonfinite0, different bits0, max abs0.
  Проверены A→B→A, LRU, anonymous/return, branch, shortening,33/52/512 prompt
  tokens и один output без decode graph. Все31 native requests завершились.
- **14 live HTTP/lifecycle scenarios PASS:**12 полных ответов /96 IDs,
  OpenAI/Anthropic JSON/SSE и cached usage, LRU/anonymous, cancel после restore,
  fresh recovery отменённой сессии, сохранность другой сессии после cancel и
  malformed native request, restart с пустым архивом. HTTP сравнивает IDs;
  полные logits проверены отдельным offline corpus.

В одном проходе,8 outputs:

| Prompt | Первый запрос, с | Возврат из RAM, с | Prefill после restore, с | Save + restore/trim, с |
| --- | ---: | ---: | ---: | ---: |
| 52 tokens | 12,629 | 5,255 | 0,666 | 3,487 |
| 512 tokens | 55,864 | 6,960 | 1,606 | 3,516 |

Это снижение задержки при возврате к диалогу, не увеличение decode tokens/s.
Порядок запросов и степень прогрева expert cache различаются; трёх повторных
A/B нет, defaults не менялись. Сериализация transposed V делает много мелких
копирований; время около3,5с при переключении — отдельная цель оптимизации.
Payload snapshot для52+7 KV равен29 968 556байт, для512+7 —263 609 916байт.

Максимумы по native memory samples обоих проходов: global RAM28,524ГиБ
(22,718%), VRAM26,932ГиБ (84,579%); process private26,479ГиБ,
working set3,216ГиБ. Это sampled values, не непрерывное измерение peaks.
Внешнее pressure в MM27-27 не создавалось. Owned native PIDs22688,6516,
61764,28320,30584,58324 завершены после проверок.

## Сборка и артефакты

Прежний EXE/source/manifest сохранены в
`build-local/minimax-m2-session-baseline`. Первая сборка не прошла из-за
отсутствующего `windows.h` в новом runtime header; исправление включено,
лог сохранён как `initial-build-fail.log`. Последующая сборка exit0.
Новый EXE допущен в Python только после offline parity, затем проверен через
обычный `MiniMaxEngine`, без обхода admission:
`275e89f8628ceecd8f0599ba92dda1abae979ea2802bfb9bf32f3669344a2ba3`.
Старый SHA:
`294ce7117876b7a112e5a29c199c8822d9f48bc333a274cbd7f6d8ff65ebbfcd`.

```powershell
& build-local/build-minimax-m2-live-api.bat
& build-local/build-minimax-m2-sessions-check.bat
& build-local/minimax-m2-cuda/bin/strata-minimax-m2-sessions-check.exe
& build-local/minimax-m2-cuda/bin/strata-minimax-m2-prefix-check.exe
python -X utf8 tools/check_minimax_m2_sessions.py --gguf H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --reference-engine build-local/minimax-m2-session-baseline/engine.exe --out build-local/minimax-m2-sessions-01
python -X utf8 tools/check_minimax_m2_sessions_http.py --gguf H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --offline build-local/minimax-m2-sessions-01 --out build-local/minimax-m2-sessions-http-01
python -X utf8 -m unittest serve.test_minimax_m2_engine serve.test_minimax_m2_server serve.test_minimax_m2_api serve.test_session_cache tools.test_minimax_m2_prefix_context tools.test_minimax_m2_http_context tools.test_minimax_m2_sessions
python -X utf8 build-local/validate-minimax-m2-sessions.py
```

Повтор GPU scripts требует новых `--out` каталогов. В offline output сохранены
оба EXE, requests/results, полные F32 logits, stdout/stderr, source snapshots;
в HTTP output — requests/responses/native results и transport logs.
`support` содержит manifest/build logs, CPU evidence и итоговые Python sources.
Сводка и hashes: [SESSION_ARCHIVE_CHECK](MINIMAX_M27_SESSION_ARCHIVE_CHECK.json).

## Остаётся

Архивирование2K/4K под реальным RAM/VRAM pressure, EOS/sampling после restore,
GPU parity при batch1/8, shift, context>4K и alternate KV dtype ещё не проверены
в этом этапе. Прежние MM27-26 PASS относятся к одному resident prefix.
Independent full-model oracle, quality gates и installer/P6 остаются открытыми.

Продолжение: [MM27-28](MINIMAX_M27_SESSION_CONTEXT.md) проверил snapshots2K/4K,
EOS/sampling и реальное давление RAM/VRAM. Границы исторического MM27-27 выше
не расширяются этими последующими проверками.
