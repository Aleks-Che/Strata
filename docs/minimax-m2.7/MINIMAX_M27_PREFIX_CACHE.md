# MM27-25: один resident KV-префикс

Дата: **2026-10-09**, `Asia/Yekaterinburg`.
[План](MINIMAX_M27_IMPLEMENTATION_PLAN.md),
[статус](MINIMAX_M27_IMPLEMENTATION_STATUS.md).

## Контракт

Native `--prefix-cache 1` и серверный `--prefix-cache` включают сохранение
одного префикса в уже выделенном GPU KV. Default off. HTTP-клиент передаёт
`X-Strata-Session-Id`; адаптер проверяет длину/управляющие символы и отправляет
в native SHA-256 этого ID. `/health` сообщает `session_id: true` только при
включённом флаге. Capability cache-admin остаётся false; архива сессий нет.

Совпадение определяется по ID сессии и точным token IDs. Смена сессии или
анонимный запрос выполняют полный prefill и вытесняют прежнее состояние.
Возврат к вытесненной сессии также начинает с полного prefill.
Состояние существует только в текущем context; restart/unload его уничтожает.

Повторно используются целые prefill-пакеты. Последний пакет нового prompt
всегда считается заново для получения logits; неполный пакет прошлого prompt
и serial decode не используются как готовый prefill. Это сохраняет размеры
пакетов относительно fresh request. Для batch16 и prompt52 повтор использует
48 позиций, для prompt512 —496. Продолжение может использовать все512 позиций,
если они были вычислены полными пакетами и совпадают по token IDs.

Перед повтором проверяются min/max KV positions и удаляется старый суффикс.
После ответа проверяется число реально вычисленных позиций:
`kv_tokens = prompt_tokens + generated_tokens - 1`. Последний выбранный токен
ещё не прошёл decode и не считается готовым KV. В result добавлены
`reused_tokens`, `evaluated_prompt_tokens` и `kv_tokens`; Python проверяет
целочисленность, границы, размер пакета и согласованность счётчиков.

Любая native ошибка, включая отмену, очищает KV и метаданные префикса.
Отмена до отправки native запроса не меняет состояние. Отклонённый на HTTP/
Python validation запрос также не обращается к native. Sampling RNG создаётся
заново для каждого запроса; кеш не продолжает старое состояние sampler.

OpenAI usage сохраняет полный `prompt_tokens` и отдельно сообщает
`prompt_tokens_details.cached_tokens`. Anthropic вычитает reused из
`input_tokens` и сообщает их в `cache_read_input_tokens`.

## Проверки

RTX5090 32ГиБ (32607МиБ), driver581.80, RAM125,555ГиБ, Windows;
target Q4_K_M из плана. Context2048, batch16, cache18432МиБ,
arena64/reserve0, file, readers2/chunk4/lookahead/D2D, RAM0/grouping off.
Strict F32 activations/KV; TF32/FA/graphs/MTP/DFlash off, greedy.
Ревизия `54467693c8354d8c64a72ec1cb083473a529b4ba`, dirty tree.

- **14 430 C++ checks PASS:** batch1/8/16, длины1..65, каждый возможный
  общий префикс/branch, сокращение, продление, смена ID/batch и invalidation.
- **29 Python test methods PASS:** Engine/supervisor, session digest и admission,
  terminal accounting, HTTP session headers/health и context boundaries.
  Это29 различных методов; промежуточные18 не прибавляются к итогу.
- **16 full-model scenarios PASS:**16–513 prompt tokens, повтор, generated
  tail, сокращение, branch, ID switch/return, анонимный запрос и границы batch.
  Все128 generated IDs и **25 608 192 F32 logits** побитно совпали с полным
  prefill. Max abs=0, nonfinite=0. Позиции KV и reused/evaluated counts совпали.
- **Регрессия относительно прежнего EXE PASS:** два fresh запроса,16 IDs и
  3 201 024 logits побитно совпали. Это отдельная проверка изменения бинарника.
- **21 live scenario PASS:**17 HTTP ответов /136 IDs, OpenAI/Anthropic JSON/SSE,
  cached usage, ID switch/anonymous, две адресные native отмены, already-cancelled
  запрос, native validation error и restart. Все полные ответы совпали с fresh.
  После prefill/decode cancel и malformed session key следующий запрос имеет
  reused0, затем повтор снова reused48. PID34892 сохранился до намеренного
  restart; новый PID2640 начал с reused0. Все owned процессы завершены.

Active prefill cancel→terminal занял93мс; decode cancel записан как0мс
монотонным таймером Python, то есть ниже его разрешения в этом измерении.
Это не утверждение о нулевой задержке. HTTP disconnect под pressure был
проверен ранее для fresh path, здесь отмена адресовалась через native adapter.

Во всех завершённых запросах прошли GPU-only, source/H2D/D2D accounting,
drained queues и global95% gates. Максимум по memory samples и phase peaks:
RAM36,036ГиБ /28,701%, VRAM27,170ГиБ /85,324%. Внешнее давление в MM27-25
не создавалось. Новых KV buffers или session snapshots не выделяется.

Один последовательный fresh/reuse corpus, timings native (миллисекунды
в raw JSON; ниже секунды):

| Сценарий | Prompt / reused | Fresh prefill | Reuse prefill | Fresh request8 | Reuse request8 |
|---|---:|---:|---:|---:|---:|
| Повтор короткого | 52 /48 | 8,235 | 0,689 | 9,104 | 1,576 |
| Повтор512 | 512 /496 | 51,236 | 1,579 | 53,060 | 3,415 |
| Продление512→513 | 513 /512 | 51,515 | 0,304 | 53,519 | 2,348 |
| Изменение суффикса512 | 512 /496 | 49,446 | 1,909 | 51,184 | 3,675 |

Это по одному измерению каждой строки, без чередующихся A/B повторов.
Разное состояние expert/OS cache влияет на timings; не переносить отношения
на новые prompt или на decode tokens/s. Подтверждённое изменение работы —
меньше вычисленных prefill tokens при побитно одинаковом результате.

Новый принятый EXE SHA-256:
`294ce7117876b7a112e5a29c199c8822d9f48bc333a274cbd7f6d8ff65ebbfcd`.
Предыдущий `f4ff4498…` сохранён в baseline и offline artifacts.
Header SHA и dependency86ebfef2 не менялись. SHA admission обновлён после
offline parity; HTTP smoke затем прошёл через обычный adapter без обхода pin.

Артефакты: `build-local/minimax-m2-prefix-01` (full logits/EXE/requests,
native source snapshot), `build-local/minimax-m2-prefix-http-01` (HTTP bodies,
native results и окончательные adapter sources). Build manifest/logs,
CPU log и policy executable/report находятся в offline `support`.
Компактная повторная проверка: [MINIMAX_M27_PREFIX_CACHE_CHECK.json](MINIMAX_M27_PREFIX_CACHE_CHECK.json).

## Воспроизведение

После сборки backend в MSVC/CUDA environment:

```powershell
build-local\minimax-m2-cuda\bin\strata-minimax-m2-prefix-check.exe
python -m unittest serve.test_minimax_m2_engine serve.test_minimax_m2_server serve.test_session_cache tools.test_minimax_m2_http_context -v
python tools/check_minimax_m2_prefix.py --gguf H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --reference-engine build-local/minimax-m2-prefix-baseline/engine.exe --out build-local/minimax-m2-prefix-NEW
python tools/check_minimax_m2_prefix_http.py --gguf H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --offline build-local/minimax-m2-prefix-NEW --out build-local/minimax-m2-prefix-http-NEW
```

GPU-проверки выполнять последовательно. Offline checker сохраняет input,
full F32 logits, результаты, EXE/source snapshots и hashes. Старый EXE служит
регрессионным эталоном того же движка; независимый model oracle он не заменяет.

Чтобы включить в изолированном сервере на этом ПК:

```powershell
python -m serve.minimax_m2_server --gguf H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --ctx 2048 --batch 16 --gpu-cache-mib 18432 --pipeline-readers 2 --pipeline-chunk-mib 4 --prefix-cache
```

Клиенту нужен стабильный `X-Strata-Session-Id`; web chat использует его после
health capability. Сам browser с prefix-on в MM27-25 отдельно не прогонялся.

## Оставшиеся части P4

Gate reuse2K/4K, один EOS414, sampling128 и реальный RAM/VRAM pressure
проверены в [MM27-26](MINIMAX_M27_PREFIX_CONTEXT.md); pressure parity относится
к token IDs, offline — ко всем logits. Полная GPU parity относится к batch16;
batch1/8 здесь проверены только CPU policy fixtures. Snapshot/restore нескольких
сессий, shift, context выше4K и другие KV dtype также остаются отдельной работой.
Предыдущий MM27-24 pressure gate относится к fresh prefill. Широкий quality
corpus, independent model oracle, MM27-06 discrepancy и live DFlash gates
остаются открытыми. Этот кеш ускоряет обработку общего prompt; скорости
вычисления новых decode-токенов он сам по себе не увеличивает.
