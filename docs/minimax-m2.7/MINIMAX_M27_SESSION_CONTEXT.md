# MM27-28: длинные сессии и давление памяти

Дата: **2026-10-09**, `Asia/Yekaterinburg`.
[План](MINIMAX_M27_IMPLEMENTATION_PLAN.md),
[статус](MINIMAX_M27_IMPLEMENTATION_STATUS.md).

## Что проверяется

Продолжение [MM27-27](MINIMAX_M27_SESSION_ARCHIVE.md): RAM snapshot/restore
нескольких сессий с prompt2032/4080, завершение EOS и seeded sampling после
восстановления. Context4096/batch16, архив6144МиБ/4 записи. Сравниваются все
выходные token IDs и F32 logits с сохранённым fresh corpus MM27-26 и с fresh
запросами в текущем процессе. Старый корпус проверяется по hashes перед запуском.
Sampler state создаётся заново; archive не должен продолжать RNG старого ответа.

Во втором проходе OpenAI/Anthropic JSON/SSE проверяются под реальной нагрузкой
RAM/VRAM. Снимок, запрашиваемый для восстановления, защищается от вытеснения;
сохранение уходящей сессии должно отклоняться, когда глобального RAM budget
недостаточно. После HTTP disconnect до первого output и native decode cancel
другой завершённый snapshot должен оставаться пригодным, а отменённая сессия —
начинать с fresh prefill. Live JSONL проверяет IDs, не экспортирует полные logits.

Движок, Python admission SHA и defaults не меняются:
EXE `275e89f8628ceecd8f0599ba92dda1abae979ea2802bfb9bf32f3669344a2ba3`.
F32 activations/KV, TF32/FA/graphs/MTP/DFlash off; greedy, кроме заданных
sampling cases T1/top_p0,95/top_k40/seed42. GPU cache18432МиБ, arena64/reserve0,
file, readers2/chunk4/lookahead/D2D, RAM expert cache0/grouping off.
RTX5090,32607МиБ, driver581.80, RAM125,555ГиБ, Windows.

## Диагностическая нагрузка

Только отдельный opt-in `pressure-holder` получил команду `RAM94`:
после read-only GGUF mapping он выделяет и заполняет собственные private pages
по32МиБ, до16ГиБ суммарно. Перед каждой аллокацией проверяет physical available
и commit; целевой предел used94,2%, аварийный guard95% для RAM и VRAM.
Повышается только прежний предел working set самого holder; настройки ОС,
pagefile и чужих процессов не меняются. `FREE`/EOF/ошибка освобождают буферы.
Если reclaim mapped pages не позволяет достичь требуемой нагрузки, проверка
должна завершиться FAIL, а не увеличивать предел. GPU chunks прежние32МиБ,
target94,5%, максимум12ГиБ; runtime самостоятельно уменьшает expert arena.

## Offline: PASS

14 requests,1 150 generated tokens,920 294 400байт полных logits.
Каждый результат совпал со своим fresh reference из MM27-26:
**230 073 600 F32 logits побитно**, nonfinite0, max abs0.
Дополнительно шесть сравнений restore с fresh того же нового процесса:
**567 IDs /113 436 288 logits побитно**. Это перекрывающиеся сравнения;
1 150 и567 не считаются числом разных сгенерированных токенов.

Проверены два возврата4K, отдельный возврат2K, EOS414 после snapshot полного
предыдущего ответа, fresh/restored sampling128 с seed42, один output без
decode graph. Сериализованный generated tail не использовался как batched
prefill; повторно считался последний пакет prompt. Byte/count limits,
позиции KV, restore flags, finite logits и GPU-only stats прошли.

Один последовательный corpus,8 outputs для длинных prompts:

| Prompt | Cold request, с | Restore request, с | Restore prefill, с | Save/restore/trim, с |
| --- | ---: | ---: | ---: | ---: |
| 2032 tokens | 222,870 | 7,086 | 2,098 | 3,592 |
| 4080 tokens | 306,510 | 7,185 | 2,076 | 3,705 |

Повторный restore4K в конце corpus занял8,962с. Payload snapshot2K —987,665МиБ,
4K —1979,689МиБ. EOS414: request110,823→99,279с; sampling128:41,675→30,710с.
Это отдельные latency measurements с разным состоянием expert cache,
не три повторных A/B и не доказательство ускорения decode. Save/restore
по-прежнему занимает несколько секунд из-за множества мелких V copies.

## HTTP/pressure: PASS

**19 scenarios PASS:**17 полных ответов /662 IDs плюс две отмены.
OpenAI/Anthropic JSON/SSE, cached usage, restore2K/4K, EOS414 и sampling128
под нагрузкой совпали с offline reference по всем IDs. Полные pressure logits
не экспортировались. После каждой отмены другая завершённая сессия
восстановилась, а отменённая2K прошла fresh prefill с тем же ответом.
Весь проход выполнялся в одном native PID8688.

Global sampled peaks: **RAM94,191% / VRAM94,430%**, оба показателя выше85%
в771 monitor samples. Expert arena сократилась18 432→18 304МиБ, один trim
освободил128МиБ. Cancel освобождает также expert cache; после него VRAM
падает и заполняется снова при recovery. Это peaks, не постоянная загрузка95%.

Holder read-only touched138 342 384 352байт GGUF, по одному байту на4КиБ
страницу; это не полный checksum и не RAM-кеш движка. Дополнительно он держал
11,625ГиБ private pages и1792МиБ GPU buffers. Его максимальный working set
93,628ГиБ, private14,255ГиБ. При RAM94 сохранение уходящего snapshot4K было
отклонено, старая запись вытеснена, а запрашиваемый snapshot2K восстановлен:
`session_saved_bytes=0`, rejected0→1, evictions0→1,
`session_restored_bytes=1 035 642 236`.

При этом6ГиБ/4 archive slots хватало для всех записей. Независимый1-секундный
монитор показал около7,374ГиБ physical available и48,968ГиБ доступного commit
рядом с RAM94. Сам holder в конце topup сообщил7 838 179 328байт physical
available. Для сохранения4K требовались blob плюс5% RAM и256МиБ запаса —
больше доступной физической памяти. Отказ вызван RAM budget, не cap/count
архива или нехваткой commit. Completed response сохранил правильные IDs.

Измеренные timings:

- Restore4K под нагрузкой: request9,808с, включая session5,647с и
  prefill2,586с. Следующие restore4K:6,666/9,308/8,974с.
- Restore2K:7,015с; защищённый restore после отказа сохранить4K:7,042с.
- EOS414:112,840с; sampling128:36,893с. Это request latency, не decode A/B.
- Disconnect до первого output, во время переноса/подготовки KV:3051,781мс
  до завершения native cancel/drain,0 outputs, опрос20мс. Флаг отмены
  проверяется после синхронного snapshot/restore; перенос сам не прерывается.
- Отмена после первого output восстановленной2K:3,187мс, дополнительных
  клиентских tokens нет. Fresh2K recovery:170,687с с0 reused tokens.

## Проверки и воспроизведение

**14 CPU methods PASS**, exit0: новые проверки ложных restore flags/bytes,
caps/timings и сохранения native failures; прежние bounded logit comparisons
и HTTP context boundaries. Новый holder собран exit0. Его SHA:
`50cf15e7c02e9c5482b67a2c333cf116e44ef6fb5f7d2abb1bdb2c03ef58f6e6`.
Прежний holder/source сохранены в
`build-local/minimax-m2-session-pressure-baseline`; native Engine EXE прежний.
Ревизия `54467693c8354d8c64a72ec1cb083473a529b4ba`, dirty tree, индекс не менялся.

```powershell
& build-local/build-minimax-m2-session-pressure.bat
python -X utf8 -m unittest tools.test_minimax_m2_sessions_context tools.test_minimax_m2_sessions tools.test_minimax_m2_prefix_context tools.test_minimax_m2_http_context
python -X utf8 tools/check_minimax_m2_sessions_context.py --gguf H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --out build-local/minimax-m2-sessions-context-01
python -X utf8 tools/check_minimax_m2_sessions_pressure.py --gguf H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --offline build-local/minimax-m2-sessions-context-01 --out build-local/minimax-m2-sessions-pressure-01
python -X utf8 build-local/finalize-minimax-m2-session-context-monitor.py
python -X utf8 build-local/validate-minimax-m2-session-context.py
```

GPU checks запускались последовательно. Отдельный read-only RAM/commit
observer был запущен во время pressure checker; точный inline source сохранён
в `build-local/minimax-m2-session-context-monitor.py`. Он завершился после
engine cleanup,656 samples. Только hash его дописываемого файла финализирован
после завершения observer; исходный pressure report до финализации сохранён
в `support/pressure-before-monitor-finalization.json`. Checks/results не менялись.
При воспроизведении нужны новые `--out` каталоги и соответствующие пути observer.

Offline artifacts: `build-local/minimax-m2-sessions-context-01`;
live artifacts: `build-local/minimax-m2-sessions-pressure-01`.
Сохранены requests/results, F32 logits, wire responses, memory samples,
EXE/source snapshots, build logs и CPU evidence. Сводка с hashes:
[SESSION_CONTEXT_CHECK](MINIMAX_M27_SESSION_CONTEXT_CHECK.json).
Owned native/holder PIDs24228/8688/58064 завершены, после cleanup GPU1310МиБ.

## Следующий шаг

Пакетный перенос V по слоям с ограниченным временным buffer: сохранить формат
state и все незатрагиваемые cells, проверить побитно state/logits и только
после этого измерить три A/B по полной задержке переключения.
Default archive0/prefix off сохранены. GPU batch1/8, shift/context>4K,
alternate KV dtype, широкий quality corpus, independent model oracle и P6
installer остаются отдельными gates. Этот этап не закрывает весь P4/P6.
