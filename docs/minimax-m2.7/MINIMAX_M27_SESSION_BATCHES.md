# MM27-31: prefix и RAM-сессии при batch 1 и 8

Дата: **2026-10-09**, `Asia/Yekaterinburg`.
[План](MINIMAX_M27_IMPLEMENTATION_PLAN.md),
[статус](MINIMAX_M27_IMPLEMENTATION_STATUS.md).

Продолжение P4 после [MM27-30](MINIMAX_M27_STATE_BULK_PRESSURE.md).
Проверяется неизменность ответа при повторном использовании KV для каждого
batch относительно fresh-пересчёта **при том же batch**. Равенство результатов
между разными batch не является условием этой проверки.

Native EXE прежний:
`98e85e80e2dbaff5dc38f03f1ab4039835fb8d7e384f4491e1d700c989c5d3d8`.
`STRATA_MM27_STATE_BULK=1`, `STRATA_MM27_TOKENWISE=0`.
Модель, dependency86ebfef2, precision и runtime defaults не меняются.

## Корпус и условия

Два последовательных GPU-процесса для каждого batch: девять уникальных
fresh-запросов с отключённым prefix/archive и 25 запросов с prefix и RAM archive.
Последний процесс начинается только после завершения предыдущего.
Каждый запрос имеет собственный fresh-эталон с теми же token IDs, output budget
и sampler. Сравниваются все выходные IDs и все F32 logits, побитно; NaN/Inf
отклоняются. Отдельно проверяются reuse positions, restore flag/bytes,
KV accounting, archive caps, GPU-only execution и время session IO.

Проверяются:

- повтор prompt, изменение token35, укорочение до33 и обратное расширение;
- prompt ровно из одного пакета: один токен при batch1, восемь при batch8;
- смена session ID, возврат из RAM archive, anonymous request и возврат;
- prompt512, restore/resident repeat, расширение512→513 и изменение token497;
- один выходной токен после cold/restore/resident request;
- sampling64: T1/top_p0,95/top_k40/seed42 после resident reuse и RAM restore.

Например, repeat512 должен переиспользовать511 позиций при batch1 и504 при
batch8; последняя порция prompt вычисляется заново. При расширении512→513
ожидается512 reused positions для обоих batch. Эти ожидания заданы явно
в диагностическом корпусе, независимо от вызова native planner.

Context1024, F32 activations/KV, TF32/FA/graphs/MTP/DFlash off.
GPU cache18432 МиБ, arena64/reserve0, file/readers2/chunk4/lookahead/D2D,
RAM expert cache0, groups0. Session archive768 МиБ/4 записи в session-процессе,
cap0/prefix off в fresh-процессе. RTX5090 32607 МиБ, driver581.80,
RAM125,555 ГиБ, Windows. Полные logits и source/EXE snapshots сохраняются.

Отдельный Windows/NVML observer измеряет RAM/commit/VRAM с интервалом0,5с.
Ошибка измерения или превышение95% завершает только собственный тестовый
процесс и отклоняет прогон. Поток закрывается до hashes. Искусственного
pressure holder в этом этапе нет.

## Результат

**Оба batch PASS.** Всего68 native requests /810 сгенерированных токенов,
из них50 session-сценариев /568 IDs /113 636 352 сравниваемых F32 logits.
Все logits совпали с соответствующим fresh побитно, все значения конечны.

| Batch | Fresh requests | Session scenarios | Сравнено IDs / logits | Пик RAM / VRAM |
|---|---:|---:|---:|---:|
| 1 | 9 | 25 PASS | 284 /56 818 176 | 24,063% /88,959% |
| 8 | 9 | 25 PASS | 284 /56 818 176 | 23,507% /87,301% |

Восемь RAM restores на batch, включая один output и sampling64, дали
ожидаемые restore flag/bytes и reused positions. Все native и независимые
отсчёты памяти в пределах95%. Четыре owned native-процесса завершены,
observer threads остановлены до финальных hashes. Давление памяти здесь
не создавалось; эти пики не заменяют MM27-30 pressure measurements.

14 CPU methods для корпуса, строгого сравнения, session evidence и observer
PASS; лог: `build-local/minimax-m2-session-batches-unit.log`.
Итоговый validator заново сравнил все raw logits и проверил coverage,
source/EXE identity, limits и cleanup: **59 условий /166 source checks /
194 artifact hashes PASS**.
[Машиночитаемый результат](MINIMAX_M27_SESSION_BATCHES_CHECK.json).

## Время отдельных запросов

Один проход, prompt512 и8 output tokens; полное native request time с session IO:

| Batch | Fresh | Resident repeat | RAM restore |
|---|---:|---:|---:|
| 1 | 159,425 с | 1,950 с | 3,824 с |
| 8 | 88,182 с | 3,029 с | 5,582 с |

Это наблюдения correctness-корпуса с разными состояниями expert cache,
без повторного speed A/B. При batch1 последний повторный prefill содержит
один токен, при batch8 — восемь; fresh вычисляет все512 токенов. По этим
одиночным timings новый default не выбирался, рост decode tokens/s не заявлен.

Добавлены только диагностический corpus runner, CPU checks и validator.
Runtime, native EXE, precision и serving defaults прежние. Следующий пункт —
GPU batch1/8 на2K/4K с fresh при том же batch, затем shift.

## Команды

```powershell
python -X utf8 -m unittest tools.test_minimax_m2_session_batches tools.test_minimax_m2_sessions tools.test_minimax_m2_sessions_context tools.test_minimax_m2_memory_observer
python -X utf8 tools/check_minimax_m2_session_batches.py --gguf H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --batch 8 --out build-local/minimax-m2-session-batch8-01
python -X utf8 tools/check_minimax_m2_session_batches.py --gguf H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --batch 1 --out build-local/minimax-m2-session-batch1-01
python -X utf8 tools/validate_minimax_m2_session_batches.py --batch1 build-local/minimax-m2-session-batch1-01 --batch8 build-local/minimax-m2-session-batch8-01
```

При повторении нужны новые output directories. Один correctness pass на batch;
это не speed A/B и не независимый model oracle. Проверяются prompts до513
при context1024: **GPU batch1/8 на2K/4K остаются отдельным gate**. Новый полный
EOS-корпус, HTTP/cancel/pressure при batch1/8, shift, context>4K и широкое качество
в этот этап не входят. Pressure batch16 уже проверен MM27-30. Старое
MM27-06 reload discrepancy остаётся OPEN.
