# MM27-30: пакетные KV copies под давлением памяти

Дата: **2026-10-09**, `Asia/Yekaterinburg`.
[План](MINIMAX_M27_IMPLEMENTATION_PLAN.md),
[статус](MINIMAX_M27_IMPLEMENTATION_STATUS.md).

Проверка новой сборки [MM27-29](MINIMAX_M27_STATE_BULK.md) на длинном корпусе
и при реальном давлении RAM/VRAM. Native EXE и defaults не меняются:
`98e85e80e2dbaff5dc38f03f1ab4039835fb8d7e384f4491e1d700c989c5d3d8`.
В обоих запусках явно задано `STRATA_MM27_STATE_BULK=1`.

## Область проверки

Первый проход повторяет14 offline requests MM27-28: cold/restore2K/4K,
EOS414, sampling128 с seed42, однократный output. Сравниваются все IDs и
полные F32 logits с прежним проверенным корпусом MM27-26 и с fresh запросами
в текущем процессе. Это проверка неизменности результатов после изменения IO.

Второй проход повторяет19 live scenarios: оба API, JSON/SSE и cached usage,
restore2K/4K, EOS/sampling под давлением памяти, GPU arena trim, отказ
сохранить4K при недостатке физической RAM с защитой запрошенной2K-сессии,
HTTP disconnect до первого output, отмена во время decode и fresh recovery.
Сравниваются IDs; live JSONL не экспортирует полные logits.

Context4096/batch16, F32 activations/KV, TF32/FA/graphs/MTP/DFlash off;
cache18432МиБ, arena64/reserve0, file/readers2/chunk4/lookahead/D2D,
RAM expert cache0, archive6144МиБ/4 записи. Greedy, кроме указанного sampling
T1/top_p0,95/top_k40/seed42. RTX5090 32607МиБ, driver581.80,
RAM125,555ГиБ, Windows. Target/header/template и dependency86ebfef2 прежние.

## Измерение памяти

Диагностический pressure holder прежний, SHA
`50cf15e7c02e9c5482b67a2c333cf116e44ef6fb5f7d2abb1bdb2c03ef58f6e6`:
read-only mapping GGUF, чтение по одному байту на4КиБ, отдельные private
buffers до16ГиБ и GPU buffers до12ГиБ. Target RAM94,2%/VRAM94,5%, guard95%.
Его buffers не являются RAM/GPU-кешем движка.

Добавлен отдельный read-only observer, который каждые0,5с читает физическую
RAM/commit через GlobalMemoryStatusEx и глобальную VRAM через System32 NVML.
Он выбирает единственный GPU по PCI bus ID и не зависит от очереди команд
holder. Поэтому заполнение RAM больше не блокирует независимые измерения.
Ошибки измерения и превышение95% отклоняют проверку; при активном запросе
посылается отмена. До вычисления hashes поток завершает запись и закрывается.
Опрос ограничен20 000 samples. Настройки ОС, pagefile и чужих процессов
не меняются. CPU tests проверяют guard, ошибки чтения и финализацию файла.

## Offline: PASS

14 requests /1150 IDs /230 073 600 F32 logits совпали с прежним эталоном
побитно. Шесть дополнительных fresh/restore сравнений:567 IDs /
113 436 288 logits также совпали. EOS414, sampling128/seed42 и single-output
прошли; NaN/Inf нет. Файл полных logits занимает920 294 400байт.

Один проход, request time с IO: cold2K224,062с → restore5,390с;
cold4K303,624с → первый restore5,561с. Другой restore4K занял8,796с
при другом состоянии expert cache и исходящей сессии. EOS fresh/restore
111,899/103,216с; sampling45,919/33,374с. Это диагностика конкретного корпуса,
не повторный speed A/B и не изменение подтверждённых медиан MM27-29.

## HTTP/pressure: PASS

Все **19 сценариев** прошли в одном native PID: 17 полных HTTP-ответов /
662 IDs совпали с offline-корпусом; ещё два сценария проверили отмену.
OpenAI/Anthropic JSON/SSE, cached usage, EOS414 и sampling128/seed42 прошли.
Полные logits проверялись offline, live pressure сравнивает token IDs.

| Измерение | Результат |
|---|---:|
| Максимум глобальной занятой RAM по всем записанным отсчётам | **94,196664%** |
| Максимум глобальной занятой VRAM | **94,403402%** |
| Отдельный observer: максимум RAM / VRAM | 94,140979% / 94,403402% |
| Отсчёты отдельного observer | 2093, без ошибок, поток завершён до hashing |
| Дополнительные private buffers holder | 11,65625 ГиБ из лимита16 ГиБ |
| GPU buffers holder | 1184 МиБ из лимита12 ГиБ |
| Физическое сокращение GPU arena | **64 МиБ**, один trim |
| Отказ сохранить исходящий4K snapshot / eviction | 1 / 1 |
| Восстановлено из защищённого2K snapshot | 1 035 642 236 байт |

Во всех записанных отсчётах RAM/VRAM оставались в пределах95%. В момент
RAM94 holder измерил7 827 099 648 байт свободной RAM. Первый независимый
отсчёт через0,250с: свободно7,3563 ГиБ RAM и45,9216 ГиБ commit. Архив также
помещался в свои caps6144 МиБ/4 записи. Таким образом, отказ admission
проверен именно при нехватке физической RAM для snapshot и резерва,
а не только при исчерпании логического лимита архива или commit.

HTTP disconnect до первого output обработан за1,644с, outputs0.
Отмена после первого decode token завершилась за3,519мс, дополнительных
токенов клиент не получил. Другая сохранённая сессия пережила обе отмены;
fresh2K recovery дал прежние8 IDs, reused_tokens0. После FREE у holder
нулевые GPU/private/touched counters; последний resident reuse прошёл.
Оба native-процесса corpus/HTTP и holder завершены, cleanup без ошибок.

В этом одном проходе: restore2K под первой pressure-фазой — 3,711 с;
первый restore4K — 9,466 с, следующие — 3,468/7,947/7,768 с.
Защищённый restore2K — 5,359 с; EOS (414 токенов) — 113,511 с;
sampling (128 токенов) — 36,033 с. Это полное HTTP время с IO.
Состояния expert cache и исходящей сессии различаются. Эти timings
не заменяют трёхпарный A/B MM27-29 и не доказывают новый speedup.

## Проверки и артефакты

18 CPU methods PASS. Итоговый validator заново сравнил все offline logits,
проверил pressure/admission/отмены/caps, идентичность EXE и завершение процессов:
**100 условий /160 проверок исходников /211 хешей артефактов PASS**.
[Машиночитаемый результат](MINIMAX_M27_STATE_BULK_PRESSURE_CHECK.json).

- `build-local/minimax-m2-state-bulk-context-01`: requests, полные logits,
  source/EXE snapshots и offline report.
- `build-local/minimax-m2-state-bulk-pressure-01`: HTTP requests/responses,
  native stderr, оба memory observers, source/EXE/holder snapshots и report.
- `build-local/minimax-m2-state-bulk-pressure-unit.log`: CPU evidence.

Изменён только диагностический код: независимый observer с guard и завершением
записи до hashes, фиксация runtime environment, stage labels и validator.
Native EXE, holder, dependency, serving и defaults не менялись.

## Команды

```powershell
$env:STRATA_MM27_STATE_BULK='1'
python -X utf8 -m unittest tools.test_minimax_m2_memory_observer tools.test_minimax_m2_sessions_context tools.test_minimax_m2_sessions tools.test_minimax_m2_prefix_context tools.test_minimax_m2_http_context
python -X utf8 tools/check_minimax_m2_sessions_context.py --gguf H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --out build-local/minimax-m2-state-bulk-context-01 --stage 'MM27-30 offline'
python -X utf8 tools/check_minimax_m2_sessions_pressure.py --gguf H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --offline build-local/minimax-m2-state-bulk-context-01 --out build-local/minimax-m2-state-bulk-pressure-01 --stage 'MM27-30 pressure'
python -X utf8 tools/validate_minimax_m2_state_bulk_pressure.py --offline build-local/minimax-m2-state-bulk-context-01 --pressure build-local/minimax-m2-state-bulk-pressure-01
```

GPU проходы последовательные, output directories при повторении должны быть
новыми. Это один pressure pass, не три A/B скорости. GPU batch1/8,
shift/context>4K, alternate KV dtype, широкий quality corpus, независимый
model oracle и MM27-06 reload discrepancy остаются отдельными gates.
