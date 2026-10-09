# MM27-26: длинный resident prefix, EOS и давление памяти

Дата: **2026-10-09**, `Asia/Yekaterinburg`.
[План](MINIMAX_M27_IMPLEMENTATION_PLAN.md),
[статус](MINIMAX_M27_IMPLEMENTATION_STATUS.md),
[контракт resident prefix](MINIMAX_M27_PREFIX_CACHE.md).

## Объём

Проверяется прежний EXE MM27-25:
`294ce7117876b7a112e5a29c199c8822d9f48bc333a274cbd7f6d8ff65ebbfcd`.
Веса, CUDA arithmetic, алгоритм кеша, sampling и defaults не меняются.
Новые инструменты воспроизводят P4 gates на полной модели; при numerical FAIL
они сохраняют исходные данные и не меняют tolerance или admission.

`check_minimax_m2_prefix_context.py` запускает один native corpus с context4096,
batch16 и включённым prefix cache. Exact chat prompts имеют2032 и4080 токенов.
Проверяются cold/repeat2K, продолжение2K→4K, repeat4K и4K с полным prefill.
Первые2K вычисляются с пустым KV; для4K есть отдельный anonymous fresh control.
Все сравнения используют полные F32 logits и точные token IDs.

Отдельные пары в том же corpus проверяют естественный EOS английского ответа,
T1/top_p0,95/top_k40/seed42 с бюджетом128 и ровно один output token.
У последнего случая decode graph отсутствует: первый output выбирается по
prefill logits. Checker требует нулевых decode compute_calls/gpu_nodes;
это не CPU fallback и не отсутствие вычисления prompt.

Сравнение logits читает блоками по65536 float, проверяет finite и биты;
одинаковые NaN/Inf, изменение одного бита и знак нуля не допускаются.
Размер файла сверяется с реальным числом output rows. Это позволяет не
дублировать большой F32 dump в RAM при проверке длинных ответов.

`check_minimax_m2_prefix_pressure.py` использует обычный pinned Python Engine
и настоящий loopback HTTP. После warm4K другой owned процесс отображает GGUF
read-only и занимает VRAM ограниченными блоками. Контролируется фактическая
глобальная RAM/VRAM, а не сумма private allocations; предел95% сохраняется.
Проверяются repeated4K обоих API, cache trim, HTTP disconnect на изменённом
суффиксе (3936 reusable positions), полный4K recovery, native cancel после
первого decode output и short recovery/repeat/release. Live стадия сравнивает
token IDs; JSONL-протокол не экспортирует logits этого pressure-прогона.

## Результаты

Offline **PASS**:11 requests, шесть fresh/reuse сравнений,567 сравниваемых
output IDs и **113 436 288 logits** побитно, finite=true/max_abs=0.
Первый full EOS также совпал с историческим английским эталоном MM27-18
по414 IDs/logit rows. CPU checker tests:6 методов PASS.

RTX5090 32607МиБ, driver581.80, RAM125,555ГиБ, Windows; target Q4_K_M
из плана, context4096/batch16, cache18432МиБ, arena64/reserve0,
file/readers2/chunk4/lookahead/D2D, RAM0/grouping off. Strict F32 KV/activations,
TF32/FA/graphs/MTP/DFlash off. Ревизия
`54467693c8354d8c64a72ec1cb083473a529b4ba`, dirty tree.

Native timings, секунды; по одному измерению строки, без чередующихся A/B:

| Пара | Prompt / reused | Fresh prefill | Reuse prefill | Fresh request | Reuse request |
|---|---:|---:|---:|---:|---:|
| 2K,8 outputs | 2032 /2016 | 221,605 | 2,096 | 224,166 | 2,976 |
| 4K,8 outputs | 4080 /4064 | 304,798 | 2,042 | 305,716 | 2,958 |
| Продолжение2K→4K,8 outputs | 4080 /2016 | 304,798 | 130,454 | 305,716 | 131,574 |
| Greedy EOS,414 outputs | 52 /48 | 9,072 | 1,083 | 114,527 | 104,785 |
| Seed42,128 outputs | 52 /48 | 11,084 | 1,136 | 41,233 | 32,964 |
| Один output | 52 /48 | 9,000 | 0,872 | 9,001 | 0,873 |

Состояние expert/OS cache и порядок запросов различаются. Это измерения
задержки при пропуске общего prefill; они не утверждают рост decode tokens/s.
Обе EOS-генерации естественно завершились на414 tokens; sampled пары
ограничены128 tokens и завершились по length.

Pressure/HTTP **PASS**:11 сценариев, в том числе9 полных HTTP ответов /72 IDs.
Все IDs совпали с offline fresh reference; OpenAI/Anthropic JSON/SSE usage
правильно разделяет prompt и cached tokens. Native GPU-only, bytes, drained
queues, KV counters и global95% gates прошли на всех завершённых запросах.

| Live сценарий | Reused | Prefill,с | Request,с |
|---|---:|---:|---:|
| Cold4K, до pressure | 0 | 394,337 | 397,002 |
| Повтор4K, до pressure | 4064 | 2,105 | 3,050 |
| Повтор4K, pressure, OpenAI JSON | 4064 | 2,653 | 3,625 |
| Повтор4K, pressure, Anthropic SSE | 4064 | 1,850 | 3,789 |
| Полный4K recovery после disconnect | 0 | 341,153 | 343,740 |
| Повтор восстановленного4K | 4064 | 1,900 | 2,879 |
| Short recovery после decode cancel | 0 | 10,055 | 11,664 |
| Short repeat под pressure | 48 | 0,678 | 2,131 |
| Short repeat после FREE | 48 | 0,727 | 2,155 |

Измеренный максимум по native samples/phase peaks и holder samples:
RAM **108,736ГиБ /86,604%**, VRAM **30,068ГиБ /94,425%**. Holder занял3360МиБ
VRAM, коснулся каждой4КиБ страницы всего128,841ГиБ GGUF и имел working set
76,593ГиБ после RAM phase. Это read-only mapping, не runtime RAM cache и не
полный checksum весов.17 monitor samples одновременно превысили85% RAM/VRAM.
Arena сократилась с16384 до16256МиБ: один pressure trim освободил128МиБ.

HTTP disconnect при3940 общих input tokens (3936 reusable) завершился
native cancel за532,660мс с polling20мс; output tokens не было. Следующий
полный4K ответ совпал с fresh и имел reused0; его повтор —reused4064.
Native cancel после первого выходного токена занял3,195мс, лишних токенов
клиент не получил. Short recovery/repeat и запрос после FREE совпали с эталоном.
Весь live цикл прошёл в native PID19336, без restart. Holder PID24152,
offline PID5456 и live native process завершены; cleanup errors нет.

При отмене существующий `mm27_pipeline_abort` освобождает expert cache, а
request handler очищает KV. Поэтому во время полного recovery GPU usage
падает, затем arena растёт снова; внешние allocations остаются выделенными.
94,425% — пик, не постоянная загрузка всего recovery. Сохранение состояния
успешных сессий должно проверяться отдельным этапом snapshot/restore.

Движок и admission не менялись, исправлений runtime не потребовалось.
Первый CPU запуск нашёл ошибку в самом mock теста (повторно использовался
изменяемый dict); mock исправлен, исходный FAIL log сохранён. GPU FAIL нет.
Default prefix cache по-прежнему off; качество ответов/P6 не переоценивались.

Артефакты: `build-local/minimax-m2-prefix-context-01` (11 native results,
1126 output rows,901088256 байт F32, EXE/sources),
`build-local/minimax-m2-prefix-pressure-01` (HTTP requests/responses, native
records и memory samples). В offline `support` сохранены manifest, CPU logs,
test source и validator. [Итоговый CHECK](MINIMAX_M27_PREFIX_CONTEXT_CHECK.json):
26 gates и318 artifact/source hash checks PASS.

## Воспроизведение

Из корня репозитория, GPU прогоны только последовательно:

```powershell
python -m unittest tools.test_minimax_m2_prefix_context -v
python tools/check_minimax_m2_prefix_context.py --gguf H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --out build-local/minimax-m2-prefix-context-NEW
python tools/check_minimax_m2_prefix_pressure.py --gguf H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --offline build-local/minimax-m2-prefix-context-NEW --out build-local/minimax-m2-prefix-pressure-NEW
```

Оба инструмента требуют существующий проверенный native EXE. Offline сохраняет
EXE/source snapshots, requests, JSON и F32 dump; live сохраняет HTTP тела,
native results, monitor samples и source snapshots. Каталоги должны быть новыми.

## Границы

Это resident KV для одной явной сессии. Архив нескольких сессий,
snapshot/restore/shift, context выше4K, альтернативные KV dtype и GPU parity
batch1/8 не входят в этот этап. Широкое качество, независимый model oracle,
старое MM27-06 discrepancy и DFlash serving остаются отдельными gates.
Filler prompts проверяют транспорт и числа, а не качество long-context retrieval.
