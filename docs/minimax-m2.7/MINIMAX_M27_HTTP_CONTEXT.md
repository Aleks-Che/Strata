# MM27-24: HTTP context4K, давление на память и recovery

Дата: **2026-10-09**, `Asia/Yekaterinburg`; commit
`54467693c8354d8c64a72ec1cb083473a529b4ba`, dirty tree.
[План](MINIMAX_M27_IMPLEMENTATION_PLAN.md),
[статус](MINIMAX_M27_IMPLEMENTATION_STATUS.md).

Продолжение [MM27-23](MINIMAX_M27_LIVE_TOOLS_WEB.md). Проверяется полный
HTTP → native Engine путь на context4096 с настоящей моделью. Model/header,
CUDA kernels, sampler, cache/pipeline и serving defaults не меняются.
Native EXE SHA-256:
`f4ff449823beee3db1b260597cac1ed9276fb53922c5816f082f77ccbf52f619`.
Header SHA-256:
`9a011b2ad6fb20db3bb0cdfe9af08665939ee1f0cb0609ffe2bfee7f31c825db`.

## Реализация проверки

`tools/check_minimax_m2_http_context.py` запускает отдельный loopback HTTP
server, resident MiniMaxEngine и `strata-minimax-m2-pressure-holder.exe`.
Holder имеет собственный CUDA context до загрузки модели: его overhead
входит в global budget. Windows job привязывает owned child к checker;
EOF/finally освобождают только его allocations/mapping. Нормальный сервер
этот helper не запускает. Новый CMake target собирает только диагностику.

- Long prompt имеет4080 IDs после настоящих template/tokenizer;8 output
  tokens плюс `CTX_SLACK=8` заполняют API context4096. Это repeated filler
  с marker в начале, проверка поздних позиций и транспорта, не long-answer quality.
- OpenAI/Anthropic × JSON/SSE отклоняют max_tokens9 для этого prompt, а также
  prompt4096/max_tokens0. Проверяется HTTP400 и отсутствие native request.
- Допустимый4080/8 проходит через OpenAI JSON и Anthropic SSE; сравниваются
  все generated IDs, native prompt count, usage, finish и drain.
- Short control совпадает с32 IDs сохранённого EN greedy reference MM27-18.
  Сравнение повторяется под нагрузкой, после каждой отмены и после её снятия.
- Под давлением обоих типов памяти отключаются два настоящих HTTP-соединения
  во время prefill. Engine обязан принять cancellation, дождаться terminal
  event и обслужить следующий запрос в том же CUDA PID.

Holder читает существующий GGUF через read-only mapping, касаясь каждого4КиБ
page до≈88% global physical RAM. Отдельная heap-копия весов не создаётся;
`ram_touched` не означает resident bytes. Максимум working set ограничен
90%-бюджетом только своего процесса, минимум не поднимается, прежний предел
восстанавливается. Read-only pages остаются вытесняемыми Windows.

GPU pressure использует шаг32МиБ, target≤94,5% global VRAM и cap12ГиБ.
Это выше cache trim threshold runtime (95% минус256МиБ), но ниже hard95%.
Проверяются уменьшение `arena_reserved` и ненулевой `cache_pressure_trims`.
Нагрузка не считается достигнутой без samples, где обе памяти заняты≥85%.
Во время запросов helper снимает global NVML/RAM samples каждые≈0,5с;
во время allocation — после шагов. При нарушении95% проверка завершается
ошибкой и освобождает свои ресурсы. Это sampling, не непрерывная гарантия
при произвольной внешней нагрузке других программ.

## Воспроизведение

Сборка из существующего verified CUDA build, после vcvars64/CUDA_PATH как в
[backend README](../../backends/minimax_m2/README.md):

```powershell
cmake -S backends/minimax_m2 -B build-local/minimax-m2-cuda -DSTRATA_MM27_RUNTIME=ON
cmake --build build-local/minimax-m2-cuda --target strata-minimax-m2-pressure-holder -j 4
python -X utf8 -m unittest -v tools.test_minimax_m2_http_context serve.test_minimax_m2_engine serve.test_minimax_m2_server serve.test_minimax_m2_api
python -X utf8 tools/check_minimax_m2_http_context.py --gguf H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --out build-local/mm27-new-http-context
```

`--out` должен быть новым каталогом. Не запускать одновременно с другим GPU
checker. Сохраняются оба EXE, source snapshots, raw HTTP requests/responses,
prompt/generated IDs, memory samples, native result/error, cache counters.
Reference: `build-local/minimax-m2-completion-01/generation.json`; hash
фиксируется в отчёте. Наличие reference не означает независимый model oracle.

Профиль: RTX5090/32607МиБ, driver581.80, RAM125,555ГиБ;
context4096/batch16, cache18432МиБ, arena64/reserve0, file reader,
readers2/chunk4/lookahead/D2D batch, RAM0/grouping off, strict F32 activations/KV,
TF32/FA/graphs/MTP/DFlash off, greedy. Dependency86ebfef2, MSVC19.44,
CUDA13.0. CPU regressions могут выполняться рядом: timings не performance A/B.

## Результаты

`build-local/minimax-m2-http-context-01/report.json`: **17 scenarios PASS**, exit0.
Сводный [CHECK](MINIMAX_M27_HTTP_CONTEXT_CHECK.json) проверяет raw evidence,
109 source snapshots, оба EXE, build/test evidence и отсутствие owned processes.
**33 CPU methods PASS**,0 skips,17,495с. Проверены implicit max_tokens0/omitted
и точная граница контекста обоих API через настоящие loopback sockets с fixture
engine. Основной EXE побитно совпадает с MM27-22; собран только pressure holder.

| Проверка | Результат |
|---|---|
|8 запросов за границей, API × JSON/SSE × два вида overflow | HTTP400; native sequence осталась0 |
|OpenAI JSON и Anthropic SSE,4080 prompt /8 generated | Все8 IDs совпали, finish=length/max_tokens, usage и drain PASS |
|5 коротких завершённых controls,52 prompt /32 generated | Все160 IDs совпали с прежним greedy reference |
|Отмена long prefill под давлением через оба API | Два terminal cancellation; recovery в том же CUDA PID59184 |
|GPU cache pressure | arena18432→18304МиБ, освобождено128МиБ,2 trims |
|Снятие нагрузки и cleanup | Контрольный ответ совпал; holder/worker/native/owned consoles завершены |

Всего7 завершённых генераций /176 generated IDs и2 отменённые. Для полных
результатов GPU-only, source/file/H2D/D2D bytes, global95%, пустые очереди и
отсутствие pending fills PASS. От socket close до освобождения request lock
наблюдалось141/156мс (проверка состояния каждые50мс), без reload процесса.

Максимум среди holder samples, native samples и native phase peaks:
**RAM109,264ГиБ /87,025%; VRAM30,085ГиБ /94,479%**. Лимит95% не нарушен.
Holder выделил1248МиБ VRAM и коснулся каждой4КиБ страницы mapping файла128,841ГиБ;
его working set на окончании GPU allocation был≈78,483ГиБ. Цель88% RAM
не удерживается принудительно: чистые страницы может вытеснять Windows.
Это нагрузка на память, а не включение runtime RAM cache или полный checksum весов.

Обе памяти заняты≥85% в native samples каждого из трёх полных pressure/recovery
запросов:36/36,25/36,25/36 samples; порог не держится во всех samples, поскольку
runtime освобождает VRAM. Во время каждого отменяемого запроса также получены
два таких внешних sample. Monitor помечает samples последним native request ID,
поэтому в его группах возможны idle intervals после ответа; внутренние native
samples относятся к самой генерации. Непрерывный максимум между samples не измерен.

| Запрос | Prefill, с | Native decode, токенов/с |
|---|---:|---:|
|4K OpenAI,8 output |405,253|2,455|
|4K Anthropic,8 output |328,856|6,676|
|Короткий перед нагрузкой,32 output |10,032|3,696|
|Короткий под нагрузкой |9,934|4,030|
|После OpenAI cancellation, нагрузка удерживается |12,684|3,401|
|После Anthropic cancellation, нагрузка удерживается |13,736|3,186|
|После снятия нагрузки |10,708|4,088|

Порядок запросов и состояние кешей отличаются,8 output — слишком мало для
выбора defaults. Значение6,676 не является ускорением Anthropic относительно
OpenAI. Большая стоимость fresh prefill — основание следующим этапом проверить
prefix reuse, а не менять sampler/precision по этому проходу.

Build helper: `build-local/build-minimax-m2-http-context.bat`; configure/build
logs, manifest и финальный CPU log скопированы в `build-local/minimax-m2-http-context-01/support`.
В первоначальном CPU запуске imported TestCase дополнительно запускал9 уже
включённых методов. Исправлен discovery import; итоговый запуск содержит33
различных метода, первоначальные42 не суммируются с ними.
Повторная проверка сохранённого evidence:
`python -X utf8 build-local/validate-minimax-m2-http-context.py` — exit0.

## Ограничения

Session reuse/shift, независимый full-model oracle, широкое качество,
старые MM27-06/MM27-18b FAIL, installer и DFlash serving — отдельные gates.
Короткий prefix под нагрузкой не доказывает качество законченного длинного
ответа. Остановленный запрос не имеет terminal result stats: drain проверяется
освобождением request lock, terminal cancellation и точным следующим ответом.

Следующий шаг — P4: prefix reuse для resident native context с явной
идентичностью сессии, учётом реально вычисленных KV positions и сравнением с
fresh prefill. Перед включением нужны token/logit parity, isolation и отмена;
snapshot/restore/shift и расширение контекста выше4K остаются отдельными пунктами.
