# MiMo: одинаковая арифметика короткого verification batch

MIMO-18, 7 октября2026. Windows, Ryzen9 9950X, RTX5090 32ГБ,
128ГБ RAM; CUDA13.0.48, MSVC19.44.35222.0, sm120, driver581.80.

Продолжение [MIMO-17](MIMO26_FLASH_TARGET_HEAD.md). Добавлены MiMo-local
CUDA dispatch для short-batch и ограничение пакета на границе KV.
Прежний русский oracle mismatch на18-м токене устранён в проверенных случаях.
Это offline greedy probe; MTP serving по-прежнему выключен.

## Изменения

Опция `--tokenwise-matmul 0..7`, default0, состоит из трёх битов:

| Бит | Изменение |
|---|---|
| 1 | Dense weight matmul использует обычный single-column GPU dispatch для каждого из2–8 столбцов |
| 2 | Quantized routed `MUL_MAT_ID` разбивается по токенам; strides и selected expert IDs сохраняются |
| 4 | FA192/128 с GQA8/16 на Blackwell использует существующую специализацию с одной query в tile; grid обрабатывает весь batch |

Режим7 включает все три. Dense flag охватывает также выходную BF16-голову,
поэтому отдельный `--target-head-columns 1` ему не нужен. У routed dispatch
согласован также sync planner. Веса не меняются; matmul остаётся на GPU.
CPU читает файлы, планирует работу, выбирает токены и сравнивает результаты.
В generation mode эти dispatch правила действуют также на подходящие
операции draft catch-up; это сравнение полного режима MTP.

Патчи применяются только при `STRATA_MIMO_SPEC_PROBE=ON` к сгенерированным
MiMo translation units. Архив `_deps` не редактируется; его SHA и исходный
FA source проверяются CMake. Ненулевой режим с DFlash пока запрещён.

Обычный prefill выполняется с mode0, verification — с выбранным mode.
Оба типа workspace прогреваются до заполнения cache. Прежний target-head
эксперимент остаётся независимым флагом для сравнения. TF32 и CUDA Graphs off.

При ненулевом mode verifier также ограничивает proposals:
`min(depth, remaining-1, 255-pos%256)`. В pinned KV cache размер используемой
области округляется до256 ячеек. Пакет, пересекающий эту границу, меняет
геометрию attention для ранних queries по сравнению с обычным decode.
На самой границе выполняется одиночный шаг; следующие proposals снова разрешены.
Это правило относится к текущему offline context512 с одной непрерывной
последовательностью, без context shift и дыр в размещении KV.

## Локализация расхождения

Teacher-forced diagnostic восстанавливает одинаковый prompt/history и сравнивает
два одиночных шага с batch2. Проверены начало генерации и пара перед прежним
сбоем. Callback сохраняет F32 активации и может менять fusion; скорость по
этому trace не измерялась. Отсутствующие view aliases исключены из сравнения.

| Режим | Наблюдение |
|---|---|
| 0, прежний путь | Первое отличие уже в `wqkv-0`, около1e-4 |
| 1, dense | QKV совпадает; первое отличие в attention следующего слоя, около1e-8 |
| 3, dense+routed | Без исправления attention oracle всё ещё расходится |
| 5, dense+FA | Первое отличие в MoE; IDs короткого oracle совпадают, logits не exact |
| 7, все три | По483 сопоставленные активации каждого токена двух проверенных пар совпали побитно, включая выход;95 отсутствующих aliases исключены |

В обычной генерации без callback русский oracle modes0/2 расходится на
индексе17, modes1/3 — на24. Mode5 сохраняет IDs, но max_abs logits1,549.
Mode7 даёт точные32 IDs и все4882432 logits; приняты15/15 proposals.
Ошибочные варианты сохранены в отчёте, а не исключены из результатов.

## Проверки на модели

Main GGUF: `MiMo-V2.6-Flash-RL-GSQ-RCO-3.5bit.gguf`,134982426368 байт.
MTP: `mtp-MiMo-V2.6-Flash-MOPD-Q4_0.gguf`, только первая обученная голова.
Context512/batch8, F32 KV, полный физический SWA для rollback, FA on,
greedy/thinking off. Global RAM/VRAM guard95%; чужие процессы не завершаются.

- 48 synthetic CUDA matrix cases: BF16/Q2_K/Q3_K/MXFP4, геометрии4096→2048
  и2048→4096, batch1/2/8, contiguous/padded activations и IDs, реальные
  размеры MiMo matmul. Все результаты bit-exact относительно независимых
  одиночных GPU вызовов. Веса, входы и canaries не изменились.
- Oracle count/code/ru: depth1 и7, по32 outputs; все192 IDs и logits exact.
- Boundary oracle: prompt127 и255, depth1/7, по16 outputs. После ограничения
  пакета все64 IDs и logits exact. До него boundary255 сохранял IDs, но
  logits отличались до1,184; этот результат также сохранён.
- Реальный Q4 MTP: count/code/ru с p_min0 и0.7 плюс boundary127/255 с p_min0.
  Все224 IDs совпали. Проверены rejection/rollback, в том числе11/19 принятых
  proposals на русском p_min0 и5/8 на boundary255 после исправления.
- У Q4 все216 строк logits после первого выхода совпали побитно. Первая
  строка, полученная из prefill с NextN features, пока не bit-exact:
  max_abs до0,000289. По коду сборка последнего FFN при включённых features
  отличается от обычного last-logit prefill; отдельная локализация этого
  остатка ещё нужна. Полная parity MTP logits не заявляется.
- 30 Python checks и106 verified-prefix/boundary checks PASS.

Короткий greedy corpus не доказывает качество stochastic sampling, длинного
контекста или всех запросов. DFlash, MTP heads2/3, streaming/cancellation,
session lifecycle и serving этим этапом не принимаются.

## Скорость

Результаты контрольной серии сохранены в
[MIMO26_FLASH_TOKENWISE_BENCHMARK.json](MIMO26_FLASH_TOKENWISE_BENCHMARK.json).
Порядок A B C C B A: no-draft, прежний MIMO-17 Q4/head-columns1,
Q4/tokenwise7/head-columns0. В каждом процессе count/code/ru, два прогрева
и три измеряемых ответа по32 outputs. Одинаковый binary и transport:
mmap, reader1, chunk8МиБ, slab16, decay65536, tensor delivery/packed guards1,
D2D/fills/early refill/prefill admission0. Cache request14ГиБ с live clamp95%.
MTP depth1/p_min0.7. Загрузка/prefill/первый output исключены из throughput;
sampling и экспорт logits входят. Разный объём фактического cache учитывается
как часть готовых режимов, а не как изоляция одного kernel.

Всего90 ответов /2880 IDs; по18 timed запросов и558 выходных шагов после
первого токена на каждый режим. Все IDs совпали. No-draft logits также
побитно совпали с сохранённым MIMO-17 и между контрольными процессами.

| Режим | Процесс1 | Процесс2 | Вместе, ток/с | Cache allocations, ГиБ | H2D, ГиБ/выходной шаг |
|---|---:|---:|---:|---:|---:|
| Без MTP | 9,159 | 9,921 | **9,525** | 11,355 | 1,142 |
| Q4, MIMO-17 head-columns1 | 9,776 | 9,982 | **9,878** | 10,520 | 1,150 |
| Q4, mode7/head-columns0 | 10,509 | 10,385 | **10,447** | 10,541 | 1,145 |

Новый режим: **+5,76% к прежнему Q4**, +9,68% к объединённому no-draft
контролю и **+5,30% к быстрейшему no-draft процессу**. No-draft controls
различаются на8,32%, поэтому выигрыш над их средним нельзя считать точным
прогнозом для другой нагрузки. Исторические10,929 ток/с MIMO-17 относятся
к другой серии; новый эксперимент не доказывает прироста над этой цифрой.

| Тема | Без MTP | Q4 MIMO-17 | Q4 mode7 |
|---|---:|---:|---:|
| Счёт | 12,641 | 12,999 | 14,797 |
| Код | 7,776 | 8,284 | 8,409 |
| Русский | 9,322 | 9,427 | 9,933 |

У нового режима222/228 proposals приняты (97,37%, покрытие222/558=39,78%),
у прежнего228/228. Target cycles558→336, против330 у прежнего Q4.
Draft+catch-up занимают1,25% generation time. Кеш почти одинаков с прежним
Q4; H2D на выходной шаг ниже лишь на0,45%. Изолированный вклад каждого из
трёх CUDA изменений в скорость здесь не измерялся.

Среднее полное время короткого timed запроса:6379,24 /6220,69 /6214,92 мс.
Новый режим ускоряет generation, но полное время запроса с prefill относительно
прежнего MTP практически не изменилось. Это не cold TTFT.

Все30 ответов нового режима /960 IDs совпали;930 decode строк logits exact,
первый prefill logit в каждом ответе отличается до0,000195.
GPU/drain audit PASS. Sampled global peaks: VRAM29,512ГиБ /92,68%,
RAM113,809ГиБ /90,64%. Предел95% соблюдён. Это секундные глобальные выборки
с другими приложениями; они могут пропустить краткие пики.

## Воспроизведение

Raw запросы, captures, logits, logs и команды:
`build-local/mimo2-validation/tokenwise`. Snapshots v1/v2/v3 сохраняют исходники,
сгенерированный CUDA и executable с SHA256. V1 — bits1/2, v2 добавляет FA,
v3 также ограничивает пакет на KV boundary. Измеряется v3:

Spec executable SHA256:
`d622b05d418094471b2cc3c18d6fb73686da13942c5f2b0cf72270f9c997295b`.

```powershell
python -m tools.check_mimo2_speculative --build build-local/mimo2-tokenwise-v3/build-local/mimo2-cuda --model H:/models/mimo-v2.6-flash/MiMo-V2.6-Flash-RL-GSQ-RCO-3.5bit.gguf --kind mtp --draft H:/models/mimo-v2.6-flash/mtp-MiMo-V2.6-Flash-MOPD-Q4_0.gguf --requests build-local/mimo2-validation/tokenwise/requests-q4.json --target-head-columns 0 --tokenwise-matmul 7 --output-dir build-local/mimo2-validation/tokenwise-new --reference-dir build-local/mimo2-validation/tokenwise/a0-none
```

Destination должен быть новым. Старые snapshots запускать сохранённым рядом
runner: нынешний передаёт дополнительные CLI flags.
[Проверки и provenance](MIMO26_FLASH_TOKENWISE_VALIDATION.json).

Основной production executable сохранён побитно, SHA256
`f56fc8f97d140e743e54d06fdd443b3016a2f1944a01b82e34988ed67532b771`.
CMake возвращён к SPEC_PROBE=OFF. Новые опции default0; MTP serving off.

Следующий MIMO-19: устранить остаток first-logit prefill и измерить cutoff
после полной decode parity; для работы без MTP остаются direct resident weights,
групповой H2D, chunks4/16 и дальнейшая работа с синхронизацией.
