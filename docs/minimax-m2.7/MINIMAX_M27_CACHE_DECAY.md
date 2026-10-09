# MM27-32 — период затухания GPU cache history

Эксперимент 2026-10-09: перенос идеи настройки frequency decay из GLM в MiniMax.
**Итог: общий default65536 сохранён.** Периоды131072/262144 дали всего
+1,25/+1,53% по медиане агрегата, замедлив новую тему на2,79/6,87%.
Для возврата к прежней теме они полезны; универсальное ускорение не подтверждено.
Новая настройка остаётся доступной в отдельной native-сборке; server EXE98e85e80…
и его admission SHA сохранены.

## Реализация

Native benchmark принимает `--cache-decay-period N`, диапазон1..4294967295,
default65536. Значение записывается в `cache_decay_period` JSON header.
`strata_mm27_cache_configure` передаёт его в конструктор общего matrix cache;
существующие вызовы MiniMax и Step сохраняют прежний default65536.
Настройка фиксируется при создании cache, не меняется между запросами.
Общий алгоритм частотной истории не изменён.

Каждые N обучающих обращений lazy decay делит частотные оценки на два.
Это не сброс всей истории. При grouping off один обучающий decode шаг делает
62×3×8=1488 обращений: периоды65536/131072/262144 соответствуют примерно
44/88/176 шагам. Prefill и первый serial шаг используют уже готовые cache hits,
но не обучают историю и не заполняют cache. Счётчики запроса сбрасываются,
частотная история и резидентные эксперты сохраняются между запросами.

## Методика

RTX5090 32607МиБ, Ryzen9 9950X, RAM125,555ГиБ; Windows, driver581.80,
CUDA13.0, MSVC19.44.35222.0. Модель:
`H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf`.
138342384352 байта; header SHA256
`9a011b2ad6fb20db3bb0cdfe9af08665939ee1f0cb0609ffe2bfee7f31c825db`.

Context2048/batch16, strict F32 activations/KV, TF32/FA/graphs/MTP/DFlash off.
Cache18ГиБ, arena64МиБ/reserve0, file reader, readers2/chunk4МиБ/lookahead1/
D2D batch1; grouping/RAM expert cache/prefix/session archive off.
ОС управляет file cache; «пустой cache» означает GPU expert cache, не холодный SSD.

Один процесс старого EXE используется только для проверки численной точности.
Затем по три отдельных процесса на период, в порядке:

1. 65536,131072,262144.
2. 131072,262144,65536.
3. 262144,65536,131072.

В каждом процессе четыре greedy запроса по128 output tokens: A — объяснение
синего неба; повтор A; B — Python LRU cache; возврат A. KV каждый раз свежий,
экспертный cache сохраняется. Это короткий decode workload, не проверка качества
или естественного завершения ответов. Данные по каждому запросу сохраняются.
Скорость агрегата = сумма decode forward tokens / сумма decode time;
первый токен относится к prefill. Полное время включает prefill и выдачу ответа,
но не загрузку модели. Сравниваются медианы трёх процессов; старый EXE в них не входит.

Сохраняются все F32 logits, IDs, H2D, hit bytes, fills, evictions, timings и
Windows/NVML samples каждые0,5с. Выход за95% RAM/VRAM или ошибка наблюдателя
останавливает только собственный дочерний процесс. Каждый процесс ограничен
Windows Job и timeout; хеширование начинается после остановки наблюдателя.
Сравнение logits читает ограниченные порции, без загрузки массива в RAM целиком.

## Измерения

Медианы трёх независимых процессов на вариант. В агрегате508 forward decode
steps: четыре ответа по128 tokens, первые токены относятся к prefill.

| Период | Decode, токенов/с | Изменение | Полное время корпуса, с | H2D decode, ГиБ | Cache fills, ГиБ | Вытеснения |
|---|---:|---:|---:|---:|---:|---:|
| 65536 | 4,120 | — | 163,804 | 984,085 | 125,150 | 40711 |
| 131072 | 4,171 | +1,25% | 161,512 | 987,351 | 77,260 | 22633 |
| 262144 | 4,183 | +1,53% | 160,484 | 988,071 | 63,944 | 17602 |

H2D, fills и количество вытеснений полностью повторились во всех трёх процессах
одного варианта. Скорости по повторам, токенов/с:

| Период | Круг1 | Круг2 | Круг3 |
|---|---:|---:|---:|
| 65536 | 4,038 | 4,145 | 4,120 |
| 131072 | 4,171 | 4,115 | 4,189 |
| 262144 | 4,109 | 4,188 | 4,183 |

Медианы отдельных сценариев:

| Сценарий | 65536 | 131072 | 262144 |
|---|---:|---:|---:|
| A, пустой GPU cache | 4,175 | 4,167 | 4,156 |
| Повтор A | 4,504 | 4,576 | 4,621 |
| B, новая тема | 3,866 | 3,758 | 3,600 |
| Возврат A | 3,988 | 4,271 | 4,499 |

На возврате A прирост составляет7,08/12,81%, но на новой теме — замедление
2,79/6,87%. Корпус содержит три запроса A и один B; агрегат благоприятен для
сохранения старой истории. Изменение default для чата со сменой тем не оправдано.
Грубое сокращение вытеснений на44,41/56,76% не уменьшило основной H2D-трафик.
Fills — отдельные D2D копии в GPU cache, их нельзя считать экономией RAM→VRAM.

Следующий кандидат — CUDA event dependencies вместо оставшихся CPU ожиданий.
В контрольном круге2 на508 decode steps было94488 scratch fences и94488 copy
fences; delivery46,899с при decode122,562с. Delivery включает ожидание источника,
а producer timers перекрываются, поэтому эти числа не предсказывают возможный
speedup. Нужны отдельный A/B, lifetime/pins/drain и cancel/pressure проверки.

## Проверки и границы

40 native requests /5120 output tokens в10 последовательных процессах.
Все4608 IDs и921894912 F32 logits девяти candidate-процессов побитно совпали
со старым EXE; finite checks PASS. Это проверка реализации относительно прежнего
движка, не независимый model oracle. Все три65536 процесса сохранили прежние
H2D/hit/fill/eviction/reuse решения, включая omitted CLI default в круге1.

19 history/admission checks,10 CLI boundary checks,5 CPU test methods и216
cache regression checks PASS. Последние включают cuda/arena, F32/mixed fixtures,
2/256МиБ caps, prefill hits, OOM/pressure, cancel, worker faults и reload с
readers2/chunk4/lookahead/D2D batch. Native cache regression выполнена после
окончания speed-серии, в `build-local/minimax-m2-cache-decay-fixture-01`.

285 итоговых gates PASS:86 source snapshot hashes,57 основных artifact hashes,
profile/accounting/drain, IDs/logits records, пересчитанные метрики и все3249
независимых memory samples. Максимумы: RAM23,608%, VRAM85,654%;95% не превышено.
Реального давления около95% в этой серии не было. Все процессы завершились,
GPU память освобождена. [Машинный итог проверки](MINIMAX_M27_CACHE_DECAY_CHECK.json).

Серия не закрывает MM27-06 native-after-cache discrepancy, wide quality/model
oracle, длинные batch1/8 sessions, shift и P6. Профили контекста и API defaults
не меняются. DFlash/MTP не участвовали.

## Воспроизведение

Собрать `strata-minimax-m2-bench`, `strata-minimax-m2-cache-decay-check` и
`strata-minimax-m2-cache-check` обычной изолированной CUDA-сборкой. До пересборки
сохранить старый EXE и manifest; benchmark принимает явные пути `--engine` и
`--baseline`. Кандидат этого опыта сохранён отдельно:
`build-local/minimax-m2-cache-decay-candidate/engine.exe`, SHA256
`36fbac1185a69a31b627eb818feb9f4193741937d42c391bcb44ebd64b4d9efa`.
Эталон98e85e80… — `build-local/minimax-m2-cache-decay-baseline/engine.exe`.

```powershell
python -m unittest tools.test_minimax_m2_cache_decay
build-local/minimax-m2-cuda/bin/strata-minimax-m2-cache-decay-check.exe > build-local/minimax-m2-cache-decay-unit.json
python tools/check_minimax_m2_cache_decay.py --gguf H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --out build-local/minimax-m2-cache-decay-01
python tools/run_minimax_m2.py --engine build-local/minimax-m2-cuda/bin/strata-minimax-m2-cache-check.exe -- build-local/minimax-m2-cache-decay-fixture-01 --pipeline-readers 2 --pipeline-chunk-mib 4 --pipeline-lookahead 1 --pipeline-d2d-batch 1
python tools/validate_minimax_m2_cache_decay.py --source build-local/minimax-m2-cache-decay-01 --fixture build-local/minimax-m2-cache-decay-unit.json --regression build-local/minimax-m2-cache-decay-fixture-01 --output docs/minimax-m2.7/MINIMAX_M27_CACHE_DECAY_CHECK.json
```

Runner не меняет server admission SHA и не устанавливает настройки сервера.
Каталог результата должен быть новым. Build-local хранит EXE, manifests,
исходники с hashes, полный корпус, JSON/F32 и логи каждого процесса.
