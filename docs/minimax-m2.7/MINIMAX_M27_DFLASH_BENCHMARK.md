# MiniMax-M2.7: измерение полезной скорости DFlash

Измерено 2026-10-08 на Windows, RTX 5090 / 32607 МиБ, driver581.80,
RAM125.555 ГиБ, CUDA13.0.48, MSVC19.44.35222.0. Это отдельный offline greedy
driver; serving и defaults MiniMax не включают DFlash.

Лучший DFlash в подтверждающей серии — Q4_K_M/depth1: **3.675 токена/с**,
off/cap18 — **3.550**. Номинальные+3.54% меньше разброса
обычных запусков; устойчивый выигрыш не подтверждён. Defaults сохранены.

Target — `H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf`.
Draft Q3_K_M, Q4_K_M, Q5_K_M находятся рядом. Идентичность, словари и границы
тензоров приведены в [инспекции](MINIMAX_M27_DFLASH.md). Полный SHA-256 target
не вычислялся; header SHA и полные SHA трёх draft проверены.

## Методика

Добавлены `strata-minimax-m2-speculative-bench`, `dflash_decode.hpp` и
`tools/bench_minimax_m2_dflash.py`. Driver получает первый токен от target,
строит блок из anchor и MASK, проверяет предложения target, принимает только
совпавший greedy prefix и target correction/bonus, удаляет rejected KV и
передаёт draft только committed features. Позиции обеих историй проверяются
после каждого цикла. Target живёт дольше заимствующего draft.

Три запроса проходят настоящий локальный chat template: объяснение цвета неба,
Python-функция суммы чётных чисел и русское объяснение RAM/VRAM. Prompt lengths
48/58/59 токенов. Каждый вариант запускается отдельным процессом; внутри него
модель и GPU cache остаются загруженными между запросами, KV очищается.
Перед измерением выполняется warmup; load/warmup записан отдельно.
Windows file cache и посторонняя нагрузка не контролировались.

Условия: context512, batch/ubatch8, target strict F32, F32 KV, FA/graphs off,
file pipeline2 readers/chunk4 МиБ, lookahead/D2D batch on, arena64 МиБ/reserve0.
Cache cap18 или24 ГиБ — верхняя граница; actual backing ограничен global95%
и внутренним запасом. RAM/VRAM guards не ослаблялись. RAM file cache управляет
Windows; полного собственного RAM cache здесь нет.

Полезная скорость = сумма `(generated_tokens - 1)` / сумма generation time.
Время включает draft, target verification, catch-up, sampling и проверки
цикла; первый токен относится к TTFT. Не считать предложенные или проверенные
токены выданными. Полные logits сохраняются после остановки таймера; их чтение
в RAM и greedy sampling входят в цикл у обоих вариантов. Дополнительно
сохранены TTFT, prefill/request time, acceptance и H2D на полезный токен.
Тексты длиной32/64 токена часто ещё содержат reasoning; это не тест законченных
ответов, длинного контекста или API throughput.

Все IDs и **каждый элемент full target logits** сравниваются с обычной
генерацией от первого output token. Gates прежние: max_abs≤5e-4, NMSE≤1e-7,
finite logits, одинаковые shapes, prompt lengths, stop reasons и IDs.
Не прошедший вариант исключается из таблиц скорости, его артефакты сохраняются.

## Почему понадобилась другая арифметика verification

Первый Q4/depth2 дал одинаковые96 output IDs, но провалил проверку logits:
max_abs1.566925 в русском запросе. Сохранённая трасса нашла первое изменение
MoE routing в позиции84, слой30. В serial scores экспертов76 и160 равны
2.3240184783935547; в batch score76 ниже на один float ULP.
Восьмой эксперт меняется с76 на160, хотя первая семёрка совпадает.
Совпадения greedy IDs недостаточно, чтобы принять такое расхождение.

`Tokenwise.cmake` добавляет opt-in `STRATA_MM27_TOKENWISE`:
bit1 считает обычные matmul (включая attention) по отдельным колонкам,
bit2 — routed matmul по токенам. Benchmark использует mode3 только во время
target verification; prefill/draft выполняются с mode0. Expert delivery и
verification graph остаются пакетными. Это сохраняет арифметику serial в
проверенном корпусе, но сокращает возможности ускорения compute.
Режим0 по умолчанию сохраняет прежний путь.

Разделить только матрицы весов оказалось недостаточно: depth4 снова менял
routing через округление attention. Окончательный вариант учитывает и attention.
В итоговом screen все42 сравнения запросов дали **побитное совпадение** logits.
Ни target weights, ни архив dependency, ни tolerance не изменялись.
Причина старого tiny native-after-cache FAIL MM27-06 этим не закрывается.

Диагностические запуски находятся в `build-local/minimax-m2-dflash-trace-01`.
Их время не является benchmark: callback читает промежуточные GPU tensors.
Неудачные ранние замеры сохранены в `minimax-m2-dflash-speed-initial-01` и
`minimax-m2-dflash-speed-screen-01`; они не входят в итоговые рекомендации.

## Screen квантов и длины блока

По32 output tokens на запрос, один запуск каждого варианта, cap18 ГиБ.
Depth означает число предложений **без anchor**, максимум7 при block_size8.
Все строки этой таблицы прошли побитное сравнение.

| Draft | Depth | Полезных токенов/с | Принято / предложено |
|---|---:|---:|---:|
| off, начало | 0 | 3.684 | — |
| Q4_K_M | 1 | 3.775 | 44/47 |
| Q4_K_M | 2 | 3.758 | 59/66 |
| Q4_K_M | 4 | 3.213 | 67/96 |
| Q4_K_M | 7 | 2.310 | 67/156 |
| Q3_K_M | 1 | 3.680 | 44/47 |
| Q3_K_M | 2 | 3.241 | 59/66 |
| Q3_K_M | 4 | 3.203 | 67/96 |
| Q3_K_M | 7 | 2.385 | 68/148 |
| Q5_K_M | 1 | 3.759 | 44/47 |
| Q5_K_M | 2 | 3.662 | 57/67 |
| Q5_K_M | 4 | 3.222 | 67/96 |
| Q5_K_M | 7 | 2.390 | 68/149 |
| off, конец | 0 | 3.699 | — |

Это отбор кандидатов, а не статистически подтверждённое ранжирование квантов.
Особенно заметен выброс первого запроса Q3/depth2. Короткие блоки Q4/depth1/2
выбраны для повторного сравнения. На cap18 off занимал около83.4% VRAM,
варианты DFlash — около93–94%; поэтому одного сравнения при равном cache cap
недостаточно для вывода о полезности draft.

## Проверка памяти и три повтора

Дополнительный32-token screen с cap24 ГиБ:

| Режим | Полезных токенов/с | Все три запроса, с | Peak global VRAM |
|---|---:|---:|---:|
| off-24 | 3.867 | 58.397 | 94.204% |
| q4-depth1-24 | 3.661 | 60.647 | 94.211% |
| q4-depth2-24 | 3.675 | 60.934 | 94.211% |

Cap24 не улучшил DFlash относительно cap18 в этом screen. Для подтверждения
выбраны Q4/depth1/2 с cap18 и off с возвращённой памятью (cap24).
По64 output tokens на запрос, три повтора, чередование прямого/обратного
порядка трёх вариантов. Затем выполнены ещё три off/cap18 с теми же запросами
и полным сравнением с первым off/cap24. Эта дополнительная серия шла после
основной, поэтому сравнение при равном cap18 не является чередующимся A/B.

| Режим | Повторы, полезных токенов/с | Медиана | Медиана времени трёх запросов, с | Acceptance |
|---|---|---:|---:|---:|
| off-18 | 3.801 / 3.550 / 3.118 | 3.550 | 91.482 | — |
| off-24 | 3.533 / 3.971 / 3.364 | 3.533 | 95.433 | — |
| q4-depth1-18 | 3.675 / 3.686 / 3.607 | 3.675 | 87.191 | 82.52% |
| q4-depth2-18 | 3.548 / 3.051 / 3.497 | 3.497 | 89.967 | 75.68% |

Для каждого повтора сначала агрегируется весь workload, затем берётся медиана
трёх агрегатов. Медианы отдельных запросов в JSON не складываются в такую
медиану: медленные выбросы приходились на разные запросы.

Q4/depth1 номинально быстрее off/cap18 на **3.54%** по медиане.
Это малый эффект на фоне разброса off и разных условий последовательных серий;
устойчивое ускорение не доказано. По этим данным DFlash default остаётся off,
экспериментальные18/2/4 и arena64/reserve0 сохранены.
Если вручную исследовать DFlash на этом корпусе, кандидат — Q4_K_M/depth1,
verify_tokenwise3, cap18. Это не универсально оптимальный production profile.

Peak global VRAM итоговых серий не выше94.212%, RAM —39.345%; guards95% прошли.
В64-token серии off/cap24 реально выделял до21.5625 ГиБ arena (resident20.89),
а on/cap18 —18 ГиБ arena (resident17.44). Нельзя считать cap24 фактической
аллокацией24 ГиБ. Внешняя нагрузка и Windows file cache не изолировались;
глобальную RAM нельзя приписывать одному процессу. Bench сохраняет sampled
global peaks; process private/working-set peaks отдельно в нём не экспортированы.

У Q4/depth1 в трёх64-token повторах draft занял3.743 с, verification150.678 с,
catch-up0.366 с: draft compute составляет около2.4% суммарного generation time.
H2D decode —2.013 ГиБ на полезный токен, у off/cap24 —1.801 ГиБ.
Таким образом, сам маленький draft здесь не основная статья времени;
проверка target и доставка экспертов остаются дорогими. Это фазовые host timings,
а не CUDA kernel profiler. Большие depth также тратят работу на отвергнутый хвост.

## Артефакты и числовые проверки

Итоговые четыре серии:29 запусков,87 сравнений запросов,3936 output tokens;
все IDs и787451904 сравниваемых элементов target logits совпали побитно.
Все запросы закончились по лимиту, а не EOS. Старые failed numerical runs
сохранены и в эти результаты не входят.

- [Screen12 DFlash и2 off](MINIMAX_M27_DFLASH_SPEED_SCREEN_CHECK.json).
- [Cap24 screen](MINIMAX_M27_DFLASH_SPEED_MEMORY_CHECK.json).
- [Три повтора64 tokens](MINIMAX_M27_DFLASH_SPEED_CONFIRM_CHECK.json).
- [Дополнительные три off/cap18](MINIMAX_M27_DFLASH_SPEED_OFF18_CHECK.json).
- [Build manifest подтверждающей серии](MINIMAX_M27_DFLASH_SPEED_BUILD_MANIFEST.json).
- [Диагностика router tie](MINIMAX_M27_DFLASH_ROUTING_DIAGNOSTIC.json).
- [Регрессии и acceptance self-test](MINIMAX_M27_DFLASH_SPEED_VALIDATION_CHECK.json).

После timings пересобраны обычный `strata-minimax-m2-bench` и runtime fixtures.
Host self-test прошёл1404 сочетания EOS/reject/bonus/budget; GPU runtime —49/49.
Сохранённый старый обычный EXE и новый обычный EXE независимо сгенерировали
три32-token запроса. Оба совпали с off нового speculative driver:
96 token IDs и19206144 logits каждый, побитно. Совпали и SHA полных logit files.
Это проверка default-mode regression и независимого serial driver;
её timings в таблицы скорости не включены. Сборки и проверки завершились exit0.
Команды дополнительной проверки:

```powershell
cmake --build build-local/minimax-m2-cuda --target strata-minimax-m2-bench strata-minimax-m2-runtime-check -j 6
python -X utf8 build-local/validate-minimax-dflash-speed.py
```

Validation script, два EXE и логи сохранены в `build-local`; runner использовал
унаследованные strict precision variables. Самостоятельно runtime fixtures
запускаются через `tools/run_minimax_m2.py`, передав executable и новый каталог.

Каждый JSON содержит hashes запускаемого EXE, source snapshot, target header,
команды, отдельные timings и полные сравнения. Полные logits, drafts inventory,
EXE/source snapshots и stderr лежат в соответствующих каталогах `build-local`,
названных выше; исходный32-token off reference —
`build-local/minimax-m2-dflash-speed-initial-01/01-off-18`.
Build основан на dependency `86ebfef2`, Strata HEAD
`ea56544abb2a52c8eeace467265cc1639ea172b0`, рабочее дерево dirty.
Screen и подтверждение различаются только переносимым setter environment
и проверкой его return code; математика и CUDA patch одинаковы.

## Воспроизведение

Собрать отдельный Windows CUDA runtime согласно backend README:

```powershell
cmake --build build-local/minimax-m2-cuda --target strata-minimax-m2-speculative-bench -j 6
python tools/run_minimax_m2.py --engine build-local/minimax-m2-cuda/bin/strata-minimax-m2-speculative-bench.exe -- --self-test
```

Config JSON — массив с уникальными `name`, `depth`, `cache_mib`;
для depth>0 также `draft` и `verify_tokenwise: 3`. Пример:

```json
[
  {"name":"off-24","depth":0,"cache_mib":24576},
  {"name":"q4-depth1-18","depth":1,"cache_mib":18432,"draft":"H:/models/MiniMax-M2.7/MiniMax-M2.7-DFlash-Q4_K_M.gguf","verify_tokenwise":3},
  {"name":"q4-depth2-18","depth":2,"cache_mib":18432,"draft":"H:/models/MiniMax-M2.7/MiniMax-M2.7-DFlash-Q4_K_M.gguf","verify_tokenwise":3}
]
```

Сохранить его как новый `build-local/dflash-config-new.json`, затем:

```powershell
python -X utf8 tools/bench_minimax_m2_dflash.py --model H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --configs build-local/dflash-config-new.json --out build-local/dflash-results-new --tokens 64 --repeats 3
```

Output directory должен отсутствовать. Порядок вариантов меняется на обратный
в каждом втором повторе. Runner сохраняет EXE, source snapshot, manifest,
запросы, полные logits, stderr и сравнения. CPU hashing/comparison выполняется
после завершения GPU-процесса. Одновременно другие GPU-тесты не запускать.
`--continue-on-parity-failure` позволяет закончить независимые варианты, но
сохраняет общий FAIL, исключает неверные timings и возвращает ненулевой exit.

## Границы этапа

Реализованы offline greedy цикл, положительный acceptance, correction/bonus,
rollback, восстановление draft features и полные числовые сравнения.
EOS/reject positions/output budget покрыты отдельно host self-test; live GPU
EOS/cancel/pressure/recovery, session reuse, sampling, длинный контекст и
API-интеграция этим замером не закрываются. DFlash остаётся экспериментальным.
