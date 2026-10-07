# MIMO-16 — повтор MTP Q4 после оптимизаций конвейера

2026-10-07, Windows, Ryzen9 9950X, RTX5090 32GB, RAM128GB,
CUDA13.0.48 / MSVC19.44.35222.0 / driver581.80.
Target: `MiMo-V2.6-Flash-RL-GSQ-RCO-3.5bit.gguf`,134982426368 байт.
Draft: `mtp-MiMo-V2.6-Flash-MOPD-Q4_0.gguf`,1264898272 байт.
Оба файла из `H:\models\mimo-v2.6-flash`.

**Генерация без MTP остаётся быстрее:10,52 против9,01 ток/с.**
Scatter-copy вместе с MTP даёт9,08 ток/с; надёжного дополнительного выигрыша
по этим повторам нет. MTP не включён по умолчанию и не добавлен в serving.

Использован лучший экономный кандидат из [MIMO-06](MIMO26_FLASH_SPECULATIVE_COMPARISON.md):
Q4_0, первая голова, depth1, `p_min=0.7`, собственные embedding/head sidecar.
Переквантования, загрузки новых моделей и изменения весов не было.

[Замеры и команды](MIMO26_FLASH_MTP_Q4_RETEST_BENCHMARK.json),
[качество, диагностика и SHA](MIMO26_FLASH_MTP_Q4_RETEST_VALIDATION.json).

## Условия сравнения

Использован текущий transport с плотными slab16 МиБ, частотным допуском decay65536,
доставкой по тензорам и пакетными guard tails. Mmap, reader1, chunk8 МиБ,
prefill admission/early refill/grouped fills off. Все модельные матрицы на CUDA.
Cache request14 ГиБ, общий live clamp95% RAM/VRAM; фактический cache различается
из-за draft и рабочих буферов. Это сравнение готовых режимов на одном PC,
а не изоляция стоимости одного вычислительного ядра при одинаковом cache.

Контекст512, batch8, F32 KV, FA on, greedy, thinking off, CUDA Graphs/TF32 off.
Физический SWA KV полный (`swa_full=true`) для rollback. Каждый запрос начинает
с чистого KV; expert cache сохраняется внутри процесса. Для MTP до наполнения
кеша прогрет target batch8 с logits всех позиций и draft; прогрев входит в load.
Без draft не удерживается этот неиспользуемый verify workspace.

Порядок **A B C C B A**: без MTP; Q4; Q4 со scatter; повтор scatter;
повтор Q4; возврат без MTP. В каждом процессе три темы: счёт, код, русский текст.
На тему два исключённых прогрева и три timed ответа по32 токена. Итого90 ответов.
Скорость —558 выходных шагов после первых токенов за два процесса каждого режима,
делённые на generation time. Draft/verify/catch-up/rollback/sampling и экспорт
logits входят в timing; load, prefill и первый выходной токен исключены.

## Результаты

| Режим | Процесс1 | Процесс2 | Объединённые ток/с | Средний cache allocations | Decode H2D на выходной шаг |
|---|---:|---:|---:|---:|---:|
| Без MTP, D2D0 | 10,663 | 10,380 | **10,520** | 12,00 ГиБ | 1,051 ГиБ |
| Q4 MTP, D2D0 | 9,168 | 8,867 | **9,015** | 9,13 ГиБ | 1,246 ГиБ |
| Q4 MTP, scatter D2D2 | 9,601 | 8,605 | **9,076** | 9,13 ГиБ | 1,250 ГиБ |

Q4 MTP медленнее на14,31%, MTP со scatter — на13,73%. Даже лучший MTP-процесс
уступает обоим контролям. Средний полный запрос:5738,34 /6631,83 /6636,48 мс
соответственно; это не cold TTFT.

| Тема | Без MTP | Q4 | Q4 +scatter |
|---|---:|---:|---:|
| Счёт | 13,139 | 10,600 | 12,516 ток/с |
| Код | 8,981 | 7,790 | 7,601 ток/с |
| Русский текст | 10,235 | 9,084 | 8,398 ток/с |

Оба варианта MTP приняли228 из228 предложений в18 timed запросах. Это100%
принятия **после confidence cutoff**; предложения покрыли228 из558 выходных
шагов (40,86%). Число target cycles снизилось с558 до330.
Draft плюс catch-up заняли около1,13% времени генерации. Остальное преимущественно
приходится на target с транспортом. Cache уменьшился примерно на2,87 ГиБ,
а H2D на выходной шаг вырос на18,5–19,0%. Изменение кеша включает влияние
весов, рабочих буферов и live clamp; отдельных замеров allocator pools нет.

Таким образом, высокий acceptance не компенсировал дополнительную доставку
экспертов и стоимость target verification. Причинность каждого kernel по этим
wall-clock метрикам не установлена. Сравнивать9,01 напрямую с прежними5,55
как чистый эффект одной оптимизации нельзя: изменились transport, прогрев,
cache policy и состояние общей памяти. Контроль для вывода — текущие10,52.

Sampled global peaks RAM/VRAM: без MTP87,97%/92,67%; Q4 89,35%/92,66%;
Q4 +scatter93,82%/92,79%. Предел95% сохранён. Это секундные выборки с другими
процессами, не гарантированные пики между выборками и не только память MiMo.

## Совпадение результатов и известная проблема target

Все60 ответов MTP из основной серии, включая прогревы, совпали по IDs с
обычным greedy:1920 проверенных выходных токенов. Второй контроль без MTP
совпал с первым побитно по всем logits и480 IDs. Первый контроль также
совпал по480 IDs с сохранённой production серией MIMO-15; это проверка IDs,
не сравнение скорости probe с обычным engine, использующим compact SWA.

**Logits MTP не побитно равны последовательному target.** Статус `pass`
в этом offline runner означает совпадение выходных IDs, а не общую lossless parity.

Повторён oracle-тест: без draft weights и без target feature extraction
пакетной проверке depth1 подаются заранее известные baseline tokens. Она снова
расходится на18-м выходном токене (zero-based17):129258 →38379. Prefill logits
побитно совпали; максимальная разница logits на этой позиции0,3382081985473633.
Первый rejection — цикл8, после изменения target top-1; все предыдущие предложения
приняты. Сигнатура совпадает с MIMO-06, транспортные оптимизации её не устранили.
Failed report сохранён как `output_mismatch`, критерии не ослаблены.

Отдельный длинный prompt через SWA128: оба варианта Q4 совпали с контролем
по16 выходным IDs, приняли7/7 предложений. Холодное время этих одиночных
boundary запросов не использовано в таблице скорости. Это дополнительная
проверка состояния; она не устраняет найденное расхождение в oracle.

30 Python tests и10 C++ verified-prefix cases PASS; GPU audit/drain всех
завершённых запросов PASS. Обычный production executable сохранён побитно,
его SHA256 `f56fc8f97d140e743e54d06fdd443b3016a2f1944a01b82e34988ed67532b771`.
Конфигурация основной сборки возвращена к `STRATA_MIMO_SPEC_PROBE=OFF`.

Это ограничение точной эквивалентности, а не доказательство худшего смысла текста.
Переход к serving, stochastic sampling, отмене и sessions требует отдельной работы.
MTP heads2/3 в этом опыте не реализованы.

## Изменения инструментов и повторение

Offline probe получил явный `--expert-cache-mib` и метрики режима копирования,
slab/decay/guards, CUDA launches, очередей и прогрева verification. Runner
принимает `--d2d-batch 0|1|2`, явно фиксирует текущие настройки transport,
проверяет ответные метрики и сохраняет environment/команды/hash requests.
Алгоритм draft/verify и арифметика модели не менялись.

Измеренный snapshot исходников/executable/manifest:
`build-local/mimo2-mtp-q4-current-measured`.
Spec executable SHA256:
`8e6e2d15dca56be6d315abe03bea39a9f80969a029ecbac11986c6141868b086`.
Raw logs, logits и requests: `build-local/mimo2-validation/mtp-q4-current`.
Полная последовательность запусков сохранена в JSON отчётах.

```powershell
python -m tools.check_mimo2_speculative --build build-local/mimo2-mtp-q4-current-measured/build-local/mimo2-cuda --model H:/models/mimo-v2.6-flash/MiMo-V2.6-Flash-RL-GSQ-RCO-3.5bit.gguf --kind mtp --draft H:/models/mimo-v2.6-flash/mtp-MiMo-V2.6-Flash-MOPD-Q4_0.gguf --requests build-local/mimo2-validation/mtp-q4-current/requests-q4.json --expert-cache-mib 14336 --d2d-batch 0 --output-dir build-local/mimo2-validation/mtp-q4-new --reference-dir build-local/mimo2-validation/mtp-q4-current/a0-none
```

Команда использует сохранённую spec-сборку. Для пересборки включить
`STRATA_MIMO_SPEC_PROBE=ON`; порядок configure/build есть в
[README](../../backends/mimo2/README.md#отдельное-сравнение-mtp--dflash).
Следующий предметный шаг — исследовать target batch2: отдельно CUDA dispatch
BF16/Q2_K/Q3_K/MXFP4 и рабочие буферы, затем применимость одиночной арифметики
и короткого MMVQ из GLM. GLM-патч не считается готовым для геометрии MiMo.
Сначала fixtures и oracle parity, затем новый Q4 A/B; включать MTP на основании
одного acceptance нельзя.
