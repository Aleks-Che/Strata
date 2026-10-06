# Удачные оптимизации GLM-5.3-Flash в Strata

Сводка на **2026-10-06**, `Asia/Yekaterinburg`. Собрана по сохранённым
проверкам 2026-10-04–05; новых замеров при составлении файла не проводилось.

Стенд: **Windows, Ryzen 9 9950X, 128 ГиБ RAM, RTX 5090 32 ГиБ**.
Основная модель: `H:\GLM-5.3-Flash-GGUF\GLM-5.3-Flash-Uncensored-IQ3_XXS.gguf`,
112,31 ГиБ. Зависимость GLM: Unsloth
`86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9`.

Здесь отдельно указаны выигрыш скорости всей генерации, сокращение работы
отдельного компонента и экономия памяти. Проценты из разных серий **не складываются**.
Повторяемый prompt B и чередование A/B — разные нагрузки. Большинство результатов
относятся к прогретым запросам, а не к первому ответу после загрузки.

## 1. Сохранённый рабочий вариант: 13,37 ток/с без MTP

Запуск: [run-glm53-no-mtp-13_37.bat](../../run-glm53-no-mtp-13_37.bat).
Профиль: [strata-glm53-no-mtp-13_37.json](../../strata-glm53-no-mtp-13_37.json).
Интерфейс — `http://127.0.0.1:8081`.

Для этого запуска сохранена отдельная копия движка
`build-local/glm53-no-mtp-13_37/strata-glm5next.exe` с SHA-256
`ae83ef59c5904fe911c0ab470fb720182851a6c1c8d56000bd8773b9f768be64`.
Последующие сборки основного backend её не перезаписывают.
BAT, JSON и бинарник — локальные файлы, исключённые из Git правилами проекта.

| Что измерено | Результат |
|---|---:|
| Прогретая генерация без MTP, медиана | **13,373 ток/с** |
| Диапазон пяти измеряемых запросов | **13,019–13,637 ток/с** |
| Медиана полного времени ответа с prefill | 7,528 с |
| Медиана основного GPU-кэша экспертов | 17,733 ГиБ |
| Передача весов H2D за запрос, включая prefill | 85,694 ГиБ |

Нагрузка: один и тот же B, **49 входных + 64 выходных токена**, greedy,
четыре warmup и пять timed запросов, состояние последовательности очищается
между запросами. Expert cache и файловый кэш ОС сохраняются. Скорость считается
как `63 * 1000 / DONE.decode_ms`, без загрузки модели, warmup и prefill.
Она не гарантируется для другого текста или длинного контекста.

Это удачная **совокупность настроек**, а не доказательство отдельного эффекта
каждого флага. В той же итоговой работе MTP1 дал 11,082 ток/с, а MTP1 с общим
scratch — 10,644 ток/с, с провалами. Поэтому выбран MTP off.
[Описание](GLM53_FLASH_COMPACTION_MTP.md),
[полные измерения](GLM53_FLASH_MTP_SHARED_SCRATCH_BENCHMARK.json).

## 2. Изменения с измеренным выигрышем скорости полной модели

### 2.1. CUDA events на целую экспертную матрицу

**Что сделано:** ожидания CPU при доставке экспертов заменены зависимостями
между CUDA streams. Одна пара событий защищает все выбранные диапазоны одной
матрицы вместо отдельной пары для каждого диапазона. Переиспользование памяти
остаётся защищено событиями завершения копирования и потребления.

**Результат:** на MTP1, prompt A 39+64, 4 warmup + 5 timed:
**8,510 → 9,586 ток/с, +12,64%**. Диапазоны: 8,489–8,661 и 9,553–9,655.
В предварительной серии число пар событий за запрос сократилось
с **69 969 до 4 608**; этот счётчик включает prefill.

**Настройка:** `STRATA_GLM_COPY_EVENTS=2`, включена в рабочих профилях.
Код: [pipeline_sched.inc](../../backends/glm5next/pipeline_sched.inc),
[runtime_memory.cpp](../../backends/glm5next/runtime_memory.cpp).
Все сравниваемые full-model logits совпали точно; проверены отмена и recovery.
[P3.3c](GLM53_FLASH_PIPELINE_MTP.md),
[benchmark](GLM53_FLASH_COPY_EVENTS_BENCHMARK.json),
[проверки](GLM53_FLASH_COPY_EVENTS_VALIDATION.json).

### 2.2. Несколько экспертных матриц в одном блоке VRAM

**Что сделано:** вместо отдельного CUDA allocation на каждую матрицу
используются блоки по 16 МиБ с выровненными slots. Последний блок уменьшается
под остаток бюджета, пустые блоки освобождаются. Это уменьшает потери на
размещение и позволяет держать больше полезных весов в той же VRAM.

**Результат:** MTP off, prompt A 39+64, одна сборка, 4 warmup + 5 timed:

| Метрика | Отдельные allocations | Slabs16 с уменьшением последнего блока |
|---|---:|---:|
| Медиана генерации | 9,865 ток/с | **11,764 ток/с** |
| Основной GPU-кэш | 12,902 ГиБ | **17,307 ГиБ** |
| Чтение весов за запрос, включая prefill | 110,680 ГиБ | **92,003 ГиБ** |

Рост скорости **19,26%**, дополнительная ёмкость кэша около **4,41 ГиБ**.
Первый запрос в этой серии стал медленнее: результат относится к прогретой
генерации. Все 45 поисковых и итоговых ответов совпали с reference;
9 912 320 сохранённых F32 logits совпали бит-в-бит.

**Настройка:** `STRATA_GLM_CACHE_SLAB_MIB=16`, включена.
Код: [expert_slab.hpp](../../backends/glm5next/expert_slab.hpp).
Это базовое размещение в общих блоках; дополнительный перенос весов между
блоками (`CACHE_COMPACT`) — отдельная оптимизация из раздела 5.
[P3.3e](GLM53_FLASH_PIPELINE_MTP.md),
[benchmark](GLM53_FLASH_SLAB_BENCHMARK.json),
[проверки](GLM53_FLASH_SLAB_VALIDATION.json).

### 2.3. Более длинная история частот основного кэша

**Что сделано:** период старения main history увеличен до 131072;
история MTP остаётся отдельной. Часто используемые матрицы реже теряют
приоритет и вытесняются из кэша.

**Результат подбора:** в серии с pool/reclaim среднее число вытеснений
сократилось примерно с **50 тысяч до 1 568** за запрос, source/H2D payload —
**145,62 → 126,32 ГиБ**. Медиана выросла **8,935 → 9,669 ток/с**.
В этом сравнении одновременно менялся reclaim: `decay4096/reclaim0` против
`decay131072/reclaim1`. Приписывать весь выигрыш одному decay нельзя.

С обычным allocator следующая серия дала **7,472 ток/с** при decay32768
и **9,866 ток/с** при decay131072. Итоговый профиль без MTP достиг
**9,902 ток/с** до внедрения slabs16. Это исторические результаты на A,
а не контроль для поздних 13,373 ток/с на B.

**Сохранены:** `STRATA_GLM_MAIN_CACHE_DECAY=131072`,
`STRATA_GLM_MEMORY_POOL=0`, `STRATA_GLM_POOL_RECLAIM=0`.
При включённых slabs именно slab allocator управляет размещением матриц;
переключение pool не следует считать дополнительным независимым ускорением.
Код: [expert_cache.hpp](../../backends/glm5next/expert_cache.hpp),
[runtime_memory.cpp](../../backends/glm5next/runtime_memory.cpp).
[P3.3d](GLM53_FLASH_PIPELINE_MTP.md),
[подбор](GLM53_FLASH_STAGING_CACHE_BENCHMARK.json),
[итоговое сравнение](GLM53_FLASH_STAGING_CACHE_MTP_BENCHMARK.json).

## 3. Рабочая основа, которую нужно сохранить

Эти механизмы подключены и проверены, но для каждого из них нет отдельного
сопоставимого замера полного инференса с единственным изменённым параметром.

| Механизм | Что он устраняет или обеспечивает | Где реализован |
|---|---|---|
| Доставка только выбранных экспертов | В RAM→VRAM идут выбранные router IDs и только cache misses; hit обслуживается из VRAM через D2D | [runtime_memory.cpp](../../backends/glm5next/runtime_memory.cpp), [expert_dispatch.hpp](../../backends/glm5next/expert_dispatch.hpp) |
| Асинхронная подготовка gate/up/down | После получения маршрута матрицы ставятся в порядок потребления; используются отдельный H2D stream и ограниченный ring | [expert_pipeline.hpp](../../backends/common/expert_pipeline.hpp), [expert_transport.hpp](../../backends/glm5next/expert_transport.hpp) |
| GPU-кэш с частотным допуском, LRU и leases | Повторные веса сохраняются между запросами; текущий план не теряет нужные ему allocations | [expert_cache.hpp](../../backends/glm5next/expert_cache.hpp) |
| Бюджеты RAM/VRAM 95% | Кэш может использовать доступную память с учётом системы, состояния модели и scratch | [expert_memory.hpp](../../backends/glm5next/expert_memory.hpp), [host_pages.hpp](../../backends/glm5next/host_pages.hpp) |
| Mmap и небольшой pinned ring | Не требуется закреплять через CUDA в RAM весь набор экспертов; память доставки ограничена размером кольца | [runtime.hpp](../../backends/glm5next/runtime.hpp), [runtime_memory.cpp](../../backends/glm5next/runtime_memory.cpp) |

Конвейер начинает работу после получения маршрута текущего слоя; предсказания
экспертов неизвестного следующего слоя здесь нет. CUDA trace подтвердил
перекрытие интервалов доставки и compute, но не доказал высокую одновременную
занятость GPU и H2D. [Описание и trace](GLM53_FLASH_PIPELINE_MTP.md).

При ранней интеграции кэша убран полный поиск/опрос eviction events на каждом
miss: введён явный LRU и уменьшено число CUDA allocations. В 90%-профиле
одиночный замер изменился **1,732 → 4,313 ток/с**. Это удачное устранение
дорогой работы CPU, но не серия для общего процента ускорения. Применявшийся
тогда async pool позднее заменён текущим размещением slabs16.
[P3.5h и ограничения замера](GLM53_FLASH_IMPLEMENTATION_STATUS.md),
[manifest](GLM53_FLASH_MEMORY_VALIDATION.json).

## 4. Ускорение отдельных операций и устранение лишней работы

### 4.1. Проверки logits без создания строк

В [runtime.hpp](../../backends/glm5next/runtime.hpp) добавлен
`require(bool, const char *)`. Диагностический литерал больше не преобразуется
в `std::string` на каждом успешном вызове внутри цикла по словарю.
Проверки NaN/Inf и сообщения об ошибках сохранены.

Инструментированный CPU benchmark: **9 912 320 → 0 allocations** на проверку
154 880 × 64 значений. В последовательных full-model сериях медиана draft
снизилась **256,6 → 133,7 мс**, но verify оставался дороже. Это не доказательство
фиксированного процента ускорения всей модели. Изменение включено в коде,
в том числе в сохранённой сборке 13,37 ток/с, и полезно также без MTP.
[Описание](GLM53_FLASH_COMPACTION_MTP.md),
[исходник microbenchmark и результаты](GLM53_FLASH_MTP_SHARED_SCRATCH_VALIDATION.json).

### 4.2. Один запуск quantized matvec для короткого пакета

MMVQ квантует входы 2–4 токенов вместе и исполняет их одним запуском CUDA-ядра.
Порядок арифметики одиночного decode сохранён. Это уменьшает число запусков
и временных allocations; однократное чтение всех весов из VRAM не гарантируется.

В fixture с весами на GPU, IQ3_XXS routed, K×M=4096×1536:
**75,0 → 58,7 мкс** для двух токенов, **128,2 → 77,7 мкс** для четырёх.
486 случаев и 15 686 784 выходных значения совпали бит-в-бит. Это локальное
ускорение матричной операции; устойчивый общий процент ускорения MTP не установлен.

`STRATA_GLM_MMVQ_TOKEN_BATCH=1` включён в локальных профилях, глобальный default
флага — 0. При MTP off может применяться к короткому prefill; одиночный decode
не меняется. Код: [mmvq_token_batch.inc](../../backends/glm5next/mmvq_token_batch.inc).
[Описание и времена](GLM53_FLASH_MMVQ_TOKEN_BATCH.md),
[матрицы](GLM53_FLASH_MMVQ_TOKEN_BATCH_MATRICES.json).

### 4.3. Восстановление MTP-кэша без ненужных выходов

При draft prefill и catch-up сохраняются MLA latent, indexer и pooled keys,
но не вычисляются ненужные query, attention output, FFN и LM head.
Обычное предсказание draft сохраняет полный NextN-блок.

90 сравнений состояния dense/sparse NextN прошли бит-в-бит; совпали следующие
draft logits/hidden states. Callback подтвердил отсутствие экспертного
`MUL_MAT_ID` в сокращённом пути. Полная модель прошла MTP1/2/3 и rollback.
Устойчивое ускорение всей генерации этим механизмом не подтверждено.

`STRATA_GLM_MTP_CACHE_ONLY=1` — default при включённом MTP; при MTP0 не используется.
Код: [mtp_cache_only.inc](../../backends/glm5next/mtp_cache_only.inc),
[mtp.hpp](../../backends/glm5next/mtp.hpp).
[Описание и проверки](GLM53_FLASH_MTP_CACHE_ONLY.md).

### 4.4. Chunk 4 МиБ: меньше накладных расходов доставки

В отдельном прогретом transport fixture IQ3_XXS, payload 227 МиБ, mmap,
один decode reader: **18,273 мс при chunk1 → 12,061 мс при chunk4**.
Chunk16 дал 11,596 мс, но требует по 64 МиБ pinned/device ring вместо
по 16 МиБ при chunk4. Это проверка доставки выбранных диапазонов без вычисления
модели и без cache, а не измерение ток/с.

В рабочем профиле выбран `--expert-chunk-mib 4`: четыре слота по 4 МиБ.
Один decode reader с обычной pinned memory также оказался лучшим из вариантов
своей full-model серии: 9,198 против 8,982 ток/с с двумя readers и 9,020
с write-combined memory. Это локальный подбор, не универсальное правило.
[Chunk benchmark](GLM53_FLASH_IQ3_XXS_CHUNK_BENCHMARK.json),
[подача весов](GLM53_FLASH_PIPELINE_MTP.md).

## 5. Удачные результаты по памяти и ожиданиям, без общего ускорения

| Механизм | Измеренный результат | Состояние и ограничение |
|---|---|---|
| Общий target/draft compute scratch | Освобождено **993,553 МиБ VRAM**; full GGUF logits точны | `STRATA_GLM_MTP_SHARED_SCRATCH=1`, default off. KV/recurrent state независимы. MTP1 с ним не выиграл по общей скорости. [Отчёт](GLM53_FLASH_COMPACTION_MTP.md) |
| Уплотнение уже выделенных GPU slabs между запросами | В пилоте возвращено **578 МиБ за 30,5 мс**; в A/B незаполненные slots уменьшились с 461 до 44/97 МиБ | `STRATA_GLM_CACHE_COMPACT=1`. Включено в отдельном запуске 13,37; обычный профиль оставляет off. Самостоятельный прирост ток/с не подтверждён. [Отчёт](GLM53_FLASH_COMPACTION.md) |
| Mmap только для экспертов, GPU-only веса загружаются через staging | Медиана working set **103,187 против 106,624 ГиБ**, разница 3,437 ГиБ в этих сериях; saved logits точны | `STRATA_GLM_EXPERT_LOAD_RAM=1` и `STRATA_GLM_EXPERT_RAM_LAYERS=0`. Это наблюдение по RAM в разных сериях; скорость не выросла, default off. [Отчёт](GLM53_FLASH_RAM_LOADING.md) |
| Раннее заполнение host-слота после завершения H2D | Суммарное CPU-ожидание слотов **0,785 → 0,206 с** за запрос; память staging не увеличена | `STRATA_GLM_EARLY_HOST_REFILL=1`, default off. Полная скорость 12,565 против контрольных 12,935/11,833 ток/с; выигрыш не установлен. В замороженной сборке 13,37 этого нового механизма ещё нет. [Отчёт](GLM53_FLASH_HOST_PIPELINE.md) |

Не следует складывать эти величины: они получены в разных режимах и не все
опции включены одновременно. Время ожиданий суммируется по CPU-потокам,
включает prefill и не равно сэкономленному времени ответа.

## 6. Полный набор настроек сохранённого запуска

| Настройка | Значение |
|---|---|
| `--mtp` | **0** |
| `--expert-pipeline` | 1 |
| `--expert-chunk-mib` | 4 |
| `--max-context` / `--batch-size` / `--threads` | 2048 / 16 / 4 |
| `--ram-target-percent` / `--vram-target-percent` | 95 / 95 |
| `STRATA_GLM_COPY_EVENTS` | 2 |
| `STRATA_GLM_MAIN_CACHE_DECAY` | 131072 |
| `STRATA_GLM_CACHE_SLAB_MIB` | 16 |
| `STRATA_GLM_CACHE_COMPACT` | **1** |
| `STRATA_GLM_MEMORY_POOL` / `STRATA_GLM_POOL_RECLAIM` | 0 / 0 |
| `STRATA_GLM_EXPERT_READ_MODE` / `STRATA_GLM_DECODE_READERS` | 0 (mmap) / 1 |
| `STRATA_GLM_WRITE_COMBINED` | 0 |
| `STRATA_GLM_MMVQ_TOKEN_BATCH` | 1 |
| `STRATA_GLM_MTP_SHARED_SCRATCH` | 0 |
| KV / TF32 / Flash Attention | F16 / off / off |
| Sampling профиля | temperature 0 |

`--mtp-cache-mib 512` присутствует в JSON, но при MTP0 не используется.
Learned warmup, дополнительный RAM warmup, private expert loading и CPU affinity
в этот запуск не добавлялись. Targets95 — реактивные бюджеты, а не гарантия
точного потолка суммарной памяти всех процессов ОС.

Для повторения benchmark из корня репозитория:

```powershell
.venv\Scripts\python.exe tools/benchmark_glm5next_mtp.py --profile strata-glm53-no-mtp-13_37.json --reference docs/GLM53/GLM53_FLASH_COMPACTION_B_REFERENCE.json --output build-local/glm53-no-mtp-13_37-repeat.json --depths 0 --warmups 4 --repeats 5 --paging-counters
```

Не добавлять второй reference: это изменит repeated-B на другую нагрузку.
Обычный `strata-glm5next.json` использует текущую сборку и не включает compaction;
он не равен отдельному сохранённому профилю 13,37.

## 7. Что не считать подтверждённым ускорением

- **Включение MTP вообще, увеличение глубины до 2/3 или cache ceiling до 1024 МиБ.**
  Некоторые ранние серии показывали небольшой выигрыш MTP1, поздние контроли
  его не подтвердили. Экономия операций внутри MTP не равна выигрышу над MTP off.
- **Принятие draft без проверки.** Такой приближённый режим не реализован и не измерен.
- **Learned warmup и дополнительный RAM scan.** Механизмы проверены, но ускорение
  первого ответа не подтверждено. [P3.3f/g](GLM53_FLASH_PIPELINE_MTP.md).
- **Native ReadFile, больше readers, write-combined.** Не дали устойчивого
  улучшения рабочего режима. [Замеры](GLM53_FLASH_COMPACTION.md).
- **Полная private RAM-загрузка.** Не дошла до READY из-за ограничений commit;
  частичная private загрузка не обошла повторный mmap-контроль.
  [Замеры](GLM53_FLASH_RAM_LOADING.md).
- **CPU affinity и streaming stores вместо memcpy.** Последняя проверка не
  дала устойчивого общего выигрыша. Отдельные удачные значения bandwidth или
  ток/с не достаточны для смены defaults. [P3.3j](GLM53_FLASH_HOST_PIPELINE.md).

Исправления TF32, tokenwise precision, GPU-only audit и точного rollback
нужны для корректности. Их нельзя отключать ради более красивой скорости
и продолжать называть результат эквивалентным проверенному GLM.

## 8. Проверки и границы результата

Для основных этапов сохранены не только timings, но и token IDs, F32 logits,
конфигурации и hashes. Slabs16 и события проверены full-model сравнением,
MTP — дополнительно forced rejection, sampling и восстановлением состояния.
Shared scratch освободил память без изменения сравниваемых logits.

В P5.3d прошли **28 CTest, 83 Python tests, 108 benchmark responses и 9 итоговых
native pipe cases**. Full GGUF проверка shared scratch включала 11 сценариев
и 9 912 320 сохранённых F32 значений; она выполнена до перегрузки `require`,
а финальная сборка с перегрузкой проверена CTest, benchmark и pipe.
[Manifest](GLM53_FLASH_MTP_SHARED_SCRATCH_VALIDATION.json).

Последующий P3.3j прошёл 32 candidate / 14 transport CTest, 83 Python tests,
полные saved logits и 9 pipe cases, но нового устойчивого ускорения не добавил.
[Manifest](GLM53_FLASH_HOST_PIPELINE_VALIDATION.json).
Длинный контекст, Linux/HIP и полные модели Qwen/DeepSeek этими GLM-замерами
не покрываются. Полный журнал: [статус внедрения](GLM53_FLASH_IMPLEMENTATION_STATUS.md).
