# DeepSeek V4 Flash 0731

Документация отдельного backend `strata-deepseek4`: запуск моделей, передача
экспертных матриц, DSpark, GPU-кэш и результаты измерений на локальном ПК.

Начать с [удачных оптимизаций](DEEPSEEK4_SUCCESSFUL_OPTIMIZATIONS.md):
подтверждённые результаты, условия сравнения и рабочий профиль на 12 октября 2026.
Сборка и подготовка модели описаны в [основном руководстве](DEEPSEEK4.md).

| Документ | Содержание |
|---|---|
| [Перенос идей Maya](DEEPSEEK4_MAYA_TRANSFER_REVIEW.md) | Заимствование VRAM для prefill, GPU-план, pinned RAM, история частот, CPU lane и другой lookahead |
| [План внедрения CPU lane по архитектуре Maya](DEEPSEEK4_CPU_LANE_IMPLEMENTATION_PLAN.md) | Целевая связка RAM/VRAM, GPU-очередь, многотокенные ядра, калибровка и критерии принятия |
| [Измерения CPU lane перед переносом](DEEPSEEK4_CPU_LANE_TRACE.md) | Трасса заданий, сохранение весов/активаций и replay; контроль влияния диагностики |
| [Поколения и общий учёт RAM/VRAM](DEEPSEEK4_RAM_TIERS.md) | Готовность и время жизни матриц, отмена поздних заполнений, ограниченная демоция GPU → RAM |
| [GPU-очередь и CPU-service](DEEPSEEK4_CPU_GPU_QUEUE.md) | GPU-выбор и mapped ring, постоянный service, проверки протокола и оставшаяся синхронизация подкачки |
| [GPU-запросы подкачки матриц](DEEPSEEK4_GPU_FETCH_QUEUE.md) | Удаление host-чтения маршрутов на выбранных ветках, отдельный I/O-service, проверки и ограничения кэширования |
| [Первый прототип CPU lane](DEEPSEEK4_CPU_LANE.md) | Эксперимент с 4/8/12/16 потоками: ускорения нет, точный restore ответа не пройден, по умолчанию выключен |
| [CPU lane v2](DEEPSEEK4_CPU_LANE_V2.md) | Целые эксперты и CUDA Graphs; fixed 14,09 против GPU 17,10 ток/с; fixed restore прошёл, auto ещё нет |
| [План и новые испытания CPU lane](DEEPSEEK4_CPU_LANE_EXPERIMENTS.md) | Выбор по токенам, общие RAM-матрицы, контроль корректности и замеры 8/12/16 потоков |
| [Пул RAM и фоновое сохранение](DEEPSEEK4_RAM_BACKGROUND.md) | Переиспользование allocations в общем бюджете и очередь на два задания: раздельные замеры |
| [RAM после решения GPU-кэша](DEEPSEEK4_RAM_ADMISSION.md) | Отложенное заполнение RAM: меньше копирования, но GPU-only остаётся быстрее; CPU12 и DSpark |
| [Прямой H2D из RAM](DEEPSEEK4_RAM_DIRECT_H2D.md) | Защищённые pinned-адреса, удаление RAM-дубликатов после GPU admission, замеры и ограничения |
| [CPU lane v2: оставшиеся отличия от Maya](DEEPSEEK4_CPU_LANE_V2_MAYA_AUDIT.md) | Раздельный CPU-кэш, общий лимит speculative window, host-планирование и ошибка калибровки |
| [CPU lane: отличия от Maya](DEEPSEEK4_CPU_LANE_MAYA_AUDIT.md) | Почему матричный прототип теряет преимущества целого CPU-эксперта; что менять и какие измерения нужны |
| [Компактные матрицы, RAM-кэш и прогноз маршрутов](DEEPSEEK4_COMPACT_RAM_PREDICTOR.md) | Плотные физические слоты, общий ограниченный RAM-кэш, оценка трёх прогнозов без подкачки |
| [Ранний расчёт готовых экспертов](DEEPSEEK4_EARLY_CACHE_HITS.md) | Основной и отдельный CUDA stream: −0,18% / −1,04% по прогретым повторам, эксперимент выключен |
| [Ранняя подгрузка hash-слоёв](DEEPSEEK4_HASH_PREFETCH.md) | Два точных режима первых трёх слоёв; +0,22% / +0,13% по прогретым повторам, оставлены выключенными |
| [Основное руководство](DEEPSEEK4.md) | Архитектура, GPU-only, GGUF Q3/Q8, сборка, запуск, API и ограничения |
| [Исходные замеры](DEEPSEEK4_BENCHMARKS.md) | Q8 и Q3 до оптимизации передачи, нагрузка CPU/GPU, RAM и VRAM |
| [Кэш матриц и pinned RAM](DEEPSEEK4_EXPERT_STREAMING.md) | Хранение gate/up/down по отдельности, staging и первое сравнение скорости |
| [DSpark](DEEPSEEK4_DSPARK.md) | Частичная загрузка draft, проверка токенов, сессии и выбранная длина черновика |
| [Подбор параметров DSpark](DEEPSEEK4_DSPARK_TUNING.md) | Стоимость draft/target, confidence, write-combined staging и результаты проб |
| [Асинхронный конвейер](DEEPSEEK4_EXPERT_PIPELINE.md) | Фоновое чтение, кольцо H2D, CUDA events, перекрытие передачи и вычислений |
| [Очередь чтения и повторное использование VRAM](DEEPSEEK4_EXPERT_PIPELINE_V2.md) | Windows overlapped reads, режим auto, динамический кэш и физические чтения SSD |
| [Частотный допуск](DEEPSEEK4_FREQUENCY_CACHE.md) | Сохранение часто используемых матриц и уменьшение лишнего D2D |
| [Вытеснение с учётом размера](DEEPSEEK4_CACHE_SIZE.md) | Уменьшение фрагментации фиксированной арены без увеличения VRAM |
| [Общий временный GPU-буфер](DEEPSEEK4_SHARED_SCRATCH.md) | Совместная временная память target/DSpark, экономия VRAM и проверки состояния |
| [Сводка удачных оптимизаций](DEEPSEEK4_SUCCESSFUL_OPTIMIZATIONS.md) | Общий результат и применяемые настройки со ссылками на доказательства |
| [Перенос оптимизаций GLM и Step](DEEPSEEK4_OPTIMIZATION_TRANSFER_REVIEW.md) | Что уже есть в DeepSeek, кандидаты для переноса и порядок их проверки |
| [Проверка трёх вариантов переноса](DEEPSEEK4_TRANSFER_TRIALS.md) | AVX2, раннее заполнение pinned-слотов и использование готового GPU-кэша при prefill |
| [Размер staging и затухание частот](DEEPSEEK4_STAGE_AND_DECAY.md) | Замеры 10 октября: stage 8/16, раздельные периоды target/DSpark и стоимость отдельных allocations |
| [Блоки динамического GPU-кэша](DEEPSEEK4_DYNAMIC_SLABS.md) | Общие slabs target/DSpark, физический резерв, давление VRAM и сравнение с отдельными allocations |
| [Короткий пакет MMVQ](DEEPSEEK4_MMVQ_TOKEN_BATCH.md) | Арифметика одиночного токена, проверка GPU-матриц и сравнение со штатным пакетным ядром DeepSeek |
| [Первое расхождение batch/serial](DEEPSEEK4_ATTENTION_PROBE.md) | Проверка активаций без изменения logits, одинаковые видимые входы Flash Attention и зависимость результата от batch и размера KV |
| [Изолированный attention и следующий участок MoE](DEEPSEEK4_ATTENTION_FIXTURE.md) | FP64-эталон, 182 GPU-сценария, уплотнение видимых ключей и локализация расхождения routed FFN |
| [BF16 router и короткий batch MMVF](DEEPSEEK4_ROUTER_MMVF.md) | Дополнительное округление активаций router, 216 GPU-сценариев и совпадение первых трёх слоёв |
| [HCA и эквивалентность полных logits](DEEPSEEK4_HCA_COMPACT.md) | Видимые K/V и маски, безопасная граница HCA, 237 GPU-сценариев и проверка контекста 4096 |
| [Одинаковый stream-K в пакетном attention](DEEPSEEK4_STABLE_STREAM_K.md) | Сетка блоков как у одиночного query, точные logits, A/B/B/A и проверки состояния |
| [Полный и поэтапный prefill](DEEPSEEK4_PREFILL_PARITY.md) | Фиксированные token IDs, контроль snapshot и наблюдения, исторические K/V и матрицы первого слоя |
| [HC и Q/K при разном batch](DEEPSEEK4_PROJECTION_PARITY.md) | Фактический CUDA dispatch, TF32, разные масштабы Q8, FP64-эталон и проверка всей модели |
| [Фиксированное разбиение MMQ для Q/K](DEEPSEEK4_FIXED_PROJECTION_MMQ.md) | Точное совпадение исторического K/V первого слоя, шесть вариантов плиток, A/B/B/A и следующий участок расхождения |
| [Расширение Q и длинный attention](DEEPSEEK4_QUERY_EXPANSION.md) | Чтение одного токена, фиксированный Q_b, граница wo_a, ограничение временной VRAM, замеры и исправление восстановления динамического кэша |
| [Выходные проекции attention](DEEPSEEK4_OUTPUT_PROJECTIONS.md) | Фиксированные wo_a/wo_b, проверка восьми групп, сравнение разбиений MMQ, граница HC и замеры модели |
| [F32-проекции HC](DEEPSEEK4_HC_PROJECTIONS.md) | Пакетный MMVF с одинаковой редукцией, группы по 1/4/8, точный вход FFN и оставшиеся расхождения router/MoE/shared expert |
| [Router на длинном prefill](DEEPSEEK4_ROUTER_LONG_PREFILL.md) | Точный BF16-router на 1–4096 токенах, отдельные источники расхождения shared/routed expert и стоимость одинаковой редукции |
| [Матрицы shared expert](DEEPSEEK4_SHARED_PROJECTIONS.md) | Фиксированный Q8_0 MMQ для gate/up/down, обход fusion, проверка SwiGLU и оставшееся расхождение routed MoE |
| [Routed MoE и короткий MMQ](DEEPSEEK4_ROUTED_MMQ.md) | Выборочная трассировка MUL_MAT_ID, контроль влияния fusion, одинаковые ID, разные форматы активаций IQ3/MXFP4 |
| [Stride ID и совпадение DSpark](DEEPSEEK4_ROUTED_ID_STRIDE.md) | Исправление выбора MMQ для top-k view, 319 точных строк verification, 8/8 greedy-сценариев и A/B/B/A без ускорения |
| [Проекции компрессора](DEEPSEEK4_COMPRESSOR_PROJECTIONS.md) | Фиксированный MMQ для CSA/HCA/indexer, устранение различий сжатого K/V и точные конечные logits на фиксированной истории |
| [Длинный prefill и полный эксперимент](DEEPSEEK4_LONG_PREFILL_AUDIT.md) | История 3072 токена, граница HCA слоя 7 и A/B/B/A всего численного набора против рабочего профиля |
| [Проекции индексатора](DEEPSEEK4_INDEXER_PROJECTIONS.md) | Трасса 128 токенов, смена одного top-k ключа и устранение расхождения истории 3072 |
| [Стоимость операций](DEEPSEEK4_KERNEL_COST.md) | Фиксированные токены, CUDA events, потери в attention и два отклонённых варианта ускорения |
| [Этапы attention и пустые блоки](DEEPSEEK4_ATTENTION_STAGES.md) | Раздельные события, точный пропуск пустых HCA-разделов и сокращение prefill на 16% в экспериментальном режиме |
| [Пакетное копирование из GPU-кэша](DEEPSEEK4_CACHE_COPY_BATCH.md) | Измерение передач экспертов, объединение D2D-запусков, защита cache allocations и A/B/B/A |
| [Прямое чтение из GPU-кэша](DEEPSEEK4_DIRECT_CACHE.md) | MMVQ читает cached-матрицы без промежуточного D2D; +1,84% в B/A/A/B, CUDA Graphs и защита от вытеснения |
| [Продолжение прямого GPU-кэша](DEEPSEEK4_DIRECT_CACHE_FOLLOWUP.md) | Новые admissions и битовая маска удержаний: меньше копий/выделений, заметного ускорения не получено |
| [Передача матриц без отдельных остановок CPU](DEEPSEEK4_ORDERED_EXPERT_COPIES.md) | Порядок в CUDA stream вместо 129 ожиданий перед весами; +8,62% позднего decode в B/A/A/B |

Отчёты сохраняют даты, параметры и ограничения своих этапов. Упоминание ещё
не реализованного механизма в раннем отчёте относится к тому этапу; текущие
настройки собраны в сводке. Проценты из разных сравнений нельзя складывать
или перемножать. Команды и пути в блоках кода даны относительно корня репозитория.

Общие [настройки VRAM](../VRAM_SETTINGS.md),
[интеграция сессий и статистики](../SESSION_INTEGRATION_RU.md) и
[архивы сессий](../SESSION_ARCHIVE.md) относятся к нескольким моделям
и остаются в общем каталоге `docs`.
