# DeepSeek V4 Flash 0731

Документация отдельного backend `strata-deepseek4`: запуск моделей, передача
экспертных матриц, DSpark, GPU-кэш и результаты измерений на локальном ПК.

Начать с [удачных оптимизаций](DEEPSEEK4_SUCCESSFUL_OPTIMIZATIONS.md):
подтверждённые результаты, условия сравнения и рабочий профиль на 11 октября 2026.
Сборка и подготовка модели описаны в [основном руководстве](DEEPSEEK4.md).

| Документ | Содержание |
|---|---|
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

Отчёты сохраняют даты, параметры и ограничения своих этапов. Упоминание ещё
не реализованного механизма в раннем отчёте относится к тому этапу; текущие
настройки собраны в сводке. Проценты из разных сравнений нельзя складывать
или перемножать. Команды и пути в блоках кода даны относительно корня репозитория.

Общие [настройки VRAM](../VRAM_SETTINGS.md),
[интеграция сессий и статистики](../SESSION_INTEGRATION_RU.md) и
[архивы сессий](../SESSION_ARCHIVE.md) относятся к нескольким моделям
и остаются в общем каталоге `docs`.
