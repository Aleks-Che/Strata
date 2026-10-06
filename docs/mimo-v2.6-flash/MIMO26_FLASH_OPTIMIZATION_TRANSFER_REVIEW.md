# MiMo: что перенести из GLM, Step и DeepSeek

Разбор **2026-10-07**, Asia/Yekaterinburg. Сопоставлены три сводки и текущий код
backend. Новых GPU-замеров в этом разборе нет; приведённые числа MiMo относятся
к сохранённым опытам 2026-10-06 на Windows, RTX5090 32 ГБ и 128 ГБ RAM.
Модель: `MiMo-V2.6-Flash-RL-GSQ-RCO-3.5bit.gguf`.

Для скорости без MTP наиболее обоснован порядок: **компактное размещение кеша →
групповые копии → частотный допуск**. Для возвращения к MTP сначала проверить
арифметику короткого пакета по примеру GLM. Проценты других моделей нельзя
переносить на MiMo или складывать.

## Что уже используется

В MiMo есть выбранные router-эксперты, GPU-only compute audit, mmap, bounded
pinned/device ring, один reader, async H2D и route pins. Есть повторное
использование подходящих allocations среди 64 старых записей. Prefill использует
готовые hits без заполнения кеша и без обновления LRU; модельные матрицы остаются
на GPU. Это уже покрывает значительную часть ранних оптимизаций трёх backend.

Поиск allocation того же округлённого размера уже находится в
[ExpertCache::store](../../backends/mimo2/expert_cache.hpp).
Поэтому STEP-12 и подбор размера DeepSeek не являются полностью новыми
оптимизациями для MiMo. Частотной истории в MiMo пока нет.

## Кандидаты для скорости без MTP

| Приоритет | Откуда | Что есть у MiMo | Что переносить |
|---|---|---|---|
| 1 | GLM slabs16 | Отдельный `cudaMalloc`, округление каждой матрицы до2 МиБ | Несколько матриц в одном GPU-блоке, учёт всей зарезервированной памяти |
| 2 | Step tensor batching; GLM copy events | Host fence после каждой доставки и cache fill | Сначала одна группа копий на scheduler input, затем отдельный опыт с CUDA events |
| 3 | DeepSeek frequency; GLM decay | Обычный LRU с admission каждого decode miss | Сравнение частоты новой матрицы и реального кандидата на вытеснение |
| Небольшое сопутствующее изменение | GLM `require(const char *)` | Временный `std::string` для каждой проверки logits | Перегрузка для литералов при сохранении NaN/Inf checks |

### 1. Размещение кеша: видимая потеря около3 ГиБ

Из [итогового MiMo benchmark](MIMO26_FLASH_SPECULATIVE_BENCHMARK.json), средние
счётчики девяти измеренных запросов на вариант:

| Вариант | Выделения кеша, ГиБ | Полезные веса, ГиБ | Разница, ГиБ |
|---|---:|---:|---:|
| Без draft | 12,842 | 9,798 | **3,044** |
| MTP Q4 | 10,149 | 7,787 | 2,362 |
| DFlash Q8 | 8,116 | 6,247 | 1,869 |

В no-draft полезно заняты76,30% выделений кеша. Разница23,70% — округление
allocations, а не свободная VRAM и не обещание вернуть все3,044 ГиБ.
Размер одной routed matrix Q2_K/Q3_K/MXFP4 —2,625/3,4375/4,25 МиБ;
нынешний allocator выделяет4/4/6 МиБ. Значения получены из
[инспекции GGUF](MIMO26_FLASH_INSPECTION.json) и
[sync_runtime.inc](../../backends/mimo2/sync_runtime.inc).

В GLM slabs16 дали9,865 →11,764 ток/с в своей серии, вместе с ростом полезной
ёмкости кеша. Источник: [сводка GLM, §2.2](../GLM53/GLM53_FLASH_SUCCESSFUL_OPTIMIZATIONS.md).
Начальная реализация для адаптации:
[expert_slab.hpp](../../backends/glm5next/expert_slab.hpp).

Нельзя просто заменить2 МиБ на256 байт и оставить отдельные `cudaMalloc`:
это вернёт уже обнаруженный недоучёт WDDM. Нужны suballocations внутри блоков,
раздельные счётчики payload/slots/reserved, учёт пустых slots и округления
самих блоков, ограничение роста по общей VRAM95%. Начать без перемещения живых
матриц между блоками; compaction — отдельный опыт. Сохранить pins и fallback.

### 2. Групповые копии: убрать отдельные ожидания

В [mimo_pipeline_copy](../../backends/mimo2/pipeline_runtime.inc) hit и miss
заканчиваются `cudaStreamSynchronize` внутри цикла по выбранным экспертам.
Заполнение кеша вызывает [mimo_d2d](../../backends/mimo2/sync_runtime.inc)
с дополнительным ожиданием. События ring уже есть, но эти ожидания остаются.

Ближайший образец — `step_pipeline_copy_tensor` в
[Step pipeline_runtime.inc](../../backends/step35/pipeline_runtime.inc):
одна группа на входной тензор scheduler, затем общий fence. В своей серии
Step это дало +19,04% без MTP. GLM events дали +12,64% в другой серии с MTP1;
это отдельные подходы и условия, не два складываемых выигрыша.
[Step](../Step-3.7-Flash/STEP37_FLASH_SUCCESSFUL_OPTIMIZATIONS.md),
[GLM](../GLM53/GLM53_FLASH_SUCCESSFUL_OPTIMIZATIONS.md).

При переносе MiMo требует сохранить guard tails до512 байт, route pins,
защиту новых fills до завершения копирования и drain на ошибке/отмене.
Нынешний `ExpertCache::store` публикует запись только после fenced fill;
асинхронная публикация потребует явного состояния готовности или отложенного
commit. Scratch может содержать живые активации: удалять backend fence без
зависимости от предыдущего consumer нельзя. Начальный tensor batching проще
проверить, чем одновременную замену всех ожиданий событиями.

### 3. Частотный допуск: меньше вытеснений и cache-fill D2D

В DeepSeek редкая новая матрица проходит через staging, сохраняя более частую
запись в кеше. Она всё равно вычисляется: routing, top-k и веса не меняются.
В своей серии это дало +12,7% на тексте и +10,6% на коде.
[Сводка DeepSeek, §6](../deepseek-v4-flash-0731/DEEPSEEK4_SUCCESSFUL_OPTIMIZATIONS.md).

Готовая ограниченная история есть в
[common/expert_frequency.hpp](../../backends/common/expert_frequency.hpp),
пример решения о допуске — в
[deepseek4/cuda_expert_transfer.inc](../../backends/deepseek4/cuda_expert_transfer.inc).
Для MiMo нужны полные ключи generation/tensor/expert, учёт даже отклонённых
decode misses и сравнение с фактической жертвой, включая подбор размера.
Prefill оставить без обучения истории. Период131072 из GLM — кандидат,
не готовый default: нужны смена темы и A/B/A, а не только повтор одного prompt.

### Небольшая CPU-оптимизация

[contract.hpp](../../backends/mimo2/contract.hpp) принимает сообщение `require`
только как `const std::string &`. Проверка `non-finite logits` вызывается для
всех152576 значений в [main.cpp](../../backends/mimo2/main.cpp) и
[draft_probe.hpp](../../backends/mimo2/draft_probe.hpp). Для32 output rows это
4 882 432 вызова с неявным созданием строки; фактические allocations зависят
от компилятора и должны измеряться. Перегрузка `const char *`, как в GLM,
убирает создание временной строки на успешном пути. Это не перенос матричных
вычислений на CPU и не доказанный прирост ток/с MiMo.

## Отдельная зацепка для корректности MTP

В GLM уже было расхождение batched logits до rollback. Его исправили
одиночной арифметикой коротких `MUL_MAT` и quantized `MUL_MAT_ID`.
См. [CandidatePatches.cmake](../../backends/glm5next/CandidatePatches.cmake)
и журнал P3.3b в [статусе GLM](../GLM53/GLM53_FLASH_IMPLEMENTATION_STATUS.md).
Затем [короткий MMVQ](../GLM53/GLM53_FLASH_MMVQ_TOKEN_BATCH.md) объединил
запуски2–4 токенов, сохранив порядок арифметики одиночного decode.

MiMo использует тот же pin dependency, но его `RoutedStrides.cmake` исправляет
strides, а не выбор одиночной арифметики. TF32/BF16 policy также не доказывает
полную эквивалентность dispatch. Поэтому это конкретная гипотеза для
[расхождения MIMO-06](MIMO26_FLASH_SPECULATIVE_DIAGNOSTIC.json), а не установленная
причина: full/SWA attention и другие операции ещё не исключены.

Сначала отдельный opt-in dispatch и replay известных tokens без draft при
batch2. Затем матричные fixtures с реальными4096×2048/2048×4096,
Q2_K/Q3_K/**MXFP4** и BF16 dense, padded strides и разными expert IDs.
GLM fixture не покрывает MXFP4 и batch8. Для DFlash нужно отдельно расширить
и проверить диапазон до8; простое включение `STRATA_GLM_*` в MiMo не поможет.
Патчи должны оставаться MiMo-local и сохранять его исправление routed strides.
После проверки арифметики — повтор full/SWA boundary, rejection и замера
MTP Q4 против нового no-draft baseline. MTP serving до этого остаётся off.

## Что пока не брать первым

- **Shared embedding/head:** в Step веса проверены на равенство; локальные
  MiMo MOPD/RL отличаются. Проверенный MiMo shared вариант оказался медленнее.
- **Shared scratch:** полезный опыт экономии памяти после correctness, но
  подтверждённые1,14 ГиБ DeepSeek не переносятся на MiMo. Общий graph buffer
  не обязательно объединит отдельные CUDA conversion pools.
- **Early refill / AVX2 / больше readers / WC:** результаты других моделей
  неоднозначны; у MiMo пока более явные потери на размещении и fences.
- **CUDA Graphs, прямой matmul по cache pointers, предсказание будущих experts:**
  больший объём изменений и проверки lifetimes; в этих сводках нет готового
  доказанного переноса для MiMo.
- **Сессии:** полезны для TTFT повторных диалогов, но это отдельный P4,
  не автоматическое ускорение обычного decode.

## Условия следующего сравнения

Для каждого изменения отдельный выключатель и одна сборка, сначала без MTP.
Проверять byte/logits parity и GPU-only audit, затем чередовать контроль/опыт
на счёте, коде и русском тексте, с прогревом и минимум тремя измеренными
повторами; отдельно новый prompt и смена темы. Фиксировать context/KV/batch,
реальный cache payload/reserved, H2D/D2D, fills/evictions, allocations/reuse,
число host fences, полное время запроса и global RAM/VRAM. Добавить контроль
физического SSD I/O: mmap bytes не означают чтение диска.

Замеры скорости MIMO-05 и MIMO-06 имеют разные условия. Сравнивать кандидат
с собственным контрольным запуском; текущие6,13 и исторические7,05 ток/с
не использовать как взаимозаменяемые baseline. Defaults менять после такого A/B.
