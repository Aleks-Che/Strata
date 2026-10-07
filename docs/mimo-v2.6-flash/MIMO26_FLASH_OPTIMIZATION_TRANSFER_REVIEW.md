# MiMo: что перенести из GLM, Step и DeepSeek

Разбор **2026-10-07**, Asia/Yekaterinburg. Сопоставлены три сводки и текущий код
backend. Новых GPU-замеров в этом разборе нет; приведённые числа MiMo относятся
к сохранённым опытам 2026-10-06 на Windows, RTX5090 32 ГБ и 128 ГБ RAM.
Модель: `MiMo-V2.6-Flash-RL-GSQ-RCO-3.5bit.gguf`.

Для скорости без MTP наиболее обоснован порядок: **компактное размещение кеша →
групповые копии → частотный допуск**. Для возвращения к MTP сначала проверить
арифметику короткого пакета по примеру GLM. Проценты других моделей нельзя
переносить на MiMo или складывать.

Продолжение от7 октября: первый кандидат реализован и измерен в
[MIMO-07](MIMO26_FLASH_SLAB_CACHE.md). Итоговый ABBA6,434→6,952 ток/с (+8,1%),
default16 включён после bit-exact corpus и проверок lifetimes. Затем
[MIMO-08](MIMO26_FLASH_TENSOR_BATCH.md) реализовал tensor delivery:7,299 ток/с,
+5,3% к последнему прогретому контролю своей серии.
[MIMO-09](MIMO26_FLASH_FREQUENCY_CACHE.md) добавил частоту с decay65536:
6,923→7,692 ток/с (+11,1%), к быстрейшему контролю+8,2%; default включён.
Decode H2D прогретых запросов−11,8%, fills−82,9%; первые запросы новых тем
копируют больше весов. Таблица ниже описывает состояние до этих трёх изменений.
Далее [MIMO-10](MIMO26_FLASH_FILL_AND_GUARDS.md) проверил grouped cache fills:
waits−19,17%, speedup нет (7,861→7,826 ток/с), default off; mixed numeric
diagnostic открыт. Перегрузка CPU guard для литералов добавлена в обычную сборку.
Её отдельный ABBA дал7,721→7,933 ток/с (+2,74%) при совпавших logits/IDs.
[MIMO-11](MIMO26_FLASH_HASH_CACHE.md) исправил распределение ключей истории:
CPU microbenchmark примерно44× быстрее, две full-model серии exact.
Средние7,532→7,976 и7,377→7,682 ток/с, но лучший контроль второй серии
быстрее нового среднего. Устойчивый end-to-end speedup пока не установлен;
cache defaults сохранены, исправление хеша включено.
[MIMO-12](MIMO26_FLASH_CACHE_INDEX.md) добавил unordered resident index,
LRU по стабильным указателям и pin count в записи. CPU replay63,330→32,505 мс;
прогретый decode8,390 против контроля8,401 ток/с. Политика кеша и logits exact,
ускорение генерации не подтверждено.
[MIMO-13](MIMO26_FLASH_HOST_PIPELINE.md) проверил early refill:7,992→7,912 ток/с,
off. Отдельное объединение512-байтных guard tails сократило H2D operations
на49,03%:10,336 ток/с, +24,02% к быстрейшему контролю8,334. Новый default on,
bytes/logits exact,224 CUDA cases и финальный corpus PASS.
[MIMO-14](MIMO26_FLASH_D2D_BATCH.md) измерил поздние и ранние пакеты D2D:
9,421→9,342 и9,619→9,418 ток/с. Второй ABBA держал cache11264 МиБ и одинаковые
cache/traffic counters; выигрыш не подтвердился, default off. Trace уменьшил
cached submissions1423,75→138, но сами intervals выросли13,885→21,184 мс.
Существенного H2D/compute overlap пока нет. Следующий приоритет — иной GPU
copy path/direct resident weights, групповые H2D, chunks4/16 и границы sync.
[MIMO-15](MIMO26_FLASH_SCATTER_COPY.md) добавил собственный scatter-copy kernel:
copy micro118,027→63,163 мкс, основной ABBA9,567→10,259 ток/с (+7,23%).
Отдельный русский ABBA с точным cache8 ГиБ и traffic дал9,131→8,035 (−12,01%).
Mode2 оставлен opt-in, default0 сохранён; все token IDs/logits exact.
Причину разброса ещё нужно локализовать перед сменой default.
[MIMO-16](MIMO26_FLASH_MTP_Q4_RETEST.md) повторил Q4 MTP с этими оптимизациями:
без draft10,520 ток/с; Q4 9,015; Q4 +scatter9,076. При принятии228/228 предложений
MTP всё ещё медленнее: cache12,00→9,13 ГиБ, H2D/выходной шаг+18,5–19,0%.
Пакетный target без draft снова расходится на18-м токене, как в MIMO-06.
Гипотеза переноса одиночной арифметики из GLM ниже остаётся задачей, не исправлением.

## Что уже используется

В MiMo есть выбранные router-эксперты, GPU-only compute audit, mmap, bounded
pinned/device ring, один reader, async H2D и route pins. Есть повторное
использование подходящих allocations среди 64 старых записей. Prefill использует
готовые hits без заполнения кеша и без обновления LRU; модельные матрицы остаются
на GPU. Это уже покрывает значительную часть ранних оптимизаций трёх backend.

Поиск allocation того же округлённого размера уже находится в
[ExpertCache::store](../../backends/mimo2/expert_cache.hpp).
Поэтому STEP-12 и подбор размера DeepSeek не являются полностью новыми
оптимизациями для MiMo. На момент исходного разбора частотной истории не было;
её последующий перенос описан в MIMO-09.

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

На момент разбора [contract.hpp](../../backends/mimo2/contract.hpp) принимал
сообщение `require` только как `const std::string &`. Проверка `non-finite logits` вызывается для
всех152576 значений в [main.cpp](../../backends/mimo2/main.cpp) и
[draft_probe.hpp](../../backends/mimo2/draft_probe.hpp). Для32 output rows это
4 882 432 вызова с неявным созданием строки; фактические allocations зависят
от компилятора и должны измеряться. Перегрузка `const char *`, как в GLM,
убирает создание временной строки на успешном пути. Это не перенос матричных
вычислений на CPU и не доказанный прирост ток/с MiMo.

MIMO-10 реализовал перегрузку. На этой машине MSVC Release guard152576 значений:
152576→0 allocations,3,22→0,057 мс в CPU microbenchmark. NaN/Inf и динамические
ошибки сохранены; отдельные end-to-end замеры и ограничения — в отчёте MIMO-10.

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

### Уточнение MIMO-17: выходная BF16-матрица и CUDA pool

Статический разбор от7 октября2026, исходники `d2b4156`, затем
[измеренный MIMO-17](MIMO26_FLASH_TARGET_HEAD.md) на9950X/RTX5090/128GB:
Q4 MTP9,880→10,929 ток/с, no-draft10,615; cache9,797→11,882 ГиБ,
H2D−14,48%. Это новый A/B с собственным контролем. Head fixture bit-exact,
историческое расхождение oracle batch2 не исправлено; default0, serving off.

В [инвентаризации](MIMO26_FLASH_INSPECTION.json) `output.weight` имеет тип BF16,
форму4096×152576 и размер1249902592 байта. В текущем CUDA dispatch для NVIDIA
Ampere+ BF16 MMVF выбирается при одном столбце. При двух и более столбцах
`StrictF32.cmake` исключает BF16 MMF, а заданный в `runtime.hpp`
`GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` приводит к cuBLAS F32. Его
`ggml_cuda_mul_mat_cublas_impl` выделяет `ggml_nelements(src0)` элементов float
и конвертирует всю матрицу весов перед умножением.

Для этой головы отдельный F32 buffer равен **2499805184 байта /2,328125 ГиБ**.
Это расчёт по коду и форме весов, а не измеренная прибавка к pool или кешу:
pool переиспользуется другими операциями, а live clamp зависит от общей VRAM.
Размер такого buffer одинаков при2 и8 выходных столбцах. Поэтому одно лишь
уменьшение all-logit прогрева с8 до2 не убирает этот расход.

Реализованный эксперимент — отдельный opt-in путь только для target `output.weight`:
проецировать каждый нужный столбец обычным GPU MMVF с F32 активациями и
накоплением, не конвертируя всю голову в F32. В
`SpeculativePatches.cmake` уже есть аналогичный приём для DFlash shared-head;
для target теперь добавлен `target_head.hpp`. Обрабатывается также all-logit warmup8,
иначе прежний прогрев заранее расширит pool. Предварительный прогрев рабочих
форм перед наполнением expert cache и предел95% сохранять.

MIMO-17 записал глобальную GPU memory по этапам: target context, draft weights/context,
обычный prefill8, verify2/all-logit warmup8; заполненный expert cache входит
в метрики запросов. Старый verify2 добавил примерно2,08 ГиБ после prefill,
новый этого роста не показал; это глобальные deltas, не отдельные pool counters.
Проверены head на одинаковых входах, oracle/rejection/SWA boundary и Q4 A/B.
Изменение одной головы не устранило расхождение target batch2:
routed matmul и attention ещё требуют отдельной локализации. Только после
этого расширять перенос GLM tokenwise/short-batch MMVQ на остальные матрицы.

Для ускорения без MTP следующий небольшой опыт уже поддержан benchmark runner:
`--axis chunk` с4/8/16 МиБ, packed guards1, D2D0, одинаковым физическим cache
и чередованием с контролем. Более крупный опыт — чтение resident experts прямо
из cache вместо cache→scratch D2D. Он требует адресации весов в CUDA kernels
и защиты cache slots до окончания consumer; простого удаления текущих fences
недостаточно. Ранее измеренные D2D intervals не равны обещанному ускорению.

### Проверенный перенос MIMO-18

[MIMO-18](MIMO26_FLASH_TOKENWISE.md) перенёс идею single-token арифметики
на MiMo-local CUDA dispatch. Одних dense/routed matmul оказалось недостаточно:
потребовались также MiMo FA query tile1 и cap пакета на KV256 boundary.
48 fixtures BF16/Q2_K/Q3_K/MXFP4 и untraced oracle depth1/7 дали bit-exact
результаты на проверенном corpus/границах. Реальный Q4 MTP сохраняет все
проверенные IDs и decode logits; первая prefill строка пока не exact.
Ни shared weights, ни изменение GGUF для этого не понадобились.
Mode7 opt-in, основной engine и serving не менялись. Новый A B C C B A:
Q4 9,878→10,447 ток/с (+5,76%); no-draft9,525, его быстрейший процесс9,921.
Почти одинаковый кеш Q4 (10,520/10,541ГиБ) и H2D (1,150/1,145ГиБ на шаг).
Полное время короткого запроса с prefill почти прежнее. Эти числа сравнивают
режимы в одной серии; результаты других моделей и прежних серий не являются контролем.

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
