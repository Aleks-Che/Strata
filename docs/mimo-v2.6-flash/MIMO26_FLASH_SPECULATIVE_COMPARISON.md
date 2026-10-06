# MiMo-V2.6-Flash-RL: сравнение MTP и DFlash

Измерено **2026-10-06** на Windows, RTX5090 32 ГБ, 128 ГБ RAM,
CUDA13.0.48/sm120, driver581.80. Target:
`H:\models\mimo-v2.6-flash\MiMo-V2.6-Flash-RL-GSQ-RCO-3.5bit.gguf`.

**Для текущего engine оставляем режим без draft: он быстрее.**
Для следующих опытов наиболее экономный кандидат —
`mtp-MiMo-V2.6-Flash-MOPD-Q4_0.gguf`, первая голова, depth1, `p_min=0.7`.
Q4 и Q8 близки по скорости; малую разницу между ними нельзя считать устойчивым
преимуществом по этим трём повторам. У Q4 меньше весов и больше места для expert cache.
Serving MTP/DFlash не включён, обычный executable не изменён.

## Одинаковый основной тест для пяти файлов

Три prompt: счёт, Python-функция, объяснение времён года по-русски.
Greedy, thinking off, 32 выходных токена; первый запрос каждого prompt исключён,
затем три прогретых повтора. Context512, batch8, F32 KV, FA on, полный физический
KV для rollback, reader1, chunk8 МиБ, cache до14 ГиБ с live clamp.
Матрицы вычисляются на CUDA. CPU обслуживает I/O, scheduling и выбор токена.
Лимиты общей RAM/VRAM95%; реальный cache учитывает веса и рабочие буферы draft.

| Режим | Новых draft-токенов максимум | Ток/с | Принято предложений | Средний expert cache, ГиБ allocations | IDs совпали с baseline |
|---|---:|---:|---:|---:|---|
| Без draft | 0 | **6,13** | — | 12,84 | Да |
| MTP Q4_0 | 1 | **5,55** | 100% | 10,15 | Да |
| MTP Q8_0 | 1 | 5,48 | 97,44% | 9,53 | Да |
| MTP BF16 | 1 | 4,88 | 97,44% | 7,89 | Да |
| DFlash Q8_0 | 7 | 5,26 | 98,46% | 8,12 | Да |
| DFlash BF16 | 7 | 4,66 | 98,46% | 3,55 | Да |

Во всех draft-строках `p_min=0.7`, собственные embedding/head из sidecar.
Для MTP загружена только первая голова; головы2/3 и их независимые смещения
не реализованы этим probe. У DFlash block8 означает anchor плюс7 новых токенов.
Acceptance считается после confidence cutoff, а не по всем предсказаниям модели.
100% принятия у Q4 не означает, что draft покрывает всю генерацию: он предложил
114 из279 токенов после первых токенов девяти измеренных запросов.

Скорость — сумма `(выходных токенов−1)` / сумма времени генерации, не среднее
арифметическое скоростей. Включены draft, verify, catch-up, rollback, выбор токенов
и запись диагностических logits; загрузка и prefill исключены. Первоначальный
контроль без draft дал6,19 ток/с, повторный6,13. Ранее опубликованные7,05 ток/с
pipeline измерялись другим тестом и не являются baseline этой таблицы.

По отдельным prompt (ток/с, те же три прогретых повтора):

| Режим | Счёт | Код | Русский текст |
|---|---:|---:|---:|
| Без draft | 6,93 | 5,50 | 6,13 |
| MTP Q4_0 | 5,50 | 5,23 | 5,95 |
| DFlash Q8_0 | 6,46 | 4,97 | 4,67 |

Точные значения, команды, IDs, memory samples summary и hashes находятся в
[benchmark](MIMO26_FLASH_SPECULATIVE_BENCHMARK.json). Это небольшой локальный
screening, не широкая оценка знаний/качества и не выбор настроек для всех контекстов.

## Почему принятие высокое, а ускорения нет

При MTP Q4 draft занял около1,4% времени генерации, при DFlash Q8 —2,4%.
Основные затраты остались в target. Draft и batched verification уменьшают
кеш экспертов: с12,84 до10,15 ГиБ у MTP Q4 и до8,12 ГиБ у DFlash Q8.
На один выходной токен после первого пришлось примерно1,47 ГиБ H2D без draft,
1,77 ГиБ с MTP Q4 и2,00 ГиБ с DFlash Q8. Это счётчик доставленных GPU bytes,
не physical SSD I/O. Уменьшение числа проходов не компенсировало дополнительные
копирования и работу verify. Полная стоимость GPU/driver paging отдельно не измерялась.

В отдельном переборе DFlash Q8 на счёте depth1/3 давали около4,9/4,6 ток/с,
depth7 —6,5. Без confidence cutoff на русском принимались15 из94 предложений;
с cutoff0.7 —12 из13. Поэтому длинный блок без фильтра оказался особенно невыгодным.
Native MTP без фильтра также был медленнее и расходился с baseline на русском.
[Все дополнительные опыты](MIMO26_FLASH_SPECULATIVE_EXPERIMENTS.json).

## Что означает проверка качества

Все пять конфигураций с cutoff0.7 совпали по выходным IDs на трёх prompt,
по четыре запуска каждого. Дополнительно MTP Q4 и DFlash Q8 совпали с baseline
на prompt длиннее SWA128 и16 выходных токенах. Эти одиночные холодные boundary
прогоны проверяют состояние/IDs; их timing не используется для рейтинга скорости.

**Общая проверка эквивалентности последовательному target пока не пройдена.**
При проверке заранее известных baseline-токенов пакетами2 и8, без draft-весов
и без feature extraction, русский ответ меняется на18-м токене. До этой позиции
все предложения приняты: расхождение возникает до первого rollback.
Prefill logits при этом bit-exact. На изменившейся позиции исходный отрыв top-1
составляет0,06897, а максимальная разница logits —0,33821 для batch2.
Значит, замена кванта draft не устраняет источник расхождения. Конкретный
kernel/attention механизм ещё не локализован. Это не доказательство, что другой
текст хуже по смыслу; это отказ от обещания lossless parity.
[Диагностика](MIMO26_FLASH_SPECULATIVE_DIAGNOSTIC.json).

До исправления/приёмки batched target и более широких проверок не переносить
эти offline-настройки в serving. Stochastic correction, streaming/cancel,
sessions/context shift и native heads2/3 требуют отдельной реализации.

## Совместимость и сделанные изменения

Все пять GGUF прошли проверки metadata, полного набора тензоров, shapes, quants,
ranges и одинакового tokenizer. Полные SHA-256 sidecars сохранены в
[инспекции](MIMO26_FLASH_DRAFT_INSPECTION.json). В main GGUF по-прежнему нет NextN.

У официальных RL и MOPD одинаковые опубликованные DFlash LFS hashes, но локальные
MOPD embedding/head отличаются от нашего target RL даже при BF16. Проверка
[происхождения](MIMO26_FLASH_DRAFT_RESEARCH.json) содержит снимки
[RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL/tree/main/dflash) и
[MOPD](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD/tree/main/dflash).
Совместимость не объявлялась по одному имени модели.

В MiMo-local probe добавлены sidecar loader, target features1/12/24/36/48,
pre-norm последнего слоя, DFlash value scale0.612, сохранение learned MASK151675,
пакетная проверка и обрезка KV до принятого префикса. Последний pre-norm capture
также явно используется в [SGLang MiMo](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/mimo_v2.py).
DeepSeek DSpark adapter не переносился целиком: его anchor/Markov семантика другая.

Проверен вариант DFlash с target RL embedding/head и отдельным MASK на GPU.
Q8/BF16 дали4,05/3,26 ток/с; улучшения на этом наборе не получено. Результаты
сохранены отдельно, в основной таблице используются исходные sidecar matrices.
Один ранний прогон остановлен memory guard при позднем выделении CUDA workspace.
Теперь полный verify batch прогревается до заполнения expert cache. Обычный
no-draft baseline не держит неиспользуемый verify workspace. Повторные завершённые
прогоны уложились в95% global limits; остановленный прогон исключён из скоростей.

28 Python tests и10 C++ проверок принятого префикса PASS. GPU audit не обнаружил
модельных вычислений на CPU; очереди transport пусты после запросов.
Сборка, hashes и сохранение production binary: [build validation](MIMO26_FLASH_SPECULATIVE_BUILD_VALIDATION.json).

Команды сборки и повторения — в [README backend](../../backends/mimo2/README.md#отдельное-сравнение-mtp--dflash).
Снимок измеренного executable/manifest: `build-local/mimo2-spec-measured`;
его можно передать runner через `--build`. Обычная CMake-конфигурация возвращена
к `STRATA_MIMO_SPEC_PROBE=OFF`.
