# MiMo: частотная политика кеша

Измерено **2026-10-07**, Asia/Yekaterinburg. Windows, RTX5090 32 ГБ,
128 ГБ RAM; `MiMo-V2.6-Flash-RL-GSQ-RCO-3.5bit.gguf`, 134 982 426 368 байт.
CUDA13.0.48, sm120, Unsloth `86ebfef2`.

На прогретых запросах ABBA дал **6,923 → 7,692 ток/с (+11,1%)** без MTP.
Относительно самого быстрого контрольного процесса прибавка **8,2%**.
Decode H2D уменьшился на11,8%, заполнение GPU-кеша — на82,9%.
Все сохранённые logits и выходные токены совпали побитово.

[Замеры](MIMO26_FLASH_FREQUENCY_BENCHMARK.json),
[сборки и проверки](MIMO26_FLASH_FREQUENCY_VALIDATION.json).

## Изменение

Вместо допуска каждого decode miss кеш сравнивает частоту новой матрицы
с частотой кандидата на вытеснение. При наличии свободного места допуск
сохраняется. При нехватке места выбирается наименее частая запись среди
64 самых старых незакреплённых entries, прежде всего с подходящим размером
allocation. Равные частоты разрешают замену; менее частый newcomer проходит
через staging, сохраняя прежнюю запись. Все выбранные router эксперты
по-прежнему вычисляются, их веса и арифметика не меняются.

Использован существующий `StrataExpertFrequencyHistory` из
[общего header](../../backends/common/expert_frequency.hpp) без его изменения.
Ключ содержит generation/tensor/expert. Счётчик до255, история ограничена
65536 ключами; учитываются hits и misses, включая отказанные admissions.
Каждые65536 наблюдений частоты лениво делятся пополам. При обычном decode
этой модели это примерно58 токенов:47 MoE ×3 матрицы ×8 экспертов на токен.
Это период наблюдений матриц, не время и не жёсткое окно последних58 токенов.

Guard tails и служебные проверки не обучают историю. При default prefill
admission off prefill также её не обучает. Reset статистики запроса сохраняет
историю; пересоздание кеша её очищает. Pins, завершение копии до публикации,
обработка ошибок и global95% сохранены. Внешнее давление памяти может
вытеснять горячие записи. Матрицы модели выполняются на GPU; CPU обновляет
небольшую историю и обслуживает прежние I/O/scheduling.

## Измерения

Один executable, порядок decay **0,65536,65536,0**. Slab16 и tensor delivery
включены в обоих режимах. Три темы: счёт, Python и русский текст. На каждую
тему первый запрос помечен прогревом, затем три измеренных запроса по32 токена.
Свежий KV на каждом запросе, кеш и история сохраняются между темами процесса.
На режим18 прогретых запросов и558 timed decode steps; всего48 запросов
и1536 output IDs, включая прогревы.

Context512, batch8, F32 KV, FA on, greedy, обычный physical SWA engine.
Cache до14 ГиБ с live clamp, mmap, reader1, chunk8 МиБ, prefill admission off.
MTP и trace выключены. Скорость — сумма decode steps / сумма generation time,
без первого output, prefill и загрузки, с sampling и записью F32 logits.

| Измерение | LRU, decay0 | Частота, decay65536 |
|---|---:|---:|
| Общая скорость, ток/с | 6,923 | 7,692 |
| Счёт, ток/с | 7,695 | 9,283 |
| Python, ток/с | 6,328 | 6,838 |
| Русский текст, ток/с | 6,879 | 7,350 |
| Среднее полное время прогретого запроса, с | 7,938 | 7,472 |
| Decode H2D за558 steps, ГиБ | 688,733 | 607,674 |
| Заполнение кеша, D2D ГиБ | 688,456 | 117,821 |
| Вытеснения | 219 422 | 37 323 |
| Host fences доставки/fills | 309 087 | 127 015 |
| Cache reserved, среднее ГиБ | 12,708 | 12,711 |
| Cache payload, среднее ГиБ | 12,120 | 12,308 |
| Peak global VRAM, ГиБ | 29,512 | 29,506 |
| Peak global RAM, ГиБ | 111,038 | 111,540 |

Счётчики fills/fences включают prefill и decode прогретых запросов; prefill
admission off исключает prefill fills. Host fences уменьшились на58,9%; каждый
fill остаётся синхронным. Дополнительные scratch fences не изменились:89 676.
Это число ожиданий CPU, не измерение CUDA execution time.
История получила629 424 наблюдения, отказала155 607 admissions и проверила
4 715 745 кандидатов; максимум20 166 ключей. Эти CPU-затраты входят в замер.

По порядку процессов: **7,112 /7,826 /7,563 /6,744 ток/с**. Контроль заметно
замедлился к концу, особенно на счёте; его запросы сохранены в общем результате.
Оба новых процесса быстрее обоих контрольных. Их объединённая скорость выше
быстрейшего контроля на8,2%; весь ABBA-прирост11,1% нельзя считать свободным
от влияния фоновой нагрузки и состояния памяти.

System disk reads прогретых запросов:0,255 /0,460 /0,164 /0,133 ГиБ.
Эти счётчики включают все процессы и диски, не изолируют модель или H:.
Peak нового режима: VRAM92,66%, RAM88,84%. Это пики выборок, не гарантия
отсутствия более коротких пиков; runtime отдельно применяет memory admission.

**Цена адаптации:** на первых запросах каждой темы суммарный decode H2D
вырос с251,799 до347,100 ГиБ за186 steps на режим. Частотная история может
удерживать экспертов прежней темы. Эти6 запросов на режим сохранены отдельно,
их скорость составила3,584 и4,153 ток/с, но prefill и состояние файловых
страниц различались. Выигрыш прогретых повторов нельзя обещать для потока
однократных запросов на постоянно меняющиеся темы.

## Проверки и ограничения

- 27 cache cases,56 slab cases с5000 смешанных операций,21 frequency cases
  с ещё5000 операций и28 Python tests PASS. Проверены pins, повторные misses,
  decay, смена generation, reset статистики, pressure/OOM и failed fill.
- Runtime212/212 PASS при decay0 и65536: F32/mixed, file/mmap, sync/pipeline,
  readers1/2, маленький/большой кеш, tails, bytes/logits, отмена, ошибка reader,
  восстановление и drain. Prefill off не обучает историю.
- Full-model corpus6 prompts, включая SWA249, EN/RU/ZH/code/numbers:
  3 509 248 F32 logits и IDs совпали с сохранённым reference.
  Invalid request, cancel→fresh и unload PASS.
- ABBA:48 запросов,1536 output IDs,234 356 736 сравнённых F32 значений,
  exact logits/IDs, memory accounting и GPU-only audit PASS.

Измеренный executable `3c96331a…` и исходники сохранены в
`build-local/mimo2-frequency-measured`. После ABBA default переключён на65536;
отдельная проверка без env/CLI overrides, её hash и результаты EN/SWA249,
invalid/cancel recovery записаны в shipping-разделе validation JSON.

Прежняя редкая F32 synthetic ошибка max_abs0,0005061785 из MIMO-07/08 здесь
не повторилась. Её причина по-прежнему не установлена; успешные текущие
запуски не доказывают исправление. Арифметика и допуски не менялись.
96 kernel и132 graph checks относятся к прежним этапам и здесь не повторялись.

Периоды16384/131072 доступны для отдельных опытов, их скорость не измерена.
65536 выбран по проверенному сравнению с0, а не объявлен глобальным оптимумом.
Длительный pressure stress, context выше512, существенный compute overlap,
async cache fills, MTP quality/serving, API и sessions остаются открытыми.

## Управление и воспроизведение

Default отдельного MiMo engine — `STRATA_MIMO_CACHE_DECAY=65536`.
`0` возвращает прежний LRU admission. Переменную задавать до запуска.
При отключённом кеше история не используется. Setup/API profile ещё отсутствует.

```powershell
$mimoModel = 'H:\models\mimo-v2.6-flash\MiMo-V2.6-Flash-RL-GSQ-RCO-3.5bit.gguf'
python -m tools.benchmark_mimo2_cache --build build-local/mimo2-cuda --model $mimoModel --cuda-bin build-local/cuda-13.0/bin/x64 --axis frequency --fixed-slab-mib 16 --order 0,65536,65536,0 --output-dir build-local/mimo2-validation/frequency-abba-new
python -m tools.check_mimo2_cuda --build build-local/mimo2-cuda --cuda-bin build-local/cuda-13.0/bin/x64 --kind runtime --cache-slab-mib 16 --pipeline-batch 1 --cache-decay 65536 --output-dir build-local/mimo2-validation/frequency-runtime-new
```

Каталоги должны быть новыми. `check_mimo2_cuda` и `check_mimo2_engine` явно
выбирают `--cache-decay` (default0); для текущих defaults также нужны
`--cache-slab-mib 16 --pipeline-batch 1`. Старые benchmark axes slab/pipeline
явно выключают частоту, сохраняя контроль прежних экспериментов.
CPU CTest: `mimo2_frequency_ownership`, target `strata-mimo2-frequency-check`.
