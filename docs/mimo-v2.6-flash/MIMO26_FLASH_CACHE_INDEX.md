# MiMo: индекс резидентных экспертов

Проверено **2026-10-07**, Asia/Yekaterinburg. Windows, AMD Ryzen9 9950X,
RTX5090 32 ГБ, 128 ГБ RAM; MSVC Release, CUDA13.0.48, sm120, Unsloth `86ebfef2`.
Модель `MiMo-V2.6-Flash-RL-GSQ-RCO-3.5bit.gguf`,134982426368 байт.

Новый индекс убирает повторные поиски резидентных записей при вытеснении
и отдельную таблицу защиты записей. В CPU-тесте144000 обращений время
63,330→32,505 мс, сокращение на48,673%.
Это тест метаданных кеша, не вычислений матриц.

**Прогретая генерация осталась около8,4 ток/с:** новое среднее8,390,
заключительный контроль8,401 (разница−0,128%).
ABBA с одинаковым бюджетом12800 МиБ:7,208 /8,410 /8,370 /8,401 ток/с.
Агрегат двух контролей7,759→8,390 (+8,138%) увеличен медленным первым
контролем; его нельзя считать устойчивым эффектом оптимизации.
Изменение сохраняется как снижение CPU-затрат на метаданные,
без заявления об ускорении decode; настройки engine остаются прежними.
Это измерение короткого корпуса на данном ПК; системная нагрузка и прогрев
могут менять время отдельных процессов. Проценты предыдущих этапов не складывать.

[Benchmark](MIMO26_FLASH_CACHE_INDEX_BENCHMARK.json),
[проверки и source hashes](MIMO26_FLASH_CACHE_INDEX_VALIDATION.json).

## Устройство индекса

Резидентные записи находятся в `unordered_map` с уже проверенным хешем MIMO-11.
Список LRU хранит указатели на записи, каждая запись содержит свой pin count.
Выбор кандидата читает запись непосредственно; ему больше не нужны поиски
в двух `std::map`. `PlanPins` резервирует вектор указателей по размеру маршрута.
Рост hash table сохраняет адреса элементов; map iterators между обращениями
не сохраняются. Полный ключ generation/tensor/expert по-прежнему сравнивается.

Сохранены прежние64 кандидата, порядок LRU, частотный допуск/затухание,
charge classes, pins будущих hits и pending tickets экспериментальных fills.
При ошибке добавления записи или LRU node выделение возвращается allocator.
GPU kernels, scheduling, shared helpers, веса и типы вычислений не менялись.
Матрицы модели выполняются на GPU. Grouped fills остаются off.

Изменены `expert_cache.hpp`, добавлены `check_cache_index.cpp` и цель CMake.
Engine SHA256 `e4a21c3bbbcc61fdc9aa6c64281ef52b88415920b97b2e2bd4f161f4582b1f1c`,
snapshot `build-local/mimo2-index-measured`.
Контроль MIMO-11 `35c789471818e7543d8732ede8c001f9dda3caed589f330c05120417b383e405`,
snapshot `build-local/mimo2-hash-measured`.

## Проверки

- До изменения снят контрольный replay ordered-map реализации. Затем новый
  индекс воспроизвёл его digests для LRU и decay65536:19200 операций,
  по4096 начальных заполнений каждого replay. Проверяются вся резидентность
  после каждого маршрута, счётчики, payload, pressure и несколько generations.
  Отдельно внутри replay: рост таблицы при живых pins, дубли и вложенные планы.
  Порядок PRNG draws зафиксирован явно; первый вариант порядка изменил trace
  и не прошёл digest. Восстановлен порядок исходного MSVC reference,
  исходные golden значения сохранены. Финальный replay PASS.
- 7 CPU CTest и29 Python regression tests PASS. Прежние CPU fill/slab tests
  также проверяют pending reservations, failures, reuse и physical budgets.
- CUDA runtime212/212 PASS: file/mmap, sync/pipeline, bytes/logits,
  ошибки/отмена и восстановление. Opt-in fill1 CUDA в этом этапе не запускался.
- Full GGUF6 cases, включая SWA249:3509248 logits exact,
  error/cancel→fresh и unload PASS.
- Full-model ABBA60 запросов /1920 output IDs /292945920 F32 значений exact.
  Сравнены20 cache/transfer counters каждого запроса с первым процессом:
  900 сравнений, различий0. Фиксированный бюджет соблюдён: True.

CPU microbenchmark делает4096 начальных fills и6000 маршрутов по24 обращения,
с pins, mixed sizes, frequency, pressure и проверкой payload. Четыре процесса
старый/новый/новый/старый: 63,319 /32,917 /32,094 /63,342 мс;
все повторили digest16589823275981314459. Время включает подготовку кеша.

## Условия замера полной модели

Cache12800 МиБ, slab16, decay65536, tensor delivery1, fill0; mmap,
reader1/chunk8 МиБ, prefill admission off. Context512, batch8, F32 KV,
FA on, greedy, без MTP и trace. На тему2 прогрева и3 измеренных запроса
по32 output tokens. Темы: счёт, код, русский текст. KV свежий; кеш и история
сохраняются в процессе. На вариант18 измеренных запросов/558 decode steps.

Скорость — сумма decode steps / generation time, без load/prefill и первого
output в числителе; sampling, проверка и запись logits включены.

| Тема | Контроль, ток/с | Новый индекс, ток/с |
|---|---:|---:|
| count | 10,115 | 10,112 |
| code | 6,612 | 7,417 |
| ru | 7,322 | 8,075 |
| Вместе | 7,759 | 8,390 |

Decode H2D: 594,088→594,088 ГиБ за558 шагов.
Средний cache reserved: 12,498→12,498 ГиБ,
payload: 12,128→12,128 ГиБ.
Новая сборка: peak global VRAM29,449 ГиБ
(92,483%), RAM109,178 ГиБ
(86,956%). Это выборки; runtime отдельно применяет95% guard.
Disk counters каждой серии в JSON включают все процессы.
Engine default request14 ГиБ/live clamp95% сохранён;12800 МиБ — только тест.

## Команды и продолжение

```powershell
& .\build-local\build-mimo2-index.cmd
python -m unittest tools.test_mimo2_gguf tools.test_mimo2_cuda tools.test_mimo2_engine tools.test_mimo2_tokenizer tools.test_mimo2_drafts
python -X utf8 build-local/mimo2-validation/run-index.py
```

Exact commands и исходные reports: `index-runtime-0`, `index-corpus`,
`index-abba` в `build-local/mimo2-validation`. Для повторения нужны свежие
output directories; GPU runs выполняются последовательно.
Исторические F32/mixed synthetic расхождения остаются открытыми: этот PASS
не объявляется их исправлением, пороги не менялись. MTP serving не включён.
Следующие кандидаты: повторное использование планов, kernel/H2D timeline,
прямое чтение cached weights без промежуточного D2D и измерение chunks4/16.
