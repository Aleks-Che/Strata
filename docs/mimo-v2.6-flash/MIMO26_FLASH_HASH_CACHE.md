# MiMo: распределение ключей частотного кеша

Проверено **2026-10-07**, Asia/Yekaterinburg. Windows, AMD Ryzen9 9950X,
RTX5090 32 ГБ, 128 ГБ RAM; CUDA13.0.48, MSVC Release, sm120, Unsloth `86ebfef2`.
Модель `MiMo-V2.6-Flash-RL-GSQ-RCO-3.5bit.gguf`, 134 982 426 368 байт.

Исправлено распределение ключей частотной истории: отдельный CPU microbenchmark
стал примерно44 раза быстрее. Изменение включено в обычную сборку,
отдельного флага нет; GPU kernels и результаты вычислений сохранены.

**Устойчивый прирост полной генерации пока не подтверждён.**
ABBA с фиксированным кешем12,5 ГиБ дал 7,377→7,682 ток/с
(+4,133%) по среднему, но новый вариант оказался на
3,455% медленнее быстрейшего контроля.
В первой серии обычного профиля среднее выросло на5,89%; значительный
разброс контролей и фоновая нагрузка мешают приписать этот процент хешу.

[Замеры с фиксированным бюджетом](MIMO26_FLASH_HASH_FIXED_BENCHMARK.json),
[первая серия с обычным live clamp](MIMO26_FLASH_HASH_BENCHMARK.json),
[проверки и исходные hashes](MIMO26_FLASH_HASH_VALIDATION.json).

## Изменение

Старый `MatrixHash` помещал номер тензора в биты16 и выше. В MSVC таблица
истории использует число корзин, равное степени двойки: при65536 корзинах
тензоры разных слоёв с одинаковым номером эксперта попадали в одну корзину.
Новый хеш перемешивает generation/tensor/expert во всех битах результата.
Сравнение полных ключей сохраняется: хеш не заменяет проверку идентичности.

Тест с141 тензором ×256 экспертами, всего36096 ключей:

| Распределение в65536 корзинах | Старый хеш | Новый хеш |
|---|---:|---:|
| Занятые корзины | 256 | 27650 |
| Самая длинная цепочка | 141 | 7 |

Отдельный CPU microbenchmark частотной истории,4 млн запросов к ключам,
каждый восьмой также обновляет счётчик, ABBA:
2152,96 /48,7006 /49,3949 /2146,13 мс.
Среднее уменьшение времени этой операции — примерно44 раза. Это **не**
ускорение всей генерации. Во всех четырёх проходах checksum истории8384815.

В engine изменён только `MatrixHash` в `expert_cache.hpp`. Порядок LRU,
частотный допуск/затухание, pins, бюджеты, GPU kernels и scheduler сохранены.
Общий `backends/common/expert_frequency.hpp` и другие backend не менялись.
Новые файлы/цели тестов проверяют CPU-метаданные; матрицы модели считает GPU.

## Полная модель

Сохранённый контроль: `build-local/mimo2-literal-measured`,
SHA256 `75b86c2cb359db8e83c2f7882947629efde1875e8e53c775694ef233cd91a748`.
Новая сборка: `build-local/mimo2-hash-measured`,
SHA256 `35c789471818e7543d8732ede8c001f9dda3caed589f330c05120417b383e405`.
Разница в engine относительно контроля ограничена хешем. Добавлены CPU checker
и явные build dependencies для frequency/fill tests: после пересборки все6 CTest
прошли. Benchmark runner получил `--warmups` и `--expert-cache-mib`; defaults
сохраняют прежнюю схему, фиксированная серия явно задаёт новые параметры.

Slab16, decay65536, tensor delivery1, синхронные fills, cache12800 МиБ,
mmap/reader1/chunk8 МиБ, prefill admission off. Context512, batch8,
F32 KV, FA on, greedy. MTP и trace выключены. KV свежий перед каждым запросом;
кеш и история сохраняются между запросами и темами одного процесса.

Каждый из4 процессов проходит3 темы. На тему первые2 запроса — прогрев,
затем3 измеренных повтора по32 output tokens. На вариант18 измеренных
запросов /558 timed decode steps. Скорость — сумма decode steps / сумма
generation time, без загрузки/prefill и первого output в числителе;
проверка logits, sampling и запись F32 включены. Это короткий warm workload.

| Тема | Старый хеш, ток/с | Новый хеш, ток/с |
|---|---:|---:|
| count | 9,408 | 8,333 |
| code | 6,502 | 6,756 |
| ru | 6,823 | 8,164 |
| Все темы | 7,377 | 7,682 |

Процессы ABBA по порядку: 6,876 /7,663 /7,702 /7,957 ток/с.
Decode H2D за558 шагов: 594,088→594,088 ГиБ.
Средний cache reserved: 12,498→12,498 ГиБ,
payload: 12,128→12,128 ГиБ.
Во всех запросах этой серии live clamp оставил ровно12800 МиБ бюджета.
Во всех4 процессах также совпали20 cache/transfer counters каждого из15
запросов: 900 сравнений с первым процессом. H2D, hits/misses,
заполнения, вытеснения, frequency history и payload совпали точно.

Во время второй серии наблюдалась активность фонового загрузчика и
MemCompression. Другие процессы не останавливались; численная величина
end-to-end прибавки зависит от системной нагрузки. Фиксированный бюджет
исключает различия cache policy в этой серии, но не устраняет все помехи
измерению времени. Отдельный CPU microbenchmark и correctness проверены.

До неё выполнен ABBA обычного профиля с cache request14 ГиБ и одним
прогревочным запросом на тему. Скорости процессов:
7,781 /7,964 /7,988 /7,299 ток/с.
Агрегат: 7,532→7,976 ток/с,
+5,889%; к быстрейшему контролю +2,507%.
Контроль заметно менялся между процессами; эта величина не считается
изолированной оценкой эффекта хеша. Поэтому добавлена вторая серия с
фиксированным бюджетом и двумя прогревами. Оба исходных отчёта сохранены.
Default cache request14 ГиБ и live clamp95% в engine **не менялись**.

Пик новой сборки: VRAM29,194 ГиБ
(91,683%), RAM107,765 ГиБ
(85,831%). Пики измерены выборками; runtime
отдельно соблюдает95% global admission. System disk reads включают другие
процессы; значения каждого процесса сохранены в JSON. Проценты разных этапов
и контрольных серий не складывать.

## Проверки

- 6 CTest PASS: прежние ownership/frequency/fill/literal проверки плюс
  1488776 сравнений истории старого и нового хеша. Проверены периоды8,
  16384,65536,131072, очистка по лимиту97/65536, несколько generations,
  UINT32_MAX tensor/expert, seed/erase и итоговые scores.
- 29 Python regression tests PASS.
- Runtime212/212 PASS: bytes/logits, file/mmap, pipeline/sync, pressure,
  ошибки/отмена и восстановление. Grouped fills остаются выключенными.
- Full GGUF:6 corpus cases, включая SWA249, 3509248 logits
  побитово совпали с сохранённым reference; error/cancel→fresh и unload PASS.
- Фиксированный ABBA:60 запросов /1920 output IDs и
  292945920 F32 значений совпали побитово.
- Обычный ABBA:48 запросов /1536 IDs и234356736 F32 значений также exact.

## Воспроизведение

Из корня репозитория, после конфигурации MiMo CUDA:

```powershell
& .\build-local\build-mimo2-hash.cmd
& .\build-local\mimo2-cuda\bin\strata-mimo2-hash-check.exe --benchmark
python -m unittest tools.test_mimo2_gguf tools.test_mimo2_cuda tools.test_mimo2_engine tools.test_mimo2_tokenizer tools.test_mimo2_drafts
python -X utf8 build-local/mimo2-validation/run-hash.py
python -m tools.benchmark_mimo2_cache --build build-local/mimo2-cuda --cuda-bin build-local/cuda-13.0/bin/x64 --model H:/models/mimo-v2.6-flash/MiMo-V2.6-Flash-RL-GSQ-RCO-3.5bit.gguf --axis binary --control-binary build-local/mimo2-literal-measured/strata-mimo2.exe --fixed-slab-mib 16 --order 0,1,1,0 --expert-cache-mib 12800 --warmups 2 --output-dir build-local/mimo2-validation/hash-fixed-abba
```

Команды runtime/corpus/ABBA и исходные reports находятся в
`build-local/mimo2-validation/run-hash.py` и каталогах `hash-runtime-0`,
`hash-corpus`, `hash-abba`, `hash-fixed-abba`; они также указаны в JSON. Повтор требует свежих
output directories. Контрольную и новую сборку запускать последовательно,
не пересобирать executable во время измерения.

## Открытые задачи

Исправление распределения хеша не локализует прежние intermittent F32/mixed
synthetic расхождения. Их исходные reports сохранены, пороги не ослаблялись;
успешный новый runtime не объявляется исправлением этих ошибок.
MTP serving и существенный overlap H2D/compute по-прежнему не подтверждены.
Далее — lookup резидентных записей, повторное использование планов scheduler
и отдельные измерения decay16384/131072 при смене темы.
