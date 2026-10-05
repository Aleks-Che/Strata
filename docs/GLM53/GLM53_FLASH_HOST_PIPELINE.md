# P3.3j: подготовка host-буфера и стоимость копирования

Дата: 2026-10-05. База Strata `b7e0976ecad062dd67d4cb0290495fa53d9763a2`
+ рабочие изменения. Ryzen 9 9950X, 128 ГиБ RAM, RTX 5090 32 ГиБ,
Windows, MSVC 19.44, CUDA 13.0.48. Модель — локальная
`GLM-5.3-Flash-Uncensored-IQ3_XXS.gguf`, 112,310 ГиБ.

**Решение:** устойчивого прироста токенов/с в этой итерации не установлено.
Рабочие настройки скорости сохранены: MTP off, token-batch MMVQ on,
обычный memcpy, без CPU affinity. Новый режим раннего заполнения host-буфера
остаётся opt-in. RAM/VRAM targets95 сохранены. Backend identity локального
профиля обновлён под новую сборку.

## Разделение времени жизни host- и device-слотов

Прежний producer ждёт завершения GPU-потребителя и только затем копирует
следующий блок в pinned RAM. Host-буфер уже свободен после завершения его
предыдущего H2D; ждать последующего GPU-потребителя для этой операции не нужно.

Добавлен `STRATA_GLM_EARLY_HOST_REFILL=1`:

1. Дождаться старого H2D (`ready`), защищающего host-буфер.
2. Подготовить следующий блок в том же host-буфере.
3. Дождаться старого потребителя (`used`), защищающего device-буфер.
4. Отправить новый H2D и опубликовать его событие.

Ожидание потребителя выполняется вне mutex отправки копий. Другой reader
может отправить более раннюю работу; блокирующий барьер в copy stream не
добавляется. Число слотов и память staging не увеличены: четыре host- и четыре
device-слота по 4 МиБ в измеренном профиле. Оба ожидания входят в `slot_wait_us`.
Прежний путь остаётся default в общем конвейере и в GLM. INFO и memory snapshots
сообщают `expert_early_host_refill`; benchmark проверяет запрошенное значение.

## Полная модель

Повторяемый prompt B: 49 входных и 64 выходных токена, greedy. Три отдельных
engine в порядке off/on/off, каждый с 4 warmup и 5 timed запросами. Все 27
ответов совпали с сохранёнными token IDs. Контекст 2048, batch16, threads4,
mmap, decode reader1, slabs16, compaction1, MTP0, F16 KV, TF32/FA off.
Файловый кэш ОС не очищался. Других наших GPU-нагрузок во время измерений не было.

| Режим | Медиана ток/с | Диапазон timed | Медиана всего запроса, с | Медиана cache, ГиБ |
|---|---:|---:|---:|---:|
| Прежний, до | 12,935 | 12,149–13,116 | 7,502 | 17,178 |
| Раннее заполнение | 12,565 | 12,464–12,801 | 7,459 | 17,061 |
| Прежний, повтор | 11,833 | 11,329–12,365 | 8,160 | 17,047 |

Ток/с = `63 * 1000 / DONE.decode_ms`, без загрузки модели и prefill.
Время всего запроса включает prefill. Медианы этих величин не обязаны
соответствовать одному и тому же запросу. Контроль заметно изменился между
сериями; ёмкость cache также не была фиксированной. Процент ускорения нового
режима из этих результатов выводить нельзя.

| Режим | Подготовка host, с | Ожидание слотов, с | Ожидание публикации потребителем, с |
|---|---:|---:|---:|
| Прежний, до | 6,093 | 0,785 | 2,717 |
| Раннее заполнение | 6,501 | 0,206 | 2,680 |
| Прежний, повтор | 6,610 | 0,868 | 2,937 |

Это медианы **сумм CPU wall-time**, включающие prefill и decode; потоки могут
работать одновременно. Эти числа нельзя складывать для получения времени
запроса или считать длительностью GPU-операций. Новый режим уменьшил ожидание
повторного использования слотов, но этого недостаточно для доказательства
ускорения генерации. Host-подготовка включает доступ к mmap и memcpy.

Максимальные sampled значения трёх серий: RAM 95,77%, VRAM 95,43%.
Targets95 — реактивные бюджеты, а не жёсткий лимит всей ОС.
Обычный локальный профиль по-прежнему не включает compaction; приведённые
скорости относятся к экспериментальному repeated-B/compaction профилю.

[Полные запросы, paging counters и memory snapshots](GLM53_FLASH_HOST_PIPELINE_BENCHMARK.json).

## Проверка CPU affinity

До изменения engine выполнены три отдельные серии на прежней сборке
`ae83ef59c5904fe911c0ab470fb720182851a6c1c8d56000bd8773b9f768be64`.
Внутри одного прогретого engine порядок timed запросов —
all/limited/limited/all, три блока. Только первый вариант имел 6 warmup;
остальные — 4. Всего 50 точных ответов. Изменялся только созданный benchmark
процесс; его исходная affinity восстановлена перед завершением, exit0.

| Ограничение | Маска | Все CPU, ток/с | Ограничение, ток/с |
|---|---|---:|---:|
| Один logical processor на каждом из 16 физических ядер | `0x55555555` | 11,987 | 11,897 |
| Первые восемь физических ядер | `0x5555` | 13,106 | 12,925 |
| Последние восемь физических ядер | `0x55550000` | 13,165 | 13,396 |

Это медианы шести timed ответов каждого варианта в своей серии. Половины
определены порядком core masks Windows, топология CCD отдельно не проверялась.
Небольшая разница последней серии не установлена как устойчивое преимущество;
affinity по умолчанию не менялась. Между сериями проценты не вычисляются.

[Полные результаты affinity](GLM53_FLASH_HOST_PIPELINE_AFFINITY.json).

## Изолированное memcpy и H2D

Добавлен диагностический benchmark с вращающимся источником 2 ГиБ,
pinned destination и блоками 64 КиБ, 256 КиБ, 1 МиБ, 3 МиБ, 4 МиБ.
Отдельно измеряются CPU copy, H2D через CUDA events и последовательность
CPU copy + H2D + ожидание. Шесть раундов ABBAAB: обычный memcpy и streaming
stores SSE2/AVX2 с SFENCE перед CUDA. Проверены обычная и write-combined pinned
память; SSE2 offset0, AVX2 offset0/17. Offset указан относительно начала
вектора, без утверждения о выравнивании его base address на 64 байта.
Все 180 финальных GPU-сравнений побайтово точны.

Ниже AVX2, обычный pinned buffer, offset0; медианы трёх раундов в ГиБ/с.

| Блок | CPU memcpy / streaming | H2D memcpy / streaming | Последовательно memcpy / streaming |
|---|---:|---:|---:|
| 64 КиБ | 37,34 / 21,29 | 19,51 / 26,94 | 2,14 / 2,23 |
| 1 МиБ | 44,82 / 22,88 | 28,75 / 50,22 | 13,41 / 12,38 |
| 3 МиБ | 23,23 / 22,99 | 52,82 / 52,83 | 14,72 / 14,65 |
| 4 МиБ | 23,39 / 23,33 | 52,86 / 52,88 | 14,65 / 14,92 |

Улучшение отдельного H2D не гарантирует ускорение подготовки и передачи вместе:
для 1 МиБ streaming stores медленнее по итоговому времени. Для больших блоков
нет последовательного выигрыша между вариантами. Production остаётся на
`std::memcpy`; новые intrinsics используются только диагностическими binaries.
AVX2 targets запускаются вручную на подходящем CPU, в общий CTest не включены.
Семантика fence проверялась по [Intel Intrinsics Reference](https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-8/cacheability-support-intrinsics-002.html).

[Все microbenchmark результаты](GLM53_FLASH_HOST_PIPELINE_COPY.json).

## Воспроизведение

Собрать candidate и отдельную конфигурацию transport согласно
[инструкции backend](../../backends/glm5next/README.md), с CUDA runtime в PATH.
Для сравнения полной модели, меняя `--early-host-refill` 0 → 1 → 0 и имя output:

```powershell
python tools/benchmark_glm5next_mtp.py --profile strata-glm5next.json --reference build-local/glm-mtp-cache-reference-b.json --output build-local/glm-refill-on-b.json --depths 0 --warmups 4 --repeats 5 --cache-compact 1 --early-host-refill 1 --paging-counters
python tools/benchmark_glm5next_affinity.py --profile strata-glm5next.json --reference build-local/glm-mtp-cache-reference-b.json --output build-local/glm-affinity.json --layout second-half --warmups 4
build-local/glm5next-transport/strata-glm5next-host-copy-avx2-bench.exe 0
```

Содержимое reference B сохранено в affinity JSON. Локальные profile, GGUF,
CUDA toolkit и build-local не распространяются в репозитории.

## Проверки и ограничения

- 32 уникальных candidate CTest: 31 в общей серии, затем отдельно добавленная
  проверка прежнего конвейера DeepSeek. Полный MMVQ matrix test также прошёл.
- 14 transport CTest, в том числе 864 сравнения матриц, guards, два consumer
  streams, native/mmap/auto, prefill/decode, отмена и повторное использование.
- Синтетически остановленный GPU-потребитель: следующий host copy завершился,
  следующий H2D не отправлялся до освобождения device-слота; все bytes точны.
- 83 Python tests, SSE2 и ручной AVX2 copy tests со смещениями 0..63,
  краями порога и недоступной страницей сразу за источником.
- Полная GGUF с early refill: все 9 912 320 сохранённых F32 logits совпали
  побитово; native pipe прошёл 9 сценариев, включая seeded repeat, отмену
  prefill/decode и правильное продолжение после отмены. Exit0.
- Все 20 зафиксированных source/binary SHA-256 совпали после измерений.
  Итоговый engine: `4da6f03b41c08f8cfddb5967addd5358b8735d1c8101cbb7b4167a1eb9ab92f8`.

[F32 проверка](GLM53_FLASH_HOST_PIPELINE_CHECK.json),
[native pipe](GLM53_FLASH_HOST_PIPELINE_PIPE.json),
[manifest, команды и hashes](GLM53_FLASH_HOST_PIPELINE_VALIDATION.json).

F32 проверка использует reference A и MTP-capable checker с `baseline-only`;
это не повторная полная проверка MTP на большой модели. Полный HTTP, длинные
контексты, Linux/HIP и полноценный inference Qwen/DeepSeek не повторялись.
В ходе benchmark контроллер менял cache budget, системные paging counters
включают другие процессы. На основе коротких серий нельзя обещать выигрыш
на других prompts или оборудовании.

Следующая задача: раздельные реальные CPU memcpy/H2D/wait measurements после
prefill при сопоставимом cache budget; сокращение повторных переносов и
возвратов страниц. Изолированный быстрый H2D и меньшее ожидание producer
сами по себе ещё не означают ускорение полного decode.
