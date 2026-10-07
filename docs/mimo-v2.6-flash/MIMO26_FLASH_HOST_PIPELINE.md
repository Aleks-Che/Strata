# MiMo: пакетная передача служебных хвостов

MIMO-13, 2026-10-07, Asia/Yekaterinburg. **Пакетная передача включена по умолчанию:
10.336 ток/с, +24.02% к быстрейшему контролю 8.334.**
Это короткие запросы на Ryzen 9 9950X / RTX 5090 /128 GB RAM, без MTP.
Отдельный опыт с early host refill выигрыша не дал; он остаётся off.

[Измерения](MIMO26_FLASH_HOST_PIPELINE_BENCHMARK.json) и
[проверки, hashes, неуспешные попытки](MIMO26_FLASH_HOST_PIPELINE_VALIDATION.json).
Исходная ревизия `e06ac9e970f828051d198ec91a65a3373aee2244`, поверх незакоммиченных MIMO-11/12.

## Скорость

| Опыт | Контроль, ток/с | Кандидат, ток/с | Решение |
|---|---:|---:|---|
| Early host refill | 7.992 | 7.912 | Off, -1.00% |
| Пакетная передача хвостов | 7.768 | 10.336 | On; средний ABBA +33.05% |

Порядок второй серии: **8.334 / 10.300 /
10.372 / 7.275 ток/с**, off/on/on/off.
Последний контроль заметно медленнее первого, поэтому основной вывод —
**+24.02% к быстрейшему контролю**, без обещания устойчивых +33%.
Оба повтора кандидата быстрее обоих контролей.

| Prompt | Контроль, ток/с | Пакетная передача, ток/с |
|---|---:|---:|
| count | 10.086 | 13.813 |
| code | 6.795 | 8.611 |
| ru | 7.150 | 9.830 |

Среднее полное время прогретого запроса: 7.191 →
5.866с; prefill 3.201 →
2.867с. Это та же серия, не отдельная оценка холодного TTFT.

## Что изменено

Для CUDA-копии выбранной матрицы иногда нужны ещё 512 байт следующего эксперта.
При попадании самой матрицы в GPU-кеш эти хвосты раньше проходили через reader,
H2D ring, события и ожидания как отдельные маленькие задания.
Теперь отсутствующие хвосты одного известного маршрута собираются в pinned RAM
и отправляются одним H2D. Затем те же байты доставляются в scheduler scratch.

Дополнительная память ограничена **2 MiB RAM + 2 MiB VRAM**: 4096 разрешённых
матриц ×512 байт. Буфер не пересекается со scratch; перед повторным использованием,
отменой, ошибкой и освобождением выполняется drain copy stream. Route pins,
scratch fence, частотный допуск и математические GPU-операции сохранены.
Общий `backends/common/expert_pipeline.hpp` не изменён этой работой.

| За 18 измеренных запросов на вариант | Контроль | Пакетная передача |
|---|---:|---:|
| H2D submissions, включая prefill | 935088 | 476626 |
| Decode H2D, GiB | 594.088 | 594.201 |
| CPU slot wait, среднее мс/запрос | 937.732 | 102.948 |
| CPU submit, среднее мс/запрос | 361.882 | 74.320 |

Число H2D-операций уменьшилось на 49.03%. Кандидат объединил
487725 хвостов в 29613 пакетов.
Decode H2D изменился лишь на 0.019%.
Средняя физическая память кеша 12798.000 →12798.667 MiB;
это измерение уменьшения числа операций, не снижения веса модели.
CPU waits включают prefill и перекрываются во времени; их нельзя складывать
или выдавать за CUDA execution time. Промежуточные D2D остаются.

Early refill уменьшил CPU slot waits 886.8 →
645.5мс/запрос, но не повысил скорость.
Его контрольные процессы иногда ограничивали кеш до 12742/12753 MiB;
это дополнительное ограничение точности сравнения около 1%.

## Проверки и память

- CUDA runtime **224/224 в обоих режимах**. Добавлены 12 проверок F32/mixed
  с несоседними экспертами, побитовыми logits, byte observer, отменой и ошибкой
  guard upload с последующим восстановлением. На каждом формате 1224 хвоста →68 пакетов.
- Шесть full-model corpus cases, включая prompt249/SWA128, **3509248 logits exact**;
  invalid ID, cancel→fresh и unload прошли. Финальная сборка повторила корпус
  при удалённых из environment обоих новых флагах; INFO подтвердил guards1/early0.
- В каждой ABBA серии 60 запросов/1920 output IDs/**292945920 F32 значений** exact.
  Это parity на проверенных запросах с тем же квантованным графом, не независимый quality benchmark.
- 7 CPU CTest и 29 Python tests PASS. Новые комбинации grouped fills/early/guards
  отдельно не измерялись; grouped fills остаются off.

| Серия | Пик глобальной VRAM | Пик глобальной RAM |
|---|---:|---:|
| Early ABBA | 29.506 GiB /92.66% | 115.071 GiB /91.65% |
| Guards ABBA | 29.506 GiB /92.66% | 112.637 GiB /89.71% |
| Final default corpus | 29.503 GiB /92.65% | 112.807 GiB /89.85% |

Пики выборочные, включают другие процессы. Лимиты 95% сохранены; матрицы считаются
на GPU. Рабочий cache request14 GiB сохранён; фиксированные 12800 MiB использовались
для A/B, оставаясь под live clamp. Измерены context512/batch8, F32 KV, FA on,
greedy, TF32/graphs off, mmap/reader1/chunk8, slab16, decay65536, prefill admission off.
Три темы, два прогревочных и три измеренных запроса по 32 токена на тему и процесс.
Decode =31 шаг ×requests /generation time; исключены load, prefill и первый output,
включены sampling и экспорт logits. Trace в замерах выключен.

Первая проверка early-off повторила известный F32 mismatch:
`f32/file_pipe_2_8/logits`, max_abs 0.0005061784759163857, nmse 5.205e-7,
worst 12975; byte checks прошли. Повтор и early-on прошли 212/212.
Старый mixed diagnostic MIMO-10 также остаётся открытым. Эти оптимизации
не объявляются исправлением обеих диагностик.

Первая попытка проверки guards остановилась на coverage guard: исходный bias
выбирал соседних экспертов, новый путь не срабатывал. Добавлен отдельный permuted
bias в fixtures; прежние случаи сохранены. Число runtime cases212/244 выросло
до 224/256. Один Python test сначала ещё ожидал старое число; после обновления
coverage expectation29 тестов прошли. Все первоначальные логи сохранены.

## Настройки и воспроизведение

`STRATA_MIMO_PACK_GUARDS=1` — новый default;0 возвращает прежние мелкие передачи.
Применяется только к mmap +tensor batching; file/per-range остаются прежними.
`STRATA_MIMO_EARLY_HOST_REFILL=0` — default;1 оставлен для отдельных опытов.
`expert_stage_mib=34` включает guard staging; `pipeline_device_bytes` считает
старый ring32 MiB, дополнительно учитывать `pipeline_guard_capacity_bytes=2 MiB`.
`pipeline_chunks` включает H2D guard batches; `pipeline_guard_ranges/batches/bytes`
показывают объединённые хвосты. Неиспользованные guard bytes при abort входят
в `pipeline_unused_bytes`.

Сборка — по [README backend](../../backends/mimo2/README.md).
Для повторения использовать новые output dirs:

```powershell
python -m tools.benchmark_mimo2_cache --build build-local/mimo2-cuda --cuda-bin build-local/cuda-13.0/bin/x64 --model H:/models/mimo-v2.6-flash/MiMo-V2.6-Flash-RL-GSQ-RCO-3.5bit.gguf --axis guards --order 0,1,1,0 --expert-cache-mib 12800 --warmups 2 --output-dir build-local/mimo2-validation/guards-abba-new
python -m tools.check_mimo2_cuda --build build-local/mimo2-cuda --cuda-bin build-local/cuda-13.0/bin/x64 --kind runtime --cache-slab-mib 16 --pipeline-batch 1 --cache-decay 65536 --pack-guards 1 --output-dir build-local/mimo2-validation/guards-runtime-new
```

Следующий MIMO-14: GPU D2D/compute timeline, промежуточные копии и chunks4/16.
Полноценный compute overlap, mixed/F32 диагностика, длинные контексты, API и
batched target parity для MTP остаются открыты.
