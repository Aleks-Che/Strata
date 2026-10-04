# GLM: конвейер и native MTP

Стенд: Windows, Ryzen 9 9950X, 128 ГиБ RAM, RTX 5090 32 ГиБ.
Модель: `GLM-5.3-Flash-Uncensored-IQ3_XXS.gguf`, 112,310 ГиБ.
Зависимость: Unsloth `86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9`.

## Что подключено

`runtime_memory.cpp` использует общий `StrataExpertPipeline`, `ExpertTransport`
и `ExpertDispatch`. После получения router IDs scheduler собирает gate/up/down
с теми же IDs в порядке будущего потребления. Cache hits удерживаются leases;
только misses поступают в ограниченный кольцевой буфер. Четыре pinned slots,
отдельный H2D stream и CUDA events защищают переиспользование slots. Producer
может читать и загружать следующие матрицы во время текущего GPU split.
Неизвестные маршруты следующего слоя не предсказываются. Все выбранные эксперты
и исходные квантованные байты сохраняются.

`mtp.hpp` запускает native NextN-блок из того же GGUF, с общими model/embedding/
output weights. Глубина — 1, 2 или 3 draft-токена. Проверка основной моделью
использует пакет carry + drafts; при отказе откатываются KDA, attention KV и
pooled indexer. Draft history восстанавливается с target features. Первый
draft-step уже использует правильную feature и повторно не исполняется.
STOP дожидается текущего GPU-вызова, затем очищает оба контекста.

Алгоритм sampling — target sample-and-match: sampler основной модели вызывается
ровно один раз на выходной токен. Greedy draft принимается только при совпадении
с этой выборкой. Лишних RNG draws нет; p/q rejection sampling здесь не используется.
Обычные penalties применяются к принятой последовательности.

Main/MTP имеют разные cache keys и историю частот. MTP ceiling входит в общий
VRAM budget. Контроллер по-прежнему использует global NVML и учитывает систему,
чужие процессы, state, scratch и allocator overhead. RAM/VRAM targets 95% —
целевые пределы контроллера, а не резервирование ровно 95% в каждый момент.

## Исправление арифметики короткой проверки

Первый реальный MTP-тест не прошёл. На первых 16 токенах IDs совпали, но max abs
logits был 1,8882; NMSE — 0,0029501. На 64 токенах разошлись и IDs.
[Отчёт до исправления](GLM53_FLASH_MTP_BEFORE_MATVEC_FIX.json).

Послойная проверка одиночного токена и первого токена пакета выявила разницу
между CUDA matvec и batched matmul. На IQ3 эти погрешности накапливались и меняли
маршрутизацию. В generated copy CUDA dispatch короткие пакеты 2–4 теперь
исполняют MUL_MAT по колонкам через тот же путь, что одиночный decode;
для MUL_MAT_ID этот режим применяется только к квантованным expert weights.
Условия CUDA graph synchronization согласованы с этим dispatch. Runtime задаёт
`STRATA_GLM_TOKENWISE_MATMUL=1` и при MTP off, чтобы precision policy была общей.
Оригинальный архив не меняется; перед патчем проверяется SHA-256 исходника.

Повтор CTest обнаружил нестабильный BF16 routed fixture при применении нового
пути к неквантованным экспертам. Для них восстановлен исходный CUDA dispatch.
В тестируемом IQ3_XXS routed weights квантованы, их путь не изменился.
[Регрессия коротких пакетов](GLM53_FLASH_TOKENWISE_QUANT.json): 108 сравнений,
из них 90 бит-в-бит (81 quantized и 9 dense F32/F16/BF16), 18 original routed
F32/F16/BF16 с прежними численными порогами. После исправления полный CTest
прошёл 14/14, затем этот тест прошёл восемь последовательных повторов.

[Полная проверка](GLM53_FLASH_MTP_CHECK.json): 39 prompt + 64 generated, F16 KV,
ctx2048/batch16, TF32/FA off, pipeline on, RAM/VRAM 95%. Все logits MTP-1/2/3
совпали с обычным decode бит-в-бит. Forced reject-first/reject-middle/accept-all,
seeded sampling, отмена draft и следующий чистый запрос также прошли.
Этот checker держит три rollback slots во всех вариантах; его скорости не
используются для выбора глубины по умолчанию.

## Сравнение скоростей

[Отдельные процессы для глубин 0/1/2/3](GLM53_FLASH_MTP_BENCHMARK.json).
Каждый процесс делает один разогревающий запрос и три измеряемых повтора.
39 prompt, 64 output, greedy, одинаковые IDs; файловый кэш ОС не очищался.
Скорость = 63 / DONE.decode_seconds; включает sampling, draft, verify, repair
и pipe output, исключает загрузку, warmup и prefill. Chunk 4 МиБ, MTP ceiling
512 МиБ, budgets 95%/95%. Rollback slots соответствуют выбранной глубине.

| MTP | Первый запрос, ток/с | Медиана прогретых, ток/с | Принято / предложено |
|---|---:|---:|---:|
| off | 6,478 | 8,584 | — |
| 1 | 5,225 | 8,730 | 31 / 32 |
| 2 | 1,829 | 7,431 | 37 / 50 |
| 3 | 4,933 | 6,555 | 41 / 66 |

Это один prompt на одной машине; большее число draft-токенов не гарантирует
ускорения. Повторы в одном процессе сохраняют expert cache, но очищают весь
sequence state. Дальнейший подбор размера кэша и контрольный повтор сохраняются
отдельными отчётами:

| Вариант | Измеряемых повторов | Медиана, ток/с |
|---|---:|---:|
| MTP 1, cache 256 МиБ, chunk 4 МиБ | 5 | 7,287 |
| MTP off, контроль в той же серии | 5 | 8,338 |
| MTP 1, cache 512 МиБ, chunk 8 МиБ | 5 | 8,457 |

[Cache 256](GLM53_FLASH_MTP_CACHE256_BENCHMARK.json),
[chunk 8](GLM53_FLASH_MTP_CHUNK8_BENCHMARK.json). Все выходные IDs совпали с
reference, процессы завершились с exit 0. В серии cache 256 первый запрос
дал 0,371 ток/с, а последующие росли от 2,815 до 8,760: состояние файлового
и expert cache заметно влияет на результат. Это не изолированная оценка
стоимости самого параметра cache ceiling.

В локальном `strata-glm5next.json` выбран **MTP 1, cache 512 МиБ, chunk 4 МиБ**,
pipeline on, ctx2048/batch16/threads4, RAM/VRAM 95%. Это самая высокая измеренная
медиана среди этих вариантов. Преимущество над off около 1,7%, сопоставимо с
разбросом; универсальная оптимальность для других запросов не подтверждена.

## Проверка конвейера и протокола

[CUDA trace](GLM53_FLASH_PIPELINE_GPU_TRACE.json) содержит шесть target graphs:
три prefill и три decode. Сумма объединённых H2D-интервалов — 1853,812 мс,
compute-split spans — 775,289 мс, их пересечение — **2,166 мс**. Это небольшое
перекрытие интервалов на фактических CUDA streams. Compute span включает
зависимости/ожидания и не равен времени занятых SM; данные не доказывают
значительное одновременное исполнение H2D и вычислительных kernels.
Оптимизация степени перекрытия остаётся отдельной задачей.

Тот же диагностический запуск подтвердил бит-в-бит все **9 912 320** baseline
F32 logits из P1 и все варианты MTP/rollback/sampling/cancel. Скорости этого
запуска не используются в таблицах: trace синхронизирует границы графов.

Рабочий профиль MTP 1 прошёл [9 native pipe сценариев](GLM53_FLASH_PIPELINE_MTP_PIPE.json)
и [8 HTTP сценариев](GLM53_FLASH_PIPELINE_MTP_API.json): OpenAI/Anthropic JSON/SSE,
disconnect во время prefill/decode, восстановление и unload. Проверены
[synthetic MTP/EOS](GLM53_FLASH_MTP_FIXTURE.json), 14 candidate CTest,
13 shared transport/cache CTest и 85 Python tests. Точные команды, версии
и границы применимости проверок: [manifest](GLM53_FLASH_PIPELINE_MTP_VALIDATION.json),
[логи/команды](GLM53_FLASH_PIPELINE_MTP_TESTS.txt).

## Параметры

| Параметр | Значение |
|---|---|
| `--expert-pipeline 0/1` | Synchronous reference / shared async pipeline |
| `--expert-chunk-mib 1..16` | Размер каждого из четырёх pinned/device slots |
| `--mtp 0..3` | Off / число native draft-токенов |
| `--mtp-cache-mib 0..32768` | MTP ceiling внутри общего VRAM budget; 0 означает bypass |
| `--ram-target-percent 95` | Глобальная цель RAM |
| `--vram-target-percent 95` | Глобальная цель VRAM |

MTP требует `batch-size >= depth + 1`. Setup экспортирует эти параметры;
CLI без профиля сохраняет opt-in для MTP и pipeline. Рабочие настройки конкретного
ПК находятся в `strata-glm5next.json`.

`STRATA_GLM_TRACE_GRAPHS=1..128` включает отдельный диагностический захват CUDA
events на фактических compute/H2D streams. Он синхронизирует измеряемые графы и
не должен быть включён в throughput benchmarks. Без него CPU submit/wait timers
не выдаются за GPU execution time или доказательство overlap.

## Ограничения

Session snapshots/reuse (P4) ещё не реализованы; запросы начинают с чистого
hybrid state. Длинный sparse-контекст, другие prompts/кванты, Linux/HIP и полные
Qwen/DeepSeek модели в этой работе не проверялись.

В [официальном config](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/c54b8d1/config.json)
`index_share_for_mtp_iteration=true` (проверено 2026-10-04). Выбранный кандидат
пересчитывает draft indexer; повторное использование его выборки между draft
итерациями остаётся отдельной оптимизацией. Наличие флага не объявляется её реализацией.
