# Step P0.4: CUDA matrix/router validation

Стенд: Windows, Ryzen 9 9950X, RTX 5090 32 ГиБ. Отдельный build directory
`build-local/step35-cuda`; Unsloth llama.cpp
`86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9`, CUDA 13.0.48, SM120a.
Это synthetic fixtures, не полная модель и не измерение скорости генерации.

## Критерии, заданные до запуска

Код: [check_kernels.cpp](../../backends/step35/check_kernels.cpp).

- Матрицы Q4_K/Q8_0/Q6_K/F32: input 512, output 64, 288 experts, 8 выбранных.
  Обычный MUL_MAT, routed broadcast input и routed per-expert input; batch1/4/17.
- Для каждого случая — contiguous и padded storage, ненулевые offsets,
  промежутки между expert matrices, token/route inputs и строками IDs.
  Read-only storage/guards не должны меняться; padded/compact GPU outputs
  должны быть bit-exact. Отдельные случаи row padding внутри quant blocks
  этим тестом не покрыты: строки весов непрерывны, padding расположен между experts.
- Веса генерируются с фиксированным seed, упаковываются CPU quantizer выбранной
  зависимости, затем деквантуются для независимого dot-product с накоплением
  в double. Сравниваются CUDA/scalar, CPU/scalar и CUDA/CPU, а не исходные
  неокруглённые веса с результатами quantized matmul.
- F32: NMSE ≤ 1e-10 **и** max absolute error ≤ 5e-5.
  Quantized: NMSE ≤ 1e-4 **и** max absolute error ≤ 0,02.
  NMSE — сумма квадратов ошибки / сумма квадратов эталона; 1e-4 соответствует
  относительной RMS-ошибке 1%. Допуск учитывает внутреннюю квантизацию activations.
  Все outputs/reference должны быть конечными.
- Router: sigmoid(logits), bias только для выбора, sorted top-8,
  нормализация исходных unbiased probabilities и умножение на 3.
  IDs должны совпадать точно; веса — NMSE ≤ 1e-10 и max abs ≤ 2e-6.
  Минимальный промежуток между сравниваемыми scores обязан быть > 1e-6,
  чтобы fixture не полагался на неопределённый порядок почти равных scores.
- Каждая CUDA graph node проверяется через `ggml_backend_supports_op` и
  запускается напрямую через CUDA backend. CPU backend используется только
  для явно указанного тестового эталона; fallback scheduler отсутствует.
- В процессе задан `NVIDIA_TF32_OVERRIDE=0`. Исходный код зависимости проверяется
  первым; если он нарушает этот режим, ошибка и отдельное исправление сохраняются.
  Численные допуски после получения результатов не расширяются.

Ожидается 78 случаев: 72 matrix/layout + 6 router/layout. Эти проверки не
доказывают корректность полного Step graph, SWA state, rollback или Strata
expert transport — для них остаются P0.5/P1/P3.

## Результаты

**78/78 PASS** после двух исправлений зависимости, применённых только в Step
build directory. Все численные допуски сохранены, все input guards неизменны,
compact/padded matrix outputs bit-exact. CTest: 2/2 PASS (version + kernel suite).

| Тип | Максимальный NMSE CUDA/scalar | Максимальная абсолютная ошибка |
|---|---:|---:|
| F32 | 1,866e-14 | 1,192e-7 |
| Q8_0 | 2,239e-5 | 0,003593 |
| Q6_K | 2,257e-5 | 0,003584 |
| Q4_K | 5,510e-5 | 0,006609 |

Числа относятся только к описанным synthetic матрицам. Это не оценка качества
ответов и не погрешность полной модели. [Полный итоговый отчёт](STEP37_FLASH_CUDA_KERNELS.json).

### Обнаруженные проблемы и исправления

1. Первый unpatched запуск выполнил 75 случаев и остановился на
   `CUDA0 unsupported SIGMOID` для padded router batch4. До остановки 14 F32
   matrix cases нарушили допуск: max abs до 0,000606 при `NVIDIA_TF32_OVERRIDE=0`.
   [Первый отчёт](STEP37_FLASH_CUDA_UNPATCHED.json),
   [исходный binary/source manifest](STEP37_FLASH_CUDA_INITIAL_MANIFEST.json).
2. Перед sigmoid добавлен явный `ggml_cont` на GPU. Это устраняет неподдержанный
   layout без CPU fallback. Повторный unpatched запуск: 78 случаев, 14 F32 FAIL,
   остальные 64 PASS. [Отчёт](STEP37_FLASH_CUDA_UNPATCHED_PACKED_ROUTER.json).
3. `StrictF32.cmake` отключает custom TF32 MMF только при
   `NVIDIA_TF32_OVERRIDE=0`. Он используется и при выборе kernel, и при расчёте
   необходимости синхронизации CUDA graph. После него осталось четыре FAIL:
   F32 routed batch4/17 с padded inputs, max abs до 317,51.
   [Промежуточный отчёт](STEP37_FLASH_CUDA_STRICT_F32_ONLY.json).
4. Причина: fallback gather вычислял индекс исходной строки как
   `token * ne11 + lane`, игнорируя padding между tokens. `RoutedStrides.cmake`
   заменяет логическое число lanes на физическое `nb12 / nb11` с проверкой
   диапазона int32. После этого **78/78 PASS**, без изменений fixture и допусков
   относительно второго запуска. Матрицы, IDs и padding не повреждаются.

F32 fallback внутри CUDA backend содержит CPU bookkeeping для группировки IDs
и stream synchronization; сами матричные операции выполняются на GPU.
Проверка не обещает, что scheduling целиком исполняется на GPU, и не измеряет
эффективность этого fallback. Локальные routed weights модели — Q4_K.

Оба изменения генерируют новые `.cu` только в `build-local/step35-cuda`.
Исходный архив, распакованные исходники и GLM/DeepSeek/Qwen не модифицированы.
Исходные SHA-256 и patch anchors проверяются CMake. Детали:
[StrictF32.cmake](../../backends/step35/StrictF32.cmake),
[RoutedStrides.cmake](../../backends/step35/RoutedStrides.cmake),
[финальный manifest](STEP37_FLASH_CUDA_BUILD_MANIFEST.json).

Tokenizer после изменений также проверен: CPU oracle **2190/2190**, CUDA-linked
oracle с обоими patches **2190/2190**; это те же inputs, не 4380 уникальных cases.
[CPU recheck](STEP37_FLASH_TOKENIZER_CPU_RECHECK.json),
[CUDA-linked parity](STEP37_FLASH_TOKENIZER_CUDA_PARITY.json).
Новые Python tests: 21/21 PASS. Общие reader/tokenizer не менялись после STEP-01.

Следующий этап — P0.5: реальный synthetic Step graph, head gates, Q/K norms,
MoE clamps, full/SWA RoPE, microbatch, окно 511/512/513/1024 и rollback. Полная
модель и GPU memory peaks в этом этапе ещё не проверены.
