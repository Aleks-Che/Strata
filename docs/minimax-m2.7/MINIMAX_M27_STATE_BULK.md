# MM27-29: пакетное копирование RAM-снимков KV

Дата: **2026-10-09**, `Asia/Yekaterinburg`.
[План](MINIMAX_M27_IMPLEMENTATION_PLAN.md),
[статус](MINIMAX_M27_IMPLEMENTATION_STATUS.md).

Продолжение [MM27-28](MINIMAX_M27_SESSION_CONTEXT.md). Путь host state IO
объединяет соседние отложенные операции одного tensor. Транспонированный V
раньше передавался отдельными строками: 1024 операции на слой, 62 слоя.
Новый путь читает ограниченный участок и упаковывает строки в прежний формат
снимка. При восстановлении сначала читает промежутки между строками, чтобы
сохранить данные других ячеек/сессий, затем отправляет заполненный участок.
Для плотного участка предварительное чтение не требуется.

CPU scratch ограничен 16 МиБ на вызов и освобождается при завершении IO.
При отказе выделения, разреженности ниже 25%, несовместимых дескрипторах или
нулевом лимите используется прежняя последовательность копирований. Проверки
границ используют вычитание без переполнения. Перекрывающиеся и переставленные
диапазоны сохраняют исходный порядок. План, который не подходит для объединения,
обходится один раз. Счётчики диагностики локальны для потока.

Формат состояния, K/V dtype, вычислительные CUDA kernels и sampler не меняются.
File/device state IO не патчатся. Изменяется только сгенерированная копия
`llama-context.cpp`; исходный dependency tree остаётся прежним. CMake проверяет
SHA исходника и единственность двух замен, manifest содержит hashes.
`STRATA_MM27_STATE_BULK=0` возвращает прежние копирования для A/B диагностики;
значение по умолчанию включает объединение. Session archive остаётся opt-in,
его default cap равен 0; prefix cache по умолчанию выключен.

## Проверка состояния и logits: PASS

Финальный EXE:
`98e85e80e2dbaff5dc38f03f1ab4039835fb8d7e384f4491e1d700c989c5d3d8`.
Старый EXE `275e89f8…`, исходники и manifest сохранены в
`build-local/minimax-m2-state-bulk-baseline` до изменений.
MSVC19.44.35222.0, CUDA13.0.48, Release/Ninja, SM120a.

1 200 CPU cases и12 CUDA checks прошли. CUDA fixture использует240 prompt
tokens при ctx512; счётчики зафиксировали21 bulk group и cap16 777 216байт.
Полный снимок5 901 216байт совпал побитно; logits продолжения также совпали.
Оба full-model режима прошли12 запросов каждый: **192 IDs /38 412 288 F32
logits** побитно совпали с MM27-28, все значения конечны. Дрейфа token IDs нет.
25 Python methods управления движком/сессиями и отдельный набор14 context
methods прошли; наборы пересекаются, их числа не складываются.

Проверочный корпус CPU сравнивает все байты, включая промежутки, пересечения,
перестановку диапазонов и metadata между записями. CUDA fixture сравнивает
полные сериализованные состояния до/после restore, logits продолжения,
разреженные позиции, сохранность второй сессии и восстановление после
отклонённого усечённого снимка. Счётчик групп подтверждает работу bulk-пути.

Full-model A/B: context4096/batch16, F32 activations/KV, TF32/FA/graphs/MTP off;
cache18432МиБ, arena64/reserve0, file/readers2/chunk4/lookahead/D2D,
RAM expert cache0, archive6144МиБ/4 записи. Greedy, 8 output tokens на запрос.
Оба режима измеряются одним EXE с `STRATA_MM27_STATE_BULK=0/1`.
Два последовательных процесса получают одинаковые cold2K/cold4K/short запросы,
затем три переключения на каждый target. Внутри процесса expert cache остаётся
прогретым; это три наблюдения на target, а не шесть независимых загрузок модели.
Измеряется полное request time с сохранением уходящей сессии, очисткой GPU KV,
restore, остаточным prefill и генерацией. Все IDs и полные F32 logits
сравниваются с проверенным корпусом старого EXE MM27-28.

RTX5090 32607МиБ, driver581.80, RAM125,555ГиБ, Windows;
target `H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf`.
Dependency `86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9`,
header `9a011b2ad6fb20db3bb0cdfe9af08665939ee1f0cb0609ffe2bfee7f31c825db`,
template `893d908f7b5cc65fdde270dcae5ea1a99647c6a7ce572ae874a57b7160069566`.

## Измерения

Медианы трёх тёплых переключений на каждый target; секунды. Порядок:
short → 2K → 4K → short. Перед восстановлением сохраняется уходящий target;
поэтому строка таблицы включает работу с двумя разными снимками.

| Возврат в сессию | Transfer legacy → bulk | TTFT legacy → bulk | Полный ответ legacy → bulk | Ускорение полного ответа |
|---|---:|---:|---:|---:|
| 2K | 4,491 → 1,919 | 6,567 → 3,968 | 7,941 → 5,277 | 1,51× |
| 4K | 4,842 → 0,551 | 6,957 → 2,604 | 8,024 → 3,608 | 2,22× |
| Short | 4,593 → 2,166 | 5,294 → 2,859 | 6,807 → 4,281 | 1,59× |

В 4K строке сам transfer ускорился8,79×. Полное время bulk во всех трёх
повторах ниже каждого legacy результата соответствующего target. Cold prefill
не включён в этот выигрыш. Старый cold2K/4K занял228,494/316,579с.
Это ускорение переключения сессий и короткого ответа после него. По семи
serial decode steps нельзя утверждать устойчивый прирост скорости decode.

Снимки содержат59/2039/4087 computed KV tokens и занимают
29 968 556 /1 035 642 236 /2 075 854 204байт. Маленький snapshot при ctx4096
ниже порога плотности; он сохраняет прежние мелкие копирования. Это объясняет
часть оставшегося transfer time в2K/short строках. Его отдельная оптимизация
потребует нового A/B.

## HTTP и границы

**10 live scenarios PASS:** обычный admission нового SHA, OpenAI/Anthropic
JSON/SSE, restore snapshot после128 output tokens при ctx512, отмена после
двух токенов, сохранность другой сессии, fresh recovery и restart.
Девять полных HTTP ответов /312 IDs совпали с прежним greedy reference;
два полученных токена отменённого ответа также совпали. Новая сборка принята
в `serve/minimax_m2_engine.py`; процессы проверки завершены.
Default archive0/prefix off сохранены.

Реальное давление RAM/VRAM94% было проверено на прежнем EXE в MM27-28.
На новой сборке этот gate впоследствии пройден в
[MM27-30](MINIMAX_M27_STATE_BULK_PRESSURE.md):19 scenarios,
RAM94,197%/VRAM94,403%, protected restore/admission/trim/обе отмены/recovery.
Сам A/B MM27-29 не заменяет эту отдельную pressure-проверку.
Полные state bytes проверены на CUDA fixture, полная модель сравнивает все
output logits. GPU batch1/8, shift, широкий quality corpus, independent model
oracle и старое reload discrepancy MM27-06 остаются открытыми.

## Команды и артефакты

```powershell
.\build-local\build-minimax-m2-state-bulk.bat
python tools/run_minimax_m2.py --engine build-local/minimax-m2-cuda/bin/strata-minimax-m2-state-bulk-check.exe -- build-local/minimax-m2-state-bulk-fixture-02
python tools/check_minimax_m2_state_bulk.py --gguf H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --out build-local/minimax-m2-state-bulk-ab-01
python tools/check_minimax_m2_state_bulk_http.py --gguf H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --offline build-local/minimax-m2-state-bulk-ab-01 --out build-local/minimax-m2-state-bulk-http-01
python -m unittest serve.test_minimax_m2_engine serve.test_minimax_m2_server tools.test_minimax_m2_sessions tools.test_minimax_m2_sessions_context
python -m unittest tools.test_minimax_m2_sessions tools.test_minimax_m2_sessions_context tools.test_minimax_m2_prefix_context tools.test_minimax_m2_http_context
python tools/validate_minimax_m2_state_bulk.py --ab build-local/minimax-m2-state-bulk-ab-01 --fixture build-local/minimax-m2-state-bulk-fixture-02 --live build-local/minimax-m2-state-bulk-http-01
```

Диагностические команды требуют новых output directories при повторном запуске.
Raw JSON/F32, stdout/stderr, EXE и source snapshots сохранены в указанных
каталогах `build-local`. `fixture-02/support` сохраняет проверенный fixture EXE,
исходники и manifest. Предварительный short-прогон на кандидате `af2482c0…`
прошёл27 full-logit сравнений и сохранён в `minimax-m2-state-bulk-short-01`;
он не заменяет приведённые выше проверки окончательной сборки.

[Итоговый CHECK](MINIMAX_M27_STATE_BULK_CHECK.json): PASS,
159 source hash checks и29 artifact hash checks. Build, fixture, оба A/B
прохода, HTTP и оба Python набора завершились с exit0. Первый запуск
валидатора до завершения HTTP отклонил ещё неполный отчёт; после завершения
всех сценариев итоговая валидация прошла.
