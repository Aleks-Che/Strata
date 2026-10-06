# Step-3.7-Flash: write-combined staging

Дата: 2026-10-06, Asia/Yekaterinburg. Поверх STEP-13/14, ревизия базы
`9c457bf526e5916d513bedccf0a25878cf36d026`; прежние незакоммиченные изменения сохранены.

**Ускорение не получено: cached pinned RAM сохраняется.** В этих сериях
WC дал −4,90% без MTP и −2,19% с MTP2. Сопутствующая CPU-нагрузка различалась,
особенно с MTP; эти числа не отделяют собственную стоимость WC от её влияния.
Оснований включать WC по умолчанию нет.

[Машинный отчёт](STEP37_FLASH_WRITE_COMBINED.json) сохраняет40 точных ответов,
timings, counters, CPU load, memory peaks, hashes, команды и проверки, включая
первую неуспешную попытку и минимальный reproducer обработки исключения.

## Изменение

Offline checker принимает `--pipeline-host cached,wc`. `cached` оставляет
обычную cacheable pinned RAM, `wc` передаёт `cudaHostAllocWriteCombined`
существующему shared primitive. Сам `backends/common/expert_pipeline.hpp`
не изменён. CPU только записывает staging payload, GPU читает его через H2D.
Четыре slots, их ёмкость, memcpy, ready/used events и cache pins одинаковы.

Новый Step API `strata_step_pipeline_config_staging()` задаёт тип host memory.
Смена типа обязательно пересоздаёт drained ring между запросами; время
переключения записывается отдельно и не входит в generation/request timers.
Существующие `config()`/`config_ex()` выбирают прежнюю cacheable RAM, даже если
предыдущий запрос включал WC. Resource guards и ограниченный cache trim сохранены.
JSON RESULT подтверждает фактический выбранный параметр `pipeline_write_combined`.
Runner отклоняет неподтверждённый WC, в том числе со старым probe binary.

## Корректность и исправление обработки отказа

- 72 CUDA ring checks PASS: 1/2 readers ×4/8/16 МиБ ×early off/on ×cached/WC,
  payload66 МиБ+513 байт, wrap, tails/guards, неверный source, cancel/restart.
- 201 runtime cases PASS: F32/mixed weights, cold/warm/eviction, побайтовое
  сравнение GPU copies и побитовое сравнение logits, prompt513. Новые проверки
  меняют host policy через оба API и принудительно освобождают cache перед
  повторными uploads. WC без readers отклоняется.

Первый runtime checker завершился с кодом−1073740791 без итогового JSON.
Причиной оказался тест намеренно неверной конфигурации: исключение из функции
с C linkage при MSVC `/EHsc`. Отдельный минимальный двухфайловый reproducer
подтвердил тот же fail-fast; добавление `noexcept(false)` дало успешный catch
и exit0. Диагностический вывод в большом checker маскировал проблему компиляции;
финальная серия201 cases прошла после удаления этого вывода.

Три Step entry points настройки конвейера теперь явно объявлены
`noexcept(false)` и в header, и в определении. Это задаёт контракт обработки
отказа, не меняя алгоритм копирования. Предположение компилятора о C linkage
описано в [документации MSVC /EH](https://learn.microsoft.com/en-us/cpp/build/reference/eh-exception-handling-model?view=msvc-170).
Это исправление конкретных configuration entry points; аудит всех остальных
throwing C APIs данным этапом не заявляется.

## Метод

Windows, Ryzen9 9950X, RAM128 ГиБ (125,555 видимо), RTX5090 32607 МиБ,
driver581.80, power limit400W без изменения. Main — локальный UD-Q4_K_S
106,323 ГиБ; MTP — официальный Q8_0, shared embedding, три GPU heads,
исходный catch-up, depth2 и p_min0.6.

Context2048, batch17, F32 KV, FA/TF32 off, reader1, slots8 МиБ,
prefill admission off, cache reuse on, cache cap16384MiB с динамическим
ограничением по глобальной VRAM. Без MTP draft не загружается. Используются
batch без MTP и batch+early с MTP2. Прогрев всей RAM не включался.

Два P1 prompts:38/34 input tokens,54/88 output tokens включая EOG. Каждый
запрос очищает KV/speculative state, expert cache сохраняется внутри процесса.
На host policy по два warmups; затем четыре парных раунда, обратный порядок
конфигураций на нечётных раундах. Decode — суммарные output tokens / время
генерации; полная пара включает prefill, исключает load и смену ring.

CPU load измеряется вокруг каждого запроса: system и child time, проценты
от всей мощности32-поточного ПК. Разность включает другие приложения, ОС
и часть работы драйвера, а не только посторонние задачи. Нагрузка менялась;
сравнение абсолютных скоростей с другими днями/сериями не является A/B тестом.

## Скорость

| Режим | Cached, токена/с | WC, токена/с | Полная пара cached →WC |
|---|---:|---:|---:|
| Без MTP /batch | 11,697 | 11,123 (−4,90%) | 15,597 →16,395с |
| Q8 MTP2 /batch+early | 13,701 | 13,401 (−2,19%) | 14,273 →14,571с |

Без MTP WC медленнее во всех четырёх раундах. Среднее wall time подготовки
весов за генерацию выросло3860,7 →4153,7мс. Это CPU counter отдельного reader,
он перекрывается с GPU compute/H2D и не складывается с ними как CUDA timeline.
Для no-MTP WC не выбирается.

С MTP2 WC медленнее в трёх из четырёх раундов. Среднее время подготовки
весов3919,5 →4006,0мс; H2D73,583 →73,841 ГиБ/ответ. Без MTP H2D74,764 →75,439
ГиБ/ответ. Кэш между запросами сохраняется, однако при смене ring его состав
может немного меняться; frozen-cache replay здесь не выполнялся.

Средняя по времени запросов CPU-нагрузка **вне model process**:

| Серия | Cached | WC |
|---|---:|---:|
| Без MTP | 11,90% | 12,93% |
| MTP2 | 11,22% | 20,19% |

Особенно для MTP2 это существенное смешение факторов. Вывод — ускорение WC
не подтверждено данным экспериментом, а не универсальное замедление WC на
любом оборудовании. Основным вариантом остаётся cached; остальные параметры
STEP-13 (reuse, batching, MTP2/early при использовании draft) сохранены.

**40/40 responses exact**, включая8 warmups,20 ответов с MTP. Это повторения
двух коротких P1 prompts. Acceptance в обоих измеряемых MTP режимах348/376
=92,55%. Два процесса завершились с кодом0, monitor95 не сработал.
Пики RAM:74,28% без MTP /77,29% с MTP; VRAM30217/30277 МиБ (92,67/92,85%).
24 существующих Step Python tests, `py_compile`, `git diff --check` PASS.

## Границы

Изменения доступны только в отдельном checker. Существующий `strata-step35.exe`
и HTTP-профиль не заменялись, shared pipeline и другие backends не менялись.
Checker ограничен greedy и480 суммарными позициями. MTP SWA512, длинные
requests, stochastic sampling, pipe/HTTP, stop/cancel/recovery и pressure
admission остаются отдельными этапами.

Следующий performance-шаг — измерить отдельно cache admission/построение
router plan и опрос глобальной памяти. Их вклад в задержку пока не измерен;
менять алгоритм кэша или снижать частоту resource checks без данных не следует.

## Воспроизведение

Raw results: `build-local/step35-cuda/step15-wc/`, полные stderr/memory samples
в подпапках `no-mtp` и `mtp2`. Для повторения выбирать новый output dir.

```powershell
$stepProbe = 'build-local/step35-cuda/bin/strata-step35-mtp-check-wc-tested.exe'
$stepModel = 'H:/models/Step-3.7-Flash/UD-Q4_K_S/Step-3.7-Flash-UD-Q4_K_S-00001-of-00004.gguf'
python -X utf8 tools/check_step35_mtp.py --engine $stepProbe --model $stepModel --depths 0 --cache-reuse on --pipeline-modes batch --pipeline-host cached,wc --rounds 4 --output-dir build-local/step35-cuda/repeat-wc-no-mtp
python -X utf8 tools/check_step35_mtp.py --engine $stepProbe --model $stepModel --draft build-local/models/step37-mtp/Step3.7-flash-mtp-Q8_0.gguf --draft-placement shared-embedding --depths 2 --cache-reuse on --pipeline-modes batch-early --pipeline-host cached,wc --rounds 4 --output-dir build-local/step35-cuda/repeat-wc-mtp2
```

Проверенный probe SHA-256:
`ba77cb78457f323c644f2777e34b07fa24e42cac727934154790378960121651`.
Рабочий engine сохранил SHA-256
`32a88bd85855395dbc79df2da323d971a9543c2847c618e7e34245a771bf9df3`.
