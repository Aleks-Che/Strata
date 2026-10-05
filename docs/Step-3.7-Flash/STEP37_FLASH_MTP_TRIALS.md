# Step-3.7-Flash: первые MTP trials

Дата: 2026-10-05. Ревизия перед работой: `3f757b0e7996c3bc0dd7859076cabdd3cc723f6d`.

**MTP работает в отдельном checker.** Финальный контроль: 9,146 → 10,602 токена/с
после прогрева, **+15,9%** к генерации. Среднее время пары запросов вместе с prefill:
19,972 → 17,859 с, **на 10,6% меньше**. Это два коротких контрольных запроса,
а не оценка произвольного диалога или длинного контекста.

[Машинный отчёт](STEP37_FLASH_MTP_TRIALS.json) сохраняет все token IDs, времена,
acceptance, memory peaks, hashes и неуспешные попытки. Полные memory samples и
stderr остались в `build-local/step35-cuda/step11-mtp/<experiment>/`.

## Стенд и метод

- Windows, Ryzen 9 9950X, RTX 5090, драйвер 581.80. Наблюдаемый power limit GPU —
  400 W; настройки питания в этой работе не менялись.
- RAM: 128 ГиБ установлено, 125,555 ГиБ видимо ОС; VRAM: 32607 МиБ.
- Trunk: локальный UD-Q4_K_S, 4 части, 106,323 ГиБ, путь из плана.
- Проверенная зависимость: Unsloth `86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9`.
- Context2048, batch/ubatch17, F32 KV, TF32/FA off, pipeline1/chunk8MiB,
  expert prefill admission off. Кэш экспертов: cap16384MiB; runtime ограничивает
  фактический объём по свободной VRAM, оставляя 5% + 256 МиБ.
- Два P1 prompt: 38/34 входных и 54/88 выходных токена, включая EOG.
  Greedy, свежие KV и speculative state на каждый запрос, `p_min=0.6`.
- Каждый проверяемый depth прогревался на обоих prompts; затем два измеряемых
  круга с обратным порядком depths во втором. GPU-кэш между запросами сохранялся.
- Токены/с = сумма выходных токенов / сумма времени генерации. Загрузка модели
  и prefill не входят в этот показатель. Время пары включает prefill и decode,
  но не загрузку модели. Базовый режим без MTP запускается **без загрузки draft**,
  чтобы освободившуюся VRAM тоже мог использовать кэш основной модели.

Всего завершено **100 проверочных запросов**, из них **60 с MTP**. Во всех
успешных сериях IDs совпали с сохранённым native reference, включая EOG.
Три незавершённые серии описаны ниже и не включены в эти 100 запросов.

## Sidecar и реализация

Скачан [официальный Q8_0](https://huggingface.co/stepfun-ai/Step-3.7-Flash-GGUF/blob/0b69336d2fd2adfdef9c66e425f7778196c31482/Step3.7-flash-mtp-Q8_0.gguf):

- 3 707 276 416 байт; SHA-256
  `469a81667a6cd6d87a85d501d57155fd90cee5af7010fd289c5169881763fd57`.
- Локально: `build-local/models/step37-mtp/Step3.7-flash-mtp-Q8_0.gguf`.
- 55 тензоров, три плотные головы `blk.45/46/47`, ширина4096. Routed experts
  внутри этого sidecar отсутствуют.
- Vocabulary/merges/control token types/template и trunk architecture metadata
  совместимы. PAD: target2 / draft1; `add_eos_token` отсутствует у target и равен
  false у draft. Checker получает уже токенизированные target IDs и не применяет
  эти draft-настройки.
- Token embedding Q8_0 побайтно совпадает с trunk:
  `139872af2cbc43f70a4709cf66ca371a8925496b2eb577b130bd519256e84c56`.

`STRATA_STEP_MTP_PROBE=ON` добавляет отдельный `strata-step35-mtp-check`.
Использован закреплённый `common_speculative` MTP driver: перенос hidden state,
teacher-forced catch-up, trained head chain, target verification и accepted-prefix
rollback. Проверка greedy выполняется основной моделью для каждого выданного токена.

`--draft-placement shared-embedding` оставляет CPU-mapped копию embedding и
неиспользуемые global output/norm в RAM. После сравнения всех quantized bytes
draft заимствует GPU embedding основной модели. Это убирает около **1069,9 МиБ**
дублирующих/неиспользуемых GPU weights. У каждой головы проверяется наличие
собственного output/norm. Основная модель живёт дольше обеих contexts и draft;
владение её GPU buffer не передаётся. Матрицы, используемые в вычислениях,
остаются на GPU.

## Скорость

Финальные серии на одной версии checker:

| Вариант | Decode, токена/с | Пара запросов с prefill, с | Accepted/proposed |
|---|---:|---:|---:|
| Без draft, большой кэш основной модели | 9,146 | 19,972 | — |
| Резидентный Q8, MTP2 | 10,274 | 18,453 | 174/188 = 92,55% |
| Общий embedding, Q8, MTP2, demand paging | **10,602** | **17,859** | 174/188 = 92,55% |

В той же финальной серии с загруженным shared draft, но выключенным MTP,
получено 8,415 токена/с. Это диагностический контроль стоимости занятой VRAM;
для сравнения с обычным приложением используется строка без draft выше.

Первый полный sweep резидентного Q8: MTP1 **10,067**, MTP2 **10,447**,
MTP3 **10,299** токена/с. Acceptance: 97,10% / 92,55% / 87,27%.
Сопутствующий контроль без draft: 9,027 токена/с. Разброс между сериями виден
в JSON; лучшие отдельные прогоны не заменяют контрольное сравнение.

Кандидат для следующего этапа: **Q8_0, depth2, p_min0.6, shared embedding,
demand paging, cache cap16384MiB**. Другие пороги и квантования пока не измерены;
это не доказательство универсального оптимума.

## RAM и размещение матриц

В основных сериях пик общей VRAM — примерно **30213–30247 МиБ**, около 92,7%.
Первый консервативный cache4GiB использовал лишь18149MiB и дал6,794 токена/с;
после увеличения cache cap он не используется как baseline.

Прогрев RAM (`--prefault-experts`) проверен отдельно:

- Запас512MiB ниже 95% оказался недостаточен: пик119,977GiB, monitor остановил
  процесс при95,557% RAM до завершения первого запроса.
- Запас8192MiB тоже оказался недостаточен для второго prompt: пик119,761GiB,
  95,386%; остановлен только тестовый процесс.
- Запас24576MiB прошёл весь sweep: **117,607GiB RAM / 93,67%**, VRAM30216MiB.
  До READY прочитано67,643GiB экспертных страниц, затем нужные страницы
  добавлялись во время генерации. Startup14,83с против3,73с у финального shared
  demand режима. Один из холодных ответов дал2,971 токена/с.
- После прогрева MTP2 дал10,612, MTP3 —10,669 токена/с. Это практически уровень
  shared demand10,602; дополнительного throughput gain от заполнения RAM
  не установлено. Prefault остаётся opt-in, его запас не является гарантией
  admission длинных запросов или внешней нагрузки.

Матрицы неиспользуемых голов также проверены в RAM. Для этого отдельная опция
сборки `STRATA_STEP_MTP_ACTIVE_CATCHUP=ON` генерирует hash-guarded копию
`common/speculative.cpp`, ограничивая catch-up запрошенным числом голов.
Оригинальные извлечённые исходники не изменяются. При depth1/2 будущие головы
не вычисляются; новые requests глубже resident head count запрещены.

Результат с одной головой в VRAM: **9,332** токена/с; с двумя — **9,432**.
Ответы и acceptance совпали с исходным driver, VRAM для кэша освободилась,
но ускорения не получилось. Эта оптимизация сборки **выключена по умолчанию**.
Причина разницы скорости этими короткими trials не установлена.

Первый resident loader trial также остановлен до генерации: upstream оставил
token embedding на CPU. Guard обнаружил это; явный GPU override исправил
размещение. Последующие resident trials успешно завершены.

## Повторить лучший проверенный вариант

Сборка и варианты описаны в [backend README](../../backends/step35/README.md).
Для этих измерений сохранён точный бинарник:

```powershell
python -X utf8 tools/check_step35_mtp.py --engine build-local/step35-cuda/bin/strata-step35-mtp-check-resident-tested.exe --model H:/models/Step-3.7-Flash/UD-Q4_K_S/Step-3.7-Flash-UD-Q4_K_S-00001-of-00004.gguf --draft build-local/models/step37-mtp/Step3.7-flash-mtp-Q8_0.gguf --draft-placement shared-embedding --depths 0,2 --cache-mib 16384 --rounds 2 --output-dir build-local/step35-cuda/step11-mtp/repeat-shared
```

Python monitor следит за общей RAM/VRAM, сохраняет peaks и завершает только
созданный им child при пересечении95%. Это опрос с интервалом, а не обещание
нулевого мгновенного превышения — оба неудачных RAM trials сохранены в отчёте.

## Границы результата

Checker отклоняет requests больше480 суммарных позиций. Не проверены SWA
rollback через512, длинный prefill/генерация, stochastic sampling, sessions,
HTTP/pipe stop/cancel и production MTP integration. MTP по умолчанию в рабочем
приложении **не включён**.

Существующий `strata-step35.exe` не пересобирался и сохранил SHA-256
`32a88bd85855395dbc79df2da323d971a9543c2847c618e7e34245a771bf9df3`.
Профили, сервер и backends Qwen/GLM/DeepSeek не изменены. Проверены24 существующих
Step Python tests, компиляция обоих вариантов probe, guard отключённого
active-heads режима и `git diff --check`.
