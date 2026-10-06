# Оптимизированный запуск Step-3.7-Flash

Отдельный файл в корне репозитория: **`START-STEP37-FAST.bat`**.
Двойной щелчок запускает сервер и открывает веб-чат на
`http://127.0.0.1:8093`. Закрытие сервера — Ctrl+C в его окне.
Если порт занят, можно запустить `START-STEP37-FAST.bat --port 8094`.

Это вариант для обычного чата с контекстом **4096 токенов**. Он использует
проверенные ускорения без MTP. Результат около 14,8 токена/с из STEP-17 относится
к отдельному MTP checker с лимитом 480 позиций; этот launcher его не запускает
и такую скорость произвольного чата не обещает.

## Конфигурация

- Основная модель: `H:\models\Step-3.7-Flash\UD-Q4_K_S`, первая из четырёх частей.
- Отдельный бинарник: `build-local/step35-cuda/bin/strata-step35-fast.exe`.
- Отдельный профиль: `build-local/step35-cuda/step18-fast-profile/step37.json`.
- GPU-кэш экспертов: cap **8192 МиБ**, с прежним динамическим ограничением
  глобальной памяти и target95. Это проверенный HTTP-бюджет; offline cap16GiB
  автоматически не переносится в запуск чата.
- Асинхронная доставка: один reader, четыре слота по 8 МиБ.
- **Allocation reuse и групповые копии включены** новыми native options
  `--expert-cache-reuse on --expert-pipeline-batch on`.
- Заполнение кэша во время prefill выключено; decode admission включён.
- MTP выключен, draft не загружается. Early refill выключен, host copy — CRT:
  отдельного устойчивого выигрыша этих вариантов без MTP не установлено.
- Batch17, F32 KV, FA/TF32 выключены. Greedy sampling по умолчанию;
  обычный native sampler доступен для настроек запроса.
- Лимит ответа автоматически ограничивается оставшимся контекстом.
- Логи и статистика находятся в каталоге нового профиля.

Прежний `strata-step35.exe`, старый HTTP-профиль и запуски других моделей
сохранены. Новый target использует тот же native pipe protocol; по умолчанию
новые опции выключены, профиль включает их явно. В INFO выводятся
`expert_cache_reuse=1`, `expert_pipeline_batch=1`, `mtp=0`.

## Сборка и повторная подготовка

Нужны ранее настроенная private CUDA-сборка Step и виртуальное окружение Strata.
В developer shell с CUDA DLL в PATH:

```powershell
cmake --build build-local/step35-cuda --target strata-step35-fast -j 4
python -X utf8 tools/prepare_step35_profile.py --model H:/models/Step-3.7-Flash/UD-Q4_K_S --engine build-local/step35-cuda/bin/strata-step35-fast.exe --output-dir build-local/step35-cuda/step18-fast-profile --cuda-dir build-local/cuda-13.0 --optimized
```

Подготовка требует новый каталог и не перезаписывает существующий профиль.
Если используется другой output directory, нужно указать его в `STEP_PROFILE`
в BAT-файле. Основной native binary не заменяется при сборке target `-fast`.

## Проверки STEP-18

Дата: 2026-10-06, Windows/Ryzen 9950X/128GiB/RTX5090.
Сборка отдельного target и 22 synthetic pipe checks прошли: greedy/seeded
sampling, A→B→A, malformed input, STOP во время prefill, восстановление после
отмены и QUIT во время запроса. Пять неверных конфигураций отклонены до loading.
34 Python profile/template/API tests прошли. BAT проверен через `--help`.

Полная модель: **10/10 HTTP-ответов** совпали с сохранёнными IDs, включая
OpenAI/Anthropic JSON/SSE, вызов инструмента и результат инструмента.
Три disconnect-проверки прошли: prefill/decode/partial tool; завершение отмены
заняло 2,639/0,212/0,422 с. После каждой отмены следующий ответ exact.
Выгрузка через Service, повторная загрузка новым процессом и ответ после неё
прошли. Это проверка серверного пути; клики в браузере не автоматизировались.

Отдельный native запрос **511 входных + 16 выходных токенов** пересёк SWA512;
IDs и все F32 logits побитово совпали с прежним контролем. Два следующих P1
ответа тоже совпали по IDs/logits. Итого 13 завершённых reference comparisons
в новых full-model сериях, помимо трёх отменённых запросов.

Пики всей системы за HTTP и native серии: RAM **93,48%**, VRAM **24565 МиБ**.
Monitor95 не срабатывал, процессы завершились с кодом 0. Forced OOM и
предельная внешняя нагрузка этим не проверены. Бюджет8GiB оставлен с учётом
наблюдаемой RAM; свободная VRAM сама по себе не доказывает запас всей системы.
Парный speed benchmark старого и нового HTTP-профиля в этом этапе не проводился;
ускорение отдельных механизмов измерено в STEP-12/13.

Данные, профиль, hashes, команды и checks:
[STEP37_FLASH_FAST_LAUNCH.json](STEP37_FLASH_FAST_LAUNCH.json).
Новый engine SHA-256:
`41971c992e0bcb5f0858f44e168d75ed8552c605fe58101e7d9f1b5abf1b8763`.
Прежний engine SHA-256 сохранён:
`32a88bd85855395dbc79df2da323d971a9543c2847c618e7e34245a771bf9df3`.
