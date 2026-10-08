# MM27-18b: sampling и завершение ответов

Измерения: **2026-10-08–09**, Windows, RTX 5090 32 ГиБ, RAM125,555 ГиБ.
[План](MINIMAX_M27_IMPLEMENTATION_PLAN.md),
[статус](MINIMAX_M27_IMPLEMENTATION_STATUS.md).

## Контракт

В отдельном MiniMax benchmark/JSONL движке реализован optional sampler.
Запрос принимает объект `sampling` с `temperature`, `top_p`, `top_k`, `seed`.
Без него остаётся прежний greedy: temperature0, первый максимальный logit
при равенстве. Остальные defaults: top_p0,95, top_k40, seed42.

Temperature допускает0 или0,01..2; top_p — (0,1]; top_k — целое0..200064,
где0 отключает фильтр. Seed — целое0..4294967294. Значение4294967295
отвергается: dependency трактует его как случайный seed. Неверные типы,
неизвестные поля и неконечные числа отклоняются до prefill. Результат
содержит фактически использованные F32 параметры и `sampling_algorithm`.

Порядок операций: **temperature → top-k → top-p → seeded distribution**.
Температура применяется до nucleus cutoff; порядок закреплён аналитическим
CPU fixture. Использован CPU sampler pinned llama.cpp, входные logits
копируются и сохраняются без преобразований. Проверяется конечность logits
и результат масштабирования. Состояние RNG принадлежит одному запросу;
повтор, ошибка или отмена не переносят RNG в следующий запрос.

Seed воспроизводим для проверенной dependency/сборки/платформы. Совпадение
последовательностей при смене компилятора или sampler implementation не
обещается. Penalties, grammar и принудительное завершение reasoning не
добавлены. CUDA math, веса, EOS200020, context budget и кеши не менялись.
Отдельный DFlash driver остаётся greedy-only; sampling туда не подключён.

Кандидат T1/top_p0,95/top_k40 соответствует
[официальному generation config базового MiniMax-M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7/raw/main/generation_config.json).
Это не конфигурация локального finetune. Она проверяется на его весах отдельно;
общий inference default остаётся greedy до выбора рабочего профиля P6.

## Условия проверки

Target: `H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf`,
138342384352 байта; header SHA-256
`9a011b2ad6fb20db3bb0cdfe9af08665939ee1f0cb0609ffe2bfee7f31c825db`.
Полный checksum target не вычислялся. Dependency86ebfef2, CUDA13.0.48,
MSVC19.44.35222.0, driver581.80, architecture120/120a.

Strict F32 activations/KV, ctx2048/batch16, GPU cache18 ГиБ,
arena64/reserve0, file2/chunk4/lookahead/D2D batch. RAM cache, grouping,
FA/graphs/MTP/DFlash выключены, `STRATA_MM27_TOKENWISE=0`.
Глобальный guard95% RAM/VRAM сохранён. GPU проверки выполняются последовательно.
Каждая группа corpus запускается в новом процессе; OS cache и внешняя нагрузка
не контролируются. В группе повторов KV очищается, GPU cache остаётся.

`sampling_ms` измеряет копирование/фильтрацию logits и выбор токенов.
Оно входит в полное `request_ms`, но исключено из forward-only decode rate,
как запись logits и внешние memory samples. Decode учитывает N−1 forward;
первый token входит в prefill. Request включает prefill, но не загрузку модели.
Число generated tokens включает reasoning и EOS.

## CPU fixtures и прежний greedy

48 CPU checks PASS: валидация конфигурации, round-trip F32 параметров,
greedy tie, неконечные logits, seed/reset и аналитическая вероятность
двух оставшихся токенов. Для logits `[2,1,0,-1]`, T0,5/top_p0,95
должны остаться два токена с первой вероятностью `1/(1+exp(-2))`.
Этот fixture проверяет порядок temperature/nucleus независимо от model logits.

Три32-token greedy запроса без `sampling` сравнивались с сохранённым
MM27-16 reference. Все96 IDs и19206144 logits совпали **побитно**;
configuration/GPU-only/bytes/drain/memory gates PASS. Это проверка регрессии,
а не новый benchmark скорости. Семантика выбора первого максимума сохранена.

Два предварительных CPU запуска завершились ошибкой при round-trip T0,01.
В первом нижняя граница сравнивалась в double с округлённым F32 значением;
исправлено сравнение с фактической F32 границей. Во втором EXE остался старым:
локализованный MSVC include output не вызвал пересборку после изменения
header. Добавлен явный `OBJECT_DEPENDS` для `main.cpp` и `check_sampling.cpp`,
оба targets пересобраны. Третий запуск прошёл48 checks; все последующие
model/pipe проверки используют эту сборку. Неудачные EXE/исходники и причины
сохранены в `build-local/minimax-m2-sampling-unit-01` и `-02`.

## Границы результата

Набор проверяет исходный русский список1..96, английское объяснение голубого
неба и китайский расчёт17×23. Проверки ответа и языка узкие; они не заменяют
широкий quality benchmark. Saved-logit CPU replay проверяет sampler,
но не является независимым full-model oracle. Сравнивать sampled logits с
greedy после расхождения token IDs некорректно; численная регрессия проверяется
отдельным greedy запуском и точным повтором одинакового seed.

Исторический [MM27-18 completion gate](MINIMAX_M27_COMPLETION_CHECK.json)
сохраняет FAIL: greedy русский запрос остановился по длине1536 без final.
Новый sampling не исправляет этот greedy результат и не устанавливает причину
старого MM27-06 reload discrepancy. P2 history/tools/API, P4 sessions,
независимый full-model oracle и DFlash live lifecycle остаются открытыми.

## Воспроизведение

Из настроенной CUDA/MSVC среды, каждый output directory должен быть новым:

```powershell
cmake --build build-local/minimax-m2-cuda --target strata-minimax-m2-bench strata-minimax-m2-sampling-check -j 6
build-local/minimax-m2-cuda/bin/strata-minimax-m2-sampling-check.exe build-local/mm27-new-sampling-unit
python -X utf8 tools/check_minimax_m2_sampling.py --model H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --out build-local/mm27-new-sampling
python -X utf8 tools/check_minimax_m2_sampling_pipe.py --source build-local/mm27-new-sampling --out build-local/mm27-new-sampling-pipe
```

Screen сохраняет EXE, исходники, команды, raw text, полные logits, CPU replay
и отдельные functional/completion gates. Pipe checker запускает скрытый Windows
supervisor, посылает CTRL_BREAK только process group своего engine и сохраняет
token events. JSONL не экспортирует logits: его replay ограничен IDs/text.
