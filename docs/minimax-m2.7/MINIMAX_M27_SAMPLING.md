# MM27-18b: sampling и завершение ответов

Измерения: **2026-10-08–09**, Windows, RTX 5090 32 ГиБ, RAM125,555 ГиБ.
[План](MINIMAX_M27_IMPLEMENTATION_PLAN.md),
[статус](MINIMAX_M27_IMPLEMENTATION_STATUS.md).
Ревизия при фиксации: `d342001b4dca70860bcd585f28004591a83e5bf3`,
dirty tree; состав до изменений закреплён отдельным baseline snapshot.

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

## Воспроизводимость запросов

Для T1/top_p0,95/top_k40 исходный русский запрос завершился по EOS при
seed42/7/1234 на398/567/1092 tokens соответственно. Все три ответа закрыли
reasoning и по-русски объяснили правильную сумму4656. Значительно различается
длина reasoning/ответа; три seed не доказывают отсутствие циклов на другом
входе или при другом seed.

В одном дополнительном процессе сначала выполнены32 tokens с seed7,
затем полный запрос с seed42. Короткий префикс совпал с исходным seed7
по32 IDs и6402048 logits; полный повтор seed42 — по398 IDs и79625472
logits. Все сравнения побитные. Частичный32-token запрос намеренно завершён
по длине и не включается в natural-completion gate.

## Полная модель: измерения корпуса

Все семь полных запросов естественно завершились по EOS с закрытым reasoning
и непустым final. Во всех4894 generated tokens, включая один короткий prefix,
979113216 logits конечны; CPU replay заново выбрал все4894 IDs без расхождений.
Это проверка сохранённых logits и sampler, а не независимое вычисление модели.

| Запрос | T / seed | Prompt / output | Decode, токенов/с | TTFT, с | Request, с | Sampling, мс/токен |
|---|---|---:|---:|---:|---:|---:|
| Русский список1..96 | 1 /42 | 343 /398 | 4,451 | 48,870 | 138,611 | 0,838 |
| Тот же список | 1 /7 | 343 /567 | 4,275 | 52,183 | 185,371 | 0,842 |
| Тот же список | 1 /1234 | 343 /1092 | 4,544 | 62,657 | 304,276 | 0,822 |
| Prefix перед повтором | 1 /7 | 343 /32 | 3,703 | 53,584 | 61,994 | 0,742 |
| Полный повтор | 1 /42 | 343 /398 | 4,322 | 36,781 | 129,132 | 0,733 |
| Пониженная температура | 0,7 /42 | 343 /1219 | 4,335 | 53,802 | 336,556 | 0,912 |
| Почему небо голубое, EN | 1 /42 | 52 /879 | 4,268 | 13,094 | 219,898 | 0,734 |
| 17×23, ZH | 1 /42 | 55 /309 | 4,268 | 12,774 | 85,316 | 0,735 |

Для русского запроса T0,7 удлинила ответ относительно T1 при seed42;
пользы от её понижения в этом одиночном сравнении не обнаружено. В английском
запросе T1 тоже дал больше tokens, чем исторический greedy414. Sampling
не гарантирует короткого reasoning. Эти числа **не являются A/B ускорения**:
последовательности токенов и объём работы различаются, timings не повторялись
для выбора быстрого профиля. T1/top_p0,95/top_k40 остаётся opt-in примером.
Глобальные sampled peaks RAM30,07%, VRAM84,66%;95%-guard не сработал.

## JSONL и потоковый текст

14 сценариев в одном загруженном процессе PASS:10 неверных sampling configs,
64-token seed42 запрос после ошибок,16-token запрос с seed7, адресная отмена
после8 токенов и64-token seed42 повтор после неё. Seed42 IDs/text совпали
с сохранённым префиксом полного английского ответа. GPU-only/bytes/drain/
memory gates прошли. Полный sampled EOS проверен в offline corpus;
в pipe этой серии проверялись ограниченные префиксы.

CTRL_BREAK адресован только тестовой process group. Ошибка отмены получена
через2,254 мс по `perf_counter`; это один замер, без гарантии latency для
другой нагрузки. Следующий запрос работает без перезапуска процесса.

Дополнительно48 real-output checks на8 ответах PASS: native token bytes,
EOS по ID, точные reasoning/content segments, token-sized и однобайтовые
UTF-8 chunks через существующий parser. Parser/API integration не менялась.

## Ручная проверка качества

Автоматические corpus gates проверяют EOS, закрытый reasoning, наличие final,
язык, ожидаемое число/ключевое слово и отсутствие длинного повторяющегося
фрагмента. Они не проверяют истинность каждого предложения.

В английском final есть ошибка: для длин волн450 и650 нм модель утверждает,
что первое излучение рассеивается примерно в10 раз сильнее. По приведённому
в том же ответе закону `1/λ⁴` отношение равно `(650/450)**4 ≈ 4,3531`.
Объяснение механизма Rayleigh scattering и natural EOS проходят узкие
автоматические gates, но **ручная проверка точности ответа — FAIL**.
Нельзя объявлять весь quality corpus успешным по наличию слова «scattering».
Причина этой фактической ошибки не установлена; независимый model oracle
в этой серии не запускался. Подбор другого seed с удачным текстом не закрывает
общую проверку качества или выбор defaults P6.

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

## Артефакты и итоговые gates

- [48 CPU checks](MINIMAX_M27_SAMPLING_UNIT_CHECK.json),
  [greedy regression](MINIMAX_M27_SAMPLING_GREEDY_CHECK.json).
- [Screen и полный CPU replay](MINIMAX_M27_SAMPLING_SCREEN_CHECK.json),
  [14 JSONL scenarios](MINIMAX_M27_SAMPLING_PIPE_CHECK.json),
  [48 token-byte/parser checks](MINIMAX_M27_SAMPLING_OUTPUT_CHECK.json).
- [Ручная проверка final:6 PASS /1 FAIL](MINIMAX_M27_SAMPLING_QUALITY_CHECK.json),
  [build manifest](MINIMAX_M27_SAMPLING_BUILD_MANIFEST.json),
  [хэши и сводные gates](MINIMAX_M27_SAMPLING_VALIDATION_CHECK.json).

Bench SHA-256
`07844c0345c12f78ded9f24d7f86154c9f360bfb574e96569e8694489268b0ef`.
Проверены188 файлов исходного снимка
`build-local/minimax-m2-sampling-baseline-01`. Shared common/Step/Hy3/GLM,
MiniMax CUDA math/cache/pipeline и serving sources не менялись. Новый sampler,
checker, main/CMake/manifest и драйверы сохранены вместе с EXE в
`build-local/minimax-m2-sampling-evidence-01`. Raw reports/logits/JSONL/logs
остаются в `build-local/minimax-m2-sampling-*`.

Сводный validation намеренно имеет **`pass=false`**:
`implementation_checks_pass=true`, `natural_completion_gate_pass=true`,
но `manual_answer_quality_pass=false` и `ready_for_api=false`.
Сборка, CPU fixtures, greedy regression, screen, pipe и output replay
завершились с exit0. Screen PASS означает его узкие автоматические gates;
отдельный ручной FAIL сохраняется в итоговом статусе.

Следующий пункт плана — P2.3–P2.6: history/tool parser и API adapter.
Проверка качества и независимого full-model oracle остаётся отдельным
условием для P6; общий default sampler пока greedy.

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
