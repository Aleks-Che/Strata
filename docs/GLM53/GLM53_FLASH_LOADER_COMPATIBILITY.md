# GLM-5.3-Flash: сопоставление GGUF с loader

Обновление 2026-10-04 на Strata `444f442`: основной тестовый файл теперь
`H:\GLM-5.3-Flash-GGUF\GLM-5.3-Flash-Uncensored-IQ3_XXS.gguf`.
Он также прошёл инспектор и тот же статический loader contract: 1 файл,
1412 тензоров, 45 основных блоков + 1 MTP, без несовпадений имён/форм.
[Новый отчёт](GLM53_FLASH_IQ3_XXS_INSPECTION.json).

```powershell
.venv/Scripts/python.exe tools/setup_glm5next.py --model-dir H:/GLM-5.3-Flash-GGUF --check-only --loader-contract --output docs/GLM53/GLM53_FLASH_IQ3_XXS_INSPECTION.json
```

Exit 0. Команда выбирает единственный GGUF в корне этой директории, не заходя
в поддиректорию UD-Q3_K_XL. При добавлении других GGUF выбор нужно уточнить;
не подменять модель молча. Типы весов отличаются: основным routed matrices нужны
IQ2_S/IQ3_S/IQ4_XS, MTP — Q2_K/Q3_K; есть BF16 среди остальных весов.
Эта проверка не доказывает поддержку этих операций на CUDA.

Ниже сохранён исходный отчёт UD-Q3_K_XL. Его ссылки на JSON и команды относятся
к прежнему профилю и не заменены результатами нового файла.

Проверено 2026-10-04. Пункт P0.2: статическое сопоставление локального
`UD-Q3_K_XL` с Unsloth llama.cpp
`86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9`.
Это кандидат для сборки P0.3, а не подтверждённая production-зависимость.
Сборка, CUDA kernels, logits и inference в эту проверку не входили.

Команда из корня Strata:

```powershell
python tools/setup_glm5next.py --model-dir H:/GLM-5.3-Flash-GGUF/UD-Q3_K_XL --check-only --loader-contract --output docs/GLM53/GLM53_FLASH_LOADER_CHECK.json
```

Результат: exit 0, все 1412 тензоров сопоставлены без пропущенных и неизвестных
имён и без несовпадений форм. [JSON отчёта](GLM53_FLASH_LOADER_CHECK.json) содержит
метаданные, списки блоков, типы и количество тензоров по семействам. Для вывода
каждого имени, shape, quant type и диапазона файла добавить `--tensor-details`.

Проверка реализована в `tools/glm5next_loader_contract.py`; `--loader-contract`
включает её после обычной проверки shards, геометрии quant rows и диапазонов.
Формы вычисляются из metadata. Значения 46/4096/288 не зашиты в формулы.

## Основание проверки

Имена и формы сверены с `load_arch_tensors`, metadata — с `load_arch_hparams`
в [glm5next.cpp выбранной ревизии](https://github.com/unslothai/llama.cpp/blob/86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9/src/models/glm5next.cpp).
Имена GGUF подтверждены таблицей
[llama-arch.cpp](https://github.com/unslothai/llama.cpp/blob/86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9/src/llama-arch.cpp).
Раздельные Q/K/V соответствуют ветви `create_tensor_qkv` в
[llama-model.cpp](https://github.com/unslothai/llama.cpp/blob/86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9/src/llama-model.cpp).

Профиль проверки ограничен полным GLM Flash с раздельными Q/K/V и скалярными
размерами FFN/head_count. Trunk-only, MTP-only и fused QKV требуют отдельного
профиля; неизвестные тензоры отклоняются, а не пропускаются.

Концевые единичные измерения дополняются до четырёх, как в
[`check_tensor_dims`](https://github.com/unslothai/llama.cpp/blob/86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9/src/llama-model-loader.cpp).
Поэтому локальный KDA conv `[4, 1, 8192]` соответствует ожидаемому
`[4, 1, 8192, 1]`. Переставлять оси или убирать внутреннюю единицу нельзя.

## Сопоставленные семейства

Числа ниже получены новым инспектором из четырёх локальных заголовков.
В коде проверки перечислены все шаблоны имён и формы каждого семейства.

| Семейство | Основная модель | MTP | Проверяемые данные |
|---|---:|---:|---|
| Embedding/output | 3 | 0 | Ширина и словарь; output допускает tied embedding в контракте loader |
| Attention/FFN norms | 90 | 2 | Ширина embedding |
| mHC | 270 | 0 | Два подблока, четыре потока, mixer 24, scale 3 |
| KDA | 510 | 0 | Q/K/V, conv, gates, beta, A, dt, norm, output |
| DSA/MLA | 88 | 8 | Low-rank projections, norms, трёхмерные K/V, output |
| Indexer | 77 | 7 | Key/query/proj, LayerNorm weight и bias, compressor gate и APE |
| Dense FFN | 9 | 0 | Gate/up/down первых трёх блоков |
| Router | 84 | 2 | Gate matrix и correction bias |
| Routed experts | 126 | 3 | Gate/up/down с осью 288 экспертов |
| Shared experts | 126 | 3 | Gate/up/down с отдельной шириной shared FFN |
| NextN | 0 | 4 | Projection и три нормы |
| Всего | 1383 | 29 | 1412 |

У модели отсутствуют `blk.45.nextn.embed_tokens.weight` и
`blk.45.nextn.shared_head_head.weight`: loader объявляет их необязательными.
Проверка допускает их отсутствие и проверяет форму, если они присутствуют.
Требовать их добавления в исходный GGUF не нужно.

## Граница основной модели и MTP

Metadata задаёт `block_count=46`, `nextn_predict_layers=1`. Полученное разделение:
основные блоки 0–44 (34 KDA, 11 DSA), MTP — блок 45. В проверенном исходнике
основной `graph` ограничивает цикл через `n_layer`, а загрузчик при выключенном
`load_mtp` помечает MTP-тензоры как пропускаемые. Отдельная ветвь строит MTP-граф.
Источник — [glm5next.cpp](https://github.com/unslothai/llama.cpp/blob/86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9/src/models/glm5next.cpp).

Это проверка разделения по исходному коду и заголовкам. Фактический trace
собранного backend при MTP off необходимо проверить в P0.3/P1; наличие native
MTP в upstream не означает его работающую интеграцию в Strata.

## Проверки и продолжение

```powershell
.venv/Scripts/python.exe -m unittest tools.test_glm5next_loader_contract tools.test_setup_glm5next tools.test_deepseek4 tools.test_shards
```

Exit 0, 37 тестов. Новые девять тестов используют сохранённые реальные заголовки
глобальных тензоров и блоков 0/3/45, перенумерованных в 0/1/2; fixture не создаётся
из формул проверяемого кода. Проверяются пропуски во всех семействах, несовместимые
имена, перестановка осей, концевые единицы, optional weights, metadata, граница MTP.
Файл `tools/fixtures/glm5next_loader_headers.json` не содержит весов.

Следующий шаг — изолированная сборка выбранной ревизии, хеш архива и тесты графов.
Quant types здесь подсчитаны и проверены на уровне GGUF-геометрии; поддержку
конкретных операций на CUDA и численную точность нужно подтвердить отдельно.
До такой проверки не менять pin DeepSeek и не включать GLM в рабочий launcher.
