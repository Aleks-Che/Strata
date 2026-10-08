# MiniMax-M2.7: локальные DFlash и новый каталог весов

Проверено 2026-10-08 на этом ПК. P5.DF-02: Q3/Q4/Q5 проходят загрузку,
GPU draft blocks, проверку borrowed embedding/head и короткий KV rollback/replay.
P5.DF-03 добавил полный offline greedy driver, корпусное сравнение всех logits
и [измерения полезной скорости](MINIMAX_M27_DFLASH_BENCHMARK.md).
Serving, live EOS/cancel/pressure/recovery ещё не закрыты.
Native MTP в основном GGUF по-прежнему отсутствует.

## Каталог и идентичность target

Актуальный target: `H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf`.
Прежнего файла непосредственно в `H:\models` больше нет.
Размер138342384352 байта и header SHA-256 совпали с прежними отчётами:
`9a011b2ad6fb20db3bb0cdfe9af08665939ee1f0cb0609ffe2bfee7f31c825db`. Основной loader contract и ranges809 tensors
прошли повторную проверку. Полный checksum target не вычислялся, поэтому
совпадение всех128,8 ГиБ payload этим не утверждается.
Исторические JSON, manifests и команды в журнале оставлены с фактическими
путями прежних запусков. Для нового запуска подставлять путь выше.

## Найденные draft

| Файл | Байты | Размер файла, МиБ |
|---|---:|---:|
| MiniMax-M2.7-DFlash-Q3_K_M.gguf | 637482496 | 607.951 |
| MiniMax-M2.7-DFlash-Q4_K_M.gguf | 788502016 | 751.974 |
| MiniMax-M2.7-DFlash-Q5_K_M.gguf | 932615680 | 889.412 |

Все файлы GGUFv3/alignment32, architecture=`dflash`, по58 тензоров,
без routed experts, собственных `token_embd.weight` и `output.weight`.
Полные SHA-256 трёх draft рассчитаны локально:

- `MiniMax-M2.7-DFlash-Q3_K_M.gguf`: `456f6839c82a60bf19ca2ed27a35f8b68966f189dd39ba7e5f76411e4b7a02fd`.
- `MiniMax-M2.7-DFlash-Q4_K_M.gguf`: `f9e0a616bab61739d73b8044fb7baace0f2d4ccb5ca80ec9bbcc602f40a76ce1`.
- `MiniMax-M2.7-DFlash-Q5_K_M.gguf`: `be249cb0993f162560def870a72ae89f7d3b37925dc2f5f46224a975df98839d`.

Общая геометрия:5 плотных блоков, hidden3072, FFN22016,
32 Q/32 KV heads, head dimension128; Q/K norms имеют128 элементов.
`fc.weight` — `[15360,3072]`, то есть проекция пяти target features.
`dflash.target_layers=[2,17,31,45,60]`, `dflash.block_size=8`,
context196608, RoPE base5000000, YaRN factor48/original4096.
`dflash.rope.dimension_count` и `dflash.attention.causal` не сериализованы.
Compiled loader подтвердил RoPE128 и freq_scale≈1/48. Его causal default=true;
probe явно переключает draft context в non-causal режим.
Не переносить flattened Q/K norm основной MiniMax на этот draft.

Типы tensors Q3-варианта:22 F32,21 Q3_K,13 Q4_K,2 Q5_K;
Q4:22 F32,32 Q4_K,4 Q6_K; Q5:22 F32,32 Q5_K,4 Q6_K.
Размеры файлов не включают runtime KV/workspace/feature buffers.

## Связь с target и обязательные проверки

Словарь target содержит200064 entries, draft —200055. Все200055 общих
строк и token types совпадают, все199744 merges равны, BOS/EOS/UNK совпали.
Девять дополнительных target entries — `[PAD200055]`…`[PAD200063]`.
MASK draft —200054, `[PAD200054]`, эта строка есть и у target.
Собственного MASK tensor нет. GPU GET_ROWS для anchor и семи MASK совпал
побитно с независимым декодированием соответствующих Q4_K строк target embedding.

Generic graph умеет брать embedding/head из target через `ctx_other`.
В прочитанной закреплённой копии `src/models/dflash.cpp` head заимствуется целиком,
а `src/llama-context.cpp` считает размер и шаг output по vocabulary draft.
Здесь размеры200064/200055 различаются. P5.DF-02 добавил адаптер:
после общей head projection выделяется префикс200055 и материализуется
непрерывный output. View без `ggml_cont` оставил бы неправильный шаг между
строками. Все строки сверены побитно с захваченными полными logits200064.
Target weights не копируются и не изменяются; исходные GGUF сохранены.

Слои `[2,17,31,45,60]` используются generic adapter как layer inputs;
MiniMax graph уже публикует `t_layer_inp`. В [конфигурации NVIDIA, revision
0a1ee27](https://huggingface.co/nvidia/MiniMax-M2.7-DFlash/blob/0a1ee27e99a5246b27a673c34620ea4ece943351/config.json)
указаны выходы `[1,16,30,44,59]`. Закреплённый `conversion/qwen.py` прибавляет1
при переводе к входу следующего слоя; геометрия локальных sidecars соответствует
этой схеме. Это сверка конфигурации, а не доказательство происхождения локальных
квантов или числовой эквивалентности полному NVIDIA reference.
Acceptance с target `ultra-uncensored-heretic` измерен на трёх коротких запросах
в P5.DF-03; общая оценка качества draft этим не устанавливается.

## Что брать из DeepSeek

Изучены [DSpark](../deepseek-v4-flash-0731/DEEPSEEK4_DSPARK.md),
[engine](../../backends/deepseek4/main.cpp) и закреплённая dependency MiniMax
`86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9` (`models/dflash.cpp`,
`common/speculative.cpp`, `models/minimax-m2.cpp`, `llama-context.cpp`).
Общий DFlash graph уже присутствует в этой dependency; offline MiniMax DFlash
driver добавлен в P5.DF-03. Native MiniMax MTP loader/graph остаётся отдельной задачей.

Переносимая схема: target живёт дольше заимствующего draft; захват hidden features;
отдельные target/draft KV и бюджеты; проверка предложений target;
принятие совпавшего префикса, откат хвоста; EOS/limit/cancel до выдачи клиенту.
DeepSeek DSpark имеет3 MoE-блока, Markov head и block_size5. Локальный MiniMax
DFlash имеет5 dense-блоков и block_size8 без Markov tensors.
DeepSeek geometry, hyper-connections и compressor snapshots сюда не переносятся.
Его опубликованные в репозитории timings не являются замером MiniMax.

## P5.DF-02: выполнено и границы проверки

Добавлены `dflash_contract.hpp`, `dflash_loader_oracle.cpp`, `DFlash.cmake`,
`check_dflash.cpp` и `tools/check_minimax_m2_dflash.py`.
Patch применяется к проверенной SHA-256 копии `dflash.cpp` в build directory;
архив dependency и распакованный исходник не изменяются. Неподдержанные
metadata, shapes, ranges, tokenizer или feature layers отклоняются до загрузки.

На RTX5090/32607 МиБ, driver581.80, RAM125.555 ГиБ, CUDA13.0.48/MSVC19.44:

| Draft | Loader | GPU checks | Peak global VRAM | Peak global RAM |
|---|---|---:|---:|---:|
| Q3_K_M | PASS, 0 allocated weight bytes | 30/30 | 92.835% | 26.438% |
| Q4_K_M | PASS, 0 allocated weight bytes | 30/30 | 93.118% | 26.398% |
| Q5_K_M | PASS, 0 allocated weight bytes | 30/30 | 93.694% | 26.341% |

Дополнительно отклонены усечённый payload, неверный feature layer и изменённый
tokenizer. Полные SHA-256 трёх draft повторно совпали с P5.DF-01.
Отдельная CPU build прошла те же3 loader/3 negative checks и4 CTest version
checks. [CPU report](MINIMAX_M27_DFLASH_CPU_CHECK.json).

Условия: context512, batch/ubatch8, F32 KV, FA/graphs off, target strict F32,
cache cap18 ГиБ, arena64 МиБ/reserve0, file pipeline2 readers/chunk4,
lookahead и D2D batch on. Для Q3_K/Q5_K dense draft действует штатная quant
арифметика dependency; флаг strict quant F32 покрывает Q4_K/Q6_K, а не все типы.
Probe содержит синхронные диагностические readbacks: его время не является
измерением скорости speculative generation.

Проверены depths1/2/4/7, повтор каждого блока после удаления noise KV,
побитное совпадение всех output rows с полным head, borrowed MASK/anchor,
target logits с feature capture on/off, перенос committed features и
восстановление draft history с нуля. GPU audit не обнаружил CPU compute или
полного копирования experts. Target verification batch-vs-serial имеет одинаковые
greedy IDs; max_abs2.575e-5…3.052e-5 проходит прежние max_abs≤5e-4/NMSE≤1e-7.
Это сравнение не побитное; tolerances не изменялись.

Восемь входных токенов — усечённая техническая строка, без chat template.
В единственном verification block принят0/7 у всех квантов; проверенный
committed prefix состоит из anchor. Это не оценка acceptance на реальных запросах
и не основание выбирать квант/depth. Положительные acceptance prefixes,
все reject positions, EOS/limit/cancel, длинный контекст и induced pressure
ещё должны пройти полный driver. Draft остаётся недоступен в bench/API defaults.

Артефакты: [probe report](MINIMAX_M27_DFLASH_PROBE_CHECK.json),
[build manifest](MINIMAX_M27_DFLASH_BUILD_MANIFEST.json).
Полные logits, EXE, source snapshots и stderr сохранены в
`build-local/minimax-m2-dflash-probe-03`; ранние probes01/02 тоже сохранены.
Baseline до правок находится в `build-local/minimax-m2-dflash-baseline-01`.
Регрессии: [runtime49/49](MINIMAX_M27_DFLASH_RUNTIME_CHECK.json) и
[обычная генерация](MINIMAX_M27_DFLASH_REGRESSION_CHECK.json), где все72 output
token IDs и14 404 608 logits побитно совпали с прежним эталоном.

## Следующие шаги P5.DFlash

1. DONE: no-allocation loader всех трёх вариантов, tensor/tokenizer/range contract,
   эффективные RoPE128 и feature layer convention. Точная provenance квантов открыта.
2. DONE для короткого probe: borrowed embedding/head, width200055/200064,
   каждая строка batch logits, MASK, освобождение draft раньше target.
3. PARTIAL: P5.DF-03 реализовал полный offline greedy driver с verification,
   correction/bonus, rollback и committed feature injection. Положительные
   prefixes и full-logit corpus parity проверены на реальных chat prompts;
   GPU audit и sampled memory95 прошли. Live EOS/cancel/pressure/recovery,
   длинный контекст и session lifecycle остаются открыты.
4. DONE для короткого screen: Q3/Q4/Q5, длины1/2/4/7 (семь proposals плюс anchor
   при block_size8), acceptance, draft/verify time и полезные токены/с.
   Исходный batch path провалил full-logit parity из-за routing tie;
   opt-in tokenwise verification дал bit-exact результат без смены tolerances.
5. Повторные A/B и возврат draft-памяти обычному cache описаны в
   [benchmark](MINIMAX_M27_DFLASH_BENCHMARK.md). Serving включать после
   correctness/lifecycle и подтверждённого полезного выигрыша.

DFlash пока выключен. Native MTP требует иных отсутствующих локальных весов;
DFlash теперь имеет локальные веса, поэтому прежний общий блокер «нет draft»
к нему не относится. P2/P4/P6 основной интеграции остаются незавершёнными.

## Воспроизведение

```powershell
$modelDir = 'H:\models\MiniMax-M2.7'
python -X utf8 tools/inspect_minimax_m2_drafts.py --model "$modelDir\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf" --draft "$modelDir\MiniMax-M2.7-DFlash-Q3_K_M.gguf" --draft "$modelDir\MiniMax-M2.7-DFlash-Q4_K_M.gguf" --draft "$modelDir\MiniMax-M2.7-DFlash-Q5_K_M.gguf" --full-hash --output build-local/minimax-m2-dflash-inspection-new.json
```

Output должен быть новым файлом. Target payload не читается; `--full-hash`
читает только три draft целиком. Запуск завершился exit0.
[Полный отчёт](MINIMAX_M27_DFLASH_INSPECTION.json) содержит metadata,
все tensor ranges, типы, tokenizer hashes и различия словарей.

Loader/probe после сборки standalone backend:

```powershell
cmake --build build-local/minimax-m2-cuda --target strata-minimax-m2-dflash_loader strata-minimax-m2-dflash-check -j 6
$modelDir = 'H:\models\MiniMax-M2.7'
python -X utf8 tools/check_minimax_m2_dflash.py --model "$modelDir\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf" --draft "$modelDir\MiniMax-M2.7-DFlash-Q3_K_M.gguf" --draft "$modelDir\MiniMax-M2.7-DFlash-Q4_K_M.gguf" --draft "$modelDir\MiniMax-M2.7-DFlash-Q5_K_M.gguf" --gpu-draft "$modelDir\MiniMax-M2.7-DFlash-Q4_K_M.gguf" --gpu-draft "$modelDir\MiniMax-M2.7-DFlash-Q3_K_M.gguf" --gpu-draft "$modelDir\MiniMax-M2.7-DFlash-Q5_K_M.gguf" --out build-local/minimax-m2-dflash-probe-new
```

Каталог output должен отсутствовать. Без `--gpu-draft` выполняются только
no-allocation и negative checks; для CPU build добавить
`--build build-local/minimax-m2-oracles`. Oracles не читают payload; Python runner
читает draft payload для SHA-256. Target полный checksum не вычисляется.
