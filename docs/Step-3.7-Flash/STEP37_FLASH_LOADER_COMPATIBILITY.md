# Step-3.7-Flash: статический контракт loader

Проверено 2026-10-05 на локальном `H:\models\Step-3.7-Flash\UD-Q4_K_S`.
Все **754 тензора** прошли проверку имён и размеров. Это проверка заголовков
основной текстовой модели; она не проверяет загрузку и численную точность полной
модели. Отдельные synthetic CUDA kernels позднее проверены в P0.4, см. конец документа.

Кандидат — Unsloth llama.cpp
`86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9`;
SHA-256 `src/models/step35.cpp`:
`4c42074b6f859b5734572eb0b1f1f07c8b783e20bfeb3a357871c77a8255c8b3`.
Изолированная сборка использует исходный архив без GLM patches.

Полная таблица по каждому тензору, включая shard, file offset, размер и тип,
находится в [GGUF inventory](STEP37_FLASH_GGUF_INVENTORY.json):
`loader_contract.tensor_mapping` и `tensor_details` соединяются по `name`.
Исполняемая проверка: [step35_loader_contract.py](../../tools/step35_loader_contract.py).

| Семейство | Количество | Размеры GGUF (быстрая ось первая) | Типы локального набора |
|---|---:|---|---|
| Embedding | 1 | `[4096,128896]` | Q8_0 |
| Output | 1 | `[4096,128896]` | Q6_K |
| Output/attention/FFN norms | 91 | `[4096]` | F32 |
| Q/K norms | 90 | `[128]` | F32 |
| RoPE factors | 1 | `[64]` | F32 |
| Q/K/V/output attention | 180 | Q `[4096,128×heads]`, K/V `[4096,1024]`, O `[128×heads,4096]` | Q8_0 |
| Head gate | 45 | `[4096,heads]` | Q8_0 |
| Dense gate/up/down | 9 | gate/up `[4096,11264]`, down `[11264,4096]` | Q8_0 |
| Router и selection bias | 84 | `[4096,288]`, `[288]` | F32 |
| Routed gate/up/down | 126 | gate/up `[4096,1280,288]`, down `[1280,4096,288]` | Q4_K |
| Shared gate/up/down | 126 | gate/up `[4096,1280]`, down `[1280,4096]` | Q8_0 |

Контракт выбранного профиля требует Q/K norms, head gate, router bias, shared
expert и rope factors. Upstream допускает отсутствие части этих тензоров для
других вариантов Step; это не повод молча принимать неполный локальный набор.
При неизвестных, пропущенных или несовпадающих тензорах inspector завершает
работу с ошибкой. Сокращённые завершающие единичные измерения допустимы, как в ggml.

## Attention и MoE

- Основные блоки 0..44; dense 0..2, MoE 3..44. Нет NextN metadata или tensors.
  Три MTP-блока из внешнего config не вычитаются из локальных 45 блоков.
- Full attention: 12 блоков 0,4,…,44, 64 query heads, 8 KV heads.
  SWA: 33 блока, 96 query heads, 8 KV heads, окно 512.
- K/V head dimension 128. Loader уменьшает rotary dimension full attention
  вдвое: 64 против 128 у SWA. Bases: 5 000 000 и 10 000.
  Graph передаёт `rope_freqs` только full attention; SWA получает `nullptr`.
  Значения самого `rope_freqs.weight` ещё не считаны и не проверены численно.
- Head gate: sigmoid от отдельной проекции нормализованного attention input;
  умножение результата attention по головам перед output projection.
- Router: sigmoid, selection bias, 8 выбранных из 288 experts,
  нормализация выбранных весов, scale 3. Shared expert прибавляется отдельно.
- Clamps: routed 7 и shared 16 на блоках 43/44, нулевые на остальных.
  Проверены metadata и места использования в `llama-graph.cpp`;
  отдельный numerical test применения clamp ещё нужен.

Sources in the pinned candidate:
[loader/graph](https://github.com/unslothai/llama.cpp/blob/86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9/src/models/step35.cpp),
[shared FFN graph](https://github.com/unslothai/llama.cpp/blob/86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9/src/llama-graph.cpp),
[common hparams](https://github.com/unslothai/llama.cpp/blob/86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9/src/llama-model.cpp).
Проверка сделана по локальной копии этих исходников.

## Границы проверки

Inspector проверяет quant block geometry и границы файлов, но наличие известного
формата не означает работоспособность соответствующего CUDA kernel. Полный
checksum 106,323 ГиБ весов не вычислялся. Structural fingerprint:
`37305e11f45387f044806d639588fe6ef9d4b10eeaad1dc8ca0f02f9cdae6502`.
Он покрывает упорядоченные header hashes и длины файлов, а не payload.

P0.4 выполнен в STEP-02: Q4_K/Q8_0/Q6_K/F32 matmul/routed matmul, top-8,
batches 1/4/17 и padded strides на RTX 5090 — 78/78 synthetic cases PASS.
[CUDA criteria, fixes and reports](STEP37_FLASH_CUDA_VALIDATION.md).
P0.5 ещё требует Step graph, независимого template rendering, full/SWA state
и rollback tests. Это не отменяет ограничений статического loader contract.
