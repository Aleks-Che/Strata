# Hy3 admission and oracles

This directory builds CPU vocabulary/template/registration oracles and CUDA
numerical checks for GGUF `hy_v3`. The CUDA checks execute small synthetic main
and MTP graphs. The opt-in Windows pipe engine executes the full checkpoint
without MTP. Selected expert transfers support a synchronous reference and
an optional bounded GPU matrix cache and file/H2D pipeline. An isolated profile
serves both HTTP APIs and the web app below.

Use a Visual Studio 2022 developer shell (MSVC and Windows SDK in PATH/INCLUDE/LIB):

```powershell
cmake -S backends/hy3 -B build-local/hy3-oracles -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_HY3_ARCHIVE=C:/work/git/my-repos/Strata/build-local/llama-glm-86ebfef2.tar.gz
cmake --build build-local/hy3-oracles --target strata-hy3-tokenizer strata-hy3-template strata-hy3-loader -j 4
ctest --test-dir build-local/hy3-oracles --output-on-failure
python -m unittest tools.test_hy3_gguf tools.test_hy3_tokenizer tools.test_step35_tokenizer tools.test_glm5next_tokenizer
```

The build checks the local archive and original loader hashes, extracts into
its private `_deps` and writes `hy3-build-manifest.json`. There are no network
downloads. Existing Qwen, DeepSeek, GLM and Step builds are not changed.

`STRATA_HY3_MTP_LOAD_FLAGS=ON` is the default. The original loader omits some
MTP tensors with `load_mtp=false`, but still registers nine optional tensors
(1,740,784,384 logical bytes for the local model). The generated Hy3 source
propagates the skip flag to optional FFN/NextN tensors as well. It leaves the
archive and extracted source untouched. OFF is only for reproducing the
upstream registration result; the loader checker requires the corrected build.

From the repository root, with Python `regex` and `jinja2` available:

```powershell
python tools/inspect_hy3_gguf.py --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --output docs/hy3/HY3_INSPECTION.json
python tools/check_hy3_loader.py --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --oracle build-local/hy3-oracles/bin/strata-hy3-loader.exe --output docs/hy3/HY3_LOADER_VALIDATION.json
python tools/check_hy3_tokenizer.py --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --oracle build-local/hy3-oracles/bin/strata-hy3-tokenizer.exe --output docs/hy3/HY3_TOKENIZER_VALIDATION.json
python tools/check_hy3_template.py --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --oracle build-local/hy3-oracles/bin/strata-hy3-template.exe --tokenizer-oracle build-local/hy3-oracles/bin/strata-hy3-tokenizer.exe --output docs/hy3/HY3_TEMPLATE_VALIDATION.json
```

The inspector admits the reviewed one-file, dense0, separate-QKV, MoE trunk
plus one embedded MTP layout. Unknown Hy3 architecture metadata, missing or
extra tensors, unsupported types, malformed rows/ranges and split files fail
closed. Small dimensions are allowed for CPU fixtures. Header hashes do not
verify weight contents. Loader checks use `no_alloc=true`, no mmap/lazy mode,
CPU placement and zero-sized dummy weight buffers; they do not read weights
or execute a graph. MTP off/on must preserve every main tensor and omit/load
all MTP tensors respectively.

The local tokenizer has no standalone CR byte token. Its native oracle drops
unmerged `\r`; Python matches this after BPE only for `hunyuan-dense`.
The report distinguishes this from lossless decoding. EOS/EOG is 120025;
120026 is a placeholder, not a stop token. No BOS is added by the oracle:
the embedded template owns it. Template tests cover the unchanged Jinja source,
`no_think`/`low`/`high`, histories, tools, results and malformed options.
OpenAI JSON-string arguments are normalized to objects before rendering.

## CUDA admission

From the same developer shell, using the local CUDA 13 toolkit and RTX 5090:

```powershell
cmake -S backends/hy3 -B build-local/hy3-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_HY3_ARCHIVE=C:/work/git/my-repos/Strata/build-local/llama-glm-86ebfef2.tar.gz -DSTRATA_HY3_CUDA=ON -DCMAKE_CUDA_COMPILER=C:/work/git/my-repos/Strata/build-local/cuda-13.0/bin/nvcc.exe -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_HY3_STRICT_F32=ON -DSTRATA_HY3_ROUTED_STRIDES=ON
cmake --build build-local/hy3-cuda --target strata-hy3-kernels-check strata-hy3-graph-check -j 4
python tools/check_hy3_cuda.py --build build-local/hy3-cuda --kind kernels --cuda-bin build-local/cuda-13.0/bin/x64 --output-dir build-local/hy3-tests/kernels-new
python tools/check_hy3_cuda.py --build build-local/hy3-cuda --kind graph --cuda-bin build-local/cuda-13.0/bin/x64 --output-dir build-local/hy3-tests/graph-new
```

Python requires `psutil`. Output directories must be new. The wrapper sets
`NVIDIA_TF32_OVERRIDE=0`, records binary/manifest hashes, and samples global RAM
and GPU0 VRAM. At 95% usage it stops only its child check. Sampled memory includes
other processes; it is not a continuous allocation guarantee or model memory
estimate. GPU logs go to the new directory. No full checkpoint is opened.

On this PC, kernels passed **150/150** and graph/state checks **86/86**.
Kernel inputs cover F32, Q8_0, Q6_K, Q5_K, Q4_K, Q3_K, IQ3_XXS and IQ4_XS,
192 experts/top8, batch1/4/17, broadcast/per-route input and padded row strides.
Packed weights are dequantized for a scalar double-accumulating reference;
explicit CPU and direct CUDA execution are also compared. Kernel tolerances:
F32 abs <=5e-5/NMSE <=1e-10; quants abs <=0.02/NMSE <=1e-4;
router abs <=2e-6/NMSE <=1e-10 with exact selected IDs. Padded vs compact
CUDA output must be bit-exact; guarded input bytes must remain unchanged.

Graph fixtures use two main blocks (dense then MoE) plus one MTP block, width256,
four Q heads/one KV head, head_dim128, 16 experts/top8 and vocab64. One fixture
uses F32; the other uses Q8_0/Q6_K, main IQ3_XXS/IQ4_XS and MTP Q3_K/Q4_K.
The checks cover CPU/CUDA logits and hidden states, serial/prefill/microbatch4,
scalar Q/K RMSNorm before NeoX RoPE, causal full GQA, dense/shared/routed SwiGLU,
MTP enorm/hnorm/concat/projection and post-final-norm hidden states. Main and
MTP use separate full F32 KV caches; save/restore and rollback/replay are
bit-exact. Loading/skipping MTP leaves main logits and hidden states identical.
The scheduler audit rejects CPU tensor math in every GPU run.

F32 graph comparisons require abs <=5e-4 and NMSE <=1e-7; scalar component
limits are tighter and recorded per check. Mixed-quant CPU/CUDA and batch-path
comparisons allow abs <=0.03 for logits, <=0.12 for hidden and NMSE <=2e-3,
because these paths quantize activations differently. This is bounded numerical
agreement, not exact quantized parity. Restore/replay and MTP-off/on comparisons
require exact bytes even with mixed weights. These synthetic checks do not
cover full-model inference; the Windows baseline below adds that comparison.
Flash Attention, F16 KV and speculative accept/reject integration remain
for later phases.

Both CUDA patches default ON in a fresh CUDA configuration, and are scoped to
generated sources under this build directory with original-source hash guards:

- `STRATA_HY3_STRICT_F32`: custom F32 MMF dispatch must respect the TF32 override.
  Without it, 14/150 tests failed even with that environment variable set.
- `STRATA_HY3_ROUTED_STRIDES`: gathered input rows must use their physical stride.
  After the F32 fix alone, four padded routed tests failed; with both, all pass.

For reproduction, configure both OFF, build/run kernels, then enable only
STRICT_F32 and repeat, then enable both. Use a new report directory each time.
The three measured reports and final graph report are linked from the status.

## Windows synchronous baseline

Keep the CUDA options above and enable `STRATA_HY3_RUNTIME=ON`:

```powershell
cmake -S backends/hy3 -B build-local/hy3-cuda -DSTRATA_HY3_RUNTIME=ON
cmake --build build-local/hy3-cuda --target strata-hy3 strata-hy3-runtime-check -j 4
python tools/check_hy3_cuda.py --build build-local/hy3-cuda --kind runtime --cuda-bin build-local/cuda-13.0/bin/x64 --output-dir build-local/hy3-tests/runtime-new
$env:PATH=(Resolve-Path build-local/cuda-13.0/bin/x64).Path+';'+$env:PATH
python tools/check_hy3_engine.py --engine build-local/hy3-cuda/bin/strata-hy3.exe --fixture build-local/hy3-tests/runtime-new/fixture/hy3-mixed.gguf --output-dir build-local/hy3-tests/pipe-new
python tools/check_hy3_model.py --engine build-local/hy3-cuda/bin/strata-hy3.exe --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --cuda-bin build-local/cuda-13.0/bin/x64 --output-dir build-local/hy3-tests/model-new --predict 24
```

The model checker uses the reviewed embedded `no_think` template. It compares
Russian, English, Chinese, code, arithmetic and a longer prompt, then A/B/A and
STOP/recovery. Binary hashes, prompt IDs, raw logit hashes, H2D byte counts,
TTFT, decode-forward time and memory samples go to `model-report.json`.
Raw logits remain in the run directory. `--smoke` selects only the short Russian
request. Sequential timings include changing OS file-cache residency and must
not be interpreted as a controlled speed comparison between transports.

Direct pipe invocation (stdin/stdout only, no network listener):

```powershell
build-local/hy3-cuda/bin/strata-hy3.exe --native H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --inspect-only
build-local/hy3-cuda/bin/strata-hy3.exe --native H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --serve --max-context 2048 --batch-size 17 --copy-mode pinned --kv f32
```

`ENC <parse_special:0|1> <UTF8 hex>` returns `IDS ...`. `GEN <count>
[sampling key=value ...] <comma-separated IDs>` emits `PP`, `T`, then `DONE`.
`STOP` cancels the current request, `QUIT` exits. Each request starts with fresh
KV and sampler state. Session keys are accepted for isolation, not persistence.
Greedy and seeded temperature/top-k/top-p/min-p/repetition sampling use native
samplers. The capability line reports the requested expert-cache cap and
pipeline readers; MTP and conversation persistence remain disabled.
Context is limited to 2048, batch to 32 and KV to F32 for this baseline.

The native header check validates the local 80+1 shape, all 1298 tensors, allowed
types, required dense/shared/router/NextN weights and non-overlapping in-file
ranges before loading weights. `--inspect-only` allocates no weight payload.
`--expert-cache-mib 0..16384` enables a bounded GPU matrix cache with pinned copy
mode; zero keeps the uncached reference path. It is a cap, not a reservation:
the controller clamps against global VRAM usage, including other applications,
and leaves 5% plus 256 MiB. It trims entries under pressure before the 95% guard.
RAM still uses the 93% process working-set target and 95% global admission guard.

The allocation/eviction controller is reused unchanged from
`backends/step35/expert_cache.hpp`. Hy3 owns a separate registry and CUDA stream.
Keys identify a live model generation, registered tensor (name/layer, type,
shape, strides and actual file mapping/offset), and expert. Beginning a new
generation drops all old entries. Cache hits copy bytes from GPU to GPU into
the native scratch; cache misses use the synchronous file/pinned path. Every
copy is fenced before admission, eviction or reuse. Entries include the actual
native next-expert padding of up to 512 bytes. A failed cache fill invalidates
entries; allocation OOM bypasses caching after the successful scratch upload.
No cache persists across model unload. The synchronous mode remains the default.

`--pipeline-readers 1` opts into the unchanged shared `StrataExpertPipeline`.
It requires pinned copy and a nonzero expert-cache cap. Reader counts 1 and 2
are admitted; `--pipeline-chunk-mib 4|8|16` defaults to 4. The shared ring has
four slots, so the default allocates 16 MiB pinned RAM plus 16 MiB GPU storage,
independently of model size. It replaces the synchronous 16 MiB host buffer.
RAM/VRAM admission precedes allocation and retains the global 95% ceiling.

After the native router returns IDs, a bounded plan follows the actual future
gate/up/down scheduler split order for that routing tensor. Only misses are
read through the registered native file handle; hits are pinned against cache
eviction until the plan drains. Independent H2D stages into ring buffers, never
directly into scheduler scratch. Each consumer waits on a recorded ready event;
slot reuse waits on its consumer event. The default delivery fences each
matrix and cache fill. Read/H2D lookahead can continue during preceding compute.

`--pipeline-batch 1` queues all selected matrices of one scheduler input tensor,
then fences the consumer stream once. It requires pipeline readers; default 0
keeps the per-matrix reference. The preceding scratch-consumer fence remains.
Cache hits stay plan-pinned; newly admitted cache destinations get local pins
until all queued D2D copies and fills complete on that same stream. Allocation
pressure cannot reuse these pending entries. Error unwinding drains the stream
before releasing pins and invalidating incomplete fills. STOP is checked between
matrix submissions and before the final fence. The shared ring is unchanged.

Every graph exit finishes the plan, including cancellation and exceptions.
Mappings remain owned until readers finish. Reader/submission errors drain the
ring and trigger recreation on the next request; incomplete cache fills are
invalidated. No source handle is changed concurrently by the failure tests.

Request metrics distinguish source/H2D bytes, D2D hit bytes, cache-fill D2D
bytes, resident/budget bytes, hits/misses, evictions/reuses and allocation OOM.
Source bytes count native file reads, including OS-cached reads; they are not
a measurement of physical SSD traffic. Cache fill and hit bytes are separate.
Pipeline counters add groups/chunks, abandoned bytes, fixed device-ring size,
reader concurrency, outstanding reader/queue bytes, consumer/slot wait time and
submission time. CPU wait sums are not GPU execution or proof of overlap.
After drain, reader-owned and queued bytes must both be zero.
`h2d_ms` times the synchronous upload path only; pipeline H2D execution time
comes from the optional CUDA trace, not that field or CPU submission time.
Batch metrics add `pipeline_copy_batches`, `pipeline_pending_fills_peak` (entry
count), `pipeline_batch_ms` (inclusive CPU wall time), and prefill/decode delivery
fence counts. The fence counter excludes preceding scratch and graph-end fences;
it does not count implicit CUDA allocator synchronization.

For a separate diagnostic run, `--trace-graphs 4 --trace-file NEW_FILE.json`
records CUDA event intervals on the actual H2D and compute streams. The file
is written at clean engine shutdown. Timing runs must leave tracing disabled.
These intervals include stream scheduling gaps; they do not measure kernel
occupancy. Tracing reuses Step's audited `GpuTrace` unchanged.

`--fixture` explicitly admits the tiny tokenizer-free fixture instead; it cannot
bypass validation of a full checkpoint. Python admission additionally validates
the full tokenizer/template contract and records header hashes. Neither hashes
nor header admission verify all weight payload bytes.

All non-routed main weights stay on GPU. Routed weights have demand mappings;
MTP tensors are skipped. The unchanged scheduler's `--copy-mode native` is the
reference. `pinned` reads only its selected expert ranges through the shared
`backends/common/expert_file.hpp`, stages at most 16 MiB and fences each H2D
before reusing staging. Without pipeline readers, one native read is in flight.
The cache and pipeline are opt-in. The scheduler rejects CPU tensor math and accidental
whole expert-tensor copies before submitting them. Selected uploads include
the candidate's <=512-byte MMQ padding; counters include these real bytes.

Mapped resident pages are bounded using GLM's existing `HostWorkingSetBudget`
helper. It caps this process based on a 93% system target after charging other
processes; limits are restored on release. The remaining margin precedes the
95% global RAM/VRAM guard. VRAM uses the shared driver-global NVML probe matched
by PCI identity, not CUDA's process view on WDDM. A separate test monitor can
stop only its child. These controls do not fill unused memory artificially.

Runtime sets `NVIDIA_TF32_OVERRIDE=0`, `GGML_OP_OFFLOAD_MIN_BATCH=1` and
`GGML_CUDA_DISABLE_FUSION=1` before backend initialization. With candidate
fusions enabled, the F32 batch4 fixture's resident/reference logits differed
by up to 0.00105551, while pinned/reference matched. Disabling fusions restored
exact equality in every runtime fixture. The offending fusion is not isolated
yet; re-enabling it requires separate numerical admission. The earlier P0
graph checker uses evaluation callbacks and is not a substitute for this
uncaptured scheduler test.

The runtime suite checks exact copied GPU bytes and logits for F32/mixed,
batch1/4/17, a transfer exceeding 16 MiB, cancellation/recovery, CPU rejection
and malformed input. HY3-04 extends it to 32 cases: injected pinned allocation
failure, a real invalid-handle ReadFile failure after one completed upload,
partial-copy cancellation, simulated RAM/VRAM limits, impossible GPU reserve,
unload and reload. Every recovery must reproduce the original logits exactly.
Test controls have no CLI/environment entry point; availability caps can only
lower real memory availability, and do not allocate pressure buffers.

These checks exposed an exception-boundary bug under MSVC `/EHsc`: throwing
memory admission helpers had C linkage, which permits the compiler to assume
they cannot throw. The RAM refusal fixture crashed with exit `0xc0000374`.
Runtime helpers now have C++ linkage; the same fixture reports the expected
refusal and recovers. Real driver OOM/device-loss recovery and external memory
pressure are not established by these synthetic failures.

Full-model speed tuning, long context, Flash Attention,
F16 KV, MTP and session persistence remain separate stages.

Hy3's separate template/parser module lives in `serve/hy3.py`. Its reviewed
fixture and CPU tests are described in `serve/fixtures/README.md`. A full-model
tool/result check uses a fixed local stub, with no external tool execution:

```powershell
python tools/check_hy3_dialogue.py --engine build-local/hy3-cuda/bin/strata-hy3.exe --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --cuda-bin build-local/cuda-13.0/bin/x64 --template-oracle build-local/hy3-oracles/bin/strata-hy3-template.exe --tokenizer-oracle build-local/hy3-oracles/bin/strata-hy3-tokenizer.exe --output-dir build-local/hy3-tests/dialogue-new
```

## Experimental HTTP profile

Export the reviewed tokenizer/template and an explicit profile into a **new**
directory. This leaves the existing profiles unchanged. The exporter checks the
model header, engine source/patch identity, template and special token IDs.

```powershell
python tools/prepare_hy3_profile.py --model H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --engine build-local/hy3-cuda/bin/strata-hy3.exe --output-dir build-local/hy3-http-new --cuda-dir build-local/cuda-13.0
python -m serve.server --engine strata --config build-local/hy3-http-new/hy3.json --port 8094
```

The server binds to `127.0.0.1`. It serves the web app, `/v1/models`, OpenAI
`/v1/chat/completions` and Anthropic `/v1/messages`, including JSON and SSE.
The profile uses the synchronous pinned baseline, context 2048, batch 17, F32 KV
and temperature 0; MTP and persistent sessions remain off. The exporter
accepts `--expert-cache-mib` with default 0; a nonzero value adds the runtime cap
to the new profile. Existing profiles are not overwritten.
`--pipeline-readers 1 --pipeline-chunk-mib 4` also exports an explicit pipeline
configuration when a nonzero cache cap is supplied.
The exporter flag `--pipeline-batch` adds native `--pipeline-batch 1` to a new
profile; it requires nonzero readers and does not alter existing profiles.
`fit_max_tokens` reduces the requested output cap to the available context space;
it never truncates the prompt. This profile is for validation, not a speed default.

Hy3 defaults to `no_think`. OpenAI accepts `reasoning_effort` (or
`reasoning.effort`); Anthropic accepts `output_config.effort`, or
`thinking.type=disabled/enabled/adaptive` (no_think/high/high). Explicit
`chat_template_kwargs.reasoning_effort` takes priority; contradictory Anthropic
thinking controls are rejected. Both APIs accept `preserved_thinking` directly
or in `chat_template_kwargs`. Efforts are exactly `no_think`, `low`, `high`.
Hard token reasoning budgets, `enable_thinking`, `clear_thinking` and media
blocks are unsupported. EOS is resolved and checked from the exported tokenizer;
only120025 ends generation for this model, not PAD or placeholder120026.

The real HTTP checker requires the same binary as the passing direct-dialogue
reference, or explicit `--allow-engine-change` to compare a new checksummed
engine against the historical IDs. It compares prompts/output IDs, runs a local
tool/result round-trip, checks prefill disconnect/recovery, and samples global
RAM/VRAM under a 95% guard:

```powershell
python tools/check_hy3_http.py --profile build-local/hy3-http-new/hy3.json --reference docs/hy3/HY3_TOOL_DIALOGUE_VALIDATION.json --output-dir build-local/hy3-tests/http-new --allow-engine-change
python -m unittest serve.test_hy3 serve.test_hy3_http tools.test_hy3_profile
node serve/test_hy3_settings_ui.cjs
```

Cache admission and the full-model capacity sweep (run sequentially, one GPU
test at a time):

```powershell
cmake --build build-local/hy3-cuda --target strata-hy3 strata-hy3-runtime-check strata-hy3-cache-check -j 4
python tools/check_hy3_cuda.py --build build-local/hy3-cuda --kind cache --cuda-bin build-local/cuda-13.0/bin/x64 --output-dir build-local/hy3-tests/cache-unit-new
python tools/check_hy3_cuda.py --build build-local/hy3-cuda --kind runtime --cuda-bin build-local/cuda-13.0/bin/x64 --output-dir build-local/hy3-tests/cache-runtime-new
python tools/check_hy3_cache_model.py --engine build-local/hy3-cuda/bin/strata-hy3.exe --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --reference docs/hy3/HY3_MODEL_VALIDATION.json --cuda-bin build-local/cuda-13.0/bin/x64 --output-dir build-local/hy3-tests/cache-model-new
python tools/check_hy3_cache_model.py --engine build-local/hy3-cuda/bin/strata-hy3.exe --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --reference docs/hy3/HY3_MODEL_VALIDATION.json --cuda-bin build-local/cuda-13.0/bin/x64 --cache-mib 8192 --pipeline-readers 0 1 2 0 --output-dir build-local/hy3-tests/pipeline-model-new
python tools/check_hy3_cache_model.py --engine build-local/hy3-cuda/bin/strata-hy3.exe --gguf H:/models/hy3/Hy3-Q3_K_M-mtp.gguf --reference docs/hy3/HY3_MODEL_VALIDATION.json --cuda-bin build-local/cuda-13.0/bin/x64 --cache-mib 8192 --pipeline-readers 2 --pipeline-batch 0 1 1 0 --output-dir build-local/hy3-tests/batch-model-new
```

The sweep defaults to 0/8192/12288/16384/0 MiB, two English/code passes per
process with fresh KV and deterministic sampling. Cache survives requests;
the OS file cache is not purged. Every output logit hash and token sequence
must equal the recorded native baseline. Each nonzero capacity also checks
STOP and a following request. See the status for measured results and limits.
`--prompts long --repeats 1` selects the recorded 362-token prompt for a separate
prefill comparison. Reference hashes and stop/recovery checks still apply.

The first HY3-08 runtime run had one intermittent resident/native mismatch at
mixed batch 4, with the pipeline disabled. Subsequent full runs and 32 focused
resident/native comparisons passed. The cause remains unresolved; the original
failure is retained in the status, and batching remains an explicit experiment.
`strata-hy3-runtime-check NEW_DIRECTORY --reference-probe` runs the focused
32 comparisons plus 28 native repeat checks; failed comparisons save logits.

[Plan](../../docs/hy3/HY3_IMPLEMENTATION_PLAN.md) and
[status with measured validation](../../docs/hy3/HY3_IMPLEMENTATION_STATUS.md).
