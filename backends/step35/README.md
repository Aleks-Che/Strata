# Step-3.7-Flash experimental native backend

Work in progress. This directory builds an isolated native pipe engine,
vocabulary/template oracles and CUDA checks. It does not install a profile or
connect Step to the HTTP server. Qwen, DeepSeek and GLM keep their own backends.
The GGUF architecture is `step35`.

## Offline MTP probe

`STRATA_STEP_MTP_PROBE=ON` adds `strata-step35-mtp-check` and builds the pinned
dependency's `llama-common` speculative driver. It defaults to OFF and requires
the validated CUDA runtime flags below. This is a separate executable, not the
pipe/HTTP engine; it does not enable MTP in any installed profile.

In the same developer shell and configured CUDA build directory:

```powershell
cmake -S backends/step35 -B build-local/step35-cuda -DSTRATA_STEP_MTP_PROBE=ON -DSTRATA_STEP_MTP_ACTIVE_CATCHUP=OFF -DSTRATA_STEP_RUNTIME=ON
cmake --build build-local/step35-cuda --target strata-step35-mtp-check -j 4
python -X utf8 tools/check_step35_mtp.py --engine build-local/step35-cuda/bin/strata-step35-mtp-check.exe --model H:/models/Step-3.7-Flash/UD-Q4_K_S/Step-3.7-Flash-UD-Q4_K_S-00001-of-00004.gguf --draft build-local/models/step37-mtp/Step3.7-flash-mtp-Q8_0.gguf --output-dir build-local/step35-cuda/step11-mtp/resident --cache-mib 16384 --depths 0,1,2,3 --rounds 2
```

Obtain the official sidecar at revision
`0b69336d2fd2adfdef9c66e425f7778196c31482` of
[`stepfun-ai/Step-3.7-Flash-GGUF`](https://huggingface.co/stepfun-ai/Step-3.7-Flash-GGUF/tree/0b69336d2fd2adfdef9c66e425f7778196c31482).
The Q8_0 file's expected SHA-256 is
`469a81667a6cd6d87a85d501d57155fd90cee5af7010fd289c5169881763fd57`.
The checker records the actual checksum and validates the sidecar directory,
trunk metadata, vocabulary and tensor placement before generation.

Every request starts with empty KV and a fresh speculative driver. The probe
compares greedy IDs with the existing native reference, includes EOG, warms each
depth, then reverses depth order on alternate rounds. It records draft, target
verification, catch-up, total request time and acceptance. Context is 2048 with
F32 KV, batch 17, pipeline reader 1 and no prefill cache admission. Requests are
limited to **480 total positions**, below the SWA window; this does not admit
rollover, sessions, stochastic sampling, cancellation or HTTP MTP.

The cache argument is a cap. Runtime reserves 5% of global VRAM plus 256 MiB;
the Python monitor terminates only its test child if global RAM or VRAM crosses
95%. Run a separate `--depths 0` trial without `--draft` for an off baseline that
can use the memory otherwise occupied by MTP.

Optional `--draft-placement shared-embedding` maps the draft embedding and
unused global output weights in RAM. It compares every quantized embedding byte
with the target's GPU tensor before borrowing that tensor. All admitted heads
must have their own output matrices/norms and use the common embedding. The
target outlives both contexts and the draft; no shared buffer ownership is
transferred. All matrices used for computation remain on GPU.

The additional build option `STRATA_STEP_MTP_ACTIVE_CATCHUP=ON` generates a
hash-guarded copy of the upstream speculative driver which catches up only the
heads requested by the current fresh sequence. It defaults to OFF. With this
option, `--draft-placement active-heads --active-heads 2 --depths 2` keeps head
47 in RAM and also shares the embedding. Use `--active-heads 1 --depths 1` to
keep heads 46 and 47 in RAM. Requests deeper than the resident head count are
rejected. The unpatched build rejects either placement with fewer than 3 heads.
This experiment saved VRAM but did not improve the completed local measurements.

`--prefault-experts` touches mapped expert pages before READY. The default
`--ram-prefault-reserve-mib 24576` leaves a startup reserve **in addition** to the
5% RAM margin: generation brings more expert pages into RAM. The 512 MiB and
8192 MiB reserve trials crossed 95% and were stopped by the monitor. The 24576 MiB
trial passed at 117.607 GiB peak RAM on this 125.555 GiB visible-RAM PC. Prefault
remains opt-in: startup increased and filling RAM did not establish a throughput
gain. These limits are observations for the tested short prompts, not admission
for longer requests or changing external memory pressure.

Measurements and remaining MTP gates:
[MTP trials](../../docs/Step-3.7-Flash/STEP37_FLASH_MTP_TRIALS.md).

`--cache-reuse off,on` adds a paired sweep of cache allocation reuse. The
checker alternates configuration order across rounds, preserving the expert
cache between requests; every request still starts with fresh KV. A replacement
may reuse an unpinned victim's allocation of the same charged size once its
previous synchronous readers have completed. It keeps the admission policy,
budget checks and trim behavior, with no spare allocation pool. This switch
defaults to `off` and is exposed only by the offline probe. The native pipe and
HTTP defaults do not enable it.

The result includes `generation_io`: decode-only H2D bytes, source-copy CPU
time, D2D wall time, consumer wait, cache hits/misses and allocation/reuse counts.
These times overlap and must not be added as a GPU timeline. The existing
top-level byte counters still cover prefill plus generation. Cache unit checks
and the runtime fixture cover reuse, pinned entries, pressure trim, mixed
allocation sizes and exact GPU bytes/logits through a 513-token prompt.
Measured results and reproduction commands:
[cache reuse trials](../../docs/Step-3.7-Flash/STEP37_FLASH_CACHE_REUSE.md).

`--pipeline-modes baseline,early,batch,batch-early` sweeps two additional
probe-only options. `early` refills the pinned host slot after its previous H2D
event; the device slot still waits for its consumer event. `batch` delivers all
selected expert matrices of one scheduler input, then fences once before that
input is used. The native scratch dependency fence remains. Plan pins protect
cached sources; temporary pins protect newly admitted cache destinations until
the copy stream has drained, including exception cleanup. There is no CUDA work
using those cache entries after the tensor-copy function returns.

Both options default to off. The original pipeline API delegates to that default;
only the probe and CUDA checks select the extended API. Changing batch delivery
alone preserves the ring. Changing early refill recreates it between requests,
outside the request timer; if needed, the cache releases the replacement memory
shortfall plus 64 MiB before the unchanged budget check. Capacity is unchanged.
`generation_io.pipeline_copy_fences` counts copy-stream host fences;
it excludes backend scratch fences. `pipeline_batch_ms` measures the complete
batched delivery wall time, including source waits and cache admission. In batch
mode individual `d2d_ms` is not measured (zero); neither field is CUDA D2D duration.
Paired measurements and limits:
[batched copy trials](../../docs/Step-3.7-Flash/STEP37_FLASH_BATCH_COPY.md).

The probe also accepts `--pipeline-readers 1,2` and
`--pipeline-chunks 4,8,16` for a Cartesian sweep with the selected copy modes.
Defaults remain one reader and 8 MiB per slot. Every result acknowledges both
settings and records `pipeline_configure_ms` separately from request time;
READY describes only the initial ring. Changing readers or capacity recreates
the drained ring and can trim a small amount of expert cache for headroom.
Older preserved probes can still run the historical fixed 1-reader/8-MiB
commands; non-default settings require explicit acknowledgement.
The runner also samples system and child CPU time around each request. Its
`cpu_load` percentages use the whole PC's CPU capacity; `other_percent` is the
system busy time minus the probe's busy time, not a list of other applications.
These observations help identify background load and do not control it.
The completed reader/chunk/depth sweeps did not establish a reproducible
additional gain; keep the STEP-13 candidate pending broader admission. Results:
[pipeline tuning](../../docs/Step-3.7-Flash/STEP37_FLASH_PIPELINE_TUNING.md).

`--pipeline-host cached,wc` compares ordinary cacheable pinned host buffers
with `cudaHostAllocWriteCombined` buffers. Both use the same four slots and
device buffers, source copies, ready/used events and residency pins. CPU only
writes the staging payload; the GPU reads it. The host policy is acknowledged
as `pipeline_write_combined` in each result. Changing it recreates the drained
ring before request timing. The default and both earlier configuration APIs
retain cacheable pinned RAM; the new staging API is opt-in.

Pipeline configuration entry points explicitly declare `noexcept(false)`.
This preserves the intended C++ guard/recovery behavior under MSVC `/EHsc`,
which otherwise assumes C-linkage functions do not throw. The runtime fixture
includes rejection of WC without readers and cold uploads after changing the
host policy through both the new and legacy APIs.
WC did not improve the completed local comparisons; cacheable RAM remains
the candidate. The paired results, unequal background CPU load and initial
exception-check failure are retained in the
[write-combined trials](../../docs/Step-3.7-Flash/STEP37_FLASH_WRITE_COMBINED.md).

`--cache-profile` enables host wall-time counters for cache lookup/admission,
victim selection, allocation/free, global memory sampling and router planning.
They include CPU waits/preemption; admission includes victim/refresh/allocation,
and plan build includes pins. Do not sum nested timers or interpret them as a
GPU timeline. Disabled timers do not read the clock. `generation_io` contains
decode deltas and the memory sample count.

`--cache-scan baseline,direct` compares the existing victim scan with direct
LRU entry pointers and one frequency lookup per candidate. The first minimum
still wins, pinned entries remain excluded, and a zero score ends the scan.
Map nodes have stable addresses; eviction removes their LRU node first. The
default is `baseline`. Every result acknowledges `cache_fast_scan` and
`cache_profile`; unsupported experimental settings are rejected by the runner.
The global budget95 policy and sampling cadence are unchanged. Measurements:
[cache profiling](../../docs/Step-3.7-Flash/STEP37_FLASH_CACHE_PROFILE.md).

`--host-copy crt,avx2` compares the CRT copy with an AVX2 temporal-store loop
for mmap-to-pinned staging. The AVX2 function is a separate compilation unit;
CPU/OS dispatch rejects unsupported explicit requests. The default remains
CRT, and legacy pipeline configuration restores CRT. Changing only this option
between drained requests preserves the ring and cache. The shared pipeline's
optional host-copy callback retains its original memcpy when absent; slot
ownership, buffers, ready/used events and cancellation rules are unchanged.

`--pipeline-profile` is a no-MTP diagnostic: host copy wall time/thread cycles
and CUDA events on H2D/compute streams, up to 128 graphs per request. It exports
raw trace files and retains graph summaries in the report. It recreates the
ring for each new trace and perturbs execution; disable it for final timing.
Uncovered GPU spans also contain untraced D2D and are not a GPU-idle metric.
Windows process page-fault deltas include soft/hard faults from all request
work, not just experts. CPU cycles are not converted to time. Byte/guard,
ring/runtime checks and paired full-model measurements:
[host copy trials](../../docs/Step-3.7-Flash/STEP37_FLASH_HOST_COPY.md).
On Windows/9950X/RTX 5090, the four-round MTP2 comparison measured
13.803 → 14.777 tokens/s (+7.06%) with AVX2 and 10.95% less process CPU time
per response. The no-MTP confirmation did not reproduce a decode gain.
Concurrent CPU load was not controlled; these are short greedy checker
results, and native/HTTP defaults remain unchanged.

Progress and remaining gates:
[plan](../../docs/Step-3.7-Flash/STEP37_FLASH_IMPLEMENTATION_PLAN.md),
[status](../../docs/Step-3.7-Flash/STEP37_FLASH_IMPLEMENTATION_STATUS.md).

## Inspect the local model

From the repository root:

```powershell
python tools/setup_step35.py --model-dir H:/models/Step-3.7-Flash/UD-Q4_K_S --inspect --output docs/Step-3.7-Flash/STEP37_FLASH_GGUF_INVENTORY.json
python -m unittest tools.test_setup_step35 tools.test_step35_tokenizer
```

The inspector reads headers, accepts a metadata-only first shard, verifies split
identity, metadata, names, shapes, quantized rows and physical ranges. It maps all
754 tensors to the audited text-trunk loader. Its fingerprint includes ordered
header hashes and file lengths. It does **not** detect corruption of weight bytes.
NextN-bearing GGUFs, fused QKV, LongRoPE and vision need separate admission work.
Some tensors optional in upstream are required by this particular model profile.

## Build the isolated oracle

Candidate: [Unsloth llama.cpp 86ebfef2](https://github.com/unslothai/llama.cpp/tree/86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9).
The local archive `build-local/llama-glm-86ebfef2.tar.gz` is reused only as input
bytes. SHA-256 must equal
`f8e524b635b726bae74fd8f84bb9249e5b09384c63207f6707c5a3f921acad99`.
Sources are extracted into the Step build directory; no GLM source patches are
applied. This candidate is validated for vocabulary, templates, synthetic graphs
and the local full UD-Q4_K_S text trunk. Pipeline, cache and MTP remain future work.

In an x64 MSVC developer shell (CMake and Ninja on PATH):

```powershell
cmake -S backends/step35 -B build-local/step35-oracle -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_STEP_ARCHIVE=C:/work/git/my-repos/Strata/build-local/llama-glm-86ebfef2.tar.gz -DSTRATA_STEP_CUDA=OFF
cmake --build build-local/step35-oracle --target strata-step35-tokenizer strata-step35-template -j 4
ctest --test-dir build-local/step35-oracle --output-on-failure
python tools/check_step35_tokenizer.py --gguf H:/models/Step-3.7-Flash/UD-Q4_K_S --oracle build-local/step35-oracle/bin/strata-step35-tokenizer.exe --output docs/Step-3.7-Flash/STEP37_FLASH_TOKENIZER_PARITY.json
```

Use a separate build directory and suitable compiler on other hosts. CPU use here
is only for vocabulary validation. It is not the planned inference execution path.
CUDA matrix/router, synthetic graph and native engine checks are described below.

`--version` returns JSON with source/archive identity. `--gguf FIRST.gguf` loads
vocabulary only and prints a ready record, vocabulary size and native EOG IDs.
Each following JSON line contains `text` and Boolean `parse_special`. Responses
contain `ids` and native `decoded_hex`. `QUIT` ends the process. `add_special`
is always false because the embedded template owns BOS. Invalid requests produce
one error response; diagnostic model logs go to stderr.

Measured on the local UD-Q4_K_S headers: **2190/2190** exact token-ID and decoded
byte comparisons pass, including every CONTROL/USER_DEFINED token in both modes,
Russian, CJK, code, Unicode, whitespace and deterministic mixed inputs.
Native EOG IDs are **1 and 128007**; PAD 2 is not EOG in this artifact/candidate.
API stop handling remains unverified. Independent Jinja rendering is checked by
the separate template fixture below; native generation uses the EOG token set.

## CUDA matrix and router fixtures (P0.4)

Use a new build directory. In an x64 MSVC developer shell, with the local CUDA
13 installation on PATH:

```powershell
$env:CUDA_PATH="$PWD\build-local\cuda-13.0"
$env:PATH="$env:CUDA_PATH\bin;$env:CUDA_PATH\bin\x64;$env:PATH"
cmake -S backends/step35 -B build-local/step35-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_STEP_ARCHIVE=C:/work/git/my-repos/Strata/build-local/llama-glm-86ebfef2.tar.gz -DSTRATA_STEP_CUDA=ON -DSTRATA_STEP_KERNEL_CHECK=ON -DSTRATA_STEP_STRICT_F32=ON -DSTRATA_STEP_ROUTED_STRIDES=ON -DCMAKE_CUDA_ARCHITECTURES=120 "-DCMAKE_CUDA_COMPILER=$env:CUDA_PATH/bin/nvcc.exe" "-DCUDAToolkit_ROOT=$env:CUDA_PATH"
cmake --build build-local/step35-cuda -j 4
ctest --test-dir build-local/step35-cuda --output-on-failure --no-tests=error
$env:NVIDIA_TF32_OVERRIDE='0'
build-local/step35-cuda/bin/strata-step35-kernels-check --output docs/Step-3.7-Flash/STEP37_FLASH_CUDA_KERNELS.json
```

On the RTX 5090: **78/78** synthetic matrix/router cases passed, plus **2/2**
CTest checks. These fixtures cover F32/Q8_0/Q6_K/Q4_K, batches 1/4/17, eight
selected experts out of 288, padded expert/input/ID strides, exact guards and
explicit CPU/scalar references. They do not measure model tok/s or validate the
full Step graph, memory delivery pipeline or sessions.

Two optional correctness patches are required for this checked configuration:

- `STRATA_STEP_STRICT_F32`: custom F32 MMF honors `NVIDIA_TF32_OVERRIDE=0`.
  The unpatched implementation still used explicit TF32 instructions.
- `STRATA_STEP_ROUTED_STRIDES`: the synchronous routed gather uses physical
  token stride, preventing reads from padding after the strict F32 path is selected.

Both patch generated files in the Step build directory. The archive, extracted
dependency sources and other backends remain untouched. Hash/anchor guards reject
a different dependency. The flags default to OFF to allow an unpatched control;
use both ON for subsequent Step graph validation. CPU-only builds keep them OFF.
The padded router fixture uses an explicit GPU `cont` operation before sigmoid,
which requires contiguous input in this candidate.

[Numerical criteria and before/after evidence](../../docs/Step-3.7-Flash/STEP37_FLASH_CUDA_VALIDATION.md).
`step-source-build.txt` records compiler, source hashes and applied patches;
the tokenizer's `--version` reports the same patch identity. Its checker admits
the original CPU build and the fully validated two-patch CUDA build.

## Native graph and state (P0.5)

Add `-DSTRATA_STEP_GRAPH_CHECK=ON` to the CUDA configure command above, rebuild,
and run CTest. This adds `strata-step35-graph-check` and `step35_native_graph`.
For a separate report:

```powershell
$env:NVIDIA_TF32_OVERRIDE='0'
build-local/step35-cuda/bin/strata-step35-graph-check build-local/step35-cuda/graph-fixture
```

On the RTX 5090, **75/75** cases pass. The fixture writes a deterministic F32 GGUF
with three actual Step layers (full/SWA/full), dense and routed/shared FFNs,
Q/K norms, partial/full RoPE, rotary factors, per-head gates and active clamps.
It compares GPU with CPU/scalar references and checks every compute node's backend.
GPU-vs-CPU logits max absolute error is 3.73e-6 (limit 5e-4; NMSE limit 1e-7).
Prefill1024, split microbatches and serial decode agree within those criteria.
No CPU tensor-math node is accepted in GPU runs.

`session_snapshot.hpp` preserves physical KV indices and masked retained SWA
history using the pinned native cache API. It also restores the append cursor.
Same-shape continuations are bit-exact across A→B→A and snapshots at prefix lengths
0/511/512/513/767/768/769/1024/1530/1536. Rollback across ring wrap uses a checkpoint
taken **before** the tentative suffix. Native sequence serialization compacts the
ring, and native tail removal can change allocation; neither guarantees exact
continuation. Do not substitute them for the checkpoint path.

This helper is deliberately limited to one unshifted, append-only Step text
sequence in the same live context. It is an in-process KV checkpoint, not a file
format, session manager, MTP implementation or sampler/logit snapshot. Its owner
context must outlive it. Memory budgets, F16 KV/Flash Attention and full-model
session behavior remain future checks. Failed restores clear both caches.

## Embedded template oracle (P0.5)

`strata-step35-template` compiles the pinned dependency's Jinja renderer without
the inference engine. Each JSON line contains `template` and `context`; the
response contains `rendered` or `error`. `QUIT` exits. The template bytes remain
unchanged. This candidate lacks `fromjson`, so the Step oracle parses string
tool arguments into objects with its native JSON parser before rendering.
Python independently renders the original string arguments through Jinja2/fromjson.
This normalization must be carried into the future Step API adapter.

```powershell
python tools/check_step35_template.py --gguf H:/models/Step-3.7-Flash/UD-Q4_K_S --oracle build-local/step35-cuda/bin/strata-step35-template.exe --tokenizer-oracle build-local/step35-cuda/bin/strata-step35-tokenizer.exe --output docs/Step-3.7-Flash/STEP37_FLASH_TEMPLATE_PARITY.json
```

**215/215** cases pass: 208 byte/token-ID comparisons, a hand-written BOS/prefix
fixture and six malformed/non-object JSON rejection cases. Inputs cover all three
reasoning levels, history and reasoning replay, text content forms, tools with
object/string arguments, multiple calls/results, and generation-prefix on/off.
The check does not enable HTTP, assistant continuation semantics or vision.
P0 configuration CTest results: CUDA **4/4**, CPU **2/2** (CPU checks are version checks).

## Native pipe and selected-copy baseline (P1)

Add `-DSTRATA_STEP_RUNTIME=ON` to the CUDA configure command (both correctness
patches are required), then build. It adds `strata-step35`, the runtime checker
and pipe lifecycle checks. Run the engine explicitly; no default profile is changed:

```powershell
build-local/step35-cuda/bin/strata-step35 --native H:/models/Step-3.7-Flash/UD-Q4_K_S/Step-3.7-Flash-UD-Q4_K_S-00001-of-00004.gguf --max-context 2048 --batch-size 17 --kv f32 --copy-mode pinned --serve
```

All non-routed weights and model operations stay on GPU. Routed weights are
mapped from their GGUF shards, without loading/pinning the entire expert set.
The scheduler copies selected ranges through one **16 MiB** pinned buffer and
synchronizes each chunk before reusing it. Full expert-tensor copies and CPU
tensor math are rejected. GPU scratch retains the native tensor layout; there
is no persistent expert cache or overlap in this baseline mode. The bounded staging buffer is
separate from scratch, KV and fixed model weights.

`--copy-mode native` selects the pinned dependency's original selected-range
copy for comparison, while retaining the GPU audit. It does not collect source
or H2D byte/timing counters; zero counters in that mode mean uninstrumented.
`--kv f16` is also supported and checked on synthetic resident/mapped graphs.
Full-model admission used F32 KV and Flash Attention off. The engine sets
`NVIDIA_TF32_OVERRIDE=0` and `GGML_OP_OFFLOAD_MIN_BATCH=1` before backend init.

The runtime patch uses generated `ggml-backend.cpp` and model-loader translation
units in the Step build only. It disables whole-file prefetch for Step, keeps
the original split-file loader, adds pre-compute GPU auditing and replaces only
selected-range transfers. The extracted dependency and other backends stay intact.
The reported patch set appends `step-sync-selected-copy-gpu-audit-mmap-demand`.

Protocol: `INFO` and `READY <context> stop session-id` on startup; requests are
`GEN <max_new> [sampling key=value...] <comma-separated token IDs>`, progress
`PP`, generated `T <id>`, then Strata-compatible `DONE`. `STOP` cancels between
microbatches/tokens, synchronizes and clears state; `QUIT` exits. `ENC 0|1 [hex]`
returns `IDS` with native tokenization (`add_special=false`). Invalid commands
return `ERR`; the next request can proceed. Each request creates a fresh sampler
and clears both KV caches. A session hash is accepted for isolation, not caching.
There is no HTTP service, tools parser, vision or MTP in this engine yet.

Checks from the repository root, with CUDA DLLs on PATH:

```powershell
ctest --test-dir build-local/step35-cuda --output-on-failure --no-tests=error
python tools/check_step35_model.py --engine build-local/step35-cuda/bin/strata-step35.exe --gguf H:/models/Step-3.7-Flash/UD-Q4_K_S --output-dir build-local/step35-cuda/model-check --predict 128
python tools/check_step35_model.py --engine build-local/step35-cuda/bin/strata-step35.exe --gguf H:/models/Step-3.7-Flash/UD-Q4_K_S --output-dir build-local/step35-cuda/request-check --request-reference build-local/step35-cuda/model-check/model-report.json
```

STEP-04 baseline CUDA CTest: **6/6**, including **27/27** transport comparisons
and rejection checks and **22/22** pipe lifecycle checks. Transport cases include
F32/F16 KV, mixed Q4_K/Q8_0/Q6_K weights, batches 1/4/17, SWA crossing, and an
18 MiB selected range split across the 16 MiB staging buffer. Full-model request
checks pass **5/5**: both native ENC modes, seeded repeat, STOP and fresh state
after cancellation.

The full-model checker runs two prompts and A→B→A with native then pinned copy,
saves F32 logits, requires exact IDs/bytes, and records process RAM and device
memory. It stops only its test process if sampled global RAM or VRAM exceeds
95%. This external test guard is not the future runtime budget controller.
These sequential timing observations do not establish a transport speedup:
disk/file-cache state differs between runs. `decode_steps/decode_forward_ms`
excludes the first sampled token and separates forward throughput from prefill.
`--logits-file FILE.f32` is a diagnostic engine option that overwrites that file
per request; its rows correspond to sampled tokens.

Reports and measured results are recorded in the implementation status. The
32-token admission includes truncated reasoning; the longer run verifies EOG
and completed answers separately. No resource-filling or speed-optimized default
is selected from this baseline. The optional cache and budget checks below extend
this baseline; the opt-in asynchronous pipeline is described below.

## Bounded GPU expert cache (P1.4/P3, experimental)

Use `--expert-cache-mib auto` with `--copy-mode pinned` to enable a cache whose
limit follows global VRAM use. An integer sets an additional MiB cap; `0` keeps
the synchronous baseline and is still the default. Existing model profiles and
the HTTP server are not changed. Native reference mode requires cache zero.

The cache keeps individual expert matrices, including the native MMQ padding.
Its registry records the live model generation, tensor name/type/shape and actual
GGUF shard offset. Re-registering a model clears entries and admission history.
Router IDs are deduplicated by the native scheduler. A hit copies bytes from cache
to native scratch on GPU; a miss uses the bounded pinned path, then may fill the
cache. Shared decaying-frequency metadata controls admission. No routing or
quantization is changed. Transfers and consumers are synchronized; no overlap or
event-protected asynchronous leases are claimed.

The controller samples global VRAM through the shared PCI-matched NVML reader on
Windows, plus global physical RAM. CUDA's per-process WDDM memory view is not used
as the global budget. Sampling occurs before each decode and after every 64 MiB
of new cache admissions. It leaves 5% VRAM plus 256 MiB, trims cache at safe
boundaries, and refuses further decode under RAM pressure (5% plus 64 MiB reserve),
unavailable/inconsistent readings or fixed/external VRAM above 95%. Failed cache
allocations bypass admission and retain synchronous delivery. Fixed model load
and arbitrary external allocations are not atomic reservations; broader startup
OOM recovery and pressure stress coverage remain part of P1.4/P3.5. RAM is not
filled with a duplicate of the entire mapped model.

`STRATA_STEP_REQUEST` adds cache hits/misses, evictions, rejected admissions,
allocation bypasses, resident/limit bytes, H2D/D2D bytes and sampled memory.
Source/H2D bytes and cache hits/misses are also split into prefill and decode.
Resident bytes count cache allocations rounded to 64 KiB; global VRAM samples
also include driver overhead and other processes. Cache-fill D2D bytes are
separate from hit D2D bytes. Per-matrix MMQ padding can make transferred bytes
slightly larger than the scheduler's `requested_bytes`. Timing counters are CPU
wall times around synchronized operations, not CUDA event durations.

```powershell
build-local/step35-cuda/bin/strata-step35 --native H:/models/Step-3.7-Flash/UD-Q4_K_S/Step-3.7-Flash-UD-Q4_K_S-00001-of-00004.gguf --max-context 2048 --batch-size 17 --kv f32 --copy-mode pinned --expert-cache-mib auto --serve
python tools/check_step35_cache.py --engine build-local/step35-cuda/bin/strata-step35.exe --gguf H:/models/Step-3.7-Flash/UD-Q4_K_S --reference docs/Step-3.7-Flash/STEP37_FLASH_MODEL_VALIDATION.json --output-dir build-local/step35-cuda/cache-check
```

The checker runs two full prompts twice per process, in cache order `0,auto,auto,0`,
then checks STOP and a fresh greedy prefix. Every output token and complete F32
logit row must match the saved P1 reference. It retains per-request timing and
memory samples, and stops its own process above 95% global RAM/VRAM. File cache
is not purged, so this is an order-balanced warm/repeat comparison, not a cold-SSD
benchmark. Initial cache filling and repeated requests must be reported separately.

## Router lookahead pipeline (P3, experimental)

Use `--expert-pipeline-readers 1` with pinned copy mode to enable asynchronous
expert delivery. `0` retains synchronous delivery and is the default; `2` is
available for measurement. `--expert-pipeline-chunk-mib 4|8|16` sets each of four
pinned host slots and four GPU slots (8 MiB per slot by default). The ring is
included in global memory accounting before cache admissions. Cache remains
independent: use `--expert-cache-mib auto` to combine both features.

After the real router IDs become available, the Step scheduler follows the
actual gate/up/down split order for that routing tensor. It removes cache hits
from the read plan and protects their residency until the plan ends. Workers
read the remaining matrices into the shared bounded ring and submit H2D on an
independent stream. CUDA ready/used events protect both host and device reuse.
Only the scheduler writes native scratch, after its previous consumers finish.
This avoids overwriting activations that alias expert scratch.

Cache hits and the final ring-to-scratch D2D still synchronize before returning.
Residency pins are not asynchronous cache consumer leases. The overlap comes
from preparing subsequent matrices while current transfers/computation proceed.
No expert prediction, changed routing, early host refill or MTP is enabled.
Cancellation drains outstanding reads before model mappings can be released.
Unexpected CUDA worker failures remain fatal to that pipeline instance.

Pipeline counters report actual host reads, H2D/D2D payload, unused prefetch,
reader concurrency, fixed ring capacities and CPU slot/consumer waits. Pipeline
`h2d_ms` is not measured by the synchronous counter and stays zero; it does not
mean copying takes no time. `d2d_bytes` includes ring-to-scratch and cache hits,
while cache-fill D2D stays in its separate counter. The legacy `d2d_ms` timer
covers synchronized cache hits/fills only, excluding ring-to-scratch transfers.

For an independent diagnostic run, add `--trace-file FILE.json --trace-graphs 8`.
This records CUDA events on the actual copy and backend compute streams and
exports their intervals and intersection. It synchronizes at graph boundaries
and must stay disabled in throughput benchmarks. The trace checker also checks
the complete answer and F32 logits against the saved reference.

```powershell
python tools/check_step35_pipeline.py --engine build-local/step35-cuda/bin/strata-step35.exe --gguf H:/models/Step-3.7-Flash/UD-Q4_K_S --reference docs/Step-3.7-Flash/STEP37_FLASH_MODEL_VALIDATION.json --output-dir build-local/step35-cuda/pipeline-check --order 0:8,1:8,1:8,0:8
python tools/check_step35_trace.py --engine build-local/step35-cuda/bin/strata-step35.exe --gguf H:/models/Step-3.7-Flash/UD-Q4_K_S --reference docs/Step-3.7-Flash/STEP37_FLASH_MODEL_VALIDATION.json --output-dir build-local/step35-cuda/pipeline-trace --readers 1
```

All timing comparisons use the same auto cache, short prompts, context 2048,
batch 17, F32 KV and disabled FA/MTP/TF32. The pipeline checker verifies complete
logits and IDs on every request, then STOP and fresh-request recovery per process.
Results and remaining coverage gaps are in the Step implementation status.

STEP-06 CUDA CTest passes **10/10**: **92** transport/runtime cases, **18** direct
ring cases, **14** cache/budget cases, **22** pipe cases in each of three modes
(off/one reader/two readers), plus the existing **78** kernels and **75** graph
checks and both oracle version checks. Runtime fixtures cover mixed/F32 weights,
F32/F16 KV, SWA crossing, warm all-hit plans, small-cache evictions and recovery
after a rejected registry/plan. The direct ring check covers an 18 MiB + 513-byte
source, guard bytes, wrap/reuse without per-chunk host synchronization, rejected
source identity, abandoned uploads and restart with changed source contents.

## Optional prefill cache admission (STEP-08)

`--expert-cache-prefill on|off` controls filling the expert cache during prefill.
The default is `on`, retaining the previous behavior. With `off`, existing cache
hits and frequency learning remain active, but misses are delivered to scratch
without allocating or evicting cache entries. Decode resumes ordinary admission.
Global memory checks still run before each microbatch. This option requires
pinned copy mode; use a nonzero cache budget to make it useful.

The request owner sets prefill/decode explicitly; batch size does not determine
the phase. A one-token prefill tail is still prefill. The scope restores the
previous phase on completion, STOP or exception. Both synchronous and pipeline
paths use the same admission gate; their CUDA synchronization is unchanged.
Metrics split `cache_fill_bytes` into `prefill_cache_fill_bytes` and
`decode_cache_fill_bytes`, and report `prefill_admission_skips` and
`prefill_admission_skip_bytes`. Skipped bytes describe candidate entries,
not saved traffic: ordinary admission might also have rejected a candidate.

```powershell
python tools/check_step35_admission.py --engine build-local/step35-cuda/bin/strata-step35.exe --gguf H:/models/Step-3.7-Flash/UD-Q4_K_S --reference docs/Step-3.7-Flash/STEP37_FLASH_MODEL_VALIDATION.json --context-reference docs/Step-3.7-Flash/STEP37_FLASH_STEP07_CONTEXT.json --output-dir build-local/step35-cuda/step08-admission
```

The output directory must be new. The checker uses context4096, batch17, F32 KV,
auto cache, one reader and 8 MiB slots. It runs `on/off/off/on` with the same
short/medium/short request sequence, checks complete F32 logits and IDs, and
checks STOP/recovery per process. `--long-only` instead checks the saved 2591-token
reference with admission off; that separate run is not a paired speed benchmark.
Logits, memory samples and per-request timing are retained. File cache is not
purged and the checker terminates only its model child above the 95% global
RAM/VRAM guard. Defaults should be chosen from full-request measurements,
including decode after prefill and subsequent short requests.

On the tested Windows PC (Ryzen 9950X, 128 GiB, RTX 5090 32 GiB), the two runs
per policy averaged 57.376 → 40.281 s for the 511-token prefill, and
60.137 → 43.550 s for that request including 16 output tokens. Decode following
this prompt slowed from 5.459 to 4.608 tokens/s; subsequent repeated short
requests gave 8.773 → 8.505 tokens/s. Use `off` for this measured prefill-heavy
workload; retain `on` as the compatibility default pending broader workloads.
All 28 reference comparisons and five STOP/recovery pairs passed. Full results,
memory samples and limitations are in the
[STEP-08 status](../../docs/Step-3.7-Flash/STEP37_FLASH_IMPLEMENTATION_STATUS.md).

## Real shard bytes and external pressure (P3.4/P3.5)

`check_step35_shards.py` selects first/middle/last available layers for gate,
up and down in each payload shard, then experts 0/middle/last. The native checker
independently reads GGUF offsets and strides, maps files without prefetch, and
compares pipeline/cache GPU bytes against separate file reads. It checks native
512-byte tails, destination guards, two consumer streams, residency pins, trim
and reload. Only sampled pages are read; original GGUF files are opened read-only.

```powershell
python tools/check_step35_shards.py --checker build-local/step35-cuda/bin/strata-step35-shards-check.exe --gguf H:/models/Step-3.7-Flash/UD-Q4_K_S --output-dir build-local/step35-cuda/step07-shards
python -m unittest tools.test_step35_shards
```

`check_step35_context.py` runs separate sync/pipeline processes with context 4096,
auto cache and F32 KV. It compares full F32 logits for a prompt over 2048 tokens,
checks a short P1 reference, and repeats the short request before/during/after
external pressure. A separate holder commits **128 MiB VRAM + 2 GiB RAM** after
checking live global headroom. Its CUDA context starts before the model so that
the engine's cache budget accounts for it. The runner also checks STOP during a
long prefill and exact fresh-request recovery.

The pressure holder is an opt-in checker, excluded from normal engine startup
and CTest. It rejects insufficient headroom and releases allocations on FREE,
EOF or error. The runner closes its own helper/model processes in cleanup.
This is a bounded pressure experiment, not forced OOM or proof at every resource
limit. It does not change the application defaults or claim throughput gains.

```powershell
python tools/check_step35_context.py --engine build-local/step35-cuda/bin/strata-step35.exe --holder build-local/step35-cuda/bin/strata-step35-pressure-holder.exe --gguf H:/models/Step-3.7-Flash/UD-Q4_K_S --reference docs/Step-3.7-Flash/STEP37_FLASH_MODEL_VALIDATION.json --output-dir build-local/step35-cuda/step07-context
```

## Text frontend adapter (STEP-09)

`serve/step35.py` implements the reviewed embedded template, request normalization
and a separate reasoning/XML-tool parser. STEP-10 adds explicit HTTP selection
through a `step35` profile. See [fixture/adapter details](../../serve/fixtures/README.md#step-37-flash-fixture-and-adapter)
for effort/replay rules, typed arguments and streaming limits.

```powershell
python -m unittest serve.test_step35
python tools/check_step35_template.py --gguf H:/models/Step-3.7-Flash/UD-Q4_K_S --oracle build-local/step35-cuda/bin/strata-step35-template.exe --tokenizer-oracle build-local/step35-cuda/bin/strata-step35-tokenizer.exe --runtime-adapter --output docs/Step-3.7-Flash/STEP37_FLASH_FRONTEND_TEMPLATE.json
python tools/check_step35_frontend.py --engine build-local/step35-cuda/bin/strata-step35.exe --gguf H:/models/Step-3.7-Flash/UD-Q4_K_S --reference docs/Step-3.7-Flash/STEP37_FLASH_MODEL_VALIDATION.json --template-oracle build-local/step35-cuda/bin/strata-step35-template.exe --tokenizer-oracle build-local/step35-cuda/bin/strata-step35-tokenizer.exe --output-dir build-local/step35-cuda/step09-frontend
```

The model checker requires a new output directory. It uses the Step adapter for
three native-oracle-checked prompts: exact P1 control, a generated typed tool call,
and continuation after a local result stub. It compares parsing at actual token,
character and whole-text boundaries. No HTTP listener or external tool runs.
The engine and cache/pipeline defaults remain as in STEP-08; the checker explicitly
uses context4096, batch17, F32 KV, auto cache, one reader and prefill admission off.

## Experimental HTTP profile (STEP-10)

Create a new directory; existing profiles are refused and no default model is
changed. Admission checks the reviewed shard headers/fingerprint, engine source
and patch set, exported control tokens and template hash. Payload checksums are
outside this admission check. The profile uses context4096, batch17, F32 KV,
one reader, prefill admission off, greedy sampling and the backend's 95% memory
budget. It caps expert cache at 8192 MiB: Step HTTP tests with auto cache at
17121 MiB and an explicit 14336 MiB cap crossed the global RAM guard on
Windows/128 GiB/RTX 5090. One earlier 14427 MiB run completed all requests;
that observation alone was insufficient to select a default. The profile does
not enable MTP; the smaller cache is a memory margin, not a measured speed win.

```powershell
python tools/prepare_step35_profile.py --model H:/models/Step-3.7-Flash/UD-Q4_K_S --engine build-local/step35-cuda/bin/strata-step35.exe --output-dir build-local/step35-cuda/my-step-profile --cuda-dir build-local/cuda-13.0
python -m serve.server --engine strata --config build-local/step35-cuda/my-step-profile/step37.json --port 8093
python -m unittest serve.test_step35 serve.test_step35_http tools.test_step35_profile
python tools/check_step35_http.py --profile build-local/step35-cuda/my-step-profile/step37.json --reference build-local/step35-cuda/step09-frontend/frontend-model-report.json --output-dir build-local/step35-cuda/my-http-check
```

The checker requires a new output directory and the same engine as its STEP-09
reference. It runs both APIs in JSON/SSE, a local tool-result dialogue, socket
disconnects during prefill/reasoning/partial tool output, and a reference request
after each disconnect. It checks native input/output IDs, not just visible text.
Its resource monitor stops only its own engine if global RAM/VRAM exceeds 95%.

Step resolves EOG from reviewed vocabulary spellings and metadata: native IDs
1 and 128007, excluding PAD2. Stop strings match raw generated text, including
reasoning and tool markup, across token boundaries. A partial tool is not executed.
Both APIs accept low/medium/high effort; omission leaves the embedded prompt
unchanged. The web setting's default is high. Hard reasoning budgets, disabling
thinking and clear-thinking overrides are rejected. No external MCP server or
interactive browser validation is implied by the local stub and HTTP tests.
