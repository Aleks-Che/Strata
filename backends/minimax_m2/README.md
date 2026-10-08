# MiniMax-M2.7 experimental backend

This standalone Windows/CUDA backend admits the reviewed single-file
`minimax-m2` Q4_K_M checkpoint. It has no server integration or model profile yet.
The synthetic fixture is admitted only by the test executable.

The synchronous baseline keeps non-routed weights, KV and model computations on
CUDA0. Routed tensors retain their original Q4_K/Q6_K encoding in a virtual file
mapping. The scheduler selects experts, then reads their ranges into one 16 MiB
pinned buffer and uploads them to GPU scratch. Each upload and scratch consumer
is fenced before reuse. The native scheduler's extra padding, at most 512 bytes
per contiguous range, is preserved. It never allocates the entire checkpoint as
a host tensor buffer. Windows may retain read data in its file cache.

`--expert-reader file|mmap|mmap-direct|mmap-decode` selects the mode-2 host delivery path.
The default remains `file` (ReadFile into pinned staging). `mmap` copies mapped
pages into that same staging buffer. `mmap-direct` submits the mapped address to
CUDA and uses no application staging allocation; the driver may stage pageable
memory internally. `mmap-decode` uses ReadFile for prefill and the first serial
decode (workspace warmup), then mapped-to-pinned copies for subsequent decode.
This avoids faulting broad prefill routes into the process's mapped working set.
Each method limits individual copies to 16 MiB and fences
before source/destination reuse. GPU cache hits do not read the host source.

All mmap methods use the existing GLM working-set budget helper without changing
other backends. It limits this process to a share of global RAM targeting 94%,
refreshes every 250 ms or sooner under pressure, and restores the prior process
limits on release. The independent hard guard still checks 95% global RAM/VRAM.
Clean file-backed pages remain reclaimable by Windows; there is no duplicate
heap copy of the expert set, locked RAM cache, or pagefile change. This is a
bounded mapped working set, not a guarantee that all requested experts are resident.

Before computing, the scheduler rejects any non-view computation assigned to
CPU and any fallback that would copy a full expert tensor. Context creation and
each decode check global RAM/VRAM against a 95% ceiling, including other
processes. During a decode, copy/compute boundaries also check memory at most
once per 25 ms; a failing budget stops the remaining work and reports the
specific RAM/VRAM error. Allocation admission reserves additional workspace. These checks are
not an OS-enforced limit against concurrent allocations by other applications.
They do not interrupt a running CUDA kernel or file read. Reports include the
number of memory checks and sampled memory peaks; they are not continuous peaks.

An optional GPU matrix cache serves selected experts through D2D copies with
explicit stream completion before reuse or eviction. There is no MTP, CUDA
Graphs or prefix reuse. Activations use
the strict F32 control; attention uses F32 KV with FA disabled.

`--pipeline-readers 1|2 --pipeline-chunk-mib 4|8|16` enables an experimental
native-file reader pipeline in mode 2 (`--expert-reader file` only). The default
is zero readers. It reuses the shared transport without modifying other backends:
four pinned slots, four GPU slots, independent H2D and delivery streams, and
CUDA events protecting slot reuse. Each ring uses 16..64 MiB depending on chunk
size. Allocation is lazy, subject to global RAM/VRAM admission and host commit
headroom. It replaces the synchronous 16 MiB staging allocation.

By default each plan contains the current tensor's selected ranges (at most 256).
`--pipeline-lookahead 1` extends a plan to at most three matrices of the same
MoE block, in the scheduler's actual split order and with the same observed router
IDs. It never predicts the next layer's experts. A scan is bounded to 32 splits;
the RAM/VRAM rings keep the same size. Reads and H2D of a future matrix can run
while CUDA computes the current matrix. Producers write only their own ring;
future scheduler scratch is not touched early. Delivery verifies source and
ranges against the plan before any copy.

GPU cache hits across the plan are protected from eviction while native reads
prepare misses. Every delivery into scheduler scratch completes before its
compute consumer; cache fills and hits also wait for their stream.
`--pipeline-d2d-batch 1` moves those per-copy waits to one delivery-stream fence
per tensor. It requires the file pipeline and works with either plan length.
New cache destinations acquire residency pins before their fills are enqueued;
the same stream orders ring delivery, cache hits, guard bytes and cache fills.
Pins prevent pending allocations from being evicted or reused. They are released
only after completion, including during exception unwinding. This can change
cache admissions when a small cache has no unpinned victim. Observers run after
the tensor fence. The previous scratch consumer is still synchronized before
each tensor, and global memory checks remain active during submission.
Cancellation or any worker/copy error drains the plan
before releasing cache pins and invalidates incomplete cache state. Cancellation
is observed at transfer boundaries and before H2D submission; an in-progress
native read may finish before the consumer reaches the drain operation. A scope
guard also drains abandoned lookahead on every scheduler exit, including failed
compute and callbacks. An unfinished plan cannot silently succeed.

`--pipeline-trace NEW.json` records CUDA event intervals on the actual H2D,
ring-delivery and compute streams for the first four plans. With lookahead enabled,
`h2d_compute_overlap_ms` reports their H2D/compute intersection; `overlap_ms` keeps
its earlier meaning of H2D/ring-delivery intersection. The event wrapper uses the
pinned CUDA backend's event representation. Keep tracing disabled in throughput
runs. Trace, pipeline, lookahead and D2D batching are opt-in; the status contains measured results.

`--gpu-cache-mib N` sets an upper bound (default 0, file mode 2 only). The live
budget includes external VRAM use and keeps 5% plus 256 MiB free. Allocation
growth also requires 1 GiB of available Windows commit. KV and workspace take
priority: creating a context empties the cache, prefill only serves hits, and
admission starts after the first serial decode has completed. The cache survives
KV clears between requests. The Step backend's bounded cache primitive is reused
without modifying it; identities and model lifecycle are specific to MiniMax.

Keys include model generation and a registered tensor/expert ID. Registration
checks name, quantization type, shape, strides, mapping identity and file offset.
Model unload invalidates all entries before unmapping. Cache entries store real
guard bytes from the following expert, with no read beyond the tensor. Entries
use allocations rounded up to 64 KiB, counted against the budget. The reported
resident bytes are the sizes requested from CUDA; the global NVML guard also
accounts for driver overhead, so the usable cache can be below the requested cap. Completed
allocations can be reused after eviction; failed fills invalidate the cache.
OOM during admission bypasses caching and retains the uploaded scratch data.

`--gpu-cache-allocator arena` packs matrices into size-class slabs of up to
64 MiB, using the Hy3 slot allocator through a MiniMax budget wrapper. The
wrapper caps total backing bytes, including empty slots and class padding,
and reserves host commit for an entire new slab. Empty slabs are returned to
CUDA immediately. Under pressure the cache evicts until physical backing fits,
even when freeing a few slots is insufficient to release a partially used slab.
The default allocator remains `cuda` (separate allocations); cache is still off
unless `--gpu-cache-mib` is positive. Other backends are unchanged.

Arena reports separate `arena_reserved` (whole slabs) from `arena_live`
(occupied slots, equal to `cache_resident`). `arena_rejects` counts refused slab
growth at a budget/commit boundary. These intentional denials also contribute
to `cache_oom` and can trigger allocation reuse; they do not mean a request
failed or that CUDA itself ran out of memory.

## Build

Configure from an x64 MSVC developer shell, using the same audited archive and
private CUDA build as the numerical checks:

```powershell
cmake -S backends/minimax_m2 -B build-local/minimax-m2-cuda -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 `
  -DCUDAToolkit_ROOT="$PWD/build-local/cuda-13.0" `
  -DCMAKE_CUDA_COMPILER="$PWD/build-local/cuda-13.0/bin/nvcc.exe" `
  -DSTRATA_MM27_ARCHIVE="$PWD/build-local/llama-glm-86ebfef2.tar.gz" `
  -DSTRATA_MM27_CUDA=ON -DSTRATA_MM27_RUNTIME=ON
cmake --build build-local/minimax-m2-cuda --target strata-minimax-m2-runtime-check strata-minimax-m2-lifecycle-check strata-minimax-m2-cache-check strata-minimax-m2-arena-check strata-minimax-m2-reload-check strata-minimax-m2-bench -j 6
```

Runtime patches are generated in this build directory, guarded by original
source hashes. Other backends and the extracted candidate sources are untouched.

## Run and verify

Use the launcher: it supplies CUDA DLL directories and the numerical settings
**before process start**. Setting these inside `main()` produced cross-batch
numerical failures on the test PC; the runtime now rejects missing settings.

```powershell
python tools/run_minimax_m2.py --engine build-local/minimax-m2-cuda/bin/strata-minimax-m2-runtime-check.exe -- build-local/mm27-new-fixture-run
python tools/check_minimax_m2_runtime.py --model H:\models\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --out build-local/mm27-new-full-run
```

Both checks require new output directories. The full-model check runs English,
Russian, Chinese, code and a longer numeric prompt. Each case uses a fresh
process for file delivery and the native mmap reference. It compares every
generated-step logit and greedy token ID. The order is not controlled for cold
versus warm caches, so these are correctness runs, not a performance A/B.

The lifecycle suite covers full-model cancellation during prefill and decode,
injected budget rejection, clear/recovery, unload/reload, real memory pressure,
and contexts 2048/4096. It runs GPU work sequentially and needs a new directory:

```powershell
python tools/check_minimax_m2_lifecycle.py --model H:\models\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --out build-local/mm27-new-lifecycle
```

The lifecycle driver retains `engine.exe` and backend source snapshots with its
report so a failing executable is not lost on the next build. The focused reload
test uses a synthetic 4K context, a fresh native oracle, a resident-weight oracle,
and nine cycles alternating cache off, separate CUDA allocations and arena.
It compares logits after every prefill batch and decode step, changes the cache
working set, and fills/frees GPU allocations between model lifetimes. It also
checks replay against the original oracle, not just against the preceding run.

```powershell
python tools/check_minimax_m2_reload.py --out build-local/mm27-new-reload --check-mm27-06-baseline
```

The optional historical hash check is for the pinned synthetic fixture and the
measured strict RTX 5090 baseline. Omit it for other hardware; the per-run exact
comparisons still apply. This driver retains its executable too. An advanced
`reload-check NEW_DIRECTORY --precision-fingerprints` diagnostic deliberately
changes numerical modes; its report is marked `DIAGNOSTIC`, not a correctness PASS.

Use `--cases fixture`, `--cases full_pressure`, or `--cases context2048 context4096`
to select part of the suite. The real pressure case approaches 90% global RAM and
VRAM using a separate read-only GGUF mapping and disposable GPU buffers, then
releases both. Injected availability ceilings only reduce real availability;
the benchmark/pipe does not expose these test controls. Long contexts use a
repeated prefix and 33 varied final tokens to exercise late positions without
repeatedly reading all experts. This is a numerical stress test, not a measure
of long-answer quality. The fixture uses the same test sequence.

For one request, save UTF-8 JSON with either an already-rendered `prompt` string
or a `tokens` array, plus `max_tokens` (1..256). The embedded MiniMax template can
be rendered with `tools/minimax_m2_template.py`; the runtime does not add role
delimiters or BOS to an already-rendered prompt. A JSON array runs multiple fresh
requests in one loaded process.

```powershell
python tools/run_minimax_m2.py -- --gguf MODEL.gguf --request request.json --output report.json --logits logits.f32
python tools/run_minimax_m2.py -- --gguf MODEL.gguf --pipe
python tools/run_minimax_m2.py -- --gguf MODEL.gguf --request request.json --gpu-cache-mib 18432
python tools/run_minimax_m2.py --engine build-local/minimax-m2-cuda/bin/strata-minimax-m2-cache-check.exe -- build-local/mm27-new-cache-fixture
python tools/check_minimax_m2_cache.py --model MODEL.gguf --out build-local/mm27-new-cache-ab --cache-mib 18432 --repeats 3 --tokens 24
python tools/run_minimax_m2.py -- --gguf MODEL.gguf --request request.json --gpu-cache-mib 18432 --gpu-cache-allocator arena
python tools/run_minimax_m2.py --engine build-local/minimax-m2-cuda/bin/strata-minimax-m2-arena-check.exe -- build-local/mm27-new-arena-check.json
python tools/check_minimax_m2_cache.py --model MODEL.gguf --out build-local/mm27-new-allocator-ab --comparison allocator --cache-mib 18432 --repeats 3 --tokens 24
python tools/check_minimax_m2_cache.py --model MODEL.gguf --out build-local/mm27-new-reader-ab --comparison reader --candidate-reader mmap-decode --cache-mib 18432 --repeats 3 --tokens 24
python tools/run_minimax_m2.py -- --gguf MODEL.gguf --request request.json --gpu-cache-mib 18432 --gpu-cache-allocator arena --expert-reader mmap-decode
python tools/check_minimax_m2_lifecycle.py --model MODEL.gguf --out build-local/mm27-new-arena-lifecycle --gpu-cache-mib 18432 --gpu-cache-allocator arena
python tools/check_minimax_m2_pipeline.py --out build-local/mm27-new-pipeline-fixtures
python tools/check_minimax_m2_pipeline.py --out build-local/mm27-new-lookahead-fixtures --lookahead
python tools/check_minimax_m2_pipeline.py --out build-local/mm27-new-d2d-fixtures --lookahead --d2d-batch
python tools/check_minimax_m2_cache.py --model MODEL.gguf --out build-local/mm27-new-d2d-ab --comparison d2d --pipeline-readers 2 --pipeline-chunk-mib 4 --cache-mib 18432 --repeats 3 --tokens 24
python tools/check_minimax_m2_cache.py --model MODEL.gguf --out build-local/mm27-new-lookahead-ab --comparison lookahead --pipeline-readers 2 --pipeline-chunk-mib 4 --cache-mib 18432 --repeats 3 --tokens 24
python tools/check_minimax_m2_cache.py --model MODEL.gguf --out build-local/mm27-new-pipeline-ab --comparison pipeline --pipeline-readers 2 --pipeline-chunk-mib 4 --cache-mib 18432 --repeats 3 --tokens 24
python tools/check_minimax_m2_lifecycle.py --model MODEL.gguf --out build-local/mm27-new-pipeline-lifecycle --gpu-cache-mib 18432 --gpu-cache-allocator arena --pipeline-readers 2 --pipeline-chunk-mib 4
```

`--pipe` reads one JSON request per line and emits `ready`, token ID events and a
final `result`, or `error`. Each valid request clears KV. It has no conversation
state, tools/reasoning parser or OpenAI API adapter. Ctrl+C sets the cancellation
flag; native lifecycle tests cover cancellation after selected copies during
both full-model prefill and decode, followed by fresh replay. The test uses the
runtime cancellation flag; it does not certify OS keyboard signal delivery.

Defaults: context 512, batch/ubatch 8, greedy sampling, eight generated tokens,
file mode 2. Context accepts 256..4096 and batch 1..16; only measured combinations
are certified by the linked reports. `--mode 1` is the native mmap copy reference
and can retain substantially more mapped pages in its process working set.
Stop ID is the export's EOS/PAD 200020; native FIM/reponame EOG aliases do not stop
this experimental conversation loop. `max_tokens` counts reasoning and EOS.

The report separates load, prefill/first-token, decode and total request timing.
For N generated tokens there are N−1 timed decode forwards; the first token
comes from prefill. In mode 2, `source_bytes` counts host bytes delivered to GPU, with
`file_bytes` and `mmap_bytes` separating ReadFile from mapped access. None of
these counters measures physical SSD traffic. For `mmap-direct`, the H2D timer
also includes page faults and driver staging; source time is zero. Memory samples are taken at
load and decode boundaries and are not continuous peak measurements. Logit files
contain little-endian F32, one vocabulary-sized row per generated token.

For file mode 2, `selected_bytes` counts scheduler requests, `cache_hit_bytes`
counts bytes served from the GPU, and `cache_guard_bytes` counts additional
following bytes uploaded for complete cache entries. Thus
`h2d_bytes + cache_hit_bytes = selected_bytes + cache_guard_bytes` for successful
requests. `cache_fill_bytes` measures scratch-to-cache D2D copies separately.
With the pipeline enabled, `pipeline_h2d_bytes` also counts uploaded lookahead
that is abandoned on failure; `pipeline_unused_bytes` reports that suffix.
Successful requests have `pipeline_h2d_bytes = pipeline_d2d_bytes = h2d_bytes`,
no unused bytes, and zero queued/reader-owned payload after every tensor plan.
The fixed pinned/device capacities are separate from logical queue occupancy.
`pipeline_plans` includes all-hit plans; `pipeline_matrices` counts their selected
tensors, `pipeline_plan_peak` is the largest plan and `pipeline_lookahead_plans`
counts plans with more than one tensor. These are distinct from transport groups,
which count only plans with misses. Completed requests have no pending plan.
`pipeline_copy_fences` counts host waits for delivery/hit/fill copies, excluding
plan completion, cleanup and producer ring events. `pipeline_copy_submissions`
counts logical range transfers and cache copies (a range can span ring chunks).
In batched mode, successful requests have one `pipeline_copy_batches`,
`pipeline_copy_fences` and `pipeline_scratch_fences` per selected tensor.
`pipeline_pending_fills_peak` counts pinned new fills, bounded by 256 per tensor;
`pipeline_abort_fences` counts exceptional batch drains. Plan setup may perform
an additional backend synchronization; it is not a scratch-fence counter.
`pipeline_read_us`, `pipeline_wait_us`, `pipeline_slot_wait_us` and
`pipeline_submit_us` are CPU time sums; they may overlap and are not GPU
durations. `pipeline_delivery_ms` includes waiting for the producer and, when
batching is disabled, per-upload D2D completion. With batching enabled it excludes
the final tensor fence; compare end-to-end request/decode time for performance.
Legacy `source_ms`/`h2d_ms` measure only the synchronous path.
The A/B script alternates process order and compares every logit and token ID;
its first request has an empty GPU cache, followed by a repeat and a longer
request on a different topic. OS file-cache state is not controlled.

Measured results and remaining work:
[implementation status](../../docs/minimax-m2.7/MINIMAX_M27_IMPLEMENTATION_STATUS.md).
MM27-06 full-model allocator A/B and cached pressure/2K/4K comparisons passed
bit-for-bit. Two earlier tiny native-after-cache reload comparisons differed.
MM27-07 replayed reconstructed tests without the extra instrumentation and ran
the stronger reload suite successfully. Numerical fingerprints did not match
the old discrepancy. Its cause remains open; allocator/cache defaults are unchanged.

MM27-08 added bounded mmap readers. Three full-model A/B pairs found faster
repeated-request decode with `mmap-decode`, but slower first and changed-topic
decode. All logits matched bit-for-bit; pressure, cancellation, reload and
2K/4K tests passed. This is an experimental option, not a general speedup.
The default reader remains `file`; see the status for complete timings.

MM27-09 added a tensor-scoped native-file pipeline. Three alternating full-model
A/B pairs with two readers and 4 MiB slots improved decode by 22.9..32.2%.
Both variants explicitly wait for cache D2D completion; the old native reload
discrepancy remains unresolved. Pipeline defaults to off. This stage overlaps
read/H2D/ring delivery, not model compute; see the status for latency, memory,
CUDA event trace and the completed correctness/pressure/context checks.

MM27-10 adds opt-in three-matrix router lookahead with unchanged ring capacities.
Three alternating full-model A/B pairs improved median decode by 17.7..31.4%
over tensor mode, with bit-exact logits. The diagnostic CUDA event trace recorded
0.775 ms of H2D/compute interval overlap in four plans;
this is not a kernel-activity profile or a complete explanation of the speedup.
All 2424 fixture, 47 lifecycle, 55 reload and 49 runtime checks passed. Defaults
remain unchanged; see the status for conditions, memory and remaining work.
