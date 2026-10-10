# MiniMax-M2.7 experimental backend

This standalone Windows/CUDA backend admits the reviewed single-file
`minimax-m2` Q4_K_M checkpoint. It has no server integration or model profile yet.
The synthetic fixture is admitted only by the test executable.

The ordinary benchmark and JSONL pipe accept optional request-local sampling:

```json
{"prompt":"<already rendered prompt>","max_tokens":1024,"sampling":{"temperature":1.0,"top_p":0.95,"top_k":40,"seed":42}}
```

Omitting `sampling` preserves greedy selection (`temperature:0`). Temperature
must be 0 or 0.01..2; top_p is in (0,1], top_k is an integer in 0..200064
(0 disables its filter), and seed is an integer in 0..4294967294. Defaults for
the other fields are top_p 0.95, top_k 40 and seed 42. Unknown sampling keys,
wrong types and non-finite values are rejected before prefill. Effective F32
settings are returned in `sampling`. Top-k/top-p are ignored in greedy mode.

The sampling chain is temperature → top-k → top-p → seeded distribution,
using the pinned llama.cpp CPU sampler on a copy of the logits. Each request
starts a fresh RNG, including after invalid requests or cancellation. A fixed
seed is reproducible for the tested build and platform; sequences are not
promised across dependency/compiler changes. No penalties or grammar are
implemented. Sampling does not change the raw logits, EOS policy, context
budget, CUDA model arithmetic or cache settings. The separate DFlash driver
remains greedy-only.

`sampling_ms` counts token selection and filtering. It is included in
`request_ms` and excluded from the forward-only decode rate. CPU checks and
saved-logit replay use `strata-minimax-m2-sampling-check`; full-model and JSONL
checks use `tools/check_minimax_m2_sampling.py` and
`tools/check_minimax_m2_sampling_pipe.py`. See
[sampling results and commands](../../docs/minimax-m2.7/MINIMAX_M27_SAMPLING.md).

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
cmake --build build-local/minimax-m2-cuda --target strata-minimax-m2-runtime-check strata-minimax-m2-lifecycle-check strata-minimax-m2-cache-check strata-minimax-m2-arena-check strata-minimax-m2-reclaim-check strata-minimax-m2-reload-check strata-minimax-m2-bench -j 6
```

Runtime patches are generated in this build directory, guarded by original
source hashes. Other backends and the extracted candidate sources are untouched.

## Run and verify

Use the launcher: it supplies CUDA DLL directories and the numerical settings
**before process start**. Setting these inside `main()` produced cross-batch
numerical failures on the test PC; the runtime now rejects missing settings.

```powershell
python tools/run_minimax_m2.py --engine build-local/minimax-m2-cuda/bin/strata-minimax-m2-runtime-check.exe -- build-local/mm27-new-fixture-run
python tools/check_minimax_m2_runtime.py --model H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --out build-local/mm27-new-full-run
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
python tools/check_minimax_m2_lifecycle.py --model H:\models\MiniMax-M2.7\MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q4_K_M.gguf --out build-local/mm27-new-lifecycle
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
or a `tokens` array, plus integer `max_tokens` (default 8, at least 1 and no
larger than the space left after prompt tokens in the context). The embedded MiniMax template can
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
runtime cancellation flag. MM27-18 also tested targeted Windows CTRL_BREAK
after 257 streamed tokens and a full 414-token replay in the same process;
physical keyboard Ctrl+C delivery was not tested.

Defaults: context 512, batch/ubatch 8, greedy sampling, eight generated tokens,
file mode 2. Context accepts 256..4096 and batch 1..16; only measured combinations
are certified by the linked reports. `--mode 1` is the native mmap copy reference
and can retain substantially more mapped pages in its process working set.
Stop ID is the export's EOS/PAD 200020; native FIM/reponame EOG aliases do not stop
this experimental conversation loop. `max_tokens` counts reasoning and EOS.
Results report the requested `max_tokens` and `stop_token_id` (200020 for EOS,
null for a length limit). Raw text and token events retain the terminal EOS;
an adapter must consume that token by ID before emitting user-facing text.
The former 256-token ceiling is removed; context remains limited to 4096.
SIGINT, and on Windows a targeted CTRL_BREAK/SIGBREAK, cancel active work.
Cancellation is checked at token boundaries as well as inside expert delivery;
the pipe returns an error, clears KV, and accepts a fresh request.

`serve/minimax_m2.py` supplies an unregistered reasoning/content parser for
the template's already-open reasoning block. It preserves whitespace and
partial text at a length limit or cancellation; `reasoning_complete` indicates
whether the closing marker arrived. Callers must decode token bytes with a
stateful UTF-8 decoder and handle EOS by token ID. Tool-like text stays literal;
tool parsing, history normalization and the HTTP adapter are still pending.
MM27-18's English response reached EOS at 414 tokens with an exact repeat;
the Russian list-sum request repeated inside reasoning through its 1536-token
limit. The complete-corpus gate remains failed. See
[completion checks](../../docs/minimax-m2.7/MINIMAX_M27_COMPLETION.md).

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

`tools/tune_minimax_m2_pipeline.py` varies cache caps, reader count and chunk
size on the same strict MM27-11 pipeline. Supply a JSON list with distinct names:

```json
[{"name":"baseline","cache_mib":18432,"readers":2,"chunk_mib":4}]
```

```powershell
python tools/tune_minimax_m2_pipeline.py --model MODEL.gguf --configs configs.json --out build-local/mm27-new-tuning --repeats 3 --tokens 24
```

It reverses configuration order on alternating repeats, retains the executable,
sources, raw logits and commands, and checks exact parity, copy accounting and
global memory limits. Optional `--reference-logits` and `--reference-report`
compare against a retained run of the same corpus and output-token limit;
otherwise the first run is the reference. Summaries include actual cache limits,
residency, hit bytes, TTFT and total request time. One screening pass is not a
confirmed performance ranking. Use repeated comparisons for promising settings;
`--tokens 256` checks longer generation, which may stop earlier on EOS.

MM27-12 screened cache caps 12/16/18/20/24 GiB, one/two readers and 4/8/16 MiB
chunks on the RTX 5090. In three confirmation pairs, cap20/two readers/chunk8
improved median decode only 0.74..1.41% over cap18/two readers/chunk4, while
median total workload time increased from 77.70 to 81.02 seconds. A near-limit
arena lost useful residency during one candidate run. The experimental
18 GiB/two readers/4 MiB setting is retained; MM27-13 below addresses reclamation.
Generic defaults remain unchanged. See the status for effective budgets and raw reports.

MM27-13 adds physical block reclamation to the MiniMax arena. Under VRAM
pressure, the cache releases whole unpinned blocks, preferring the least live
payload and then the oldest most recent access. One plan or pending-fill pin
protects the entire block. If pinned blocks prevent reclaim, the existing error
path drains the pipeline before cleanup. The sampled budget is preserved;
refresh inside cache admission uses the same policy. Step/Hy3 keep their
previous cache policy: the shared helper's group callback is opt-in.

`cache_pressure_trims`, `cache_pressure_groups`, `cache_pressure_evicted_bytes`
and `cache_pressure_released_bytes` report pressure-specific work separately
from ordinary expert replacement. In a controlled four-block CUDA fixture,
lowering the budget from 256 to 255 MiB with interleaved LRU entries discarded
all 256 MiB with the old algorithm, versus one 64 MiB block with the new one.
This demonstrates reclamation efficiency, not a full-model throughput result.

Three full-model old/new pairs at cap20/two readers/chunk8 on this RTX 5090
measured median decode 3.392→3.421, 4.382→4.663 and 4.148→4.177 tokens/s for
first/repeated/changed-topic requests. Median total workload was 80.513→77.432 s.
The repeated request gained 6.41%; first/new-topic differences were under 1%.
All token IDs and full logits matched bit-for-bit. OS cache/external pressure
were uncontrolled, and these pairs do not compare different cache caps.
All 4920 pipeline, 47 lifecycle and 55 reload checks passed; the historical
MM27-06 discrepancy remains open. See the status for individual measurements.

```powershell
python tools/run_minimax_m2.py --engine build-local/minimax-m2-cuda/bin/strata-minimax-m2-reclaim-check.exe -- build-local/mm27-new-reclaim.json
python tools/check_minimax_m2_reclaim.py --model MODEL.gguf --baseline-dir build-local/minimax-m2-reclaim-baseline --out build-local/mm27-new-reclaim-ab
```

The A/B driver needs a retained old `engine.exe`, its `backends/` source snapshot
and a `manifest.json` mapping relative paths to SHA-256 hashes. It verifies those
hashes before running three alternating old/new pairs with identical settings
(default cap20, two readers, chunk8). It preserves both binaries and checks
full logit bits, token IDs, copy accounting, bounded buffers and sampled global
RAM/VRAM. Use a new output directory for every run; run GPU tests sequentially.

MiniMax also accepts `--arena-block-mib 8|16|32|64` (default 64). This is the
target physical allocation size, separate from `--pipeline-chunk-mib`.
Selected expert matrices are still copied individually. A matrix larger than
the target receives one larger allocation; a smaller cache cap can reduce the
block. The existing 2 MiB allocator rounding and global memory/commit guards
remain in effect. Hy3's default block size remains 64 MiB.

For a block-size sweep, add optional `"arena_block_mib":16` to each tuning
configuration. `check_lifecycle` and the cache fixture accept the same CLI flag.
Keep cache cap, readers and pipeline chunk fixed when comparing block sizes.
Physical size classes currently pack equal-sized matrix slots together;
semantic grouping by layer/expert is a separate placement policy, not enabled
by this option. See MM27-14 in the status for measurements and grouping design.

MM27-14 checked all four sizes. A 256→255 MiB controlled budget change retained
248/240/224/192 MiB for 8/16/32/64 MiB blocks, with exact surviving bytes.
Smaller blocks nevertheless caused frequent allocation/reclamation near the
VRAM limit in the full model. Three cap20/two-reader/chunk8 pairs measured
64→32 MiB median decode 3.409→3.103, 4.638→4.255 and 4.208→3.912 tokens/s;
median workload time was 78.251→79.423 seconds. Default 64 MiB is retained.
All 1872 fixture and 37 lifecycle checks passed, as did full logit/token parity.
These timings do not compare smaller blocks at the recommended 18 GiB cap.
The option above changes physical block size only; MM27-16 separately adds
placement of an expert's three matrices together (see below).

`--arena-growth-reserve-mib N` (0..1024, default 0) adds a global VRAM reserve
only when the arena needs a new physical block. Existing slots can be filled or
reused inside that reserve, and the cache still shrinks at its original memory
threshold. This leaves a band between growth and shrinking to avoid repeatedly
allocating and freeing blocks when available VRAM fluctuates. The configured
cache cap, host commit guard and global 95% guard remain unchanged. The option
has no effect with the `cuda` allocator. The tuning JSON accepts optional
`"arena_growth_reserve_mib":32`; cache/lifecycle checks accept the CLI flag too.

MM27-15's controlled 8 MiB block test alternates a 255/256 MiB budget 20 times.
With reserve 0 it performs 20 frees and 20 new allocations; with reserve 32 MiB
it performs none, retaining 224 MiB. Surviving bytes match exactly. It can grow
back to the full 256 MiB cap when global headroom returns, and still releases
blocks under larger pressure. See the status for full-model timing and gates.

`--cache-group-experts 0|1` (default 0) enables MiniMax expert triplets. It
requires a nonzero arena cache. Each `(generation, layer, expert)` owns one
slot containing gate/up/down; replacement removes all three, and a pin on any
component protects the whole slot. Components become readable only after their
copy completion fence. New groups remain pinned until the router plan ends.
Admission and frequency training happen once per group per plan, preventing
late admission at the second or third matrix after the first was bypassed.

Use the measured combination with `--pipeline-readers 2 --pipeline-chunk-mib 4
--pipeline-lookahead 1 --pipeline-d2d-batch 1`. Complete triplets were checked
at request boundaries with three-matrix lookahead. Other paths can leave safe
partially populated groups and were not benchmarked as a fast profile. GGUF
layout and transport ring sizes are unchanged; three source ranges still need
their own transfers. The tuning JSON accepts `"group_experts":true`; the
cache and lifecycle checkers accept the CLI flag.

`cache_expert_groups`, `cache_partial_expert_groups`, `cache_ready_matrices`,
`cache_pending_matrices`, `cache_ready_bytes`, `cache_group_admissions` and
`cache_group_plan_pins_peak` expose completeness and ownership. Hits/misses
still count matrix lookups; evictions/reuses count entire groups in this mode.
`cache_pressure_groups` continues to count physical arena blocks.

MM27-16 passed 488 GPU/cache/lifecycle checks, including actual full-model
pressure at sampled global RAM 89.95% and VRAM 94.07%. Six full-model runs
matched the retained pre-change executable bit-for-bit (576 tokens and
115,236,864 logits). Three alternating cap18 pairs measured median aggregate
forward decode 3.912→3.930 tokens/s, with mixed per-request results. Median
three-request time was 86.935→86.154 seconds; a repeatable speed advantage was
not established. Default grouping remains off. These timings use the ordinary
benchmark's forward-only decode interval, not DFlash useful-token timing.
See [group cache measurements](../../docs/minimax-m2.7/MINIMAX_M27_GROUP_CACHE.md)
for packing density, individual runs, commands, hashes and retained failures.

`--ram-cache-mib N` (default 0, maximum 131072) adds a MiniMax-only LRU of
read-only GGUF views to the file pipeline. It requires pipeline readers and
the `file` reader; GPU grouping remains optional. This is not a heap copy of
the weights. The initial cap is limited by physical RAM headroom; subsequent
pressure can shrink it. Accounting includes the allocation-granularity prefix
and page rounding, with at most 65536 entries. Host commit headroom is checked
for mapping metadata. The existing process working-set target is 94%, with
an independent global 95% RAM/VRAM guard. Windows can reclaim these clean pages;
a view-cache hit does not guarantee that every page is physically resident.

Prefill can reuse existing entries but does not admit new views. Decode reads
may map missing chunks after an observed GPU admission refusal on an earlier
access. Successful GPU fills therefore avoid creating RAM mappings on their first access. This
bounded history is cleared with the model and on subsequent GPU admission.
RAM-only configurations do not require a GPU refusal. When the GPU cache finishes a matrix fill, overlapping
RAM views are retired. Views used by readers remain alive until the last copy
finishes. A later GPU miss can map that range again. Admission refusal falls
back to the original `ReadFile` path. Read errors propagate after draining;
they are not hidden as successful cache misses. Model/context replacement and
pipeline abort clear views, and release restores the process working-set limit.

`ram_cache_bytes/budget/cap` measure charged views, not physical residency.
`ram_cache_mapped_bytes/file_bytes` split successful source copies between
mapped reads and native-file fallback; `ram_cache_hits` counts existing-view
reuse. Admissions, evictions, GPU drops, prefill bypasses, active readers and
peak charged bytes are reported separately. The older `file_bytes` continues
to count bytes delivered through the file pipeline, including this managed
source reader; it is not a physical SSD traffic counter. The tuning JSON uses
`"ram_cache_mib":65536`; cache/lifecycle checkers accept the CLI flag too.
`ram_cache_gpu_waits/history_entries` expose deferred admission and its bounded
metadata (at most 65536 ranges).

MM27-17 passed 816 unit/pipeline/lifecycle checks and full-logit comparisons
for 1344 generated tokens, including three 256-token continuations. A first
eager-mapping prototype was slow; deferring admission reduced transient views.
Three final short-corpus pairs nevertheless measured median aggregate decode
3.941 tokens/s off versus 3.485 with RAM64, and workload time 85.944 versus
89.556 seconds. RAM64 reached 55.859 GiB of charged views in the long probe;
that probe has no paired current off timing. Default RAM cache remains 0.
See [RAM cache results](../../docs/minimax-m2.7/MINIMAX_M27_RAM_CACHE.md)
for the scope, retained reports and commands.

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

MiniMax weights now reside in `H:\models\MiniMax-M2.7`. This directory also
contains DFlash Q3_K_M/Q4_K_M/Q5_K_M sidecars. Static inventory and draft hashes
are recorded in [DFlash inspection](../../docs/minimax-m2.7/MINIMAX_M27_DFLASH.md).
The isolated runtime now compacts the borrowed 200064-row head output to the
draft's 200055-token vocabulary. No shared weight is copied or changed.
`strata-minimax-m2-dflash_loader` validates registration without weight allocation;
`strata-minimax-m2-dflash-check` exercises real draft blocks and KV rollback.
Run them with `tools/check_minimax_m2_dflash.py` (commands in the DFlash document).
Q3/Q4/Q5 passed 30 GPU checks each, including every output row against the full
head, borrowed MASK embeddings, feature capture and replay. These diagnostic
probes are not speed benchmarks. The probe's Q3_K/Q5_K weights use the dependency's usual quant
arithmetic; the strict quant F32 patch covers Q4_K/Q6_K only.

`strata-minimax-m2-speculative-bench` now provides a separate offline greedy
DFlash driver. `tools/bench_minimax_m2_dflash.py` compares complete target logits
and token IDs before admitting a run to its useful-speed summaries. It retains
EXE/source snapshots, full logits, memory samples and draft/verify/catch-up costs.
See [DFlash benchmark](../../docs/minimax-m2.7/MINIMAX_M27_DFLASH_BENCHMARK.md)
for commands, the Q3/Q4/Q5 depth1/2/4/7 screen and repeated comparisons.

Batched target arithmetic changed a near-tied eighth routed expert on one
prompt. Opt-in `--verify-tokenwise 3` splits ordinary (including attention)
and routed matmuls by token during target verification; model weights, delivery
pipeline and tolerances are unchanged. Modes1/2 are diagnostics; mode0 preserves
the existing default. The final screen matches serial logits bit-for-bit.
This does not resolve the separate historical MM27-06 reload discrepancy.
Normal bench/API speculation stays disabled. GPU EOS/cancel/pressure/recovery,
long-context/session lifecycle and sampling remain separate gates.

MM27-22 connects the resident JSONL benchmark to the existing OpenAI/Anthropic
HTTP service through `serve.minimax_m2_engine`. Start the isolated experimental
profile with `python -m serve.minimax_m2_server`; see
[native API instructions](../../docs/minimax-m2.7/MINIMAX_M27_NATIVE_API.md)
for the model path, extra dependency, explicit cache/pipeline flags and checks.
The native Windows cancellation handler now uses `SetConsoleCtrlHandler`:
MSVC resets a `signal(SIGBREAK, ...)` handler after its first delivery, which
terminated the resident process on a second HTTP cancellation. The failed run
and old binary are retained. This entry point does not register MiniMax in the
main installer and does not enable DFlash, sessions or alternate precision.

[MM27-23 live tools and web checks](../../docs/minimax-m2.7/MINIMAX_M27_LIVE_TOOLS_WEB.md)
cover two generated calls, reversed results, tool errors across both APIs and
browser history/repeated Stop. The adapter canonicalizes tool-definition field
order; the web client omits the session header when health advertises no support.
HTTP pressure/4K passed in [MM27-24](../../docs/minimax-m2.7/MINIMAX_M27_HTTP_CONTEXT.md).
Broad answer quality and installer registration remain open.

MM27-25 adds optional resident KV prefix reuse with native `--prefix-cache 1`
or server `--prefix-cache`. HTTP clients must send `X-Strata-Session-Id`.
Only complete prefill batches from that session are reused; the last prompt
batch is evaluated again. A different session or an anonymous request clears
the previous prefix. Native error/cancel and restart invalidate it. There is
one resident prefix, no session archive or cache-admin capability. Defaults
remain off. See [contract and validation](../../docs/minimax-m2.7/MINIMAX_M27_PREFIX_CACHE.md).

[MM27-26](../../docs/minimax-m2.7/MINIMAX_M27_PREFIX_CONTEXT.md) extends the
prefix checks to 2032/4080-token prompts, full EOS414, seeded sampling128 and
one-token outputs: 113436288 compared logits are bit-identical. The live 4K
pressure run checks both APIs, expert arena trim, cancellation and fresh recovery.
Shift and GPU parity at other batch sizes remain open. Runtime, admission SHA
and defaults were unchanged by this audit.

MM27-27 adds opt-in RAM checkpoints for inactive sessions:
`--prefix-cache --session-cache-mib 600 --session-cache-slots 2` on the server
(native uses `--prefix-cache 1`). The RAM cap includes blob/token metadata;
the active GPU sequence is separate. Oldest inactive entries are evicted first,
with the requested checkpoint protected. Restore consumes its checkpoint;
cancel/error discards partial active KV while keeping completed other sessions.
Restart/unload clears everything. No on-disk format or cache-admin API is added.
Copy/restore time is included in TTFT and request time. Default archive cap is0.
See [session archive](../../docs/minimax-m2.7/MINIMAX_M27_SESSION_ARCHIVE.md)
for the full-logit regression, HTTP lifecycle evidence and remaining checks.

[MM27-28](../../docs/minimax-m2.7/MINIMAX_M27_SESSION_CONTEXT.md) verifies RAM
restore at2K/4K, EOS414 and seeded sampling128:1150 output IDs and230073600
logits match the saved reference;567 IDs/113436288 logits also match fresh
requests in the same process. Nineteen live pressure scenarios pass with
RAM94.191%/VRAM94.430%, protected restore after rejected snapshot admission,
128MiB arena trim and cancellation/recovery. Only the diagnostic holder gains
bounded private RAM topup; the engine binary and defaults are unchanged.

[MM27-29](../../docs/minimax-m2.7/MINIMAX_M27_STATE_BULK.md) coalesces host KV
state transfers with at most16MiB CPU scratch. The snapshot format is unchanged;
restore preserves gaps and other sequences. Sparse ranges keep the old path.
`STRATA_MM27_STATE_BULK=0` disables coalescing for diagnostics; default is on.
1200 CPU cases/12 CUDA checks,192 output IDs/38412288 full-model logits and10
live HTTP/lifecycle scenarios pass. Three warm switches per target give median
request2K7.941→5.277s and4K8.024→3.608s; this is session latency, not sustained
decode throughput. The admitted EXE is98e85e80…; archive default remains0.
[MM27-30](../../docs/minimax-m2.7/MINIMAX_M27_STATE_BULK_PRESSURE.md) repeats
the complete offline corpus on the bulk binary:1150 IDs/230073600 logits match
the saved reference bit-for-bit. All19 live pressure scenarios pass, with
RAM94.197%/VRAM94.403%,64MiB arena trim, rejected4K snapshot admission,
protected2K restore and both cancellation/recovery paths. An independent
Windows/NVML observer records physical RAM, commit and global VRAM every0.5s;
all recorded samples stay within95%. Eighteen CPU methods and100 final gates
pass. Runtime, holder and defaults are unchanged; this is not a new speed A/B.
[MM27-31](../../docs/minimax-m2.7/MINIMAX_M27_SESSION_BATCHES.md) verifies
GPU batch1/8 at context1024 with prompts up to513 tokens. All50 prefix/archive
scenarios match fresh at the same batch size:568 IDs and113636352 full logits.
Coverage includes branches, shorter/extended prompts, RAM restores, one-token
outputs and seeded sampling64. Four sequential native processes finish cleanly;
14 CPU methods and59 final gates pass. Runtime and defaults are unchanged.
Next: GPU batch1/8 prefix/archive at2K/4K, then shift. Sparse-copy tuning,
context>4K and broader quality remain open. Long EOS/pressure evidence remains
specific to the previously tested batch16 profile.

MM27-32 adds native `--cache-decay-period N` (1..4294967295, default65536).
It controls the number of trained expert-matrix lookups between halvings of
admission scores. Prefill/workspace warmup does not train this history. Request
counter resets preserve it. Other backends keep their existing65536 default.
The value is fixed when the GPU cache is configured and appears in the JSON
header. With grouping off,65536/131072/262144 correspond to about44/88/176
trained decode steps on this62-layer/top8 model.

This is a native diagnostic setting; the server admission SHA remains98e85e80….
The retained candidate is `build-local/minimax-m2-cache-decay-candidate/engine.exe`;
pass it explicitly to `tools/run_minimax_m2.py --engine ... -- ...` to test the
new flag without replacing the admitted server executable. See
[cache decay](../../docs/minimax-m2.7/MINIMAX_M27_CACHE_DECAY.md) for the full-logit
comparison, repeated-process timings and the boundary/admission fixture.
Three runs per period give4.120/4.171/4.183 tokens/s at65536/131072/262144 on
the four-request128-token corpus. Longer histories slow the new-topic request
by2.79/6.87% while helping return to the old topic; default65536 is retained.
All4608 candidate IDs and921894912 logits match the old EXE bit-for-bit;
19 history checks,216 cache regressions and285 final evidence gates pass.

MM27-33 adds native `--pipeline-events 0|1|2`, default0. It requires file
pipeline readers and D2D batching. Mode1 replaces matrix scratch/copy CPU
fences with CUDA stream event dependencies. Mode2 also uses async split
compute; the public graph boundary drains each CUDA backend on every exit.
Pending cache fills remain pinned until copy retirement, before the next
matrix's cache decisions. Other scheduler inputs retain upstream waits.
Strict F32 operators still contain internal CPU waits; mode2 `compute_ms`
measures CPU submission, as identified by `compute_ms_scope`.

The three-pair A/B on RTX5090, context2048/batch16, cache18GiB/readers2/chunk4,
four fresh-KV128-token requests, gives **4.086→4.266 tokens/s (+4.40%)**.
Full request time falls164.030→157.808s; all pairs improve4.07–4.40%.
Mode1 alone did not improve the screening aggregate. All4608 IDs and921894912
logits match the retained synchronous EXE exactly, with identical cache
decisions.1256 CUDA fixture checks,6 CPU methods,8 CLI cases and357 evidence
gates pass. Those initial checks cover the short corpus; MM27-34 below adds
full-model context/pressure checks and server admission.

Use `tools/run_minimax_m2.py --engine build-local/minimax-m2-copy-events-candidate/engine.exe -- ...`
and append `--pipeline-events 2` to the experimental18/2/4/lookahead/D2D profile.
Candidate SHA13a885cc… is retained separately; admitted server EXE98e85e80…
and native default0 are preserved. See
[matrix events](../../docs/minimax-m2.7/MINIMAX_M27_COPY_EVENTS.md) for raw
artifacts, reproduction, exact timings and the remaining gates.

MM27-34 validates that same events EXE at context4096/batch16 with prefix
and RAM archive:1150 IDs/230073600 logits match the prior reference exactly;
six fresh/restore pairs also match. All19 live pressure scenarios pass,
including17 HTTP answers/662 IDs, EOS414, seeded sampling128, physical RAM
admission rejection,128MiB GPU trim and both cancellations/recovery.
Sampled global peaks: RAM94.199%, VRAM94.451%. Six real CLI/auth/unload/reload
checks,51 CPU methods and111 final evidence gates pass.

The experimental server now accepts that exact13a885cc… binary with explicit
`--pipeline-events 2`; append
`--engine build-local/minimax-m2-copy-events-candidate/engine.exe --pipeline-events 2`
to its18/2/4/batch16 launch command. The legacy98e85e80… executable still
works with mode0; default0 and the main bench binary are preserved. See
[context and pressure](../../docs/minimax-m2.7/MINIMAX_M27_COPY_EVENTS_CONTEXT.md)
for the launch command, memory measurement timing, raw artifacts and scope.
No new speed A/B was performed in MM27-34. GPU batch1/8 at2K/4K, shift,
context>4K and quality/oracle gates remain open.

MM27-35 adds native `--router-host-ids 0|1` (default0). The strict Q4_K/Q6_K
CUDA path can consume an owned snapshot of the IDs already fetched by the
scheduler. It requires exact node/storage/layout identity and a single split
lifetime; mismatches, tokenwise slices and eval callbacks use the original D2H.
The permutation-table H2D fence and graph-exit synchronization remain.

Three full-model pairs on RTX5090/RAM125.555GiB, context2048/batch16 and the
same18/2/4/events2 profile: median decode4.292→4.458tok/s (+3.87%), all pairs
+2.56–6.74%. All3072 IDs/614596608 F32 logits match the retained reference;
transport/cache decisions are identical.97 router-ID checks,230 cache/fault/
reload checks,12 Python methods and4 CLI cases pass. Candidate612354a1… is
retained in `build-local/minimax-m2-router-ids-candidate`; legacy98e85e80… and
accepted events13a885cc… binaries are preserved. New server admission requires
long-context/session/real-pressure validation. See
[host router IDs](../../docs/minimax-m2.7/MINIMAX_M27_ROUTER_IDS.md).

Short full-model batch1/8 pairs also match exactly (32 IDs/6402048 logits).
Final evidence:695 gates/163 hashes pass;2144 independent memory samples,
RAM22.213%/VRAM84.406% peaks, all10 native processes exited. This preserves
cache18GiB for an equal comparison; it is not a full-model pressure test.

MM27-36 validates that same router EXE at context4096/batch16 with prefix and
RAM archive:1150 IDs/230073600 logits match the old reference; six restore
pairs also match. All19 live pressure scenarios pass (17 HTTP answers/662 IDs,
EOS414, sampling128,128MiB GPU trim, protected2K restore after4K admission
rejection, both cancellations/recovery). Peaks:RAM94.180%/VRAM94.430%.
Six real CLI/auth/unload/reload checks,60 CPU methods and116 evidence gates
pass. All owned processes exited.

The experimental server accepts exact612354a1… with boolean `--router-host-ids`
and `--pipeline-events 2` (readers required). Append
`--engine build-local/minimax-m2-router-ids-candidate/engine.exe --pipeline-events 2 --router-host-ids`
to its18/2/4/batch16 profile. Legacy98e85e80… and events13a885cc… remain admitted;
defaults stay off. See [context and pressure](../../docs/minimax-m2.7/MINIMAX_M27_ROUTER_IDS_CONTEXT.md)
for the validated session profile and evidence. No new speed A/B was performed.
The next candidate is permutation-table H2D synchronization with bounded host
buffer lifetime; batch1/8 at2K/4K, shift and quality/oracle gates remain open.


MM27-37 implements native `--sort-table-async 0|1` (default 0). Four pinned
4096-byte host slots retain each permutation table until its CUDA copy event
completes. Oversized tables fall back; graph-exit and failure cleanup drain the
sources. Arithmetic, the device pool, CUDA graph policy and expert copies stay
unchanged. The candidate is `build-local/minimax-m2-sort-table-candidate/engine.exe`
(SHA `692f2e76d97192c3357d209d2013d92b63735607e6c4eeaeac42b31c71a839ff`).

Three pairs on the same RTX 5090/Q4_K_M A,A,B,A x128 corpus, router IDs on:
median 4.465481 → 4.482868 tokens/s (+0.3894%); pair changes +0.4450%,
-0.1769%, +0.1561%. This does not establish a useful speedup. The native option
stays off and this EXE is not admitted by the server. There were 97,464 uploads
per workload, 16 KiB pinned memory and zero slot waits/fallbacks. All 3,072 IDs
and 614,596,608 logits matched the retained reference. Short batch 1/8 pairs
added 32 exact IDs and 6,402,048 compared logits. 106 native, 236 cache/fault/
reload checks, 17 Python methods and 4 CLI checks passed; the final validator
passed 903 gates and retained 178 hashes. Real long/session/pressure validation
for this candidate is deferred until a useful gain is demonstrated.

During the short batch checks, system-wide physical reads fell to 38–78 MiB
while engine source reads were 286–418 GiB. Those counters include every disk,
other processes and model loading; they do not isolate H: or decode. Next:
profile file-cache → pinned staging → H2D/ring/D2D with separate prefill/decode
windows and a bounded CUDA timeline on the admitted router EXE. See the
[MM27-37 report](../../docs/minimax-m2.7/MINIMAX_M27_SORT_TABLE.md) and
[recomputed evidence](../../docs/minimax-m2.7/MINIMAX_M27_SORT_TABLE_CHECK.json).
