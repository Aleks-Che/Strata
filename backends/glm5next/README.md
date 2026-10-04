# GLM candidate build and tokenizer oracle

This is preparation for P0.3/P2.1b/P3, not a Strata inference backend or launcher.
The protocol and native reader tests build without a model, CUDA or llama.cpp. The oracle
target requires the audited Unsloth archive; its full build and token-ID parity
have not yet been checked. Qwen and DeepSeek dependencies/build files are unchanged.

Candidate: `86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9`, as in the
[loader report](../../docs/GLM53/GLM53_FLASH_LOADER_COMPATIBILITY.md).
Download the [exact commit archive](https://github.com/unslothai/llama.cpp/archive/86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9.tar.gz)
and record/review its SHA-256 before configuring. There is no default archive hash
and no download during configure. A matching hash checks archive bytes against the
supplied value; it does not independently authenticate that the supplied archive
is the requested commit. This remains an audit candidate, not a production pin.

From the Strata root, in an initialized compiler environment:

```text
cmake -S backends/glm5next -B build-local/glm5next-protocol -G Ninja -DSTRATA_GLM_PROTOCOL_TESTS_ONLY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-local/glm5next-protocol --config Release
ctest --test-dir build-local/glm5next-protocol -C Release --output-on-failure --no-tests=error
python -m unittest tools.test_glm5next_build
```

On Windows, use the x64 Native Tools environment, including Windows SDK headers,
libraries and `rc.exe`/`mt.exe`. `cl.exe` on PATH alone is insufficient. The checks
were built with MSVC 19.44.35222.0 and Windows SDK 10.0.26100.0. The protocol uses a mock
encoder: passing it says nothing about GLM tokenization or CUDA kernels.

On Windows, this configuration also runs `glm5next_expert_file_test` and
`deepseek4_expert_file_compat_test` against the same
[shared native reader](../common/expert_file.hpp), through its direct and DeepSeek
compatibility includes. Each run checks 384 matrix reads for eight quant layouts
with mixed gate/up/down, guards and partial chunks; 160 concurrent reads after
unmapping and closing the original file; reload identity; offsets above 4 GiB;
short reads, invalid ranges and cancellation before submission. The large-offset
fixture uses a sparse temporary file and requires a filesystem supporting sparse
files. Payloads are synthetic packed bytes, not numerical quantization fixtures.
These tests cover file-to-RAM delivery, not GLM graph integration, H2D or decoding.
The existing DeepSeek CUDA pipeline test remains a separate regression.

For GPU byte transport fixtures (P3.4a), use a separate directory and the local
CUDA toolkit. This builds C++ against the CUDA runtime; it does not build the
Unsloth candidate or compile model kernels with nvcc:

```text
cmake -S backends/glm5next -B build-local/glm5next-transport -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_GLM_PROTOCOL_TESTS_ONLY=ON -DSTRATA_GLM_TRANSPORT_TESTS_CUDA=ON -DCUDAToolkit_ROOT=/path/to/cuda
cmake --build build-local/glm5next-transport --config Release
ctest --test-dir build-local/glm5next-transport -C Release -V --no-tests=error
```

On Windows, initialize the x64 compiler environment above and place the CUDA
runtime DLL directory on PATH. The opt-in GPU test fails if no CUDA device is
available. Default protocol/reader-only builds still need neither CUDA nor GPU.
On Windows the suite contains twelve tests, including the native route planner, transport lifetime, range parser, GPU cache,
global-memory reader and DeepSeek compatibility checks (frequency history, VRAM policy and memory sampling).
The synthetic GPU test checks 432 matrix transfers
in 18 cases: three mixed gate/up/down groups, eight distinct expert IDs including
first/last, mmap/native/auto modes and prefill/decode reader policies. It uses
4096x2048 matrix geometry (transposed for down), synthetic bytes for all eight
routed types, 262161-byte staging chunks, two consumer streams and destination
guards. Native mode protects the mmap view with PAGE_NOACCESS and removes the
registry entry after planning; queued reads must retain their file handles.
Source/H2D/D2D/chunk counters must match the consumed plan. Auto-mode counters
report the actual native/mmap choice, not physical SSD reads.

P3.6a adds shared wait/staging telemetry, also available through
`ExpertTransport::counters()`: split reader slot waits and consumer publication
waits, fixed pinned/ring capacities, current/peak reader-owned and queued payload
bytes, and abandoned bytes. See [field definitions](../common/README.md).
The CUDA fixtures validate allocation bounds, empty plans, full ring, short tail,
cancel/restart and the preserved legacy wait total. Allocation-size overflow is
rejected before allocation. These are CPU timers and byte accounting; no new
CUDA timing events or graph synchronization were added. GLM INFO/monitor and GPU
execution timelines still require runtime integration. Passing these checks does
not establish an inference throughput change.

P3.7a adds an explicit warm transport benchmark to the range checker. It uses
real selected GGUF ranges, no inference or cache. For example, with the CUDA
runtime DLL directory on PATH:

```text
python -m tools.benchmark_glm5next_transfer --gguf H:/GLM-5.3-Flash-GGUF/GLM-5.3-Flash-Uncensored-IQ3_XXS.gguf --checker build-local/glm5next-transport/strata-glm5next-transfer-check.exe --output docs/GLM53/GLM53_FLASH_IQ3_XXS_TRANSPORT_BENCHMARK.json
```

Default selection is layers 3/11/45, experts 0–6/287, all three projections:
72 matrices. Each of mmap/native/auto × 1/2/4 readers × prefill/decode runs in
a fresh process, in seeded shuffled order. Defaults are 1 MiB chunks, two warmup
passes and five samples. All selected mapped bytes are first compared against
independent stdio reads. Every pass checks complete GPU payloads and inter-matrix
guards outside the timer. The measured region includes pipeline start, ordered
transfers, finish and final consumer-stream synchronization, with no per-matrix
sync. Allocations, warmup, destination initialization and D2H validation are
excluded. Destination storage is capped at 512 MiB; host expected/actual buffers
each match that storage. Four pinned and four device-ring slots are additional.

The report contains raw samples, CPU wait counters, exact source/H2D/D2D bytes,
configuration, binary/header hashes and file sizes/mtimes. Decode mmap/auto still
uses one active reader for any configured reader count; native decode honors
the count. `effective_readers` states that existing policy. Source counters do
not measure physical disk I/O. This test deliberately warms only selected ranges;
it does not establish whole-model RAM residency, compute overlap or token rate.
The parser accepts `--benchmark READERS DECODE WARMUPS REPEATS` with the existing
plain range manifest on stdin, rejecting cache/dispatch manifests.

On Windows / Ryzen 9 9950X / RTX 5090 / CUDA 13.0 (2026-10-04), 18 cases per
profile passed with 9072 matrix comparisons each. For the 227 MiB IQ3_XXS
selection, mmap prefill medians for 1/2/4 readers were 18.301/17.380/17.582 ms;
mmap decode with one configured/active reader was 18.029 ms. For the 294.5 MiB
UD-Q3_K_XL selection, mmap prefill medians were 23.230/23.011/23.159 ms.
Decode mmap/auto differences are small and overlap sample ranges. These are
component observations, not new runtime defaults. Reports:
[IQ3_XXS](../../docs/GLM53/GLM53_FLASH_IQ3_XXS_TRANSPORT_BENCHMARK.json),
[UD-Q3_K_XL](../../docs/GLM53/GLM53_FLASH_UD_Q3_K_XL_TRANSPORT_BENCHMARK.json).
Protocol checks: `python -m unittest tools.test_glm5next_transfer_benchmark`.
Cold storage, representative routed workloads, chunk-size sweeps and integrated
GLM compute remain necessary before choosing production reader settings.

`expert_plan.hpp` provides the native route-to-range planner (P3.1b), without CUDA,
llama.cpp or payload reads. The loader supplies the full model identity and load
generation, main/total/leading-dense block counts, one layer's three routed tensor
layouts and shard sizes/data starts. `plan_experts` validates all three layouts,
quant row geometry, byte counts, tensor overlap, file bounds and transposed down
shape before returning full `ExpertKey` values. The shared key definition now lives
in `expert_key.hpp`; the GPU cache includes it without changing key semantics.

The planner supports the eight routed quant types listed above, rejects dense
layers and invalid IDs, and deduplicates routes in first-use order. Each unique ID
keeps its complete gate/up/down triple, regardless of descriptor input order.
Duplicate IDs may come from a batch; compute still applies each token's original
router weights. This planner only deduplicates weight transfers. Empty routes
produce an empty plan after layout validation. `max_routes` (default 4096) bounds
input IDs, including duplicates, and thus at most three times that many output
keys; larger batches must be split by the caller. Offsets are 64-bit, including
offsets above 4 GiB and a final expert ending exactly at EOF. An unused metadata-only
shard may end before its aligned data start, as in the split GLM profile.

The planner owns only metadata. The graph adapter must resolve returned shard
offsets to valid source views, retain mappings/file owners, protect current-plan
cache leases, and drain the pipeline before releasing sources on all exit paths.
It does not create a GLM graph, execute a router or own CUDA events. The CPU test
checks 384 keys from independent fixture geometry plus malformed layouts/routes.
The GPU byte test now gets its 24-matrix plans from this native planner, with
duplicate router IDs and independent expected offsets; its 432 comparisons remain
synthetic packed bytes, not model logits or a measured speedup.

`expert_memory.hpp` adds an opt-in controller for one combined main/MTP cache per
device (P3.5c). Pass the configured byte cap, the shared `StrataVramPolicy`, and a
probe returning global free/total bytes for the cache's CUDA device. GLM accepts
configured-byte mode 0 and total-device-usage mode 2; matrix-count mode 1 is not
a byte limit for mixed quants. Reserve validation (minimum 128 MiB) and limit
arithmetic match DeepSeek. Other allocations, including fixed weights, state,
ring, workspace and other processes, count against available device memory.

Construct the controller and call `refresh()` at dispatch boundaries, after
allocating those buffers. The controller caps the cache by both the configured
limit and the reserve-aware byte limit. It trims idle entries using the cache's
existing events/pins/leases; retired allocations remain charged. Pending work
can defer trimming. `Status` reports the sampled free/total, target, resident and
deferred bytes, sample validity, trim result and sample/failure counts. It is a
snapshot at refresh, not a live memory measurement or an atomic reservation.

P3.5f handles memory lost between that sample and cache allocation: only
`cudaErrorMemoryAllocation` returned by the cache matrix allocator produces an
empty lease and increments both `bypasses` and `allocation_bypasses`. No upload
is submitted for that entry; `ExpertDispatch` delivers the complete matrix via
its existing uncached transport into the already allocated destination. Planned
hits remain pinned. Victims evicted to make room before allocation are not restored.
Later misses can allocate normally; there is no automatic retry or budget change.
Other allocation errors, event failures and upload exceptions still propagate.
An expected CUDA last error is cleared only if it is `cudaErrorMemoryAllocation`.

The optional third cache constructor argument is an allocation callback for
failure injection; successful pointers must be compatible with `cudaFree`, and
failure must leave no allocation/work behind. Default construction uses
`cudaMalloc`. CUDA fixtures inject OOM and non-OOM allocation results while
checking real device bytes, accounting, source lifetime, pins, recovery and
fatal-error cancellation/restart. Dispatch checks cover mmap/native/auto,
LRU/frequency and prefill/decode, including guards and exact source/H2D totals.
These tests do not exhaust physical VRAM or verify recovery of a full GLM graph.

Admissions pause until the first valid sample. A failed, inconsistent or throwing
probe leaves existing hits available but prevents new cache allocations/evictions
for misses; dispatch still transports those weights through bypass. The cache's
`paused_bypasses` counter distinguishes this case. A valid refresh resumes normal
admission; a controller's destructor does not silently resume after probe failure.
Returning to manual control requires explicit `set_budget` and
`set_admission_enabled(true)` between dispatch scopes. The cache must outlive the
controller and neither may be controlled concurrently or from another device.

The three-argument controller constructor now supplies an owning live probe from
`global_memory.hpp` (P3.5d). On Windows it uses the shared PCI-matched NVML reader,
as DeepSeek does, and reports failure without CUDA per-process fallback. The
four-argument constructor still accepts an injected probe. On non-Windows CUDA
the factory queries the current device with `cudaMemGetInfo`; this path has not
been tested here. Tests inject memory snapshots while allocating/checking real CUDA buffers:
main/MTP pressure, pins, retired leases, pending events, invalid samples, recovery,
total-device targets, and dispatch bypass payloads. These are synthetic component
checks, not pressure from another application, a full GLM run or a speed result.
Sampling frequency and pre-allocation workspace reservations remain runtime work.
Windows reader tests also check missing symbols, failed initialization, PCI/handle
errors, invalid NVML values, recovery after read failure, two simulated devices,
400 serialized concurrent samples and balanced shutdown. Both the direct and
DeepSeek include paths run a live probe/controller smoke test. On RTX 5090 with
CUDA runtime/driver 13000/13000, the real probe returned valid global data and the
controller admitted and verified a 64-byte fixture under a 1 MiB cap (2026-10-04).
These smoke tests accept an unavailable live NVML result only when the outputs
are zero and the controller keeps admissions paused; their log explicitly states
which branch ran. This does not connect the absent GLM inference graph or test
pressure from another application. A reader that failed initialization must be
recreated; read failures re-resolve the PCI handle on the next sample.

An explicit external-process check is available as P3.5e; it is not part of default
CTest. Build the CUDA transport configuration, set the CUDA runtime DLL PATH, then:

```text
python -m tools.check_glm5next_memory --checker build-local/glm5next-transport/strata-glm5next-memory-pressure-check.exe --output docs/GLM53/GLM53_FLASH_EXTERNAL_MEMORY.json
```

The runner launches a controller and a separate CUDA holder on the same PCI GPU.
Both contexts initialize before the controller chooses a total-device usage
target 128 MiB above baseline usage. The cache cap is 64 MiB (two synthetic 32 MiB
main/MTP entries), with 128 MiB reserve. The holder allocates and touches 256 MiB.
This crosses the configured usage target while leaving physical VRAM available;
it does not attempt to cause OOM. The controller must trim the unpinned entry,
retain and verify the pinned entry, then finish trimming when the pin is released.
After the holder frees its allocation, the budget and both entries must recover.
Each policy, LRU and frequency, verifies five full payloads. The checker requires
NVML and fails if unavailable, with an 8-second deadline for observing each
pressure/recovery transition. Every worker command has a runner timeout.

The JSON records PCI identity, CUDA versions, child PIDs, checker/runner hashes,
the actual device usage target and global memory/cache snapshots for all stages.
The runner drains/exits its own children on success and closes their stdin on
failure, killing only a child that fails to exit within three seconds. GPU buffers
are freed on normal EOF/exit, including early cancellation. Both policies passed
on RTX 5090, CUDA runtime/driver 13000/13000 (2026-10-04). This is real external
allocation with synthetic cache payloads, not GLM inference, full memory exhaustion,
speed measurement or control of allocations racing between two memory samples.
Other active applications can prevent the expected target transitions and make
the check fail. Runner protocol/cleanup tests: `python -m unittest tools.test_glm5next_memory_check`.

`expert_transport.hpp` adds an owning transport adapter (P3.3a). Construct one
`ExpertTransport` per host owner/device, then `begin(plan, sources, decode)` with
native `ExpertKey` ranges and full offset-zero `ExpertSourceView` mappings. Each
view carries model/generation/shard identity and a shared owner of its mapping
and native registration; a file handle alone does not preserve mmap memory.
All keys, source identities, ranges and duplicates validate before submission.
Plans are bounded to 12288 matrices and 1048576 staging chunks. Source mappings
referenced by the plan are retained internally, so the caller can drop its own
references after `begin`. No payload copies occur during validation.

Call `transfer(index, destination, stream)` in increasing index order. Bad order,
null destinations and nested `begin` calls fail without consuming more jobs.
`finish()` requires that every planned matrix was consumed; if not, it first drains
and cancels the remainder, then reports the incomplete plan. `cancel()` explicitly
discards that remainder and is idempotent. Both retire source owners after the
common pipeline finishes reading. Destruction (including exception unwinding)
joins pipeline workers and drains its GPU ring before destroying source owners.
A pipeline failure destroys that pipeline before releasing mappings; recreate the
adapter after such an error. Validation mistakes do not poison an idle adapter.

The adapter reuses the shared pipeline's CUDA slot events. `finish`/`cancel` do
not promise completed destination consumers: callers retain destination memory
and streams until their work finishes, and retain any associated cache leases.
Calls, including cancellation, must be serialized on one host owner; a server
cancellation signal must be handled there between transfers. Native read cleanup
may wait for in-flight reads, so cancellation is not a bounded-latency guarantee.
This adapter does not integrate the model graph, filter cache hits or implement
cache admission. The graph adapter must construct the ordered miss/bypass plan.

Windows CUDA fixtures use native planned triples and real temporary mappings:
cancel after 0/1/7 matrices, complete all 24, restart with another source generation,
check 192 delivered matrices plus untouched suffixes/guards across six modes, and
unwind an exception with a prefetched ring. External mapping owners are dropped
before transfers; weak ownership checks verify retirement. Native mode protects
the mapped pages with PAGE_NOACCESS. These are synthetic byte/lifetime checks;
actual I/O/CUDA failure injection and real GLM request cancellation remain untested.

`expert_dispatch.hpp` joins the planner output, cache and transport (P3.2b.2).
Construct `ExpertDispatch(cache, transport, keys, source_views, decode)`, call
`copy(index, destination, stream)` in plan order, then `finish()`. Cache and
transport must outlive the scope and have one host owner; do not mutate either
externally while the dispatch is active. The scope first pins the resident part
of the entire plan, probes residency without counting accesses, and gives only
the ordered misses to `ExpertTransport`. Every actual access uses `get()` exactly
once. An admitted miss fills the cache through transport, a hit copies cached bytes,
and a bypass transfers all selected bytes directly to the destination. Consumer
leases for hits and newly loaded entries remain held through the plan. All-hit
plans can omit source views and must produce no source/H2D work.

`cancel()`, incomplete `finish()` and exception destruction drain the remaining
miss plan before releasing leases, pins and source views. Queue destination users
on their supplied streams and keep destination memory/streams alive until those
users complete. Cache-to-destination D2D copies are not included in the pipeline's
D2D counter; it counts ring-to-cache/bypass transfers only. An unexpected residency
change is an error and cancels the scope; this is a diagnostic for external
invalidation, not support for simultaneous reload. Duplicate keys and nested
scopes are rejected. Source/layout validation remains the loader/planner's job;
the transport additionally checks binding and ranges for misses. The adapter
copies weights to caller destinations; it does not execute a GLM graph or expose
cache pointers for compute in place.

Windows CUDA fixtures exercise mixed hit/admitted-miss/bypass plans, all-hit plans,
zero cache budget, frequency rejection, cancellation and exception cleanup with
native planned ranges, mixed quants and guarded destinations. They run mmap/native/
auto with LRU and frequency policies; dispatch fixtures use prefill reader policy.
Existing transport fixtures separately test decode. No dispatch speedup, real
GGUF dispatch, numerical logits or model graph integration has been measured.

On non-Windows the GPU test has only the six host-memory cases (144 matrix
comparisons); native file transport is not implemented there. Only Windows was
run for P3.4a. Neither synthetic byte parity nor the DeepSeek pipeline regression
establishes GLM dequantization, graph correctness, model output or throughput.

`expert_cache.hpp` implements the isolated GPU cache component (P3.2b.1).
`ExpertKey` carries every field of the Python reference key: complete model
identity, load generation, main/MTP, layer, expert, projection, quant type, shape,
shard, offset and length. The loader must supply a fresh generation on each reload
and validate the tensor layout; the cache validates key structure, not quant math.
The component is not yet connected to a GLM loader/router/graph.

Before processing any misses from a routed plan, call `cache.protect_plan(keys)`
and keep the returned `PlanPins` alive for the plan (P3.5b). It pins every currently
resident key, including uploads still in flight, so an early miss cannot evict a
later planned hit. Duplicate keys are pinned once; absent keys are not allocated.
All keys validate first, and the input is bounded to 12288 matrix keys. Protection
does not count an access, update frequency/LRU, enqueue CUDA work or wait for upload
readiness. `size()` reports the number of unique entries actually pinned.

Pins expose no data pointers. Actual consumers still need a normal `get()` lease,
which orders upload readiness and records consumer events on release. Keep leases
for newly loaded matrices through the rest of the plan too; those matrices were
absent when pins were taken. End/cancel the plan by destroying the pins or calling
their idempotent `release()`. Multiple guards may overlap. Explicit invalidation
removes entries from lookup but retains their bytes/source owners while pinned;
new generations cannot exceed the shared byte budget. Pins can outlive the cache
object. Retiring the last reference to an invalidated allocation may wait for its
GPU work, like cache teardown. This protects residency only; it is not a snapshot
or permission to obtain old-generation data after reload. The model loader must
cancel/rebuild plans when changing generations. Use the same single host owner as
the cache. CUDA fixtures cover early miss/later hit, both admission policies,
overlapping guards, budget trim, invalidation, teardown and exception cancellation.

Use one host owner and one CUDA device per cache, including its leases. A miss
requires a shared source owner and an upload callback; the callback can invoke
`StrataExpertPipeline::transfer` and must order all writes onto the supplied
stream. It must not reenter the cache. A hit waits on upload readiness and does
not invoke the callback. Hold a `Lease` for every matrix needed by the current
plan. Queue its consumers on the lease's stream before release or destruction;
obtain a separate lease for another stream. Release records a CUDA event, so
eviction cannot free bytes still consumed by the GPU. Source ownership remains
with the allocation until retirement. A failed upload is drained and never
published as a cache hit.

LRU eviction considers only unleased entries whose upload and consumer events
have completed. If none can be evicted within the byte budget, `get` returns an
empty lease without uploading; the caller must use its uncached transfer path.
`set_budget` may return false while live leases/events prevent trimming; retry
after consumers finish. Reported resident bytes include invalidated allocations
still held by leases. The budget covers matrix allocations, not allocator/event
overhead or other model memory. A future VRAM controller must supply this limit.

`invalidate(model, generation)` removes only that identity/generation. Released
entries may require a blocking drain here; active leases remain valid, including
after the cache object is destroyed. Counters report hits/misses, bypasses,
evictions and invalidations. Allocation reuse is not implemented yet.
The CUDA test covers key separation, byte parity, LRU/budgets,
pending uploads/consumers, source lifetime, reload, upload failure and an upload
through the shared pipeline followed by a hit with no second H2D. Its payloads
are synthetic; no GLM cache speedup or real model cache parity has been measured.

P3.5a adds opt-in frequency admission: `ExpertCache(limit, {true, 4096, 131072})`.
The original one-argument constructor keeps LRU behavior without a history table.
The shared `backends/common/expert_frequency.hpp` retains DeepSeek's pointer-key
API; GLM uses all twelve `ExpertKey` fields. Counts include hits and valid misses,
even bypassed matrices. Values saturate at 255 and halve each decay period of
accesses, including for untouched keys. At the configured key limit, an unseen key
clears the metadata history; residency is unchanged. The limit counts keys, not
host bytes (GLM identities contain strings). One clock covers both main and MTP,
whose counts are separate. These defaults are not tuned on GLM inference.

With free capacity a miss is admitted. Under pressure, its score must be at least
that of every idle LRU victim needed for its byte size; ties admit. All victims
are planned before any eviction, so a frequency rejection or insufficient idle
space cannot partially empty the cache. Active leases and pending CUDA events
remain protected. A rejection returns an empty lease without calling the uploader;
the caller must still deliver that selected matrix through the uncached pipeline.
Explicit budget trimming ignores admission scores. Model/generation invalidation
also clears matching nonresident history, preserving other identities.
`admissions` counts successful inserts; `admission_rejects` counts frequency
rejections and is a subset of `bypasses`. Budget/event bypasses are not frequency
rejections. CUDA fixtures check decay, repeated-miss promotion, bounded history,
mixed-size atomic eviction, pending events and full uncached pipeline delivery.

For read-only byte checks of actual GGUF ranges (P3.4b), the Windows CUDA-test
configuration also builds `strata-glm5next-transfer-check.exe`. Run the inspector
and range planner through this wrapper from the repository root:

```text
python -m tools.check_glm5next_transfer --gguf H:/GLM-5.3-Flash-GGUF/GLM-5.3-Flash-Uncensored-IQ3_XXS.gguf --checker build-local/glm5next-transport/strata-glm5next-transfer-check.exe --layers 3 11 45 --experts 0 1 2 3 4 5 6 287 --modes mmap native auto --chunk-bytes 262161 --timeout 120 --output docs/GLM53/GLM53_FLASH_IQ3_XXS_GPU_TRANSFER.json
```

Keep the CUDA runtime DLL on PATH as above. For UD-Q3_K_XL, pass its first shard
and a separate report. These layers cover all eight routed types across the two
local profiles. Eight IDs select 72 matrices per model, including main and MTP
weights; deduplication is performed by the existing Python planner. Reported
offsets belong only to the inspected model, never reuse them across profiles.

The checker reads the range manifest on stdin, maps source files read-only,
uses the shared CUDA pipeline and compares every returned byte (plus destination
guards) with an independent `fseek`/`fread` baseline. It bounds matrix/chunk sizes,
range count and chunk metadata, and rejects ranges outside the actual source file.
Files and mappings live through completion. `--test-parser` runs without a GPU.
The runner validates every indexed result and all transport counters; failures or
timeouts replace an older success report with `status=error` and return exit 1.
The JSON records selected matrices, header hashes, binary hash, source sizes and
mtime, GPU/runtime and counters. Source or executable changes during the check
are rejected. Header hashes are not hashes of all weights; they and binary hashes
record identity, not independent authenticity. Source files remain unchanged.

Each process uses prefill-style slices and a fresh mapping. Native/mmap counters
describe the source access path, not physical SSD traffic. This checker serializes
comparison after each matrix and is not a speed benchmark, full GLM model run,
numerical quantization test or MTP execution. Without `--cache-check` it does not
exercise the cache. Python runner contracts:
`python -m unittest tools.test_glm5next_transfer_check`.

For isolated cache parity on the same real ranges (P3.4c), add `--cache-check` and
use a separate output, for example `docs/GLM53/GLM53_FLASH_IQ3_XXS_GPU_CACHE.json`.
The wrapper sends a `GLM_CACHE_RANGES_V1` manifest with the complete matrix key.
Its model identity hashes the observed shard paths, sizes, mtimes and header
hashes. This separates the checked file sets but is neither a full payload hash
nor a production loader/session fingerprint. The default range protocol remains
unchanged. Cached reports use schema version 2 and record the identity and stages.

Each range gets a fresh LRU cache with exactly that matrix's byte budget. Five
stages are compared against the independent stdio baseline: cold generation 1,
a hit on another CUDA stream, generation 2 insertion forcing eviction, generation
1 reloading and evicting generation 2, then explicit invalidation of generation 1
and insertion of generation 3. Generations simulate loader keys; the checker does
not reload a model or remap files. Every stage copies cached bytes into a guarded
verification destination. Misses use the shared pipeline and retained mapping
owner; hits must not invoke the uploader, read source bytes or add H2D traffic.
Per-stage deltas and per-range cache counters are checked in C++; Python requires
all ordered range results and exact aggregate counters. Pipeline D2D counters
exclude the additional cache-to-verification copies.

For 72 ranges each mode must report 360 byte comparisons, 72 hits, 288 misses,
144 evictions and 72 invalidations, with no bypass. Source/H2D bytes and chunks
must equal four passes over the ranges. No timings are used as a benchmark:
stages synchronize for comparison. This checks selected packed weights and cache
lifetime, not dequantization, mixed-range cache planning, frequency admission on
real weights, GLM graph integration, numerical logits or MTP execution.

To check the native `ExpertDispatch` scope on the selected real GGUF ranges, use
`--dispatch-check` instead of `--cache-check`:

```text
python -m tools.check_glm5next_transfer --gguf H:/GLM-5.3-Flash-GGUF/GLM-5.3-Flash-Uncensored-IQ3_XXS.gguf --checker build-local/glm5next-transport/strata-glm5next-transfer-check.exe --dispatch-check --layers 3 11 45 --experts 0 1 2 3 4 5 6 287 --modes mmap native auto --chunk-bytes 262161 --timeout 120 --output docs/GLM53/GLM53_FLASH_IQ3_XXS_GPU_DISPATCH.json
```

`GLM_DISPATCH_RANGES_V1` carries the same complete keys as the cache manifest,
requires distinct triples and exercises LRU/frequency with prefill/decode reader
policies. Each consecutive gate/up/down triple provides A/C/D; B aliases C's bytes
under generation 2. A cache of A+C bytes first holds A/B. The next plan C/A/D must
evict B, preserve the later hit A, and deliver D through bypass. An all-hit C/A plan
has no source views or source/H2D traffic. Cancellation after A in an A/D plan must
release pins and leases; invalidation, generation 3 loading and zero-budget bypass
then reuse the same transport. Alternating streams and guarded destinations are
compared with independent stdio reads, with synchronization after each matrix.

There are 14 byte comparisons per triple/policy pair: 24 triples × 2 admission
policies × 2 reader policies × 3 I/O modes = 4032 comparisons per model. Both local
profiles passed on RTX 5090, CUDA runtime/driver 13000/13000 (2026-10-04).
Reports preserve file/header identities, checker hash, actual transfer counters
and scenario results. Source/H2D counters include cancelled read-ahead; per-phase
assertions require exact bytes for completed plans and zero for all-hit plans.
Auto prefill may read resident pages through mmap; auto decode uses mmap. Native
reads are not evidence of physical SSD I/O. Ring D2D excludes cache-to-destination
copies. This is a serialized packed-byte check; generation changes are simulated
keys, not model reload. It does not validate dequantization, numerical GLM outputs,
graph cancellation, frequency rejection on real weights, or inference throughput.

For the real oracle, use a **different** build directory. Substitute the verified
local archive and its recorded hash in this command (not executed here):

```text
cmake -S backends/glm5next -B build-local/glm5next-oracle -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_GLM_ARCHIVE=/path/to/commit.tar.gz -DSTRATA_GLM_ARCHIVE_SHA256=<reviewed-64-digit-hash> -DSTRATA_GLM_CUDA=OFF
cmake --build build-local/glm5next-oracle --target strata-glm5next-tokenizer
build-local/glm5next-oracle/bin/strata-glm5next-tokenizer --version
build-local/glm5next-oracle/bin/strata-glm5next-tokenizer --gguf H:/GLM-5.3-Flash-GGUF/GLM-5.3-Flash-Uncensored-IQ3_XXS.gguf
```

The executable has `.exe` on Windows; multi-config generators may add `Release/`.
Source extraction and upstream generated files stay under this private build tree.
`glm-source-build.txt` records the requested revision, checked archive hash,
compiler/CUDA settings and that this harness applies no source patches. It is
configure evidence, not proof of a successful build or an unmodified extracted tree.
Use a fresh build tree for a reproducible run. `STRATA_GLM_CUDA=ON` compiles candidate
CUDA support, but the oracle itself always uses `vocab_only=true`, `n_gpu_layers=0`
and `load_mtp=false`; it never creates an inference context or computes layers.
Future inference must separately preserve the initial `NVIDIA_TF32_OVERRIDE=0`
and flash-attention-off correctness settings from the implementation plan.

The oracle uses the [candidate's public llama API](https://github.com/unslothai/llama.cpp/blob/86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9/include/llama.h),
validates `general.architecture=glm5next` and `tokenizer.ggml.pre=glm4`, and emits:

```text
READY_TOKENIZER
```

Send `ENC 0 <hex>` or `ENC 1 <hex>` with UTF-8 bytes encoded as hex, one request
per line. `1` enables special-token parsing; both modes use `add_special=false`.
An omitted hex field encodes empty text. Responses are `IDS` followed by decimal
token IDs, or one `ERR` line; malformed requests do not end the process. `QUIT`
ends it. LF and CRLF are accepted. Text is limited to 1 MiB per request. Diagnostics
from model loading go to stderr. `--version` prints the requested source revision
and archive hash without loading a model.

After building against the actual candidate archive, run from the Strata root:

```text
python -m tools.check_glm5next_tokenizer --gguf H:/GLM-5.3-Flash-GGUF/GLM-5.3-Flash-Uncensored-IQ3_XXS.gguf --oracle /path/to/strata-glm5next-tokenizer --archive-sha256 <reviewed-64-digit-hash> --output docs/GLM53/GLM53_FLASH_IQ3_XXS_TOKENIZER_PARITY.json --timeout 60
```

The first test model is now the single-file Uncensored-IQ3_XXS GGUF above.
Its actual routed tensor types are IQ2_S/IQ3_S/IQ4_XS and Q2_K/Q3_K for MTP;
the filename is not a tensor encoding contract. The earlier four-part UD-Q3_K_XL
model remains a separate comparison profile. Use its first shard and a separate
output report when testing it. Never reuse token/state validation results solely
because both files report the same architecture.

The runner verifies the oracle's `--version` revision/hash against the candidate
and the supplied archive hash before loading the GGUF. These fields are reported
by the executable; they do not independently authenticate its build. The report
also records the binary hash, GGUF header hash and embedded-template hash.

The shared corpus in `tools/glm5next_tokenizer_corpus.py` contains 18 plain inputs
and 18 prompts rendered from the actual GGUF template: multi-turn conversation,
two tool calls with reversed result order, and a new user turn after tools, with
low/high/max effort and both clear_thinking values. Both tokenizers receive each
text with parse_special off/on, for 72 comparisons. This tests token-ID parity on
shared inputs; it is not an independent comparison of template rendering.

Exit 0 requires every ID sequence to match. Exit 1 indicates a mismatch or an
execution/protocol/provenance error; the JSON status distinguishes `pass`, `fail`
and `error`. For comparisons the JSON stores input text, both ID lists and the
first differing index. An error replaces any previous success report. No real
candidate parity result exists yet. Runner tests use a scripted subprocess:
`python -m unittest tools.test_glm5next_oracle_check`.
