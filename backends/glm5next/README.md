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
On Windows the suite contains eight tests, including the native route planner, range parser, GPU cache
and the unchanged DeepSeek frequency-history regression through its compatibility include.
The synthetic GPU test checks 432 matrix transfers
in 18 cases: three mixed gate/up/down groups, eight distinct expert IDs including
first/last, mmap/native/auto modes and prefill/decode reader policies. It uses
4096x2048 matrix geometry (transposed for down), synthetic bytes for all eight
routed types, 262161-byte staging chunks, two consumer streams and destination
guards. Native mode protects the mmap view with PAGE_NOACCESS and removes the
registry entry after planning; queued reads must retain their file handles.
Source/H2D/D2D/chunk counters must match the consumed plan. Auto-mode counters
report the actual native/mmap choice, not physical SSD reads.

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
