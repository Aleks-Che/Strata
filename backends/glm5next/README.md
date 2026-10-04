# GLM candidate build and tokenizer oracle

This is preparation for P0.3/P2.1b/P3.1a/P3.4, not a Strata inference backend or launcher.
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
On Windows the suite contains five tests, including the real-range manifest parser.
The synthetic GPU test checks 432 matrix transfers
in 18 cases: three mixed gate/up/down groups, eight distinct expert IDs including
first/last, mmap/native/auto modes and prefill/decode reader policies. It uses
4096x2048 matrix geometry (transposed for down), synthetic bytes for all eight
routed types, 262161-byte staging chunks, two consumer streams and destination
guards. Native mode protects the mmap view with PAGE_NOACCESS and removes the
registry entry after planning; queued reads must retain their file handles.
Source/H2D/D2D/chunk counters must match the consumed plan. Auto-mode counters
report the actual native/mmap choice, not physical SSD reads.

On non-Windows the GPU test has only the six host-memory cases (144 matrix
comparisons); native file transport is not implemented there. Only Windows was
run for P3.4a. Neither synthetic byte parity nor the DeepSeek pipeline regression
establishes GLM dequantization, graph correctness, model output or throughput.

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
numerical quantization test, cache check or MTP execution. Python runner contracts:
`python -m unittest tools.test_glm5next_transfer_check`.

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
