# GLM candidate build and tokenizer oracle

This is preparation for P0.3/P2.1b, not a Strata inference backend or launcher.
The stdlib protocol test builds without a model, CUDA or llama.cpp. The oracle
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
were built with MSVC 19.44.35222.0 and Windows SDK 10.0.26100.0. They use a mock
encoder: passing them says nothing about GLM tokenization or CUDA kernels.

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
