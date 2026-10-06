# Hy3 admission and oracles

This directory builds CPU vocabulary/template/registration oracles and CUDA
numerical checks for GGUF `hy_v3`. The CUDA checks execute small synthetic main
and MTP graphs. There is no full-model generation engine, installed profile or
HTTP API yet.

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
require exact bytes even with mixed weights. Full-model logits/greedy parity,
Flash Attention, F16 KV, speculative accept/reject integration and speed remain
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

[Plan](../../docs/hy3/HY3_IMPLEMENTATION_PLAN.md) and
[status with measured validation](../../docs/hy3/HY3_IMPLEMENTATION_STATUS.md).
