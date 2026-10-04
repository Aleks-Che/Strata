# Shared expert transport

`expert_file.hpp` is the native file-reading layer used by DeepSeek and prepared
for GLM's byte-range planner. It has no llama, ggml or CUDA dependency. The old
DeepSeek include forwards here, so its loader hook and pipeline use this code.

- `add` registers a complete, offset-zero mapping and retains a separate file
  handle. Duplicate or overlapping mappings are rejected. `remove` prevents new
  lookups; existing `shared_ptr<Source>` owners keep the handle alive.
- `Request::read` translates a checked mapped address to a file offset.
  `read_at` takes the explicit 64-bit offset from a validated range plan. It does
  not dereference the mapping and can run after the original mapping is closed.
- Use one `Request` per reader thread. Hold the source and destination until the
  call returns. Native chunks are limited to `MAXDWORD` bytes. Short reads throw;
  cancellation returns false. An in-flight cancellation is drained before return.
- A native source handle does not keep mmap pages alive. Any path that copies
  mmap memory must separately retain that mapping. This layer does not own CUDA
  events, staging buffers, model generations, or cache entries.

Native I/O remains Windows-only. Non-Windows registration is still a no-op and
the existing pipeline falls back to mmap; POSIX native reads were not added.

The Windows tests live in `test_expert_file.cpp` and are built by the isolated
[GLM test configuration](../glm5next/README.md). They use real temporary files
with synthetic bytes; they do not require a model or GPU.
