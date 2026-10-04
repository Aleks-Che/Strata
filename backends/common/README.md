# Shared expert transport

`expert_frequency.hpp` supplies bounded, saturating, lazily decayed admission
history without CUDA or model dependencies. `StrataExpertFrequency` preserves
DeepSeek's pointer-key behavior; `StrataExpertFrequencyHistory<Key, Hash>` supports
GLM's full model/generation/matrix keys and selective history invalidation. The
history neither owns weights nor chooses eviction victims. Cache callers record
accesses and compare scores; see [GLM admission rules](../glm5next/README.md).

`expert_pipeline.hpp` provides the existing bounded four-slot CUDA pipeline:
mapped/native reads -> pinned staging -> independent H2D stream -> consumer D2D.
It now depends on `expert_slice.hpp`, `expert_file.hpp` and the CUDA runtime,
without a llama/ggml dependency. DeepSeek's old header forwards to this module;
its backend API still uses the same `StrataExpertSlice` layout. The move does not
change scheduling, counters, events or cancellation behavior.

`start` prepares source ranges; `transfer` consumes them in order, using CUDA
events to protect staging/ring slots across consumer streams. Call `finish` on
every graph exit before releasing mappings, including when a planned suffix was
not consumed. The caller must serialize start/transfer/finish; the internal reader
threads provide concurrency, not support for multiple graph owners. `finish`
drains abandoned H2D work, but consumed D2D work can still be in flight: retain
destinations until their consumer streams complete. This module does not own
expert cache entries or supply GLM router/graph hooks.

`counters()` returns a locked snapshot. Byte counters, high-water marks and CPU
wall-time sums accumulate for the pipeline's lifetime, across start/finish/restart:

| Field | Meaning |
|---|---|
| `slot_wait_us` | Reader CPU time waiting for a slot's previous CUDA consumer event; includes call overhead |
| `consumer_wait_us` | Consumer CPU time waiting for the next H2D submission to be published; includes condition-variable overhead |
| `wait_us` | Legacy sum of those two waits, preserved for DeepSeek |
| `read_us` / `submit_us` | Existing CPU read/copy-attempt time / H2D enqueue time, including copy-stream lock contention |
| `pinned_bytes` / `device_ring_bytes` | Fixed four-slot host/device allocations, each `slots * chunk_bytes` |
| `reader_owned_bytes` / `reader_owned_peak` | Current/peak logical payload bytes assigned to readers, including slot waits and H2D submission |
| `queued_bytes` / `queued_peak` | Current/peak published H2D payload bytes awaiting consumption, including copies not yet complete |
| `unused_bytes` | Cumulative published payload bytes abandoned at finish/cancel, complementing `unused` chunks |

After successful finish, current reader-owned and queued bytes are zero. Peaks
and fixed capacities remain. Slot/event and allocator overhead, source mappings,
cache and destination allocations are excluded. The fixed capacities describe
allocated staging, while logical payload peaks describe queue occupancy; they
must not be added as independent allocations. `reader_owned_bytes` can describe a
new job waiting for a still-running consumer of that slot. CPU wait sums may
overlap across threads and are not elapsed wall time, H2D/compute execution times,
or proof of GPU overlap. CUDA timeline measurement remains a separate task.

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

The optional `STRATA_GLM_TRANSPORT_TESTS_CUDA` configuration also exercises this
shared pipeline on GPU with eight GLM packed layouts. See the same GLM build
instructions. This is byte transport validation; it does not execute GLM kernels.

`vram_policy.hpp` shares the existing `StrataVramPolicy` byte-limit arithmetic
between DeepSeek and the GLM cache controller. The DeepSeek header forwards here;
its modes, validation and arithmetic are unchanged. The GLM test configuration
runs the original `test_vram_policy.cpp` through that compatibility include.
This header does not sample the GPU. Callers must supply trustworthy global
free/total memory and account for all resident cache allocations, including
retired allocations retained by consumers. See GLM's `expert_memory.hpp` for
refresh and failed-sample admission handling.

`device_memory.hpp` shares the Windows CUDA-to-NVML reader with DeepSeek and GLM.
It loads `nvml.dll` from System32, maps the CUDA device's PCI bus ID to an NVML
handle, and reports global free/total bytes. There is no per-process CUDA fallback
on WDDM. Sampling and handle lookup are serialized. Read failures or invalid
values discard the handle for rebinding; failed samples clear both output values.
Initialization failures require recreating the reader to retry initialization.
`diagnostic()` identifies the last stage and API error code. Destroying a reader
balances its successful NVML initialization and unloads its DLL. No NVML SDK is
needed. Injected API functions in tests validate failures and PCI identity without
requiring NVML to fail on the test machine; live smoke checks use the real DLL.
