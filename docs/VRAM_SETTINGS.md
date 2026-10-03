# Live GPU cache settings

The gear button at the top right opens **GPU memory**. Live changes are supported
by the DeepSeek CUDA backend built from this source, including DSpark. The Qwen
engine and older DeepSeek executables show the settings as unavailable.

Choose a mode and click **Apply**:

- **Configured cache budgets** uses the startup `--expert-cache-mib` and
  `--draft-expert-cache-mib` limits. Applying this mode also enables live memory
  monitoring and the free-memory reserve.
- **Manual matrix limit** caps the number of cached matrices across the target
  and DSpark. A matrix is one gate, up or down weight matrix; it is not a complete
  expert. Zero releases the streamed-expert cache. The free-memory reserve can
  reduce residency below the requested count.
- **Automatic · target GPU usage** limits total occupied VRAM on the selected
  GPU, including other applications. For example, a target of 24 GiB leaves less
  room for cached matrices when another application allocates memory. It does
  not promise that Strata alone will occupy 24 GiB.

**Keep free** is an additional reserve in MiB (minimum 128; initial value 1024).
Both the target and the reserve must fit. One GiB is 1024 MiB. The current total
GPU usage, cached matrix count, cache bytes and free memory appear above the form.
The Monitor tab also shows the current matrix count and cache allocation.

The engine samples memory once per second at graph boundaries and while idle.
A Windows build reads global device memory through NVML, matching the CUDA
device by PCI bus ID; `cudaMemGetInfo` alone did not track another process's
allocation in the Windows test. Without a working NVML reading, Windows pauses
cache growth. Linux uses the CUDA memory reading. NVIDIA documents the global
memory fields in its [NVML reference](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html).
A long prompt chunk or other GPU operation can delay a sample and a settings
change until that operation ends. This is not a hard reservation against sudden
allocations by other applications. Dense weights, context/compute buffers and
layers placed with `--gpu-expert-layers` / `--draft-gpu-expert-layers` stay allocated.
If these and other applications exceed the target, all cached matrices are
released and the dialog reports that the target cannot be reached.

The first application of live settings releases the original fixed cache arenas.
Subsequent cache allocations belong to individual matrices. Eviction waits for
pending copies before returning memory to CUDA. Changing count or target then
releases only the least recently used matrices needed to meet the limit. When
memory becomes available, the cache fills on demand; no inference runs just to
fill an idle cache. Growth keeps an extra 64 MiB margin to avoid repeatedly
allocating and freeing at the boundary. Missing telemetry pauses growth. Failed
cache allocations fall back to streaming the affected matrices and are counted
in the dialog. This does not protect unrelated fixed GPU allocations from OOM.

Settings are saved atomically beside the launch configuration as
`<profile>.vram-settings.json`. They apply before the first expert-cache allocation
on the next model load and survive engine restarts. Without a launch configuration,
settings last for the server process. With no saved settings, startup cache
allocation keeps its existing behavior until **Apply** is clicked.

## HTTP and engine protocol

`GET /vram/settings` returns `settings`, capability flag `supported`, `alive`,
`pending`, persistence status and a `live` snapshot. `POST /vram/settings` accepts:

```json
{"mode":"vram","matrices":0,"target_mib":24576,"reserve_mib":1024}
```

Modes are `config`, `count` and `vram`. All three numeric fields are required
integers. `matrices` is 0–1,000,000; `target_mib` is 0–1,048,576 (positive in `vram`
mode); `reserve_mib` is 128–1,048,576. Settings can be submitted during generation.
HTTP 200 means the desired settings were recorded; `pending` stays true until
the engine reports that revision applied. Configured API-key authentication and
the existing same-origin JSON write checks protect the endpoint.

The engine advertises `vram-control` in `READY`, accepts
`VRAM_SET <revision> <mode-number> <matrices> <target-mib> <reserve-mib>` and emits
`VRAM_STATUS <json>`. The stdin reader queues settings, and the inference thread
applies them between graph operations. These messages never enter the token
queue. The `STRATA_VRAM_POLICY` child-process environment variable carries the
same fields without `VRAM_SET` on reload.

## Verification

`python -m unittest serve.test_vram_settings` checks validation, atomic saving,
HTTP access checks, updates during a request, persistence across reloads and
control-message isolation. Build the DeepSeek CMake targets
`deepseek4_vram_policy_test` and `deepseek4_expert_transfer_test` for the budget
arithmetic and CUDA tests. The CUDA test covers matrix-byte parity, partial
eviction, a shared target/draft count cap, disabling/refilling the cache, and an
extra 256 MiB GPU allocation that forces eviction. It returns 77 when no CUDA
device is available. Performance of the per-matrix live allocator has not been
benchmarked against the fixed arena.

Validated on 3 October 2026 on Windows with an RTX 5090 32 GiB: 122 Python tests
and all four DeepSeek C++ test executables passed. With the Q3 DeepSeek model and
DSpark loaded, the limit changed from 64 to 0 to 128 matrices during a 16-token
response, which completed successfully. In a separate idle test, another CUDA
process allocated and touched 1 GiB; the engine released its 128 cached matrices
(465,895,424 bytes) and restored the available budget after that process exited.
