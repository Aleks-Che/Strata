# Session archive and token statistics

The local integration ports features from `local-inference/Strata` into the newer
Strata engine and server. It preserves upstream MCP tools, model unload/load,
reasoning budgets, request monitoring, multi-GPU code, segmented KV retention and
the local Q8_0 PLE support. Whole-session parking still requires a single GPU.

## Using the web app

- **Statistics** shows processed input, cached input, model output and request
  totals. Choose 24 hours, 7 days, 30 days or all time. Charts and the activity table
  update after requests finish, including known work from interrupted requests.
- **Monitor → Session archive** lists active and saved sessions, token counts,
  snapshot RAM, last use and the last save result. Retained KV for the active
  session is reported separately and consumes the same cache budget.
- **Release session** frees one entire parked snapshot, including its internal
  checkpoints. It leaves the active session intact. Requests must be idle with
  no queue. The next request for a released history may need to read it again.
- **Auto-release session after … hours** expires parked sessions between requests
  or while idle. It is off initially; the initial interval is 2 hours. Settings
  are saved beside the run config in `*.archive-settings.json`. Changing settings
  is allowed while busy; releasing memory waits for the engine FIFO.

Age colors are visual cues: up to 30 minutes, 30–60 minutes, 1–2 hours and over
2 hours. They do not independently cause eviction. Snapshots stay in **RAM** and
are lost on model unload or process restart. This feature does not serialize KV
to disk. Chat text in the browser, the RAM archive and token statistics are
separate stores; releasing a snapshot does not delete the browser's chat text.

## Configuration

Add engine arguments, choosing a budget that leaves room for model weights,
ordinary prompt checkpoints, other apps and file caching:

```json
["--conversation-cache-mib", "2048",
 "--conversation-cache-slots", "4",
 "--conversation-cache-min-free-mib", "8192"]
```

The archive budget is a cap, not an allocation at startup. The default budget is
0, slots 4, physical memory floor 2560 MiB. `--prompt-cache 0` disables parking.
An incoming transfer image and retained KV also count against the budget.
Insufficient physical RAM or Windows commit skips parking and reports the reason;
old parked entries can be evicted under pressure. The configured floor is checked
at capture boundaries, not a reservation against memory other software may use.

`--conversation-cache-ttl SECONDS` adds engine-side idle expiry (default 0: off).
It runs even without the Python server. When both TTL and the web auto-release
policy are enabled, either may remove an eligible snapshot. A long-idle active
history is not given a fresh engine TTL merely because it was parked later.

`--conversation-cache-shared-prefix` opts into exact prefix reuse across named
sessions and replacement of identical histories (default off). Token positions,
image identity, steering mode and valid draft KV still have to match. Checkpoint
retention keeps the current upstream pinned-root/LRU policy. Session IDs are
matching hints, not authentication or per-user access controls.

The real server saves usage to `data/usage-statistics.sqlite` by default. Override
with config `statistics_file` or `--statistics-file PATH`; relative paths are
resolved against the repository. Mock servers use memory unless explicitly given
a path. The database stores minute aggregates and totals, not prompts, session IDs
or API keys. Its counts start from the first request handled by this integration;
older benchmark logs and another checkout's database are not imported.

SQLite commits both aggregate tables together. Failed writes remain pending in
RAM for retry, with a visible warning. Pending, uncommitted increments cannot
survive a process crash. A reasoning-budget request can make two engine passes:
both passes' actual input/cache work is counted under one request; injected
wrap-up text is not model output. MCP turns are separate model requests.

## HTTP and engine protocol

Both completion APIs accept an optional single `X-Strata-Session-Id` header, up to
256 UTF-8 bytes, nonempty and without control characters. The server hashes it
before sending `session=<sha256>` to the engine, outside the prompt. An old engine
rejects an explicitly requested session ID instead of pretending to isolate it.

| Endpoint | Result |
| --- | --- |
| `GET /statistics?range=24h` | Totals, time buckets and persistence status; ranges `24h`, `7d`, `30d`, `all` |
| `GET /cache/entries` | Active/saved inventory, RAM, generation ID, policy and save diagnostics |
| `POST /cache/release` | JSON `{"id":"session-1","generation":"…"}`; releases one parked session |
| `POST /cache/settings` | JSON `{"enabled":true,"hours":2}`; valid interval 0.1–8760 hours |
| `GET /metrics` | Adds `conversation_cache`, transfer phase and per-request cache timings |

These endpoints honor the configured API key. Archive writes also require JSON
and a same-origin or explicitly trusted page. Release returns 409 while busy or
for an old generation, 404 for a vanished entry, and 501 for an unsupported engine.
Only whole saved sessions can be released.

The engine advertises `session-id cache-admin` in `READY`. `CACHE_ENTRIES` publishes
a completed inventory. `CACHE` lines carry gauges, save/restore times and failure
reasons. `CACHE_DROP <request-id> <entry-id>` is acknowledged by `CACHE_DROPPED`;
the Python reader keeps those control replies outside the generation token queue.
Stable entry IDs are not reused within a process; the server generation changes
after a restart. Model unload clears the inventory.

## Verification

Python tests: `python -m unittest discover -s serve -p "test_*.py"`.
The Windows process-containment test needs ordinary process visibility; sandboxed
`tasklist` can report no child even when it is alive.

CMake `STRATA_BUILD_CONVERSATION_TESTS=ON` enables CPU cache/memory tests;
`STRATA_BUILD_TESTS=ON` also enables snapshot and validation tests. The added cases
cover namespace isolation, whole-session deletion, stale IDs, TTL boundaries,
memory/commit limits, MTP valid ranges and superseded history isolation.

Local measurements and the Q8_0 configuration are in
[the Russian integration report](SESSION_INTEGRATION_RU.md).
