"""Compare native MTP depths in separate, warmed pipe engines on this PC.

Uses the exact saved prompt/continuation. Never changes the working profile.
Warmup requests prime routing/cache; the following requests are timed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.server import StrataEngine, child_env, engine_args


def replace(args, flag, value):
    args = list(args)
    if flag in args:
        args[args.index(flag) + 1] = str(value)
    else:
        args += [flag, str(value)]
    return args


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile', type=Path, required=True)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--second-reference', type=Path, help='Alternate requests with a second saved prompt/continuation')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--depths', default='0,1,2,3')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--warmups', type=int, default=1)
    ap.add_argument('--mtp-cache-mib', type=int, default=512)
    ap.add_argument('--chunk-mib', type=int, default=4)
    ap.add_argument('--copy-events', type=int, choices=(0, 1, 2), help='0: host waits; 1: events per range; 2: events per tensor')
    ap.add_argument('--decode-readers', type=int, choices=(1, 2, 3, 4))
    ap.add_argument('--read-mode', type=int, choices=(0, 1, 2), help='0 mmap; 1 native cached reads; 2 native cold prefill')
    ap.add_argument('--load-ram', type=int, choices=(0, 1), help='Copy expert weights into private RAM instead of mmap')
    ap.add_argument('--ram-layers', type=int, help='With --load-ram 1, keep layers at or above this index mapped')
    ap.add_argument('--paging-counters', action='store_true', help='Windows machine-wide paging/disk deltas per request')
    ap.add_argument('--write-combined', type=int, choices=(0, 1))
    ap.add_argument('--early-host-refill', type=int, choices=(0, 1), help='Prepare the next host payload while the prior GPU slot is still in use')
    ap.add_argument('--main-cache-decay', type=int)
    ap.add_argument('--pool-reclaim', type=int, choices=(0, 1))
    ap.add_argument('--memory-pool', type=int, choices=(0, 1))
    ap.add_argument('--cache-slab-mib', type=int, help='0 disables packed cache; otherwise 4..256 MiB per slab')
    ap.add_argument('--cache-compact', type=int, choices=(0, 1), help='Compact idle expert slabs between requests')
    ap.add_argument('--cache-only', type=int, choices=(0, 1), help='MTP catch-up: full control / cache-only graph')
    ap.add_argument('--token-batch', type=int, choices=(0, 1), help='Quantized verify matvec: serial / batch single-token kernels')
    ap.add_argument('--shared-scratch', type=int, choices=(0, 1), help='Share setup-time CUDA compute buffers between serialized target and draft')
    a = ap.parse_args()
    depths = [int(x) for x in a.depths.split(',')]
    if a.repeats < 2 or a.warmups < 1 or not depths or any(x not in range(4) for x in depths):
        ap.error('at least one warmup, two measured repeats and depths 0..3 required')
    if a.main_cache_decay is not None and not 1 <= a.main_cache_decay <= 1048576:
        ap.error('main cache decay must be 1..1048576')
    if a.cache_slab_mib is not None and a.cache_slab_mib != 0 and not 4 <= a.cache_slab_mib <= 256:
        ap.error('cache slab must be 0 or 4..256 MiB')
    if a.ram_layers is not None and (a.load_ram != 1 or not 0 <= a.ram_layers <= 4096):
        ap.error('ram layers must be 0..4096 and require --load-ram 1')
    cfg = json.loads(a.profile.read_text(encoding='utf-8'))
    paging = None
    if a.paging_counters:
        from tools.windows_memory_counters import PagingCounters
        paging = PagingCounters()
        paging.sample()  # Fail before loading the model if PDH is unavailable.
    reference = json.loads(a.reference.read_text(encoding='utf-8'))
    references = [reference]
    if a.second_reference:
        references.append(json.loads(a.second_reference.read_text(encoding='utf-8')))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'running', 'configuration': cfg, 'reference': str(a.reference.resolve()),
              'binary_sha256': hashlib.sha256(Path(cfg['exe']).read_bytes()).hexdigest(),
              'warmup_requests': a.warmups, 'timed_requests': a.repeats,
              'timing': '(output tokens - 1) * 1000 / DONE.decode_ms; excludes load/warmup/prefill, includes target sampling, draft, verify, repair and pipe writes',
              'second_reference': str(a.second_reference.resolve()) if a.second_reference else None,
              'scope': 'saved prompts in round-robin order, sequential engines, OS file cache not cleared', 'variants': []}
    def save():
        a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    save()
    for depth in depths:
        variant = dict(cfg)
        variant['env'] = dict(cfg.get('env') or {})
        for name, value in (('STRATA_GLM_COPY_EVENTS', a.copy_events),
                            ('STRATA_GLM_DECODE_READERS', a.decode_readers),
                            ('STRATA_GLM_EXPERT_READ_MODE', a.read_mode),
                            ('STRATA_GLM_EXPERT_LOAD_RAM', a.load_ram),
                            ('STRATA_GLM_EXPERT_RAM_LAYERS', a.ram_layers),
                            ('STRATA_GLM_WRITE_COMBINED', a.write_combined),
                            ('STRATA_GLM_EARLY_HOST_REFILL', a.early_host_refill),
                            ('STRATA_GLM_MAIN_CACHE_DECAY', a.main_cache_decay),
                            ('STRATA_GLM_POOL_RECLAIM', a.pool_reclaim),
                            ('STRATA_GLM_MEMORY_POOL', a.memory_pool),
                            ('STRATA_GLM_CACHE_SLAB_MIB', a.cache_slab_mib),
                            ('STRATA_GLM_CACHE_COMPACT', a.cache_compact),
                            ('STRATA_GLM_MTP_CACHE_ONLY', a.cache_only),
                            ('STRATA_GLM_MMVQ_TOKEN_BATCH', a.token_batch),
                            ('STRATA_GLM_MTP_SHARED_SCRATCH', a.shared_scratch)):
            if value is not None:
                variant['env'][name] = str(value)
        for flag, value in (('--expert-pipeline', 1), ('--expert-chunk-mib', a.chunk_mib),
                            ('--mtp', depth), ('--mtp-cache-mib', a.mtp_cache_mib)):
            variant['args'] = replace(variant['args'], flag, value)
        variant['log'] = str(a.output.with_name(a.output.stem + f'-mtp{depth}.engine.log').resolve())
        result = {'depth': depth, 'configuration': variant, 'runs': [], 'status': 'running'}
        report['variants'].append(result)
        engine = None
        try:
            start = time.monotonic()
            engine = StrataEngine(variant['exe'], engine_args(variant), cwd=variant.get('cwd'),
                                  env=child_env(variant), log=variant['log'])
            result['startup_seconds'] = time.monotonic() - start
            result['info'] = engine.info
            assert engine.info['gpu_only'] == 1 and engine.info['expert_pipeline'] == 1
            assert engine.info['spec'] == depth
            if a.copy_events is not None:
                assert engine.info['expert_copy_events'] == a.copy_events
            if a.decode_readers is not None:
                assert engine.info['expert_decode_readers'] == a.decode_readers
            if a.read_mode is not None:
                wanted_read = ('mmap', 'native', 'auto')[a.read_mode]
                if a.read_mode == 0 and variant['env'].get('STRATA_GLM_EXPERT_LOAD_RAM') == '1':
                    wanted_read = 'hybrid' if 'STRATA_GLM_EXPERT_RAM_LAYERS' in variant['env'] else 'ram'
                assert engine.info['expert_read_mode'] == wanted_read
            if a.load_ram is not None:
                storage = ('hybrid' if 'STRATA_GLM_EXPERT_RAM_LAYERS' in variant['env'] else 'ram') if a.load_ram else 'mmap'
                assert engine.info['expert_storage'] == storage
            if a.ram_layers is not None:
                assert engine.info['expert_ram_layers'] == a.ram_layers
            if a.write_combined is not None:
                assert engine.info['expert_write_combined'] == a.write_combined
            if a.early_host_refill is not None:
                assert engine.info['expert_early_host_refill'] == a.early_host_refill
            if a.main_cache_decay is not None:
                assert engine.info['main_cache_decay'] == a.main_cache_decay
            if a.pool_reclaim is not None:
                assert engine.info['expert_pool_reclaim'] == a.pool_reclaim
            if a.memory_pool is not None:
                assert engine.info['expert_memory_pool'] == a.memory_pool
            if a.cache_slab_mib is not None:
                assert engine.info['expert_cache_slab_mib'] == a.cache_slab_mib
            if a.cache_compact is not None:
                assert engine.info['expert_cache_compact'] == a.cache_compact
            if a.cache_only is not None:
                assert engine.info['mtp_cache_only'] == (a.cache_only if depth else 0)
            if a.token_batch is not None:
                assert engine.info['mmvq_token_batch'] == a.token_batch
            if a.shared_scratch is not None:
                assert (engine.info['mtp_shared_scratch_bytes'] > 0) == bool(a.shared_scratch and depth)
            for i in range(a.repeats + a.warmups):
                ref = references[i % len(references)]
                ids, expected = ref['prompt_ids'], ref['generated_ids']
                paging_before = paging.sample() if paging else None
                start = time.perf_counter(); first = None; tokens = []
                for token in engine.generate(ids, len(expected), {'temperature': 0}, threading.Event()):
                    if token is not None:
                        if first is None:
                            first = time.perf_counter() - start
                        tokens.append(token)
                wall = time.perf_counter() - start
                done = dict(engine.last)
                assert done['finish'] == 'length' and len(tokens) == len(expected), done
                run = {'warmup': i < a.warmups, 'prompt_index': i % len(references), 'tokens': tokens, 'equals_reference': tokens == expected, 'done': done,
                       'first_token_seconds': first, 'request_wall_seconds': wall,
                       'tokens_per_second': (len(tokens) - 1) * 1000 / done['decode_ms']}
                if paging:
                    run['system_paging_delta'] = paging.difference(paging_before, paging.sample())
                result['runs'].append(run)
                print(f"MTP {depth}, run {i}: {run['tokens_per_second']:.3f} tok/s, reference={run['equals_reference']}", flush=True)
                save()
            result['median_tokens_per_second'] = statistics.median(r['tokens_per_second'] for r in result['runs'] if not r['warmup'])
            result['status'] = 'pass' if all(r['equals_reference'] for r in result['runs']) else 'token_mismatch'
        except Exception as exc:
            result['status'], result['error'] = 'error', str(exc)
        finally:
            if engine:
                proc = engine.proc
                engine.close()
                result['exit_code'] = proc.returncode
                if proc.returncode != 0:
                    result['status'] = 'error'
                text = Path(variant['log']).read_text(encoding='utf-8', errors='replace')
                for marker, key in (('STRATA_GLM_MEMORY ', 'memory_snapshots'), ('STRATA_GLM_MTP ', 'mtp_statistics')):
                    result[key] = [json.loads(line.split(marker, 1)[1]) for line in text.splitlines() if marker in line]
                if a.shared_scratch and depth and (not result['mtp_statistics'] or
                        not all(s.get('shared_scratch_active') for s in result['mtp_statistics'])):
                    result['status'] = 'error'
                    result['error'] = 'MTP scratch sharing detached or statistics are missing'
            save()
    valid = [v for v in report['variants'] if v['status'] == 'pass']
    report['fastest_matching_depth'] = max(valid, key=lambda v: v['median_tokens_per_second'])['depth'] if valid else None
    report['status'] = 'pass' if len(valid) == len(depths) else 'incomplete_or_mismatch'
    save()
    if paging:
        paging.close()
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
