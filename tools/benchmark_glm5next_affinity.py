"""Windows CPU placement comparison inside one warmed GLM pipe engine.

Changes only the process this tool starts, restores its original affinity,
and preserves the selected profile. Uses one saved prompt in ABBA order.
"""
import argparse
import ctypes
from ctypes import wintypes as w
import hashlib
import json
from pathlib import Path
import statistics
import struct
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.server import StrataEngine, child_env, engine_args
from tools.windows_memory_counters import PagingCounters


def topology():
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.GetLogicalProcessorInformationEx.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(w.DWORD)]
    api.GetProcessAffinityMask.argtypes = [w.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    api.SetProcessAffinityMask.argtypes = [w.HANDLE, ctypes.c_size_t]
    size = w.DWORD()
    api.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))
    if not size.value:
        raise ctypes.WinError(ctypes.get_last_error())
    buf = ctypes.create_string_buffer(size.value)
    if not api.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(size)):
        raise ctypes.WinError(ctypes.get_last_error())
    cores, offset = [], 0
    while offset < size.value:
        relationship, length = struct.unpack_from('<II', buf.raw, offset)
        count = struct.unpack_from('<H', buf.raw, offset + 30)[0]
        if relationship != 0 or length < 48 or count != 1:
            raise RuntimeError('this benchmark requires a single Windows processor group')
        mask, group = struct.unpack_from('<QH', buf.raw, offset + 32)
        if group:
            raise RuntimeError('multiple processor groups are not supported')
        cores.append(mask)
        offset += length
    return api, cores


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile', type=Path, required=True)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--warmups', type=int, default=6)
    ap.add_argument('--blocks', type=int, default=3, help='Four timed requests per ABBA block')
    ap.add_argument('--layout', choices=('physical', 'first-half', 'second-half'), default='physical')
    ap.add_argument('--compaction', choices=(0, 1), type=int, default=1)
    a = ap.parse_args()
    if sys.platform != 'win32' or ctypes.sizeof(ctypes.c_void_p) != 8:
        ap.error('requires 64-bit Windows')
    if a.warmups < 1 or a.blocks < 2:
        ap.error('at least one warmup and two ABBA blocks required')
    api, cores = topology()
    cfg = json.loads(a.profile.read_text(encoding='utf-8'))
    cfg['env'] = dict(cfg.get('env') or {})
    cfg['env']['STRATA_GLM_CACHE_COMPACT'] = str(a.compaction)
    cfg['log'] = str(a.output.with_suffix('.engine.log').resolve())
    reference = json.loads(a.reference.read_text(encoding='utf-8'))
    report = {'status': 'running', 'configuration': cfg, 'reference': reference,
              'binary_sha256': hashlib.sha256(Path(cfg['exe']).read_bytes()).hexdigest(),
              'core_masks': [hex(m) for m in cores], 'layout': a.layout, 'runs': [],
              'method': 'One engine; same prompt, initial warmups then all/limited/limited/all blocks. No OS cache flush; system paging counters include other processes.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    engine, original, handle = None, 0, None
    paging = PagingCounters()
    try:
        start = time.monotonic()
        engine = StrataEngine(cfg['exe'], engine_args(cfg), cwd=cfg.get('cwd'), env=child_env(cfg), log=cfg['log'])
        report.update(startup_seconds=time.monotonic()-start, info=engine.info)
        assert engine.info['spec'] == 0 and engine.info['gpu_only'] == 1
        handle = w.HANDLE(int(engine.proc._handle))
        def get_mask():
            own, system = ctypes.c_size_t(), ctypes.c_size_t()
            if not api.GetProcessAffinityMask(handle, ctypes.byref(own), ctypes.byref(system)):
                raise ctypes.WinError(ctypes.get_last_error())
            return own.value
        original = get_mask()
        selected = cores if a.layout == 'physical' else cores[:len(cores)//2] if a.layout == 'first-half' else cores[len(cores)//2:]
        limited = 0
        for core in selected:
            available = core & original
            if available:
                limited |= available & -available
        if not limited or limited == original:
            raise RuntimeError('no distinct CPU placement to compare')
        report['affinity_masks'] = {'all': hex(original), 'limited': hex(limited)}
        schedule = [('all', True)] * a.warmups + [(label, False) for _ in range(a.blocks) for label in ('all', 'limited', 'limited', 'all')]
        for i, (label, warmup) in enumerate(schedule):
            mask = original if label == 'all' else limited
            if not api.SetProcessAffinityMask(handle, mask) or get_mask() != mask:
                raise RuntimeError('failed to apply/verify affinity to the owned engine')
            before = paging.sample()
            start = time.perf_counter(); first = None; tokens = []
            for token in engine.generate(reference['prompt_ids'], len(reference['generated_ids']), {'temperature': 0}, threading.Event()):
                if token is not None:
                    if first is None: first = time.perf_counter() - start
                    tokens.append(token)
            done = dict(engine.last)
            assert done['finish'] == 'length' and tokens == reference['generated_ids'], done
            run = {'warmup': warmup, 'placement': label, 'affinity': hex(mask), 'tokens': tokens,
                   'equals_reference': True, 'done': done, 'first_token_seconds': first,
                   'request_wall_seconds': time.perf_counter()-start,
                   'tokens_per_second': (len(tokens)-1)*1000/done['decode_ms'],
                   'system_paging_delta': paging.difference(before, paging.sample())}
            report['runs'].append(run)
            print(f"{i}: {label}, {run['tokens_per_second']:.3f} tok/s, reference=True", flush=True)
            save()
        report['summary'] = {label: {'median_tokens_per_second': statistics.median(r['tokens_per_second'] for r in report['runs'] if not r['warmup'] and r['placement'] == label),
            'median_wall_seconds': statistics.median(r['request_wall_seconds'] for r in report['runs'] if not r['warmup'] and r['placement'] == label)} for label in ('all', 'limited')}
        report['status'] = 'pass'
    except Exception as exc:
        report.update(status='error', error=str(exc))
    finally:
        if engine:
            if original and engine.proc.poll() is None:
                report['affinity_restored'] = bool(api.SetProcessAffinityMask(handle, original))
                if not report['affinity_restored']: report['status'] = 'error'
            proc = engine.proc
            engine.close(); report['exit_code'] = proc.returncode
            if proc.returncode: report['status'] = 'error'
            report['memory_snapshots'] = [json.loads(line.split('STRATA_GLM_MEMORY ', 1)[1]) for line in Path(cfg['log']).read_text(encoding='utf-8', errors='replace').splitlines() if 'STRATA_GLM_MEMORY ' in line]
        paging.close(); save()
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
