"""Recheck MM27-31 raw logits, corpus coverage, source identity and cleanup."""
import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.minimax_m2_engine import EXE_SHA256
from tools.check_minimax_m2_prefix import save
from tools.check_minimax_m2_session_batches import compare_runs, make_corpus
from tools.check_minimax_m2_sessions_context import sha
from tools.minimax_m2_memory_observer import memory_within_limit


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def process_exited(pid):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x100000, False, pid)
    if not handle:
        return ctypes.get_last_error() == 87
    try:
        return kernel.WaitForSingleObject(handle, 0) == 0
    finally:
        kernel.CloseHandle(handle)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch1', type=Path, required=True)
    p.add_argument('--batch8', type=Path, required=True)
    p.add_argument('--out', type=Path, default=ROOT/'docs/minimax-m2.7/MINIMAX_M27_SESSION_BATCHES_CHECK.json')
    args = p.parse_args(); gates, sources, artifacts, batches, processes = {}, {}, {}, {}, {}
    binary = ROOT/'build-local/minimax-m2-cuda/bin/strata-minimax-m2-bench.exe'
    gates['current_admitted_binary'] = sha(binary) == EXE_SHA256
    for batch, folder in [(1,args.batch1.resolve()), (8,args.batch8.resolve())]:
        r = read(folder/'report.json'); prefix = 'batch'+str(batch)+'_'
        gates[prefix+'completed'] = r['pass'] and r['batch'] == batch
        gates[prefix+'binary'] = r['engine_sha256'] == sha(folder/'engine.exe') == EXE_SHA256
        gates[prefix+'environment'] = r['environment']['STRATA_MM27_STATE_BULK'] == '1' and r['environment']['STRATA_MM27_TOKENWISE'] == '0'
        assert r['pass'], 'incomplete/failed report: '+str(folder)
        requests = read(folder/'sessions.requests.json'); expected = r['cases']
        by_name = {c['name']:req for c,req in zip(expected,requests)}
        # Reconstruct the declared fixed corpus; do not accept missing scenarios.
        fresh, rebuilt, cases = make_corpus(by_name['cold']['tokens'], by_name['long-extend']['tokens'],
                                            by_name['branch']['tokens'][35], batch)
        gates[prefix+'corpus'] = requests == rebuilt and read(folder/'fresh.requests.json') == fresh and len(expected) == 25
        gates[prefix+'expectations'] = all(all(actual[k] == value for k,value in case.items()) for actual,case in zip(expected,cases))
        comparisons = compare_runs(folder,batch,cases)
        gates[prefix+'all_logit_rows'] = all(c['pass'] for c in comparisons)
        memory = []; totals = []; generated = 0
        for name in ['fresh','sessions']:
            info = r['runs'][name]; processes[info['pid']] = process_exited(info['pid'])
            gates[prefix+name+'_cleanup'] = info['exit_code'] == 0 and info['process_exited'] and processes[info['pid']]
            rows = [json.loads(line) for line in (folder/(name+'.memory.jsonl')).read_text().splitlines()]
            gates[prefix+name+'_observer'] = bool(rows) and len(rows) == info['observer']['samples'] and info['observer']['stopped'] and not info['observer']['error']
            gates[prefix+name+'_limit95'] = all(memory_within_limit(m) for m in rows)
            memory.extend(rows)
            d = read(folder/(name+'.json')); generated += sum(x['generated_tokens'] for x in d['results'])
            native = [d['memory_before'],d['memory_loaded']]+[m for c in d['results'] for m in c['memory_samples']]
            totals.extend(native)
            gates[prefix+name+'_native95'] = all(0 <= m[k+'_available'] <= m[k+'_total'] and m[k+'_available']*20 >= m[k+'_total'] for m in native for k in ['ram','vram'])
        for relative,digest in r['sources'].items():
            sources[(folder/'sources'/relative).relative_to(ROOT).as_posix()] = sha(folder/'sources'/relative) == digest
        for name,digest in r['artifacts'].items():
            gates[prefix+'artifact_'+name] = sha(folder/name) == digest
        # Documentation can record this result after the immutable run snapshot.
        gates[prefix+'current_sources'] = all(sha(ROOT/name) == digest for name,digest in r['sources'].items() if Path(name).suffix != '.md')
        batches[batch] = {'requests': len(fresh)+len(requests), 'comparisons': len(comparisons),
                          'generated_tokens': generated, 'compared_tokens': sum(c['generated_tokens'] for c in comparisons),
                          'compared_floats': sum(c['logits']['floats'] for c in comparisons), 'pairs': comparisons,
                          'sampled_peak_percent': {
                              'ram': max([100*(1-m['ram_free']/m['ram_total']) for m in memory]+[100*(1-m['ram_available']/m['ram_total']) for m in totals]),
                              'vram': max([100*(1-m['gpu_free']/m['gpu_total']) for m in memory]+[100*(1-m['vram_available']/m['vram_total']) for m in totals])}}
        for file in folder.rglob('*'):
            if file.is_file(): artifacts[file.relative_to(ROOT).as_posix()] = {'sha256':sha(file),'bytes':file.stat().st_size}
    unit = ROOT/'build-local/minimax-m2-session-batches-unit.log'; unit_text = unit.read_text(encoding='utf-8')
    gates['cpu_tests'] = 'Ran 14 tests' in unit_text and '\nOK\n' in unit_text
    gates['source_snapshots'] = all(sources.values())
    result = {'pass':all(gates.values()), 'stage':'MM27-31 boundary', 'engine_sha256':EXE_SHA256,
              'context':1024, 'batches':batches, 'gates':gates, 'source_checks':sources, 'owned_pids_exited':processes,
              'artifacts':artifacts, 'unit_log_sha256':sha(unit), 'validator_sha256':sha(Path(__file__)),
              'limitations':['Prompts up to513; GPU batch1/8 at2K/4K still open.',
                             'Offline full logits; no new HTTP/cancel/pressure or EOS corpus.',
                             'Same engine fresh reference, not an independent model oracle or speed A/B.']}
    save(args.out,result)
    print(('PASS' if result['pass'] else 'FAIL'),len(gates),'gates;',len(sources),'source checks;',len(artifacts),'artifact hashes')
    if not result['pass']: raise SystemExit([k for k,v in gates.items() if not v])


if __name__ == '__main__':
    main()
