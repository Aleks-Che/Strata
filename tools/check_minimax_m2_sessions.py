"""Opt-in full-model RAM session audit. Sequential GPU runs; retain full F32 logits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.winjob import contain
from tools.check_minimax_m2_prefix import save, sha
from tools.check_minimax_m2_prefix_context import snapshot, native_checks, NV
from tools.run_minimax_m2 import runtime_environment
from tools.strata_tokenizer import Tokenizer


def compare_files(a, arow, b, brow, rows):
    count = rows*NV
    if rows <= 0 or min(arow, brow) < 0 or a.stat().st_size < (arow*NV+count)*4 or b.stat().st_size < (brow*NV+count)*4:
        raise ValueError('invalid/truncated logits range')
    different, finite, maximum = 0, True, 0.0
    with a.open('rb') as x, b.open('rb') as y:
        x.seek(arow*NV*4); y.seek(brow*NV*4)
        remaining = count
        while remaining:
            n = min(65536, remaining)
            left, right = np.frombuffer(x.read(n*4), dtype='<f4'), np.frombuffer(y.read(n*4), dtype='<f4')
            if len(left) != n or len(right) != n:
                raise ValueError('logits changed/truncated')
            different += int(np.count_nonzero(left.view('<u4') != right.view('<u4')))
            ok = bool(np.isfinite(left).all() and np.isfinite(right).all()); finite &= ok
            if ok: maximum = max(maximum, float(np.max(np.abs(left.astype('f8')-right))))
            remaining -= n
    return {'pass': different == 0 and finite, 'floats': count, 'different_bits': different,
            'finite': finite, 'max_abs': maximum if finite else None}


def make_corpus(tok):
    old = json.loads((ROOT/'build-local/minimax-m2-completion-01/requests.json').read_text(encoding='utf-8'))[0]
    base = tok.encode(old['prompt'], parse_special=True)
    branch = base.copy(); branch[35] = tok.encode('red')[0]
    long = (base+tok.encode(' The blue sky reflects scattered light.'*150))[:512]
    prompts = [base, branch, base[:33], long]
    # name, reference prompt, session, expected reused, archive restore, output budget
    cases = [('a-cold',0,'a',0,False,8), ('a-resident',0,'a',48,False,8),
             ('b-cold',1,'b',0,False,8), ('a-restored',0,'a',48,True,8),
             ('c-cold',2,'c',0,False,8), ('b-restored',1,'b',48,True,8),
             ('a-evicted',0,'a',0,False,8), ('c-restored',2,'c',32,True,8),
             ('anonymous',0,None,0,False,8), ('a-after-anonymous',0,'a',48,True,8),
             ('b-evicted',1,'b',0,False,8), ('b-branch',0,'b',32,False,8),
             ('long-cold',3,'long',0,False,8), ('a-before-long-restore',0,'a',48,True,8),
             ('long-restored',3,'long',496,True,8), ('long-shorter',0,'long',48,False,8),
             ('d-one',0,'d',0,False,1), ('long-one-restored',0,'long',48,True,1),
             ('d-one-restored',0,'d',48,True,1), ('d-one-resident',0,'d',48,False,1)]
    requests = []
    for name, ref, session, reuse, restored, limit in cases:
        request = {'tokens': prompts[ref], 'max_tokens': limit}
        if session: request['session_key'] = hashlib.sha256(session.encode()).hexdigest()
        requests.append(request)
    return prompts, cases, requests


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gguf', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reference-engine', type=Path, required=True)
    p.add_argument('--engine', type=Path, default=ROOT/'build-local/minimax-m2-cuda/bin/strata-minimax-m2-bench.exe')
    p.add_argument('--cuda-root', type=Path, default=ROOT/'build-local/cuda-13.0')
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    report = {'pass': False, 'stage': 'MM27-27', 'model': str(args.gguf.resolve()),
              'engine_sha256': sha(args.engine), 'reference_sha256': sha(args.reference_engine), 'runs': {}, 'cases': []}
    snapshot(args.out, report)
    for file in (Path(__file__), ROOT/'tools/check_minimax_m2_prefix_context.py'):
        dest = args.out/'sources'/file.resolve().relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, dest); report['sources'][file.resolve().relative_to(ROOT).as_posix()] = sha(dest)
    shutil.copyfile(args.engine, args.out/'engine.exe'); shutil.copyfile(args.reference_engine, args.out/'reference.exe')
    report['gpu'] = subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
    env = runtime_environment(args.cuda_root); env['STRATA_MM27_TOKENWISE'] = '0'
    report['environment'] = {k:v for k,v in env.items() if k.startswith(('STRATA_','GGML_','LLAMA_','NVIDIA_'))}
    def run(name, binary, requests, cap=None):
        save(args.out/(name+'.requests.json'), requests)
        command = [str(binary.resolve()), '--gguf',str(args.gguf.resolve()),'--ctx','2048','--batch','16','--mode','2',
                   '--prefix-cache','1','--gpu-cache-mib','18432','--gpu-cache-allocator','arena',
                   '--pipeline-readers','2','--pipeline-chunk-mib','4','--pipeline-lookahead','1','--pipeline-d2d-batch','1',
                   '--request',str((args.out/(name+'.requests.json')).resolve()),
                   '--output',str((args.out/(name+'.json')).resolve()),'--logits',str((args.out/(name+'.f32')).resolve())]
        if cap is not None: command += ['--session-cache-mib', str(cap), '--session-cache-slots','2']
        report['runs'][name] = {'command':command}; save(args.out/'report.json',report)
        print('START '+name+' '+str(len(requests))+' requests',flush=True)
        flags = {}
        if os.name == 'nt':
            startup = subprocess.STARTUPINFO(); startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW; startup.wShowWindow = 0
            flags = {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP, 'startupinfo': startup}
        with (args.out/(name+'.stdout.log')).open('wb') as out, (args.out/(name+'.stderr.log')).open('wb') as err:
            proc = subprocess.Popen(command, stdout=out, stderr=err, env=env, **flags); contain(proc)
            report['runs'][name]['pid'] = proc.pid; save(args.out/'report.json',report)
            try: rc = proc.wait(timeout=1800)
            finally:
                if proc.poll() is None: proc.kill(); proc.wait(timeout=15)
        report['runs'][name]['exit_code'] = rc
        assert rc == 0, name
        data = json.loads((args.out/(name+'.json')).read_text(encoding='utf-8'))
        checks = [native_checks(r,data) for r in data['results']]
        report['runs'][name]['checks'] = checks
        assert all(all(c.values()) for c in checks), (name,checks)
        assert (args.out/(name+'.f32')).stat().st_size == sum(r['generated_tokens'] for r in data['results'])*NV*4
        save(args.out/'report.json',report); print('DONE '+name,flush=True)
        return data['results']
    try:
        prompts, cases, requests = make_corpus(Tokenizer.from_gguf(args.gguf))
        fresh = [{'tokens':ids,'max_tokens':8} for ids in prompts]
        ref = run('reference',args.reference_engine,fresh)
        cold = run('fresh',args.engine,fresh,0)
        for i, (a,b) in enumerate(zip(ref,cold)):
            parity = compare_files(args.out/'reference.f32',i*8,args.out/'fresh.f32',i*8,8)
            report['cases'].append({'name':'fresh-regression-'+str(i),'logits':parity,
                                    'pass':a['token_ids']==b['token_ids'] and parity['pass']})
        restored = run('sessions',args.engine,requests,600)
        offset = 0
        for case, result in zip(cases,restored):
            name, idx, session, reuse, restore, count = case
            parity = compare_files(args.out/'fresh.f32',idx*8,args.out/'sessions.f32',offset,count); offset += count
            check = (result['token_ids']==cold[idx]['token_ids'][:count] and result['reused_tokens']==reuse and
                     result['session_restore']==restore and result['session_archive_bytes']<=600*1024*1024 and
                     result['session_archive_entries']<=2 and result['request_ms']>=result['session_ms'] and parity['pass'])
            report['cases'].append({'name':name,'logits':parity,'expected_reused':reuse,'expected_restore':restore,'pass':check})
        assert restored[-1]['session_archive_evictions']>0
        rejected = run('budget-rejection',args.engine,[requests[0],requests[2],requests[3]],1)
        for i,r in enumerate(rejected):
            ref_idx = [0,1,0][i]
            parity = compare_files(args.out/'fresh.f32',ref_idx*8,args.out/'budget-rejection.f32',i*8,8)
            report['cases'].append({'name':'budget-rejection-'+str(i),'logits':parity,
                'pass':parity['pass'] and r['token_ids']==cold[ref_idx]['token_ids'] and r['reused_tokens']==0 and
                       r['session_archive_bytes']==0 and not r['session_restore'] and r['session_archive_rejected']==i})
        report['pass'] = all(c['pass'] for c in report['cases'])
        assert report['pass'], [c['name'] for c in report['cases'] if not c['pass']]
    except BaseException:
        report['error'] = traceback.format_exc(); raise
    finally:
        save(args.out/'report.json',report)
    print('PASS '+str(len(report['cases']))+' full-logit comparisons',flush=True)


if __name__ == '__main__': main()
