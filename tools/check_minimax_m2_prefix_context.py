"""Opt-in MM27-26 full-logit prefix audit: 2K/4K, EOS, sampling and one-token output.

Uses the admitted executable without changes. Run alone. Retains all logits;
comparisons read bounded chunks, including when RAM pressure is tested later.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.minimax_m2_api import MiniMaxAPITemplate
from serve.minimax_m2_engine import EXE_SHA256
from serve.server import Service
from serve.winjob import contain
from tools.check_minimax_m2_completion_pipe import result_checks
from tools.check_minimax_m2_http_context import exact_prompt
from tools.check_minimax_m2_prefix import save, sha
from tools.run_minimax_m2 import runtime_environment
from tools.strata_tokenizer import Tokenizer

NV = 200064


def native_checks(result, header):
    checks = result_checks(result, header)
    # A one-token answer samples prefill logits without any decode graph.
    if result['generated_tokens'] == 1:
        pre, dec = result['prefill'], result['decode']
        checks['gpu_only'] = (pre['gpu_nodes'] > 0 and pre['rejected_cpu_nodes'] == pre['rejected_full_copies'] == 0 and
                              dec['gpu_nodes'] == dec['compute_calls'] == dec['rejected_cpu_nodes'] == dec['rejected_full_copies'] == 0)
    checks['kv_accounting'] = (result['kv_tokens'] == result['prompt_tokens']+result['generated_tokens']-1 and
                               result['reused_tokens']+result['evaluated_prompt_tokens'] == result['prompt_tokens'] and
                               0 <= result['reused_tokens'] < result['prompt_tokens'] and
                               result['reused_tokens'] % header['batch'] == 0)
    return checks


def compare_ranges(path, first, second, count, chunk_floats=65536):
    """Offsets/count in floats, bit equality plus finite checks; no tolerance."""
    if count <= 0 or first < 0 or second < 0 or chunk_floats < 1 or path.stat().st_size < 4*max(first+count, second+count):
        raise ValueError('invalid/truncated logits range')
    different, finite, maximum = 0, True, 0.0
    with path.open('rb') as a, path.open('rb') as b:
        a.seek(first*4); b.seek(second*4)
        remaining = count
        while remaining:
            n = min(remaining, chunk_floats)
            x, y = np.frombuffer(a.read(n*4), dtype='<f4'), np.frombuffer(b.read(n*4), dtype='<f4')
            if len(x) != n or len(y) != n:
                raise ValueError('logits changed/truncated during comparison')
            different += int(np.count_nonzero(x.view('<u4') != y.view('<u4')))
            valid = bool(np.isfinite(x).all() and np.isfinite(y).all())
            finite &= valid
            if valid:
                maximum = max(maximum, float(np.max(np.abs(x.astype('f8')-y))))
            remaining -= n
    return {'pass': finite and different == 0, 'floats': count, 'different_bits': different,
            'finite': finite, 'max_abs': maximum if finite else None}


def snapshot(out, report):
    sources = [*sorted((ROOT/'backends/minimax_m2').glob('*')),
               *sorted((ROOT/'serve').glob('*minimax*.py')), ROOT/'serve/server.py', ROOT/'serve/winjob.py',
               ROOT/'serve/fixtures/minimax_m27_chat_template.jinja', ROOT/'tools/strata_tokenizer.py',
               ROOT/'tools/run_minimax_m2.py', ROOT/'tools/check_minimax_m2_completion_pipe.py',
               ROOT/'tools/check_minimax_m2_http_context.py', ROOT/'tools/check_minimax_m2_prefix.py', Path(__file__)]
    report['sources'] = {}
    for source in sources:
        if not source.is_file():
            continue
        rel = source.resolve().relative_to(ROOT)
        dest = out/'sources'/rel; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest); report['sources'][rel.as_posix()] = sha(dest)


def make_corpus(tok):
    svc = Service(None, tok, MiniMaxAPITemplate(ROOT/'serve/fixtures/minimax_m27_chat_template.jinja'))
    small_req, small = exact_prompt(svc, 2032)
    large_req, large = exact_prompt(svc, 4080)
    english = {'model': 'minimax-m2.7', 'messages': [{'role': 'user', 'content':
               'Explain in a few sentences why the sky appears blue during the day.'}], 'max_tokens': 512, 'temperature': 0}
    messages, tools, kwargs = svc.normalize_request(english, 'openai')
    short = tok.encode(svc.template.render(messages, tools=tools, **kwargs), parse_special=True)
    common = next((i for i,(a,b) in enumerate(zip(small,large)) if a != b), min(len(small),len(large)))
    sample = {'temperature': 1, 'top_p': .95, 'top_k': 40, 'seed': 42}
    specs = [('2k-initial', small, 'long', 8, {}, 0), ('2k-repeat', small, 'long', 8, {}, 2016),
             ('4k-extend', large, 'long', 8, {}, common//16*16), ('4k-repeat', large, 'long', 8, {}, 4064),
             ('4k-fresh', large, None, 8, {}, 0), ('eos-initial', short, 'short', 512, {}, 0),
             ('eos-repeat', short, 'short', 512, {}, 48), ('sample-reuse', short, 'short', 128, sample, 48),
             ('sample-fresh', short, None, 128, sample, 0), ('one-initial', short, 'short', 1, {}, 0),
             ('one-repeat', short, 'short', 1, {}, 48)]
    requests = [{'tokens': ids, 'max_tokens': maximum, 'sampling': sampling,
                 **({'session_key': hashlib.sha256(key.encode()).hexdigest()} if key else {})}
                for _,ids,key,maximum,sampling,_ in specs]
    cases = [{'name': name, 'expected_reused': reuse} for name,_,_,_,_,reuse in specs]
    return requests, cases, {'2k':small_req,'4k':large_req,'english':english}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cuda-root', type=Path, default=ROOT/'build-local/cuda-13.0')
    args = parser.parse_args()
    binary = ROOT/'build-local/minimax-m2-cuda/bin/strata-minimax-m2-bench.exe'
    assert sha(binary) == EXE_SHA256
    args.out.mkdir(parents=True, exist_ok=False)
    report = {'pass': False, 'stage': 'MM27-26 offline', 'engine_sha256': EXE_SHA256,
              'model': str(args.gguf.resolve()), 'cases': [], 'pairs': []}
    snapshot(args.out, report); shutil.copyfile(binary, args.out/'engine.exe')
    env = runtime_environment(args.cuda_root); env['STRATA_MM27_TOKENWISE'] = '0'
    report['environment'] = {k:v for k,v in env.items() if k.startswith(('STRATA_', 'GGML_', 'LLAMA_', 'NVIDIA_'))}
    report['gpu'] = subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
    try:
        requests, cases, http = make_corpus(Tokenizer.from_gguf(args.gguf))
        report['cases'] = cases
        save(args.out/'requests.json', requests); save(args.out/'http-requests.json',http)
        command = [str(binary), '--gguf', str(args.gguf.resolve()), '--ctx','4096','--batch','16','--mode','2',
                   '--prefix-cache','1','--gpu-cache-mib','18432','--gpu-cache-allocator','arena',
                   '--pipeline-readers','2','--pipeline-chunk-mib','4','--pipeline-lookahead','1','--pipeline-d2d-batch','1',
                   '--request',str((args.out/'requests.json').resolve()),'--output',str((args.out/'generation.json').resolve()),
                   '--logits',str((args.out/'generation.f32').resolve())]
        report['command'] = command; save(args.out/'report.json',report)
        print('START native ctx4096 corpus: '+str(len(cases))+' cases',flush=True)
        with (args.out/'stdout.log').open('wb') as out, (args.out/'stderr.log').open('wb') as err:
            proc = subprocess.Popen(command,stdout=out,stderr=err,env=env,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            try:
                if not contain(proc):
                    raise RuntimeError('cannot contain native child')
                report['native_pid'] = proc.pid; end = time.monotonic()+3600; last = -1
                while proc.poll() is None:
                    f = args.out/'generation.f32'; rows = f.stat().st_size//(NV*4) if f.exists() else 0
                    if rows >= last+64 or rows in (0,8,16,24,32,40) and rows!=last:
                        print('F32 output rows saved: '+str(rows),flush=True); last = rows
                    if time.monotonic()>end:
                        raise TimeoutError('offline corpus exceeded one hour')
                    time.sleep(2)
                report['exit_code'] = proc.returncode
                assert proc.returncode == 0, 'native failed; inspect stderr'
            finally:
                if proc.poll() is None:
                    proc.kill(); proc.wait(timeout=15)
        d=json.loads((args.out/'generation.json').read_text(encoding='utf-8'))
        offsets=[]; total=0
        for case,r in zip(cases,d['results']):
            offsets.append(total); total += NV*r['generated_tokens']
            case.update(native_checks=native_checks(r,d),reused=r['reused_tokens'],prompt_tokens=r['prompt_tokens'],
                        generated=r['generated_tokens'],stop=r['stop_reason'],prefill_ms=r['prefill_ms'],request_ms=r['request_ms'])
            case['pass'] = all(case['native_checks'].values()) and case['reused']==case['expected_reused']
        assert len(d['results'])==len(cases) and (args.out/'generation.f32').stat().st_size==total*4
        for first,second in [(0,1),(4,2),(4,3),(5,6),(8,7),(9,10)]:
            a,b=d['results'][first],d['results'][second]
            pair={'fresh':cases[first]['name'],'reuse':cases[second]['name'],'tokens_equal':a['token_ids']==b['token_ids']}
            if a['generated_tokens']==b['generated_tokens']:
                pair['logits']=compare_ranges(args.out/'generation.f32',offsets[first],offsets[second],NV*a['generated_tokens'])
            pair['pass']=pair['tokens_equal'] and pair.get('logits',{}).get('pass',False)
            report['pairs'].append(pair)
        report['natural_eos']=all(d['results'][i]['stop_reason']=='eos' and d['results'][i]['token_ids'][-1]==200020 for i in (5,6))
        report['pass']=report['natural_eos'] and all(c['pass'] for c in cases) and all(c['pass'] for c in report['pairs'])
        assert report['pass'], 'prefix context parity failed'
    except BaseException:
        report['error']=traceback.format_exc()
        raise
    finally:
        report['artifacts']={f.name:sha(f) for f in args.out.iterdir() if f.is_file() and f.name!='report.json'}
        save(args.out/'report.json',report)
        print('REPORT pass='+str(report['pass']),flush=True)


if __name__=='__main__':
    main()
