"""Sampling JSONL validation, seed isolation and targeted Windows cancel/recovery."""
import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_minimax_m2_completion_pipe import result_checks
from tools.run_minimax_m2 import runtime_environment
from tools.strata_tokenizer import Tokenizer
from tools.tune_minimax_m2_pipeline import save, sha


def worker(args):
    if not ctypes.windll.kernel32.GetConsoleCP():
        raise RuntimeError('isolated supervisor requires its hidden console')
    screen = json.loads((args.source/'sampling-report.json').read_text(encoding='utf-8'))
    ref = next(c for c in screen['cases'] if c['name'] == 'english-42')
    if not (screen['functional_pass'] and ref['quality']['pass'] and ref['finite_logits'] and ref['sampling_ok']):
        raise ValueError('selected English reference and numerical gates must pass first')
    data = json.loads(Path(ref['report']).read_text(encoding='utf-8'))
    expected = data['results'][0]
    original = next(c['request'] for g in screen['groups'] for c in g if c['name'] == 'english-42')
    request = dict(original, max_tokens=64)
    ids = expected['token_ids'][:64]
    assert len(ids) > 8
    tokenizer = Tokenizer.from_gguf(screen['model'])
    expected_text = tokenizer.decode(ids)
    command = [str((args.out/'engine.exe').resolve()), '--gguf', screen['model'], '--pipe', '--ctx', '2048', '--batch', '16',
               '--gpu-cache-mib', '18432', '--gpu-cache-allocator', 'arena', '--pipeline-readers', '2',
               '--pipeline-chunk-mib', '4', '--pipeline-lookahead', '1', '--pipeline-d2d-batch', '1']
    env = runtime_environment(args.cuda_root);env['STRATA_MM27_TOKENWISE'] = '0'
    report = {'pass': False, 'scope': 'one live process, invalid sampling requests, seed switch, CTRL_BREAK after 8 tokens, prefix replay; no pipe logits',
              'engine_sha256': sha(args.out/'engine.exe'), 'driver_sha256': sha(Path(__file__)),
              'reference': ref['report'], 'reference_sha256': sha(Path(ref['report'])), 'command': command, 'cases': []}
    process = None
    try:
        with (args.out/'events.jsonl').open('w', encoding='utf-8') as transcript, (args.out/'engine.stderr.log').open('w', encoding='utf-8') as stderr:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                                       env=env, encoding='utf-8', creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            inbox = queue.Queue()
            def reader():
                try:
                    for line in process.stdout:
                        inbox.put((time.perf_counter(), line))
                finally:
                    inbox.put((time.perf_counter(), None))
            thread = threading.Thread(target=reader, daemon=True);thread.start()
            def event(timeout=180):
                when,line = inbox.get(timeout=timeout)
                if line is None:
                    raise RuntimeError('unexpected engine EOF')
                transcript.write(line);transcript.flush()
                return when,json.loads(line)
            _,header = event(300);assert header['event'] == 'ready';report['header'] = header
            assert all(header[k] == data[k] for k in ['context','batch','kv','strict_f32','flash_attention',
                'graphs','gpu_cache_mib','gpu_cache_allocator','pipeline_readers','pipeline_chunk_mib',
                'pipeline_lookahead','pipeline_d2d_batch','ram_cache_mib','cache_group_experts'])
            def run(name,r,cancel_after=0):
                process.stdin.write(json.dumps(r,ensure_ascii=False)+'\n');process.stdin.flush()
                emitted,signal_at,start = [],None,time.perf_counter()
                while True:
                    remaining = 600-(time.perf_counter()-start)
                    if remaining <= 0:
                        raise TimeoutError(name)
                    when,e = event(min(180,remaining))
                    if e['event'] == 'token':
                        emitted.append(e['id'])
                        if cancel_after and len(emitted) >= cancel_after and signal_at is None:
                            signal_at = time.perf_counter()
                            if not ctypes.windll.kernel32.GenerateConsoleCtrlEvent(1,process.pid):
                                raise ctypes.WinError()
                    elif e['event'] in ('result','error'):
                        return {'name':name,'request':r,'streamed_token_ids':emitted,'terminal':e,
                                'cancel_latency_ms':(when-signal_at)*1000 if signal_at else None}
                    else:
                        raise RuntimeError('unexpected protocol event')
            def record(c):
                report['cases'].append(c);save(args.out/'pipe-report.json',report)
                print(c['name']+': '+('PASS' if c['pass'] else 'FAIL'),flush=True)
                if not c['pass']:
                    raise RuntimeError('failed gate: '+c['name'])
            bad = [None,True,[],{'temperature':True},{'temperature':.0001},{'temperature':3},
                   {'top_p':0},{'top_k':200065},{'seed':4294967295},{'min_p':.1}]
            for i,config in enumerate(bad):
                c = run(f'invalid_sampling_{i}',dict(request,sampling=config))
                c['pass'] = c['terminal']['event']=='error' and not c['streamed_token_ids'];record(c)
            def valid(name,r,reference=True):
                c = run(name,r);result = c['terminal'];checks = result_checks(result,header)
                checks['sampling_config'] = all(math.isclose(result['sampling'][k],v,rel_tol=1e-7,abs_tol=1e-8) for k,v in r['sampling'].items())
                checks['algorithm'] = result['sampling_algorithm']=='temperature->top_k->top_p->dist'
                checks['events'] = c['streamed_token_ids']==result['token_ids'] and len(c['streamed_token_ids'])==result['generated_tokens']
                checks['budget'] = result['max_tokens']==r['max_tokens'] and 1<=result['generated_tokens']<=r['max_tokens']
                if reference:
                    checks['reference_ids_text'] = result['token_ids']==ids and result['text']==expected_text
                    checks['stop'] = result['stop_reason']==('eos' if ids[-1]==200020 else 'length')
                c['checks'] = checks;c['pass'] = all(checks.values());record(c)
            valid('seed42_after_invalid',request)
            valid('different_seed_request',dict(request,max_tokens=16,sampling=dict(request['sampling'],seed=7)),False)
            c = run('cancel_sampling_after_8',request,cancel_after=8);emitted = c['streamed_token_ids']
            c['pass'] = (c['terminal']['event']=='error' and 'cancelled' in c['terminal']['message'] and
                         8<=len(emitted)<len(ids) and emitted==ids[:len(emitted)] and
                         c['cancel_latency_ms'] is not None and 0<=c['cancel_latency_ms']<30000)
            record(c)
            valid('seed42_after_cancel',request)
            process.stdin.close();report['exit_code'] = process.wait(timeout=90);thread.join(timeout=5)
            extras = []
            while not inbox.empty():
                _,line = inbox.get_nowait()
                if line is not None:extras.append(line)
            report['extra_events'] = extras
            report['pass'] = report['exit_code']==0 and not extras and all(c['pass'] for c in report['cases'])
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        if process and process.poll() is None:
            process.kill();process.wait(timeout=30)  # Only our failing test child.
        save(args.out/'pipe-report.json',report)
    return 0 if report['pass'] else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--engine',type=Path,default=ROOT/'build-local/minimax-m2-cuda/bin/strata-minimax-m2-bench.exe')
    p.add_argument('--cuda-root',type=Path,default=ROOT/'build-local/cuda-13.0')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args = p.parse_args()
    if os.name!='nt':p.error('Windows CTRL_BREAK check requires Windows')
    if args.worker:return worker(args)
    args.out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(args.engine,args.out/'engine.exe');shutil.copyfile(Path(__file__),args.out/Path(__file__).name)
    command = [sys.executable,'-X','utf8',str(Path(__file__).resolve()),'--worker','--source',str(args.source.resolve()),
               '--out',str(args.out.resolve()),'--cuda-root',str(args.cuda_root.resolve())]
    save(args.out/'supervisor-command.json',command)
    startup = subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=subprocess.SW_HIDE
    with (args.out/'supervisor.stdout.log').open('w',encoding='utf-8') as stdout,(args.out/'supervisor.stderr.log').open('w',encoding='utf-8') as stderr:
        r = subprocess.run(command,stdout=stdout,stderr=stderr,startupinfo=startup,creationflags=subprocess.CREATE_NEW_CONSOLE)
    print('sampling pipe exit:',r.returncode);return r.returncode


if __name__=='__main__':
    raise SystemExit(main())
