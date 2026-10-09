"""MM27-32: sequential three-period cache experiment with retained full logits.

One old-EXE correctness run, then three fresh processes per period in balanced
Latin-square order. GPU cache starts empty; OS file cache is uncontrolled.
The corpus is A, A, B, A with fresh KV each time, preserving only expert history.
No server/default promotion is performed by this diagnostic.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from serve.winjob import contain
from tools.check_minimax_m2_prefix import save
from tools.check_minimax_m2_prefix_context import snapshot, native_checks, NV
from tools.check_minimax_m2_sessions import compare_files
from tools.gguf_reader import GGUFFile
from tools.minimax_m2_loader_contract import validate_loader_contract
from tools.minimax_m2_template import renderer, text_context
from tools.minimax_m2_memory_observer import MemoryObserver
from tools.run_minimax_m2 import runtime_environment

PERIODS=(65536,131072,262144)
NAMES=('a_empty','a_repeat','b_new_topic','a_return')


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def schedule(rounds):
    return [(r+1,p) for r in range(rounds) for p in PERIODS[r%3:]+PERIODS[:r%3]]


def metrics(results):
    steps=sum(r['decode_forward_tokens'] for r in results)
    ms=sum(r['decode_ms'] for r in results)
    dec=[r['decode'] for r in results]
    hit=sum(s['cache_hit_bytes'] for s in dec);h2d=sum(s['h2d_bytes'] for s in dec)
    return {'decode_tok_s':1000*steps/ms,'decode_ms':ms,'decode_steps':steps,
            'request_ms':sum(r['request_ms'] for r in results),
            'prefill_ms':sum(r['prefill_ms'] for r in results),
            'decode_h2d_gib':h2d/2**30,'decode_hit_percent':100*hit/(hit+h2d),
            'decode_evictions':sum(s['cache_evictions'] for s in dec),
            'decode_fill_gib':sum(s['cache_fill_bytes'] for s in dec)/2**30,
            'decode_delivery_ms':sum(s['pipeline_delivery_ms'] for s in dec)}


def summarize(runs):
    output=[]
    for idx,name in [(-1,'all_requests'),*enumerate(NAMES)]:
        item={'name':name,'periods':{}}
        for p in PERIODS:
            samples=[metrics(r['results'] if idx==-1 else [r['results'][idx]])
                     for r in runs if r['round']>0 and r['period']==p]
            item['periods'][str(p)]={'samples':samples,
                'median':{k:statistics.median(s[k] for s in samples) for k in samples[0]}}
        base=item['periods']['65536']['median']
        for p in PERIODS:
            entry=item['periods'][str(p)];med=entry['median']
            entry['decode_gain_percent']=100*(med['decode_tok_s']/base['decode_tok_s']-1)
            entry['request_time_reduction_percent']=100*(1-med['request_ms']/base['request_ms'])
        output.append(item)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gguf',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--engine',type=Path,default=ROOT/'build-local/minimax-m2-cache-decay-candidate/engine.exe')
    p.add_argument('--baseline',type=Path,default=ROOT/'build-local/minimax-m2-cache-decay-baseline/engine.exe')
    p.add_argument('--cuda-root',type=Path,default=ROOT/'build-local/cuda-13.0')
    p.add_argument('--rounds',type=int,choices=[1,2,3],default=3)
    p.add_argument('--tokens',type=int,choices=[128,256],default=128)
    args=p.parse_args();args.out=args.out.resolve();args.gguf=args.gguf.resolve()
    args.out.mkdir(parents=True,exist_ok=False)
    report={'pass':False,'stage':'MM27-32','model':str(args.gguf),'model_size':args.gguf.stat().st_size,
            'engine_sha256':sha(args.engine),'baseline_sha256':sha(args.baseline),
            'context':2048,'batch':16,'cache_mib':18432,'names':NAMES,'tokens_requested':args.tokens,
            'method':'old EXE correctness only; fresh candidate process per period/round; Latin-square order; OS file cache uncontrolled',
            'runs':[],'cli_checks':[]}
    snapshot(args.out,report)
    for source in [Path(__file__),ROOT/'tools/check_minimax_m2_sessions.py',ROOT/'tools/minimax_m2_memory_observer.py',
                   ROOT/'backends/step35/expert_cache.hpp',ROOT/'backends/common/expert_frequency.hpp',
                   ROOT/'backends/hy3/gpu_arena.hpp',ROOT/'backends/common/expert_file.hpp']:
        rel=source.resolve().relative_to(ROOT);dest=args.out/'sources'/rel
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest);report['sources'][rel.as_posix()]=sha(dest)
    for source,name in [(args.engine,'candidate.exe'),(args.baseline,'baseline.exe')]:
        shutil.copyfile(source,args.out/name)
        manifest=source.parent/'minimax-m2-build-manifest.json'
        if manifest.is_file():shutil.copyfile(manifest,args.out/(name+'.manifest.json'))
    env=runtime_environment(args.cuda_root);env['STRATA_MM27_TOKENWISE']='0';env['STRATA_MM27_STATE_BULK']='1'
    report['environment']={k:v for k,v in env.items() if k.startswith(('STRATA_','GGML_','LLAMA_','NVIDIA_'))}
    report['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
    try:
        for value,valid in [('0',False),('-1',False),('1x',False),('4294967296',False),('18446744073709551616',False),
                            ('1',True),('65536',True),('131072',True),('262144',True),('4294967295',True)]:
            result=subprocess.run([str(args.out/'candidate.exe'),'--cache-decay-period',value,'--version'],
                                  env=env,capture_output=True,text=True,timeout=15,creationflags=subprocess.CREATE_NO_WINDOW)
            check={'value':value,'valid':valid,'exit':result.returncode,'stderr':result.stderr,
                   'pass':result.returncode==(0 if valid else 2)}
            report['cli_checks'].append(check);assert check['pass'],check
        g=GGUFFile(args.gguf);contract=validate_loader_contract(g.metadata,g.tensors)
        with args.gguf.open('rb') as f:report['header_sha256']=hashlib.sha256(f.read(g.header_end)).hexdigest()
        report['source_revision']=contract['source_revision'];template=renderer(g.metadata['tokenizer.chat_template'])
        def request(text):
            return {'prompt':template.render(**text_context({'messages':[{'role':'user','content':text}],
                                                            'add_generation_prompt':True})), 'max_tokens':args.tokens}
        a=request('Explain in a few sentences why the sky appears blue during the day.')
        b=request('Write a Python LRU cache with a fixed capacity using OrderedDict. Explain get and put, then show a short usage example.')
        save(args.out/'requests.json',[a,a,b,a])
        for round_no,period in [(0,65536),*schedule(args.rounds)]:
            name='baseline' if round_no==0 else f'r{round_no}-p{period}'
            base=args.out/name;exe='baseline.exe' if round_no==0 else 'candidate.exe'
            command=[str(args.out/exe),'--gguf',str(args.gguf),'--ctx','2048','--batch','16','--mode','2',
                     '--prefix-cache','0','--session-cache-mib','0','--ram-cache-mib','0',
                     '--gpu-cache-mib','18432','--gpu-cache-allocator','arena','--arena-block-mib','64',
                     '--arena-growth-reserve-mib','0','--cache-group-experts','0','--expert-reader','file',
                     '--pipeline-readers','2','--pipeline-chunk-mib','4','--pipeline-lookahead','1','--pipeline-d2d-batch','1',
                     '--request',str(args.out/'requests.json'),'--output',str(base.with_suffix('.json')),
                     '--logits',str(base.with_suffix('.f32'))]
            # Round 1 at 65536 also tests the new EXE's omitted/default setting.
            if round_no>0 and (round_no,period)!=(1,65536):command+=['--cache-decay-period',str(period)]
            run={'name':name,'round':round_no,'period':period,'command':command,'pass':False}
            report['runs'].append(run);save(args.out/'report.json',report);print('START '+name,flush=True)
            with base.with_suffix('.stdout.log').open('wb') as out,base.with_suffix('.stderr.log').open('wb') as err:
                proc=subprocess.Popen(command,stdout=out,stderr=err,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
                observer=None
                try:
                    assert contain(proc),'job containment';run['pid']=proc.pid
                    def stop_child():
                        if proc.poll() is None:proc.kill()
                    observer=MemoryObserver(base.with_suffix('.memory.jsonl'),on_error=stop_child);observer.start()
                    save(args.out/'report.json',report);start=time.monotonic();last=-1
                    while proc.poll() is None:
                        f=base.with_suffix('.f32');rows=f.stat().st_size//(NV*4) if f.exists() else 0
                        if rows//64!=last:
                            last=rows//64;print(f'{name}: {rows} logit rows',flush=True)
                        if time.monotonic()-start>1200:raise TimeoutError(name)
                        time.sleep(2)
                    run['exit_code']=proc.returncode;assert proc.returncode==0,'inspect '+name+'.stderr.log'
                finally:
                    if proc.poll() is None:proc.kill();proc.wait(timeout=15)
                    if observer is not None:observer.close();run['observer']=observer.summary()
            assert run['observer']['samples']>0 and not run['observer']['error'] and run['observer']['stopped']
            memory=[json.loads(line) for line in base.with_suffix('.memory.jsonl').read_text().splitlines()]
            run['observed_peak_percent']={axis:max(100*(1-s[axis+'_free']/s[axis+'_total']) for s in memory) for axis in ['ram','gpu']}
            data=json.loads(base.with_suffix('.json').read_text(encoding='utf-8'));results=data['results']
            assert len(results)==4 and data.get('cache_decay_period',65536)==period
            assert data['context']==2048 and data['batch']==16 and data['strict_f32'] and not data['graphs'] and not data['mtp']
            run['results']=results;run['checks']=[];rows=0
            for r in results:
                checks=native_checks(r,data)
                checks['fresh_kv']=r['reused_tokens']==0 and not r['session_restore']
                checks['no_prefill_training_fills']=r['prefill']['cache_fill_bytes']==r['prefill']['cache_evictions']==0
                run['checks'].append(checks);assert all(checks.values()),checks;rows+=r['generated_tokens']
            assert base.with_suffix('.f32').stat().st_size==rows*NV*4
            if round_no==0:
                reference=results
                run['logits']=compare_files(base.with_suffix('.f32'),0,base.with_suffix('.f32'),0,rows)
            else:
                assert [r['token_ids'] for r in results]==[r['token_ids'] for r in reference],'token ID mismatch'
                run['logits']=compare_files(args.out/'baseline.f32',0,base.with_suffix('.f32'),0,rows)
            assert run['logits']['pass'];run['pass']=True;run['metrics']=metrics(results)
            save(args.out/'report.json',report)
            print('PASS '+name+' '+json.dumps(run['metrics']),flush=True)
        report['summary']=summarize(report['runs']);report['pass']=True
    except BaseException:
        report['error']=traceback.format_exc();raise
    finally:
        report['artifacts']={f.name:sha(f) for f in args.out.iterdir() if f.is_file() and f.name!='report.json'}
        save(args.out/'report.json',report);print('REPORT pass='+str(report['pass']),flush=True)


if __name__=='__main__':main()
