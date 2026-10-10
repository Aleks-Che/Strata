"""MM27-37 full-model pinned sort-table A/B against the retained MM27-32 corpus.

One process per mode/repetition; fresh KV for A,A,B,A, warm expert history.
Full F32 output is retained and compared to the old synchronous EXE's output.
Run alone. Screening does not establish a default; repeat the selected pair.
"""
import argparse
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
from tools.check_minimax_m2_cache_decay import sha, metrics, NAMES
from tools.check_minimax_m2_prefix import save
from tools.check_minimax_m2_prefix_context import snapshot, native_checks, NV
from tools.check_minimax_m2_sessions import compare_files
from tools.minimax_m2_memory_observer import MemoryObserver
from tools.run_minimax_m2 import runtime_environment


from tools.check_minimax_m2_copy_events import event_checks
from tools.check_minimax_m2_router_ids import router_checks, summarize
from tools.validate_minimax_m2_router_ids import TRANSPORT


def table_checks(result, mode):
    checks={'table_copies':True,'table_bytes':True,'table_drained':True,'table_bounded':True,'table_no_fallback':True}
    for phase in ['prefill','decode']:
        s=result[phase];n=s['pipeline_matrices']
        checks['table_copies'] &= s['sort_table_copies']==(n if mode else 0)
        # The last MoE block keeps only the requested output row per prefill
        # microbatch; preceding 61 blocks process every input row (3 matrices).
        rows=183*result['evaluated_prompt_tokens']+n//62 if phase=='prefill' else 186*(result['generated_tokens']-1)
        checks['table_bytes'] &= s['sort_table_bytes']==(64*rows if mode else 0)
        checks['table_drained'] &= s['sort_table_pending']==0
        checks['table_bounded'] &= (s['sort_table_pinned_bytes']==16384 and
            (1<=s['sort_table_peak_pending']<=4 if n else s['sort_table_peak_pending']==0)) if mode else (
            s['sort_table_pinned_bytes']==s['sort_table_peak_pending']==s['sort_table_reuse_waits']==s['sort_table_drain_waits']==0)
        checks['table_no_fallback'] &= s['sort_table_fallbacks']==0
    return checks



def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=ROOT/'build-local/minimax-m2-cache-decay-01')
    p.add_argument('--engine',type=Path,default=ROOT/'build-local/minimax-m2-sort-table-candidate/engine.exe')
    p.add_argument('--cuda-root',type=Path,default=ROOT/'build-local/cuda-13.0')
    p.add_argument('--modes',type=int,nargs='+',default=[0,1]);p.add_argument('--repeats',type=int,choices=[1,3],default=1)
    a=p.parse_args();a.out=a.out.resolve()
    if a.modes != [0,1]:p.error('modes must be 0 1 (sort-table async off/on)')
    old=json.loads((a.reference/'report.json').read_text(encoding='utf-8'));assert old['pass']
    for name in ['requests.json','baseline.json','baseline.f32']:
        assert sha(a.reference/name)==old['artifacts'][name],name
    reference=json.loads((a.reference/'baseline.json').read_text(encoding='utf-8'))['results']
    gguf=Path(old['model']);assert gguf.stat().st_size==old['model_size']
    with gguf.open('rb') as f:assert __import__('hashlib').sha256(f.read(8287960)).hexdigest()==old['header_sha256']
    a.out.mkdir(parents=True,exist_ok=False)
    report={'pass':False,'stage':'MM27-37','model':str(gguf),'reference_report_sha256':sha(a.reference/'report.json'),
            'reference_logits_sha256':old['artifacts']['baseline.f32'],'engine_sha256':sha(a.engine),
            'modes':a.modes,'repeats':a.repeats,'runs':[],
            'scope':'sequential fresh processes; ctx2048/batch16/cache18 GiB/decay65536; events2; A,A,B,A x128; router IDs on; pinned table off/on; OS file cache uncontrolled'}
    snapshot(a.out,report)
    for source in [Path(__file__),ROOT/'tools/check_minimax_m2_router_ids.py',ROOT/'tools/validate_minimax_m2_router_ids.py',ROOT/'tools/check_minimax_m2_copy_events.py',ROOT/'tools/check_minimax_m2_cache_decay.py',ROOT/'tools/check_minimax_m2_sessions.py',
                   ROOT/'tools/minimax_m2_memory_observer.py',ROOT/'backends/step35/expert_cache.hpp',
                   ROOT/'backends/common/expert_frequency.hpp',ROOT/'backends/common/expert_pipeline.hpp',
                   ROOT/'backends/common/expert_file.hpp',ROOT/'backends/hy3/gpu_arena.hpp']:
        rel=source.resolve().relative_to(ROOT);dest=a.out/'sources'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,dest);report['sources'][rel.as_posix()]=sha(dest)
    shutil.copyfile(a.engine,a.out/'engine.exe');shutil.copyfile(a.engine.parent/'minimax-m2-build-manifest.json',a.out/'manifest.json')
    shutil.copyfile(a.reference/'requests.json',a.out/'requests.json')
    env=runtime_environment(a.cuda_root);env['STRATA_MM27_TOKENWISE']='0';env['STRATA_MM27_STATE_BULK']='1'
    report['environment']={k:v for k,v in env.items() if k.startswith(('STRATA_','GGML_','LLAMA_','NVIDIA_'))}
    report['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
    try:
        for repeat in range(1,a.repeats+1):
            for mode in a.modes if repeat%2 else a.modes[::-1]:
                name=f'r{repeat}-e{mode}';base=a.out/name
                command=[str(a.out/'engine.exe'),'--gguf',str(gguf),'--ctx','2048','--batch','16','--mode','2',
                    '--gpu-cache-mib','18432','--gpu-cache-allocator','arena','--cache-decay-period','65536',
                    '--pipeline-readers','2','--pipeline-chunk-mib','4','--pipeline-lookahead','1','--pipeline-d2d-batch','1',
                    '--pipeline-events','2','--router-host-ids','1','--sort-table-async',str(mode),'--request',str(a.out/'requests.json'),'--output',str(base.with_suffix('.json')),
                    '--logits',str(base.with_suffix('.f32'))]
                run={'name':name,'repeat':repeat,'mode':mode,'command':command,'pass':False};report['runs'].append(run)
                print('START '+name,flush=True);save(a.out/'report.json',report)
                with base.with_suffix('.stdout.log').open('wb') as out,base.with_suffix('.stderr.log').open('wb') as err:
                    proc=subprocess.Popen(command,stdout=out,stderr=err,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
                    observer=None
                    try:
                        assert contain(proc),'job containment';run['pid']=proc.pid
                        def stop():
                            if proc.poll() is None:proc.kill()
                        observer=MemoryObserver(base.with_suffix('.memory.jsonl'),on_error=stop);observer.start()
                        save(a.out/'report.json',report);start=time.monotonic();last=-1
                        while proc.poll() is None:
                            f=base.with_suffix('.f32');rows=f.stat().st_size//(NV*4) if f.exists() else 0
                            if rows//128!=last:last=rows//128;print(f'{name}: {rows} logit rows',flush=True)
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
                assert data['pipeline_events']==2 and data['router_host_ids'] is True and data['sort_table_async']==bool(mode) and data['cache_decay_period']==65536 and len(results)==4
                run['results']=results;run['checks']=[];rows=0
                for r in results:
                    checks={**native_checks(r,data),**event_checks(r,2),**router_checks(r,1),**table_checks(r,mode)}
                    checks['fresh_kv']=r['reused_tokens']==0 and not r['session_restore']
                    checks['prefill_no_fill']=r['prefill']['cache_fill_bytes']==r['prefill']['cache_evictions']==0
                    run['checks'].append(checks);assert all(checks.values()),checks;rows+=r['generated_tokens']
                assert base.with_suffix('.f32').stat().st_size==rows*NV*4
                assert [r['token_ids'] for r in results]==[r['token_ids'] for r in reference]
                run['logits']=compare_files(a.reference/'baseline.f32',0,base.with_suffix('.f32'),0,rows)
                assert run['logits']['pass']
                if len(report['runs'])>1:
                    first=report['runs'][0]['results']
                    assert all(r[phase][key]==first[i][phase][key] for i,r in enumerate(results) for phase in ['prefill','decode'] for key in TRANSPORT),'changed expert transport'
                run['metrics']=metrics(results);run['pass']=True
                save(a.out/'report.json',report);print('PASS '+name+' '+json.dumps(run['metrics']),flush=True)
        report['summary']=summarize(report['runs'],a.modes);report['pass']=True
    except BaseException:
        report['error']=traceback.format_exc();raise
    finally:
        report['artifacts']={f.name:sha(f) for f in a.out.iterdir() if f.is_file() and f.name!='report.json'}
        save(a.out/'report.json',report);print('REPORT pass='+str(report['pass']),flush=True)


if __name__=='__main__':main()
