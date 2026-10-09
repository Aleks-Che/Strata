"""MM27-35 short full-model same-batch parity at batch1/8. Run after speed A/B."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from serve.winjob import contain
from tools.run_minimax_m2 import runtime_environment
from tools.minimax_m2_memory_observer import MemoryObserver
from tools.check_minimax_m2_prefix import save
from tools.check_minimax_m2_cache_decay import sha
from tools.check_minimax_m2_prefix_context import native_checks, NV
from tools.check_minimax_m2_router_ids import router_checks
from tools.check_minimax_m2_copy_events import event_checks
from tools.check_minimax_m2_sessions import compare_files
from tools.validate_minimax_m2_router_ids import ENGINE_SHA, TRANSPORT


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'build-local/minimax-m2-router-ids-ab-01'
    speed=json.loads((source/'report.json').read_text(encoding='utf-8'));assert speed['pass']
    assert sha(source/'engine.exe')==ENGINE_SHA
    shutil.copyfile(source/'engine.exe',a.out/'engine.exe');shutil.copyfile(Path(__file__),a.out/'driver.py')
    original=json.loads((source/'requests.json').read_text(encoding='utf-8'))
    # Same two complete prompts as the speed corpus; only shorten the outputs.
    requests=[dict(original[i],max_tokens=8) for i in [0,2]];save(a.out/'requests.json',requests)
    report={'pass':False,'stage':'MM27-35','engine_sha256':ENGINE_SHA,'runs':[],
            'speed_report_sha256':sha(source/'report.json'),'scope':'ctx2048, same-batch1/8 off/on, two complete prompts,8 outputs each'}
    env=runtime_environment(ROOT/'build-local/cuda-13.0');env['STRATA_MM27_TOKENWISE']='0';env['STRATA_MM27_STATE_BULK']='1'
    try:
        for batch in [1,8]:
            baseline=None
            for mode in [0,1]:
                name=f'b{batch}-r{mode}';base=a.out/name
                command=[str(a.out/'engine.exe'),'--gguf',speed['model'],'--ctx','2048','--batch',str(batch),'--mode','2',
                    '--gpu-cache-mib','18432','--gpu-cache-allocator','arena','--pipeline-readers','2','--pipeline-chunk-mib','4',
                    '--pipeline-lookahead','1','--pipeline-d2d-batch','1','--pipeline-events','2','--router-host-ids',str(mode),
                    '--request',str(a.out/'requests.json'),'--output',str(base.with_suffix('.json')),'--logits',str(base.with_suffix('.f32'))]
                run={'name':name,'batch':batch,'mode':mode,'command':command,'pass':False};report['runs'].append(run);save(a.out/'report.json',report)
                print('START '+name,flush=True)
                with base.with_suffix('.stdout.log').open('wb') as out,base.with_suffix('.stderr.log').open('wb') as err:
                    proc=subprocess.Popen(command,env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW);observer=None
                    try:
                        assert contain(proc);run['pid']=proc.pid
                        def stop():
                            if proc.poll() is None:proc.kill()
                        observer=MemoryObserver(base.with_suffix('.memory.jsonl'),on_error=stop);observer.start()
                        run['exit_code']=proc.wait(timeout=300);assert run['exit_code']==0,name
                    finally:
                        if proc.poll() is None:proc.kill();proc.wait(timeout=15)
                        if observer is not None:observer.close();run['observer']=observer.summary()
                assert run['observer']['samples'] and not run['observer']['error'] and run['observer']['stopped']
                header=json.loads(base.with_suffix('.json').read_text(encoding='utf-8'));run['results']=header['results'];run['checks']=[]
                assert header['batch']==batch and header['router_host_ids']==bool(mode) and header['pipeline_events']==2
                for v in run['results']:
                    checks={**native_checks(v,header),**event_checks(v,2),**router_checks(v,mode)}
                    run['checks'].append(checks);assert all(checks.values()),checks
                rows=sum(v['generated_tokens'] for v in run['results']);assert rows==16
                assert base.with_suffix('.f32').stat().st_size==rows*NV*4
                if mode:
                    assert [v['token_ids'] for v in baseline['results']]==[v['token_ids'] for v in run['results']]
                    run['logits']=compare_files(a.out/f'b{batch}-r0.f32',0,base.with_suffix('.f32'),0,rows);assert run['logits']['pass']
                    assert all(v[phase][key]==baseline['results'][i][phase][key] for i,v in enumerate(run['results']) for phase in ['prefill','decode'] for key in TRANSPORT)
                else:baseline=run
                run['pass']=True;save(a.out/'report.json',report);print('PASS '+name,flush=True)
        report['pass']=True
    except BaseException:
        report['error']=traceback.format_exc();raise
    finally:
        report['artifacts']={f.name:sha(f) for f in a.out.iterdir() if f.is_file() and f.name!='report.json'}
        save(a.out/'report.json',report)


if __name__=='__main__':main()
