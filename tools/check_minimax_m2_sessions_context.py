"""Opt-in MM27-28 full-logit audit: RAM restore at 2K/4K, EOS and seeded sampling."""
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

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from serve.minimax_m2_engine import EXE_SHA256, EVENTS_EXE_SHA256
from serve.winjob import contain
from tools.check_minimax_m2_prefix import save
from tools.check_minimax_m2_prefix_context import make_corpus as prefix_corpus, snapshot, native_checks
from tools.check_minimax_m2_sessions import compare_files
from tools.run_minimax_m2 import runtime_environment
from tools.strata_tokenizer import Tokenizer
from tools.check_minimax_m2_copy_events import event_checks
from tools.minimax_m2_memory_observer import MemoryObserver


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def make_corpus(tok):
    old,_,http=prefix_corpus(tok)
    small,large,short=old[0]['tokens'],old[4]['tokens'],old[5]['tokens']
    sampled=old[8]['sampling']
    specs=[('2k-cold',small,'a',8,{},0,False,0),
           ('short-between-2k',short,'s',8,{},0,False,5),
           ('2k-restored',small,'a',8,{},2016,True,0),
           ('4k-cold',large,'b',8,{},0,False,4),
           ('short-between-4k',short,'s',8,{},48,True,5),
           ('4k-restored',large,'b',8,{},4064,True,4),
           ('eos-cold',short,'e',512,{},0,False,5),
           ('short-between-eos',short,'s',8,{},48,True,5),
           ('eos-restored',short,'e',512,{},48,True,5),
           ('sample-fresh',short,None,128,sampled,0,False,8),
           ('sample-restored',short,'e',128,sampled,48,True,8),
           ('one-cold',short,'one',1,{},0,False,9),
           ('4k-restored-again',large,'b',8,{},4064,True,4),
           ('one-restored',short,'one',1,{},48,True,9)]
    requests=[{'tokens':ids,'max_tokens':n,'sampling':sampling,
               **({'session_key':hashlib.sha256(key.encode()).hexdigest()} if key else {})}
              for _,ids,key,n,sampling,_,_,_ in specs]
    cases=[{'name':name,'expected_reused':reuse,'expected_restore':restore,'reference_index':ref}
           for name,_,_,_,_,reuse,restore,ref in specs]
    return requests,cases,http


def inspect_case(result,header,expected):
    checks=native_checks(result,header)
    checks['reuse']=result['reused_tokens']==expected['expected_reused']
    checks['restore']=type(result['session_restore']) is bool and result['session_restore']==expected['expected_restore']
    checks['restore_bytes']=(result['session_restored_bytes']>0)==result['session_restore']
    checks['archive_limits']=0<=result['session_archive_bytes']<=header['session_cache_mib']*2**20 and 0<=result['session_archive_entries']<=header['session_cache_slots']
    checks['session_time_included']=0<=result['session_ms']<=result['request_ms'] and result['ttft_ms']>=result['session_ms']
    return checks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gguf',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=ROOT/'build-local/minimax-m2-prefix-context-01')
    p.add_argument('--cuda-root',type=Path,default=ROOT/'build-local/cuda-13.0')
    p.add_argument('--stage',default='MM27-28 offline',help='Evidence label; does not change the corpus')
    p.add_argument('--engine',type=Path,default=ROOT/'build-local/minimax-m2-cuda/bin/strata-minimax-m2-bench.exe')
    p.add_argument('--pipeline-events',type=int,choices=(0,2),default=0)
    args=p.parse_args();binary=args.engine.resolve();digest=sha(binary)
    old=json.loads((args.reference/'report.json').read_text(encoding='utf-8'))
    assert old['pass'] and digest in (EXE_SHA256,EVENTS_EXE_SHA256)
    assert not args.pipeline_events or digest==EVENTS_EXE_SHA256
    for name in ['generation.json','generation.f32','requests.json']:
        assert sha(args.reference/name)==old['artifacts'][name],name
    args.out.mkdir(parents=True,exist_ok=False)
    report={'pass':False,'stage':args.stage,'engine_sha256':digest,'pipeline_events':args.pipeline_events,
            'reference_report_sha256':sha(args.reference/'report.json'),'reference_engine_sha256':old['engine_sha256'],
            'model':str(args.gguf.resolve()),'cases':[],'pairs':[]}
    snapshot(args.out,report);shutil.copyfile(binary,args.out/'engine.exe')
    for source in [Path(__file__),ROOT/'tools/check_minimax_m2_sessions.py',ROOT/'tools/check_minimax_m2_copy_events.py',
                   ROOT/'tools/minimax_m2_memory_observer.py']:
        dest=args.out/'sources'/source.resolve().relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,dest);report['sources'][source.resolve().relative_to(ROOT).as_posix()]=sha(dest)
    env=runtime_environment(args.cuda_root);env['STRATA_MM27_TOKENWISE']='0';env['STRATA_MM27_STATE_BULK']='1'
    report['environment']={k:v for k,v in env.items() if k.startswith(('STRATA_','GGML_','LLAMA_','NVIDIA_'))}
    report['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
    try:
        requests,cases,http=make_corpus(Tokenizer.from_gguf(args.gguf));report['cases']=cases
        save(args.out/'requests.json',requests);save(args.out/'http-requests.json',http)
        command=[str(binary),'--gguf',str(args.gguf.resolve()),'--ctx','4096','--batch','16','--mode','2',
                 '--prefix-cache','1','--session-cache-mib','6144','--session-cache-slots','4',
                 '--gpu-cache-mib','18432','--gpu-cache-allocator','arena','--pipeline-readers','2',
                 '--pipeline-chunk-mib','4','--pipeline-lookahead','1','--pipeline-d2d-batch','1',
                 '--request',str((args.out/'requests.json').resolve()),'--output',str((args.out/'generation.json').resolve()),
                 '--logits',str((args.out/'generation.f32').resolve())]
        if digest==EVENTS_EXE_SHA256:command+=['--pipeline-events',str(args.pipeline_events)]
        report['command']=command;save(args.out/'report.json',report)
        print('START ctx4096 RAM archive: '+str(len(cases))+' cases',flush=True)
        with (args.out/'stdout.log').open('wb') as out,(args.out/'stderr.log').open('wb') as err:
            proc=subprocess.Popen(command,stdout=out,stderr=err,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
            observer=None
            try:
                if not contain(proc):raise RuntimeError('cannot contain native child')
                def stop_child():
                    if proc.poll() is None:proc.kill()
                observer=MemoryObserver(args.out/'physical-commit-monitor.jsonl',on_error=stop_child);observer.start()
                report['native_pid']=proc.pid;save(args.out/'report.json',report)
                end=time.monotonic()+3600
                while proc.poll() is None:
                    if time.monotonic()>end:raise TimeoutError('one hour native limit')
                    time.sleep(2)
                report['exit_code']=proc.returncode;assert proc.returncode==0,'inspect stderr'
            finally:
                if proc.poll() is None:proc.kill();proc.wait(timeout=15)
                if observer:observer.close();report['independent_monitor']=observer.summary()
            assert report['independent_monitor']['samples'] and not report['independent_monitor']['error']
        data=json.loads((args.out/'generation.json').read_text(encoding='utf-8'))
        if digest==EVENTS_EXE_SHA256:assert data['pipeline_events']==args.pipeline_events
        reference=json.loads((args.reference/'generation.json').read_text(encoding='utf-8'))['results']
        assert len(data['results'])==len(cases)
        offsets=[];total=0
        for r in reference:offsets.append(total);total+=r['generated_tokens']
        row=0;rows=[]
        for case,r in zip(cases,data['results']):
            rows.append(row);idx=case['reference_index'];count=r['generated_tokens']
            case['checks']=inspect_case(r,data,case)
            if digest==EVENTS_EXE_SHA256:case['checks'].update(event_checks(r,args.pipeline_events))
            case['logits']=compare_files(args.reference/'generation.f32',offsets[idx],args.out/'generation.f32',row,count)
            case['tokens_equal']=r['token_ids']==reference[idx]['token_ids'][:count]
            case['pass']=all(case['checks'].values()) and case['tokens_equal'] and case['logits']['pass']
            row+=count
        assert (args.out/'generation.f32').stat().st_size==row*200064*4
        for first,second in [(0,2),(3,5),(3,12),(6,8),(9,10),(11,13)]:
            a,b=data['results'][first],data['results'][second]
            pair={'fresh':cases[first]['name'],'restored':cases[second]['name'],'tokens_equal':a['token_ids']==b['token_ids']}
            pair['logits']=compare_files(args.out/'generation.f32',rows[first],args.out/'generation.f32',rows[second],a['generated_tokens'])
            pair['pass']=pair['tokens_equal'] and pair['logits']['pass'];report['pairs'].append(pair)
        report['natural_eos']=all(data['results'][i]['stop_reason']=='eos' and data['results'][i]['generated_tokens']==414 for i in [6,8])
        report['pass']=report['natural_eos'] and all(c['pass'] for c in cases) and all(p['pass'] for p in report['pairs'])
        assert report['pass'],[c['name'] for c in cases if not c['pass']]
    except BaseException:
        report['error']=traceback.format_exc();raise
    finally:
        report['artifacts']={f.name:sha(f) for f in args.out.iterdir() if f.is_file() and f.name!='report.json'}
        save(args.out/'report.json',report);print('REPORT pass='+str(report['pass']),flush=True)


if __name__=='__main__':main()
