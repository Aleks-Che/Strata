"""Opt-in MM27-29: full-logit 2K/4K restore audit and three warm A/B observations.

Runs two native processes sequentially, with identical cold setup and request
order. Repetitions share each process's warm expert cache; they are not six
independent process launches. Both modes are checked against the old MM27-28 EXE.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from serve.winjob import contain
from tools.check_minimax_m2_prefix import save
from tools.check_minimax_m2_prefix_context import snapshot, native_checks, NV
from tools.check_minimax_m2_sessions import compare_files
from tools.run_minimax_m2 import runtime_environment


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gguf',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=ROOT/'build-local/minimax-m2-sessions-context-01')
    p.add_argument('--cuda-root',type=Path,default=ROOT/'build-local/cuda-13.0')
    args=p.parse_args()
    binary=ROOT/'build-local/minimax-m2-cuda/bin/strata-minimax-m2-bench.exe'
    old=json.loads((args.reference/'report.json').read_text(encoding='utf-8'))
    assert old['pass'] and old['engine_sha256']=='275e89f8628ceecd8f0599ba92dda1abae979ea2802bfb9bf32f3669344a2ba3'
    for name in ['generation.json','generation.f32','requests.json']:
        assert sha(args.reference/name)==old['artifacts'][name],name
    requests=json.loads((args.reference/'requests.json').read_text(encoding='utf-8'))
    reference=json.loads((args.reference/'generation.json').read_text(encoding='utf-8'))['results']
    offsets=[];rows=0
    for r in reference:offsets.append(rows);rows+=r['generated_tokens']
    # Repeated pairs include saving the outgoing snapshot, clearing GPU KV,
    # restoring the target, residual prefill and eight output tokens.
    indices=[0,3,1]*4
    corpus=[requests[i] for i in indices]
    args.out.mkdir(parents=True,exist_ok=False)
    report={'pass':False,'stage':'MM27-29','engine_sha256':sha(binary),
            'reference_report_sha256':sha(args.reference/'report.json'),
            'reference_engine_sha256':old['engine_sha256'],'model':str(args.gguf.resolve()),
            'method':'sequential legacy then bulk processes; 3 warm switches per target per mode; same cold setup',
            'runs':{},'comparisons':[]}
    snapshot(args.out,report);shutil.copyfile(binary,args.out/'engine.exe')
    for source in [Path(__file__),ROOT/'tools/check_minimax_m2_sessions.py']:
        rel=source.resolve().relative_to(ROOT);dest=args.out/'sources'/rel
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest);report['sources'][rel.as_posix()]=sha(dest)
    save(args.out/'requests.json',corpus)
    env=runtime_environment(args.cuda_root);env['STRATA_MM27_TOKENWISE']='0'
    report['environment']={k:v for k,v in env.items() if k.startswith(('STRATA_','GGML_','LLAMA_','NVIDIA_'))}
    report['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
    try:
        for mode in ['legacy','bulk']:
            env['STRATA_MM27_STATE_BULK']='0' if mode=='legacy' else '1'
            command=[str(binary),'--gguf',str(args.gguf.resolve()),'--ctx','4096','--batch','16','--mode','2',
                     '--prefix-cache','1','--session-cache-mib','6144','--session-cache-slots','4',
                     '--gpu-cache-mib','18432','--gpu-cache-allocator','arena','--pipeline-readers','2',
                     '--pipeline-chunk-mib','4','--pipeline-lookahead','1','--pipeline-d2d-batch','1',
                     '--request',str((args.out/'requests.json').resolve()),
                     '--output',str((args.out/(mode+'.json')).resolve()),'--logits',str((args.out/(mode+'.f32')).resolve())]
            run={'command':command,'STRATA_MM27_STATE_BULK':env['STRATA_MM27_STATE_BULK'],'checks':[]};report['runs'][mode]=run
            print('START '+mode,flush=True);save(args.out/'report.json',report)
            with (args.out/(mode+'.stdout.log')).open('wb') as out,(args.out/(mode+'.stderr.log')).open('wb') as err:
                proc=subprocess.Popen(command,stdout=out,stderr=err,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
                try:
                    assert contain(proc),'job containment';run['pid']=proc.pid;save(args.out/'report.json',report)
                    run['exit_code']=proc.wait(timeout=2400);assert proc.returncode==0,'inspect stderr'
                finally:
                    if proc.poll() is None:proc.kill();proc.wait(timeout=15)
            data=json.loads((args.out/(mode+'.json')).read_text(encoding='utf-8'))
            assert len(data['results'])==len(indices)
            for j,(idx,r) in enumerate(zip(indices,data['results'])):
                c=native_checks(r,data)
                c['reuse']=r['reused_tokens']==(0 if j<3 else [2016,4064,48][j%3])
                c['restore']=r['session_restore']==(j>=3)
                c['ids']=r['token_ids']==reference[idx]['token_ids']
                c['budget']=r['session_archive_bytes']<=6144*1024**2 and r['session_archive_entries']<=4
                c['timing']=r['request_ms']>=r['session_ms']
                parity=compare_files(args.reference/'generation.f32',offsets[idx],args.out/(mode+'.f32'),j*8,8)
                run['checks'].append({'request':j,'checks':c,'logits':parity,'pass':all(c.values()) and parity['pass']})
            assert (args.out/(mode+'.f32')).stat().st_size==len(indices)*8*NV*4
            assert all(c['pass'] for c in run['checks']),run['checks']
            run['pass']=True;save(args.out/'report.json',report);print('PASS '+mode,flush=True)
        a=json.loads((args.out/'legacy.json').read_text())['results']
        b=json.loads((args.out/'bulk.json').read_text())['results']
        for i,label in enumerate(['2k','4k','short']):
            x=[a[j] for j in range(3+i,12,3)];y=[b[j] for j in range(3+i,12,3)]
            comparison={'target':label,'repetitions':3}
            for metric in ['session_ms','ttft_ms','request_ms','decode_tokens_per_second']:
                av=[r[metric] for r in x];bv=[r[metric] for r in y]
                am,bm=statistics.median(av),statistics.median(bv)
                comparison[metric]={'legacy':av,'bulk':bv,'legacy_median':am,'bulk_median':bm,
                                    'ratio_legacy_over_bulk':am/bm if bm else None}
            report['comparisons'].append(comparison)
        report['pass']=True
    except BaseException:
        report['error']=traceback.format_exc();raise
    finally:
        report['artifacts']={f.name:sha(f) for f in args.out.iterdir() if f.is_file() and f.name!='report.json'}
        save(args.out/'report.json',report);print('REPORT pass='+str(report['pass']),flush=True)


if __name__=='__main__':main()
