"""Offline greedy MiMo target/MTP/DFlash comparison with recorded token IDs and memory."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
from .check_mimo2_engine import Engine, compare_logits
from .check_mimo2_cuda import sha, check_ceiling, gpu_memory, memory
from .inspect_mimo2_drafts import inspect


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--draft',type=Path)
    p.add_argument('--kind',choices=['none','mtp','dflash'],default='none')
    p.add_argument('--share-target',action='store_true',help='DFlash: RL target embedding/head plus separate learned MASK')
    p.add_argument('--build',type=Path,default=Path('build-local/mimo2-cuda'))
    p.add_argument('--cuda-bin',type=Path,default=Path('build-local/cuda-13.0/bin/x64'))
    p.add_argument('--requests',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--reference-dir',type=Path);p.add_argument('--timeout',type=int,default=1800)
    a=p.parse_args();dest=a.output_dir.resolve();dest.mkdir(parents=True,exist_ok=False)
    binary=(a.build/'bin/strata-mimo2-spec-check.exe').resolve()
    report=dict(status='error',scope='Offline greedy experiment; all tested positions retained in F32 KV, context512/batch8. MTP head0 only. No serving integration.',
        command=[str(binary),'--model',str(a.model.resolve()),'--kind',a.kind],binary_sha256=sha(binary),
        manifest=json.loads((a.build/'mimo2-build-manifest.json').read_text()),requests=[],samples=[])
    process=None
    try:
        assert report['manifest'].get('speculative_probe'), 'Configure/build with STRATA_MIMO_SPEC_PROBE=ON'
        if a.share_target:
            assert a.kind=='dflash'
            report['command']+=['--share-target','1']
        if a.draft:
            report['inspection']=inspect(a.model,[a.draft])
            assert report['inspection']['drafts'][0]['contract']['kind']==a.kind
            report['command']+=['--draft',str(a.draft.resolve())]
        else: assert a.kind=='none'
        report['before']=memory(gpu_memory());check_ceiling(report['before'])
        env=os.environ.copy();env['PATH']=str(a.cuda_bin.resolve())+os.pathsep+env['PATH']
        process=Engine(report['command'],env,dest,a.timeout)
        assert process.read()=='READY'
        import time
        report['load_and_workspace_warmup_seconds']=time.monotonic()-process.start
        requests=json.loads(a.requests.read_text(encoding='utf8'))
        for index,request in enumerate(requests):
            q=dict(request);q['logits']=str(dest/f'{index}.f32')
            process.send(json.dumps(q));r=json.loads(process.read());r['request']=request
            report['requests'].append(r)
            if a.reference_dir:
                refs=json.loads((a.reference_dir/'spec-report.json').read_text(encoding='utf8'))['requests']
                matches=[i for i,ref in enumerate(refs) if request['tokens']==ref['request']['tokens'] and request['predict']==ref['request']['predict']]
                assert matches, 'No baseline for this exact prompt/length'
                ref_index=index if index in matches else matches[0]
                ref=refs[ref_index];r['reference_index']=ref_index
                r['ids_equal']=r['ids']==ref['ids']
                r['first_id_mismatch']=next((j for j,(x,y) in enumerate(zip(r['ids'],ref['ids'])) if x!=y),None)
                x=np.fromfile(q['logits'],dtype='<f4');y=np.fromfile(a.reference_dir/f'{ref_index}.f32',dtype='<f4')
                if x.shape==y.shape:
                    r['logits_comparison']=compare_logits(x,y)
                    if r['first_id_mismatch'] is not None:
                        end=(r['first_id_mismatch']+1)*152576
                        r['logits_same_history']=compare_logits(x[:end],y[:end])
            print(a.kind,request.get('name'),round(r['tokens_per_second'],3),'t/s',r['accepted'],'/',r['proposed'],'equal',r.get('ids_equal'),flush=True)
            (dest/'spec-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        process.close();assert process.process.returncode==0
        report['status']='pass' if all(r.get('ids_equal',True) for r in report['requests']) else 'output_mismatch'
    except Exception as e:
        report['error']=repr(e)
    finally:
        if process:
            process.close();report['samples']=process.samples;report['exit_code']=process.process.returncode
        (dest/'spec-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(report['status'],report.get('error',''),dest/'spec-report.json',flush=True)
    return 0 if report['status']=='pass' else 1


if __name__=='__main__': sys.exit(main())
