"""Compare full-model MTP logits with independent target-only batch execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from check_hy3_cuda import check_ceiling, memory, gpu_memory
from check_step35_model import Monitor
from inspect_hy3_gguf import inspect_model


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checker','gguf','reference','sweep','output-dir','cuda-bin'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    directory=args.output_dir.resolve();directory.mkdir(parents=True,exist_ok=False)
    ref=json.loads(args.reference.read_text(encoding='utf8'))
    sweep=json.loads(args.sweep.read_text(encoding='utf8'))
    assert ref['status']==sweep['status']=='pass'
    assert ref['header_sha256']==sweep['header_sha256']==inspect_model(args.gguf)['header_sha256']
    report=dict(status='error',scope='teacher-forced target-only batches 1..4; no draft weights, hidden extraction or KV rollback',
                checker_sha256=sha(args.checker),sweep_sha256=sha(args.sweep),reference_sha256=sha(args.reference),checks=[])
    env=dict(os.environ);env['PATH']=str(args.cuda_bin.resolve())+os.pathsep+env.get('PATH','')
    command=[str(args.checker.resolve()),'--teacher-forced',str(args.gguf.resolve()),str(args.reference.resolve()),str(directory/'oracle')]
    report['command']=command
    monitor=Monitor();child=None
    try:
        check_ceiling(memory(gpu_memory()))
        with (directory/'stderr.log').open('wb') as err,(directory/'stdout.log').open('wb') as out:
            child=subprocess.Popen(command,env=env,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            monitor.start(child);report['exit_code']=child.wait(timeout=600)
        if monitor.error: raise RuntimeError(monitor.error)
        report['oracle']=json.loads((directory/'oracle/teacher-report.json').read_text())
        assert report['exit_code']==0 and report['oracle']['status']=='pass'
        for depth in (0,1,2,3):
            run_index=next(i for i,v in enumerate(sweep['runs']) if v['mtp_depth']==depth)
            run=sweep['runs'][run_index]
            raw_dir=args.sweep.parent/f"{run_index:02d}-cache-{run['cache_mib']}-readers-{run['pipeline_readers']}-batch-{run['pipeline_batch']}-mtp-{depth}"
            source=raw_dir/'logits.f32' # Last recovery request is English, checked below.
            expected=next(r for r in run['requests'] if r['name']=='en' and r['repeat']==1)
            assert sha(source)==expected['logits_sha256'], 'recovery differs from recorded English request'
            actual=directory/'oracle'/f'batch{depth+1}.f32'
            a=np.fromfile(actual,dtype=np.float32);b=np.fromfile(source,dtype=np.float32)
            assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            delta=a.astype(np.float64)-b
            row=dict(depth=depth,batch=depth+1,oracle_sha256=sha(actual),mtp_sha256=sha(source),
                     exact=actual.read_bytes()==source.read_bytes(),max_abs=float(np.max(np.abs(delta))),
                     nmse=float(np.sum(delta*delta)/max(np.sum(b.astype(np.float64)**2),1e-30)))
            report['checks'].append(row)
            assert row['exact'], 'MTP logits differ from independent target batching'
        report['status']='pass'
    except Exception as error:
        report['error']=str(error)
    finally:
        if child and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: child.kill();child.wait(timeout=10)
        monitor.close();report.update(memory_samples=monitor.samples,memory_guard_error=monitor.error)
        (directory/'mtp-logits-report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(report['status'],report.get('error',''),flush=True)
    return int(report['status']!='pass')


if __name__=='__main__':
    raise SystemExit(main())
