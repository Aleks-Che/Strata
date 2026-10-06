"""Full-model cache-off/on exact logits and a bounded cache capacity sweep.

Repeats a fixed English/code corpus with fresh KV; OS file cache is not purged.
Every run owns one engine under the sampled 95% global memory guard.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from check_hy3_engine import Engine
from check_hy3_model import metrics
from check_hy3_cuda import check_ceiling, memory, gpu_memory
from check_step35_model import Monitor
from inspect_hy3_gguf import inspect_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('engine', 'gguf', 'reference', 'output-dir', 'cuda-bin'):
        parser.add_argument('--'+arg, type=Path, required=True)
    parser.add_argument('--cache-mib', type=int, nargs='+', default=[0,8192,12288,16384,0])
    parser.add_argument('--pipeline-readers', type=int, nargs='+', default=[0])
    parser.add_argument('--pipeline-chunk-mib', type=int, choices=(4,8,16), default=4)
    parser.add_argument('--pipeline-batch', type=int, choices=(0,1), nargs='+', default=[0])
    parser.add_argument('--prompts', choices=('en','code','long'), nargs='+', default=['en','code'])
    parser.add_argument('--repeats', type=int, choices=(1,2,3), default=2)
    args = parser.parse_args()
    if any(not 0 <= cap <= 16384 for cap in args.cache_mib):
        parser.error('cache capacities must be 0..16384 MiB')
    if any(not 0 <= readers <= 2 for readers in args.pipeline_readers) or (any(args.pipeline_readers) and 0 in args.cache_mib):
        parser.error('pipeline readers must be 0..2 with nonzero cache cap')
    if any(args.pipeline_batch) and 0 in args.pipeline_readers:
        parser.error('tensor batching requires pipeline readers')
    model, binary, directory = args.gguf.resolve(), args.engine.resolve(), args.output_dir.resolve()
    ref = json.loads(args.reference.read_text(encoding='utf8'))
    inventory = inspect_model(model)
    if ref['status'] != 'pass' or inventory['header_sha256'] != ref['header_sha256']:
        raise ValueError('passing reference for the same reviewed Hy3 header required')
    prompts = [next(p for p in ref['prompts'] if p['name']==name) for name in args.prompts]
    reference = {r['name']:r for r in ref['runs'][1]['requests']}
    directory.mkdir(parents=True, exist_ok=False)
    os.environ['PATH'] = str(args.cuda_bin.resolve())+os.pathsep+os.environ.get('PATH','')
    report = dict(status='error', scope='full-model exact historical logits/IDs; sequential repeated capacity sweep; OS file cache not purged',
        engine_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), reference_engine_sha256=ref['engine_sha256'],
        model=str(model), header_sha256=inventory['header_sha256'], context=2048, batch=17, kv='f32',
        mtp=False, pipeline_readers=args.pipeline_readers, pipeline_chunk_mib=args.pipeline_chunk_mib,
        pipeline_batch=args.pipeline_batch, repeats=args.repeats,
        copy_mode='pinned', temperature=0, predict=ref['predict'], prompts=prompts, runs=[])
    def save():
        (directory/'cache-model-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    try:
        variants=[(cap,readers,batch) for cap in args.cache_mib for readers in args.pipeline_readers for batch in args.pipeline_batch]
        for index, (capacity,readers,batch_copy) in enumerate(variants):
            work = directory/f'{index:02d}-cache-{capacity}-readers-{readers}-batch-{batch_copy}'
            work.mkdir()
            row = dict(cache_mib=capacity, pipeline_readers=readers, pipeline_batch=batch_copy, requests=[])
            report['runs'].append(row)
            engine, monitor = None, Monitor()
            try:
                check_ceiling(memory(gpu_memory()))
                print('Loading cache',capacity,'MiB; readers',readers,'batch',batch_copy,flush=True)
                start=time.perf_counter()
                engine=Engine(binary,model,work,mode='pinned',logits=True,on_start=monitor.start,
                    extra_args=['--expert-cache-mib',str(capacity),'--pipeline-readers',str(readers),
                                '--pipeline-chunk-mib',str(args.pipeline_chunk_mib),'--pipeline-batch',str(batch_copy)])
                row.update(ready_seconds=time.perf_counter()-start,info=engine.info)
                for repeat in range(args.repeats):
                    for prompt in prompts:
                        print('cache',capacity,'repeat',repeat,prompt['name'],flush=True)
                        result=engine.generate(prompt['ids'],ref['predict'],sampling='temperature=0',
                            on_progress=lambda n,total: print('prefill',n,'/',total,flush=True))
                        raw=(work/'logits.f32').read_bytes()
                        expected=reference[prompt['name']]
                        result.update(name=prompt['name'],repeat=repeat,metrics=metrics(engine)[-1],
                            logits_sha256=hashlib.sha256(raw).hexdigest(),logits_bytes=len(raw))
                        result['exact_reference']=(result['ids']==expected['ids'] and
                            result['logits_sha256']==expected['logits_sha256'] and len(raw)==expected['logits_bytes'])
                        row['requests'].append(result)
                        save()
                        if not result['exact_reference']:
                            raise ValueError('cache changed full-model logits/IDs')
                        m=result['metrics']
                        assert m['source_bytes']==m['h2d_bytes'] and m['cache_bytes']<=m['cache_budget']<=capacity*2**20
                        assert not m['rejected_cpu_nodes'] and not m['rejected_full_copies']
                        if readers:
                            assert m['pipeline_groups'] and not m['pipeline_unused_bytes']
                            assert not m['pipeline_queued'] and not m['pipeline_reader_owned']
                            assert m['staging_bytes']==m['pipeline_device_bytes']==4*args.pipeline_chunk_mib*2**20
                        if batch_copy:
                            assert m['pipeline_copy_batches'] and m['pipeline_copy_batches']==m['pipeline_copy_fences']
                            assert m['pipeline_copy_fences']<m['ranges']
                        print('exact; decode',round(m['decode_steps']*1000/m['decode_forward_ms'],3),
                            'tok/s; H2D',round(m['decode_h2d_bytes']/2**30,3),'GiB; hits',m['decode_cache_hits'],
                            'misses',m['decode_cache_misses'],flush=True)
                if capacity:
                    cancelled=engine.generate(prompts[0]['ids'],ref['predict'],cancel=True)
                    assert cancelled['finish']=='cancel'
                    recovered=engine.generate(prompts[0]['ids'],ref['predict'],sampling='temperature=0')
                    expected=reference[prompts[0]['name']]
                    assert recovered['ids']==expected['ids']
                    assert hashlib.sha256((work/'logits.f32').read_bytes()).hexdigest()==expected['logits_sha256']
                    row['cancel_recovery']=dict(cancelled=cancelled,recovered=recovered,exact_reference=True)
                engine.close()
                row['exit_code']=engine.process.returncode
                assert row['exit_code']==0
            finally:
                if engine and not engine.stderr.closed: engine.close()
                monitor.close()
                row.update(memory_samples=monitor.samples,monitor_error=monitor.error)
                save()
            if monitor.error: raise RuntimeError(monitor.error)
        report['status']='pass'
    except Exception as error:
        report['error']=str(error)
    save()
    print(report['status'],report.get('error',''),directory/'cache-model-report.json',flush=True)
    return int(report['status']!='pass')


if __name__=='__main__':
    raise SystemExit(main())
