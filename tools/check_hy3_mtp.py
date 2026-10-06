"""Native Hy3 MTP sweep: exact greedy IDs, numerical logits and useful output speed.

Runs sequentially with a sampled 95% global memory ceiling. MTP-off must remain
bit-exact with the historical reference; MTP batches require exact greedy IDs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from check_hy3_engine import Engine
from check_hy3_model import metrics
from check_hy3_cuda import check_ceiling, memory, gpu_memory
from check_step35_model import Monitor
from inspect_hy3_gguf import inspect_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('engine', 'gguf', 'reference', 'output-dir', 'cuda-bin'):
        parser.add_argument('--'+arg, type=Path, required=True)
    parser.add_argument('--cache-mib', type=int, nargs='+', default=[8192])
    parser.add_argument('--pipeline-readers', type=int, nargs='+', default=[2])
    parser.add_argument('--pipeline-chunk-mib', type=int, choices=(4,8,16), default=4)
    parser.add_argument('--pipeline-batch', type=int, choices=(0,1), nargs='+', default=[0])
    parser.add_argument('--gpu-cache-policy', choices=('all','decode'), nargs='+', default=['all'])
    parser.add_argument('--prompts', choices=('ru','en','zh','code','numbers','long'), nargs='+', default=['en','code'])
    parser.add_argument('--repeats', type=int, choices=(1,2,3), default=2)
    parser.add_argument('--profile-delivery', action='store_true', help='diagnostic CPU timers; not an uninstrumented speed benchmark')
    parser.add_argument('--depths', type=int, nargs='+', choices=(0,1,2,3), default=[0,1,2,3,0])
    parser.add_argument('--variants', nargs='+', help='explicit GPU-cache-MiB:depth[:RAM-cache-MiB|auto[:frequency|lru]] order')
    parser.add_argument('--lifecycle', action='store_true', help='also compare sampling fallback and cancel during generation')
    args = parser.parse_args()
    try:
        capacities_depths=[]
        for value in args.variants or [f'{c}:{d}' for c in args.cache_mib for d in args.depths]:
            fields=value.split(':')
            if len(fields) not in (2,3,4): raise ValueError()
            c,d=map(int,fields[:2]);ram=fields[2] if len(fields)>=3 else '0'
            policy=fields[3] if len(fields)==4 else 'frequency'
            if policy not in ('frequency','lru'): raise ValueError()
            if not 0<=c<=16384 or not 0<=d<=3 or (ram!='auto' and not 0<=int(ram)<=1048576): raise ValueError()
            capacities_depths.append((c,d,ram,policy))
    except ValueError:
        parser.error('variants must be GPU-MiB:depth[:RAM-MiB|auto[:frequency|lru]], GPU0..16384, depth0..3, RAM0..1048576')
    if capacities_depths[0][1] != 0:
        parser.error('first depth must be zero for same-build logits comparison')
    if any(not 0 <= cap <= 16384 for cap in args.cache_mib):
        parser.error('cache capacities must be 0..16384 MiB')
    if any(not 0 <= readers <= 2 for readers in args.pipeline_readers) or (any(args.pipeline_readers) and any(c==0 for c,d,ram,policy in capacities_depths)):
        parser.error('pipeline readers must be 0..2 with nonzero cache cap')
    if any(args.pipeline_batch) and 0 in args.pipeline_readers:
        parser.error('tensor batching requires pipeline readers')
    if args.profile_delivery and 0 in args.pipeline_readers:
        parser.error('delivery profiling requires pipeline readers')
    if args.lifecycle and 'en' not in args.prompts:
        parser.error('lifecycle checks require the English prompt to cancel before EOS')
    model, binary, directory = args.gguf.resolve(), args.engine.resolve(), args.output_dir.resolve()
    ref = json.loads(args.reference.read_text(encoding='utf8'))
    inventory = inspect_model(model)
    if ref['status'] != 'pass' or inventory['header_sha256'] != ref['header_sha256']:
        raise ValueError('passing reference for the same reviewed Hy3 header required')
    prompts = [next(p for p in ref['prompts'] if p['name']==name) for name in args.prompts]
    reference = {r['name']:r for r in ref['runs'][1]['requests']}
    directory.mkdir(parents=True, exist_ok=False)
    os.environ['PATH'] = str(args.cuda_bin.resolve())+os.pathsep+os.environ.get('PATH','')
    report = dict(status='error', scope='MTP greedy IDs vs historical reference; off logits bit-exact; on numerical diagnostics vs same-build off; sequential sweep, OS cache not purged',
        engine_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), reference_engine_sha256=ref['engine_sha256'],
        model=str(model), header_sha256=inventory['header_sha256'], context=2048, batch=17, kv='f32',
        variants=capacities_depths, lifecycle=args.lifecycle, pipeline_readers=args.pipeline_readers, pipeline_chunk_mib=args.pipeline_chunk_mib,
        pipeline_batch=args.pipeline_batch, gpu_cache_policy=args.gpu_cache_policy, repeats=args.repeats,
        profile_delivery=args.profile_delivery,
        copy_mode='pinned', temperature=0, predict=ref['predict'], prompts=prompts, runs=[])
    def save():
        (directory/'mtp-model-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    off_logits={}
    off_sampled=None
    try:
        variants=[(cap,readers,batch,depth,ram,policy,gpu_policy) for cap,depth,ram,policy in capacities_depths for readers in args.pipeline_readers for batch in args.pipeline_batch for gpu_policy in args.gpu_cache_policy]
        for index, (capacity,readers,batch_copy,depth,ram,policy,gpu_policy) in enumerate(variants):
            work = directory/f'{index:02d}-cache-{capacity}-readers-{readers}-batch-{batch_copy}-mtp-{depth}'
            work.mkdir()
            row = dict(mtp_depth=depth, cache_mib=capacity, ram_cache_mib=ram, ram_cache_policy=policy, gpu_cache_policy=gpu_policy, pipeline_readers=readers, pipeline_batch=batch_copy, requests=[])
            report['runs'].append(row)
            engine, monitor = None, Monitor()
            try:
                check_ceiling(memory(gpu_memory()))
                print('Loading cache',capacity,'MiB; readers',readers,'batch',batch_copy,'MTP',depth,'RAM cache',ram,flush=True)
                start=time.perf_counter()
                engine=Engine(binary,model,work,mode='pinned',logits=True,on_start=monitor.start,
                    extra_args=['--expert-cache-mib',str(capacity),'--pipeline-readers',str(readers),
                                '--pipeline-chunk-mib',str(args.pipeline_chunk_mib),'--pipeline-batch',str(batch_copy),'--mtp',str(depth)]+
                               (['--gpu-cache-policy',gpu_policy] if gpu_policy!='all' else [])+
                               (['--ram-cache-mib',ram,'--ram-cache-policy',policy] if ram!='0' else [])+
                               (['--profile-delivery','1'] if args.profile_delivery else []))
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
                        result['ids_exact']=result['ids']==expected['ids']
                        result['exact_reference']=(result['ids_exact'] and
                            result['logits_sha256']==expected['logits_sha256'] and len(raw)==expected['logits_bytes'])
                        values=np.frombuffer(raw,dtype=np.float32)
                        key=prompt['name']
                        if not depth: off_logits[key]=values.copy()
                        base=off_logits[key]
                        if values.shape==base.shape:
                            delta=values.astype(np.float64)-base
                            result['logits_comparison']={'max_abs':float(np.max(np.abs(delta))),
                                'nmse':float(np.sum(delta*delta)/max(np.sum(base.astype(np.float64)**2),1e-30)),
                                'bit_exact':raw==base.tobytes(),'finite':bool(np.isfinite(values).all())}
                        row['requests'].append(result)
                        save()
                        if not result['ids_exact'] or (not depth and not result['exact_reference']):
                            raise ValueError('MTP changed greedy IDs or off baseline logits')
                        assert values.size==len(result['ids'])*120832 and np.isfinite(values).all()
                        m=result['metrics']
                        if gpu_policy=='decode':
                            assert m['gpu_cache_policy']=='decode' and not m['prefill_gpu_fill_bytes']
                        assert m['source_bytes']==m['h2d_bytes'] and m['cache_bytes']<=m['cache_budget']<=capacity*2**20
                        assert not m['rejected_cpu_nodes'] and not m['rejected_full_copies']
                        if ram!='0':
                            h=m['ram_cache']
                            assert h['bytes']<=h['budget'] and not h['pending_bytes']
                            assert h['hit_bytes']+h['file_bytes']==m['source_bytes']
                            assert h['policy']==policy
                            if policy=='frequency': assert m['prefill_ram_fill_bytes']==0
                            assert m['ram_reused_payload_bytes']+m['ram_unreused_payload_bytes']==h['bytes']
                            assert 0<=m['ram_reused_payload_bytes']<=h['bytes']
                        if readers:
                            assert m['pipeline_groups'] and not m['pipeline_unused_bytes']
                            assert not m['pipeline_queued'] and not m['pipeline_reader_owned']
                            assert m['staging_bytes']==m['pipeline_device_bytes']==4*args.pipeline_chunk_mib*2**20
                        if batch_copy:
                            assert m['pipeline_copy_batches'] and m['pipeline_copy_batches']==m['pipeline_copy_fences']
                            assert m['pipeline_copy_fences']<m['ranges']
                        print('IDs exact; useful decode',round((len(result['ids'])-1)*1000/m['generation_wall_ms'],3),
                            'tok/s; H2D',round(m['decode_h2d_bytes']/2**30,3),'GiB; hits',m['decode_cache_hits'],
                            'misses',m['decode_cache_misses'],'MTP',m['mtp'],flush=True)
                        if ram!='0': print('RAM cache',m['ram_cache'],flush=True)
                if capacity:
                    cancelled=engine.generate(prompts[0]['ids'],ref['predict'],cancel=True)
                    assert cancelled['finish']=='cancel'
                    recovered=engine.generate(prompts[0]['ids'],ref['predict'],sampling='temperature=0')
                    expected=reference[prompts[0]['name']]
                    assert recovered['ids']==expected['ids']
                    if not depth: assert hashlib.sha256((work/'logits.f32').read_bytes()).hexdigest()==expected['logits_sha256']
                    row['cancel_recovery']=dict(cancelled=cancelled,recovered=recovered,ids_exact=True)
                if args.lifecycle:
                    short=next((p for p in prompts if p['name']=='en'),prompts[0])
                    engine.send(f"GEN {ref['predict']} temperature=0 "+','.join(map(str,short['ids'])))
                    output=[];stop_at=None
                    while True:
                        line=engine.line()
                        if line.startswith('T '):
                            output.append(int(line[2:]))
                            if len(output)==2:
                                stop_at=time.perf_counter();engine.send('STOP')
                        elif line.startswith('DONE '):
                            fields=line.split()
                            assert fields[5]=='cancel' and int(fields[1])==len(output)
                            assert int(fields[6])<=int(fields[7]) and stop_at is not None
                            assert output==reference[short['name']]['ids'][:len(output)]
                            row['decode_cancel']=dict(ids=output,done=line,settle_seconds=time.perf_counter()-stop_at)
                            break
                        elif not line.startswith('PP '): raise ValueError(line)
                    recovery=engine.generate(short['ids'],ref['predict'],sampling='temperature=0')
                    assert recovery['ids']==reference[short['name']]['ids']
                    row['decode_cancel_recovery']=recovery
                    sampled=engine.generate(short['ids'],ref['predict'],sampling='temperature=0.7 seed=42 top_k=32')
                    sampled['metrics']=metrics(engine)[-1]
                    sampled['logits_sha256']=hashlib.sha256((work/'logits.f32').read_bytes()).hexdigest()
                    assert sampled['metrics']['mtp_depth']==0 and sampled['metrics']['mtp']['proposed']==0
                    if not depth: off_sampled=sampled
                    assert sampled['ids']==off_sampled['ids'] and sampled['logits_sha256']==off_sampled['logits_sha256']
                    row['sampling_fallback']=sampled
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
    print(report['status'],report.get('error',''),directory/'mtp-model-report.json',flush=True)
    return int(report['status']!='pass')


if __name__=='__main__':
    raise SystemExit(main())
