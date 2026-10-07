"""Sequential A/B MiMo cache or binary runs with exact logits and 95% limits."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import psutil

from .check_mimo2_engine import Engine, compare_logits
from .check_mimo2_cuda import check_ceiling, gpu_memory, memory, sha
from .gguf_reader import GGUFFile
from .mimo2_template import renderer
from .prepare_mimo2_spec_requests import QUESTIONS


def disks():
    # Whole-system counters: includes unrelated processes, never labelled model I/O.
    return {name: x.read_bytes for name, x in psutil.disk_io_counters(perdisk=True).items()}


def metrics(rows):
    records=[r['metrics'] for r in rows]
    steps=sum(r['decode_steps'] for r in records)
    ms=sum(r['generation_ms'] for r in records)
    return dict(requests=len(rows), timed_tokens=steps, generation_ms=ms,
        tokens_per_second=steps*1000/ms,
        median_tokens_per_second=statistics.median(r['decode_steps']*1000/r['generation_ms'] for r in records),
        mean_cache_bytes=statistics.mean(r['cache_bytes'] for r in records),
        mean_payload_bytes=statistics.mean(r['cache_payload_bytes'] for r in records),
        mean_slot_bytes=statistics.mean(r['cache_slot_bytes'] for r in records),
        decode_h2d_bytes=sum(r['decode_h2d_bytes'] for r in records),
        cache_evictions=sum(r['cache_evictions'] for r in records),
        physical_slab_allocations=sum(r['cache_slab_allocations'] for r in records),
        copy_fences=sum(r.get('pipeline_copy_fences',0) for r in records),
        scratch_fences=sum(r.get('pipeline_scratch_fences',0) for r in records),
        tensor_batches=sum(r.get('pipeline_copy_batches',0) for r in records),
        frequency_rejected=sum(r.get('cache_frequency_rejected',0) for r in records),
        mean_request_ms=statistics.mean(r['request_ms'] for r in records))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--cuda-bin', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--order', default='0,16,16,0')
    p.add_argument('--axis', choices=['slab','pipeline','frequency','fill','binary'], default='slab')
    p.add_argument('--control-binary', type=Path, help='Preserved control executable for the binary axis; fill batches off')
    p.add_argument('--fixed-cache-decay', type=int, choices=[0,16384,65536,131072], default=65536,
                   help='Frequency period for fill and binary axes')
    p.add_argument('--fixed-slab-mib', type=int, choices=[0,16,32], default=16,
                   help='Slab layout for non-slab axes; slab axis always uses pipeline0')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--predict', type=int, default=32)
    p.add_argument('--timeout', type=int, default=2400)
    a=p.parse_args()
    order=[int(x) for x in a.order.split(',')]
    allowed=(0,16,32) if a.axis=='slab' else (0,1) if a.axis in ('pipeline','fill','binary') else (0,16384,65536,131072)
    if not order or order[0]!=0 or any(x not in allowed for x in order):
        p.error('Start with control 0; selected axis accepts '+str(allowed))
    if not 1<=a.repeats<=10 or not 2<=a.predict<=128:p.error('invalid repeat/predict count')
    if (a.axis=='binary')!=bool(a.control_binary):p.error('binary axis requires --control-binary; other axes cannot use it')
    if a.control_binary and not a.control_binary.is_file():p.error('control executable not found')
    dest=a.output_dir.resolve();dest.mkdir(parents=True, exist_ok=False)
    model=a.model.resolve();build=a.build.resolve();binary=build/'bin/strata-mimo2.exe'
    manifest=json.loads((build/'mimo2-build-manifest.json').read_text(encoding='utf8'))
    assert manifest['runtime'] and not manifest['speculative_probe']
    template=renderer(GGUFFile(model).metadata['tokenizer.chat_template'])
    prompts=[(name,template.render(messages=[dict(role='user',content=text)],enable_thinking=False,add_generation_prompt=True)) for name,text in QUESTIONS]
    report=dict(status='running',model=str(model),model_bytes=model.stat().st_size,
        binary_sha256=sha(binary),manifest=manifest,order=order,repeats=a.repeats,axis=a.axis,fixed_slab_mib=a.fixed_slab_mib,fixed_cache_decay=a.fixed_cache_decay,
        scope='GPU-only no-draft; context512/batch8/F32 KV/FA on, fresh KV; cache14 GiB/live clamp, mmap/reader1/chunk8, prefill admission off. First request per prompt excluded. Throughput excludes first output, prefill and loading; includes sampling/logits export. Disk counters cover all system I/O.',
        rounds=[],memory_ceiling=.95)
    if a.control_binary:report['control_binary_sha256']=sha(a.control_binary)
    engine=None
    try:
        reference_ids=None;reference_logits=None
        for index,value in enumerate(order):
            slab=value if a.axis=='slab' else a.fixed_slab_mib
            batch=value if a.axis=='pipeline' else int(a.axis in ('frequency','fill','binary'))
            decay=value if a.axis=='frequency' else a.fixed_cache_decay if a.axis in ('fill','binary') else 0
            fill=value if a.axis=='fill' else 0
            check_ceiling(memory(gpu_memory()))
            directory=dest/f'round-{index}-{a.axis}-{value}';directory.mkdir()
            logits=directory/'logits.f32'
            round_binary=a.control_binary.resolve() if a.axis=='binary' and value==0 else binary
            command=[str(round_binary),'--native',str(model),'--serve','--max-context','512','--batch-size','8',
                '--expert-cache-mib','14336','--expert-cache-prefill','off','--expert-reader','mmap',
                '--expert-readers','1','--expert-chunk-mib','8','--logits-file',str(logits)]
            env=os.environ.copy();env['PATH']=str(a.cuda_bin.resolve())+os.pathsep+env.get('PATH','')
            env['STRATA_MIMO_CACHE_SLAB_MIB']=str(slab)
            env['STRATA_MIMO_PIPELINE_BATCH']=str(batch)
            env['STRATA_MIMO_CACHE_DECAY']=str(decay)
            env['STRATA_MIMO_CACHE_FILL_BATCH']=str(fill)
            entry=dict(index=index,slab_mib=slab,pipeline_batch=batch,cache_decay=decay,cache_fill_batch=fill,command=command,cases=[]);report['rounds'].append(entry)
            entry['variant']=value;entry['binary_sha256']=sha(round_binary)
            engine=Engine(command,env,directory,a.timeout);entry['samples']=engine.samples
            while True:
                line=engine.read()
                if line.startswith('INFO '):entry['info']=line
                if line.startswith('READY '):break
            assert f'expert_cache_slab_mib={slab}' in entry['info']
            assert f'expert_pipeline_batch={batch}' in entry['info']
            assert f'expert_cache_decay={decay}' in entry['info']
            assert f'expert_cache_fill_batch={fill}' in entry['info']
            entry['load_wall_seconds']=time.monotonic()-engine.start
            for name,prompt in prompts:
                engine.send('ENC 1 '+prompt.encode('utf8').hex());line=engine.read()
                assert line.startswith('IDS ');ids=[int(n) for n in line.split()[1:]]
                for repeat in range(a.repeats+1):
                    before=disks();case=engine.request(ids,a.predict);after=disks()
                    case.update(name=name,repeat=repeat,warmup=repeat==0,
                        system_disk_read_bytes={k:after[k]-before.get(k,after[k]) for k in after})
                    entry['cases'].append(case)
                    print(index,a.axis,value,name,repeat,case['done'],flush=True)
            engine.close();assert engine.process.returncode==0
            assert len(engine.records)==len(entry['cases'])
            for case,record in zip(entry['cases'],engine.records):
                assert record['finish'] in ('length','stop') and record['generated']==len(case['tokens'])
                assert not record['rejected_cpu_nodes'] and not record['rejected_full_copies']
                assert not record['pipeline_queued'] and not record['pipeline_reader_owned']
                assert 0<=record['cache_payload_bytes']<=record['cache_slot_bytes']<=record['cache_bytes']<=record['cache_limit']<=14*2**30
                assert record['requested_bytes']==record['h2d_bytes']+record['cache_hit_bytes']
                assert record['cache_slab_mib']==slab
                assert record['pipeline_batch']==batch
                assert record['cache_decay']==decay and record['cache_history_keys']<=65536
                assert record['cache_fill_batch']==fill and record['cache_pending']==0
                if fill:assert record['pipeline_fill_submissions']>=record['pipeline_fill_batches']>0
                if decay:assert record['cache_frequency_updates']>0 and record['cache_history_keys']>0
                assert (record['pipeline_copy_batches']>0)==bool(batch)
                case['metrics']=record
            engine=None
            values=np.fromfile(logits,dtype='<f4').reshape(-1,152576)
            ids=[(r['ids'],r['tokens']) for r in entry['cases']]
            if index==0:reference_ids=ids;reference_logits=values
            assert ids==reference_ids,'token IDs differ'
            entry['logits_comparison']=compare_logits(values,reference_logits)
            assert entry['logits_comparison']['pass_'],'logits differ'
            entry['logits_sha256']=sha(logits)
            warm=[r for r in entry['cases'] if not r['warmup']]
            entry['aggregate']=metrics(warm)
            entry['per_prompt']={name:metrics([r for r in warm if r['name']==name]) for name,_ in prompts}
            entry['peak_global_vram_bytes']=max(s['gpu']['used'] for s in entry['samples'])
            entry['peak_global_ram_bytes']=max(s['ram_total']-s['ram_available'] for s in entry['samples'])
            entry['status']='pass'
            (dest/'benchmark.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
            print('ROUND PASS',index,a.axis,value,entry['aggregate'],flush=True)
        report['status']='pass'
    except Exception as error:
        report['status']='error';report['error']=repr(error)
    finally:
        if engine:engine.close()
        (dest/'benchmark.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(report['status'],report.get('error',''),dest/'benchmark.json',flush=True)
    return int(report['status']!='pass')


if __name__=='__main__':raise SystemExit(main())
