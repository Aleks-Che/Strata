"""Aggregate completed offline MiMo draft runs; never count warmup as throughput."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(rows):
    ms=sum(r['generation_ms'] for r in rows)
    tokens=sum(len(r['ids'])-1 for r in rows)
    proposed=sum(r['proposed'] for r in rows)
    accepted=sum(r['accepted'] for r in rows)
    return dict(request_count=len(rows), generated_after_first=tokens, generation_ms=ms,
        tokens_per_second=tokens*1000/ms, accepted=accepted, proposed=proposed,
        acceptance=accepted/proposed if proposed else None,
        cycles=sum(r['cycles'] for r in rows),
        target_ms=sum(r['target_ms'] for r in rows), draft_ms=sum(r['draft_ms'] for r in rows),
        catchup_ms=sum(r['catchup_ms'] for r in rows),
        mean_prefill_ms=sum(r['prefill_ms'] for r in rows)/len(rows),
        mean_cache_allocated_bytes=sum(r['cache_bytes'] for r in rows)/len(rows),
        mean_cache_payload_bytes=sum(r['cache_payload_bytes'] for r in rows)/len(rows),
        decode_h2d_bytes=sum(r['decode_h2d_bytes'] for r in rows),
        ids_equal=all(r['ids_equal'] for r in rows) if all('ids_equal' in r for r in rows) else None)


def summarize(label, path):
    report=json.loads(path.read_text(encoding='utf8'))
    if report['status'] not in ('pass','output_mismatch'): raise ValueError('Incomplete run: '+str(path))
    warm=[r for r in report['requests'] if not r['request'].get('warmup',False)]
    groups=defaultdict(list)
    for r in warm: groups[(r['depth'],round(r['p_min'],3))].append(r)
    configurations=[]
    for (depth, threshold),rows in groups.items():
        cases=defaultdict(list)
        for r in rows: cases[r['request']['name'].split('-d')[0]].append(r)
        configurations.append(dict(depth=depth,p_min=threshold,aggregate=metrics(rows),
            cases={name:metrics(items) for name,items in cases.items()}))
    samples=report['samples']
    return dict(label=label,path=str(path.resolve()),report_sha256=digest(path),
        status=report['status'],binary_sha256=report['binary_sha256'],manifest=report['manifest'],
        command=report['command'],configurations=configurations,
        all_ids_equal=all(r.get('ids_equal',True) for r in report['requests']),
        output_tokens_checked=sum(len(r['ids']) for r in report['requests'] if 'ids_equal' in r),
        peak_global_vram_bytes=max(s['gpu']['used'] for s in samples),
        peak_global_ram_bytes=max(s['ram_total']-s['ram_available'] for s in samples),
        peak_process_rss_bytes=max(s.get('process_rss',0) for s in samples),
        cpu_model_nodes_rejected=sum(r['rejected_cpu_nodes'] for r in report['requests']),
        outputs=[dict(name=r['name'],ids=r['ids'],ids_equal=r.get('ids_equal'),
            logits_comparison=r.get('logits_comparison')) for r in report['requests']])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='append',required=True,help='label=directory containing spec-report.json')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    assert a.output.suffix=='.json' and not a.output.exists()
    runs=[summarize(label,Path(folder)/'spec-report.json') for label,folder in (r.split('=',1) for r in a.run)]
    report=dict(scope='Offline greedy MiMo; RTX5090/128GB PC, context512/batch8/F32 KV/swa_full. Per-prompt first request excluded. MTP head0 only; no stochastic or broad quality claim.',
        status='pass' if all(r['status']=='pass' for r in runs) else 'output_mismatch',runs=runs)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    for run in runs:
        for c in run['configurations']:
            print(run['label'],c['depth'],c['p_min'],round(c['aggregate']['tokens_per_second'],3),
                  't/s',c['aggregate']['acceptance'],'acceptance',run['status'])


if __name__=='__main__': main()
