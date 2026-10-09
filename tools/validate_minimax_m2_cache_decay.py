"""Validate retained MM27-32 evidence, hashes, memory samples and aggregation."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.check_minimax_m2_cache_decay import sha, metrics, schedule, summarize, NV
from tools.check_minimax_m2_prefix import save
from tools.check_minimax_m2_prefix_context import native_checks
from tools.minimax_m2_memory_observer import memory_within_limit


def validate(directory,fixture,regression):
    report=json.loads((directory/'report.json').read_text(encoding='utf-8'))
    gates={}
    def check(name,value):gates[name]=bool(value)
    check('completed',report['pass'] and not report.get('error'))
    check('ten_processes',[(r['round'],r['period']) for r in report['runs']]==[(0,65536),*schedule(3)])
    check('ten_cli_boundaries',len(report['cli_checks'])==10 and all(c['pass'] for c in report['cli_checks']))
    for kind,items in [('artifacts',report['artifacts']),('sources',report['sources'])]:
        for rel,digest in items.items():
            path=directory/('sources' if kind=='sources' else '')/rel
            check(kind+'/'+rel,path.is_file() and sha(path)==digest)
    check('candidate_identity',sha(directory/'candidate.exe')==report['engine_sha256'])
    check('baseline_identity',sha(directory/'baseline.exe')==report['baseline_sha256']==
          '98e85e80e2dbaff5dc38f03f1ab4039835fb8d7e384f4491e1d700c989c5d3d8')
    baseline=json.loads((directory/'baseline.json').read_text(encoding='utf-8'))['results']
    ids=rows=0
    for run in report['runs']:
        name=run['name'];data=json.loads((directory/(name+'.json')).read_text(encoding='utf-8'));results=data['results']
        check(name+'/pass',run['pass'] and run['exit_code']==0)
        check(name+'/results',results==run['results'] and len(results)==4)
        check(name+'/config',data.get('cache_decay_period')==run['period'] if run['round'] else 'cache_decay_period' not in data)
        check(name+'/profile',data['context']==2048 and data['batch']==16 and data['gpu_cache_mib']==18432 and
              data['gpu_cache_allocator']=='arena' and data['arena_block_mib']==64 and data['arena_growth_reserve_mib']==0 and
              not data['cache_group_experts'] and data['pipeline_readers']==2 and data['pipeline_chunk_mib']==4 and
              data['pipeline_lookahead'] and data['pipeline_d2d_batch'] and not data['prefix_cache'] and
              data['ram_cache_mib']==data['session_cache_mib']==0 and data['kv']=='F32' and data['strict_f32'] and
              not data['flash_attention'] and not data['graphs'] and not data['mtp'])
        check(name+'/native',all(all(native_checks(r,data).values()) for r in results))
        check(name+'/fresh_kv',all(r['reused_tokens']==0 and not r['session_restore'] for r in results))
        check(name+'/prefill',all(r['prefill']['cache_fill_bytes']==r['prefill']['cache_evictions']==0 for r in results))
        count=sum(r['generated_tokens'] for r in results);ids+=count
        check(name+'/size',(directory/(name+'.f32')).stat().st_size==count*NV*4)
        logit=run['logits'];check(name+'/full_logits',logit['pass'] and logit['finite'] and
                               logit['different_bits']==logit['max_abs']==0 and logit['floats']==count*NV)
        check(name+'/ids',[r['token_ids'] for r in results]==[r['token_ids'] for r in baseline])
        if run['round']:rows+=count
        if run['period']==65536:
            keys=('h2d_bytes','cache_hit_bytes','cache_fill_bytes','cache_evictions','cache_reuses')
            check(name+'/default_policy',all(r[phase][key]==b[phase][key] for r,b in zip(results,baseline)
                                            for phase in ('prefill','decode') for key in keys))
        check(name+'/aggregation',run['metrics']==metrics(results))
        samples=[json.loads(s) for s in (directory/(name+'.memory.jsonl')).read_text().splitlines()]
        check(name+'/observer',len(samples)==run['observer']['samples']>0 and run['observer']['stopped'] and not run['observer']['error'])
        check(name+'/memory95',all(memory_within_limit(s) for s in samples))
    check('summary',report['summary']==summarize(report['runs']))
    unit=json.loads(fixture.read_text(encoding='utf-8-sig'))
    check('cache_fixture',unit['pass'] and len(unit['cases'])==19 and all(c['pass'] for c in unit['cases']))
    lifecycle=json.loads((regression/'cache-report.json').read_text(encoding='utf-8'))
    check('cache_lifecycle',lifecycle['pass'] and lifecycle['cases']==len(lifecycle['tests'])==216 and
          lifecycle['failures']==0 and all(c['pass'] for c in lifecycle['tests']) and len(lifecycle['runs'])==8)
    regression_hashes={f.relative_to(regression).as_posix():sha(f) for f in sorted(regression.rglob('*')) if f.is_file()}
    return {'pass':all(gates.values()),'stage':'MM27-32','report_sha256':sha(directory/'report.json'),
            'engine_sha256':report['engine_sha256'],'fixture_sha256':sha(fixture),'gates':gates,
            'cache_regression':{'cases':lifecycle['cases'],'artifacts':regression_hashes},
            'validator_sha256':sha(Path(__file__)),
            'requests':sum(len(r['results']) for r in report['runs']),'output_tokens':ids,
            'compared_candidate_ids':rows,'compared_candidate_logits':rows*NV,'summary':report['summary']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--fixture',type=Path,required=True);p.add_argument('--regression',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=validate(a.source,a.fixture,a.regression);save(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('summary','gates')}))
    print('gates',len(result['gates']),'failed',[k for k,v in result['gates'].items() if not v])
    if not result['pass']:raise SystemExit(1)


if __name__=='__main__':main()
