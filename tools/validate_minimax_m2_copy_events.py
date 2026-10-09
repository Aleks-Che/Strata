"""Verify MM27-33 retained runs, event accounting, full-logit records and fixtures."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.check_minimax_m2_copy_events import event_checks, summarize, sha, metrics, NV
from tools.check_minimax_m2_prefix import save
from tools.check_minimax_m2_prefix_context import native_checks
from tools.minimax_m2_memory_observer import memory_within_limit


def validate(directories,fixture_root):
    gates={};summary=[];engines=set();requests=rows=samples=0;peaks={'ram':0,'gpu':0}
    def check(name,value):gates[name]=bool(value)
    for directory in directories:
        d=json.loads((directory/'report.json').read_text(encoding='utf-8'));prefix=directory.name
        engines.add(d['engine_sha256']);check(prefix+'/pass',d['pass'] and not d.get('error'))
        check(prefix+'/engine',sha(directory/'engine.exe')==d['engine_sha256'])
        check(prefix+'/run_order',[(r['repeat'],r['mode']) for r in d['runs']]==
              [(n,m) for n in range(1,d['repeats']+1) for m in (d['modes'] if n%2 else d['modes'][::-1])])
        for group,items in [('artifacts',d['artifacts']),('sources',d['sources'])]:
            for rel,digest in items.items():
                path=directory/('sources' if group=='sources' else '')/rel
                check(prefix+'/'+group+'/'+rel,path.is_file() and sha(path)==digest)
        baseline=None
        for run in d['runs']:
            label=prefix+'/'+run['name'];mode=run['mode']
            header=json.loads((directory/(run['name']+'.json')).read_text(encoding='utf-8'));results=header['results']
            check(label+'/completed',run['pass'] and run['exit_code']==0 and results==run['results'] and len(results)==4)
            check(label+'/profile',header['pipeline_events']==mode and header['cache_decay_period']==65536 and
                  header['context']==2048 and header['batch']==16 and header['gpu_cache_mib']==18432 and
                  header['gpu_cache_allocator']=='arena' and header['arena_block_mib']==64 and header['arena_growth_reserve_mib']==0 and
                  header['pipeline_readers']==2 and header['pipeline_chunk_mib']==4 and header['pipeline_lookahead'] and header['pipeline_d2d_batch'] and
                  not header['cache_group_experts'] and not header['prefix_cache'] and header['session_cache_mib']==header['ram_cache_mib']==0 and
                  header['strict_f32'] and header['kv']=='F32' and not header['graphs'] and not header['flash_attention'] and not header['mtp'])
            check(label+'/native',all(all(native_checks(r,header).values()) for r in results))
            check(label+'/events',all(all(event_checks(r,mode).values()) for r in results))
            check(label+'/fresh_kv',all(r['reused_tokens']==0 and not r['session_restore'] for r in results))
            check(label+'/prefill',all(r['prefill']['cache_fill_bytes']==r['prefill']['cache_evictions']==0 for r in results))
            if baseline is None:baseline=results
            check(label+'/ids',[r['token_ids'] for r in results]==[r['token_ids'] for r in baseline])
            keys=('h2d_bytes','cache_hit_bytes','cache_fill_bytes','cache_evictions','cache_reuses')
            check(label+'/same_cache_decisions',all(r[phase][key]==b[phase][key] for r,b in zip(results,baseline)
                                                   for phase in ('prefill','decode') for key in keys))
            count=sum(r['generated_tokens'] for r in results);requests+=len(results);rows+=count
            check(label+'/size',(directory/(run['name']+'.f32')).stat().st_size==count*NV*4)
            logit=run['logits'];check(label+'/full_logits',logit['pass'] and logit['finite'] and
                logit['different_bits']==logit['max_abs']==0 and logit['floats']==count*NV)
            check(label+'/metrics',run['metrics']==metrics(results))
            memory=[json.loads(s) for s in (directory/(run['name']+'.memory.jsonl')).read_text().splitlines()];samples+=len(memory)
            check(label+'/memory',len(memory)==run['observer']['samples']>0 and run['observer']['stopped'] and
                  not run['observer']['error'] and all(memory_within_limit(s) for s in memory))
            for axis in peaks:peaks[axis]=max(peaks[axis],*(100*(1-s[axis+'_free']/s[axis+'_total']) for s in memory))
        check(prefix+'/summary',d['summary']==summarize(d['runs'],d['modes']))
        summary.append({'directory':str(directory),'report_sha256':sha(directory/'report.json'),'summary':d['summary']})
    check('same_executable',len(engines)==1)
    fixtures=[]
    for name in ['e0','e1','e2','e2-groups','e2-ram','e2-single']:
        path=fixture_root/f'minimax-m2-copy-events-fixture-{name}-01/cache-report.json'
        f=json.loads(path.read_text(encoding='utf-8'))
        check('fixture/'+name,f['pass'] and f['failures']==0 and f['cases']==len(f['tests']) and all(t['pass'] for t in f['tests']))
        check('fixture/'+name+'/profile',f['pipeline_events']==int(name[1]) and f['pipeline_readers']==2 and
              f['pipeline_chunk_mib']==4 and f['pipeline_d2d_batch'] and
              f['pipeline_lookahead']==(name!='e2-single') and f['cache_group_experts']==(name=='e2-groups') and
              f['ram_cache_mib']==(64 if name=='e2-ram' else 0))
        fixtures.append({'name':name,'cases':f['cases'],'report_sha256':sha(path)})
    cli_path=fixture_root/'minimax-m2-copy-events-cli.json';cli=json.loads(cli_path.read_text())
    check('cli',cli['pass'] and len(cli['cases'])==8 and all(c['pass'] for c in cli['cases']))
    return {'pass':all(gates.values()),'stage':'MM27-33','gates':gates,'requests':requests,'output_tokens':rows,
            'compared_logits':rows*NV,'independent_memory_samples':samples,'observed_peak_percent':peaks,
            'engines':sorted(engines),'fixtures':fixtures,'cli_sha256':sha(cli_path),'validator_sha256':sha(Path(__file__)),
            'results':summary}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--fixture-root',type=Path,default=ROOT/'build-local');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=validate(a.runs,a.fixture_root);save(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('gates','results','fixtures')}))
    print('gates',len(result['gates']),'failures',[k for k,v in result['gates'].items() if not v])
    if not result['pass']:raise SystemExit(1)


if __name__=='__main__':main()
