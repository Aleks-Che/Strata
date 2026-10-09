"""Verify retained MM27-29 evidence and publish a compact repository report."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from serve.minimax_m2_engine import EXE_SHA256
from tools.check_minimax_m2_prefix import save
from tools.check_minimax_m2_state_bulk import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ab',type=Path,required=True)
    p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--live',type=Path,required=True)
    p.add_argument('--out',type=Path,default=ROOT/'docs/minimax-m2.7/MINIMAX_M27_STATE_BULK_CHECK.json')
    args=p.parse_args()
    reports={key:json.loads((path/'report.json').read_text(encoding='utf-8'))
             for key,path in [('ab',args.ab),('fixture',args.fixture),('live',args.live)]}
    assert all(r['pass'] for r in reports.values())
    assert reports['ab']['engine_sha256']==reports['live']['engine_sha256']==EXE_SHA256
    assert reports['live']['offline_report_sha256']==sha(args.ab/'report.json')
    binary=ROOT/'build-local/minimax-m2-cuda/bin/strata-minimax-m2-bench.exe'
    assert sha(binary)==EXE_SHA256
    checked_sources=checked_artifacts=0
    for path,key in [(args.ab,'ab'),(args.live,'live')]:
        r=reports[key]
        for relative,digest in r['sources'].items():
            assert sha(path/'sources'/relative)==digest,(key,relative);checked_sources+=1
        for relative,digest in r['artifacts'].items():
            assert sha(path/relative)==digest,(key,relative);checked_artifacts+=1
    for name in ['state_bulk.hpp','StateBulk.cmake','check_state_bulk.cpp','CMakeLists.txt','build_manifest.json.in']:
        rel='backends/minimax_m2/'+name
        assert sha(ROOT/rel)==reports['ab']['sources'][rel]==reports['live']['sources'][rel],rel
    fixture=reports['fixture']
    assert fixture['cpu_cases']==1200 and all(c['pass'] for c in fixture['gpu_checks'])
    assert fixture['gpu_io']['bulk_groups']>0 and fixture['gpu_io']['max_scratch_bytes']==16*1024**2
    fixture_exe=ROOT/'build-local/minimax-m2-cuda/bin/strata-minimax-m2-state-bulk-check.exe'
    assert sha(fixture_exe)==sha(args.fixture/'support'/fixture_exe.name)
    for name in ['state_bulk.hpp','check_state_bulk.cpp','StateBulk.cmake']:
        assert sha(args.fixture/'support'/name)==sha(ROOT/'backends/minimax_m2'/name)
    manifest_path=ROOT/'build-local/minimax-m2-cuda/minimax-m2-build-manifest.json'
    assert sha(manifest_path)==sha(args.fixture/'support'/manifest_path.name)
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    for entry in manifest['sampling_support_sha256'].split(';'):
        if entry:
            name,digest=entry.split(':');assert sha(ROOT/'backends/minimax_m2'/name)==digest,name
    assert sha(ROOT/'build-local/minimax-m2-cuda/strata-minimax-m2-context.cpp')==manifest['state_generated_sha256']
    runs=reports['ab']['runs']
    assert set(runs)=={'legacy','bulk'} and len(reports['ab']['comparisons'])==3
    assert all(r['pass'] and r['exit_code']==0 and len(r['checks'])==12 and all(c['pass'] for c in r['checks']) for r in runs.values())
    assert all(c['repetitions']==3 for c in reports['ab']['comparisons'])
    assert len(reports['live']['cases'])==10 and all(c['pass'] for c in reports['live']['cases'])
    floats=sum(c['logits']['floats'] for r in runs.values() for c in r['checks'])
    tokens=sum(len(r['token_ids']) for mode in runs for r in json.loads((args.ab/(mode+'.json')).read_text())['results'])
    memory={}
    for mode in runs:
        data=json.loads((args.ab/(mode+'.json')).read_text())
        samples=[data['memory_before'],data['memory_loaded']]+[m for r in data['results'] for m in r['memory_samples']]
        phases=[r[k] for r in data['results'] for k in ['prefill','decode']]
        peaks={axis:max([m[axis+'_total']-m[axis+'_available'] for m in samples]+
                        [s['sampled_'+axis+'_used_peak'] for s in phases]) for axis in ['ram','vram']}
        memory[mode]={axis+'_peak_percent':100*peaks[axis]/data['memory_loaded'][axis+'_total'] for axis in peaks}
        memory[mode]['process_private_peak']=max([m['process_private'] for m in samples]+[s['sampled_private_peak'] for s in phases])
        memory[mode]['process_working_set_peak']=max([m['process_working_set'] for m in samples]+[s['sampled_working_set_peak'] for s in phases])
        assert all(memory[mode][axis+'_peak_percent']<=95 for axis in peaks)
    for count,name in [(25,'unit'),(14,'context-unit')]:
        log=(ROOT/('build-local/minimax-m2-state-bulk-'+name+'.log')).read_text(encoding='utf-8')
        assert ('Ran '+str(count)+' tests') in log and '\nOK\n' in log
    support=[binary,fixture_exe,manifest_path,
             ROOT/'build-local/build-minimax-m2-state-bulk.bat',
             ROOT/'build-local/minimax-m2-state-bulk-build.log',
             ROOT/'build-local/minimax-m2-state-bulk-configure.log',
             ROOT/'build-local/minimax-m2-state-bulk-unit.log',
             ROOT/'build-local/minimax-m2-state-bulk-context-unit.log',Path(__file__).resolve(),
             *sorted((args.fixture/'support').resolve().iterdir())]
    result={'pass':True,'stage':'MM27-29','engine_sha256':EXE_SHA256,
            'reference_engine_sha256':reports['ab']['reference_engine_sha256'],
            'tokens_compared':tokens,'f32_logits_compared':floats,
            'cpu_fixture_cases':fixture['cpu_cases'],'cuda_checks':len(fixture['gpu_checks']),
            'cuda_bulk_groups':fixture['gpu_io']['bulk_groups'],'scratch_cap_bytes':16*1024**2,
            'http_lifecycle_cases':len(reports['live']['cases']),
            'python_test_runs':[{'methods':25,'log':'build-local/minimax-m2-state-bulk-unit.log'},
                                {'methods':14,'log':'build-local/minimax-m2-state-bulk-context-unit.log'}],
            'source_hash_checks':checked_sources,'artifact_hash_checks':checked_artifacts,
            'sampled_memory':memory,
            'comparisons':reports['ab']['comparisons'],
            'evidence':{str(path.relative_to(ROOT) if path.is_absolute() else path):sha(path/'report.json')
                        for path in [args.ab,args.fixture,args.live]},
            'support_sha256':{path.relative_to(ROOT).as_posix():sha(path) for path in support},
            'limits':['A/B repetitions share the warm cache within each of two sequential processes.',
                      'State bytes are compared on a CUDA fixture; the full model compares all output logits.',
                      'Live API checks compare IDs, not full logits.',
                      'Real 94% RAM/VRAM pressure has not been rerun on this new executable.',
                      'Decode speedup, GPU batch1/8, shift and broad quality are not established by this stage.']}
    save(args.out,result);print(json.dumps({k:result[k] for k in ['pass','tokens_compared','f32_logits_compared','source_hash_checks','artifact_hash_checks']}))


if __name__=='__main__':main()
