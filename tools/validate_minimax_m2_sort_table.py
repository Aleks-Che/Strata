"""Recompute MM27-37 full-logit, transport, fixture and memory evidence."""
import argparse
import ctypes
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.check_minimax_m2_router_ids import router_checks, summarize
from tools.check_minimax_m2_sort_table import table_checks
from tools.check_minimax_m2_copy_events import event_checks
from tools.check_minimax_m2_prefix_context import native_checks, NV
from tools.check_minimax_m2_cache_decay import sha, metrics
from tools.check_minimax_m2_prefix import save
from tools.check_minimax_m2_sessions import compare_files
from tools.windows_memory_counters import PagingCounters

ENGINE_SHA='692f2e76d97192c3357d209d2013d92b63735607e6c4eeaeac42b31c71a839ff'
TRANSPORT=('ranges','selected_bytes','h2d_bytes','source_bytes','cache_hits','cache_misses',
           'cache_hit_bytes','cache_fill_bytes','cache_guard_bytes','cache_evictions','cache_reuses',
           'pipeline_matrices','pipeline_copy_events','graph_exit_fences')


def exited(pid):
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32];k.OpenProcess.restype=ctypes.c_void_p
    k.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32];k.WaitForSingleObject.restype=ctypes.c_uint32
    k.CloseHandle.argtypes=[ctypes.c_void_p]
    h=k.OpenProcess(0x100000,False,pid)
    if not h:return ctypes.get_last_error()==87
    try:return k.WaitForSingleObject(h,0)==0
    finally:k.CloseHandle(h)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'build-local/minimax-m2-sort-table-ab-02')
    p.add_argument('--out',type=Path,default=ROOT/'docs/minimax-m2.7/MINIMAX_M27_SORT_TABLE_CHECK.json')
    a=p.parse_args();r=json.loads((a.run/'report.json').read_text());gates=[];artifacts={}
    def gate(name,ok):gates.append({'name':name,'pass':bool(ok)})
    def remember(path):artifacts[path.resolve().relative_to(ROOT).as_posix()]=sha(path)
    gate('run_pass',r['pass']);gate('engine',sha(a.run/'engine.exe')==r['engine_sha256']==ENGINE_SHA)
    manifest=json.loads((a.run/'manifest.json').read_text())
    gate('manifest_revision',manifest['source_revision']=='86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9' and
         manifest['archive_sha256']=='f8e524b635b726bae74fd8f84bb9249e5b09384c63207f6707c5a3f921acad99')
    gate('manifest_runtime_patch',manifest['runtime_enabled']=='ON' and 'mm27-opt-in-pinned-sort-table-ring' in manifest['patches'])
    build=ROOT/'build-local/minimax-m2-cuda'
    gate('generated_cuda',sha(build/'strata-mm27-quant-f32-cuda.cu')==manifest['cuda_quant_dispatch_generated_sha256'])
    for entry in manifest['runtime_generated_sha256'].split(';'):
        if entry:
            name,digest=entry.split(':');gate('generated_runtime/'+name,sha(build/name)==digest)
    for field in ['group_cache_support_sha256','sampling_support_sha256','dflash_support_sha256']:
        for entry in manifest[field].split(';'):
            if entry:
                name,digest=entry.split(':')
                gate('manifest_support/'+name,sha(a.run/'sources/backends/minimax_m2'/name)==digest)
    gate('three_alternating_pairs',[(v['repeat'],v['mode']) for v in r['runs']]==[(1,0),(1,1),(2,1),(2,0),(3,0),(3,1)])
    for name,digest in r['artifacts'].items():
        path=a.run/name;gate('artifact/'+name,sha(path)==digest);remember(path)
    for name,digest in r['sources'].items():
        path=a.run/'sources'/name;gate('source/'+name,sha(path)==digest)
        if name.startswith('backends/minimax_m2/') and not name.endswith('.md'):
            gate('current_backend/'+name,sha(ROOT/name)==digest)
        remember(path)
    reference=ROOT/'build-local/minimax-m2-cache-decay-01'
    old=json.loads((reference/'report.json').read_text());gate('reference_report',sha(reference/'report.json')==r['reference_report_sha256'])
    gate('reference_logits',sha(reference/'baseline.f32')==r['reference_logits_sha256']==old['artifacts']['baseline.f32'])
    ref=json.loads((reference/'baseline.json').read_text())['results']
    gate('reference_tokens',sha(reference/'baseline.json')==old['artifacts']['baseline.json'])
    comparisons=[];peaks={'ram':0,'gpu':0};rows=0;requests=0;pids=[]
    baseline=r['runs'][0]['results']
    for run in r['runs']:
        name=run['name'];base=a.run/name;header=json.loads(base.with_suffix('.json').read_text())
        gate(name+'/reported_results',header['results']==run['results'])
        gate(name+'/flags',header['router_host_ids'] is True and header['sort_table_async']==bool(run['mode']) and header['pipeline_events']==2 and
             header['context']==2048 and header['batch']==16 and header['cache_decay_period']==65536)
        gate(name+'/tokens', [v['token_ids'] for v in run['results']]==[v['token_ids'] for v in ref])
        count=sum(v['generated_tokens'] for v in run['results']);rows+=count;requests+=len(run['results'])
        gate(name+'/logit_size',base.with_suffix('.f32').stat().st_size==count*NV*4)
        comp=compare_files(reference/'baseline.f32',0,base.with_suffix('.f32'),0,count)
        comparisons.append({'name':name,**comp});gate(name+'/full_logits',comp['pass'])
        for i,v in enumerate(run['results']):
            for key,ok in {**native_checks(v,header),**event_checks(v,2),**router_checks(v,1),**table_checks(v,run['mode'])}.items():
                gate(f'{name}/{i}/{key}',ok)
            gate(f'{name}/{i}/same_transport',all(v[phase][key]==baseline[i][phase][key] for phase in ['prefill','decode'] for key in TRANSPORT))
        memory=[json.loads(line) for line in base.with_suffix('.memory.jsonl').read_text().splitlines()]
        gate(name+'/independent_observer',len(memory)==run['observer']['samples'] and len(memory)>0 and
             not run['observer']['error'] and run['observer']['stopped'])
        for axis in peaks:
            peak=max(100*(1-v[axis+'_free']/v[axis+'_total']) for v in memory);peaks[axis]=max(peaks[axis],peak)
            gate(name+'/'+axis+'_95_percent',peak<=95)
        pids.append({'pid':run['pid'],'exited':exited(run['pid'])});gate(name+'/process_exited',pids[-1]['exited'])
        gate(name+'/metrics',run['metrics']==metrics(run['results']))
    fixture_counts={}
    for name,file,count in [('fixtures','build-local/minimax-m2-sort-table-fixtures-02/report.json',106),
                            ('cache','build-local/minimax-m2-sort-table-cache-01/cache-report.json',236)]:
        path=ROOT/file;f=json.loads(path.read_text());remember(path);fixture_counts[name]=f['cases']
        gate(name+'/tests',f['pass'] and f['failures']==0 and f['cases']==len(f['tests'])==count and all(t['pass'] for t in f['tests']))
    for name in ['minimax-m2-sort-table-fixtures-02.stdout.log','minimax-m2-sort-table-fixtures-02.stderr.log',
                 'minimax-m2-sort-table-cache-01.stdout.log','minimax-m2-sort-table-cache-01.stderr.log',
                 'minimax-m2-sort-table-configure.log','minimax-m2-sort-table-build.log','build-minimax-m2-sort-table.bat']:
        remember(ROOT/'build-local'/name)
    candidate=ROOT/'build-local/minimax-m2-sort-table-candidate'
    for path in candidate.glob('*'):
        if path.is_file():remember(path)
    gate('candidate_unchanged',sha(candidate/'engine.exe')==ENGINE_SHA)
    gate('legacy_preserved',sha(ROOT/'build-local/minimax-m2-cuda/bin/strata-minimax-m2-bench.exe')=='98e85e80e2dbaff5dc38f03f1ab4039835fb8d7e384f4491e1d700c989c5d3d8')
    gate('events_preserved',sha(ROOT/'build-local/minimax-m2-copy-events-candidate/engine.exe')=='13a885cc75352bb39cc4231174da3d9f03705d5f9c150117935d6f05ea9d2401')
    gate('router_preserved',sha(ROOT/'build-local/minimax-m2-router-ids-candidate/engine.exe')=='612354a12513eb1ce616ac2cc370bdd314a5c4725760bb586f6b9dd0ee1e6177')
    unit=ROOT/'build-local/minimax-m2-sort-table-unit.log';remember(unit)
    gate('unit_tests','Ran 17 tests' in unit.read_text() and unit.read_text().rstrip().endswith('OK'))
    cli=ROOT/'build-local/minimax-m2-sort-table-cli.json';c=json.loads(cli.read_text());remember(cli)
    gate('early_cli_rejections',len(c)==4 and all(v['pass'] for v in c))
    summary=summarize(r['runs'],[0,1]);gate('summary_recomputed',summary==r['summary'])
    pairs=[]
    for repeat in [1,2,3]:
        v={run['mode']:metrics(run['results']) for run in r['runs'] if run['repeat']==repeat}
        pairs.append({'repeat':repeat,'off_tok_s':v[0]['decode_tok_s'],'on_tok_s':v[1]['decode_tok_s'],
                      'gain_percent':100*(v[1]['decode_tok_s']/v[0]['decode_tok_s']-1)})
    batch_dir=ROOT/'build-local/minimax-m2-sort-table-batches-01'
    br=json.loads((batch_dir/'report.json').read_text());remember(batch_dir/'report.json')
    gate('short_batches/report',br['pass'] and br['engine_sha256']==ENGINE_SHA and br['speed_report_sha256']==sha(a.run/'report.json'))
    gate('short_batches/paging_helper',sha(batch_dir/'windows_memory_counters.py')==sha(ROOT/'tools/windows_memory_counters.py'))
    gate('short_batches/order',[(v['batch'],v['mode']) for v in br['runs']]==[(1,0),(1,1),(8,0),(8,1)])
    for name,digest in br['artifacts'].items():
        path=batch_dir/name;gate('short_batches/artifact/'+name,sha(path)==digest);remember(path)
    batch_comparisons=[]
    for run in br['runs']:
        name=run['name'];base=batch_dir/name;header=json.loads(base.with_suffix('.json').read_text())
        gate(name+'/reported_results',run['pass'] and run['results']==header['results'])
        if 'system_io_delta' in run:
            gate(name+'/system_io_delta',PagingCounters.difference(run['system_io_before'],run['system_io_after'])==run['system_io_delta'])
        gate(name+'/flags',header['batch']==run['batch'] and header['router_host_ids'] is True and header['sort_table_async']==bool(run['mode']) and header['pipeline_events']==2)
        for i,v in enumerate(run['results']):
            for key,ok in {**native_checks(v,header),**event_checks(v,2),**router_checks(v,1),**table_checks(v,run['mode'])}.items():gate(f'{name}/{i}/{key}',ok)
        memory=[json.loads(line) for line in base.with_suffix('.memory.jsonl').read_text().splitlines()]
        gate(name+'/observer',len(memory)==run['observer']['samples']>0 and not run['observer']['error'] and run['observer']['stopped'])
        for axis in peaks:
            peak=max(100*(1-v[axis+'_free']/v[axis+'_total']) for v in memory);peaks[axis]=max(peaks[axis],peak);gate(name+'/'+axis+'_95_percent',peak<=95)
        pids.append({'pid':run['pid'],'exited':exited(run['pid'])});gate(name+'/process_exited',pids[-1]['exited'])
        if run['mode']:
            off=json.loads((batch_dir/f"b{run['batch']}-r0.json").read_text())['results']
            gate(name+'/tokens',[v['token_ids'] for v in off]==[v['token_ids'] for v in run['results']])
            gate(name+'/transport',all(v[phase][key]==off[i][phase][key] for i,v in enumerate(run['results']) for phase in ['prefill','decode'] for key in TRANSPORT))
            comp=compare_files(batch_dir/f"b{run['batch']}-r0.f32",0,base.with_suffix('.f32'),0,16)
            batch_comparisons.append({'batch':run['batch'],**comp});gate(name+'/full_logits',comp['pass'])
    table_counters=[]
    for run in r['runs']:
        table_counters.append({'name':run['name'],'decode_cpu_submission_ms':sum(v['decode']['compute_ms'] for v in run['results']),
            'decode_delivery_ms':sum(v['decode']['pipeline_delivery_ms'] for v in run['results']),**{key:sum(v[phase][key] for v in run['results'] for phase in ['prefill','decode'])
            for key in ['sort_table_copies','sort_table_bytes','sort_table_reuse_waits','sort_table_drain_waits','sort_table_fallbacks']}})
    for name in ['report.json','corrected-evidence.json']:
        remember(ROOT/'build-local/minimax-m2-sort-table-ab-01'/name)
    remember(a.run/'report.json');remember(Path(__file__))
    for name in ['tools/check_minimax_m2_sort_table.py','tools/test_minimax_m2_sort_table.py','tools/check_minimax_m2_sort_table_batches.py']:remember(ROOT/name)
    output={'pass':all(v['pass'] for v in gates),'stage':'MM27-37','engine_sha256':ENGINE_SHA,'requests':requests,
        'token_ids':rows,'full_logits':rows*NV,'full_logit_comparisons':comparisons,'pairs':pairs,'summary':summary,
        'independent_peak_percent':peaks,'sort_table_counters':table_counters,'fixtures':fixture_counts,'short_batch_comparisons':batch_comparisons,'short_batch_system_io':[{'name':v['name'],'scope':v.get('system_io_scope'), 'delta':v.get('system_io_delta'),'engine_source_bytes':sum(x[p]['source_bytes'] for x in v['results'] for p in ['prefill','decode']),'unavailable':v.get('system_io_unavailable')} for v in br['runs']],'pids':pids,'gates':gates,'artifacts':artifacts,
        'scope':'ctx2048/batch16 speed pairs; short full-model batch1/8; tiny batch1/8/16 + fault/reload; native opt-in only, no server promotion'}
    save(a.out,output);print(json.dumps({k:output[k] for k in ['pass','token_ids','full_logits','pairs','independent_peak_percent']}))
    print(f'{len(gates)} gates; {len(artifacts)} hashes');return 0 if output['pass'] else 1


if __name__=='__main__':raise SystemExit(main())
