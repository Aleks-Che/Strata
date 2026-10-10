"""Recheck MM27-30 full logits, pressure evidence, identities and cleanup."""
import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from serve.minimax_m2_engine import EXE_SHA256, EVENTS_EXE_SHA256, ROUTER_EXE_SHA256
from tools.check_minimax_m2_prefix import save
from tools.check_minimax_m2_prefix_context import native_checks
from tools.check_minimax_m2_sessions import compare_files
from tools.check_minimax_m2_sessions_context import sha,inspect_case
from tools.minimax_m2_memory_observer import memory_within_limit
from tools.check_minimax_m2_copy_events import event_checks
from tools.check_minimax_m2_router_ids import router_checks


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def ram_sample_evidence(samples,target,blob,interval=.5):
    """Corroborate the target near its timestamp; separately check post-target admission bytes.

    Eviction can free an archive before the next poll. Select by timestamp,
    never by the largest reported memory use, and retain both observations.
    """
    nearest=min(samples,key=lambda m:abs(m['monotonic']-target['monotonic']))
    after=next(m for m in samples if m['monotonic']>=target['monotonic'])
    reserve=after['ram_total']//20+256*2**20
    return {'target_sample':nearest,'post_sample':after,'required_bytes':reserve+blob,
            'post_percent_target_met':.05<=after['ram_free']/after['ram_total']<=.065,
            'target_verified':abs(nearest['monotonic']-target['monotonic'])<=interval+.05 and
                .05<=nearest['ram_free']/nearest['ram_total']<=.065,
            'post_budget_short':0<=after['monotonic']-target['monotonic']<2 and
                target['ram_free']<reserve+blob and after['ram_free']<reserve+blob and after['commit_free']>reserve+blob}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline',type=Path,required=True)
    parser.add_argument('--pressure',type=Path,required=True)
    parser.add_argument('--reference',type=Path,default=ROOT/'build-local/minimax-m2-prefix-context-01')
    parser.add_argument('--out',type=Path,default=ROOT/'docs/minimax-m2.7/MINIMAX_M27_STATE_BULK_PRESSURE_CHECK.json')
    parser.add_argument('--engine',type=Path,default=ROOT/'build-local/minimax-m2-cuda/bin/strata-minimax-m2-bench.exe')
    parser.add_argument('--pipeline-events',type=int,choices=(0,2),default=0)
    parser.add_argument('--router-host-ids',action='store_true')
    parser.add_argument('--stage',default='MM27-30')
    parser.add_argument('--unit-log',type=Path,default=ROOT/'build-local/minimax-m2-state-bulk-pressure-unit.log')
    parser.add_argument('--python-methods',type=int,default=18)
    parser.add_argument('--startup',type=Path,help='Optional real CLI/auth/unload/reload smoke report directory')
    args=parser.parse_args();off=args.offline.resolve();live=args.pressure.resolve();old=args.reference.resolve()
    a,h,baseline=read(off/'report.json'),read(live/'report.json'),read(old/'report.json')
    assert a['pass'] and h['pass'] and baseline['pass'],'reports must be complete and passing'
    d=read(off/'generation.json');ref=read(old/'generation.json')['results']
    gates={};sources={};artifacts={}
    def gate(name,ok):gates[name]=bool(ok)
    gate('offline_cases',len(a['cases'])==len(d['results'])==14 and len(a['pairs'])==6)
    gate('pressure_cases',len(h['cases'])==19 and all(c['pass'] for c in h['cases']))
    binary=args.engine.resolve();digest=sha(binary)
    gate('same_admitted_engine',digest in (EXE_SHA256,EVENTS_EXE_SHA256,ROUTER_EXE_SHA256) and
         a['engine_sha256']==h['engine_sha256']==digest==sha(off/'engine.exe')==sha(live/'engine.exe'))
    if args.pipeline_events or digest in (EVENTS_EXE_SHA256,ROUTER_EXE_SHA256):
        speed_path=ROOT/'docs/minimax-m2.7'/('MINIMAX_M27_ROUTER_IDS_CHECK.json' if digest==ROUTER_EXE_SHA256 else 'MINIMAX_M27_COPY_EVENTS_CHECK.json');speed=read(speed_path)
        if digest==ROUTER_EXE_SHA256:
            gate('same_short_ab_candidate',speed['pass'] and all(g['pass'] for g in speed['gates']) and digest==speed['engine_sha256'])
        else:
            gate('same_short_ab_candidate',speed['pass'] and all(speed['gates'].values()) and digest in speed['engines'])
        gate('events_identity',digest in (EVENTS_EXE_SHA256,ROUTER_EXE_SHA256) and a['pipeline_events']==h['pipeline_events']==
             d['pipeline_events']==h['header']['pipeline_events']==args.pipeline_events and
             d['cache_decay_period']==h['header']['cache_decay_period']==65536)
        profile={'context':4096,'batch':16,'gpu_cache_mib':18432,'gpu_cache_allocator':'arena',
                 'arena_block_mib':64,'arena_growth_reserve_mib':0,'pipeline_readers':2,'pipeline_chunk_mib':4,
                 'pipeline_lookahead':True,'pipeline_d2d_batch':True,'prefix_cache':True,'session_cache_mib':6144,
                 'session_cache_slots':4,'ram_cache_mib':0,'cache_group_experts':False,'kv':'F32',
                 'strict_f32':True,'graphs':False,'flash_attention':False,'mtp':False}
        gate('events_profile',all(type(header.get(k)) is type(v) and header[k]==v
             for header in (d,h['header']) for k,v in profile.items()))
        gate('offline_event_drains',all(all(event_checks(r,args.pipeline_events).values()) for r in d['results']))
        gate('live_event_drains',all(all(event_checks(c['native'],args.pipeline_events).values()) for c in h['cases'] if 'native' in c))
        offline_memory=[json.loads(line) for line in (off/'physical-commit-monitor.jsonl').read_text().splitlines()]
        gate('offline_independent_monitor',len(offline_memory)==a['independent_monitor']['samples']>0 and
             not a['independent_monitor']['error'] and a['independent_monitor']['stopped'] and
             all(memory_within_limit(m) for m in offline_memory))
    gate('router_option_capability',not args.router_host_ids or (digest==ROUTER_EXE_SHA256 and args.pipeline_events==2))
    if digest==ROUTER_EXE_SHA256:
        gate('router_identity',all(header.get('router_host_ids') is args.router_host_ids for header in (a,h,d,h['header'])))
        gate('offline_router_reuse',all(all(router_checks(r,args.router_host_ids).values()) for r in d['results']))
        gate('live_router_reuse',all(all(router_checks(c['native'],args.router_host_ids).values()) for c in h['cases'] if 'native' in c))
        gate('current_router_sources',all(sha(ROOT/'backends/minimax_m2'/n)==h['sources']['backends/minimax_m2/'+n] for n in
             ['router_ids.hpp','RouterIds.cmake','RuntimePatches.cmake','QuantF32.cmake','sync_runtime.inc','sync_runtime.h']))
    gate('bulk_enabled',a['environment']['STRATA_MM27_STATE_BULK']==h['runtime_environment']['STRATA_MM27_STATE_BULK']=='1' and 'mm27-bounded-host-state-bulk' in d['patches'])
    gate('linked_reports',a['reference_report_sha256']==sha(old/'report.json') and h['offline_report_sha256']==sha(off/'report.json'))
    gate('reference_hashes',all(sha(old/name)==baseline['artifacts'][name] for name in ['generation.json','generation.f32','requests.json']))
    gate('offline_checks',all(all(inspect_case(r,d,c).values()) and c['pass'] for c,r in zip(a['cases'],d['results'])))
    offsets=[];total=0
    for r in ref:offsets.append(total);total+=r['generated_tokens']
    rows=[];total=0;parities=[]
    for c,r in zip(a['cases'],d['results']):
        rows.append(total);idx=c['reference_index'];n=r['generated_tokens']
        parities.append(compare_files(old/'generation.f32',offsets[idx],off/'generation.f32',total,n))
        gate('tokens_'+c['name'],r['token_ids']==ref[idx]['token_ids'][:n]);total+=n
    gate('offline_logit_size',(off/'generation.f32').stat().st_size==total*200064*4)
    for first,second in [(0,2),(3,5),(3,12),(6,8),(9,10),(11,13)]:
        parities.append(compare_files(off/'generation.f32',rows[first],off/'generation.f32',rows[second],d['results'][first]['generated_tokens']))
    gate('all_logits_recompared',all(p['pass'] for p in parities))
    gate('eos414',a['natural_eos'] and all(d['results'][i]['stop_reason']=='eos' and d['results'][i]['generated_tokens']==414 for i in [6,8]))
    gate('seed128',all(d['results'][i]['generated_tokens']==128 and d['results'][i]['sampling']==ref[8]['sampling'] and d['results'][i]['sampling']['seed']==42 for i in [9,10]))
    refs={'2k':d['results'][0]['token_ids'],'4k':d['results'][3]['token_ids'],'short':d['results'][1]['token_ids'],
          'eos':d['results'][6]['token_ids'],'sample':d['results'][9]['token_ids']}
    complete=[c for c in h['cases'] if 'native' in c]
    gate('http_answers',len(complete)==17 and sum(c['native']['generated_tokens'] for c in complete)==662 and
         all(c['status']==200 and c['native']['token_ids']==refs[c['kind']] for c in complete))
    gate('live_eos_sampling',next(c for c in complete if c['kind']=='eos')['native']['stop_reason']=='eos' and
         next(c for c in complete if c['kind']=='sample')['native']['sampling']==ref[8]['sampling'])
    gate('live_native_checks',all(all(native_checks(c['native'],h['header']).values()) for c in complete))
    gate('same_cuda_process',all(c['native_pid']==h['header']['native_pid'] for c in complete))
    gate('protected_ram_admission',h['admission']['saved_bytes']==0 and h['admission']['restored_bytes']>0 and
         h['admission']['after_rejected']>h['admission']['before_rejected'] and h['admission']['after_evictions']>h['admission']['before_evictions'])
    gate('physical_ram_target',.05<=h['pressure_ram94']['ram_free']/h['pressure_ram94']['ram_total']<=.065)
    external=[json.loads(line) for line in (live/'physical-commit-monitor.jsonl').read_text().splitlines()]
    blob=d['results'][5]['session_restored_bytes']
    ram_evidence=ram_sample_evidence(external,h['pressure_ram94'],blob,h['independent_monitor']['interval_seconds'])
    near=ram_evidence['post_sample']
    before=next(c for c in complete if c['name']=='4k-before-ram94')['native']
    gate('independent_physical_ram_target',ram_evidence['target_verified'])
    gate('ram_not_archive_or_commit_limit',ram_evidence['post_budget_short'] and
         before['session_archive_entries']+1<=4 and before['session_archive_bytes']+blob+2**20<6144*2**20)
    gate('independent_monitor_finalized',len(external)==h['independent_monitor']['samples'] and len(external)>0 and
         h['independent_monitor']['stopped'] and not h['independent_monitor']['error'] and
         sha(live/'physical-commit-monitor.jsonl')==h['artifacts']['physical-commit-monitor.jsonl'])
    gate('independent_memory95',all(memory_within_limit(m) for m in external))
    gate('bounded_private_pressure',0<h['pressure_ram94']['ram_private_bytes']<=16*2**30 and
         h['released']['ram_private_bytes']==h['released']['gpu_bytes']==h['released']['ram_touched']==0)
    gate('gpu_trim',h['trim']['before']>h['trim']['after'] and h['trim']['count']>0)
    gate('concurrent_ram_gpu_pressure',h['both_over85_samples']>0 and any(all(m[k+'_free']<.15*m[k+'_total'] for k in ['ram','gpu']) for m in external))
    for c in h['cases']:
        if c['name']=='restored-prefill-http-disconnect':gate('prefill_cancel',c['last']['generated']==0 and 'cancel' in c['error']['message'].lower())
        if c['name']=='restored-decode-cancel':gate('decode_cancel',not c['extra_tokens'] and c['first_token']==refs['2k'][0] and 'cancel' in c['error']['message'].lower())
    gate('fresh_after_cancel',next(c for c in complete if c['name']=='2k-fresh-recovery')['native']['reused_tokens']==0)
    gate('holder_cleanup',h['holder_exit_code']==0 and h['engine_closed'] and not h['cleanup_errors'] and not h['monitor_error'])
    samples=[d['memory_before'],d['memory_loaded']]
    for r in d['results']:samples+=r['memory_samples']
    for c in complete:samples+=c['native']['memory_samples']
    gate('native_memory95',all(0<=m[k+'_total']-m[k+'_available']<=m[k+'_total']*.95 for m in samples for k in ['ram','vram']))
    mon=h['monitor']+h['holder_records']
    gate('holder_memory95',all(.05*m[k+'_total']<=m[k+'_free']<=m[k+'_total'] for m in mon for k in ['ram','gpu']))
    gate('archive_caps',all(r['session_archive_bytes']<=6144*2**20 and r['session_archive_entries']<=4 for r in d['results']+[c['native'] for c in complete]))
    for folder,r in [(off,a),(live,h)]:
        for rel,source_digest in r['sources'].items():sources[(folder/'sources'/rel).relative_to(ROOT).as_posix()]=sha(folder/'sources'/rel)==source_digest
        for name,artifact_digest in r['artifacts'].items():gate('artifact_'+folder.name+'_'+name,sha(folder/name)==artifact_digest)
    gate('source_snapshots',all(sources.values()))
    gate('current_core_sources',all(sha(ROOT/'backends/minimax_m2'/n)==h['sources']['backends/minimax_m2/'+n] for n in
         ['main.cpp','sessions.hpp','session_runtime.hpp','prefix.hpp','runtime.hpp','pressure_holder.cpp','StateBulk.cmake','state_bulk.hpp']))
    gate('current_adapter',sha(ROOT/'serve/minimax_m2_engine.py')==h['sources']['serve/minimax_m2_engine.py'])
    gate('holder_executable',h['holder_sha256']==sha(live/'holder.exe')==sha(ROOT/'build-local/minimax-m2-cuda/bin/strata-minimax-m2-pressure-holder.exe'))
    unit=args.unit_log
    unit_text=unit.read_text(encoding='utf-8');gate('cpu_evidence',f'Ran {args.python_methods} tests' in unit_text and '\nOK\n' in unit_text)
    if args.startup:
        startup=args.startup.resolve();s=read(startup/'report.json')
        gate('startup_profile',s['pass'] and s['engine_sha256']==digest and s['pipeline_events']==args.pipeline_events and s.get('router_host_ids',False) is args.router_host_ids and s['server_stopped'])
        expected=[('auth-required',401),('web-assets',200),('prefix-0',200),('unload-0',200),('prefix-1',200),('unload-1',200)]
        gate('startup_cases',[(c['name'],c['status']) for c in s['cases']]==expected)
        answers=[read(startup/('prefix-'+str(i)+'.txt')) for i in range(2)]
        gate('startup_reload_answers',all(v['usage']['completion_tokens']==8 and v['choices'][0]['finish_reason']=='length' for v in answers)
             and answers[0]['choices'][0]['message']==answers[1]['choices'][0]['message'])
        gate('startup_source_hashes',all(sha(startup/'sources'/name)==digest for name,digest in s['sources'].items()))
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    owned={}
    for pid in [a['native_pid'],h['holder_pid'],h['header']['native_pid']]+([s['server_pid']] if args.startup else []):
        handle=kernel.OpenProcess(0x100000,False,pid)
        if handle:owned[pid]=kernel.WaitForSingleObject(handle,0)==0;kernel.CloseHandle(handle)
        else:owned[pid]=ctypes.get_last_error()==87
    gate('owned_pids_exited',all(owned.values()))
    for folder in [off,live]+([args.startup.resolve()] if args.startup else []):
        for f in folder.rglob('*'):
            if f.is_file():artifacts[f.relative_to(ROOT).as_posix()]={'sha256':sha(f),'bytes':f.stat().st_size}
    summary={'pass':all(gates.values()),'stage':args.stage,'engine_sha256':digest,'pipeline_events':args.pipeline_events,'router_host_ids':args.router_host_ids,
             'gates':gates,'source_checks':sources,'owned_pids_exited':owned,
             'offline_requests':len(a['cases']),'native_generated_tokens':total,
             'old_reference_compared_floats':sum(c['logits']['floats'] for c in a['cases']),
             'fresh_restore_compared_floats':sum(p['logits']['floats'] for p in a['pairs']),
             'http_scenarios':len(h['cases']),'http_full_tokens':sum(c['native']['generated_tokens'] for c in complete),'python_methods':args.python_methods,
             'global_sampled_peak_percent':{
                 'ram':max([100*(1-m['ram_available']/m['ram_total']) for m in samples]+[100*(1-m['ram_free']/m['ram_total']) for m in mon+external]),
                 'vram':max([100*(1-m['vram_available']/m['vram_total']) for m in samples]+[100*(1-m['gpu_free']/m['gpu_total']) for m in mon+external])},
             'ram94_physical_commit_sample':near,'ram94_evidence':ram_evidence,'independent_monitor':h['independent_monitor'],
             'independent_sampled_peak_percent':{k:max(100*(1-m[k+'_free']/m[k+'_total']) for m in external) for k in ['ram','gpu']},
             'limitations':['Pressure checks token IDs, not full logits.','GPU batch16 only; no shift or context beyond4096.',
                            'Single pressure pass; not a repeated speed A/B.'],
             'artifacts':artifacts,'unit_log_sha256':sha(unit),'validator_sha256':sha(Path(__file__))}
    if digest in (EVENTS_EXE_SHA256,ROUTER_EXE_SHA256):
        summary['short_ab_check_sha256']=sha(speed_path)
        summary['offline_independent_monitor']=a['independent_monitor']
        summary['offline_independent_peak_percent']={k:max(100*(1-m[k+'_free']/m[k+'_total']) for m in offline_memory) for k in ['ram','gpu']}
    if digest==ROUTER_EXE_SHA256:
        fields=('router_ids_published','router_ids_hits','router_ids_misses','router_ids_bytes',
                'pipeline_copy_events','graph_exit_fences')
        summary['router_counters']={name:{key:sum(r[phase][key] for r in results for phase in ('prefill','decode'))
             for key in fields} for name,results in [('offline',d['results']),('live_complete',[c['native'] for c in complete])]}
    if args.startup:summary['startup_cases']=len(s['cases'])
    save(args.out,summary)
    print(('PASS' if summary['pass'] else 'FAIL'),len(gates),'gates;',len(sources),'source checks;',len(artifacts),'artifact hashes')
    if not summary['pass']:raise SystemExit([name for name,ok in gates.items() if not ok])


if __name__=='__main__':main()
