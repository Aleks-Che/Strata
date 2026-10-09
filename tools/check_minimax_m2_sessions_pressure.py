"""MM27-28: live session restore/admission/cancel under bounded external RAM/VRAM pressure."""
import argparse
from copy import deepcopy
import http.client
import json
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from serve.minimax_m2_api import MiniMaxAPITemplate
from serve.minimax_m2_engine import MiniMaxEngine,EXE_SHA256,EVENTS_EXE_SHA256
from serve.server import Service,Server,make_handler
from tools.check_minimax_m2_http_context import Holder
from tools.check_minimax_m2_live_tools import collect_sse
from tools.check_minimax_m2_prefix import save
from tools.check_minimax_m2_prefix_context import snapshot,native_checks
from tools.check_minimax_m2_sessions_context import sha
from tools.strata_tokenizer import Tokenizer
from tools.minimax_m2_memory_observer import MemoryObserver
from tools.check_minimax_m2_copy_events import event_checks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gguf',type=Path,required=True)
    p.add_argument('--offline',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cuda-root',type=Path,default=ROOT/'build-local/cuda-13.0')
    p.add_argument('--holder',type=Path,default=ROOT/'build-local/minimax-m2-cuda/bin/strata-minimax-m2-pressure-holder.exe')
    p.add_argument('--stage',default='MM27-28 pressure',help='Evidence label; does not change the corpus')
    p.add_argument('--engine',type=Path,default=ROOT/'build-local/minimax-m2-cuda/bin/strata-minimax-m2-bench.exe')
    p.add_argument('--pipeline-events',type=int,choices=(0,2),default=0)
    args=p.parse_args();binary=args.engine.resolve();digest=sha(binary)
    audit=json.loads((args.offline/'report.json').read_text(encoding='utf-8'))
    assert audit['pass'] and audit['engine_sha256']==digest and digest in (EXE_SHA256,EVENTS_EXE_SHA256)
    assert not args.pipeline_events or digest==EVENTS_EXE_SHA256
    assert audit.get('pipeline_events',0)==args.pipeline_events
    for name in ['generation.json','requests.json','http-requests.json']:
        assert sha(args.offline/name)==audit['artifacts'][name],name
    args.out.mkdir(parents=True,exist_ok=False)
    report={'pass':False,'stage':args.stage,'engine_sha256':digest,'pipeline_events':args.pipeline_events,
            'offline_report_sha256':sha(args.offline/'report.json'),'cases':[],'monitor':[],'monitor_error':None}
    snapshot(args.out,report)
    for relative in ['tools/minimax_m2_memory_observer.py','tools/test_minimax_m2_memory_observer.py','tools/check_minimax_m2_copy_events.py']:
        source=ROOT/relative;dest=args.out/'sources'/relative
        shutil.copyfile(source,dest);report['sources'][relative]=sha(dest)
    for source,name in [(binary,'engine.exe'),(args.holder,'holder.exe'),(Path(__file__),'checker.py')]:shutil.copyfile(source,args.out/name)
    report['holder_sha256']=sha(args.holder)
    bodies=json.loads((args.offline/'http-requests.json').read_text(encoding='utf-8'))
    results=json.loads((args.offline/'generation.json').read_text(encoding='utf-8'))['results']
    requests=json.loads((args.offline/'requests.json').read_text(encoding='utf-8'))
    refs={'2k':results[0]['token_ids'],'4k':results[3]['token_ids'],'short':results[1]['token_ids'],
          'eos':results[6]['token_ids'],'sample':results[9]['token_ids']}
    bodies['short']={**bodies['english'],'max_tokens':8}
    bodies['eos']=deepcopy(bodies['english'])
    bodies['sample']={**bodies['english'],'max_tokens':128,'temperature':1,'top_p':.95,'top_k':40,'seed':42}
    holder=engine=server=server_thread=monitor_thread=observer=None;stop=threading.Event()
    def record():
        if holder:report['holder_records']=list(holder.records)
        save(args.out/'report.json',report)
    try:
        holder=Holder(args.holder.resolve(),args.gguf.resolve(),args.cuda_root,args.out)
        report['holder_ready']=holder.expect('ready');report['holder_pid']=holder.proc.pid
        engine=MiniMaxEngine(args.gguf,binary,args.cuda_root,args.out/'native.stderr.log',context=4096,batch=16,
                             gpu_cache_mib=18432,pipeline_readers=2,pipeline_chunk_mib=4,prefix_cache=True,
                             session_cache_mib=6144,session_cache_slots=4,request_timeout=1200,pipeline_events=args.pipeline_events)
        report['header'],report['command']=engine.header,engine.command;pid=engine.header['native_pid']
        report['runtime_environment']={k:v for k,v in engine.env.items() if k.startswith(('STRATA_','GGML_','LLAMA_','NVIDIA_'))}
        def cancel_on_observer_error():
            if engine._gate.locked():engine._send({'command':'cancel','request_id':engine._sequence})
        observer=MemoryObserver(args.out/'physical-commit-monitor.jsonl',sequence=lambda:engine._sequence,
                                on_error=cancel_on_observer_error)
        observer.start()
        tok=Tokenizer.from_gguf(args.gguf)
        svc=Service(engine,tok,MiniMaxAPITemplate(ROOT/'serve/fixtures/minimax_m27_chat_template.jinja'),model_name='minimax-m2.7')
        server=Server(('127.0.0.1',0),make_handler(svc));server_thread=threading.Thread(target=server.serve_forever,daemon=True);server_thread.start()
        def monitor():
            try:
                while not stop.is_set():
                    m=holder.command('SAMPLE','sample');report['monitor'].append({**m,'native_sequence':engine._sequence})
                    stop.wait(.5)
            except Exception:
                report['monitor_error']=traceback.format_exc()
                if engine._gate.locked():engine._send({'command':'cancel','request_id':engine._sequence})
        monitor_thread=threading.Thread(target=monitor,daemon=True);monitor_thread.start()
        def post(name,kind,session,reuse,restored,api='openai',stream=False):
            assert not observer.error and not report['monitor_error'],'memory observer failed before request'
            print('START '+name,flush=True);body={**bodies[kind],'stream':stream}
            if stream and api=='openai':body['stream_options']={'include_usage':True}
            save(args.out/(name+'.request.json'),body)
            conn=http.client.HTTPConnection(*server.server_address,timeout=1250);started=time.perf_counter()
            try:
                conn.request('POST','/v1/chat/completions' if api=='openai' else '/v1/messages',json.dumps(body),
                             {'Content-Type':'application/json','X-Strata-Session-Id':session})
                response=conn.getresponse();wire=response.read().decode('utf-8')
            finally:conn.close()
            (args.out/(name+'.response.txt')).write_text(wire,encoding='utf-8')
            r=deepcopy(engine.last_result)
            row={'name':name,'kind':kind,'session':session,'pass':False,'api':api,'stream':stream,'status':response.status,
                 'wall_ms':1000*(time.perf_counter()-started),'native':r,'error':deepcopy(engine.last_error),'native_pid':engine.header['native_pid']}
            report['cases'].append(row);record()
            assert response.status==200 and r and not engine.last_error,row['error']
            row['checks']=native_checks(r,engine.header)
            if digest==EVENTS_EXE_SHA256:row['checks'].update(event_checks(r,args.pipeline_events))
            assert all(row['checks'].values()) and r['token_ids']==refs[kind]
            assert r['reused_tokens']==reuse and r['session_restore']==restored,(name,r['reused_tokens'],r['session_restore'])
            assert r['session_archive_bytes']<=6144*2**20 and r['session_archive_entries']<=4
            assert engine.alive() and engine.header['native_pid']==pid and not engine._gate.locked()
            answer=collect_sse(api,wire) if stream else json.loads(wire);usage=answer['usage'];row['usage']=usage
            if api=='openai':assert usage['prompt_tokens_details']['cached_tokens']==reuse and usage['prompt_tokens']==r['prompt_tokens']
            else:assert usage['cache_read_input_tokens']==reuse and usage['input_tokens']==r['prompt_tokens']-reuse
            assert usage['completion_tokens' if api=='openai' else 'output_tokens']==len(refs[kind])
            assert not report['monitor_error'],report['monitor_error']
            assert not observer.error,observer.error
            row['pass']=True;record();print('DONE '+name+' reused='+str(reuse)+' restored='+str(restored),flush=True)
            return r
        post('4k-cold','4k','long',0,False)
        post('2k-cold','2k','medium',0,False)
        warm=post('short-cold','short','short',0,False)
        report['pressure_ram']=holder.command('RAM','ram');report['pressure_gpu']=holder.command('GPU','gpu');record()
        stressed=post('4k-restore-pressure','4k','long',4064,True,'openai',True)
        report['trim']={'before':warm['decode']['arena_reserved'],'after':stressed['decode']['arena_reserved'],
                        'count':sum(stressed[k]['cache_pressure_trims'] for k in ['prefill','decode'])}
        assert report['trim']['before']>report['trim']['after'] and report['trim']['count']>0
        post('2k-restore-pressure','2k','medium',2016,True,'anthropic',True)
        post('4k-before-long-completions','4k','long',4064,True)
        eos=post('eos-restored-pressure','eos','short',48,True,'anthropic',True)
        assert eos['stop_reason']=='eos' and eos['generated_tokens']==414
        post('4k-between-completions','4k','long',4064,True)
        sampled=post('sample-restored-pressure','sample','short',48,True)
        assert sampled['generated_tokens']==128 and sampled['sampling']==results[9]['sampling']
        before=post('4k-before-ram94','4k','long',4064,True)
        report['pressure_ram94']=holder.command('RAM94','ram94');record()
        m=report['pressure_ram94'];assert .05<=m['ram_free']/m['ram_total']<=.065,'RAM93.5..95 target not reached'
        protected=post('2k-protected-admission','2k','medium',2016,True,'anthropic')
        report['admission']={'before_rejected':before['session_archive_rejected'],'after_rejected':protected['session_archive_rejected'],
                             'before_evictions':before['session_archive_evictions'],'after_evictions':protected['session_archive_evictions'],
                             'saved_bytes':protected['session_saved_bytes'],'restored_bytes':protected['session_restored_bytes']}
        assert protected['session_saved_bytes']==0 and protected['session_archive_rejected']>before['session_archive_rejected']
        assert protected['session_archive_evictions']>before['session_archive_evictions'] and protected['session_restored_bytes']>0
        post('short-before-prefill-cancel','short','short',0,False)
        branch=deepcopy(bodies['2k']);content=branch['messages'][0]['content'];left,right=content.rsplit(' A'*128,1)
        branch['messages'][0]['content']=left+' B'*128+right
        messages,tools,kwargs=svc.normalize_request(branch,'openai')
        branch_ids=tok.encode(svc.template.render(messages,tools=tools,**kwargs),parse_special=True)
        original=requests[0]['tokens'];shared=next(i for i,(a,b) in enumerate(zip(original,branch_ids)) if a!=b)
        assert 1500<shared<2000 and len(branch_ids)<=2032
        save(args.out/'cancel-branch.request.json',branch);save(args.out/'cancel-branch.ids.json',branch_ids)
        sequence=engine._sequence;conn=http.client.HTTPConnection(*server.server_address,timeout=10)
        try:
            conn.request('POST','/v1/chat/completions',json.dumps(branch),{'Content-Type':'application/json','X-Strata-Session-Id':'medium'})
            end=time.monotonic()+10
            while engine._sequence==sequence and time.monotonic()<end:time.sleep(.02)
            assert engine._sequence==sequence+1 and engine._gate.locked()
            time.sleep(.6);started=time.perf_counter()
        finally:conn.close()
        end=time.monotonic()+40
        while engine._gate.locked() and time.monotonic()<end:time.sleep(.02)
        row={'name':'restored-prefill-http-disconnect','pass':False,'cancel_ms':1000*(time.perf_counter()-started),
             'common_prompt_tokens':shared,'error':deepcopy(engine.last_error),'last':deepcopy(engine.last)}
        report['cases'].append(row);record()
        assert engine.alive() and engine.header['native_pid']==pid and not engine._gate.locked()
        assert engine.last_result is None and engine.last_error and 'cancel' in engine.last_error['message'].lower() and engine.last['generated']==0
        row['pass']=True
        post('short-survives-prefill-cancel','short','short',48,True)
        post('2k-fresh-recovery','2k','medium',0,False)
        post('short-before-decode-cancel','short','short',48,True,'anthropic',True)
        cancel=threading.Event();gen=engine.generate(original,8,{},cancel,session_id='medium')
        try:
            item=next(gen)
            while item is None:item=next(gen)
            started=time.perf_counter();cancel.set();rest=[t for t in gen if t is not None]
        finally:gen.close()
        row={'name':'restored-decode-cancel','pass':False,'first_token':item,'extra_tokens':rest,
             'cancel_ms':1000*(time.perf_counter()-started),'error':deepcopy(engine.last_error)}
        report['cases'].append(row);record()
        assert item==refs['2k'][0] and not rest and engine.alive() and engine.header['native_pid']==pid
        assert engine.last_result is None and engine.last_error and 'cancel' in engine.last_error['message'].lower()
        row['pass']=True
        post('short-survives-decode-cancel','short','short',48,True)
        report['released']=holder.command('FREE','freed')
        post('short-after-release','short','short',48,False)
        report['both_over85_samples']=sum(all(m[a+'_free']<=.15*m[a+'_total'] for a in ['gpu','ram']) for m in report['monitor'])
        assert report['both_over85_samples']>0
        report['pass']=all(c['pass'] for c in report['cases']) and not report['monitor_error']
    except BaseException:
        report['error']=traceback.format_exc();print(report['error'],flush=True)
    finally:
        stop.set();report['cleanup_errors']=[]
        def cleanup(action):
            try:action()
            except BaseException:report['cleanup_errors'].append(traceback.format_exc())
        if monitor_thread:cleanup(lambda:monitor_thread.join(5))
        if holder:cleanup(holder.close);report['holder_exit_code']=holder.proc.poll()
        if server:cleanup(server.shutdown);cleanup(server.server_close);cleanup(lambda:server_thread.join(5))
        if engine:cleanup(engine.close);report['engine_closed']=not engine.alive()
        if observer:
            cleanup(observer.close);report['independent_monitor']=observer.summary()
            if observer.error or not observer.count:report['pass']=False
        report['pass'] &= report.get('holder_exit_code')==0 and report.get('engine_closed') is True and not report['cleanup_errors'] and not report['monitor_error'] and not (monitor_thread and monitor_thread.is_alive())
        report['artifacts']={f.name:sha(f) for f in args.out.iterdir() if f.is_file() and f.name!='report.json'}
        record();print('REPORT pass='+str(report['pass']),flush=True)
    return 0 if report['pass'] else 1


if __name__=='__main__':raise SystemExit(main())
