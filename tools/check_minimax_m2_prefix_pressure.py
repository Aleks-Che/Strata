"""Opt-in live 4K prefix reuse/rollback under real bounded RAM and VRAM pressure.

Requires the full-logit context audit. HTTP output is compared by exact token
IDs; pressure-stage logits are not exported by the resident JSONL protocol.
"""
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serve.minimax_m2_api import MiniMaxAPITemplate
from serve.minimax_m2_engine import MiniMaxEngine, EXE_SHA256
from serve.server import Service, Server, make_handler
from tools.check_minimax_m2_http_context import Holder
from tools.check_minimax_m2_live_tools import collect_sse
from tools.check_minimax_m2_prefix import save, sha
from tools.check_minimax_m2_prefix_context import native_checks, snapshot
from tools.strata_tokenizer import Tokenizer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf',type=Path,required=True)
    parser.add_argument('--offline',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cuda-root',type=Path,default=ROOT/'build-local/cuda-13.0')
    parser.add_argument('--holder',type=Path,default=ROOT/'build-local/minimax-m2-cuda/bin/strata-minimax-m2-pressure-holder.exe')
    args=parser.parse_args()
    binary=ROOT/'build-local/minimax-m2-cuda/bin/strata-minimax-m2-bench.exe'
    audit=json.loads((args.offline/'report.json').read_text(encoding='utf-8'))
    assert audit['pass'] and audit['engine_sha256']==sha(binary)==EXE_SHA256
    for name in ['generation.json','requests.json','http-requests.json']:
        assert sha(args.offline/name)==audit['artifacts'][name], 'offline artifact changed: '+name
    args.out.mkdir(parents=True,exist_ok=False)
    report={'pass':False,'stage':'MM27-26 pressure','engine_sha256':EXE_SHA256,
            'offline_report_sha256':sha(args.offline/'report.json'),'cases':[],'monitor':[],'monitor_error':None}
    snapshot(args.out,report)
    for source,name in [(binary,'engine.exe'),(args.holder,'holder.exe'),(Path(__file__),'checker.py')]:
        shutil.copyfile(source,args.out/name)
    report['holder_sha256']=sha(args.holder)
    requests=json.loads((args.offline/'http-requests.json').read_text(encoding='utf-8'))
    native_requests=json.loads((args.offline/'requests.json').read_text(encoding='utf-8'))
    reference=json.loads((args.offline/'generation.json').read_text(encoding='utf-8'))['results']
    long=deepcopy(requests['4k']); short={**requests['english'],'max_tokens':8}
    expected_long=reference[4]['token_ids']; expected_short=reference[5]['token_ids'][:8]
    holder=engine=server=server_thread=monitor_thread=None
    stop=threading.Event()

    def record():
        if holder:
            report['holder_records']=list(holder.records)
        save(args.out/'report.json',report)

    try:
        holder=Holder(args.holder.resolve(),args.gguf.resolve(),args.cuda_root,args.out)
        report['holder_ready']=holder.expect('ready'); report['holder_pid']=holder.proc.pid
        engine=MiniMaxEngine(args.gguf,binary,args.cuda_root,args.out/'native.stderr.log',context=4096,batch=16,
                             gpu_cache_mib=18432,pipeline_readers=2,pipeline_chunk_mib=4,prefix_cache=True,request_timeout=1200)
        report['header'],report['command']=engine.header,engine.command
        native_pid=engine.header['native_pid']
        tok=Tokenizer.from_gguf(args.gguf)
        svc=Service(engine,tok,MiniMaxAPITemplate(ROOT/'serve/fixtures/minimax_m27_chat_template.jinja'),model_name='minimax-m2.7')
        server=Server(('127.0.0.1',0),make_handler(svc))
        server_thread=threading.Thread(target=server.serve_forever,daemon=True); server_thread.start()

        def monitor():
            try:
                while not stop.is_set():
                    m=holder.command('SAMPLE','sample')
                    report['monitor'].append({**m,'native_sequence':engine._sequence})
                    stop.wait(.5)
            except Exception:
                report['monitor_error']=traceback.format_exc()
                if engine._gate.locked():
                    engine._send({'command':'cancel','request_id':engine._sequence})
        monitor_thread=threading.Thread(target=monitor,daemon=True); monitor_thread.start()

        def post(name,req,expected_reused,api='openai',stream=False):
            print('START '+name,flush=True)
            body={**req,'stream':stream}; save(args.out/(name+'.request.json'),body)
            connection=http.client.HTTPConnection(*server.server_address,timeout=1250)
            started=time.perf_counter()
            try:
                connection.request('POST','/v1/chat/completions' if api=='openai' else '/v1/messages',json.dumps(body),
                                   {'Content-Type':'application/json','X-Strata-Session-Id':'pressure-session'})
                response=connection.getresponse(); wire=response.read().decode('utf-8')
            finally:
                connection.close()
            (args.out/(name+'.response.txt')).write_text(wire,encoding='utf-8')
            result=deepcopy(engine.last_result)
            row={'name':name,'pass':False,'api':api,'stream':stream,'status':response.status,'wall_ms':1000*(time.perf_counter()-started),
                 'native':result,'error':deepcopy(engine.last_error),'native_pid':engine.header['native_pid']}
            report['cases'].append(row); record()
            assert response.status==200 and result and not engine.last_error, row['error']
            row['checks']=native_checks(result,engine.header)
            assert all(row['checks'].values()),row['checks']
            expected=expected_long if req['messages']==long['messages'] else expected_short
            assert result['token_ids']==expected and result['reused_tokens']==expected_reused
            assert engine.header['native_pid']==native_pid and not engine._gate.locked() and engine.alive()
            answer=collect_sse(api,wire) if stream else json.loads(wire)
            usage=answer['usage']; row['usage']=usage
            if api=='openai':
                assert usage['prompt_tokens']==result['prompt_tokens'] and usage['completion_tokens']==len(expected)
                assert usage['prompt_tokens_details']['cached_tokens']==expected_reused
            else:
                assert usage['input_tokens']==result['prompt_tokens']-expected_reused and usage['output_tokens']==len(expected)
                assert usage['cache_read_input_tokens']==expected_reused
            assert not report['monitor_error'],report['monitor_error']
            row['pass']=True; record()
            print('DONE '+name+' reused='+str(expected_reused)+' prefill_ms='+str(round(result['prefill_ms'],3)),flush=True)
            return result

        post('4k-warm',long,0)
        warm=post('4k-repeat',long,4064,'anthropic',True)
        report['pressure_ram']=holder.command('RAM','ram'); record()
        report['pressure_gpu']=holder.command('GPU','gpu'); record()
        stressed=post('4k-pressure-openai',long,4064)
        report['trim']={'before':warm['decode']['arena_reserved'],'after':stressed['decode']['arena_reserved'],
                        'count':sum(stressed[p]['cache_pressure_trims'] for p in ('prefill','decode'))}
        assert report['trim']['before']>report['trim']['after'] and report['trim']['count']>0,report['trim']
        post('4k-pressure-anthropic',long,4064,'anthropic',True)

        # Retain >3K exact tokens but make at least 128 suffix tokens missing.
        branch=deepcopy(long)
        content=branch['messages'][0]['content']; left,right=content.rsplit(' A'*128,1)
        branch['messages'][0]['content']=left+' B'*128+right
        messages,tools,kwargs=svc.normalize_request(branch,'openai')
        branch_ids=tok.encode(svc.template.render(messages,tools=tools,**kwargs),parse_special=True)
        original=native_requests[2]['tokens']
        shared=next(i for i,(a,b) in enumerate(zip(original,branch_ids)) if a!=b)
        assert len(branch_ids)<=4080 and 3000<=shared<4000
        save(args.out/'cancel-branch.request.json',branch); save(args.out/'cancel-branch.ids.json',branch_ids)
        before=engine._sequence
        connection=http.client.HTTPConnection(*server.server_address,timeout=10)
        try:
            connection.request('POST','/v1/chat/completions',json.dumps(branch),
                               {'Content-Type':'application/json','X-Strata-Session-Id':'pressure-session'})
            end=time.monotonic()+10
            while engine._sequence==before and time.monotonic()<end:
                time.sleep(.02)
            assert engine._sequence==before+1 and engine._gate.locked()
            time.sleep(.6)
            started=time.perf_counter()
        finally:
            connection.close()
        end=time.monotonic()+40
        while engine._gate.locked() and time.monotonic()<end:
            time.sleep(.02)
        row={'name':'pressure-reused-prefill-http-disconnect','pass':False,'cancel_ms':1000*(time.perf_counter()-started),
             'common_prompt_tokens':shared,'error':deepcopy(engine.last_error),'last':deepcopy(engine.last)}
        report['cases'].append(row); record()
        assert engine.alive() and engine.header['native_pid']==native_pid and not engine._gate.locked()
        assert engine.last_result is None and engine.last_error and 'cancel' in engine.last_error['message'].lower()
        assert engine.last['generated']==0
        row['pass']=True
        post('4k-pressure-cancel-recovery',long,0)
        post('4k-pressure-recovered-repeat',long,4064,'anthropic',True)

        cancel=threading.Event()
        gen=engine.generate(original,8,{},cancel,session_id='pressure-session')
        try:
            item=next(gen)
            while item is None:
                item=next(gen)
            started=time.perf_counter(); cancel.set()
            rest=[x for x in gen if x is not None]
        finally:
            gen.close()
        row={'name':'pressure-reused-decode-cancel','pass':False,'cancel_ms':1000*(time.perf_counter()-started),
             'first_token':item,'extra_tokens':rest,'error':deepcopy(engine.last_error)}
        report['cases'].append(row); record()
        assert item==expected_long[0] and not rest and engine.alive() and engine.header['native_pid']==native_pid
        assert engine.last_result is None and engine.last_error and 'cancel' in engine.last_error['message'].lower()
        row['pass']=True
        post('short-pressure-cancel-recovery',short,0)
        post('short-pressure-repeat',short,48,'anthropic',True)
        report['released']=holder.command('FREE','freed')
        post('short-after-release',short,48)
        samples=[m for m in report['monitor'] if m['gpu_bytes'] and m['ram_touched']]
        report['both_over85_samples']=sum(all(m[a+'_free']<=.15*m[a+'_total'] for a in ('gpu','ram')) for m in samples)
        assert report['both_over85_samples']>0,'both physical memory pressure gates not reached'
        report['pass']=all(c['pass'] for c in report['cases']) and not report['monitor_error']
    except BaseException:
        report['error']=traceback.format_exc()
        print(report['error'],flush=True)
    finally:
        stop.set()
        report['cleanup_errors']=[]
        def cleanup(action):
            try:
                action()
            except BaseException:
                report['cleanup_errors'].append(traceback.format_exc())
        if monitor_thread:
            cleanup(lambda:monitor_thread.join(310))
        if holder:
            cleanup(holder.close); report['holder_exit_code']=holder.proc.poll()
        if server:
            cleanup(server.shutdown); cleanup(server.server_close); cleanup(lambda:server_thread.join(5))
        if engine:
            cleanup(engine.close); report['engine_closed']=not engine.alive()
        report['pass'] &= (report.get('holder_exit_code')==0 and report.get('engine_closed') is True and
                          not report['cleanup_errors'] and not (monitor_thread and monitor_thread.is_alive()))
        report['artifacts']={f.name:sha(f) for f in args.out.iterdir() if f.is_file() and f.name!='report.json'}
        record(); print('REPORT pass='+str(report['pass']),flush=True)
    return 0 if report['pass'] else 1


if __name__=='__main__':
    raise SystemExit(main())
