"""Live loopback HTTP and cancellation/restart checks for bounded RAM KV sessions."""
import argparse
from copy import deepcopy
import http.client
import json
from pathlib import Path
import sys
import threading
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.minimax_m2_api import MiniMaxAPITemplate
from serve.minimax_m2_engine import MiniMaxEngine, NativeRequestError, EXE_SHA256
from serve.server import Service, Server, make_handler
from tools.check_minimax_m2_prefix import save, sha
from tools.check_minimax_m2_prefix_context import snapshot, native_checks
from tools.strata_tokenizer import Tokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gguf', type=Path, required=True)
    p.add_argument('--offline', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--engine', type=Path, default=ROOT/'build-local/minimax-m2-cuda/bin/strata-minimax-m2-bench.exe')
    p.add_argument('--cuda-root', type=Path, default=ROOT/'build-local/cuda-13.0')
    args = p.parse_args()
    offline = json.loads((args.offline/'report.json').read_text(encoding='utf-8'))
    assert offline['pass'] and offline['engine_sha256'] == sha(args.engine) == EXE_SHA256
    reference = json.loads((args.offline/'fresh.json').read_text(encoding='utf-8'))['results'][0]['token_ids']
    ids = json.loads((args.offline/'fresh.requests.json').read_text(encoding='utf-8'))[0]['tokens']
    args.out.mkdir(parents=True,exist_ok=False)
    report = {'pass':False,'stage':'MM27-27 HTTP','engine_sha256':EXE_SHA256,
              'offline_report_sha256':sha(args.offline/'report.json'),'cases':[]}
    snapshot(args.out,report)
    source = args.out/'sources/tools/check_minimax_m2_sessions_http.py'
    source.write_bytes(Path(__file__).read_bytes());report['sources']['tools/check_minimax_m2_sessions_http.py']=sha(source)
    engine = server = thread = None
    base = {'model':'minimax-m2.7','messages':[{'role':'user','content':
            'Explain in a few sentences why the sky appears blue during the day.'}], 'max_tokens':8,'temperature':0}
    try:
        engine = MiniMaxEngine(args.gguf,args.engine,args.cuda_root,args.out/'native.stderr.log',context=2048,batch=16,
                              gpu_cache_mib=18432,pipeline_readers=2,pipeline_chunk_mib=4,prefix_cache=True,
                              session_cache_mib=128,session_cache_slots=2)
        report['header'],report['command'] = deepcopy(engine.header),engine.command
        initial_pid = engine.header['native_pid']
        tok = Tokenizer.from_gguf(args.gguf)
        svc = Service(engine,tok,MiniMaxAPITemplate(ROOT/'serve/fixtures/minimax_m27_chat_template.jinja'),model_name='minimax-m2.7')
        server = Server(('127.0.0.1',0),make_handler(svc))
        thread = threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        port = server.server_address[1]
        def record(name,reuse,restore):
            r = deepcopy(engine.last_result)
            checks = native_checks(r,engine.header)
            case = {'name':name,'native':r,'checks':checks,'native_pid':engine.header['native_pid'],
                    'pass':all(checks.values()) and r['token_ids']==reference and r['reused_tokens']==reuse and
                           r['session_restore']==restore and engine.last['reused']==reuse and
                           r['session_archive_bytes']<=128*1024*1024 and r['session_archive_entries']<=2}
            report['cases'].append(case);save(args.out/'report.json',report);assert case['pass'],name
            return case
        def post(name,session,reuse,restore,api='openai',stream=False):
            print('START '+name,flush=True)
            req = {**base,'stream':stream}
            if stream and api=='openai': req['stream_options']={'include_usage':True}
            save(args.out/(name+'.request.json'),req)
            headers = {'Content-Type':'application/json'}
            if session is not None: headers['X-Strata-Session-Id']=session
            conn = http.client.HTTPConnection('127.0.0.1',port,timeout=900)
            try:
                conn.request('POST','/v1/chat/completions' if api=='openai' else '/v1/messages',json.dumps(req),headers)
                response = conn.getresponse();body = response.read().decode('utf-8');status=response.status
            finally: conn.close()
            (args.out/(name+'.response.txt')).write_text(body,encoding='utf-8')
            assert status==200,(name,status,body)
            case = record(name,reuse,restore);case.update(api=api,stream=stream,status=status)
            if stream:
                events = [json.loads(s[6:]) for s in body.splitlines() if s.startswith('data: ') and s!='data: [DONE]']
                assert events and not any('error' in e or e.get('type')=='error' for e in events)
                usage = next(e['usage'] for e in reversed(events) if 'usage' in e)
            else: usage=json.loads(body)['usage']
            if api=='openai': assert usage['prompt_tokens_details']['cached_tokens']==reuse
            else: assert usage.get('cache_read_input_tokens',0)==reuse
            case['usage']=usage;save(args.out/'report.json',report)
        post('a-cold','a',0,False)
        post('b-cold','b',0,False,'anthropic',True)
        post('a-restore','a',48,True,'openai',True)
        post('c-cold','c',0,False)
        post('b-restore','b',48,True,'anthropic')
        post('a-evicted','a',0,False)
        post('anonymous',None,0,False)
        post('a-after-anonymous','a',48,True)
        # Restore B, cancel during decode, then require fresh B. A's immutable
        # checkpoint survives the failure; no partial B is ever archived.
        print('START cancel-restored-b',flush=True)
        gen = engine.generate(ids,32,{},threading.Event(),session_id='b')
        received=[]
        for token in gen:
            if token is not None:
                received.append(token)
                if len(received)==2:break
        gen.close()
        report['cases'].append({'name':'cancel-restored-b','ids':received,'error':deepcopy(engine.last_error),
                               'pass':received==reference[:2] and bool(engine.last_error) and
                                      'cancel' in engine.last_error['message'].lower() and engine.alive()})
        post('b-after-cancel','b',0,False)
        post('a-survives-cancel','a',48,True)
        send = engine._send
        def invalid(item):
            if item.get('command')=='generate':
                item=deepcopy(item);item['request']['session_key']='invalid'
            return send(item)
        rejected=False
        try:
            with patch.object(engine,'_send',side_effect=invalid):
                list(engine.generate(ids,8,{},threading.Event(),session_id='a'))
        except NativeRequestError: rejected=True
        report['cases'].append({'name':'invalid-request','error':deepcopy(engine.last_error),'pass':rejected and engine.alive()})
        post('b-survives-other-error','b',48,True)
        assert engine.header['native_pid']==initial_pid
        old=engine.proc;engine.restart()
        report['restart']={'previous_pid':initial_pid,'new_pid':engine.header['native_pid'],'supervisor_exit':old.poll()}
        assert old.poll() is not None and engine.header['native_pid']!=initial_pid
        post('b-after-restart','b',0,False)
        report['pass']=all(c['pass'] for c in report['cases']);assert report['pass']
    except BaseException:
        report['error']=traceback.format_exc();raise
    finally:
        errors=[]
        for action in ([engine.close] if engine else [])+([server.shutdown,server.server_close] if server else [])+([lambda:thread.join(5)] if thread else []):
            try:action()
            except BaseException as e:errors.append(str(e))
        if errors:report['cleanup_errors']=errors;report['pass']=False
        save(args.out/'report.json',report)
    print('PASS '+str(len(report['cases']))+' HTTP/lifecycle scenarios',flush=True)


if __name__=='__main__':main()
