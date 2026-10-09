"""MM27-29 admitted-engine HTTP restore/cancel/restart regression at ctx512.

128 generated tokens make the archived V dense enough to exercise bulk IO.
Reference output is the previously audited MM27-28 greedy EOS sequence.
"""
import argparse
from copy import deepcopy
import http.client
import json
from pathlib import Path
import sys
import threading
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from serve.minimax_m2_api import MiniMaxAPITemplate
from serve.minimax_m2_engine import MiniMaxEngine, EXE_SHA256
from serve.server import Service, Server, make_handler
from tools.check_minimax_m2_prefix import save
from tools.check_minimax_m2_prefix_context import snapshot, native_checks
from tools.check_minimax_m2_state_bulk import sha
from tools.strata_tokenizer import Tokenizer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gguf',type=Path,required=True)
    p.add_argument('--offline',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=ROOT/'build-local/minimax-m2-sessions-context-01')
    p.add_argument('--cuda-root',type=Path,default=ROOT/'build-local/cuda-13.0')
    args=p.parse_args()
    binary=ROOT/'build-local/minimax-m2-cuda/bin/strata-minimax-m2-bench.exe'
    audited=json.loads((args.offline/'report.json').read_text(encoding='utf-8'))
    old=json.loads((args.reference/'report.json').read_text(encoding='utf-8'))
    assert audited['pass'] and audited['engine_sha256']==EXE_SHA256==sha(binary)
    assert old['pass'] and sha(args.reference/'report.json')==audited['reference_report_sha256']
    for name in ['generation.json','http-requests.json','requests.json']:
        assert sha(args.reference/name)==old['artifacts'][name],name
    reference=json.loads((args.reference/'generation.json').read_text())['results'][6]['token_ids']
    base=json.loads((args.reference/'http-requests.json').read_text())['english']
    ids=json.loads((args.reference/'requests.json').read_text())[1]['tokens']
    args.out.mkdir(parents=True,exist_ok=False)
    report={'pass':False,'stage':'MM27-29 live','engine_sha256':EXE_SHA256,
            'offline_report_sha256':sha(args.offline/'report.json'),'cases':[]}
    snapshot(args.out,report)
    source=args.out/'sources/tools/check_minimax_m2_state_bulk_http.py'
    source.write_bytes(Path(__file__).read_bytes());report['sources']['tools/check_minimax_m2_state_bulk_http.py']=sha(source)
    engine=server=thread=None
    try:
        engine=MiniMaxEngine(args.gguf,binary,args.cuda_root,args.out/'native.stderr.log',context=512,batch=16,
                            gpu_cache_mib=18432,pipeline_readers=2,pipeline_chunk_mib=4,prefix_cache=True,
                            session_cache_mib=256,session_cache_slots=2)
        assert engine.env.get('STRATA_MM27_STATE_BULK')!='0','bulk IO must be enabled for this audit'
        report['bulk_environment']=engine.env.get('STRATA_MM27_STATE_BULK','default-on')
        report['header'],report['command']=deepcopy(engine.header),engine.command
        tok=Tokenizer.from_gguf(args.gguf)
        svc=Service(engine,tok,MiniMaxAPITemplate(ROOT/'serve/fixtures/minimax_m27_chat_template.jinja'),model_name='minimax-m2.7')
        server=Server(('127.0.0.1',0),make_handler(svc));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        def post(name,key,count,reuse,restored,api='openai',stream=False):
            req={**base,'max_tokens':count,'stream':stream}
            if api=='openai' and stream:req['stream_options']={'include_usage':True}
            conn=http.client.HTTPConnection(*server.server_address,timeout=600)
            print('START '+name,flush=True)
            save(args.out/(name+'.request.json'),req)
            try:
                conn.request('POST','/v1/chat/completions' if api=='openai' else '/v1/messages',json.dumps(req),
                             {'Content-Type':'application/json','X-Strata-Session-Id':key})
                response=conn.getresponse();body=response.read().decode('utf-8');status=response.status
            finally:conn.close()
            (args.out/(name+'.response.txt')).write_text(body,encoding='utf-8')
            r=deepcopy(engine.last_result);checks=native_checks(r,engine.header)
            checks.update(status=status==200,ids=r['token_ids']==reference[:count],reuse=r['reused_tokens']==reuse,
                          restore=r['session_restore']==restored,budget=r['session_archive_bytes']<=256*1024**2)
            if stream:
                events=[json.loads(line[6:]) for line in body.splitlines() if line.startswith('data: ') and line!='data: [DONE]']
                checks['stream']=bool(events) and not any('error' in e or e.get('type')=='error' for e in events)
                usage=next(e['usage'] for e in reversed(events) if 'usage' in e)
            else:usage=json.loads(body)['usage']
            checks['usage']=(usage['prompt_tokens_details']['cached_tokens'] if api=='openai' else usage.get('cache_read_input_tokens',0))==reuse
            report['cases'].append({'name':name,'native':r,'checks':checks,'pass':all(checks.values())})
            save(args.out/'report.json',report);assert all(checks.values()),(name,checks)
        post('a-long','a',128,0,False)
        post('b-cold','b',8,0,False,'anthropic',True)
        post('a-bulk-restored','a',8,48,True,'openai',True)
        assert report['cases'][-1]['native']['session_restored_bytes']>80*1024**2
        post('b-restored','b',8,48,True,'anthropic')
        post('a-long-again','a',128,48,True)
        post('b-before-cancel','b',8,48,True)
        cancel=threading.Event();gen=engine.generate(ids,8,{},cancel,session_id='a');received=[]
        try:
            for token in gen:
                if token is not None:
                    received.append(token)
                    if len(received)==2:cancel.set()
        finally:gen.close()
        ok=received==reference[:2] and engine.alive() and bool(engine.last_error) and 'cancel' in engine.last_error['message'].lower()
        report['cases'].append({'name':'cancel-bulk-restored','ids':received,'error':deepcopy(engine.last_error),'pass':ok});assert ok
        post('b-survives-cancel','b',8,48,True)
        post('a-fresh-after-cancel','a',8,0,False)
        pid=engine.header['native_pid'];old_proc=engine.proc;engine.restart()
        report['restart']={'old_pid':pid,'new_pid':engine.header['native_pid'],'old_supervisor_exit':old_proc.poll()}
        assert old_proc.poll() is not None and engine.header['native_pid']!=pid
        post('a-after-restart','a',8,0,False)
        report['pass']=all(c['pass'] for c in report['cases'])
    except BaseException:
        report['error']=traceback.format_exc();raise
    finally:
        errors=[]
        for action in ([engine.close] if engine else [])+([server.shutdown,server.server_close] if server else [])+([lambda:thread.join(5)] if thread else []):
            try:action()
            except BaseException as e:errors.append(str(e))
        if errors:report['cleanup_errors']=errors;report['pass']=False
        report['artifacts']={f.name:sha(f) for f in args.out.iterdir() if f.is_file() and f.name!='report.json'}
        save(args.out/'report.json',report)
    print('PASS '+str(len(report['cases']))+' live cases',flush=True)


if __name__=='__main__':main()
