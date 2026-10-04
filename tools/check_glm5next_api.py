"""Explicit real-model loopback test of server.main with the GLM profile.

Checks OpenAI/Anthropic JSON and SSE, disconnect/next-request recovery and unload.
The child runs the normal server main; stdin QUIT triggers its Ctrl+C cleanup.
"""
import argparse
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def child():
    import _thread
    from serve.server import main
    sys.argv=[str(ROOT/'serve/server.py'),*sys.argv[2:]]
    def stop():
        for line in sys.stdin:
            if line.strip()=='QUIT':
                _thread.interrupt_main(); return
    threading.Thread(target=stop,daemon=True).start()
    return main()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile',type=Path,required=True)
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--port',type=int,default=8081)
    a=ap.parse_args()
    # Never send test messages to a pre-existing local server.
    with socket.socket() as probe: probe.bind(('127.0.0.1',a.port))
    reference=json.loads(a.reference.read_text(encoding='utf-8'))
    expected=reference['output_text'].split('</think>',1)[1]
    report={'status':'error','cases':[],'profile':str(a.profile.resolve())}
    def connect(): return http.client.HTTPConnection('127.0.0.1',a.port,timeout=240)
    def request(path,payload=None):
        c=connect()
        try:
            c.request('GET' if payload is None else 'POST',path,None if payload is None else json.dumps(payload).encode(),
                      {'Content-Type':'application/json','Origin':f'http://127.0.0.1:{a.port}'})
            response=c.getresponse(); data=response.read().decode('utf-8')
            assert response.status==200,(response.status,data)
            return data
        finally: c.close()
    def wait_for(predicate,timeout=90):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            if predicate(): return
            time.sleep(.1)
        raise AssertionError('server state did not settle')
    def payload(api,stream=False,count=64):
        return {'model':'glm-5.3-flash','messages':[{'role':'user','content':
            'Explain in English how asynchronous expert offloading helps run a mixture-of-experts language model on one GPU. Give three practical points.'}],
            'temperature':0,'max_tokens':count,'stream':stream,
            **({'reasoning_effort':'low'} if api=='openai' else {'output_config':{'effort':'low'}})}
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--server-child','--engine','strata',
             '--config',str(a.profile.resolve()),'--host','127.0.0.1','--port',str(a.port),'--statistics-file',':memory:']
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.with_suffix('.server.log').open('w',encoding='utf-8') as log:
        p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,text=True,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        try:
            def ready():
                assert p.poll() is None,'server exited during startup'
                try: return json.loads(request('/health')).get('status')=='ok'
                except OSError: return False
            wait_for(ready,120)
            report['models']=json.loads(request('/v1/models'))
            metrics=json.loads(request('/metrics')); report['engine']=metrics['engine']
            assert report['engine']['architecture']=='glm5next' and report['engine']['gpu_only']==1
            for api in ('openai','anthropic'):
                path='/v1/chat/completions' if api=='openai' else '/v1/messages'
                for stream in (False,True):
                    start=time.monotonic(); raw=request(path,payload(api,stream))
                    if not stream:
                        data=json.loads(raw)
                        if api=='openai':
                            text=data['choices'][0]['message']['content']
                            assert data['choices'][0]['finish_reason']=='length'
                        else:
                            text=''.join(x.get('text','') for x in data['content'] if x['type']=='text')
                            assert data['stop_reason']=='max_tokens'
                    else:
                        entries=[l[6:] for l in raw.splitlines() if l.startswith('data: ')]
                        if api=='openai':
                            assert entries.count('[DONE]')==1 and entries[-1]=='[DONE]'
                            data=[json.loads(e) for e in entries if e!='[DONE]']
                            text=''.join(c.get('delta',{}).get('content','') or '' for e in data for c in e.get('choices',[]))
                        else:
                            data=[json.loads(e) for e in entries]
                            assert sum(e['type']=='message_stop' for e in data)==1
                            text=''.join(e.get('delta',{}).get('text','') for e in data if e['type']=='content_block_delta')
                    assert text==expected,(api,stream,text,expected)
                    report['cases'].append({'name':api+(' SSE' if stream else ' JSON'),'wall_seconds':time.monotonic()-start,'response':data})
                    print(report['cases'][-1]['name']+' passed',flush=True)
            for phase in ('prefill','decode'):
                c=connect(); response=None
                try:
                    c.request('POST','/v1/chat/completions',json.dumps(payload('openai',True,256)).encode(),{'Content-Type':'application/json'})
                    if phase=='prefill':
                        wait_for(lambda: json.loads(request('/metrics'))['live']['state']=='reading',20)
                    else:
                        response=c.getresponse(); assert response.status==200
                        while True:
                            line=response.readline().decode('utf-8')
                            assert line,'stream ended before disconnect'
                            if line.startswith('data: ') and line.strip()!='data: [DONE]':
                                event=json.loads(line[6:])
                                if any(c.get('delta',{}).get('content') for c in event.get('choices',[])): break
                    start=time.monotonic()
                    if response: response.close()
                    c.close()
                    wait_for(lambda: json.loads(request('/metrics'))['live']['state']=='idle',90)
                    metrics=json.loads(request('/metrics'))
                    assert metrics['requests'][0]['finish'] in ('cancel','disconnect'),metrics['requests'][0]
                    report['cases'].append({'name':'HTTP disconnect '+phase,'cancel_seconds':time.monotonic()-start,'last_request':metrics['requests'][0]})
                    data=json.loads(request('/v1/chat/completions',payload('openai',False,8)))
                    assert data['choices'][0]['message']['content']==expected_prefix(reference,8)
                    report['cases'].append({'name':'clean after HTTP '+phase,'response':data})
                    print('HTTP '+phase+' disconnect/recovery passed',flush=True)
                finally:
                    if response: response.close()
                    c.close()
            report['unload']=json.loads(request('/unload',{}))
            assert report['unload']['status']=='unloaded'
            report['metrics_after_unload']=json.loads(request('/metrics'))
            assert report['metrics_after_unload']['live']['state']=='unloaded'
            report['status']='pass'
        except Exception as e:
            report['error']=str(e)
            raise
        finally:
            if p.poll() is None:
                p.stdin.write('QUIT\n'); p.stdin.flush()
                try: p.wait(timeout=60)
                except subprocess.TimeoutExpired: p.kill(); p.wait()
            report['server_exit_code']=p.returncode
            if p.returncode!=0:
                report['status']='error'
                report['shutdown_error']='server did not exit cleanly'
            a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    assert p.returncode==0,('server shutdown',p.returncode)


def expected_prefix(reference,count):
    from tools.strata_tokenizer import Tokenizer
    # Use the tested GGUF tokenizer, not byte slicing inside a Unicode token.
    tok=Tokenizer.from_gguf(reference['configuration']['model'])
    return tok.decode(reference['generated_ids'][:count]).split('</think>',1)[1]


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--server-child': sys.exit(child())
    main()
