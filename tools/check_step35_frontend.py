"""Real Step template/tokenizer/parser smoke, with a local tool-result stub.

No HTTP server or external tool is invoked. A new output directory is required.
The native render/token-ID oracles check each actual prompt before generation.
"""
import argparse
import codecs
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.step35 import StepTemplate, StepOutputParser
from tools.check_step35_engine import Engine
from tools.check_step35_model import Monitor
from tools.check_step35_tokenizer import validate_provenance
from tools.gguf_reader import GGUFFile
from tools.setup_step35 import first_shard
from tools.strata_tokenizer import Tokenizer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parsed(chunks, tools=None):
    parser=StepOutputParser(tools=tools,stream_tools=True);events=[]
    for chunk in chunks:events.extend(parser.feed(chunk))
    events.extend(parser.finish())
    return {'reasoning':''.join(e.text for e in events if e.kind=='reasoning'),
            'content':''.join(e.text for e in events if e.kind=='content'),
            'calls':[{'name':e.call.name,'arguments':e.call.arguments} for e in events if e.kind=='tool_call']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('engine','gguf','reference','template-oracle','tokenizer-oracle','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    model=first_shard(args.gguf).resolve()
    reference=json.loads(args.reference.read_text(encoding='utf8'))
    if reference['status']!='pass' or Path(reference['model']).resolve()!=model:
        p.error('a passing reference for this GGUF is required')
    directory=args.output_dir.resolve();directory.mkdir(parents=True,exist_ok=False)
    meta=GGUFFile(model).metadata;vocab=meta['tokenizer.ggml.tokens']
    source=meta['tokenizer.chat_template'];(directory/'chat_template.jinja').write_bytes(source.encode())
    bos,eos=vocab[meta['tokenizer.ggml.bos_token_id']],vocab[meta['tokenizer.ggml.eos_token_id']]
    template=StepTemplate(directory/'chat_template.jinja',bos_token=bos,eos_token=eos)
    tokenizer=Tokenizer.from_gguf(model)
    tok_version=json.loads(subprocess.run([str(args.tokenizer_oracle.resolve()),'--version'],capture_output=True,check=True).stdout)
    validate_provenance(tok_version)
    report={'status':'error','scope':'real engine pipe/template/parser; local tool result stub, no HTTP or external tools',
            'engine_sha256':sha(args.engine),'reference_sha256':sha(args.reference),'model':str(model),
            'template_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'tokenizer_oracle_provenance':tok_version,'tokenizer_oracle_sha256':sha(args.tokenizer_oracle),
            'template_oracle_sha256':sha(args.template_oracle),
            'context':4096,'batch':17,'cache':'auto','readers':1,'prefill_admission':'off','kv':'f32','mtp':False,
            'requests':[]}
    def save():
        (directory/'frontend-model-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    def prompt(messages,tools=None):
        rendered=template.render(messages,tools,reasoning_effort='low')
        ids=tokenizer.encode(rendered,parse_special=True)
        context={'messages':messages,'tools':tools,'reasoning_effort':'low','add_generation_prompt':True,'bos_token':bos,'eos_token':eos}
        oracle=subprocess.run([str(args.template_oracle.resolve())],input=(json.dumps({'template':source,'context':context},ensure_ascii=False)+'\nQUIT\n').encode(),capture_output=True,check=True,timeout=30)
        if json.loads(oracle.stdout)['rendered']!=rendered:raise ValueError('native template mismatch')
        oracle=subprocess.run([str(args.tokenizer_oracle.resolve()),'--gguf',str(model)],input=(json.dumps({'text':rendered,'parse_special':True},ensure_ascii=False)+'\nQUIT\n').encode(),capture_output=True,check=True,timeout=30)
        row=json.loads(oracle.stdout.decode().splitlines()[1])
        if row['ids']!=ids or row['decoded_hex']!=rendered.encode().hex():raise ValueError('native tokenizer mismatch')
        return rendered,ids
    engine=None;monitor=Monitor()
    try:
        # Reviewed native EOG set for this fixed model. API stop resolution is
        # deliberately not implemented by this harness or inferred from PAD.
        if vocab[1]!='<｜end▁of▁sentence｜>' or meta['tokenizer.ggml.eos_token_id']!=128007:
            raise ValueError('unreviewed native EOG metadata')
        started=time.perf_counter()
        engine=Engine(args.engine.resolve(),model,directory,logits=True,on_start=monitor.start,
            extra_args=['--max-context','4096','--expert-cache-mib','auto','--expert-pipeline-readers','1','--expert-cache-prefill','off'])
        report.update(info=engine.info,ready_seconds=time.perf_counter()-started)
        print('Real Step engine READY',flush=True)
        def generate(name,messages,tools=None,count=256):
            rendered,ids=prompt(messages,tools)
            print(f'{name}: {len(ids)} input tokens; native prompt/IDs exact',flush=True)
            last=time.perf_counter()
            def progress(done,total):
                nonlocal last
                if time.perf_counter()-last>25:
                    print(f'{name}: prefill {done}/{total}',flush=True);last=time.perf_counter()
            result=engine.generate(ids,count,sampling='temperature=0',on_progress=progress)
            result.update(name=name,prompt_ids=ids,prompt=rendered,messages=messages,tools=tools,
                          logits_sha256=sha(directory/'logits.f32'),logits_bytes=(directory/'logits.f32').stat().st_size)
            report['requests'].append(result)
            if result['finish']!='stop' or result['ids'][-1] not in (1,128007):raise ValueError(name+': incomplete native response')
            decoder=codecs.getincrementaldecoder('utf8')(errors='strict')
            chunks=[decoder.decode(tokenizer.token_bytes(t)) for t in result['ids'][:-1]]
            chunks.append(decoder.decode(b'',final=True))
            text=''.join(chunks);result['raw_text']=text
            value=parsed(chunks,tools)
            if value!=parsed([text],tools) or value!=parsed(list(text),tools):raise ValueError('parser depends on chunking')
            result.update(parsed=value,native_prompt_exact=True,token_boundary_parser_exact=True)
            print(f'{name}: {len(result["ids"])} output, {len(value["calls"])} tools; complete and chunk-exact',flush=True)
            save();return result
        baseline=generate('P1_A',[{'role':'user','content':reference['prompts'][0]['user']}],count=reference['predict'])
        expected=reference['runs'][0]['requests'][0]
        if baseline['prompt_ids']!=reference['prompts'][0]['ids'] or baseline['ids']!=expected['ids'] or baseline['logits_sha256']!=expected['logits_sha256']:
            raise ValueError('P1 exact reference changed')
        if baseline['parsed']['content'].strip()!='4' or baseline['parsed']['calls']:raise ValueError('P1 parser mismatch')
        baseline['P1_exact']=True
        tools=[{'type':'function','function':{'name':'lookup_weather','description':'Return a local test weather forecast.',
                'parameters':{'type':'object','properties':{'city':{'type':'string'},'days':{'type':'integer'},'include_wind':{'type':'boolean'}},
                              'required':['city','days','include_wind']}}}]
        messages=[{'role':'user','content':'Use lookup_weather once with city="Yekaterinburg", days=2, include_wind=true. Do not guess the weather. Once data is available, give a one-sentence answer.'}]
        tool=generate('tool_call',messages,tools)
        expected_call={'name':'lookup_weather','arguments':{'city':'Yekaterinburg','days':2,'include_wind':True}}
        if tool['parsed']['calls']!=[expected_call]:raise ValueError('model did not produce the expected typed call')
        history=messages+[{'role':'assistant','content':tool['parsed']['content'],'reasoning_content':tool['parsed']['reasoning'],
            'tool_calls':[{'id':'local_step_test','type':'function','function':expected_call}]},
            {'role':'tool','tool_call_id':'local_step_test','content':'Local test result: Yekaterinburg, day 1: 7 C, clear, light wind; day 2: 8 C, clear, light wind.'}]
        continuation=generate('tool_continuation',history,tools)
        if continuation['parsed']['calls'] or not continuation['parsed']['content'].strip():raise ValueError('missing final tool response')
        engine.close();report['exit_code']=engine.process.returncode
        if engine.process.returncode:raise ValueError('engine exit failed')
        report['status']='pass'
    except Exception as error:
        report['error']=str(error)
    finally:
        if engine and not engine.stderr.closed:engine.close()
        monitor.close();report.update(memory_samples=monitor.samples,monitor_error=monitor.error)
        if monitor.error:report.update(status='error',error=monitor.error)
        if engine:
            report['metrics']=[json.loads(s.split(' ',1)[1]) for s in engine.stderr_path.read_text(encoding='utf8',errors='replace').splitlines() if s.startswith('STRATA_STEP_REQUEST ')]
        save()
    print(report['status'],report.get('error',''),flush=True)
    return int(report['status']!='pass')


if __name__=='__main__':
    raise SystemExit(main())
