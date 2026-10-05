"""Step full-model context growth and bounded external RAM/VRAM pressure.

Separate sequential sync/pipeline processes, exact F32 logits, no speed claim.
The holder is started before the model so its CUDA context is already budgeted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import threading

from check_step35_engine import Engine
from check_step35_model import Monitor
from check_step35_template import renderer
from gguf_reader import GGUFFile
from setup_step35 import first_shard
from strata_tokenizer import Tokenizer


class Holder:
    def __init__(self, binary, directory):
        self.stderr=(directory/'holder-stderr.log').open('w',encoding='utf8')
        self.process=subprocess.Popen([str(binary)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=self.stderr,text=True,encoding='utf8',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.lines=queue.Queue()
        def pump():
            for line in self.process.stdout:self.lines.put(line)
            self.lines.put(None)
        self.reader=threading.Thread(target=pump,daemon=True);self.reader.start()
        self.records=[]
    def expect(self, phase):
        line=self.lines.get(timeout=30)
        if line is None:raise ValueError('holder exited before '+phase)
        record=json.loads(line);self.records.append(record)
        if record.get('phase')!=phase:raise ValueError('unexpected holder reply: '+str(record))
        if record['gpu_free']<record['gpu_total']*0.05 or record['ram_free']<record['ram_total']*0.05:
            raise ValueError('holder global memory guard')
        return record
    def command(self, command, phase):
        self.process.stdin.write(command+'\n');self.process.stdin.flush();return self.expect(phase)
    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close() # EOF releases only this helper's allocations.
            try:self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=10)
        self.reader.join(timeout=3)
        if not self.process.stdin.closed:self.process.stdin.close()
        self.process.stdout.close();self.stderr.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',type=Path,required=True)
    parser.add_argument('--holder',type=Path,required=True)
    parser.add_argument('--gguf',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    directory=args.output_dir.resolve();directory.mkdir(parents=True,exist_ok=True)
    model=first_shard(args.gguf)
    reference=json.loads(args.reference.read_text(encoding='utf8'))
    if reference['status']!='pass' or Path(reference['model']).resolve()!=model.resolve():
        parser.error('a passing reference for the same model is required')
    meta=GGUFFile(model).metadata;vocab=meta['tokenizer.ggml.tokens']
    template=renderer(meta['tokenizer.chat_template']);tokenizer=Tokenizer.from_gguf(model)
    body=''.join(f'Record {i:03d}: the red boat crossed the river while the green train waited beside the old station.\n' for i in range(128))
    body+='Which color was the boat? Answer briefly.'
    rendered=template.render(messages=[{'role':'user','content':body}],tools=None,add_generation_prompt=True,
        reasoning_effort='low',bos_token=vocab[meta['tokenizer.ggml.bos_token_id']],eos_token=vocab[meta['tokenizer.ggml.eos_token_id']])
    long_ids=tokenizer.encode(rendered,parse_special=True)
    if not 2048<len(long_ids)<4000:raise ValueError('stress prompt must cross previous 2048 context limit')
    report={'status':'error','scope':'context4096, prompt over2048, bounded real external pressure; no benchmark claim',
        'engine_sha256':hashlib.sha256(args.engine.read_bytes()).hexdigest(),
        'holder_sha256':hashlib.sha256(args.holder.read_bytes()).hexdigest(),
        'reference_sha256':hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        'model':str(model),'context':4096,'batch':17,'kv':'f32','cache':'auto','mtp':False,
        'long_prompt':{'text':body,'ids':long_ids},'runs':[]}
    def save():
        (directory/'context-report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    def metrics(engine):
        return [json.loads(line.split(' ',1)[1]) for line in engine.stderr_path.read_text(encoding='utf8',errors='replace').splitlines()
                if line.startswith('STRATA_STEP_REQUEST ')]
    controls={}
    try:
        print(f'Long prompt: {len(long_ids)} tokens, context=4096',flush=True)
        for readers in (0,1):
            work=directory/f'readers-{readers}';work.mkdir(exist_ok=True)
            run={'readers':readers,'requests':[]};report['runs'].append(run)
            monitor=Monitor();engine=holder=None
            try:
                holder=Holder(args.holder.resolve(),work);run['holder_ready']=holder.expect('ready')
                engine=Engine(args.engine.resolve(),model,work,logits=True,on_start=monitor.start,
                    extra_args=['--max-context','4096','--expert-cache-mib','auto','--expert-pipeline-readers',str(readers)])
                run['info']=engine.info
                def generate(name, ids, count):
                    result=engine.generate(ids,count,sampling='temperature=0')
                    raw=(work/'logits.f32').read_bytes()
                    result.update(name=name,logits_sha256=hashlib.sha256(raw).hexdigest(),logits_bytes=len(raw))
                    run['requests'].append(result)
                    key='long' if name=='long' else 'short'
                    if key not in controls:
                        controls[key]=result
                        (work/(key+'-reference.f32')).write_bytes(raw)
                    expected=controls[key]
                    if result['ids']!=expected['ids'] or result['logits_sha256']!=expected['logits_sha256']:
                        raise ValueError('exact logits/IDs differ: '+name)
                    result['exact']=True
                    print(f'readers={readers} {name}: exact, {len(result["ids"])} output, prefill={result["prefill_ms"]/1000:.2f}s',flush=True)
                    save();return result
                short=reference['prompts'][0]['ids']
                result=generate('short',short,reference['predict'])
                expected=reference['runs'][0]['requests'][0]
                run['short_matches_P1']=result['ids']==expected['ids'] and result['logits_sha256']==expected['logits_sha256']
                if not run['short_matches_P1']:raise ValueError('context4096 short request differs from P1')
                generate('long',long_ids,16)
                # Re-establish the same short working set before pressure.
                generate('before_pressure',short,reference['predict'])
                before=metrics(engine)[-1];run['before_pressure_metrics']=before
                allocated=holder.command('ALLOC','allocated')
                if allocated['gpu_bytes']!=128*2**20 or allocated['ram_bytes']!=2*2**30:
                    raise ValueError('wrong holder allocation size')
                generate('under_pressure',short,reference['predict'])
                after=metrics(engine)[-1];run['under_pressure_metrics']=after
                run['cache_reduction_bytes']=before['cache_bytes']-after['cache_bytes']
                if run['cache_reduction_bytes']<=0:raise ValueError('cache did not shrink under real external pressure')
                holder.command('FREE','freed')
                generate('after_pressure',short,reference['predict'])
                # Cancel a long prompt after progress, then check fresh state.
                cancelled=engine.generate(long_ids,128,cancel=True);run['cancelled']=cancelled
                if cancelled['finish']!='cancel':raise ValueError('STOP did not cancel long prompt')
                generate('after_cancel',short,reference['predict'])
                engine.close();run['exit_code']=engine.process.returncode
                if run['exit_code']:raise ValueError('engine exit failed')
            finally:
                if holder:
                    holder.close();run['holder_records']=holder.records;run['holder_exit_code']=holder.process.returncode
                if engine and not engine.stderr.closed:engine.close()
                monitor.close();run['monitor_error']=monitor.error;run['memory_samples']=monitor.samples
                if engine:run['metrics']=metrics(engine)
                save()
            if monitor.error:raise ValueError(monitor.error)
            if run['holder_exit_code']:raise ValueError('holder exit failed')
        report['status']='pass'
    except Exception as error:report['error']=str(error)
    save();print(report['status'],report.get('error',''),flush=True)
    return int(report['status']!='pass')


if __name__=='__main__':
    raise SystemExit(main())
