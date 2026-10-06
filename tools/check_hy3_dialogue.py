"""Real Hy3 tool/result continuation with a local fixed-value stub (no API).

The checkpoint is read-only. Only the spawned engine can be stopped by the
global 95% RAM/VRAM monitor. This is correctness, not a performance benchmark.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.hy3 import Hy3Template, Hy3OutputParser, TEMPLATE_SHA256
from check_hy3_engine import Engine
from check_hy3_cuda import check_ceiling, memory, gpu_memory
from check_step35_model import Monitor
from gguf_reader import GGUFFile
from inspect_hy3_gguf import inspect_model
from strata_tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--gguf', type=Path, required=True)
    parser.add_argument('--cuda-bin', type=Path, required=True)
    parser.add_argument('--template-oracle', type=Path, required=True)
    parser.add_argument('--tokenizer-oracle', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    model, binary = args.gguf.resolve(), args.engine.resolve()
    inventory = inspect_model(model)
    source = GGUFFile(model).metadata['tokenizer.chat_template']
    if hashlib.sha256(source.encode()).hexdigest() != TEMPLATE_SHA256:
        raise ValueError('unreviewed model template')
    fixture = directory/'chat_template.jinja'
    fixture.write_bytes(source.encode('utf8'))
    template, tokenizer = Hy3Template(fixture), Tokenizer.from_gguf(model)
    tools = [{'type': 'function', 'function': {'name': 'get_temperature',
        'description': 'Get the temperature for a city.', 'parameters': {'type': 'object',
        'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}]
    history = [{'role': 'user', 'content': 'What is the temperature in Paris? Use get_temperature, then answer with the returned number and C.'}]
    report = dict(status='error', scope='real checkpoint tool/parser/result continuation with a fixed local stub; no HTTP or external tools',
        model=str(model), engine_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        header_sha256=inventory['header_sha256'], template_sha256=TEMPLATE_SHA256,
        context=2048, batch=17, kv='f32', mtp=False, cache=False, pipeline=False, copy_mode='pinned',
        reasoning_effort='no_think', temperature=0, turns=[])
    path = directory/'dialogue-report.json'
    def save():
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    engine, monitor = None, Monitor()
    try:
        check_ceiling(memory(gpu_memory()))
        os.environ['PATH'] = str(args.cuda_bin.resolve())+os.pathsep+os.environ.get('PATH', '')
        engine = Engine(binary, model, directory, mode='pinned', logits=False, on_start=monitor.start)
        report['info'] = engine.info
        for index in range(2):
            context = dict(messages=history, tools=tools, reasoning_effort='no_think', add_generation_prompt=True)
            rendered = template.render(**context)
            ids = tokenizer.encode(rendered, parse_special=True)
            native = subprocess.run([str(args.template_oracle.resolve())],
                input=(json.dumps({'template': source, 'context': context})+'\nQUIT\n').encode(),
                capture_output=True, check=True, timeout=30)
            native_prompt = json.loads(native.stdout.splitlines()[0])['rendered']
            encoded = subprocess.run([str(args.tokenizer_oracle.resolve()), '--gguf', str(model)],
                input=(json.dumps({'text': rendered, 'parse_special': True})+'\nQUIT\n').encode(),
                capture_output=True, check=True, timeout=30)
            native_ids = json.loads(encoded.stdout.splitlines()[1])['ids']
            if rendered != native_prompt or ids != native_ids:
                raise ValueError('runtime prompt differs from native template/tokenizer')
            turn = dict(index=index, context=json.loads(json.dumps(context)), prompt=rendered, prompt_ids=ids,
                        native_prompt_and_ids_exact=True)
            report['turns'].append(turn)
            save()
            print('Generating turn', index, 'prompt tokens', len(ids), flush=True)
            result = engine.generate(ids, 96, sampling='temperature=0',
                on_progress=lambda done, total: print('prefill', done, '/', total, flush=True))
            text = b''.join(tokenizer.token_bytes(i) for i in result['ids'] if i != 120025).decode('utf8')
            output = Hy3OutputParser(thinking=False, tools=tools)
            events = output.feed(text)+output.finish()
            calls = [event.call for event in events if event.kind == 'tool_call']
            content = ''.join(e.text for e in events if e.kind == 'content')
            reasoning = ''.join(e.text for e in events if e.kind == 'reasoning')
            turn.update(result=result, raw_text=text, content=content, reasoning=reasoning,
                        calls=[dict(id=c.id, name=c.name, arguments=c.arguments) for c in calls])
            save()
            print('Turn', index, ascii(text), flush=True)
            if result['finish'] != 'stop':
                raise ValueError('tool dialogue did not reach EOS')
            if index == 0:
                if len(calls) != 1 or calls[0].name != 'get_temperature' or calls[0].arguments != {'city': 'Paris'}:
                    raise ValueError('expected one complete typed temperature call for Paris')
                call = calls[0]
                history.append(dict(role='assistant', content=content, reasoning_content=reasoning,
                    tool_calls=[dict(id=call.id, type='function', function=dict(name=call.name, arguments=call.arguments))]))
                stub = {'city': 'Paris', 'temperature_c': 17}
                history.append(dict(role='tool', tool_call_id=call.id, content=json.dumps(stub)))
                report['local_stub_result'] = stub
            elif calls or '17' not in content or not content.strip():
                raise ValueError('expected final answer using the local stub result')
        report['status'] = 'pass'
    except Exception as error:
        report['error'] = str(error)
    finally:
        try:
            if engine:
                engine.close()
                report['exit_code'] = engine.process.returncode
                if engine.process.returncode:
                    report.update(status='error', exit_error='engine did not exit cleanly')
        finally:
            monitor.close()
            report.update(memory_samples=monitor.samples, monitor_error=monitor.error)
            if monitor.error:
                report.update(status='error', error=monitor.error)
            save()
    print(report['status'], report.get('error', ''), path, flush=True)
    return int(report['status'] != 'pass')


if __name__ == '__main__':
    raise SystemExit(main())
