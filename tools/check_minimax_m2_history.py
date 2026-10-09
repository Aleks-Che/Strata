"""History unit/golden/native-Jinja/token-ID checks; reads GGUF metadata, no GPU."""
import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.minimax_m2_history import MiniMaxChatTemplate, prepare_minimax_context, TEMPLATE_SHA256
from tools.check_minimax_m2_oracles import provenance, run_lines, template_corpus
from tools.gguf_reader import GGUFFile
from tools.inspect_minimax_m2_gguf import inspect_model
from tools.minimax_m2_history_cases import history_cases, numeric_cases, USER
from tools.strata_tokenizer import Tokenizer
from tools.minimax_m2_template import renderer, text_context


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bin-dir', type=Path, default=ROOT/'build-local/minimax-m2-oracles/bin')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {'pass': False, 'scope': 'unit methods, handwritten prompts, native Jinja with isolated float JSON fix, native token IDs; no inference/API',
              'model': str(args.gguf.resolve()), 'source_sha256': {}, 'cases': []}
    try:
        for rel in ['serve/minimax_m2_history.py','serve/test_minimax_m2_history.py','tools/minimax_m2_history_cases.py',
                    'tools/check_minimax_m2_history.py','serve/fixtures/minimax_m27_chat_template.jinja',
                    'serve/minimax_m2.py','serve/test_minimax_m2.py','serve/frontend.py','tools/minimax_m2_template.py',
                    'tools/test_minimax_m2_tokenizer.py','tools/strata_tokenizer.py','tools/check_minimax_m2_oracles.py','.gitattributes',
                    'backends/minimax_m2/TemplateJSON.cmake','backends/minimax_m2/template_oracle.cpp',
                    'backends/minimax_m2/CMakeLists.txt','backends/minimax_m2/build_manifest.json.in']:
            dest = args.out/'sources'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/rel,dest);report['source_sha256'][rel] = sha(dest)
        suffix = '.exe' if sys.platform == 'win32' else ''
        binaries = {}
        for kind in ['template-json','template','tokenizer']:
            source = args.bin_dir/('strata-minimax-m2-'+kind+suffix)
            dest = args.out/source.name;shutil.copyfile(source,dest);binaries[kind]=dest.resolve()
        report['binary_sha256'] = {kind:sha(path) for kind,path in binaries.items()}
        token_version = provenance(binaries['tokenizer'])
        version = json.loads(subprocess.run([str(binaries['template-json']),'--version'],capture_output=True,check=True,timeout=15).stdout)
        assert version == dict(token_version,renderer='native-jinja-json-float-roundtrip',
                               patches=token_version['patches']+';jinja-json-float-roundtrip')
        report['provenance'] = {'template-json':version,'tokenizer':token_version,
                                'template':provenance(binaries['template'],template=True)}
        manifest_path = args.bin_dir.parent/'minimax-m2-build-manifest.json'
        shutil.copyfile(manifest_path,args.out/'build-manifest.json')
        report['build_manifest_sha256'] = sha(args.out/'build-manifest.json')
        build = json.loads(manifest_path.read_text())
        assert build['template_json_oracle']['patch_sha256'] == sha(ROOT/'backends/minimax_m2/TemplateJSON.cmake')
        inventory = inspect_model(args.gguf)
        report['model_header_sha256'] = inventory['header_sha256']
        gguf = GGUFFile(args.gguf);source = gguf.metadata['tokenizer.chat_template']
        fixture = (ROOT/'serve/fixtures/minimax_m27_chat_template.jinja').read_bytes()
        assert source.encode('utf-8') == fixture and hashlib.sha256(fixture).hexdigest() == TEMPLATE_SHA256
        report['template_sha256'] = TEMPLATE_SHA256
        suite = unittest.defaultTestLoader.loadTestsFromNames([
            'serve.test_minimax_m2_history','serve.test_minimax_m2','tools.test_minimax_m2_tokenizer'])
        log = io.StringIO();unit = unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
        (args.out/'unit.log').write_text(log.getvalue(),encoding='utf-8')
        report['unit'] = {'pass':unit.wasSuccessful(),'methods':unit.testsRun,'failures':len(unit.failures),'errors':len(unit.errors)}
        cases = history_cases()
        # Additional renderer/tokenizer parity for schema and structured values.
        # The unit suite separately checks the exact argument values and IDs.
        tool = {'type':'function','function':{'name':'f','description':'Пример 中文','parameters':{
            'type':'object','properties':{'nested':{'type':'object'},'s':{'type':'string'}}},'strict':True}}
        call = {'role':'assistant','content':None,'reasoning_content':'R','tool_calls':[{
            'id':'n','function':{'name':'f','arguments':json.dumps({'nested':{'a':[1,2.5,True,None]},'s':'x\n<xml>ü'},ensure_ascii=False)}}]}
        for flag in [False,True]:
            cases.append({'name':'typed-schema/prompt='+str(flag),'request':{'messages':[USER,call,
                {'role':'tool','tool_call_id':'n','content':[{'type':'text','text':'结果'},'!']}],
                'tools':[tool],'add_generation_prompt':flag}})
        numbers = numeric_cases();cases.extend(numbers)
        report['numeric_prompts'] = len(numbers)
        contexts = [prepare_minimax_context(c['request']) for c in cases]
        raw_requests = [{'template':source,'context':context} for context in contexts]
        save(args.out/'requests.json',cases);save(args.out/'contexts.json',contexts)
        native = run_lines(binaries['template-json'],raw_requests)
        save(args.out/'native-rendered.json',native)
        assert len(native)==len(cases)
        token_requests = [{'text':r.get('rendered',''),'parse_special':True} for r in native]
        encoded = run_lines(binaries['tokenizer'],token_requests,args.gguf.resolve())
        save(args.out/'native-tokens.json',encoded)
        assert len(encoded)==len(cases)+1 and encoded[0]['ready'] and encoded[0]['vocab_size']==200064
        tokenizer = Tokenizer.from_gguf(args.gguf)
        template = MiniMaxChatTemplate(source)
        for case,context,actual,tokens in zip(cases,contexts,native,encoded[1:]):
            original = deepcopy(case['request'])
            rendered = template.render(case['request']);ids = tokenizer.encode(rendered,parse_special=True)
            checks = {'input_unchanged':case['request']==original,'normalization_idempotent':prepare_minimax_context(context)==context,
                      'native_render_equal':actual.get('rendered')==rendered,'token_ids_equal':tokens.get('ids')==ids,
                      'decoded_bytes_equal':tokens.get('decoded_hex')==rendered.encode().hex()==b''.join(tokenizer.token_bytes(i) for i in ids).hex(),
                      'one_bos':ids.count(200034)==1,'generation_prefix':rendered.endswith(']~b]ai\n<think>\n')==context['add_generation_prompt']}
            if 'expected' in case:
                checks['handwritten_prompt_equal'] = rendered==case['expected']
            row = {'name':case['name'],'pass':all(checks.values()),'checks':checks,
                   'tokens':len(ids),'render_sha256':hashlib.sha256(rendered.encode()).hexdigest()}
            if not row['pass']:row.update(python=rendered,context=context,native=actual)
            report['cases'].append(row)
        report['handwritten_prompts'] = sum('expected' in c for c in cases)
        report['native_prompts'] = len(cases)
        report['checks'] = sum(len(c['checks']) for c in report['cases'])
        report['prompt_tokens_compared'] = sum(c['tokens'] for c in report['cases'])
        legacy = template_corpus()
        legacy_requests = [{'template':source,'context':text_context(c['context'])} for c in legacy]
        old = run_lines(binaries['template'],legacy_requests)
        new = run_lines(binaries['template-json'],legacy_requests)
        assert len(old)==len(new)==len(legacy)
        py = renderer(source)
        report['legacy'] = [{'name':c['name'],'pass':before.get('rendered')==after.get('rendered')==py.render(**r['context'])}
                            for c,r,before,after in zip(legacy,legacy_requests,old,new)]
        save(args.out/'legacy-native.json',{'requests':legacy_requests,'raw':old,'json_float_fix':new})
        report['pass'] = report['unit']['pass'] and all(c['pass'] for c in report['cases']+report['legacy'])
        report['artifact_sha256'] = {name:sha(args.out/name) for name in ['requests.json','contexts.json','native-rendered.json','native-tokens.json','unit.log','legacy-native.json']}
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        save(args.out/'history-report.json',report)
    print(json.dumps({k:report[k] for k in ['pass','unit','handwritten_prompts','native_prompts','checks','prompt_tokens_compared']}))
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
