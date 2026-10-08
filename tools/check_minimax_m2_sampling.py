"""MiniMax sampling completion screen, seeded repeat and raw-logit replay.

Three seeds on the previously repeating Russian prompt, one lower temperature,
English/Chinese checks, and a repeat after another request has advanced its RNG.
Quality gates are narrow corpus checks, not an independent model oracle.
"""
import argparse
import collections
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_minimax_m2_completion import CONFIG, compare, rows
from tools.gguf_reader import GGUFFile
from tools.minimax_m2_loader_contract import validate_loader_contract
from tools.minimax_m2_template import renderer, text_context
from tools.run_minimax_m2 import PRECISION_ENV, runtime_environment
from tools.tune_minimax_m2_pipeline import audit, save, sha


def quality(result, language, expected, required=True):
    text = result['text']
    parts = text.split('</think>', 1)
    final = parts[1].removesuffix('[e~[').strip() if len(parts) == 2 else ''
    words = re.findall(r'\w+', text.lower())
    repetitions = max(collections.Counter(tuple(words[i:i+12]) for i in range(max(0, len(words)-11))).values(), default=0)
    eos = result['stop_reason'] == 'eos' and result['stop_token_id'] == 200020 and result['token_ids'].count(200020) == 1 and result['token_ids'][-1] == 200020
    checks = {'natural_eos': eos, 'reasoning_closed': len(parts) == 2, 'final_present': bool(final),
              'no_repeated_12_word_loop': repetitions <= 8}
    if language == 'russian':
        checks['answer'] = expected in final
        checks['language'] = len(re.findall('[А-Яа-яЁё]', final)) >= 20
    elif language == 'chinese':
        checks['answer'] = expected in final or '三百九十一' in final
        checks['language'] = len(re.findall('[\u4e00-\u9fff]', final)) >= 4
    else:
        checks['answer'] = 'scatter' in final.lower() or 'rayleigh' in final.lower()
        checks['language'] = len(re.findall('[a-zA-Z]', final)) >= 30
    return {'pass': all(checks.values()) if required else True, 'required': required, 'checks': checks,
            'final': final, 'max_12_word_repetitions': repetitions}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tokens', type=int, default=1536)
    p.add_argument('--engine', type=Path, default=ROOT/'build-local/minimax-m2-cuda/bin/strata-minimax-m2-bench.exe')
    p.add_argument('--replay-engine', type=Path, default=ROOT/'build-local/minimax-m2-cuda/bin/strata-minimax-m2-sampling-check.exe')
    p.add_argument('--cuda-root', type=Path, default=ROOT/'build-local/cuda-13.0')
    args = p.parse_args()
    if not 257 <= args.tokens <= 1705:
        p.error('tokens must be 257..1705 for the 2048-token context')
    g = GGUFFile(args.model)
    contract = validate_loader_contract(g.metadata, g.tensors)
    template = renderer(g.metadata['tokenizer.chat_template'])
    russian = 'Прочитай список и объясни на русском, как найти его сумму: '+', '.join(map(str, range(1, 97)))
    english = 'Explain in a few sentences why the sky appears blue during the day.'
    chinese = '请用中文简要说明 17 × 23 的计算过程，并给出结果。'
    def case(name, text, seed=42, temperature=1., language='russian', expected='4656', tokens=None):
        return {'name': name, 'language': language, 'expected': expected, 'complete': tokens is None,
                'request': {'prompt': template.render(**text_context({'messages': [{'role': 'user', 'content': text}], 'add_generation_prompt': True})),
                    'max_tokens': args.tokens if tokens is None else tokens,
                    'sampling': {'temperature': temperature, 'top_p': .95, 'top_k': 40, 'seed': seed}}}
    groups = [[case('russian-42', russian)], [case('russian-7', russian, seed=7)], [case('russian-1234', russian, seed=1234)],
              [case('rng-prefix-7', russian, seed=7, tokens=32), case('russian-42-repeat', russian)],
              [case('russian-temp07', russian, temperature=.7)],
              [case('english-42', english, language='english', expected='scattering')],
              [case('chinese-42', chinese, language='chinese', expected='391')]]
    args.out.mkdir(parents=True, exist_ok=False)
    exe, replay_exe = args.out/'engine.exe', args.out/'sampling-check.exe'
    shutil.copyfile(args.engine, exe);shutil.copyfile(args.replay_engine, replay_exe)
    shutil.copytree(ROOT/'backends/minimax_m2', args.out/'sources')
    shutil.copyfile(Path(__file__), args.out/'sources'/Path(__file__).name)
    for rel in ['tools/check_minimax_m2_completion.py', 'tools/tune_minimax_m2_pipeline.py', 'tools/minimax_m2_template.py', 'tools/run_minimax_m2.py']:
        shutil.copyfile(ROOT/rel, args.out/'sources'/Path(rel).name)
    with args.model.open('rb') as f:
        header_sha = __import__('hashlib').sha256(f.read(g.header_end)).hexdigest()
    env = runtime_environment(args.cuda_root);env['STRATA_MM27_TOKENWISE'] = '0'
    report = {'pass': False, 'scope': 'narrow completion/answer gates; fresh process per group; sequential GPU runs; OS cache and external load uncontrolled',
              'model': str(args.model), 'model_header_sha256': header_sha, 'source_revision': contract['source_revision'],
              'engine_sha256': sha(exe), 'replay_engine_sha256': sha(replay_exe), 'driver_sha256': sha(Path(__file__)),
              'precision_environment': dict(PRECISION_ENV, STRATA_MM27_TOKENWISE='0'), 'groups': groups,
              'runs': [], 'cases': [], 'comparisons': []}
    locations = {}
    try:
        for index, group in enumerate(groups):
            base = args.out/f'{index+1:02d}-{group[-1]["name"]}'
            request_file = base.with_suffix('.request.json')
            save(request_file, [c['request'] for c in group])
            command = [str(exe.resolve()), '--gguf', str(args.model.resolve()), '--request', str(request_file),
                       '--output', str(base.with_suffix('.json')), '--logits', str(base.with_suffix('.f32')),
                       '--ctx', '2048', '--batch', '16', '--gpu-cache-mib', '18432', '--gpu-cache-allocator', 'arena',
                       '--pipeline-readers', '2', '--pipeline-chunk-mib', '4', '--pipeline-lookahead', '1', '--pipeline-d2d-batch', '1']
            run_report = {'name': base.name, 'command': command}
            report['runs'].append(run_report);save(args.out/'sampling-report.json', report)
            print(base.name+': running', flush=True)
            with base.with_suffix('.stdout.log').open('w', encoding='utf-8') as stdout, base.with_suffix('.stderr.log').open('w', encoding='utf-8') as stderr:
                run = subprocess.run(command, env=env, stdout=stdout, stderr=stderr, timeout=1800)
            run_report['exit_code'] = run.returncode
            if run.returncode:
                raise RuntimeError(base.name+': engine failed; retained diagnostics')
            data = json.loads(base.with_suffix('.json').read_text(encoding='utf-8'))
            results = data['results'];assert len(results) == len(group)
            run_report['gates'] = audit(data, CONFIG, args.tokens)
            run_report['gates']['requests'] = all(1 <= r['generated_tokens'] <= c['request']['max_tokens'] and
                 r['generated_tokens'] == len(r['token_ids']) and (r['stop_reason'] == 'eos' or r['generated_tokens'] == c['request']['max_tokens'])
                 for r,c in zip(results,group))
            values = rows(base.with_suffix('.f32'), results)
            offset = 0
            for c, result in zip(group, results):
                n = result['generated_tokens'];finite = all(np.isfinite(values[j:min(j+16,offset+n)]).all() for j in range(offset,offset+n,16))
                sampling_ok = all(math.isclose(result['sampling'][k],v,rel_tol=1e-7,abs_tol=1e-8) for k,v in c['request']['sampling'].items())
                sampling_ok &= result['sampling_algorithm'] == 'temperature->top_k->top_p->dist'
                item = {'name': c['name'], 'run': base.name, 'report': str(base.with_suffix('.json')), 'logits': str(base.with_suffix('.f32')),
                        'offset': offset, 'sampling': result['sampling'], 'sampling_ok': bool(sampling_ok), 'finite_logits': bool(finite),
                        'generated_tokens': n, 'prompt_tokens': result['prompt_tokens'], 'stop_reason': result['stop_reason'],
                        'decode_tok_s': result['decode_tokens_per_second'], 'sampling_ms': result['sampling_ms'],
                        'sampling_ms_per_token': result['sampling_ms']/n, 'request_ms': result['request_ms'],
                        'quality': quality(result,c['language'],c['expected'],c['complete'])}
                report['cases'].append(item);locations[c['name']] = item
                reference_name = {'russian-42-repeat': 'russian-42', 'rng-prefix-7': 'russian-7'}.get(c['name'])
                if reference_name:
                    prior = locations[reference_name]
                    prior_data = json.loads(Path(prior['report']).read_text(encoding='utf-8'))['results']
                    old = rows(Path(prior['logits']), prior_data)
                    comparison = compare(values[offset:offset+n], old[prior['offset']:prior['offset']+n])
                    reference_ids = prior_data[0]['token_ids'][:n]
                    comparison['token_ids_equal'] = result['token_ids'] == reference_ids
                    comparison['pass'] &= comparison['token_ids_equal'] and (c['name'] != 'russian-42-repeat' or n == prior['generated_tokens'])
                    comparison.update({'actual': c['name'], 'reference': reference_name});report['comparisons'].append(comparison)
                    del old
                offset += n
                print(json.dumps({k:item[k] for k in ['name','generated_tokens','stop_reason','sampling_ms_per_token','quality']},ensure_ascii=False),flush=True)
            del values
            replay_dir = args.out/(base.name+'-replay')
            replay_cmd = [str(replay_exe.resolve()), '--replay', str(base.with_suffix('.json')), str(base.with_suffix('.f32')), str(replay_dir)]
            with base.with_suffix('.replay.log').open('w', encoding='utf-8') as log:
                replay = subprocess.run(replay_cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
            run_report['replay'] = json.loads((replay_dir/'sampling-report.json').read_text(encoding='utf-8'))
            run_report['replay_command'] = replay_cmd
            run_report['report_sha256'] = sha(base.with_suffix('.json'));run_report['logits_sha256'] = sha(base.with_suffix('.f32'))
            run_report['pass'] = all(run_report['gates'].values()) and replay.returncode == 0 and run_report['replay']['pass']
            save(args.out/'sampling-report.json', report)
            if not run_report['pass']:
                raise RuntimeError('functional/replay gate failed: '+base.name)
        report['functional_pass'] = all(r['pass'] for r in report['runs']) and all(c['finite_logits'] and c['sampling_ok'] for c in report['cases']) and all(c['pass'] for c in report['comparisons'])
        report['completion_pass'] = all(c['quality']['pass'] for c in report['cases'])
        report['pass'] = report['functional_pass'] and report['completion_pass']
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        save(args.out/'sampling-report.json', report)
    print(json.dumps({k:report[k] for k in ['pass','functional_pass','completion_pass']}),flush=True)
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
