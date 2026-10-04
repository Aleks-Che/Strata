"""Train expert placement hints, then compare fresh MTP-off engines.

Measures startup, first token, complete request, and warmed decode separately.
The learned file is frozen for alternating restart trials and an unseen prompt.
Never edits the working launch profile. Requires an unused output/warm-file path.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.server import StrataEngine, child_env, engine_args
from serve.glm5next import GLMTemplate
from tools.strata_tokenizer import Tokenizer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile', type=Path, required=True)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--warm-file', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--cold-repeats', type=int, default=3)
    a = ap.parse_args()
    if a.cold_repeats < 2 or a.output.exists() or a.warm_file.exists():
        ap.error('use at least two restart pairs and unused output/warm-file paths')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(a.profile.read_text(encoding='utf-8'))
    reference = json.loads(a.reference.read_text(encoding='utf-8'))
    assert cfg['args'][cfg['args'].index('--mtp') + 1] == '0', 'MTP must be off'
    report = {'status': 'running', 'configuration': cfg, 'binary_sha256': digest(Path(cfg['exe'])),
              'method': 'Train on saved prompt A with nine requests; freeze hints. Alternate three fresh-engine pairs, reversing order each pair. Pair 0 also has four warmups plus five timed repeats. Unseen prompt B is not used in training. OS file cache is not cleared. Placement metadata contains no prompt text or state.',
              'cases': []}

    def save():
        a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    def run(name, ids, expected, count, mode, prompt_name):
        variant = deepcopy(cfg)
        variant.setdefault('env', {})['STRATA_GLM_EXPERT_PROFILE'] = str(a.warm_file.resolve()) if mode != 'off' else ''
        variant['env']['STRATA_GLM_EXPERT_PROFILE_READ_ONLY'] = '0' if mode == 'train' else '1'
        variant['log'] = str(a.output.with_name(a.output.stem + '-' + name + '.engine.log').resolve())
        assert not Path(variant['log']).exists(), 'use a fresh log basename'
        case = {'name': name, 'mode': mode, 'prompt': prompt_name, 'prompt_ids': ids,
                'configuration': variant, 'runs': [], 'status': 'running'}
        report['cases'].append(case)
        save()
        engine = None
        print('START ' + name, flush=True)
        try:
            started = time.perf_counter()
            engine = StrataEngine(variant['exe'], engine_args(variant), cwd=variant.get('cwd'),
                                  env=child_env(variant), log=variant['log'])
            case['startup_seconds'] = time.perf_counter() - started
            case['info'] = engine.info
            assert engine.info['spec'] == 0 and engine.info['gpu_only'] == 1
            assert (engine.info['expert_warm_entries'] > 0) == (mode == 'learned')
            for i in range(count):
                begin = time.perf_counter()
                tokens, first = [], None
                for token in engine.generate(ids, 64, {'temperature': 0}, threading.Event()):
                    if token is not None:
                        if first is None:
                            first = time.perf_counter() - begin
                        tokens.append(token)
                wall = time.perf_counter() - begin
                done = dict(engine.last)
                assert len(tokens) == 64 and done['finish'] == 'length', done
                if expected is None:
                    expected = tokens
                assert tokens == expected, (name, i, 'reference IDs differ')
                result = {'tokens': tokens, 'equals_reference': True, 'done': done,
                          'first_token_seconds': first, 'request_wall_seconds': wall,
                          'tokens_per_second': 63 * 1000 / done['decode_ms']}
                case['runs'].append(result)
                save()
                print(f'{name} #{i}: first={first:.3f}s, total={wall:.3f}s, decode={result["tokens_per_second"]:.3f} tok/s', flush=True)
            case['status'] = 'pass'
        finally:
            if engine:
                process = engine.proc
                engine.close()
                case['exit_code'] = process.returncode
                assert process.returncode == 0
                lines = Path(variant['log']).read_text(encoding='utf-8', errors='replace').splitlines()
                case['memory_snapshots'] = [json.loads(line.split('STRATA_GLM_MEMORY ', 1)[1])
                                            for line in lines if 'STRATA_GLM_MEMORY ' in line]
            save()
        return expected

    run('train', reference['prompt_ids'], reference['generated_ids'], 9, 'train', 'A')
    frozen_hash = digest(a.warm_file)
    report['learned_file'] = {'path': str(a.warm_file.resolve()), 'sha256': frozen_hash,
                              'entries': len(json.loads(a.warm_file.read_text())['entries'])}
    for repeat in range(a.cold_repeats):
        for mode in (('off', 'learned') if repeat % 2 == 0 else ('learned', 'off')):
            run(f'A-{repeat}-{mode}', reference['prompt_ids'], reference['generated_ids'],
                9 if repeat == 0 else 1, mode, 'A')
            assert digest(a.warm_file) == frozen_hash
    prompt = 'Напиши на Python стабильную сортировку слиянием. Объясни сложность и почему равные элементы сохраняют порядок. Приведи пример с кортежами.'
    tokenizer = Tokenizer.from_gguf(cfg['args'][1])
    template = GLMTemplate(Path(cfg['tokenizer']) / 'chat_template.jinja')
    rendered = template.render([{'role': 'user', 'content': prompt}], reasoning_effort='low')
    ids = tokenizer.encode(rendered, parse_special=True)
    report['unseen_prompt'] = prompt
    expected = run('B-off', ids, None, 3, 'off', 'B')
    run('B-learned', ids, expected, 3, 'learned', 'B')
    assert digest(a.warm_file) == frozen_hash
    report['cold_summary'] = {}
    for mode in ('off', 'learned'):
        cases = [c for c in report['cases'] if c['mode'] == mode and c['prompt'] == 'A']
        report['cold_summary'][mode] = {
            'median_startup_seconds': statistics.median(c['startup_seconds'] for c in cases),
            'median_first_token_seconds': statistics.median(c['runs'][0]['first_token_seconds'] for c in cases),
            'median_launch_to_first_token_seconds': statistics.median(c['startup_seconds'] + c['runs'][0]['first_token_seconds'] for c in cases),
            'median_first_request_seconds': statistics.median(c['runs'][0]['request_wall_seconds'] for c in cases),
            'warm_decode_tokens_per_second': statistics.median(r['tokens_per_second'] for r in cases[0]['runs'][4:]),
        }
    report['status'] = 'pass'
    save()


if __name__ == '__main__':
    main()
