"""Bounded real-model DSpark session/rollback/cancellation checks on GPU.

python tools/validate_deepseek4_speculative.py --config strata-deepseek4-ud-q3-k-xl-dspark.json
"""
import argparse
import json
from pathlib import Path
import sys
import threading
import time
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.server import StrataEngine, child_env
from serve.deepseek import DeepSeekTemplate
from tools.strata_tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--tokens', type=int, default=64, choices=range(8, 129))
    parser.add_argument('--baseline', action='store_true', help='Run the same state checks without DSpark')
    parser.add_argument('--check-causality', action='store_true', help='Expensive diagnostic: vary future draft tokens, compare anchor logits')
    parser.add_argument('--output', type=Path, default=ROOT / 'bench/results/deepseek4-dspark-validation.json')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text('utf-8'))
    if cfg.get('architecture') != 'deepseek4' or (not args.baseline and '--draft-model' not in cfg['args']):
        parser.error('Requires a DeepSeek DSpark profile')
    tok = Tokenizer.from_gguf(cfg['args'][cfg['args'].index('--native') + 1])
    template = DeepSeekTemplate(Path(cfg['tokenizer']) / 'chat_template.jinja')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    engine = None
    # A process-level deadline also covers a stuck load or cancelled CUDA call.
    def timeout():
        results['timeout'] = True
        if engine:
            engine.proc.kill()
        else:
            for child in psutil.Process().children(recursive=True):
                try:
                    if child.name() in ('strata-deepseek4.exe', 'strata-deepseek4'):
                        child.kill()
                except psutil.Error:
                    pass
    timer = threading.Timer(480, timeout)
    timer.start()
    try:
        env = child_env(cfg)
        if args.check_causality:
            env['STRATA_SPEC_CHECK_CAUSAL'] = '1'
        engine = StrataEngine(cfg['exe'], cfg['args'], cwd=cfg['cwd'], log=str(args.output.with_suffix('.log')), env=env)
        assert engine.info.get('gpu_only') == 1
        assert engine.info.get('speculative') == ('none' if args.baseline else 'dspark')
        results['engine_info'] = engine.info
        def run(name, ids, count, session, cancel_after=None, sampling=None):
            event = threading.Event()
            output = []
            start = time.monotonic()
            for token in engine.generate(ids, count, sampling or {'temperature': 0}, event, session_id=session):
                if token is not None:
                    output.append(token)
                if cancel_after and len(output) >= cancel_after:
                    event.set()
            entry = {'token_ids': output, 'timings': dict(engine.last), 'elapsed_seconds': time.monotonic() - start}
            results[name] = entry
            print(name, json.dumps(entry['timings']), flush=True)
            assert 0 <= entry['timings'].get('drafts_accepted', 0) <= entry['timings'].get('drafts_offered', 0)
            assert len(output) <= count
            return output
        def prompt(text):
            return tok.encode(template.render([{'role': 'user', 'content': text}], enable_thinking=False), parse_special=True)
        ids = prompt(' '.join(f'Запись {i}: синий квадрат.' for i in range(32)) + '\nОбъясни подробно, как работает оперативная память компьютера.')
        assert len(ids) > 128  # exercise compressed-attention state, not just SWA
        a = run('session_a', ids, args.tokens, 'validation-a')
        run('session_b', prompt('Напиши несколько предложений про океан.'), 16, 'validation-b')
        b = run('session_a_restored', ids, args.tokens, 'validation-a')
        assert a == b, 'Target/draft session restore changed greedy output'
        assert results['session_a_restored']['timings']['reused'] == len(ids) - 1
        assert results['session_a_restored']['timings']['cache_source'] == 'ram'
        continuation = ids + a + tok.encode('\nПродолжай:')
        c = run('active_extension', continuation, 16, 'validation-a')
        assert results['active_extension']['timings']['cache_source'] == 'active'
        restored_extension = run('active_extension_restored', continuation, 16, 'validation-a')
        assert c == restored_extension, 'Restoring the continued session changed output'
        assert results['active_extension_restored']['timings']['cache_source'] == 'ram'
        d = run('fresh_extension', continuation, 16, 'validation-fresh')
        # Batch-size-dependent quantized-target numerics are tracked separately
        # from exact state round trips. Keep this result visible, never claim
        # lossless parity with a full prefill or non-speculative decode.
        results['full_prefill_identical'] = c == d
        print('full_prefill_identical', c == d, flush=True)
        run('single_token', ids, 1, 'validation-a')
        assert results['single_token']['timings']['drafts_offered'] == 0
        run('cancel', ids, 256, 'validation-cancel', cancel_after=4)
        assert results['cancel']['timings']['finish'] == 'cancel'
        e = run('after_cancel', ids, args.tokens, 'validation-a')
        assert a == e, 'Cancellation contaminated the restored session'
        sampling = {'temperature': 0.6, 'seed': 123, 'penalty_repeat': 1.05}
        f = run('seeded', ids, 24, 'validation-seeded', sampling=sampling)
        g = run('seeded_repeat', ids, 24, 'validation-seeded', sampling=sampling)
        assert f == g, 'Seeded sampling/history changed after cache restore'
        if args.check_causality:
            import re
            log = args.output.with_suffix('.log').read_text('utf-8', errors='replace')
            diffs = [float(value) for value in re.findall(r'STRATA_SPEC_CAUSAL .*?max_logit_diff=([\d.e+-]+)', log)]
            results['causality_max_logit_diffs'] = diffs
            assert diffs and max(diffs) <= 1e-5, 'Future draft tokens affected anchor logits'
        results['passed'] = True
    finally:
        timer.cancel()
        if engine:
            engine.close()
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n', 'utf-8')


if __name__ == '__main__':
    main()
