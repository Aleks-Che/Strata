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


def compare_greedy_results(reference, results):
    names = ('session_a', 'session_b', 'session_a_restored', 'active_extension',
             'active_extension_restored', 'fresh_extension', 'single_token', 'after_cancel')
    return {name: {
        'same_input_ids': reference[name]['input_ids'] == results[name]['input_ids'],
        'same_output_ids': reference[name]['token_ids'] == results[name]['token_ids'],
    } for name in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--tokens', type=int, default=64, choices=range(8, 129))
    parser.add_argument('--baseline', action='store_true', help='Run the same state checks without DSpark')
    parser.add_argument('--allow-cpu-lane', action='store_true', help='Explicitly validate the optional hybrid lane; keep all state/output assertions')
    parser.add_argument('--warmup-requests', type=int, default=0, choices=range(9),
                        help='Fresh 16-token requests before state checks, e.g. to exercise runtime cost updates')
    parser.add_argument('--check-causality', action='store_true', help='Expensive diagnostic: vary future draft tokens, compare anchor logits')
    parser.add_argument('--check-serial', action='store_true', help='Expensive diagnostic: compare every verification row with the same tokens evaluated one at a time; timings are not benchmarks')
    parser.add_argument('--serial-rounds', type=int, default=3, choices=range(1, 65), help='Maximum serial-probe rounds per request')
    parser.add_argument('--greedy-reference', type=Path, help='Require identical input/output IDs in the eight greedy scenarios of a prior --baseline result')
    parser.add_argument('--output', type=Path, default=ROOT / 'bench/results/deepseek4-dspark-validation.json')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text('utf-8'))
    if cfg.get('architecture') != 'deepseek4' or (not args.baseline and '--draft-model' not in cfg['args']):
        parser.error('Requires a DeepSeek DSpark profile')
    serial_path = args.output.with_suffix('.serial.jsonl')
    if args.check_serial and (args.baseline or serial_path.exists()):
        parser.error('Serial probe requires DSpark and a fresh output .serial.jsonl path')
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
                    if Path(child.exe()).resolve() == Path(cfg['exe']).resolve():
                        child.kill()
                except psutil.Error:
                    pass
    timer = threading.Timer(480, timeout)
    timer.start()
    try:
        env = child_env(cfg)
        if args.check_causality:
            env['STRATA_SPEC_CHECK_CAUSAL'] = '1'
        if args.check_serial:
            env['STRATA_SPEC_SERIAL_REPORT'] = str(serial_path.resolve())
            env['STRATA_SPEC_CHECK_ROUNDS'] = str(args.serial_rounds)
        engine = StrataEngine(cfg['exe'], cfg['args'], cwd=cfg['cwd'], log=str(args.output.with_suffix('.log')), env=env)
        assert engine.info.get('gpu_only') == 1 or (args.allow_cpu_lane and engine.info.get('expert_compute') == 'hybrid'
                                                   and int(engine.info.get('expert_cpu_threads', 0)) > 0)
        assert engine.info.get('speculative') == ('none' if args.baseline else 'dspark')
        results['engine_info'] = engine.info
        def run(name, ids, count, session, cancel_after=None, sampling=None, cancel_seconds=None):
            event = threading.Event()
            output = []
            start = time.monotonic()
            def stop_during_prefill():
                event.set()
                # generate() can be waiting for a prompt-progress line. Send
                # STOP directly so this exercises cancellation inside a graph.
                engine.write_command('STOP\n')
            cancel_timer = threading.Timer(cancel_seconds, stop_during_prefill) if cancel_seconds else None
            if cancel_timer:
                cancel_timer.start()
            try:
                for token in engine.generate(ids, count, sampling or {'temperature': 0}, event, session_id=session):
                    if token is not None:
                        output.append(token)
                    if cancel_after and len(output) >= cancel_after:
                        event.set()
            finally:
                if cancel_timer:
                    cancel_timer.cancel()
                    cancel_timer.join()
            entry = {'input_ids': list(ids), 'token_ids': output, 'timings': dict(engine.last), 'elapsed_seconds': time.monotonic() - start}
            results[name] = entry
            print(name, json.dumps(entry['timings']), flush=True)
            assert 0 <= entry['timings'].get('drafts_accepted', 0) <= entry['timings'].get('drafts_offered', 0)
            assert len(output) <= count
            return output
        def prompt(text):
            return tok.encode(template.render([{'role': 'user', 'content': text}], enable_thinking=False), parse_special=True)
        ids = prompt(' '.join(f'Запись {i}: синий квадрат.' for i in range(32)) + '\nОбъясни подробно, как работает оперативная память компьютера.')
        assert len(ids) > 128  # exercise compressed-attention state, not just SWA
        for i in range(args.warmup_requests):
            run(f'warmup_{i}', ids, 16, f'validation-warmup-{i}')
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
        run('cancel_prefill', ids, 256, 'validation-cancel-prefill', cancel_seconds=0.05)
        assert results['cancel_prefill']['timings']['finish'] == 'cancel'
        assert not results['cancel_prefill']['token_ids'], 'Timed cancellation missed prefill'
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
        if args.check_serial:
            records = [json.loads(line) for line in serial_path.read_text('utf-8').splitlines()]
            assert records, 'No serial-probe records'
            assert all(not x['activation_observer'] for x in records), 'This validation requires final logits only'
            assert all(x['replay_max_abs'] == 0 for x in records), 'Serial probe changed replay logits'
            rows = [row for record in records for row in record['rows']]
            used = [row for record in records for row in record['rows'][:len(record['output_ids'])]]
            results['serial_probe'] = {
                'path': str(serial_path), 'packets': len(records), 'rows': len(rows),
                'max_abs': max(row['max_abs'] for row in rows),
                'changed_argmax': sum(row['argmax_a'] != row['argmax_b'] for row in rows),
                'used_rows': len(used),
                'used_changed_argmax': sum(row['argmax_a'] != row['argmax_b'] for row in used),
                'replay_max_abs': max(x['replay_max_abs'] for x in records),
            }
        if args.greedy_reference:
            reference = json.loads(args.greedy_reference.read_text('utf-8'))
            comparisons = compare_greedy_results(reference, results)
            results['greedy_reference'] = {'path': str(args.greedy_reference), 'comparisons': comparisons}
            assert all(x['same_input_ids'] and x['same_output_ids'] for x in comparisons.values()), 'Greedy reference input/output IDs differ'
        results['passed'] = True
    finally:
        timer.cancel()
        timer.join()
        if engine:
            engine.close()
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n', 'utf-8')


if __name__ == '__main__':
    main()
