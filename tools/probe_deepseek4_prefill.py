"""Compare fixed DeepSeek token histories on GPU; this is not a speed benchmark.

Input JSON is an array of token IDs. Split endpoints are exclusive offsets
before the final token, which is always evaluated separately in both paths.
"""
import argparse
import json
from pathlib import Path
import sys
import threading
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.server import StrataEngine, child_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--tokens-json', type=Path, required=True)
    parser.add_argument('--splits', default='')
    parser.add_argument('--nodes', default='@attention0,hc_init,attn_norm-0,qr-0,qr_norm-0,q-0,kv-0,attn_raw-0,l_last-0,result_output', help='Comma-separated tensor names or @weight:, @expert: (MUL_MAT_ID with inputs and IDs), @attention:, @attention-sinks: selectors')
    parser.add_argument('--history-row', type=int, default=-1, help='Read and compare only this token inside its segment and the full prefill, including visible attention inputs')
    parser.add_argument('--history-rows', type=int, default=1, help='Compare 1..128 tokens starting at history-row; the entire window must fit in one segment')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=480)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text('utf-8'))
    if cfg.get('architecture') != 'deepseek4' or '--draft-model' in cfg['args']:
        parser.error('Requires a DeepSeek profile without DSpark')
    tokens = json.loads(args.tokens_json.read_text('utf-8'))
    if not isinstance(tokens, list) or not 2 <= len(tokens) <= 4096 or any(type(x) is not int or x < 0 for x in tokens):
        parser.error('tokens-json must contain 2..4096 nonnegative integer IDs')
    if not -1 <= args.history_row < len(tokens) - 1:
        parser.error('history-row must be -1 or a position inside the prefix')
    if not 1 <= args.history_rows <= 128 or (args.history_row < 0 and args.history_rows != 1):
        parser.error('history-rows must be 1..128 and requires history-row for a window')
    try:
        ends = [int(x) for x in args.splits.split(',')] if args.splits else []
    except ValueError:
        parser.error('splits must contain integer endpoints')
    if any(b <= a or b > len(tokens) - 1 for a, b in zip([0] + ends, ends)):
        parser.error('splits must increase within prefix')
    if not ends or ends[-1] < len(tokens) - 1:
        ends.append(len(tokens) - 1)
    if args.history_row >= 0 and args.history_row + args.history_rows > next(x for x in ends if x > args.history_row):
        parser.error('history window crosses a split or the end of the prefix')
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace = args.output.with_suffix('.jsonl')
    if args.output.exists() or trace.exists():
        parser.error('Choose a fresh output path so probe records cannot be mixed')
    env = child_env(cfg)
    env.update(STRATA_PREFILL_PROBE=str(args.output), STRATA_PREFILL_PROBE_SPLITS=args.splits,
               STRATA_SPEC_TRACE_NODES=str(trace), STRATA_SPEC_TRACE_NODE=args.nodes)
    env['STRATA_PREFILL_PROBE_HISTORY_ROW'] = str(args.history_row)
    env['STRATA_PREFILL_PROBE_HISTORY_ROWS'] = str(args.history_rows)
    engine = None
    expired = threading.Event()

    def timeout():
        expired.set()
        if engine:
            engine.proc.kill()
        else:
            expected = Path(cfg['exe']).name.lower()
            for p in psutil.Process().children(recursive=True):
                try:
                    if p.name().lower() == expected:
                        p.kill()
                except psutil.Error:
                    pass
    timer = threading.Timer(args.timeout, timeout)
    timer.start()
    try:
        engine = StrataEngine(cfg['exe'], cfg['args'], cwd=cfg['cwd'],
                              log=str(args.output.with_suffix('.log')), env=env)
        assert engine.info.get('gpu_only') == 1 and engine.info.get('speculative') == 'none'
        output = [x for x in engine.generate(tokens, 1, {'temperature': 0}, threading.Event(), session_id='prefill-probe') if x is not None]
        assert not expired.is_set(), 'Probe timed out'
        result = json.loads(args.output.read_text('utf-8'))
        result.update(config=cfg, engine_info=engine.info, normal_generation=output, timings=dict(engine.last))
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', 'utf-8')
        print(json.dumps({k: result[k] for k in ['full', 'segmented', 'control_difference', 'restored_difference', 'normal_generation']}, indent=2))
    finally:
        timer.cancel()
        timer.join()
        if engine:
            engine.close()


if __name__ == '__main__':
    main()
