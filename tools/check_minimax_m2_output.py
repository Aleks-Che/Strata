"""Replay retained MiniMax completions through the unregistered text parser.

Checks the real token bytes against native text, then token-sized and single
byte UTF-8 delivery. EOS is consumed by ID, never by removing literal text.
"""
import argparse
import codecs
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.minimax_m2 import MiniMaxReasoningParser
from tools.strata_tokenizer import Tokenizer
from tools.tune_minimax_m2_pipeline import save, sha


def replay(chunks):
    decoder = codecs.getincrementaldecoder('utf-8')('strict')
    parser = MiniMaxReasoningParser()
    events = []
    for chunk in chunks:
        events.extend(parser.feed(decoder.decode(chunk)))
    events.extend(parser.feed(decoder.decode(b'', final=True)))
    events.extend(parser.finish())
    return {'reasoning': ''.join(e.text for e in events if e.kind == 'reasoning'),
            'content': ''.join(e.text for e in events if e.kind == 'content'),
            'reasoning_complete': parser.reasoning_complete,
            'event_kinds': sorted(set(e.kind for e in events))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True, help='native generation.json with results')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    native = json.loads(args.report.read_text(encoding='utf-8'))
    tokenizer = Tokenizer.from_gguf(native['model'])
    args.out.mkdir(parents=True, exist_ok=False)
    report = {'pass': False, 'reference': str(args.report), 'reference_sha256': sha(args.report),
              'source_sha256': {rel: sha(ROOT/rel) for rel in ['serve/minimax_m2.py', 'tools/check_minimax_m2_output.py',
                'serve/frontend.py', 'tools/strata_tokenizer.py']}, 'cases': []}
    for i, result in enumerate(native['results']):
        ids = result['token_ids']
        raw = [tokenizer.token_bytes(t) for t in ids]
        eos = result['stop_reason'] == 'eos'
        terminal = ids[-1] == 200020 and ids.count(200020) == 1 if eos else 200020 not in ids
        body = raw[:-1] if eos else raw
        text = b''.join(body).decode('utf-8', errors='strict')
        expected = text.split('</think>', 1)
        pieces = replay(body)
        one_byte = replay(bytes([b]) for chunk in body for b in chunk)
        checks = {'native_token_bytes_equal': b''.join(raw).decode('utf-8', errors='strict') == result['text'],
                  'terminal_id': terminal,
                  'completion_state': pieces['reasoning_complete'] == (len(expected) == 2) and
                                      (not eos or (len(expected) == 2 and bool(expected[1].strip()))),
                  'exact_segments': pieces['reasoning'] == expected[0] and pieces['content'] == (expected[1] if len(expected) == 2 else ''),
                  'byte_stream_equal': pieces == one_byte,
                  'no_executable_calls': set(pieces['event_kinds']) <= {'content', 'reasoning'}}
        report['cases'].append({'request': i, 'pass': all(checks.values()), 'checks': checks,
                                'generated_tokens': len(ids), **pieces})
    report['checks'] = sum(len(c['checks']) for c in report['cases'])
    report['pass'] = bool(report['cases']) and all(c['pass'] for c in report['cases'])
    save(args.out/'output-report.json', report)
    print(json.dumps({'pass': report['pass'], 'requests': len(report['cases']), 'checks': report['checks']}))
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
