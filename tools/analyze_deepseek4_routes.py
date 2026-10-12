"""Evaluate bounded expert predictors before observing the next routed batch.

Only learned routing is scored. No weights are loaded and no graph is changed.
The trace records scheduler execution order, including speculative retries and
session restores; those events must not be mistaken for accepted token order.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path


def analyze(rows):
    histories = defaultdict(lambda: deque(maxlen=32))
    totals = defaultdict(Counter)
    layers = defaultdict(lambda: defaultdict(Counter))
    observed = skipped_hash = 0
    for row in rows:
        if row['hash']:
            skipped_hash += 1
            continue
        key = (row['owner'], row['matrix'])
        history = histories[key]
        current = row['ids']
        if not current:
            continue
        k = len(current[0])
        if history and not row['prefill']:
            # All candidate sets are formed before any current ID is observed.
            previous = list(history[-1])
            predictions = {'previous': previous[:k]}
            for window in (8, 32):
                counts = Counter(e for route in list(history)[-window:] for e in route)
                predictions[f'frequency_{window}'] = sorted(counts, key=lambda e: (-counts[e], e))[:k]
            first = set(current[0])
            union = set(e for route in current for e in route)
            for method, predicted in predictions.items():
                values = dict(batches=1, offered=len(predicted), useful=len(set(predicted) & union),
                              first_hits=len(set(predicted) & first), first_targets=len(first),
                              union_targets=len(union))
                totals[method].update(values)
                layers['/'.join(key)][method].update(values)
            observed += 1
        history.extend(current)

    def report(counts):
        c = dict(counts)
        c['precision'] = c['useful'] / c['offered'] if c['offered'] else 0
        c['waste_fraction'] = 1 - c['precision']
        c['first_recall'] = c['first_hits'] / c['first_targets'] if c['first_targets'] else 0
        c['union_recall'] = c['useful'] / c['union_targets'] if c['union_targets'] else 0
        return c

    per_layer = {layer: {m: report(v) for m, v in methods.items()} for layer, methods in layers.items()}
    # A screening threshold, not evidence of net speedup: GPU residency and
    # copy/compute overlap still need separate evaluation on a held-out trace.
    candidates = {layer: [m for m, v in methods.items()
                         if v['batches'] >= 100 and v['precision'] >= .9]
                  for layer, methods in per_layer.items()}
    return dict(learned_batches=observed, hash_rows_skipped=skipped_hash,
                methods={m: report(v) for m, v in totals.items()}, layers=per_layer,
                screening_candidates={k: v for k, v in candidates.items() if v},
                live_prefetch_enabled=False, weights_loaded=0,
                scope='scheduler order; speculative retries and restored sessions included')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.trace.open(encoding='utf-8') as source:
        result = analyze(json.loads(line) for line in source)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'layers'}, indent=2))


if __name__ == '__main__':
    main()
