"""Inspect local MiniMax DFlash sidecars; this does not enable speculative inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.gguf_reader import GGUFFile
from tools.inspect_hy3_gguf import file_hash, protect_output, validate_ranges
from tools.inspect_minimax_m2_gguf import inspect_model


def array_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()


def inspect(target_path, draft_paths, full_hash=False):
    target_path = Path(target_path).resolve()
    target = GGUFFile(target_path)
    target_report = inspect_model(target_path)
    result = {'status': 'pass', 'scope': 'static header/range/tokenizer inspection; no runtime, acceptance or speed validation',
              'target': str(target_path), 'target_file_bytes': target_path.stat().st_size,
              'target_header_sha256': target_report['header_sha256'],
              'target_full_sha256': None, 'target_loader_contract_pass': True, 'drafts': []}
    for path in draft_paths:
        path = Path(path).resolve()
        draft = GGUFFile(path)
        meta = draft.metadata
        if draft.alignment != 32 or meta.get('general.architecture') != 'dflash':
            raise ValueError(f'Expected aligned DFlash GGUF: {path}')
        rows = validate_ranges(draft.tensors, draft.data_start, path.stat().st_size, draft.alignment)
        names = {t.name for t in draft.tensors}
        target_meta = target.metadata
        tokens = meta.get('tokenizer.ggml.tokens', [])
        target_tokens = target_meta['tokenizer.ggml.tokens']
        types = meta.get('tokenizer.ggml.token_type', [])
        target_types = target_meta['tokenizer.ggml.token_type']
        layers = meta.get('dflash.target_layers', [])
        hidden = target_meta['minimax-m2.embedding_length']
        checks = {
            'target_hidden_width': meta.get('dflash.embedding_length') == hidden,
            'feature_layers_in_target': bool(layers) and len(set(layers)) == len(layers) and
                all(type(i) is int and 0 <= i < target_meta['minimax-m2.block_count'] for i in layers),
            'feature_projection_shape': any(t.name == 'fc.weight' and t.shape == [len(layers)*hidden, hidden] for t in draft.tensors),
            'tokenizer_family': all(meta.get('tokenizer.ggml.'+k) == target_meta.get('tokenizer.ggml.'+k) for k in ('model', 'pre')),
            'vocabulary_prefix': bool(tokens) and len(tokens) <= len(target_tokens) and tokens == target_tokens[:len(tokens)],
            'token_types_prefix': len(types) == len(tokens) and types == target_types[:len(types)],
            'merges_equal': meta.get('tokenizer.ggml.merges') == target_meta.get('tokenizer.ggml.merges'),
            'bos_eos_unk_equal': all(meta.get('tokenizer.ggml.'+k+'_token_id') == target_meta.get('tokenizer.ggml.'+k+'_token_id')
                                     for k in ('bos', 'eos', 'unknown')),
        }
        mask = meta.get('tokenizer.ggml.mask_token_id')
        checks['mask_in_shared_prefix'] = type(mask) is int and 0 <= mask < len(tokens)
        if not all(checks.values()):
            raise ValueError(f'Static mismatch in {path.name}: {[k for k, v in checks.items() if not v]}')
        row = {'path': str(path), 'file_bytes': path.stat().st_size,
               'header_sha256': file_hash(path, draft.header_end),
               'full_sha256': file_hash(path, path.stat().st_size) if full_hash else None,
               'gguf_version': draft.version, 'alignment': draft.alignment,
               'header_end': draft.header_end, 'data_start': draft.data_start,
               'metadata': {k: v for k, v in meta.items() if not k.startswith('tokenizer.')},
               'tokenizer': {k: {'count': len(v), 'json_sha256': array_hash(v)} if isinstance(v, list) else v
                             for k, v in meta.items() if k.startswith('tokenizer.')},
               'tensor_count': len(rows), 'types': dict(Counter(t.type_name for t in draft.tensors)),
               'payload_bytes': sum(r['bytes'] for r in rows), 'tensors': rows, 'static_checks': checks,
               'target_vocab_size': len(target_tokens), 'draft_vocab_size': len(tokens),
               'vocabulary_equal': tokens == target_tokens,
               'target_only_tokens': [{'id': i, 'token': target_tokens[i], 'type': target_types[i]}
                                      for i in range(len(tokens), len(target_tokens))],
               'mask': {'id': mask, 'token': tokens[mask], 'target_token': target_tokens[mask]},
               'own_embedding': 'token_embd.weight' in names, 'own_output_head': 'output.weight' in names,
               'own_mask_embedding': 'mask_embd.weight' in names,
               'head_vocab_adapter_required': len(tokens) != len(target_tokens) and 'output.weight' not in names,
               'runtime_compatible': None, 'acceptance_measured': False, 'speed_measured': False}
        result['drafts'].append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--draft', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--full-hash', action='store_true', help='Hash draft payloads; target payload is not read')
    args = parser.parse_args()
    output = protect_output(args.output, args.model, *args.draft)
    if output.exists() or output.suffix != '.json':
        raise ValueError('Use a fresh .json report')
    report = inspect(args.model, args.draft, args.full_hash)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f'PASS: {len(report["drafts"])} sidecars, static inspection only; {output}')


if __name__ == '__main__':
    main()
