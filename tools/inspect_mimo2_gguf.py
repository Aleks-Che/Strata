"""Inspect and admit the reviewed MiMo text GGUF without reading weight payloads."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.gguf_reader import GGUFFile
from tools.mimo2_loader_contract import validate_loader_contract
# These helpers only validate generic GGUF ranges/files, not Hy3 architecture.
from tools.inspect_hy3_gguf import file_hash, protect_output, validate_ranges


def inspect_model(path, *, allow_fixture=False):
    path = Path(path).resolve()
    g = GGUFFile(path)
    details = validate_ranges(g.tensors, g.data_start, path.stat().st_size, g.metadata.get('general.alignment', 32))
    contract = validate_loader_contract(g.metadata, g.tensors, allow_fixture=allow_fixture)
    mapping = {r['name']: r for r in contract['tensor_mapping']}
    groups = Counter()
    for r in details:
        r.update({k: mapping[r['name']][k] for k in ('scope', 'family', 'layer')})
        groups['main_routed' if r['family'] == 'routed' else 'main_other'] += r['bytes']
    metadata = {}
    for k, v in g.metadata.items():
        if k == 'tokenizer.chat_template':
            metadata[k] = {'utf8_bytes': len(v.encode('utf-8')), 'sha256': hashlib.sha256(v.encode('utf-8')).hexdigest()}
        elif k.startswith('tokenizer.') and isinstance(v, list):
            metadata[k] = {'count': len(v), 'json_sha256': hashlib.sha256(json.dumps(v, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()}
        else:
            metadata[k] = v
    return {'schema_version': 1, 'status': 'pass', 'fixture': allow_fixture, 'model': str(path),
            'scope': 'header/range/static contract; no weight payload reads, hash or inference',
            'file_bytes': path.stat().st_size, 'gguf_version': g.version, 'alignment': g.alignment,
            'header_end': g.header_end, 'data_start': g.data_start,
            'header_sha256': file_hash(path, g.header_end), 'full_file_sha256': None,
            'metadata_count': len(g.metadata), 'metadata': metadata,
            'special_tokens': {k: {'id': v, 'token': g.metadata['tokenizer.ggml.tokens'][v]}
                for k, v in g.metadata.items() if k.startswith('tokenizer.ggml.') and k.endswith('_token_id')},
            'tensor_count': len(details), 'tensor_types': Counter(t.type_name for t in g.tensors),
            'payload_bytes': sum(groups.values()), 'group_bytes': groups,
            'main_allmiss_expert_bytes_per_token': sum(r['expert_bytes']*contract['metadata']['expert_used_count']
                for r in details if r['family'] == 'routed'),
            'loader_contract': contract, 'tensors': details}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = protect_output(args.output, args.gguf)
        report = inspect_model(args.gguf)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(f'PASS: {report["tensor_count"]} tensors; {report["payload_bytes"]} payload bytes; {output}')
        return 0
    except (ValueError, OSError, KeyError, struct.error, OverflowError) as exc:
        print(f'MiMo inspection failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
