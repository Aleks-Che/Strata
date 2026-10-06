"""Read-only Hy3 header/range admission. Never load or hash model weight payloads."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import sys

if __package__:
    from .gguf_reader import BLOCK_GEOMETRY, GGML_TYPES, GGUFFile
    from .hy3_loader_contract import validate_loader_contract
else:
    from gguf_reader import BLOCK_GEOMETRY, GGML_TYPES, GGUFFile
    from hy3_loader_contract import validate_loader_contract


def file_hash(path, count):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while count:
            chunk = stream.read(min(count, 1024 * 1024))
            if not chunk:
                raise ValueError('Truncated input while hashing')
            digest.update(chunk)
            count -= len(chunk)
    return digest.hexdigest()


def protect_output(output, *inputs):
    output = Path(output).resolve()
    for source in inputs:
        source = Path(source).resolve()
        if output == source or (output.exists() and source.exists() and output.samefile(source)):
            raise ValueError('Output must differ from all inputs')
    return output


def validate_ranges(tensors, data_start, file_size, alignment):
    if type(alignment) is not int or not 1 <= alignment <= 2**20 or alignment & (alignment - 1):
        raise ValueError('Invalid GGUF alignment')
    if not 0 <= data_start <= file_size <= 2**63 - 1 or data_start % alignment:
        raise ValueError('Invalid file/data bounds')
    details, names = [], set()
    for t in tensors:
        if not t.name or t.name in names:
            raise ValueError(f'Empty or duplicate tensor name: {t.name}')
        names.add(t.name)
        if not 1 <= len(t.shape) <= 4 or any(type(d) is not int or d <= 0 for d in t.shape):
            raise ValueError(f'Invalid tensor shape: {t.name}')
        geometry = BLOCK_GEOMETRY.get(t.type_name)
        if geometry is None or GGML_TYPES.get(t.type_id) != t.type_name:
            raise ValueError(f'Unknown tensor encoding: {t.name}')
        if t.shape[0] % geometry[0]:
            raise ValueError(f'Invalid quantized row: {t.name}')
        size = t.expected_bytes()
        if type(t.offset) is not int or t.offset < 0 or t.offset % alignment:
            raise ValueError(f'Unaligned or invalid offset: {t.name}')
        start, end = data_start + t.offset, data_start + t.offset + size
        if end > 2**63 - 1 or end > file_size:
            raise ValueError(f'Truncated or overflowing tensor range: {t.name}')
        row = {'name': t.name, 'shape': t.shape, 'type': t.type_name, 'type_id': t.type_id,
               'offset': t.offset, 'file_offset': start, 'bytes': size,
               'row_bytes': t.shape[0] // geometry[0] * geometry[1]}
        if t.name.endswith('_exps.weight'):
            if len(t.shape) != 3:
                raise ValueError(f'Expected rank-3 experts: {t.name}')
            row['expert_bytes'] = size // t.shape[2]
        details.append(row)
    end = data_start
    for t in sorted(details, key=lambda r: r['file_offset']):
        if t['file_offset'] < end:
            raise ValueError(f'Overlapping tensor ranges: {t["name"]}')
        end = t['file_offset'] + t['bytes']
    return details


def inspect_model(path):
    path = Path(path).resolve()
    parsed = GGUFFile(path)
    meta = parsed.metadata
    details = validate_ranges(parsed.tensors, parsed.data_start, path.stat().st_size,
                              meta.get('general.alignment', 32))
    contract = validate_loader_contract(meta, parsed.tensors)
    mapping = {row['name']: row for row in contract['tensor_mapping']}
    groups = Counter()
    for row in details:
        row.update({k: mapping[row['name']][k] for k in ('scope', 'family', 'layer')})
        group = row['scope'] + ('_routed' if row['family'] == 'routed' else '_other')
        groups[group] += row['bytes']
    summary = {}
    for k, v in meta.items():
        if k == 'tokenizer.chat_template':
            summary[k] = {'utf8_bytes': len(v.encode('utf-8')), 'sha256': hashlib.sha256(v.encode('utf-8')).hexdigest()}
        elif k.startswith('tokenizer.') and isinstance(v, list):
            summary[k] = {'count': len(v), 'json_sha256': hashlib.sha256(json.dumps(v, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()}
        else:
            summary[k] = v
    special = {k: {'id': v, 'token': meta['tokenizer.ggml.tokens'][v]}
               for k, v in meta.items() if k.startswith('tokenizer.ggml.') and k.endswith('_token_id')}
    return {'schema_version': 1, 'status': 'pass', 'model': str(path),
            'scope': 'header and static loader contract only; weights not read or hashed; no inference',
            'file_bytes': path.stat().st_size, 'gguf_version': parsed.version,
            'alignment': parsed.alignment, 'header_end': parsed.header_end, 'data_start': parsed.data_start,
            'header_sha256': file_hash(path, parsed.header_end), 'full_file_sha256': None,
            'metadata_count': len(meta), 'metadata': summary, 'special_tokens': special,
            'tensor_count': len(details), 'tensor_types': Counter(t.type_name for t in parsed.tensors),
            'payload_bytes': sum(groups.values()), 'group_bytes': groups,
            'main_allmiss_expert_bytes_per_token': sum(r['expert_bytes'] * contract['metadata']['expert_used_count']
                for r in details if r['scope'] == 'main' and r['family'] == 'routed'),
            'loader_contract': contract, 'tensors': details}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = protect_output(args.output, args.gguf)
        report = inspect_model(args.gguf)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f'PASS: {report["tensor_count"]} tensors; {report["payload_bytes"]} payload bytes; {output}')
        return 0
    except (ValueError, OSError, KeyError, struct.error, OverflowError) as exc:
        print(f'Hy3 inspection failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
