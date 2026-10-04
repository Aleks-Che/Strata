"""Compare exact Python GLM tokenizer IDs with the compiled candidate oracle.

Run as python -m tools.check_glm5next_tokenizer --help. No model forward pass,
network or oracle build is performed. A report is not a template-rendering oracle:
both tokenizers receive the same prompts rendered by Strata's GLMTemplate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile

from serve.glm5next import GLMTemplate
from tools.gguf_reader import GGUFFile
from tools.glm5next_loader_contract import LOADER_SHA
from tools.glm5next_tokenizer_corpus import cases
from tools.strata_tokenizer import Tokenizer


def sha256_file(path, limit=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        while limit is None or limit > 0:
            chunk = source.read(1024 * 1024 if limit is None else min(limit, 1024 * 1024))
            if not chunk:
                break
            digest.update(chunk)
            if limit is not None:
                limit -= len(chunk)
    return digest.hexdigest()


def oracle_version(command, archive_sha256, timeout):
    if not re.fullmatch('[0-9a-fA-F]{64}', archive_sha256):
        raise ValueError('Expected a reviewed 64-digit archive SHA-256')
    result = subprocess.run([*command, '--version'], capture_output=True, timeout=timeout, check=True)
    fields = {}
    for line in result.stdout.decode('ascii').splitlines():
        key, separator, value = line.partition('=')
        if not separator or key in fields:
            raise ValueError('Malformed or duplicate oracle version field')
        fields[key] = value
    expected = {'requested_revision': LOADER_SHA, 'archive_sha256': archive_sha256.lower()}
    if fields != expected:
        raise ValueError(f'Oracle provenance mismatch: expected {expected}, got {fields}')
    return fields


def compare_ids(tokenizer, inputs, command, gguf, timeout):
    entries, requests = [], []
    for name, text in inputs:
        raw = text.encode('utf-8')
        if len(raw) > 1024 * 1024:
            raise ValueError('Comparison input exceeds oracle 1 MiB limit')
        for special in (False, True):
            entries.append({'name': name, 'text': text, 'parse_special': special,
                            'python_ids': tokenizer.encode(text, parse_special=special)})
            requests.append(f'ENC {int(special)} {raw.hex()}\n')
    if not entries:
        raise ValueError('Empty comparison corpus')
    result = subprocess.run([*command, '--gguf', str(gguf)],
                            input=(''.join(requests) + 'QUIT\n').encode('ascii'),
                            capture_output=True, timeout=timeout, check=True)
    lines = result.stdout.decode('ascii').splitlines()
    if not lines or lines[0] != 'READY_TOKENIZER' or len(lines) != len(entries) + 1:
        raise ValueError('Oracle must emit READY_TOKENIZER and exactly one IDS line per request')
    for entry, line in zip(entries, lines[1:]):
        if not re.fullmatch(r'IDS(?: [0-9]+)*', line):
            raise ValueError(f'Invalid oracle response for {entry["name"]}: {line[:200]}')
        ids = [int(value) for value in line.split()[1:]]
        if any(value >= len(tokenizer.tokens) for value in ids):
            raise ValueError(f'Oracle ID outside vocabulary: {entry["name"]}')
        expected = entry['python_ids']
        entry['oracle_ids'] = ids
        entry['match'] = ids == expected
        if ids != expected:
            entry['first_difference'] = next((i for i, (a, b) in enumerate(zip(expected, ids)) if a != b),
                                              min(len(expected), len(ids)))
    mismatches = sum(not entry['match'] for entry in entries)
    return {'status': 'pass' if not mismatches else 'fail', 'case_count': len(entries),
            'mismatch_count': mismatches, 'cases': entries,
            'oracle_stderr_tail': result.stderr[-8192:].decode('utf-8', errors='replace')}


def check(gguf, oracle, archive_sha256, timeout):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be finite and positive')
    gguf, oracle = Path(gguf).resolve(), Path(oracle).resolve()
    # Reject the wrong binary/build before loading the GGUF or invoking --gguf.
    provenance = oracle_version([str(oracle)], archive_sha256, min(timeout, 15))
    header = GGUFFile(gguf)
    meta = header.metadata
    if meta.get('general.architecture') != 'glm5next' or meta.get('tokenizer.ggml.pre') != 'glm4':
        raise ValueError('Expected glm5next/glm4 GGUF metadata')
    source = meta.get('tokenizer.chat_template')
    if not isinstance(source, str) or not source:
        raise ValueError('GGUF has no embedded chat template')
    tokenizer = Tokenizer.from_gguf(gguf)
    with tempfile.TemporaryDirectory() as directory:
        template_path = Path(directory) / 'chat_template.jinja'
        template_path.write_text(source, encoding='utf-8', newline='')
        inputs = cases(GLMTemplate(template_path))
    report = compare_ids(tokenizer, inputs, [str(oracle)], gguf, timeout)
    report.update({'oracle_provenance': provenance, 'oracle_path': str(oracle),
                   'oracle_binary_sha256': sha256_file(oracle), 'gguf_first_shard': str(gguf),
                   'gguf_header_bytes': header.header_end,
                   'gguf_header_sha256': sha256_file(gguf, header.header_end),
                   'template_sha256': hashlib.sha256(source.encode('utf-8')).hexdigest(),
                   'vocab_size': len(tokenizer.tokens)})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', type=Path, required=True)
    parser.add_argument('--oracle', type=Path, required=True)
    parser.add_argument('--archive-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=60)
    args = parser.parse_args(argv)
    # A typo in the report destination must never overwrite model/binary inputs.
    for source in (args.gguf, args.oracle):
        if args.output.resolve() == source.resolve() or (
                args.output.exists() and source.exists() and args.output.samefile(source)):
            parser.error('--output must differ from GGUF and oracle inputs')
    report = {'schema_version': 1, 'status': 'error',
              'scope': 'token IDs for common input text; not independent template rendering or GPU inference'}
    try:
        report.update(check(args.gguf, args.oracle, args.archive_sha256, args.timeout))
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        if isinstance(exc, subprocess.CalledProcessError):
            report['oracle_stderr_tail'] = (exc.stderr or b'')[-8192:].decode('utf-8', errors='replace')
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'{report["status"]}: {report.get("case_count", 0)} cases, report: {args.output}')
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
