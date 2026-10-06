"""Validate compiled no-allocation registration and complete MTP-off omission."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

if __package__:
    from .hy3_loader_contract import ARCHIVE_SHA256, LOADER_SHA
    from .inspect_hy3_gguf import file_hash, inspect_model, protect_output
else:
    from hy3_loader_contract import ARCHIVE_SHA256, LOADER_SHA
    from inspect_hy3_gguf import file_hash, inspect_model, protect_output


def validate_registration(report, inventory):
    if (report.get('source_revision'), report.get('archive_sha256'), report.get('patches')) != (
            LOADER_SHA, ARCHIVE_SHA256, 'hy3-mtp-load-flags'):
        raise ValueError('Wrong Hy3 loader provenance')
    runs = report.get('runs', [])
    if len(runs) != 2 or [r.get('load_mtp') for r in runs] != [False, True]:
        raise ValueError('Expected MTP-off and MTP-on registrations')
    main = [t for t in inventory['tensors'] if t['scope'] == 'main']
    mtp = {t['name']: t['bytes'] for t in inventory['tensors'] if t['scope'] == 'mtp'}
    h = inventory['loader_contract']['metadata']
    for r in runs:
        if (r['main_blocks'], r['all_blocks'], r['main_tensors'], r['main_logical_bytes'], r['allocated_weight_bytes']) != (
                h['main_blocks'], h['block_count'], len(main), sum(t['bytes'] for t in main), 0):
            raise ValueError('Main registration or zero-allocation check failed')
        actual = {t['name']: t['bytes'] for t in r['mtp_tensors']}
        if len(actual) != len(r['mtp_tensors']) or actual != (mtp if r['load_mtp'] else {}):
            raise ValueError('MTP registration differs: off must omit all MTP tensors')
        if r['mtp_logical_bytes'] != sum(actual.values()):
            raise ValueError('MTP byte total differs')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', type=Path, required=True)
    parser.add_argument('--oracle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = protect_output(args.output, args.gguf, args.oracle)
        inventory = inspect_model(args.gguf)
        native = subprocess.run([str(args.oracle.resolve()), '--gguf', str(args.gguf.resolve())],
                                capture_output=True, check=True, timeout=60)
        report = json.loads(native.stdout)
        validate_registration(report, inventory)
        report.update(binary_sha256=file_hash(args.oracle, args.oracle.stat().st_size),
                      gguf_header_sha256=inventory['header_sha256'], mtp_off_skips_all_weights=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print('PASS: trunk unchanged, MTP off=0/on=all, zero allocated weight bytes;', output)
        return 0
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
        print(f'Hy3 loader check failed: {exc}', file=sys.stderr)
        if isinstance(exc, subprocess.CalledProcessError):
            print(exc.stderr[-3000:].decode(errors='replace'), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
