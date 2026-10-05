#!/usr/bin/env python3
"""Guarded, short-context greedy Step MTP trials; never updates a serve profile."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

from check_step35_model import Monitor
from gguf_reader import GGUFFile

OFFICIAL_Q8_SHA256 = '469a81667a6cd6d87a85d501d57155fd90cee5af7010fd289c5169881763fd57'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inspect_draft(model, draft):
    target, sidecar = GGUFFile(model), GGUFFile(draft)
    a, b = target.metadata, sidecar.metadata
    if b.get('general.architecture') != 'step35' or b.get('step35.block_count') != 48 or b.get('step35.nextn_predict_layers') != 3:
        raise ValueError('expected Step 45+3 sidecar')
    # Padding and automatic EOS insertion differ between the publishers. The
    # probe uses target-tokenized IDs and never inserts draft special tokens.
    keys = ['tokenizer.ggml.model', 'tokenizer.ggml.pre', 'tokenizer.ggml.tokens',
            'tokenizer.ggml.token_type', 'tokenizer.ggml.merges',
            'tokenizer.ggml.bos_token_id', 'tokenizer.ggml.eos_token_id',
            'tokenizer.chat_template']
    for key in keys:
        if a.get(key) != b.get(key):
            raise ValueError(f'incompatible sidecar: {key}')
    for key, value in a.items():
        if not key.startswith('step35.') or key == 'step35.block_count':
            continue
        other = b.get(key)
        if isinstance(value, list) and isinstance(other, list):
            other = other[:len(value)]
        if value != other:
            raise ValueError(f'incompatible architecture metadata: {key}')
    size, previous, names = draft.stat().st_size, 0, set()
    tensors = []
    for tensor in sorted(sidecar.tensors, key=lambda t: t.offset):
        count = tensor.expected_bytes()
        if count is None or tensor.name in names or tensor.offset < previous or tensor.offset % sidecar.alignment:
            raise ValueError('invalid draft tensor directory')
        end = sidecar.data_start + tensor.offset + count
        if end > size:
            raise ValueError(f'truncated draft tensor: {tensor.name}')
        if tensor.name.startswith('blk.') and tensor.name.split('.')[1] not in ('45', '46', '47'):
            raise ValueError('sidecar contains unexpected trunk block')
        previous = tensor.offset + count
        names.add(tensor.name)
        tensors.append(dict(name=tensor.name, shape=tensor.shape, type=tensor.type_name, bytes=count))
    for i in range(45, 48):
        for tail in ('nextn.eh_proj', 'nextn.enorm', 'nextn.hnorm', 'ffn_gate', 'ffn_up', 'ffn_down'):
            if f'blk.{i}.{tail}.weight' not in names:
                raise ValueError(f'missing dense MTP tensor: {i}/{tail}')
    if any('_exps.' in name for name in names):
        raise ValueError('this probe admits a dense draft only')
    return dict(sha256=sha256(draft), bytes=size, tensors=tensors,
                tokenizer_equal_keys=keys,
                special_metadata_differences={k: {'target': a.get(k), 'draft': v} for k, v in b.items()
                                             if k.startswith('tokenizer.') and a.get(k) != v})


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = json.loads(args.reference.read_text(encoding='utf8'))
    if reference['status'] != 'pass' or Path(reference['model']).resolve() != args.model.resolve():
        raise ValueError('a passing reference for this exact model path is required')
    depths = [int(x) for x in args.depths.split(',')]
    if any(x not in range(4) for x in depths) or (not args.draft and any(depths)):
        raise ValueError('depths must be 0..3; nonzero depth needs --draft')
    if max(depths)>args.active_heads or (args.draft_placement!='active-heads' and args.active_heads!=3):
        raise ValueError('requested depth/placement is incompatible with --active-heads')
    report = dict(status='running', scope='greedy short-context MTP; not HTTP admission',
                  engine=str(args.engine.resolve()), engine_sha256=sha256(args.engine),
                  model=str(args.model.resolve()), draft=str(args.draft.resolve()) if args.draft else None,
                  cache_mib=args.cache_mib, draft_placement=args.draft_placement,
                  active_heads=args.active_heads,
                  prefault_experts=args.prefault_experts,
                  ram_prefault_reserve_mib=args.ram_prefault_reserve_mib,
                  p_min=args.p_min, rounds=args.rounds, runs=[])
    report_path = args.output_dir/'mtp-report.json'

    def save():
        temporary = report_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
        temporary.replace(report_path)

    save()
    process, monitor = None, Monitor()
    try:
        if args.draft:
            report['sidecar'] = inspect_draft(args.model, args.draft)
            if report['sidecar']['sha256'] != args.draft_sha256.lower():
                raise ValueError('sidecar SHA-256 differs from the expected publication')
            save()
        command = [str(args.engine.resolve()), '--model', str(args.model.resolve()), '--cache-mib', str(args.cache_mib)]
        if args.prefault_experts:
            command += ['--prefault-experts', 'on', '--ram-prefault-reserve-mib', str(args.ram_prefault_reserve_mib)]
        if args.draft:
            command += ['--draft', str(args.draft.resolve()), '--draft-placement', args.draft_placement,
                        '--active-heads', str(args.active_heads)]
        env = os.environ.copy()
        cuda = Path(__file__).resolve().parents[1]/'build-local/cuda-13.0/bin'
        env['PATH'] = str(cuda)+os.pathsep+str(cuda/'x64')+os.pathsep+env['PATH']
        with (args.output_dir/'stderr.log').open('w', encoding='utf8') as stderr:
            start = time.perf_counter()
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                                       text=True, encoding='utf8', env=env, bufsize=1,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            monitor.start(process)
            lines = queue.Queue()

            def reader():
                for line in process.stdout:
                    lines.put(line.rstrip())
                lines.put(None)
            thread = threading.Thread(target=reader, daemon=True)
            thread.start()

            def receive(prefix):
                deadline = time.monotonic()+1200
                while time.monotonic() < deadline:
                    try:
                        line = lines.get(timeout=1)
                    except queue.Empty:
                        if process.poll() is not None:
                            raise RuntimeError(f'probe exited {process.returncode}; see stderr.log')
                        continue
                    if line is None:
                        raise RuntimeError(f'probe closed stdout; see stderr.log ({monitor.error})')
                    if line.startswith(prefix):
                        return json.loads(line[len(prefix):])
                    raise RuntimeError(f'unexpected probe output: {line[:200]}')
                raise TimeoutError('probe request/startup timed out')

            report['ready'] = receive('READY ')
            report['startup_seconds'] = time.perf_counter()-start
            print(f"READY in {report['startup_seconds']:.2f}s", flush=True)
            save()
            # Warm every tested depth on both prompts (also the draft graphs),
            # then reverse order on alternate rounds. Keep warmup observations.
            plan = [(-1, d) for d in depths] + [(r, d) for r in range(args.rounds)
                                          for d in (depths if r % 2 == 0 else depths[::-1])]
            for round_id, depth in plan:
                for index, prompt in enumerate(reference['prompts'][:2]):
                    payload = dict(tokens=prompt['ids'], predict=128, depth=depth, p_min=args.p_min)
                    start = time.perf_counter()
                    process.stdin.write(json.dumps(payload)+'\n'); process.stdin.flush()
                    result = receive('RESULT ')
                    result.update(round=round_id, prompt=index, client_seconds=time.perf_counter()-start)
                    result['exact_reference_ids'] = result['ids'] == reference['runs'][0]['requests'][index]['ids']
                    report['runs'].append(result)
                    save()
                    print(f"round={round_id} depth={depth} prompt={index}: {result['decode_tokens_per_second']:.3f} tok/s, "
                          f"accepted={result['accepted']}/{result['proposed']}, exact={result['exact_reference_ids']}", flush=True)
                    if not result['exact_reference_ids']:
                        raise ValueError('greedy token mismatch; performance is not admitted')
            process.stdin.write('QUIT\n'); process.stdin.flush()
            process.wait(timeout=60)
            report['exit_code'] = process.returncode
            if process.returncode:
                raise RuntimeError('probe did not exit cleanly')
            if monitor.error:
                raise RuntimeError(monitor.error)
            report['status'] = 'pass'
    except Exception as error:
        report['status'], report['error'] = 'error', str(error)
    finally:
        if process and process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
        monitor.close()
        report['monitor_error'] = monitor.error
        report['memory_samples'] = monitor.samples
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--draft', type=Path)
    parser.add_argument('--draft-sha256', default=OFFICIAL_Q8_SHA256,
                        help='expected publication checksum; defaults to the official Q8_0')
    parser.add_argument('--reference', type=Path, default=Path('docs/Step-3.7-Flash/STEP37_FLASH_MODEL_VALIDATION.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cache-mib', type=int, default=16384, help='cap; runtime also reserves 5%% global VRAM + 256 MiB')
    parser.add_argument('--draft-placement', choices=('resident', 'shared-embedding', 'active-heads'), default='resident')
    parser.add_argument('--active-heads', type=int, choices=(1,2,3), default=3)
    parser.add_argument('--prefault-experts', action='store_true', help='touch mapped expert pages until RAM headroom is reached')
    parser.add_argument('--ram-prefault-reserve-mib', type=int, default=24576,
                        help='additional startup reserve below the 95%% ceiling for pages needed by generation')
    parser.add_argument('--depths', default='0,1,2,3')
    parser.add_argument('--p-min', type=float, default=0.6)
    parser.add_argument('--rounds', type=int, default=2)
    args = parser.parse_args()
    if (not 1 <= args.rounds <= 8 or not 0 <= args.p_min <= 1 or not 0 <= args.cache_mib <= 16384
            or not 512 <= args.ram_prefault_reserve_mib <= 32768):
        parser.error('invalid round/probability/cache configuration')
    report = run(args)
    print(report['status'], report.get('error',''), flush=True)
    return int(report['status'] != 'pass')


if __name__ == '__main__':
    raise SystemExit(main())
