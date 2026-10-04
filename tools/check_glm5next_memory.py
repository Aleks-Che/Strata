"""Opt-in real NVML/cache pressure check using a separate bounded CUDA holder.

No model inference or throughput measurement. Requires Windows NVIDIA NVML.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import queue
import subprocess
import tempfile
import threading

MIB = 1 << 20


class Worker:
    def __init__(self, command, timeout):
        self.timeout = timeout
        self.errors = tempfile.TemporaryFile()
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=self.errors, text=True, encoding='utf-8')
        except BaseException:
            self.errors.close()
            raise
        self.lines = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                self.lines.put(line)
        finally:
            self.lines.put(None)

    def expect(self, phase):
        try:
            line = self.lines.get(timeout=self.timeout)
        except queue.Empty as exc:
            raise TimeoutError(f'Timed out waiting for {phase}') from exc
        if line is None:
            self.errors.seek(0)
            raise ValueError(f'Worker ended before {phase}: {self.errors.read().decode("utf-8", errors="replace")[-4096:]}')
        value = json.loads(line)
        if not isinstance(value, dict) or value.get('phase') != phase:
            raise ValueError(f'Expected {phase}, got {value!r}')
        return value

    def send(self, command):
        self.process.stdin.write(command + '\n')
        self.process.stdin.flush()

    def finish(self):
        self.send('QUIT')
        self.process.stdin.close()
        code = self.process.wait(timeout=self.timeout)
        self.reader.join(timeout=self.timeout)
        if code or self.reader.is_alive() or self.lines.get(timeout=self.timeout) is not None:
            raise ValueError(f'Worker did not exit cleanly: {code}')

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if not self.process.stdin.closed:
            try:
                self.process.stdin.close()  # EOF lets the worker release its CUDA allocations.
            except OSError:
                pass  # An exited child may already have closed its pipe.
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()  # Only this runner's child, never an unrelated process.
            self.process.wait(timeout=3)
        self.reader.join(timeout=3)
        self.process.stdout.close()
        self.errors.close()


def validate_stage(record, phase):
    expected = {
        'warm': (64*MIB, 64*MIB, 0, 1, 0, 2, 0),
        'protected': (0, 32*MIB, 32*MIB, 0, 1, 2, 1),
        'trimmed': (0, 0, 0, 1, 1, 2, 2),
        'recovered': (64*MIB, 64*MIB, 0, 1, 1, 4, 2),
    }[phase]
    fields = ('target_bytes', 'resident_bytes', 'deferred_bytes', 'trim_complete', 'hits', 'misses', 'evictions')
    if (record.get('phase') != phase or record.get('sample_valid') != 1 or
            tuple(record.get(k) for k in fields) != expected):
        raise ValueError(f'Unexpected controller result for {phase}: {record!r}')
    free, total = record.get('free_bytes'), record.get('total_bytes')
    if type(free) is not int or type(total) is not int or not 0 <= free <= total or total == 0:
        raise ValueError(f'Invalid live memory values for {phase}')
    target = record.get('device_target_mib')
    if type(target) is not int or not 0 < target <= 1048576:
        raise ValueError(f'Invalid device usage target for {phase}')


def run_pair(checker, policy, timeout):
    with ExitStack() as stack:
        holder = stack.enter_context(Worker([str(checker), '--holder'], timeout))
        holder_ready = holder.expect('ready')
        controller = stack.enter_context(Worker([str(checker), '--controller-' + policy], timeout))
        ready = controller.expect('ready')
        if (not ready.get('pci') or ready != holder_ready or holder.process.pid == controller.process.pid or
                ready.get('cuda_runtime', 0) <= 0 or ready.get('cuda_driver', 0) <= 0):
            raise ValueError('Holder and controller must be distinct processes on the same GPU')
        stages = []
        controller.send('START')
        warm = controller.expect('warm'); validate_stage(warm, 'warm'); stages.append(warm)
        holder.send('ALLOC')
        allocated = holder.expect('allocated')
        if allocated.get('bytes') != 256*MIB:
            raise ValueError('Unexpected external pressure allocation size')
        for command, phase in (('PRESSURE', 'protected'), ('TRIM', 'trimmed')):
            controller.send(command)
            result = controller.expect(phase); validate_stage(result, phase); stages.append(result)
        holder.send('FREE'); holder.expect('freed')
        controller.send('RECOVER')
        recovered = controller.expect('recovered'); validate_stage(recovered, 'recovered'); stages.append(recovered)
        if len({s['device_target_mib'] for s in stages}) != 1:
            raise ValueError('Device usage target changed during pressure scenario')
        controller.finish(); holder.finish()
        return {'status': 'pass', 'policy': policy, 'holder_pid': holder.process.pid,
                'controller_pid': controller.process.pid, 'device': ready, 'stages': stages}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checker', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=20)
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('timeout must be positive and finite')
    if (args.output.suffix.lower() != '.json' or args.output.resolve() == args.checker.resolve() or
            (args.output.exists() and args.checker.exists() and args.output.samefile(args.checker))):
        parser.error('Output must be a JSON report distinct from checker')
    result = {'schema_version': 1, 'status': 'error',
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Real external CUDA allocation and NVML/cache controller; synthetic weights, no GLM graph or throughput',
              'holder_bytes': 256*MIB, 'cache_cap_bytes': 64*MIB, 'matrix_bytes': 32*MIB,
              'policy_mode': 2, 'reserve_mib': 128, 'byte_comparisons_per_policy': 5,
              'worker_timeout_seconds': args.timeout, 'sample_retry_deadline_seconds': 8,
              'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'checker': str(args.checker.resolve()), 'runs': []}
    try:
        result['checker_sha256'] = hashlib.sha256(args.checker.read_bytes()).hexdigest()
        for policy in ('lru', 'frequency'):
            result['runs'].append(run_pair(args.checker.resolve(), policy, args.timeout))
        if hashlib.sha256(args.checker.read_bytes()).hexdigest() != result['checker_sha256']:
            raise ValueError('Checker changed during pressure validation')
        result['status'] = 'pass'
    except (OSError, ValueError, subprocess.SubprocessError, queue.Empty) as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'{result["status"]}: {args.output}')
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
