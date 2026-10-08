"""Launch the experimental MiniMax baseline with precision set before process start.

Examples (from repository root):
  python tools/run_minimax_m2.py -- --gguf MODEL.gguf --request request.json --output report.json
  python tools/run_minimax_m2.py --engine build-local/minimax-m2-cuda/bin/strata-minimax-m2-runtime-check.exe -- NEW_DIRECTORY
The child inherits stdin/stdout, including the native --pipe JSONL protocol.
"""
import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PRECISION_ENV = {
    'NVIDIA_TF32_OVERRIDE': '0', 'GGML_CUDA_CUBLAS_COMPUTE_TYPE': 'f32',
    'GGML_CUDA_DISABLE_GRAPHS': '1', 'LLAMA_GRAPH_REUSE_DISABLE': '1',
    'GGML_OP_OFFLOAD_MIN_BATCH': '1', 'STRATA_MM27_QUANT_F32': '1',
}


def runtime_environment(cuda_root):
    env = dict(os.environ, **PRECISION_ENV)
    root = Path(cuda_root).resolve()
    if not root.is_dir():
        raise ValueError(f'CUDA directory not found: {root}')
    env['PATH'] = os.pathsep.join([str(root / 'bin'), str(root / 'bin' / 'x64'), env.get('PATH', '')])
    return env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine', type=Path, default=ROOT / 'build-local/minimax-m2-cuda/bin/strata-minimax-m2-bench.exe')
    p.add_argument('--cuda-root', type=Path, default=ROOT / 'build-local/cuda-13.0')
    p.add_argument('args', nargs=argparse.REMAINDER)
    args = p.parse_args()
    if not args.engine.is_file():
        p.error(f'engine not found: {args.engine}')
    rest = args.args[1:] if args.args[:1] == ['--'] else args.args
    try:
        return subprocess.call([str(args.engine.resolve()), *rest], env=runtime_environment(args.cuda_root))
    except (OSError, ValueError) as exc:
        p.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())
