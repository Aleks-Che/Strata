"""Reference byte-range plan for GLM routed experts; no payload reads or GPU.

Consumes inspect_model(tensor_details=True). Runtime code must additionally keep
source handles alive and wait for CUDA consumers before reusing cache/staging.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from tools.gguf_reader import BLOCK_GEOMETRY
from tools.setup_glm5next import inspect_model


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


@dataclass(frozen=True)
class ExpertMatrix:
    branch: str
    layer: int
    expert: int
    projection: str
    quant: str
    columns: int
    rows: int
    shard: str
    file_offset: int
    bytes: int

    def cache_key(self, model_identity: str, load_generation: int):
        """Caller assigns a new generation on every reload, including same path.

        model_identity identifies the complete model, not just its architecture.
        The key includes layout and file range, so distinct quantizations/branches
        cannot alias. It does not replace source lifetime or CUDA event tracking.
        """
        if not isinstance(model_identity, str) or not model_identity:
            raise ValueError('model_identity must be a nonempty string')
        integer(load_generation, 'load_generation')
        return (model_identity, load_generation, self.branch, self.layer, self.expert,
                self.projection, self.quant, self.columns, self.rows, self.shard,
                self.file_offset, self.bytes)

    def chunks(self, max_bytes):
        """Yield exact (file offset, bytes) ranges, without rounding into neighbours.

        The limit applies to reads, not GPU slots. A chunk may split a quant block;
        consumers must assemble all raw bytes before interpreting the matrix.
        """
        integer(max_bytes, 'max_bytes', 1)
        end = self.file_offset + self.bytes
        offset = self.file_offset
        while offset < end:
            size = min(max_bytes, end - offset)
            yield offset, size
            offset += size


def plan_expert_reads(report, layer, expert_ids):
    """Deduplicate routed IDs in first-use order; preserve each gate/up/down triple.

    GGUF shapes are [input columns, output rows, experts], with contiguous packed
    rows. Never add general.alignment between expert slices inside one tensor.
    Reject invalid routes/layouts before returning any plan to the caller.
    """
    if report.get('architecture') != 'glm5next':
        raise ValueError('Expected a glm5next inspection report')
    integer(layer, 'layer')
    main = integer(report['main_blocks'], 'main_blocks', 1)
    blocks = integer(report['block_count'], 'block_count', main)
    if layer >= blocks:
        raise ValueError('layer is outside block_count')
    details = report.get('tensor_details')
    if not isinstance(details, list):
        raise ValueError('Inspection must include tensor_details')
    selected = []
    for expert in expert_ids:
        integer(expert, 'expert ID')
        if expert not in selected:
            selected.append(expert)
    branch = 'main' if layer < main else 'mtp'
    layouts = []
    for projection in ('gate', 'up', 'down'):
        name = f'blk.{layer}.ffn_{projection}_exps.weight'
        matches = [t for t in details if t['name'] == name]
        if len(matches) != 1:
            raise ValueError(f'Missing or duplicate routed matrix: {name}')
        tensor = matches[0]
        shape = tensor['shape']
        if len(shape) != 3:
            raise ValueError(f'Expected [columns, rows, experts]: {name}')
        columns, rows, experts = [integer(n, 'shape dimension', 1) for n in shape]
        geometry = BLOCK_GEOMETRY.get(tensor['type'])
        if geometry is None or columns % geometry[0]:
            raise ValueError(f'Unsupported quant or partial quantized row: {name}')
        size = columns // geometry[0] * geometry[1] * rows
        if integer(tensor['bytes'], 'tensor bytes', 1) != size * experts:
            raise ValueError(f'Tensor byte count disagrees with layout: {name}')
        if any(e >= experts for e in selected):
            raise ValueError(f'Expert ID outside tensor: {name}')
        shard_no = integer(tensor['shard'], 'shard')
        if shard_no >= len(report['shard_details']):
            raise ValueError(f'Shard index outside report: {name}')
        shard = report['shard_details'][shard_no]
        start = integer(tensor['file_offset'], 'file_offset')
        if start < shard['data_start'] or start + size * experts > shard['file_bytes']:
            raise ValueError(f'Matrix range outside shard data: {name}')
        layouts.append((projection, tensor['type'], columns, rows, experts, shard['name'], start, size))
    gate, up, down = layouts
    if gate[2:5] != up[2:5] or (gate[3], gate[2], gate[4]) != down[2:5]:
        raise ValueError('gate/up/down shapes or expert counts disagree')
    return [ExpertMatrix(branch, layer, expert, projection, quant, columns, rows,
                         shard, start + expert * size, size)
            for expert in selected
            for projection, quant, columns, rows, _, shard, start, size in layouts]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', type=Path, required=True, help='First GGUF shard')
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--experts', type=int, nargs='+', required=True)
    parser.add_argument('--chunk-bytes', type=int, default=1024 * 1024)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        integer(args.chunk_bytes, 'chunk-bytes', 1)
        report = inspect_model(args.gguf, tensor_details=True)
        matrices = plan_expert_reads(report, args.layer, args.experts)
        result = {'schema_version': 1, 'first_shard': report['first_shard'],
                  'validation_scope': 'reference byte ranges from headers; no payload reads, runtime cache or GPU validation',
                  'layer': args.layer, 'requested_experts': args.experts,
                  'chunk_bytes': args.chunk_bytes, 'matrix_count': len(matrices),
                  'total_bytes': sum(m.bytes for m in matrices),
                  'matrices': [{**asdict(m), 'chunks': list(m.chunks(args.chunk_bytes))} for m in matrices]}
        args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    except (ValueError, OSError) as exc:
        parser.exit(1, f'{exc}\n')
    print(f'{len(matrices)} matrices, {result["total_bytes"]} bytes; plan saved to {args.output}')


if __name__ == '__main__':
    main()
