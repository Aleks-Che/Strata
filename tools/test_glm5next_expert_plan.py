"""Reference GLM expert addressing; synthetic raw payloads, no GPU."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from tools.glm5next_expert_plan import plan_expert_reads
from tools.setup_glm5next import inspect_model
from tools.test_setup_glm5next import model_fixture, write_shard


class ExpertPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'model.gguf'
        meta, tensors = model_fixture()
        meta['glm5next.expert_feed_forward_length'] = 512
        for tensor in tensors:
            if '_exps.' in tensor['name']:
                tensor['shape'] = [512, 256, 2] if 'ffn_down_' in tensor['name'] else [256, 512, 2]
        write_shard(self.path, meta, tensors)
        self.report = inspect_model(self.path, tensor_details=True)

    def plan(self, layer=1, experts=(0, 1)):
        return plan_expert_reads(self.report, layer, experts)

    def test_all_five_encodings_and_transposed_down_shape(self):
        # Independent byte geometry, not imported from implementation's table.
        quant_bytes = {'IQ3_XXS': 98, 'IQ4_XS': 136, 'Q6_K': 210, 'Q3_K': 110, 'Q4_K': 144}
        seen = set()
        for layer, branch in ((1, 'main'), (2, 'mtp')):
            matrices = self.plan(layer)
            for m in matrices:
                seen.add(m.quant)
                self.assertEqual(m.branch, branch)
                self.assertEqual((m.columns, m.rows), (512, 256) if m.projection == 'down' else (256, 512))
                self.assertEqual(m.bytes, 512 * quant_bytes[m.quant])
                t = next(t for t in self.report['tensor_details']
                         if t['name'] == f'blk.{layer}.ffn_{m.projection}_exps.weight')
                self.assertEqual(m.file_offset, t['file_offset'] + m.expert * m.bytes)
                if m.expert == 1:
                    self.assertEqual(m.file_offset + m.bytes, t['file_offset'] + t['bytes'])
        self.assertEqual(seen, set(quant_bytes))

    def test_deduplicate_in_first_use_order_without_losing_triples(self):
        matrices = self.plan(experts=[1, 0, 1, 0])
        self.assertEqual([(m.expert, m.projection) for m in matrices],
                         [(e, p) for e in (1, 0) for p in ('gate', 'up', 'down')])
        self.assertEqual(self.plan(experts=[]), [])

    def test_bounded_chunks_preserve_raw_payload_and_last_partial_chunk(self):
        # Fill independent whole-tensor ranges. Each byte varies with its absolute
        # file position, so an off-by-one, padding or cross-expert read is visible.
        for t in self.report['tensor_details']:
            if '_exps.' in t['name']:
                payload = bytes((i + t['file_offset']) % 251 for i in range(t['bytes']))
                with self.path.open('r+b') as f:
                    f.seek(t['file_offset'])
                    f.write(payload)
        for layer in (1, 2):
            for m in self.plan(layer):
                for limit in (997, m.bytes, m.bytes + 1):
                    with self.subTest(quant=m.quant, expert=m.expert, limit=limit):
                        chunks = list(m.chunks(limit))
                        self.assertEqual(sum(n for _, n in chunks), m.bytes)
                        self.assertTrue(all(0 < n <= limit for _, n in chunks))
                        self.assertEqual(chunks[0][0], m.file_offset)
                        self.assertEqual(sum(chunks[-1]), m.file_offset + m.bytes)
                        with self.path.open('rb') as f:
                            f.seek(m.file_offset)
                            expected = f.read(m.bytes)
                            delivered = []
                            for offset, size in chunks:
                                f.seek(offset)
                                delivered.append(f.read(size))
                        self.assertEqual(b''.join(delivered), expected)
                        if limit == 997:
                            self.assertLess(chunks[-1][1], limit)

    def test_cache_key_separates_model_generation_branch_and_layout(self):
        m = self.plan()[0]
        key = m.cache_key('model-A', 1)
        self.assertNotEqual(key, m.cache_key('model-B', 1))
        self.assertNotEqual(key, m.cache_key('model-A', 2))
        changes = {'branch': 'mtp', 'layer': 2, 'expert': 1, 'projection': 'up',
                   'quant': 'Q4_K', 'columns': 512, 'rows': 256, 'shard': 'other.gguf',
                   'file_offset': m.file_offset + 1, 'bytes': m.bytes + 1}
        for field, value in changes.items():
            with self.subTest(field=field):
                self.assertNotEqual(key, replace(m, **{field: value}).cache_key('model-A', 1))
        for identity, generation in (('', 1), ('m', -1), ('m', True)):
            with self.assertRaises(ValueError):
                m.cache_key(identity, generation)

    def test_invalid_routes_and_dense_layers_fail_before_plan(self):
        for expert in (-1, 2, True, 0.5, '1'):
            with self.subTest(expert=expert), self.assertRaises(ValueError):
                self.plan(experts=[0, expert])
        for layer in (-1, 0, 3, True):
            with self.subTest(layer=layer), self.assertRaises(ValueError):
                self.plan(layer)
        for limit in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                list(self.plan()[0].chunks(limit))

    def test_invalid_layouts_and_file_ranges_rejected(self):
        for change in ({'shape': [255, 512, 2]}, {'shape': [256, 512]},
                       {'type': 'UNKNOWN'}, {'bytes': 1}, {'shard': 1},
                       {'file_offset': 0}, {'file_offset': 1 << 40},
                       {'shape': [512, 256, 2]}):
            report = deepcopy(self.report)
            next(t for t in report['tensor_details'] if t['name'] == 'blk.1.ffn_gate_exps.weight').update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                plan_expert_reads(report, 1, [0])
        report = deepcopy(self.report)
        report['tensor_details'].append(next(t for t in report['tensor_details']
                                             if t['name'] == 'blk.1.ffn_up_exps.weight'))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            plan_expert_reads(report, 1, [0])

    def test_offsets_above_32_bits_are_preserved(self):
        report = deepcopy(self.report)
        shift = 1 << 35
        for t in report['tensor_details']:
            t['file_offset'] += shift
        for s in report['shard_details']:
            s['data_start'] += shift
            s['file_bytes'] += shift
        for before, after in zip(self.plan(), plan_expert_reads(report, 1, [0, 1])):
            self.assertEqual(after.file_offset, before.file_offset + shift)
            self.assertEqual(after.bytes, before.bytes)


if __name__ == '__main__':
    unittest.main()
