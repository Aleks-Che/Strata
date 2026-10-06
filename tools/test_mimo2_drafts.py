"""Negative sidecar admission checks from the recorded five-file header inventory.

Uses header/tensor descriptors only; no model files, GPU or downloads.
"""
from dataclasses import replace
import json
from pathlib import Path
import unittest
from .gguf_reader import TensorInfo
from .inspect_mimo2_drafts import validate_draft


class DraftAdmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1]/'docs/mimo-v2.6-flash/MIMO26_FLASH_DRAFT_INSPECTION.json'
        records = json.loads(path.read_text(encoding='utf8'))['drafts']
        vocabulary = {'tokenizer.ggml.model':'gpt2', 'tokenizer.ggml.pre':'qwen2',
            'tokenizer.ggml.tokens':[str(i) for i in range(152576)],
            'tokenizer.ggml.token_type':[1]*152576, 'tokenizer.ggml.merges':[],
            'tokenizer.ggml.eos_token_id':151645, 'tokenizer.ggml.padding_token_id':151643,
            'tokenizer.ggml.add_bos_token':False, 'tokenizer.ggml.mask_token_id':151675}
        cls.target = {**next(r['metadata'] for r in records if r['contract']['kind']=='mtp'), **vocabulary}
        for name in ('head_count_kv', 'sliding_window_pattern'):
            key = 'mimo2.attention.'+name
            cls.target[key] = cls.target[key][:48]
        cls.cases = [(dict(r['metadata'], **vocabulary),
            [TensorInfo(t['name'],t['shape'],t['type_id'],t['type'],t['offset']) for t in r['tensors']]) for r in records]

    def test_five_reviewed_layouts(self):
        self.assertEqual(len(self.cases),5)
        for meta, tensors in self.cases:
            result=validate_draft(meta,tensors,self.target)
            self.assertIsNone(result['runtime_compatible'])
            self.assertFalse(result['acceptance_measured'])

    def test_tokenizer_mismatch(self):
        for key, value in [('tokens',['wrong']),('merges',['wrong pair']),('eos_token_id',0),('add_bos_token',True)]:
            meta,tensors=self.cases[0]
            with self.subTest(key=key), self.assertRaisesRegex(ValueError,'tokenizer mismatch'):
                validate_draft({**meta,'tokenizer.ggml.'+key:value},tensors,self.target)

    def test_wrong_mimo_specific_dflash_geometry(self):
        meta,tensors=next(c for c in self.cases if c[0]['general.architecture']=='dflash')
        for key, value in [('target_layers',[0,11,23,35,47]),('attention.value_scale',1.0),
                           ('rope.dimension_count',128),('attention.causal',True),('block_size',16)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_draft({**meta,'dflash.'+key:value},tensors,self.target)
        for key in ('dflash.sample_from_anchor','dflash.selector_rank','dflash.hyper_connection.count'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_draft({**meta,key:1},tensors,self.target)
        with self.assertRaises(ValueError):
            validate_draft({**meta,'tokenizer.ggml.mask_token_id':0},tensors,self.target)

    def test_missing_duplicate_extra_weights(self):
        for meta,tensors in self.cases:
            for bad in (tensors[:-1],tensors+[tensors[0]],tensors+[replace(tensors[0],name='blk.99.weight')]):
                with self.assertRaises(ValueError): validate_draft(meta,bad,self.target)

    def test_bad_shape_and_mixed_quant(self):
        for meta,tensors in self.cases:
            matrix=next(i for i,t in enumerate(tensors) if len(t.shape)==2)
            for change in ({'shape':[4096,1]}, {'type_name':'MXFP4','type_id':39},
                           {'type_name':'Q4_0' if tensors[matrix].type_name!='Q4_0' else 'BF16'}):
                bad=list(tensors);bad[matrix]=replace(bad[matrix],**change)
                with self.assertRaises(ValueError): validate_draft(meta,bad,self.target)

    def test_wrong_mtp_offset_or_heads(self):
        meta,tensors=next(c for c in self.cases if c[0]['general.architecture']=='mimo2')
        for key,value in [('mimo2.block_count',48),('mimo2.nextn_predict_layers',1),
                          ('mimo2.attention.head_count_kv',[8]*51)]:
            with self.assertRaises(ValueError): validate_draft({**meta,key:value},tensors,self.target)
        bad=[replace(t,name=t.name.replace('blk.48.','blk.0.')) for t in tensors]
        with self.assertRaises(ValueError): validate_draft(meta,bad,self.target)


if __name__ == '__main__': unittest.main()
