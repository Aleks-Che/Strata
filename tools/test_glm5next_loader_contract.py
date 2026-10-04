"""Loader preflight tests against recorded GGUF headers, with no weights or GPU."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

if __package__:
    from .gguf_reader import TensorInfo
    from .glm5next_loader_contract import LOADER_SHA, validate_loader_contract
else:
    from gguf_reader import TensorInfo
    from glm5next_loader_contract import LOADER_SHA, validate_loader_contract


class LoaderContractTests(unittest.TestCase):
    def setUp(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/glm5next_loader_headers.json").read_text(encoding="utf-8"))
        self.meta = fixture["metadata"]
        self.tensors = {name: TensorInfo(name, shape, type_id, type_name, 0)
                        for name, shape, type_id, type_name in fixture["tensors"]}

    def check(self):
        return validate_loader_contract(self.meta, self.tensors)

    def test_recorded_headers_and_mtp_boundary(self):
        report = self.check()
        self.assertEqual(report["source_revision"], LOADER_SHA)
        self.assertEqual(report["validated_tensors"], 89)
        self.assertEqual((report["main_tensors"], report["mtp_tensors"]), (60, 29))
        self.assertEqual(report["main_graph_blocks"], [0, 1])
        self.assertEqual(report["mtp_weight_blocks"], [2])
        self.assertEqual(report["kda_main_blocks"], [0])
        self.assertEqual(report["dsa_main_blocks"], [1])
        self.assertEqual(report["families"]["main.mhc"]["tensors"], 12)
        self.assertNotIn("mtp.mhc", report["families"])

    def test_missing_tensor_from_every_family(self):
        for name in ("token_embd.weight", "blk.0.attn_norm.weight", "blk.0.ssm_a",
                     "blk.0.ssm_conv1d_q.weight", "blk.0.hc_ffn_fn.weight", "blk.0.ffn_down.weight",
                     "blk.1.attn_k_b.weight", "blk.1.indexer_compressor_gate.weight",
                     "blk.1.indexer.k_norm.bias", "blk.1.exp_probs_b.bias",
                     "blk.1.ffn_down_shexp.weight", "blk.1.ffn_up_exps.weight",
                     "blk.2.nextn.hnorm.weight"):
            with self.subTest(name=name):
                tensor = self.tensors.pop(name)
                with self.assertRaisesRegex(ValueError, "Loader tensor names: missing=") as failure:
                    self.check()
                self.assertIn(name, str(failure.exception))
                self.tensors[name] = tensor

    def test_no_silent_tensor_renaming(self):
        tensor = self.tensors.pop("blk.1.indexer_compressor_ape.weight")
        tensor.name = "blk.1.indexer.kpool_ape.weight"
        self.tensors[tensor.name] = tensor
        with self.assertRaisesRegex(ValueError, "unexpected=.*indexer.kpool_ape"):
            self.check()

    def test_wrong_axis_order_and_nontrivial_shapes(self):
        for name, shape in (("blk.0.ssm_conv1d_k.weight", [4, 8192, 1]),
                            ("blk.1.attn_k_b.weight", [512, 256, 64]),
                            ("blk.1.indexer_compressor_ape.weight", [4, 128]),
                            ("blk.0.hc_attn_fn.weight", [24, 16384]),
                            ("blk.1.ffn_up_exps.weight", [4096, 2048, 287])):
            with self.subTest(name=name):
                original = self.tensors[name].shape
                self.tensors[name].shape = shape
                with self.assertRaisesRegex(ValueError, "Loader tensor shape"):
                    self.check()
                self.tensors[name].shape = original

    def test_trailing_singletons_only(self):
        self.tensors["blk.0.ssm_conv1d_q.weight"].shape.append(1)
        self.tensors["blk.1.exp_probs_b.bias"].shape.extend([1, 1, 1])
        self.check()
        self.tensors["blk.0.ssm_conv1d_q.weight"].shape.append(1)
        with self.assertRaisesRegex(ValueError, "Loader tensor shape"):
            self.check()

    def test_optional_mtp_embeddings_and_output(self):
        del self.tensors["output.weight"]
        report = self.check()
        self.assertEqual(len(report["optional_tensors_absent"]), 3)
        for name in ("blk.2.nextn.embed_tokens.weight", "blk.2.nextn.shared_head_head.weight"):
            tensor = deepcopy(self.tensors["token_embd.weight"])
            tensor.name = name
            self.tensors[name] = tensor
        self.assertEqual(self.check()["optional_tensors_absent"], ["output.weight"])
        self.tensors["blk.2.nextn.embed_tokens.weight"].shape = [4096, 1]
        with self.assertRaisesRegex(ValueError, "Loader tensor shape"):
            self.check()

    def test_wrong_architecture_or_mtp_in_main(self):
        self.meta["general.architecture"] = "glm5-next"
        with self.assertRaisesRegex(ValueError, "general.architecture"):
            self.check()
        self.meta["general.architecture"] = "glm5next"
        self.meta["glm5next.nextn_predict_layers"] = 0
        with self.assertRaisesRegex(ValueError, "nextn_predict_layers"):
            self.check()
        self.meta["glm5next.nextn_predict_layers"] = 2
        with self.assertRaises(ValueError):
            self.check()

    def test_invalid_and_missing_architecture_parameters(self):
        original = deepcopy(self.meta)
        bad_values = {
            "attention.head_count_kv": [[0, 1], [0, 1, 0], [0, 2, 1], [1, 1, 1]],
            "attention.indexer.top_k": [0, 2049], "attention.indexer.kpool": [0, 1],
            "kda.gate_lower_bound": [0, float("nan"), float("inf")],
            "ssm.conv_kernel": [1], "rope.dimension_count": [64],
            "hyper_connection.count": [0, True], "expert_used_count": [289],
            "expert_weights_norm": [1], "swiglu_clamp_exp": [[10.0], [1.0, 2.0, float("nan")]],
        }
        for key, values in bad_values.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    self.meta = deepcopy(original)
                    self.meta["glm5next." + key] = value
                    with self.assertRaisesRegex(ValueError, "Loader metadata"):
                        self.check()
        for key in ("kda.gate_lower_bound", "attention.indexer.kpool", "attention.layer_norm_epsilon"):
            with self.subTest(missing=key):
                self.meta = deepcopy(original)
                del self.meta["glm5next." + key]
                with self.assertRaisesRegex(ValueError, "Loader metadata"):
                    self.check()

    def test_optional_metadata_defaults(self):
        for key in ("expert_shared_feed_forward_length", "swiglu_clamp_exp", "swiglu_clamp_shexp"):
            del self.meta["glm5next." + key]
        self.check()


if __name__ == "__main__":
    unittest.main()
