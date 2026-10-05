"""Step splitting regressions with a small vocabulary; full IDs use the native oracle."""
import unittest

from tools.strata_tokenizer import BYTE_TO_UNICODE, Tokenizer
from tools.check_step35_tokenizer import ARCHIVE_SHA256, LOADER_SHA, VALIDATED_PATCH_SETS, validate_provenance


def fixture(pre="deepseek-v3"):
    tokens = list(BYTE_TO_UNICODE.values())
    merges = []
    # Include tempting cross-boundary merges so incorrect splitting still
    # round-trips but fails these exact-piece expectations.
    for word in ("123456", "456", "中文abc", "abc中文", "a\u0301", "!ABC"):
        mapped = "".join(BYTE_TO_UNICODE[b] for b in word.encode())
        for n in range(2, len(mapped) + 1):
            if mapped[:n] not in tokens:
                tokens.append(mapped[:n])
                merges.append(mapped[:n - 1] + " " + mapped[n - 1])
    types = [1] * len(tokens)
    tokens.extend(["<|im_end|>", "<think>"])
    types.extend([3, 4])
    return Tokenizer(tokens, merges, types, pre=pre)


class StepTokenizerTests(unittest.TestCase):
    def test_oracle_provenance(self):
        record = {"architecture": "step35", "requested_revision": LOADER_SHA,
                  "archive_sha256": ARCHIVE_SHA256, "patches": "none"}
        for patches in VALIDATED_PATCH_SETS:
            validate_provenance({**record, "patches": patches})
        for bad in ([], {}, {**record, "architecture": "glm5next"}, {**record, "requested_revision": "main"},
                    {**record, "archive_sha256": "0" * 64}, {**record, "patches": "unknown"},
                    {**record, "patches": "cuda-f32-mmf-respect-tf32-override"}):
            with self.subTest(record=bad), self.assertRaises(ValueError):
                validate_provenance(bad)

    def test_sequential_cjk_and_digit_boundaries(self):
        tok = fixture()
        pieces = lambda text: [tok.token_bytes(i).decode("utf-8") for i in tok.encode(text)]
        self.assertEqual(pieces("123456"), ["123", "456"])
        self.assertEqual(pieces("中文abc"), ["中文", "abc"])
        self.assertEqual(pieces("abc中文"), ["abc", "中文"])
        self.assertEqual(pieces("a\u0301"), ["a\u0301"])
        self.assertEqual(pieces("!ABC"), ["!ABC"])

    def test_control_and_user_defined_tokens(self):
        tok = fixture()
        self.assertEqual(tok.encode("<think>", False), [tok.ids["<think>"]])
        self.assertEqual(tok.encode("<|im_end|>", True), [tok.ids["<|im_end|>"]])
        self.assertNotEqual(tok.encode("<|im_end|>", False), [tok.ids["<|im_end|>"]])
        for parse_special in (False, True):
            text = "Привет <think>中文123456\n<|im_end|>"
            self.assertEqual(tok.decode(tok.encode(text, parse_special)), text)
        self.assertEqual(tok.encode(""), [])

    def test_does_not_enable_glm_ignore_merges(self):
        tokens = list(BYTE_TO_UNICODE.values()) + ["ab", "bc", "abc"]
        tok = Tokenizer(tokens, ["b c", "a b", "ab c"], pre="deepseek-v3")
        self.assertEqual(tok.encode("abc"), [tok.ids["a"], tok.ids["bc"]])


if __name__ == "__main__":
    unittest.main()
