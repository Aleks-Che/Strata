"""Compare Hy3 Python token IDs and decoded bytes with the isolated native oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

if __package__:
    from .gguf_reader import GGUFFile
    from .inspect_hy3_gguf import file_hash as header_hash, protect_output
    from .hy3_loader_contract import LOADER_SHA, ARCHIVE_SHA256
    from .strata_tokenizer import Tokenizer
else:
    from gguf_reader import GGUFFile
    from inspect_hy3_gguf import file_hash as header_hash, protect_output
    from hy3_loader_contract import LOADER_SHA, ARCHIVE_SHA256
    from strata_tokenizer import Tokenizer

VALIDATED_PATCH_SETS = ("none", "hy3-mtp-load-flags")


def validate_provenance(provenance):
    if not isinstance(provenance, dict) or provenance.get("patches") not in VALIDATED_PATCH_SETS:
        raise ValueError("Unvalidated Hy3 oracle patch set")
    if provenance != {"architecture": "hy_v3", "requested_revision": LOADER_SHA,
                       "archive_sha256": ARCHIVE_SHA256, "patches": provenance["patches"]}:
        raise ValueError("Wrong Hy3 oracle provenance")


def corpus(meta):
    texts = ["", "Hello, world!", "Привет! Объясни, как работает конвейер.",
             "你好世界。 日本語のカタカナとひらがな 한글", "مرحبا بالعالم Ελληνικά हिन्दी",
             "1234567890 ١٢٣٤٥٦٧ １２３４５６７ 3.14159265 -123e+456",
             "a\u0301_б e\u0308 !ABC /test +foo 123中文456カタカナ",
             " leading  spaces\t\tend  ", "\r\n\n\r abc\n\n", "\x00\x01\x1b\x7f",
             "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n",
             '{"string":"строка","number":12,"list":[true,null]}',
             "👨‍👩‍👧‍👦 🚀 🏳️‍🌈 🧑🏽‍💻", "\u2028\u2029\u00a0\u200b\ufeff",
             "x" * 300, "abc " * 257,
             "<｜begin▁of▁sentence｜><|im_start|>user\nПривет<|im_end|>\n<|im_start|>assistant\n<think:opensource>\n",
             "<think:opensource>рассуждение</think:opensource>ответ<|im_end|>",
             '<tool_call>\n<function=weather>\n<parameter=city>\nЕкатеринбург\n</parameter>\n</function>\n</tool_call>',
             '<tool_response>{"temperature":5}</tool_response>', "<tool_calls></tool_calls>"]
    texts.extend(["<｜hy_begin_of_sentence:opensource｜><｜hy_User:opensource｜>Привет<｜hy_Assistant:opensource｜>",
                  "<tool_calls:opensource><tool_call:opensource>weather<tool_sep:opensource><arg_key:opensource>city</arg_key:opensource><arg_value:opensource>Екатеринбург</arg_value:opensource></tool_call:opensource></tool_calls:opensource>"])
    cases = [(f"text-{i:02d}", text) for i, text in enumerate(texts)]
    tokens, types = meta["tokenizer.ggml.tokens"], meta["tokenizer.ggml.token_type"]
    for index, (token, kind) in enumerate(zip(tokens, types)):
        if kind in (3, 4):
            cases.append((f"special-{index}", "A" + token + "Б" + token))
    rng = random.Random(3053)
    fragments = ["abc", "!ABC", "Привет", "123456789", "中文", "カタカナ", " ", "\t", "\n", "\r\n",
                 "a\u0301", "🧑🏽‍💻", "foo_bar", "<think:opensource>", "<|im_end|>", "\x00", " ", ":", "ß", "१२३४५"]
    for i in range(256):
        cases.append((f"mixed-{i:03d}", "".join(rng.choices(fragments, k=rng.randrange(1, 40)))))
    return [{"name": name, "text": text, "parse_special": special}
            for name, text in cases for special in (False, True)]


def compare(gguf, oracle, timeout=120, *, tokenizer=None):
    version = subprocess.run([str(oracle), "--version"], check=True, capture_output=True, timeout=15)
    provenance = json.loads(version.stdout)
    validate_provenance(provenance)
    header = GGUFFile(gguf)
    meta = header.metadata
    if (meta.get("general.architecture"), meta.get("tokenizer.ggml.pre")) != ("hy_v3", "hunyuan-dense"):
        raise ValueError("Expected hy3/hunyuan-dense model")
    from_export = tokenizer is not None
    tokenizer = Tokenizer.from_gguf(gguf) if tokenizer is None else tokenizer
    cases = corpus(meta)
    requests = "".join(json.dumps(c, ensure_ascii=True) + "\n" for c in cases) + "QUIT\n"
    native = subprocess.run([str(oracle), "--gguf", str(gguf)], input=requests.encode(),
                            check=True, capture_output=True, timeout=timeout)
    lines = native.stdout.splitlines()
    if len(lines) != len(cases) + 1:
        raise ValueError("Oracle response count differs from request count")
    info = json.loads(lines[0])
    if info.get("ready") is not True or info.get("add_special") is not False or info.get("vocab_size") != len(tokenizer.tokens):
        raise ValueError("Oracle vocabulary/ready mismatch")
    if info.get('bos') != meta['tokenizer.ggml.bos_token_id'] or info.get('eos') != meta['tokenizer.ggml.eos_token_id']:
        raise ValueError('Oracle BOS/EOS differ from metadata')
    if info.get('eog_ids') != [meta['tokenizer.ggml.eos_token_id']]:
        raise ValueError('Unreviewed Hy3 EOG set')
    failures = 0
    for case, line in zip(cases, lines[1:]):
        response = json.loads(line)
        ids = response.get("ids")
        if not isinstance(ids, list) or any(type(i) is not int or not 0 <= i < len(tokenizer.tokens) for i in ids):
            raise ValueError(f"Invalid oracle IDs: {case['name']}")
        python_ids = tokenizer.encode(case["text"], parse_special=case["parse_special"])
        python_bytes = b"".join(tokenizer.token_bytes(i) for i in python_ids)
        expected_bytes = case['text'].replace('\r', '').encode()
        match = ids == python_ids and response["decoded_hex"] == python_bytes.hex() == expected_bytes.hex()
        case.update({"python_ids": python_ids, "native_ids": ids, "match": match,
                     "lossless": python_bytes == case['text'].encode(),
                     "expected_cr_omissions": case['text'].count('\r')})
        if not match:
            failures += 1
            case["native_decoded_hex"] = response["decoded_hex"]
            case["python_decoded_hex"] = python_bytes.hex()
    return {"schema_version": 1, "status": "pass" if failures == 0 else "fail",
            "python_tokenizer_source": "supplied exported tokenizer" if from_export else "GGUF",
            "scope": "exact token IDs and decoded bytes, add_special=false; the local vocabulary/oracle drops CR; not inference",
            "cr_omission_case_count": sum(c['expected_cr_omissions'] > 0 for c in cases),
            "case_count": len(cases), "mismatch_count": failures, "native_info": info,
            "oracle_provenance": provenance, "oracle_binary_sha256": header_hash(oracle, oracle.stat().st_size),
            "gguf_header_sha256": header_hash(gguf, header.header_end), "gguf": str(gguf),
            "template_sha256": hashlib.sha256(meta["tokenizer.chat_template"].encode()).hexdigest(),
            "cases": cases, "oracle_stderr_tail": native.stderr[-4096:].decode("utf-8", errors="replace")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        gguf, oracle = args.gguf.resolve(), args.oracle.resolve()
        output = protect_output(args.output, gguf, oracle)
        # Also protect the other shards when --gguf is a directory.
        for source in [oracle, *gguf.parent.glob("*.gguf")]:
            if source.resolve() == output or (output.exists() and output.samefile(source)):
                raise ValueError("Output must differ from model and oracle inputs")
        report = compare(gguf, oracle)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f'{report["status"]}: {report["case_count"]} cases, {report["mismatch_count"]} mismatches; {output}')
        return 0 if report["status"] == "pass" else 1
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"Hy3 oracle check failed: {exc}", file=sys.stderr)
        if isinstance(exc, subprocess.CalledProcessError):
            print((exc.stderr or b"")[-4096:].decode("utf-8", errors="replace"), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
