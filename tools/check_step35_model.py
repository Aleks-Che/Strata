"""Full Step GGUF admission: native/pinned selected-copy exact logits and IDs.

Timings describe these sequential runs, not a controlled speed comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time

import psutil

if __package__:
    from .check_step35_engine import Engine
    from .check_step35_template import renderer, TEMPLATE_SHA
    from .gguf_reader import GGUFFile
    from .setup_step35 import first_shard
    from .strata_tokenizer import Tokenizer
else:
    from check_step35_engine import Engine
    from check_step35_template import renderer, TEMPLATE_SHA
    from gguf_reader import GGUFFile
    from setup_step35 import first_shard
    from strata_tokenizer import Tokenizer


class Monitor:
    def __init__(self):
        self.samples, self.error = [], None
        self.stop = threading.Event()
        self.thread = None

    def start(self, process):
        def loop():
            child = psutil.Process(process.pid)
            started = time.perf_counter()
            index, gpu = 0, None
            while not self.stop.is_set() and process.poll() is None:
                try:
                    ram = psutil.virtual_memory()
                    mem = child.memory_info()
                    if index % 4 == 0:
                        result = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=memory.total,memory.used",
                            "--format=csv,noheader,nounits"], capture_output=True, check=True, text=True, timeout=10,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                        total, used = map(float, result.stdout.strip().split(","))
                        gpu = {"total": int(total*2**20), "used": int(used*2**20)}
                    sample = {"seconds": time.perf_counter()-started, "process_rss": mem.rss,
                              "process_peak_wset": getattr(mem, "peak_wset", None),
                              "process_private": getattr(mem, "private", None),
                              "system_ram_total": ram.total, "system_ram_available": ram.available,
                              "gpu": gpu}
                    self.samples.append(sample)
                    if ram.available < ram.total*0.05 or (gpu and gpu["used"] > gpu["total"]*0.95):
                        self.error = "95% global RAM/VRAM ceiling crossed; stopped only this test process"
                        process.terminate()
                        break
                except psutil.NoSuchProcess:
                    break
                except Exception as error:
                    self.error = str(error)
                    process.terminate()
                    break
                index += 1
                self.stop.wait(0.5)
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=12)


def run(binary, model, output, predict):
    meta = GGUFFile(model).metadata
    source = meta["tokenizer.chat_template"]
    if hashlib.sha256(source.encode()).hexdigest() != TEMPLATE_SHA:
        raise ValueError("unreviewed Step template")
    tokenizer, template = Tokenizer.from_gguf(model), renderer(source)
    vocab = meta["tokenizer.ggml.tokens"]
    prompts = []
    for text in ("Сколько будет 2 + 2? Ответь одной цифрой.", "Name the first three prime numbers. Give only the numbers."):
        rendered = template.render(messages=[{"role": "user", "content": text}], tools=None,
                                   add_generation_prompt=True, reasoning_effort="low",
                                   bos_token=vocab[meta["tokenizer.ggml.bos_token_id"]],
                                   eos_token=vocab[meta["tokenizer.ggml.eos_token_id"]])
        prompts.append({"user": text, "rendered": rendered, "ids": tokenizer.encode(rendered, parse_special=True)})
    report = {"status": "error", "scope": "full-model native/pinned selected-copy exact comparison; sequential timing observations",
              "model": str(model), "predict": predict, "context": 2048, "batch": 17, "kv": "F32", "mtp": False,
              "engine_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "prompts": prompts, "runs": []}
    report_file = output / "model-report.json"
    def save():
        report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
    save()
    try:
        for mode in ("native", "pinned"):
            directory = output / mode
            directory.mkdir(exist_ok=True)
            monitor = Monitor()
            engine = None
            record = {"copy_mode": mode, "requests": []}
            report["runs"].append(record)
            try:
                print(f"Loading full Step GGUF: copy={mode}", flush=True)
                start = time.perf_counter()
                engine = Engine(binary, model, directory, mode=mode, kv="f32", logits=True, on_start=monitor.start)
                record["ready_seconds"] = time.perf_counter()-start
                record["info"] = engine.info
                print(f"READY {mode}: {record['ready_seconds']:.2f} s", flush=True)
                for index, prompt in enumerate(prompts):
                    result = engine.generate(prompt["ids"], predict, sampling="temperature=0")
                    result["text"] = b"".join(tokenizer.token_bytes(i) for i in result["ids"]).decode("utf8", errors="replace")
                    path = directory / f"prompt-{index}.f32"
                    raw = (directory / "logits.f32").read_bytes()
                    path.write_bytes(raw)
                    result["logits_sha256"] = hashlib.sha256(raw).hexdigest()
                    result["logits_bytes"] = len(raw)
                    assert len(raw) == len(result["ids"])*len(vocab)*4
                    record["requests"].append(result)
                    if mode == "pinned":
                        reference = report["runs"][0]["requests"][index]
                        result["exact_native_ids"] = result["ids"] == reference["ids"]
                        result["exact_native_logits"] = raw == (output / "native" / path.name).read_bytes()
                        if not result["exact_native_ids"] or not result["exact_native_logits"]:
                            raise ValueError(f"native/pinned mismatch on prompt {index}")
                    print(f"{mode} prompt {index}: {len(result['ids'])} tokens, prefill {result['prefill_ms']/1000:.2f}s, generation {result['generation_wall_ms']/1000:.2f}s", flush=True)
                    save()
                # Third request tests A -> B -> A after a full native state clear.
                repeat = engine.generate(prompts[0]["ids"], predict, sampling="temperature=0")
                repeat_raw = (directory/"logits.f32").read_bytes()
                record["A_B_A_exact"] = repeat["ids"] == record["requests"][0]["ids"] and repeat_raw == (directory/"prompt-0.f32").read_bytes()
                if not record["A_B_A_exact"]:
                    raise ValueError("full-model A/B/A differs")
                record["repeat"] = repeat
                engine.close()
                record["exit_code"] = engine.process.returncode
                if record["exit_code"] != 0:
                    raise ValueError("engine did not exit cleanly")
            finally:
                if engine and not engine.stderr.closed:
                    engine.close()
                monitor.close()
                record["memory_samples"] = monitor.samples
                record["monitor_error"] = monitor.error
                record["requests_metrics"] = [json.loads(line.split(" ", 1)[1]) for line in
                    (directory/"stderr.log").read_text(encoding="utf8", errors="replace").splitlines()
                    if line.startswith("STRATA_STEP_REQUEST ")]
                save()
            if monitor.error:
                raise ValueError(monitor.error)
        report["status"] = "pass"
    except Exception as error:
        report["error"] = str(error)
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--predict", type=int, default=32)
    parser.add_argument("--request-reference", type=Path, help="Run tokenizer/sampling/cancel checks against an existing admission report")
    args = parser.parse_args()
    if not 1 <= args.predict <= 256:
        parser.error("predict must be 1..256")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.request_reference:
        report = request_checks(args.engine.resolve(), first_shard(args.gguf), args.output_dir.resolve(), args.request_reference)
    else:
        report = run(args.engine.resolve(), first_shard(args.gguf), args.output_dir.resolve(), args.predict)
    print(report["status"], report.get("error", ""), flush=True)
    return int(report["status"] != "pass")


def request_checks(binary, model, output, reference_path):
    reference = json.loads(reference_path.read_text(encoding="utf8"))
    if reference["status"] != "pass" or reference["engine_sha256"] != hashlib.sha256(binary.read_bytes()).hexdigest():
        raise ValueError("reference must be a passing admission from the same binary")
    if Path(reference["model"]).resolve() != model.resolve():
        raise ValueError("reference is for another GGUF")
    report = {"status": "error", "scope": "full-model tokenizer, seeded sampling, cancellation and subsequent fresh state",
              "reference": str(reference_path), "engine_sha256": reference["engine_sha256"], "cases": []}
    engine, monitor = None, Monitor()
    try:
        engine = Engine(binary, model, output, on_start=monitor.start)
        tokenizer = Tokenizer.from_gguf(model)
        text = reference["prompts"][0]["rendered"]
        for special in (False, True):
            engine.send(f"ENC {int(special)} " + text.encode().hex())
            response = engine.line().split()
            if response[0] != "IDS" or list(map(int,response[1:])) != tokenizer.encode(text,parse_special=special):
                raise ValueError("pipe token IDs differ")
            report["cases"].append({"name": f"native_ENC_special_{special}", "pass": True})
        ids = reference["prompts"][0]["ids"]
        sampling = "temperature=0.7 top_k=32 top_p=0.95 seed=123"
        first = engine.generate(ids,16,sampling=sampling)
        second = engine.generate(ids,16,sampling=sampling)
        if not first["ids"] or first["ids"] != second["ids"]:
            raise ValueError("seeded full-model sampling differs")
        report["cases"].append({"name": "seeded_sampling_repeat", "pass": True, "ids": first["ids"]})
        cancelled = engine.generate(ids,512,cancel=True)
        if cancelled["finish"] != "cancel":
            raise ValueError("STOP did not cancel request")
        report["cases"].append({"name": "STOP", "pass": True, "result": cancelled})
        after = engine.generate(ids,16,sampling="temperature=0")
        expected = reference["runs"][1]["requests"][0]["ids"][:16]
        if after["ids"] != expected:
            raise ValueError("state after cancel differs from greedy reference")
        report["cases"].append({"name": "fresh_greedy_after_cancel", "pass": True, "ids": after["ids"]})
        engine.close()
        if engine.process.returncode:
            raise ValueError("engine exit failed")
        report["status"] = "pass"
    except Exception as error:
        report["error"] = str(error)
    finally:
        if engine and not engine.stderr.closed:
            engine.close()
        monitor.close()
        report["memory_samples"] = monitor.samples
        report["monitor_error"] = monitor.error
        if monitor.error:
            report.update(status="error", error=monitor.error)
    report["case_count"] = len(report["cases"])
    (output/"requests-report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    return report


if __name__ == "__main__":
    raise SystemExit(main())
