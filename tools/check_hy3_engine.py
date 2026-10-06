"""Exercise the Hy3 pipe process on its synthetic fixture or real GGUF."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time


class Engine:
    def __init__(self, binary, model, directory, mode="pinned", kv="f32", logits=False, on_start=None, extra_args=()):
        self.command = [str(binary), "--native", str(model), "--max-context", "2048", "--batch-size", "17",
                        "--copy-mode", mode, "--kv", kv, "--serve"]
        self.command += list(extra_args)
        if logits:
            self.command += ["--logits-file", str(directory / "logits.f32")]
        self.stderr_path = directory / "stderr.log"
        self.stderr = self.stderr_path.open("w", encoding="utf8")
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.stderr, text=True, encoding="utf8", bufsize=1,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.lines = queue.Queue()
        if on_start is not None:
            on_start(self.process)
        self.pump = threading.Thread(target=self._read, daemon=True)
        self.pump.start()
        self.info = []
        try:
            while True:
                line = self.line(600)
                self.info.append(line)
                if line.startswith("READY "):
                    break
        except BaseException:
            self.close()
            raise

    def _read(self):
        for line in self.process.stdout:
            self.lines.put(line.strip())
        self.lines.put(None)

    def line(self, timeout=600):
        line = self.lines.get(timeout=timeout)
        if line is None:
            raise RuntimeError(f"engine exited: {self.process.poll()}; {self.stderr_path}")
        return line

    def send(self, line):
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()

    def generate(self, tokens, count=16, sampling="", cancel=False, on_progress=None):
        start = time.perf_counter()
        self.send(f"GEN {count} {sampling} " + ",".join(map(str, tokens)))
        output, progress = [], []
        first = None
        while True:
            line = self.line()
            if line.startswith("PP "):
                progress.append(line)
                if on_progress is not None:
                    on_progress(int(line.split()[1]), len(tokens))
                if cancel:
                    self.send("STOP")
                    cancel = False
            elif line.startswith("T "):
                output.append(int(line[2:]))
                if first is None:
                    first = time.perf_counter() - start
            elif line.startswith("DONE "):
                fields = line.split()
                assert int(fields[1]) == len(output) and int(fields[2]) == len(tokens)
                return {"ids": output, "done": line, "finish": fields[5], "prefill_ms": float(fields[3]),
                        "generation_wall_ms": float(fields[4]), "ttft_seconds": first,
                        "wall_seconds": time.perf_counter() - start, "progress": progress}
            else:
                raise RuntimeError(f"unexpected generation response: {line}")

    def close(self):
        if self.process.poll() is None:
            try:
                self.send("QUIT")
                self.process.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait(timeout=10)
        # A memory-guard termination may leave a broken pipe. Preserve the
        # original execution error and let callers save the monitor evidence.
        for stream in (self.stderr, self.process.stdin, self.process.stdout):
            try:
                stream.close()
            except OSError:
                pass


def fixture(binary, model, output, extra_args=()):
    report = {"status": "error", "scope": "synthetic pipe lifecycle, sampling, isolation and cancellation", "cases": []}
    engine = None
    try:
        engine = Engine(binary, model, output, extra_args=["--fixture", *extra_args])
        assert "architecture=hy_v3" in engine.info[0] and "gpu_only=1" in engine.info[0]
        report["info"] = engine.info
        def passed(name):
            report["cases"].append({"name": name, "pass": True})
        a = engine.generate([11,18,25,32], sampling="temperature=0")
        assert len(a["ids"]) == 16
        passed("greedy_16")
        engine.generate([10,21,31,44], sampling="temperature=0")
        assert engine.generate([11,18,25,32], sampling="temperature=0")["ids"] == a["ids"]
        passed("A_B_A_greedy")
        sampling = "temperature=0.7 top_k=16 top_p=0.95 seed=123 penalty_repeat=1.05"
        sampled = engine.generate([11,18,25,32], sampling=sampling)
        assert engine.generate([11,18,25,32], sampling=sampling)["ids"] == sampled["ids"]
        passed("seeded_sampling_repeat")
        for invalid in ("", "GEN", "GEN 0 1", "GEN 2048 1", "GEN 1 64", "GEN 1 -1", "GEN 1 1,", "GEN 1 1,,2",
                        "GEN 1 temperature=nan 1", "GEN 1 seed=4294967296 1", "GEN 1 top_p=0 1", "GEN 1 unknown=1 1",
                        "GEN 1 session=bad 1", "ENC 2 00", "ENC 1 gg"):
            engine.send(invalid)
            assert engine.line().startswith("ERR ")
            passed("reject/" + invalid)
        result = engine.generate([i % 64 for i in range(1024)], count=512, cancel=True)
        assert result["finish"] == "cancel" and len(result["ids"]) < 512
        passed("stop_during_prefill")
        assert engine.generate([11,18,25,32], sampling="temperature=0")["ids"] == a["ids"]
        passed("fresh_state_after_cancel_and_errors")
        session = "session=" + "a"*64
        assert engine.generate([11,18,25,32], sampling=session)["ids"] == a["ids"]
        passed("session_key_isolation")
        engine.send("GEN 512 " + ",".join(str(i % 64) for i in range(512)))
        assert engine.line().startswith("PP ")
        engine.close()
        assert engine.process.returncode == 0
        passed("quit_during_request")
        report.update(status="pass", exit_code=engine.process.returncode, greedy_ids=a["ids"], sampled_ids=sampled["ids"])
    except Exception as error:
        report["error"] = str(error)
    finally:
        if engine is not None and not engine.stderr.closed:
            engine.close()
    report["case_count"] = len(report["cases"])
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine',type=Path,required=True)
    p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    report=fixture(args.engine.resolve(),args.fixture.resolve(),args.output_dir.resolve())
    report['engine_sha256']=hashlib.sha256(args.engine.read_bytes()).hexdigest()
    report['fixture_sha256']=hashlib.sha256(args.fixture.read_bytes()).hexdigest()
    (args.output_dir/'pipe-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(report['status'],report['case_count'],'checks',report.get('error',''))
    return int(report['status']!='pass')

if __name__=='__main__':
    raise SystemExit(main())
