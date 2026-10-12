"""Check trace validation and prevent adding nested/missing CUDA measurements."""
import csv
import tempfile
import unittest
from pathlib import Path

from analyze_deepseek4_cpu_trace import summarize


class TraceTests(unittest.TestCase):
    def row(self, **changes):
        keys = "job tokens rows unique_experts input_bytes output_bytes select_us lease_us lease_calls lease_failures queued_us callback_us callback_end_us pack_us prepare_us gate_up_us hidden_quant_us down_us gu_dispatch_us gu_first_worker_us gu_last_worker_us gu_workers down_dispatch_us down_first_worker_us down_last_worker_us down_workers retire_us input_ms callback_stream_ms output_ms join_ms completed gpu_events route_get_us route_wait_us route_reused route_measured".split()
        row = dict.fromkeys(keys, 0)
        row.update(phase="decode", owner="target", layer="blk.0", job=1, tokens=1, rows=1,
                   unique_experts=1, queued_us=10, callback_us=20, callback_end_us=30, completed=1,
                   select_us=10, lease_us=4, route_measured=1)
        row.update(changes)
        return row

    def read(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"trace.tsv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)
            return summarize(path)

    def test_host_trace_does_not_claim_zero_gpu_wait(self):
        item = self.read([self.row()])["phases"]["decode"]
        self.assertIsNone(item["executed_spans"]["join_ms"]["sum"])
        self.assertEqual(item["select_us"], 10)
        self.assertEqual(item["lease_us"], 4)
        self.assertEqual(item["callback_queue_us"], 10)

    def test_gpu_spans_exclude_declined_and_incomplete_jobs(self):
        rows = [self.row(gpu_events=1, join_ms=0.25),
                self.row(job=2, rows=0, gpu_events=1, join_ms=90),
                self.row(job=3, completed=0, gpu_events=1, join_ms=80)]
        item = self.read(rows)["phases"]["decode"]
        self.assertEqual(item["declined"], 1)
        self.assertEqual(item["incomplete"], 1)
        self.assertEqual(item["executed_spans"]["join_ms"]["sum"], 0.25)

    def test_duplicate_replay_job_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate job"):
            self.read([self.row(), self.row()])

    def test_invalid_callback_order_rejected(self):
        with self.assertRaisesRegex(ValueError, "chronology"):
            self.read([self.row(callback_us=40)])

    def test_nonfinite_event_rejected(self):
        with self.assertRaises(ValueError):
            self.read([self.row(join_ms=float("nan"))])


if __name__ == "__main__":
    unittest.main()
