"""Exercises real Linux chrt/taskset/proc and cleanup with an HTTP test double."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scheduler_lab.experiment import run_trial


@unittest.skipUnless(sys.platform == "linux" and shutil.which("chrt") and shutil.which("taskset"), "Linux scheduling tools required")
class LinuxIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        fixture = Path("tests/fixtures/ollama").resolve()
        (self.root / "ollama").write_bytes(fixture.read_bytes())
        (self.root / "ollama").chmod(0o755)
        (self.root / "stress-ng").write_bytes(Path("tests/fixtures/stress-ng").read_bytes())
        (self.root / "stress-ng").chmod(0o755)
        self.environment = patch.dict(os.environ, {"PATH": str(self.root) + os.pathsep + os.environ["PATH"]})
        self.environment.start()
        self.affinity = os.sched_getaffinity(0)
        if len(self.affinity) < 2:
            self.skipTest("Need at least two allowed CPUs")
        workload = {max(self.affinity)}
        os.sched_setaffinity(0, self.affinity - workload)
        self.args = argparse.Namespace(cpu_set=workload, models_dir=str(self.root), timeout=10, settle=0, tokens=16)
        self.lock = {"model": "test:3b", "digest": "test-only-digest", "prompt": "test",
                     "sha256": hashlib.sha256(b"test").hexdigest(), "prompt_eval_count": 128}

    def tearDown(self):
        os.sched_setaffinity(0, self.affinity)
        self.environment.stop()
        self.temp.cleanup()

    def execute(self, policy, timeout=None):
        directory = self.root / policy
        if timeout:
            self.args.timeout = timeout
        trial = {"trial_id": "test-only", "replication": 1, "policy": policy,
                 "contention_pct": 0, "stress_workers": 0}
        row = run_trial(self.args, trial, self.lock, directory)
        log = (directory / "server.log").read_text()
        for pid in re.findall(r"(?:server|runner)=(\d+)", log):
            # A reparented killed runner can briefly remain a zombie until PID 1 reaps it.
            status = Path(f"/proc/{pid}/status")
            if status.exists():
                self.assertIn("Z (zombie)", status.read_text())
        return row, directory

    def test_other_and_batch_apply_to_descendants(self):
        for policy, expected in (("other", 0), ("batch", 3)):
            row, directory = self.execute(policy)
            self.assertEqual(row["status"], "ok", row["error"])
            samples = [json.loads(line) for line in (directory / "proc.jsonl").read_text().splitlines()]
            self.assertTrue(any(len({t["pid"] for t in s["tasks"]}) > 1 for s in samples))
            self.assertTrue(all(t["policy"] == expected for s in samples for t in s["tasks"]))

    def test_watchdog_stops_stalled_server_and_runner(self):
        with patch.dict(os.environ, {"TEST_OLLAMA_HANG": "1"}):
            row, _ = self.execute("other", timeout=1)
        self.assertEqual(row["status"], "failed")
        self.assertIn("watchdog", row["error"])

    def test_fifo_when_host_grants_permission(self):
        permission = subprocess.run(["chrt", "-f", "50", "true"], capture_output=True)
        if permission.returncode:
            self.skipTest("Host does not grant FIFO priority 50")
        row, directory = self.execute("fifo")
        self.assertEqual(row["status"], "ok", row["error"])
        samples = [json.loads(line) for line in (directory / "proc.jsonl").read_text().splitlines()]
        self.assertTrue(all(t["policy"] == 1 and t["priority"] == 50 for s in samples for t in s["tasks"]))

    def test_contended_trial_records_background_cpu_service(self):
        directory = self.root / "contended"
        trial = {"trial_id": "test-only", "replication": 1, "policy": "other",
                 "contention_pct": 100, "stress_workers": 1}
        self.args.settle = 0.1
        self.args.tokens = 64
        row = run_trial(self.args, trial, self.lock, directory)
        self.assertEqual(row["status"], "ok", row["error"])
        self.assertGreater(row["background_cpu_seconds"], 0)
        self.assertTrue((directory / "stress_proc.jsonl").exists())
        self.assertEqual(json.loads((directory / "background.json").read_text())["violations"], [])
        log = (directory / "stress.log").read_text()
        for pid in re.findall(r"stress=(\d+)", log):
            self.assertFalse(Path(f"/proc/{pid}").exists())
