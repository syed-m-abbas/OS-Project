"""Linux process supervision and per-thread /proc telemetry."""

import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import threading
import time


def parse_cpus(text):
    cpus = set()
    for part in text.split(","):
        bounds = part.strip().split("-")
        if len(bounds) == 1:
            cpus.add(int(bounds[0]))
        elif len(bounds) == 2:
            lo, hi = map(int, bounds)
            if hi < lo:
                raise ValueError("Descending CPU range")
            cpus.update(range(lo, hi + 1))
        else:
            raise ValueError("Invalid CPU list")
    if not cpus or min(cpus) < 0:
        raise ValueError("CPU IDs must be nonnegative")
    return cpus


def cpu_text(cpus):
    return ",".join(map(str, sorted(cpus)))


def process_tree(root):
    pending, found = [root], set()
    while pending:
        pid = pending.pop()
        if pid in found:
            continue
        found.add(pid)
        try:
            for task in Path(f"/proc/{pid}/task").iterdir():
                try:
                    pending.extend(map(int, (task / "children").read_text().split()))
                except (FileNotFoundError, ProcessLookupError):
                    pass
        except (FileNotFoundError, ProcessLookupError):
            pass
    return found


def snapshot(root):
    rows = []
    for pid in sorted(process_tree(root)):
        try:
            tasks = list(Path(f"/proc/{pid}/task").iterdir())
        except FileNotFoundError:
            continue
        for task in tasks:
            try:
                tid = int(task.name)
                status = dict(line.split(":", 1) for line in (task / "status").read_text().splitlines() if ":" in line)
                # comm can include spaces and parentheses. Fields after the last ')' start at field 3.
                stat = (task / "stat").read_text().rsplit(")", 1)[1].split()
                row = {"pid": pid, "tid": tid, "starttime": int(stat[19]),
                       "voluntary": int(status["voluntary_ctxt_switches"]),
                       "involuntary": int(status["nonvoluntary_ctxt_switches"]),
                       "cpu_ticks": int(stat[11]) + int(stat[12]),
                       "policy": os.sched_getscheduler(tid),
                       "priority": os.sched_getparam(tid).sched_priority,
                       "cpus": sorted(os.sched_getaffinity(tid))}
                rows.append(row)
            except (FileNotFoundError, ProcessLookupError):
                continue
    return rows


class Telemetry:
    """Accumulate deltas by (TID, starttime), retaining samples of exited threads."""

    def __init__(self, root, path, cpus, policy, priority, interval=0.2):
        self.root, self.path, self.cpus = root, path, set(cpus)
        self.policy, self.priority, self.interval = policy, priority, interval
        self.previous, self.totals = {}, dict(voluntary=0, involuntary=0, cpu_ticks=0)
        self.violations, self.errors = set(), []
        self.seen, self.exited = set(), set()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.handle = None
        self.lock = threading.Lock()

    def sample(self):
        with self.lock:
            rows = snapshot(self.root)
            active = set()
            for row in rows:
                key = (row["tid"], row["starttime"])
                active.add(key)
                if row["policy"] != self.policy or row["priority"] != self.priority or set(row["cpus"]) != self.cpus:
                    self.violations.add(f"pid={row['pid']} tid={row['tid']} policy/priority/affinity mismatch")
                if key in self.previous:
                    for field in self.totals:
                        self.totals[field] += max(0, row[field] - self.previous[key][field])
                self.previous[key] = row
            self.exited.update(self.seen - active)
            self.seen.update(active)
            if self.handle:
                self.handle.write(json.dumps({"monotonic_ns": time.perf_counter_ns(), "tasks": rows}) + "\n")
                self.handle.flush()
            return rows

    def _loop(self):
        while not self.stop_event.wait(self.interval):
            try:
                self.sample()
            except Exception as error:
                self.errors.append(repr(error))
                break

    def start(self):
        self.handle = self.path.open("w", encoding="utf-8")
        try:
            self.sample()
            self.thread.start()
        except BaseException:
            self.handle.close()
            raise

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        try:
            self.sample()
        finally:
            self.handle.close()

    def summary(self, seconds):
        return {**self.totals, "threads_seen": len(self.seen), "threads_exited": len(self.exited),
                "voluntary_per_s": self.totals["voluntary"] / seconds,
                "involuntary_per_s": self.totals["involuntary"] / seconds,
                "violations": sorted(self.violations), "errors": self.errors}


class ManagedProcess:
    def __init__(self, command, log_path, env=None):
        self.log = log_path.open("wb")
        try:
            self.process = subprocess.Popen(command, env=env, stdout=self.log, stderr=subprocess.STDOUT,
                                            start_new_session=True)
        except BaseException:
            self.log.close()
            raise

    def stop(self):
        # Also kill surviving descendants if the process-group leader has exited.
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.process.wait(timeout=5)
        self.log.close()


def machine_metadata():
    paths = ["/proc/cpuinfo", "/proc/meminfo", "/proc/sys/kernel/sched_rt_runtime_us",
             "/proc/sys/kernel/sched_rt_period_us", "/proc/self/cgroup"]
    result = {"platform": platform.platform(), "kernel": platform.release(),
              "python": platform.python_version(), "clock_ticks": os.sysconf("SC_CLK_TCK")}
    for name in paths:
        try:
            result[name] = Path(name).read_text()
        except OSError as error:
            result[name] = str(error)
    result["governors"] = {str(p): p.read_text().strip() for p in Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_governor")}
    result["topology"] = {str(p): p.read_text().strip() for p in Path("/sys/devices/system/cpu").glob("cpu*/topology/thread_siblings_list")}
    for name, command in {"ollama_version": ["ollama", "--version"],
                          "stress_ng_version": ["stress-ng", "--version"],
                          "util_linux_version": ["chrt", "--version"],
                          "git_revision": ["git", "rev-parse", "HEAD"]}.items():
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=10)
            result[name] = {"returncode": completed.returncode,
                            "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip()}
        except (OSError, subprocess.TimeoutExpired) as error:
            result[name] = str(error)
    return result
