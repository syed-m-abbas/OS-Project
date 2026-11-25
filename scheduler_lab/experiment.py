"""Plan and execute isolated, auditable experiments on an owned Ollama daemon."""

from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import sys
import threading
import time

from .linux import ManagedProcess, Telemetry, cpu_text, machine_metadata, snapshot
from .ollama import api, generate, request_payload

POLICIES = {"other": ("-o", 0, 0), "batch": ("-b", 3, 0), "fifo": ("-f", 1, 50)}


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if type(config.get("replications")) is not int or config["replications"] < 1:
        raise ValueError("replications must be a positive integer")
    conditions = config.get("conditions", [])
    if not conditions:
        raise ValueError("At least one condition is required")
    seen = set()
    for condition in conditions:
        policy, percent = condition.get("policy"), condition.get("contention_pct")
        if policy not in POLICIES or type(percent) is not int or percent not in (0, 50, 100, 200):
            raise ValueError("Conditions require policy other/batch/fifo and contention_pct 0/50/100/200")
        if (policy, percent) in seen:
            raise ValueError("Duplicate condition")
        seen.add((policy, percent))
    return config


def make_plan(config, cores):
    rng, plan = random.Random(config.get("seed", 2026)), []
    for replication in range(1, config["replications"] + 1):
        conditions = list(config["conditions"])
        rng.shuffle(conditions)
        for condition in conditions:
            workers = math.ceil(cores * condition["contention_pct"] / 100)
            plan.append({"trial_id": f"trial-{len(plan) + 1:03d}", "replication": replication,
                         **condition, "stress_workers": workers,
                         "actual_worker_density_pct": workers / cores * 100})
    return plan


def model_record(base, model):
    models = api(base, "/api/tags")["models"]
    for record in models:
        if model in (record.get("name"), record.get("model")):
            return record
    raise ValueError(f"Model {model!r} is not installed. Pull the exact tag and retry.")


def calibrate(args):
    prompt = Path(args.prompt).read_text(encoding="utf-8").strip()
    record = model_record(args.url, args.model)
    # Adjust a simple suffix, verifying the tokenizer's actual count at each step.
    suffix_count, visited = 0, set()
    for _ in range(20):
        candidate = prompt + " x" * suffix_count
        payload = request_payload(args.model, candidate, args.threads, tokens=2)
        payload["stream"] = False
        # Avoid prefix-cache effects on calibration counts between candidates.
        payload["keep_alive"] = 0
        result = api(args.url, "/api/generate", payload, timeout=args.timeout)
        count = result.get("prompt_eval_count")
        if not isinstance(count, int) or count <= 0:
            raise ValueError("Ollama did not return a valid prompt_eval_count")
        if count == args.prompt_tokens:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                raise ValueError(f"Refusing to overwrite existing prompt lock: {output}")
            lock = {"model": args.model, "digest": record["digest"], "prompt": candidate,
                    "sha256": hashlib.sha256(candidate.encode()).hexdigest(),
                    "prompt_eval_count": count, "raw": True, "model_details": record.get("details"),
                    "ollama_version": api(args.url, "/api/version")}
            output.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
            return lock
        next_suffix = suffix_count + args.prompt_tokens - count
        if next_suffix < 0 or next_suffix in visited:
            raise ValueError(f"Cannot calibrate this base text (count={count}). Use a shorter/different base prompt.")
        visited.add(suffix_count)
        suffix_count = next_suffix
    raise ValueError("Prompt calibration did not converge")


def available_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def validate_linux(args, plan):
    if sys.platform != "linux":
        raise ValueError("Real trials require Linux or WSL2. Use 'plan' or 'demo' on Windows.")
    for name in ("ollama", "taskset", "chrt", "stress-ng"):
        if not shutil.which(name):
            raise ValueError(f"Missing executable: {name}. See scripts/setup_ubuntu.sh.")
    allowed = os.sched_getaffinity(0)
    if not args.cpu_set <= allowed:
        raise ValueError(f"Workload CPUs must be a subset of {sorted(allowed)}")
    if not allowed - args.cpu_set:
        raise ValueError("Reserve at least one CPU outside the workload for the controller and watchdog")
    if os.sched_getscheduler(0) != os.SCHED_OTHER:
        raise ValueError("Run the controller under SCHED_OTHER")
    if not Path(args.models_dir).is_dir():
        raise ValueError("--models-dir must point to the existing Ollama models directory")
    if any(t["policy"] == "fifo" for t in plan):
        runtime = int(Path("/proc/sys/kernel/sched_rt_runtime_us").read_text())
        period = int(Path("/proc/sys/kernel/sched_rt_period_us").read_text())
        if not 0 < runtime < period:
            raise ValueError("FIFO trials require nonzero RT throttling that reserves normal-task bandwidth")
        probe = subprocess.run(["chrt", "-f", "50", "true"], capture_output=True, text=True)
        if probe.returncode:
            raise ValueError("FIFO priority 50 unavailable. Run the harness with CAP_SYS_NICE/root. " + probe.stderr.strip())


def run_trial(args, trial, lock, directory):
    directory.mkdir()
    (directory / "condition.json").write_text(json.dumps(trial, indent=2) + "\n", encoding="utf-8")
    row = {**trial, "status": "failed", "error": "", "data_kind": "measured"}
    expired = threading.Event()
    start = time.monotonic()
    try:
        with ExitStack() as stack:
            port = available_port()
            base = f"http://127.0.0.1:{port}"
            flag, policy_id, priority = POLICIES[trial["policy"]]
            env = {**os.environ, "OLLAMA_HOST": f"127.0.0.1:{port}",
                   "OLLAMA_MODELS": str(Path(args.models_dir).resolve()),
                   "OLLAMA_NUM_PARALLEL": "1", "OLLAMA_MAX_LOADED_MODELS": "1",
                   "OLLAMA_KEEP_ALIVE": "10m", "CUDA_VISIBLE_DEVICES": "-1",
                   "ROCR_VISIBLE_DEVICES": "-1"}
            command = ["taskset", "-c", cpu_text(args.cpu_set), "chrt", flag, str(priority), "ollama", "serve"]
            (directory / "launch.json").write_text(json.dumps({"command": command,
                "environment": {k: v for k, v in env.items() if k.startswith("OLLAMA_") or k.endswith("VISIBLE_DEVICES")}}, indent=2), encoding="utf-8")
            server = ManagedProcess(command, directory / "server.log", env)
            stack.callback(server.stop)
            # This normal-priority watchdog runs on the reserved control CPUs.
            def expire():
                expired.set()
                try:
                    os.killpg(server.process.pid, 9)
                except ProcessLookupError:
                    pass
            timer = threading.Timer(args.timeout, expire)
            timer.daemon = True
            timer.start()
            stack.callback(timer.cancel)
            while True:
                if server.process.poll() is not None:
                    raise RuntimeError("Private Ollama daemon exited. Inspect server.log")
                try:
                    version = api(base, "/api/version", timeout=1)
                    break
                except (OSError, ValueError):
                    if time.monotonic() - start > min(30, args.timeout):
                        raise RuntimeError("Ollama startup timed out")
                    time.sleep(0.1)
            record = model_record(base, lock["model"])
            if record["digest"] != lock["digest"]:
                raise RuntimeError("Model digest differs from calibrated prompt lock")
            if record.get("details", {}).get("quantization_level") != "Q4_K_M":
                raise RuntimeError("Proposal requires Q4_K_M quantization. Pull the exact model tag.")
            (directory / "model.json").write_text(json.dumps({"version": version, "record": record,
                "show": api(base, "/api/show", {"model": lock["model"]})}, indent=2), encoding="utf-8")
            # A different warmup prompt avoids deliberately priming the measured prompt's prefix cache.
            generate(base, request_payload(lock["model"], "Write eight words about astronomy.",
                     len(args.cpu_set), tokens=8), timeout=args.timeout)
            loaded = api(base, "/api/ps")["models"]
            if len(loaded) != 1 or loaded[0].get("digest") != lock["digest"] or loaded[0].get("size_vram") != 0:
                raise RuntimeError("CPU-only loaded-model verification failed")
            (directory / "loaded_model.json").write_text(json.dumps(loaded, indent=2), encoding="utf-8")
            tasks = snapshot(server.process.pid)
            if not tasks or not any(t["pid"] != server.process.pid for t in tasks):
                raise RuntimeError("No descendant inference runner found after warmup")
            stress = None
            stress_telemetry = None
            if trial["stress_workers"]:
                stress = ManagedProcess(["taskset", "-c", cpu_text(args.cpu_set), "chrt", "-o", "0",
                    "stress-ng", "--cpu", str(trial["stress_workers"]), "--cpu-method", "matrixprod",
                    "--timeout", f"{math.ceil(args.timeout + 10)}s", "--metrics-brief"], directory / "stress.log")
                stack.callback(stress.stop)
                time.sleep(args.settle)
                if stress.process.poll() is not None:
                    raise RuntimeError("stress-ng exited before the measured request")
                stress_telemetry = Telemetry(stress.process.pid, directory / "stress_proc.jsonl",
                                            args.cpu_set, 0, 0)
                stress_telemetry.start()
                stack.callback(stress_telemetry.stop)
            telemetry = Telemetry(server.process.pid, directory / "proc.jsonl", args.cpu_set, policy_id, priority)
            telemetry.start()
            stack.callback(telemetry.stop)
            begin = time.perf_counter()
            metrics = generate(base, request_payload(lock["model"], lock["prompt"], len(args.cpu_set),
                               tokens=args.tokens), directory / "events.jsonl", timeout=args.timeout)
            seconds = time.perf_counter() - begin
            telemetry.sample()
            # Read final snapshots before context managers stop the workers.
            if stress_telemetry:
                stress_telemetry.sample()
            if metrics["prompt_eval_count"] != lock["prompt_eval_count"]:
                raise RuntimeError("Measured prompt token count differs from prompt lock")
            if not metrics["chunk_token_count_match"] or metrics["gap_count"] < 2:
                raise RuntimeError("Stream chunks do not match token count, or fewer than three token chunks arrived")
            if stress and stress.process.poll() is not None:
                raise RuntimeError("stress-ng exited during measurement")
            row.update(metrics)
            row.update({"measurement_seconds": seconds, "prompt_sha256": lock["sha256"], "model_digest": lock["digest"],
                        "workload_cpus": sorted(args.cpu_set), "threads": len(args.cpu_set)})
        # stop callbacks sample and join telemetry, so inspect their final state here.
        context = telemetry.summary(seconds)
        (directory / "context_switches.json").write_text(json.dumps(context, indent=2), encoding="utf-8")
        if context["violations"] or context["errors"]:
            raise RuntimeError("Inference thread audit failed: " + repr(context))
        row.update({"voluntary_switches": context["voluntary"], "involuntary_switches": context["involuntary"],
                    "voluntary_per_s": context["voluntary_per_s"], "involuntary_per_s": context["involuntary_per_s"],
                    "threads_seen": context["threads_seen"], "threads_exited": context["threads_exited"]})
        if stress_telemetry:
            background = stress_telemetry.summary(seconds)
            (directory / "background.json").write_text(json.dumps(background, indent=2), encoding="utf-8")
            if background["violations"] or background["errors"]:
                raise RuntimeError("Background thread audit failed")
            row["background_cpu_seconds"] = background["cpu_ticks"] / os.sysconf("SC_CLK_TCK")
            row["background_capacity_pct"] = 100 * row["background_cpu_seconds"] / (seconds * len(args.cpu_set))
        row["status"] = "ok"
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
        if expired.is_set():
            row["error"] = "Trial exceeded total watchdog deadline. " + row["error"]
    finally:
        row["trial_wall_seconds"] = time.monotonic() - start
        (directory / "result.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    return row


def run_matrix(args):
    config = load_config(args.config)
    plan = make_plan(config, len(args.cpu_set))
    validate_linux(args, plan)
    lock = json.loads(Path(args.prompt_lock).read_text(encoding="utf-8"))
    if hashlib.sha256(lock["prompt"].encode()).hexdigest() != lock["sha256"]:
        raise ValueError("Prompt lock checksum mismatch")
    if lock.get("prompt_eval_count") != 128 or lock.get("raw") is not True:
        raise ValueError("Proposal trials require a calibrated raw 128-token prompt")
    output = Path(args.output)
    if output.exists():
        raise ValueError("Use a new output directory for each experiment to preserve raw evidence")
    output.mkdir(parents=True)
    metadata = machine_metadata()
    metadata.update({"data_kind": "measured", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config, "arguments": {k: (sorted(v) if isinstance(v, set) else v) for k, v in vars(args).items() if k != "handler"},
        "plan": plan, "prompt_lock": lock})
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    original_affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, original_affinity - args.cpu_set)
    failures = 0
    try:
        for trial in plan:
            print(f"{trial['trial_id']}: {trial['policy']} / {trial['contention_pct']}%", flush=True)
            result = run_trial(args, trial, lock, output / trial["trial_id"])
            with (output / "trials.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result) + "\n")
            if result["status"] != "ok":
                failures += 1
                print(result["error"], file=sys.stderr, flush=True)
                if not args.keep_going:
                    break
            time.sleep(args.cooldown)
    finally:
        os.sched_setaffinity(0, original_affinity)
    return 1 if failures else 0
