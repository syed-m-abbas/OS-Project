"""Synthetic fixtures for software verification. These do not test any hypothesis."""

import json
import random

from .experiment import load_config, make_plan
from .metrics import latency_metrics


def create_demo(output):
    if output.exists():
        raise ValueError("Use a new demo output directory")
    config = load_config("configs/proposal.json")
    plan = make_plan(config, 3)
    output.mkdir(parents=True)
    metadata = {"data_kind": "synthetic", "warning": "Synthetic software demonstration. No inference measurements.",
                "plan": plan, "config": config}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    rng = random.Random(7)
    with (output / "trials.jsonl").open("w", encoding="utf-8") as handle:
        for trial in plan:
            # Policy is deliberately absent from this toy latency model.
            load = trial["contention_pct"] / 100
            gaps = [rng.lognormvariate(3.2 + load * 0.15, 0.15 + load * 0.08) for _ in range(127)]
            arrivals = [rng.uniform(150, 250) * (1 + load * 0.1)]
            for gap in gaps:
                arrivals.append(arrivals[-1] + gap)
            row = {**trial, **latency_metrics(arrivals), "data_kind": "synthetic", "status": "ok", "error": "",
                   "eval_count": 128, "generation_tps": 1000 / (sum(gaps) / len(gaps)),
                   "involuntary_per_s": rng.uniform(4, 12) * (1 + load), "voluntary_per_s": rng.uniform(30, 50)}
            directory = output / trial["trial_id"]
            directory.mkdir()
            events = [{"arrival_ms": a, "event": {"response": "x", "done": False}} for a in arrivals]
            events.append({"arrival_ms": arrivals[-1] + 1, "event": {"response": "", "done": True, "eval_count": 128}})
            (directory / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
            (directory / "result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            handle.write(json.dumps(row) + "\n")
    print(f"Wrote 35 SYNTHETIC demo trials to {output}")
