"""Ollama NDJSON transport. Arrival timestamps use a monotonic clock."""

import json
import time
import urllib.request

from .metrics import latency_metrics


def api(base, path, payload=None, timeout=30):
    request = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    # Loopback traffic must not go through a user's HTTP proxy.
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout) as response:
        result = json.load(response)
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


def request_payload(model, prompt, threads, tokens=128, seed=42):
    return {
        "model": model, "prompt": prompt, "raw": True, "stream": True,
        "keep_alive": "10m",
        "options": {"temperature": 0, "seed": seed, "num_predict": tokens,
                    "num_thread": threads, "num_gpu": 0, "num_ctx": 2048},
    }


def generate(base, payload, events_path=None, timeout=180, on_event=None):
    request = urllib.request.Request(base + "/api/generate", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    start = time.perf_counter_ns()
    arrivals, final = [], None
    evidence = events_path.open("w", encoding="utf-8") if events_path else None
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout) as response:
            for line in response:
                if not line.strip():
                    continue
                now = time.perf_counter_ns()
                event = json.loads(line)
                elapsed = (now - start) / 1e6
                if evidence:
                    evidence.write(json.dumps({"arrival_ms": elapsed, "event": event}) + "\n")
                if "error" in event:
                    raise RuntimeError(event["error"])
                if event.get("response"):
                    arrivals.append(elapsed)
                if on_event:
                    on_event(event)
                if event.get("done"):
                    final = event
                    break
    finally:
        if evidence:
            evidence.close()
    if final is None:
        raise RuntimeError("Stream ended without a done event")
    if not arrivals or final.get("eval_count", 0) < 2 or final.get("eval_duration", 0) <= 0:
        raise RuntimeError("Insufficient generated tokens or missing generation timing")
    measured = latency_metrics(arrivals)
    measured.update({
        "wall_seconds": (time.perf_counter_ns() - start) / 1e9,
        "eval_count": final["eval_count"], "prompt_eval_count": final.get("prompt_eval_count"),
        "eval_duration_ns": final["eval_duration"],
        "generation_tps": final["eval_count"] / (final["eval_duration"] / 1e9),
        "chunk_token_count_match": len(arrivals) == final["eval_count"],
        "done_reason": final.get("done_reason"),
    })
    return measured
