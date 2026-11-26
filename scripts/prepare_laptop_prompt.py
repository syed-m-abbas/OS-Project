"""Prepare a natural 128-token raw Llama instruction, without repetitive padding."""
import json
from pathlib import Path
from scheduler_lab.ollama import api, request_payload

words = ("Explain how an operating system shares CPU time between runnable programs. "
         "Describe fairness response time throughput and context switching. Compare normal "
         "batch and real time scheduling with clear practical examples. Discuss how background "
         "computation changes interactive responsiveness and why processor affinity matters. "
         "Include limitations concerning shared caches memory bandwidth and virtual machines. "
         "Write a detailed continuous explanation suitable for an undergraduate operating systems "
         "course. Use complete sentences and focus on the relationship between scheduling "
         "decisions and observable application latency. Explain the tradeoffs carefully. "
         "Describe why a fast average can hide long pauses during token generation. "
         "Connect each example to a measurable outcome and distinguish observations "
         "from assumptions about the underlying hardware and software behavior.").split()
prefix = "<|start_header_id|>user<|end_header_id|>\n\n"
suffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
best = None
for n in range(len(words), 0, -1):
    candidate = prefix + " ".join(words[:n]) + suffix
    payload = request_payload("llama3.2:3b-instruct-q4_K_M", candidate, 3, tokens=2)
    payload.update(stream=False, keep_alive=0)
    result = api("http://127.0.0.1:11434", "/api/generate", payload, timeout=300)
    count = result["prompt_eval_count"]
    print(f"words={n} tokens={count}", flush=True)
    if count == 128:
        best = candidate
        break
    if count < 128:
        raise RuntimeError("Token count skipped 128; adjust the natural wording and rerun")
if best is None:
    raise RuntimeError("No exact prompt found")
Path("prompts/laptop-instruct-128.txt").write_text(best, encoding="utf-8")
