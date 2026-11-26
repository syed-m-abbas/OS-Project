"""Trial-level summaries, paired inference, and publication figures."""

import csv
import itertools
import json
import math
from pathlib import Path
import statistics

from .metrics import latency_metrics, percentile

METRICS = ["ttft_ms", "generation_tps", "itl_p99_ms", "tail_jitter_ms", "involuntary_per_s", "voluntary_per_s"]


def paired_permutation_p(differences):
    """Exact two-sided sign-flip test on independent replication-level differences."""
    if not differences:
        return None
    if len(differences) > 20:
        raise ValueError("Exact sign-flip test is limited to 20 pairs")
    observed = abs(statistics.mean(differences))
    count = sum(abs(statistics.mean(d * s for d, s in zip(differences, signs))) >= observed - 1e-12
                for signs in itertools.product((-1, 1), repeat=len(differences)))
    return count / 2 ** len(differences)


def holm_adjust(pvalues):
    ordered = sorted(enumerate(pvalues), key=lambda item: item[1])
    adjusted, previous = [None] * len(pvalues), 0
    for rank, (index, value) in enumerate(ordered):
        previous = max(previous, min(1.0, value * (len(pvalues) - rank)))
        adjusted[index] = previous
    return adjusted


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze_results(source, output):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        raise RuntimeError("Install analysis dependencies: python -m pip install -e '.[analysis]'") from error
    if output.exists():
        raise ValueError("Use a new analysis output directory")
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    kind = metadata.get("data_kind")
    if kind not in ("measured", "synthetic"):
        raise ValueError("Input must identify its measured or synthetic provenance")
    raw = [json.loads(line) for line in (source / "trials.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    declared_plan = {t["trial_id"]: t for t in metadata.get("plan", [])}
    valid, gaps_by_trial, excluded, identifiers, pairs = [], {}, [], set(), set()
    for row in raw:
        if row["trial_id"] in identifiers:
            raise ValueError("Duplicate trial ID in raw results")
        identifiers.add(row["trial_id"])
        if declared_plan:
            expected_condition = declared_plan.get(row["trial_id"])
            if expected_condition is None or any(row.get(k) != expected_condition.get(k) for k in ("replication", "policy", "contention_pct", "stress_workers")):
                raise ValueError("Trial condition differs from the recorded plan")
        if row.get("data_kind") != kind:
            raise ValueError("Mixed or missing data provenance")
        if row["status"] != "ok":
            excluded.append({"trial_id": row["trial_id"], "reason": row.get("error", "failed")})
            continue
        pair = (row["policy"], row["contention_pct"], row["replication"])
        if pair in pairs:
            raise ValueError("Duplicate condition within a replication block")
        pairs.add(pair)
        events = [json.loads(line) for line in (source / row["trial_id"] / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        arrivals = [e["arrival_ms"] for e in events if e["event"].get("response")]
        if any(not math.isfinite(x) or x < 0 for x in arrivals) or any(b < a for a, b in zip(arrivals, arrivals[1:])):
            raise ValueError("Invalid event timestamps")
        computed = latency_metrics(arrivals)
        if len(arrivals) != row["eval_count"] or len(arrivals) < 3:
            raise ValueError("A valid trial must have at least three chunks matching the token count")
        for metric in ("ttft_ms", "itl_p99_ms", "tail_jitter_ms"):
            if not math.isclose(computed[metric], row[metric], rel_tol=1e-8, abs_tol=1e-6):
                raise ValueError(f"Raw event/summary mismatch: {row['trial_id']} {metric}")
        for metric in METRICS:
            if not isinstance(row.get(metric), (int, float)) or not math.isfinite(row[metric]) or row[metric] < 0:
                raise ValueError(f"Invalid metric {metric}")
        valid.append(row)
        gaps_by_trial[row["trial_id"]] = [b - a for a, b in zip(arrivals, arrivals[1:])]
    if not valid:
        raise ValueError("No valid trials to analyze")
    if kind == "measured" and len({(r["model_digest"], r["prompt_sha256"], tuple(r["workload_cpus"]), r["threads"]) for r in valid}) != 1:
        raise ValueError("Controlled model/prompt/CPU settings vary within this experiment")
    output.mkdir(parents=True)
    rng = np.random.default_rng(2026)
    groups = {}
    for row in valid:
        groups.setdefault((row["policy"], row["contention_pct"]), []).append(row)
    summaries = []
    for (policy, contention), rows in sorted(groups.items()):
        summary = {"policy": policy, "contention_pct": contention, "n": len(rows), "data_kind": kind}
        for metric in METRICS:
            values = np.array([r[metric] for r in rows])
            boot = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
            summary.update({metric + "_mean": float(values.mean()),
                metric + "_std": float(values.std(ddof=1)) if len(values) > 1 else None,
                metric + "_ci_low": float(np.percentile(boot, 2.5)),
                metric + "_ci_high": float(np.percentile(boot, 97.5))})
        summaries.append(summary)
    comparisons = []
    for percent in (100, 200):
        baseline = {r["replication"]: r for r in groups.get(("other", percent), [])}
        for policy in ("batch", "fifo"):
            alternative = {r["replication"]: r for r in groups.get((policy, percent), [])}
            common = sorted(baseline.keys() & alternative.keys())
            if not common:
                continue
            differences = [alternative[i]["itl_p99_ms"] - baseline[i]["itl_p99_ms"] for i in common]
            reductions = [(1 - alternative[i]["itl_p99_ms"] / baseline[i]["itl_p99_ms"]) * 100
                          for i in common if baseline[i]["itl_p99_ms"] > 0]
            comparisons.append({"policy": policy, "contention_pct": percent, "metric": "itl_p99_ms",
                "pairs": len(common), "mean_difference_ms": statistics.mean(differences),
                "mean_paired_reduction_pct": statistics.mean(reductions) if reductions else None,
                "p_exact_two_sided": paired_permutation_p(differences), "data_kind": kind})
    for comparison, adjusted in zip(comparisons, holm_adjust([c["p_exact_two_sided"] for c in comparisons])):
        comparison["p_holm"] = adjusted
    expected = len(metadata.get("plan", []))
    planned_ids = {t["trial_id"] for t in metadata.get("plan", [])}
    complete = bool(expected) and len(valid) == expected and identifiers == planned_ids and not excluded
    write_csv(output / "trials.csv", [{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in raw])
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "comparisons.csv", comparisons)
    report = {"data_kind": kind, "complete_matrix": complete, "planned_trials": expected,
        "valid_trials": len(valid), "excluded": excluded, "missing_trial_ids": sorted(planned_ids - identifiers),
        "summaries": summaries, "comparisons": comparisons,
        "notes": ["Trial p99 values are the independent observations, not pooled tokens.",
                  "Bootstrap intervals use 10000 trial resamples and are unstable with five trials.",
                  "With five pairs, the smallest two-sided exact sign-flip p-value is 0.0625.",
                  "Holm adjustment covers the available four primary p99 comparisons.",
                  "Failed runs are listed. Successful outliers are retained."]}
    (output / "analysis.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 300})
    colors = {"other": "#355C7D", "batch": "#22856D", "fifo": "#B85A3A"}
    def finish(fig, name):
        if kind == "synthetic":
            fig.text(0.5, 0.01, "SYNTHETIC DEMO DATA - NO RESEARCH FINDINGS", ha="center", color="#a32323", fontsize=9)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(output / (name + ".png"))
        fig.savefig(output / (name + ".pdf"))
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    levels = sorted({s["contention_pct"] for s in summaries})
    for offset, policy in enumerate(("other", "batch", "fifo")):
        available = [s for s in summaries if s["policy"] == policy]
        x = [levels.index(s["contention_pct"]) + (offset - 1) * 0.24 for s in available]
        means = [s["generation_tps_mean"] for s in available]
        errors = [[s["generation_tps_mean"] - s["generation_tps_ci_low"] for s in available],
                  [s["generation_tps_ci_high"] - s["generation_tps_mean"] for s in available]]
        ax.bar(x, means, width=0.23, yerr=errors, capsize=3, color=colors[policy], label=policy.upper())
    ax.set(xticks=range(len(levels)), xticklabels=levels, xlabel="Nominal contention (%)", ylabel="Generation throughput (tokens/s)")
    ax.legend()
    finish(fig, "throughput")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharey=True)
    for ax, level in zip(axes, (100, 200)):
        for policy in ("other", "batch", "fifo"):
            gaps = sorted(g for r in groups.get((policy, level), []) for g in gaps_by_trial[r["trial_id"]])
            if gaps:
                ax.step(gaps, np.arange(1, len(gaps) + 1) / len(gaps), where="post", label=policy.upper(), color=colors[policy])
        ax.set(title=f"{level}% contention", xlabel="Inter-token arrival gap (ms)")
    axes[0].set_ylabel("Empirical cumulative probability")
    axes[-1].legend(fontsize=8)
    finish(fig, "jitter_cdf")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for policy in ("other", "batch", "fifo"):
        rows = sorted([s for s in summaries if s["policy"] == policy], key=lambda s: s["contention_pct"])
        ax.errorbar([s["contention_pct"] for s in rows], [s["involuntary_per_s_mean"] for s in rows],
                    yerr=[s["involuntary_per_s_std"] or 0 for s in rows], marker="o", capsize=3,
                    label=policy.upper(), color=colors[policy])
    ax.set(xlabel="Nominal contention (%)", ylabel="Involuntary context switches / s (sampled)")
    ax.legend()
    finish(fig, "context_switches")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for policy in ("other", "batch", "fifo"):
        rows = sorted([s for s in summaries if s["policy"] == policy], key=lambda s: s["contention_pct"])
        ax.errorbar([s["contention_pct"] for s in rows], [s["ttft_ms_mean"] for s in rows],
                    yerr=[s["ttft_ms_std"] or 0 for s in rows], marker="o", capsize=3,
                    color=colors[policy], label=policy.upper())
    ax.set(xlabel="Nominal contention (%)", ylabel="Time to first response token (ms)")
    ax.legend()
    finish(fig, "ttft")
    # Fixed numeric cells only, so raw prompt/model strings cannot inject LaTeX.
    lines = [r"\begin{tabular}{llrrr}", r"\hline", r"Policy & Load & $n$ & TTFT (ms) & p99 (ms) \\", r"\hline"]
    for s in summaries:
        lines.append(f"{s['policy'].upper()} & {s['contention_pct']}\\% & {s['n']} & {s['ttft_ms_mean']:.1f} & {s['itl_p99_ms_mean']:.1f} " + r"\\")
    lines.extend([r"\hline", r"\end{tabular}"])
    if kind == "synthetic":
        lines.insert(0, r"\textbf{Synthetic demonstration only.}\par")
    (output / "results_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Analyzed {len(valid)}/{expected} valid {kind} trials. Complete matrix: {complete}")

