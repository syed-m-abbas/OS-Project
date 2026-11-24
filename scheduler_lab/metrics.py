"""Explicit, dependency-free metric definitions shared by collection and analysis."""

import math
import statistics


def percentile(values, q):
    """Linear interpolation, equivalent to NumPy's default percentile method."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def latency_metrics(arrivals_ms):
    gaps = [b - a for a, b in zip(arrivals_ms, arrivals_ms[1:])]
    return {
        "ttft_ms": arrivals_ms[0] if arrivals_ms else None,
        "chunks": len(arrivals_ms),
        "gap_count": len(gaps),
        "itl_mean_ms": statistics.mean(gaps) if gaps else None,
        "itl_std_ms": statistics.stdev(gaps) if len(gaps) > 1 else None,
        "itl_p50_ms": percentile(gaps, 50),
        "itl_p95_ms": percentile(gaps, 95),
        "itl_p99_ms": percentile(gaps, 99),
        "tail_jitter_ms": (percentile(gaps, 99) - percentile(gaps, 50)) if gaps else None,
    }
