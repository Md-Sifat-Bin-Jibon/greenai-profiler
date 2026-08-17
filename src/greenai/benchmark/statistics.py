"""Benchmark statistics helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel, Field


class LatencyStats(BaseModel):
    """Latency and throughput summary for a set of timed iterations."""

    iterations: int
    warmup: int
    batch_size: int
    mean_seconds: float
    median_seconds: float
    p95_seconds: float
    p99_seconds: float
    min_seconds: float
    max_seconds: float
    stddev_seconds: float
    throughput_samples_per_sec: float
    raw_seconds: list[float] = Field(default_factory=list, repr=False)


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute percentile of empty sequence.")
    if pct <= 0:
        return sorted_values[0]
    if pct >= 100:
        return sorted_values[-1]
    k = (len(sorted_values) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return d0 + d1


def compute_latency_stats(
    samples: Sequence[float],
    *,
    warmup: int,
    batch_size: int,
    keep_raw: bool = False,
) -> LatencyStats:
    """Compute latency/throughput statistics from per-iteration seconds."""
    if not samples:
        raise ValueError("At least one timed sample is required.")
    ordered = sorted(samples)
    mean = sum(samples) / len(samples)
    variance = sum((x - mean) ** 2 for x in samples) / len(samples)
    stddev = math.sqrt(variance)
    median = _percentile(ordered, 50)
    mean_latency = mean
    throughput = (batch_size / mean_latency) if mean_latency > 0 else 0.0
    return LatencyStats(
        iterations=len(samples),
        warmup=warmup,
        batch_size=batch_size,
        mean_seconds=mean,
        median_seconds=median,
        p95_seconds=_percentile(ordered, 95),
        p99_seconds=_percentile(ordered, 99),
        min_seconds=ordered[0],
        max_seconds=ordered[-1],
        stddev_seconds=stddev,
        throughput_samples_per_sec=throughput,
        raw_seconds=list(samples) if keep_raw else [],
    )
