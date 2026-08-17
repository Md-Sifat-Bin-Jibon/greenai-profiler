"""Throughput helpers."""

from __future__ import annotations

from greenai.benchmark.statistics import LatencyStats


def throughput_from_latency(stats: LatencyStats) -> float:
    """Return samples/sec derived from mean latency and batch size."""
    return stats.throughput_samples_per_sec
