"""Benchmarking package."""

from __future__ import annotations

from greenai.benchmark.memory import MemoryResult, measure_memory
from greenai.benchmark.runner import BenchmarkConfig, BenchmarkResult, run_benchmark
from greenai.benchmark.statistics import LatencyStats, compute_latency_stats

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "LatencyStats",
    "MemoryResult",
    "compute_latency_stats",
    "measure_memory",
    "run_benchmark",
]
