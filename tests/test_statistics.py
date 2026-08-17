"""Benchmark statistics tests."""

from __future__ import annotations

import pytest

from greenai.benchmark.statistics import compute_latency_stats


def test_compute_latency_stats() -> None:
    samples = [0.01, 0.02, 0.03, 0.04, 0.05]
    stats = compute_latency_stats(samples, warmup=2, batch_size=2)
    assert stats.iterations == 5
    assert stats.warmup == 2
    assert stats.batch_size == 2
    assert stats.min_seconds == 0.01
    assert stats.max_seconds == 0.05
    assert stats.median_seconds == 0.03
    assert stats.mean_seconds == pytest.approx(0.03)
    assert stats.throughput_samples_per_sec == pytest.approx(2 / 0.03)


def test_empty_samples_raise() -> None:
    with pytest.raises(ValueError):
        compute_latency_stats([], warmup=0, batch_size=1)
