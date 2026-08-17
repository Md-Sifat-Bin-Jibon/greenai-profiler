"""Latency measurement for model inference."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from greenai.benchmark.statistics import LatencyStats, compute_latency_stats
from greenai.utils.timing import synchronize_if_cuda, timed_section


def measure_latency(
    forward_fn: Callable[[], Any],
    *,
    device: str = "cpu",
    warmup: int = 10,
    iterations: int = 50,
    batch_size: int = 1,
    keep_raw: bool = False,
    on_iteration: Callable[[], None] | None = None,
) -> LatencyStats:
    """Time repeated calls to ``forward_fn`` with optional CUDA sync."""
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be >= 0 and iterations must be >= 1.")

    for _ in range(warmup):
        forward_fn()
    synchronize_if_cuda(device)

    samples: list[float] = []
    for _ in range(iterations):
        with timed_section(device) as elapsed:
            forward_fn()
        samples.append(elapsed())
        if on_iteration is not None:
            on_iteration()

    return compute_latency_stats(
        samples,
        warmup=warmup,
        batch_size=batch_size,
        keep_raw=keep_raw,
    )
