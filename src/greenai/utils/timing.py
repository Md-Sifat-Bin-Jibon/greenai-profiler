"""Timing utilities with optional CUDA synchronization."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager


def synchronize_if_cuda(device: str) -> None:
    """Synchronize CUDA if the device string indicates a CUDA device."""
    if not device.startswith("cuda"):
        return
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def timed_section(device: str = "cpu") -> Iterator[Callable[[], float]]:
    """Context manager that yields a callable returning elapsed seconds.

    Performs CUDA synchronization before start and before reading elapsed time
    when ``device`` refers to CUDA.
    """
    synchronize_if_cuda(device)
    start = time.perf_counter()
    elapsed_holder = {"value": 0.0}

    def elapsed() -> float:
        synchronize_if_cuda(device)
        elapsed_holder["value"] = time.perf_counter() - start
        return elapsed_holder["value"]

    try:
        yield elapsed
    finally:
        # Ensure a final sync/time capture if caller never invoked elapsed().
        if elapsed_holder["value"] == 0.0:
            synchronize_if_cuda(device)
            elapsed_holder["value"] = time.perf_counter() - start
