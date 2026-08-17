"""NVIDIA GPU helpers via NVML or PyTorch."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NvidiaMemoryStats:
    """CUDA memory statistics for a single device."""

    allocated_bytes: int | None
    reserved_bytes: int | None
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    source: str


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def reset_peak_memory_stats(device: int = 0) -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
    except ImportError:
        return


def memory_stats(device: int = 0) -> NvidiaMemoryStats:
    """Collect CUDA allocator stats when available."""
    try:
        import torch
    except ImportError:
        return NvidiaMemoryStats(None, None, None, None, "unavailable")

    if not torch.cuda.is_available():
        return NvidiaMemoryStats(None, None, None, None, "cuda_unavailable")

    return NvidiaMemoryStats(
        allocated_bytes=int(torch.cuda.memory_allocated(device)),
        reserved_bytes=int(torch.cuda.memory_reserved(device)),
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
        source="torch.cuda",
    )


def power_draw_watts(device_index: int = 0) -> float | None:
    """Return instantaneous GPU power draw in watts, or None if unavailable."""
    try:
        import pynvml  # type: ignore[import-untyped]
    except ImportError:
        return None

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        milliwatts = pynvml.nvmlDeviceGetPowerUsage(handle)
        pynvml.nvmlShutdown()
        return float(milliwatts) / 1000.0
    except Exception:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()
        return None


def nvidia_summary() -> dict[str, Any]:
    stats = memory_stats()
    return {
        "cuda_available": cuda_available(),
        "power_watts": power_draw_watts(),
        "memory": stats.__dict__,
    }
