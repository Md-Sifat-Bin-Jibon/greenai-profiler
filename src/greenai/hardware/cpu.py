"""CPU helpers."""

from __future__ import annotations

import os
from typing import Any

import psutil


def process_rss_bytes(pid: int | None = None) -> int:
    """Return resident set size for the current (or given) process."""
    process = psutil.Process(pid or os.getpid())
    return int(process.memory_info().rss)


def cpu_percent(interval: float = 0.0) -> float:
    """Return system-wide CPU utilization percentage."""
    return float(psutil.cpu_percent(interval=interval))


def cpu_snapshot() -> dict[str, Any]:
    """Return a small CPU utilization snapshot."""
    return {
        "cpu_percent": cpu_percent(interval=0.0),
        "rss_bytes": process_rss_bytes(),
    }
