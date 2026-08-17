"""Memory measurement for CPU and CUDA."""

from __future__ import annotations

from pydantic import BaseModel, Field

from greenai.hardware.cpu import process_rss_bytes
from greenai.hardware.nvidia import memory_stats, reset_peak_memory_stats


class MemoryResult(BaseModel):
    """Memory snapshot. RSS is process RSS, not model-only memory."""

    cpu_rss_bytes: int | None = None
    gpu_allocated_bytes: int | None = None
    gpu_reserved_bytes: int | None = None
    gpu_peak_allocated_bytes: int | None = None
    gpu_peak_reserved_bytes: int | None = None
    source: str = "mixed"
    notes: list[str] = Field(default_factory=list)


def measure_memory(*, device: str = "cpu", reset_peaks: bool = False) -> MemoryResult:
    """Collect CPU RSS and CUDA allocator stats when applicable."""
    notes = [
        "CPU RSS is process resident memory, not equivalent to model parameter memory.",
    ]
    if reset_peaks and device.startswith("cuda"):
        reset_peak_memory_stats()

    cpu_rss = process_rss_bytes()
    gpu = memory_stats() if device.startswith("cuda") else None

    return MemoryResult(
        cpu_rss_bytes=cpu_rss,
        gpu_allocated_bytes=None if gpu is None else gpu.allocated_bytes,
        gpu_reserved_bytes=None if gpu is None else gpu.reserved_bytes,
        gpu_peak_allocated_bytes=None if gpu is None else gpu.peak_allocated_bytes,
        gpu_peak_reserved_bytes=None if gpu is None else gpu.peak_reserved_bytes,
        source="psutil" if gpu is None else f"psutil+{gpu.source}",
        notes=notes,
    )
