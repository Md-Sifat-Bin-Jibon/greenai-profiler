"""Memory measurement tests."""

from __future__ import annotations

from greenai.benchmark.memory import measure_memory


def test_measure_memory_cpu() -> None:
    result = measure_memory(device="cpu")
    assert result.cpu_rss_bytes is not None
    assert result.cpu_rss_bytes > 0
    assert result.gpu_allocated_bytes is None
    assert any("RSS" in note or "rss" in note.lower() for note in result.notes)
