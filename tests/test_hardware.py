"""Hardware detection tests."""

from __future__ import annotations

from greenai.hardware import HardwareInfo, detect_hardware


def test_detect_hardware_returns_structured_info() -> None:
    info = detect_hardware()
    assert isinstance(info, HardwareInfo)
    assert info.python_version
    assert info.os_name
    assert info.ram_total_bytes is None or info.ram_total_bytes > 0
