"""Energy backend tests with mocks."""

from __future__ import annotations

import pytest

from greenai.hardware.energy import (
    EnergyStatus,
    IntelRaplMonitor,
    NvidiaEnergyMonitor,
    UnsupportedEnergyMonitor,
    select_energy_monitor,
)


def test_unsupported_monitor() -> None:
    monitor = UnsupportedEnergyMonitor()
    assert monitor.available() is False
    monitor.start()
    result = monitor.stop(inferences=10)
    assert result.status == EnergyStatus.UNAVAILABLE
    assert result.total_energy_joules is None
    assert "No supported" in (result.reason or "")


def test_select_energy_monitor_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(NvidiaEnergyMonitor, "available", lambda self: False)
    monkeypatch.setattr(IntelRaplMonitor, "available", lambda self: False)
    monitor = select_energy_monitor()
    assert isinstance(monitor, UnsupportedEnergyMonitor)


def test_nvidia_stop_without_samples() -> None:
    monitor = NvidiaEnergyMonitor()

    class FakeNvml:
        @staticmethod
        def nvmlShutdown() -> None:
            return None

        @staticmethod
        def nvmlDeviceGetPowerUsage(handle: object) -> int:
            raise RuntimeError("no power")

    monitor._pynvml = FakeNvml
    monitor._handle = object()
    monitor._t0 = 0.0
    monitor._samples = []
    result = monitor.stop(inferences=5)
    assert result.status == EnergyStatus.UNAVAILABLE


def test_nvidia_integrates_power_samples() -> None:
    monitor = NvidiaEnergyMonitor()

    class FakeNvml:
        @staticmethod
        def nvmlShutdown() -> None:
            return None

    monitor._pynvml = FakeNvml
    monitor._handle = object()
    monitor._t0 = __import__("time").perf_counter() - 1.0
    monitor._samples = [100.0, 100.0]
    result = monitor.stop(inferences=10)
    assert result.status == EnergyStatus.MEASURED
    assert result.total_energy_joules is not None
    assert result.energy_per_inference_joules is not None
    assert result.average_power_watts == 100.0
