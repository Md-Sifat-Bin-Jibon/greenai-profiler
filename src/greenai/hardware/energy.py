"""Extensible energy measurement backends.

Energy is never fabricated. Backends either measure, estimate with an explicit
label, or report unavailable.
"""

from __future__ import annotations

import contextlib
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EnergyStatus(str, Enum):  # noqa: UP042 - keep str Enum for broad Python support
    MEASURED = "measured"
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"


class EnergyResult(BaseModel):
    """Energy measurement outcome for a profiling interval."""

    status: EnergyStatus
    method: str
    total_energy_joules: float | None = None
    energy_per_inference_joules: float | None = None
    inferences: int | None = None
    average_power_watts: float | None = None
    duration_seconds: float | None = None
    reason: str | None = None
    notes: list[str] = Field(default_factory=list)

    def to_display(self) -> dict[str, Any]:
        return self.model_dump()


class EnergyMonitor(ABC):
    """Abstract energy monitor."""

    name: str = "base"

    @abstractmethod
    def available(self) -> bool:
        """Return True if this backend can measure on the current host."""

    @abstractmethod
    def start(self) -> None:
        """Begin an energy measurement window."""

    @abstractmethod
    def stop(self, inferences: int) -> EnergyResult:
        """End the window and return an EnergyResult."""


class UnsupportedEnergyMonitor(EnergyMonitor):
    """Fallback monitor that always reports unavailable."""

    name = "unsupported"

    def available(self) -> bool:
        return False

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self, inferences: int) -> EnergyResult:
        duration = time.perf_counter() - getattr(self, "_t0", time.perf_counter())
        return EnergyResult(
            status=EnergyStatus.UNAVAILABLE,
            method="none",
            inferences=inferences,
            duration_seconds=duration,
            reason="No supported hardware energy interface detected.",
        )


class NvidiaEnergyMonitor(EnergyMonitor):
    """Measure energy by integrating NVML instantaneous power over time.

    Methodology: sample GPU power draw at a short interval while the workload
    runs, then approximate energy as mean(power) * duration. This is a
    measured estimate of energy from hardware power telemetry, not a direct
    joule counter. Instantaneous power alone is never reported as energy.
    """

    name = "nvidia_nvml"

    def __init__(self, device_index: int = 0, sample_interval_s: float = 0.05) -> None:
        self.device_index = device_index
        self.sample_interval_s = sample_interval_s
        self._t0 = 0.0
        self._samples: list[float] = []
        self._pynvml: Any | None = None
        self._handle: Any | None = None

    def available(self) -> bool:
        try:
            import pynvml  # type: ignore[import-untyped]

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            _ = pynvml.nvmlDeviceGetPowerUsage(handle)
            pynvml.nvmlShutdown()
            return True
        except Exception:
            with contextlib.suppress(Exception):
                import pynvml  # type: ignore[import-untyped]

                pynvml.nvmlShutdown()
            return False

    def start(self) -> None:
        import pynvml  # type: ignore[import-untyped]

        self._pynvml = pynvml
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        self._samples = []
        self._t0 = time.perf_counter()
        self._sample_once()

    def _sample_once(self) -> None:
        if self._pynvml is None or self._handle is None:
            return
        try:
            milliwatts = self._pynvml.nvmlDeviceGetPowerUsage(self._handle)
            self._samples.append(float(milliwatts) / 1000.0)
        except Exception:
            return

    def poll(self) -> None:
        """Optional mid-run sample; call from long workloads."""
        now = time.perf_counter()
        if not self._samples or (now - self._t0) >= self.sample_interval_s * len(self._samples):
            self._sample_once()

    def stop(self, inferences: int) -> EnergyResult:
        self._sample_once()
        duration = time.perf_counter() - self._t0
        with contextlib.suppress(Exception):
            if self._pynvml is not None:
                self._pynvml.nvmlShutdown()

        if not self._samples or duration <= 0:
            return EnergyResult(
                status=EnergyStatus.UNAVAILABLE,
                method=self.name,
                inferences=inferences,
                duration_seconds=duration,
                reason="NVML power samples could not be collected.",
            )

        avg_power = sum(self._samples) / len(self._samples)
        total_energy = avg_power * duration
        per_inf = total_energy / inferences if inferences > 0 else None
        return EnergyResult(
            status=EnergyStatus.MEASURED,
            method=self.name,
            total_energy_joules=total_energy,
            energy_per_inference_joules=per_inf,
            inferences=inferences,
            average_power_watts=avg_power,
            duration_seconds=duration,
            notes=[
                "Energy approximated as mean(NVML power samples) x duration.",
                "Does not subtract idle/baseline power.",
            ],
        )


class IntelRaplMonitor(EnergyMonitor):
    """Read Intel RAPL energy counters via powercap sysfs when present."""

    name = "intel_rapl"
    _RAPL_PATH = "/sys/class/powercap/intel-rapl:0/energy_uj"

    def __init__(self) -> None:
        self._start_uj: int | None = None
        self._t0 = 0.0

    def available(self) -> bool:
        try:
            with open(self._RAPL_PATH, encoding="utf-8") as handle:
                _ = handle.read()
            return True
        except OSError:
            return False

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self._start_uj = self._read_uj()

    def _read_uj(self) -> int | None:
        try:
            with open(self._RAPL_PATH, encoding="utf-8") as handle:
                return int(handle.read().strip())
        except OSError:
            return None

    def stop(self, inferences: int) -> EnergyResult:
        end_uj = self._read_uj()
        duration = time.perf_counter() - self._t0
        if self._start_uj is None or end_uj is None:
            return EnergyResult(
                status=EnergyStatus.UNAVAILABLE,
                method=self.name,
                inferences=inferences,
                duration_seconds=duration,
                reason="RAPL energy counter could not be read.",
            )
        # Counter may wrap; treat negative delta as unavailable rather than guess.
        delta_uj = end_uj - self._start_uj
        if delta_uj < 0:
            return EnergyResult(
                status=EnergyStatus.UNAVAILABLE,
                method=self.name,
                inferences=inferences,
                duration_seconds=duration,
                reason="RAPL counter wrapped or decreased; refusing to guess.",
            )
        total_j = delta_uj / 1_000_000.0
        per_inf = total_j / inferences if inferences > 0 else None
        avg_power = total_j / duration if duration > 0 else None
        return EnergyResult(
            status=EnergyStatus.MEASURED,
            method=self.name,
            total_energy_joules=total_j,
            energy_per_inference_joules=per_inf,
            inferences=inferences,
            average_power_watts=avg_power,
            duration_seconds=duration,
            notes=["Measured via Intel RAPL powercap energy_uj counter."],
        )


def select_energy_monitor(prefer: str | None = None) -> EnergyMonitor:
    """Choose the best available energy monitor for this host."""
    candidates: list[EnergyMonitor] = [
        NvidiaEnergyMonitor(),
        IntelRaplMonitor(),
    ]
    if prefer:
        for monitor in candidates:
            if monitor.name == prefer and monitor.available():
                return monitor
    for monitor in candidates:
        if monitor.available():
            return monitor
    return UnsupportedEnergyMonitor()
