"""Hardware detection and energy backends."""

from __future__ import annotations

from greenai.hardware.detector import HardwareInfo, detect_hardware
from greenai.hardware.energy import (
    EnergyMonitor,
    EnergyResult,
    EnergyStatus,
    IntelRaplMonitor,
    NvidiaEnergyMonitor,
    UnsupportedEnergyMonitor,
    select_energy_monitor,
)

__all__ = [
    "EnergyMonitor",
    "EnergyResult",
    "EnergyStatus",
    "HardwareInfo",
    "IntelRaplMonitor",
    "NvidiaEnergyMonitor",
    "UnsupportedEnergyMonitor",
    "detect_hardware",
    "select_energy_monitor",
]
