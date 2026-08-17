"""Hardware information models and detection."""

from __future__ import annotations

import contextlib
import importlib
import platform
import sys
from typing import Any

import psutil
from pydantic import BaseModel, Field


class HardwareInfo(BaseModel):
    """Structured snapshot of the host used for a profiling run."""

    os_name: str
    os_release: str
    os_version: str
    machine: str
    python_version: str
    cpu_name: str | None = None
    cpu_cores_physical: int | None = None
    cpu_cores_logical: int | None = None
    ram_total_bytes: int | None = None
    gpu_vendor: str | None = None
    gpu_name: str | None = None
    gpu_memory_bytes: int | None = None
    cuda_available: bool = False
    cuda_version: str | None = None
    pytorch_version: str | None = None
    notes: list[str] = Field(default_factory=list)

    def to_display_dict(self) -> dict[str, Any]:
        """Return a flat dict suitable for terminal rendering."""
        return self.model_dump()


def _windows_cpu_name() -> str | None:
    """Read the CPU brand string from the Windows registry."""
    # winreg only exists on Windows; typed as Any so non-Windows type checks pass.
    winreg: Any = importlib.import_module("winreg")
    key = winreg.OpenKey(
        winreg.HKEY_LOCAL_MACHINE,
        r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
    )
    try:
        name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
    finally:
        winreg.CloseKey(key)
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def _cpu_name() -> str | None:
    if platform.system() == "Windows":
        try:
            resolved = _windows_cpu_name()
        except (OSError, ImportError):
            resolved = None
        if resolved is not None:
            return resolved
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if platform.system() == "Darwin":
        try:
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (OSError, FileNotFoundError):
            pass
    return platform.processor() or None


def _pytorch_info() -> tuple[str | None, bool, str | None]:
    try:
        import torch
    except ImportError:
        return None, False, None
    version = getattr(torch, "__version__", None)
    cuda_available = bool(torch.cuda.is_available())
    cuda_version = None
    if cuda_available:
        cuda_version = getattr(torch.version, "cuda", None)
    return version, cuda_available, cuda_version


def _nvidia_gpu_info() -> tuple[str | None, str | None, int | None, list[str]]:
    notes: list[str] = []
    try:
        import pynvml  # type: ignore[import-untyped]
    except ImportError:
        # Fall back to torch device properties when NVML is absent.
        try:
            import torch

            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return "NVIDIA", props.name, int(props.total_memory), notes
        except Exception:
            notes.append("NVIDIA GPU details unavailable.")
        return None, None, None, notes

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total = int(mem.total)
        pynvml.nvmlShutdown()
        return "NVIDIA", str(name), total, notes
    except Exception as exc:
        notes.append(f"NVML probe failed: {exc}")
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()
        return None, None, None, notes


def detect_hardware() -> HardwareInfo:
    """Detect host hardware and runtime details without hardcoding values."""
    pytorch_version, cuda_available, cuda_version = _pytorch_info()
    gpu_vendor, gpu_name, gpu_memory, notes = _nvidia_gpu_info()

    vm = psutil.virtual_memory()
    return HardwareInfo(
        os_name=platform.system(),
        os_release=platform.release(),
        os_version=platform.version(),
        machine=platform.machine(),
        python_version=sys.version.split()[0],
        cpu_name=_cpu_name(),
        cpu_cores_physical=psutil.cpu_count(logical=False),
        cpu_cores_logical=psutil.cpu_count(logical=True),
        ram_total_bytes=int(vm.total),
        gpu_vendor=gpu_vendor,
        gpu_name=gpu_name,
        gpu_memory_bytes=gpu_memory,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        pytorch_version=pytorch_version,
        notes=notes,
    )
