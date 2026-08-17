"""High-level benchmark runner combining latency, memory, and energy."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from greenai.benchmark.latency import measure_latency
from greenai.benchmark.memory import MemoryResult, measure_memory
from greenai.benchmark.statistics import LatencyStats
from greenai.hardware.energy import EnergyMonitor, EnergyResult, select_energy_monitor
from greenai.hardware.nvidia import reset_peak_memory_stats
from greenai.models.base import BaseModelAdapter, ModelInfo


class BenchmarkConfig(BaseModel):
    """Configuration for a benchmark run."""

    device: str = "cpu"
    batch_size: int = 1
    input_shape: list[int] | None = None
    warmup: int = 10
    iterations: int = 50
    keep_raw: bool = False
    measure_energy: bool = True
    energy_backend: str | None = None


class BenchmarkResult(BaseModel):
    """Combined benchmark outcome."""

    model: ModelInfo
    latency: LatencyStats
    memory: MemoryResult
    energy: EnergyResult | None = None
    config: BenchmarkConfig
    notes: list[str] = Field(default_factory=list)


def run_benchmark(
    adapter: BaseModelAdapter,
    config: BenchmarkConfig | None = None,
    *,
    energy_monitor: EnergyMonitor | None = None,
) -> BenchmarkResult:
    """Run warmup + timed inference and collect memory/energy metrics."""
    cfg = config or BenchmarkConfig()
    model = adapter.load()
    example = adapter.create_example_input(cfg.batch_size, cfg.input_shape)
    info = adapter.inspect()

    def forward() -> Any:
        return model(example)

    notes: list[str] = []
    if cfg.device.startswith("cuda"):
        reset_peak_memory_stats()

    monitor = energy_monitor
    if cfg.measure_energy:
        monitor = monitor or select_energy_monitor(cfg.energy_backend)
        monitor.start()

    # For NVIDIA monitor, poll between iterations when possible.
    latency = measure_latency(
        forward,
        device=cfg.device,
        warmup=cfg.warmup,
        iterations=cfg.iterations,
        batch_size=cfg.batch_size,
        keep_raw=cfg.keep_raw,
    )

    energy: EnergyResult | None = None
    if cfg.measure_energy and monitor is not None:
        # Extra polling not needed for RAPL/NVML stop which samples endpoints.
        energy = monitor.stop(inferences=cfg.iterations * cfg.batch_size)
    elif not cfg.measure_energy:
        notes.append("Energy measurement skipped by configuration.")

    memory = measure_memory(device=cfg.device, reset_peaks=False)
    return BenchmarkResult(
        model=info,
        latency=latency,
        memory=memory,
        energy=energy,
        config=cfg,
        notes=notes,
    )
