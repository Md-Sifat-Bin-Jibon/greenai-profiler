"""Layer-wise latency profiling for PyTorch models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from greenai.profiling.hooks import HookRecorder, LayerSample


class LayerProfileEntry(BaseModel):
    name: str
    module_type: str
    latency_seconds: float
    latency_percent: float
    parameter_count: int | None = None
    input_shape: list[int] | None = None
    output_shape: list[int] | None = None
    dtype: str | None = None
    energy_status: str = "unavailable"
    energy_joules: float | None = None
    energy_notes: str = "Layer energy is unavailable; hardware cannot isolate energy per layer."


class LayerProfileResult(BaseModel):
    layers: list[LayerProfileEntry]
    total_latency_seconds: float
    iterations: int
    notes: list[str] = Field(default_factory=list)


def _aggregate(samples: list[LayerSample]) -> dict[str, LayerSample]:
    """Average repeated samples per layer name."""
    buckets: dict[str, list[LayerSample]] = {}
    for sample in samples:
        buckets.setdefault(sample.name, []).append(sample)
    aggregated: dict[str, LayerSample] = {}
    for name, group in buckets.items():
        avg_latency = sum(s.latency_seconds for s in group) / len(group)
        first = group[0]
        aggregated[name] = LayerSample(
            name=name,
            module_type=first.module_type,
            latency_seconds=avg_latency,
            input_shape=first.input_shape,
            output_shape=first.output_shape,
            parameter_count=first.parameter_count,
            dtype=first.dtype,
        )
    return aggregated


def profile_layers(
    model: Any,
    example_input: Any,
    *,
    device: str = "cpu",
    warmup: int = 5,
    iterations: int = 20,
) -> LayerProfileResult:
    """Profile named leaf modules via forward hooks."""
    recorder = HookRecorder(device=device)
    handles = []
    for name, module in model.named_modules():
        if name == "":
            continue
        # Profile leaf modules only to avoid double-counting parent containers.
        if any(True for _ in module.children()):
            continue
        handles.append(module.register_forward_pre_hook(recorder.pre_hook(name)))
        handles.append(module.register_forward_hook(recorder.forward_hook(name)))

    try:
        model.eval()
        for _ in range(warmup):
            model(example_input)
        recorder.samples.clear()
        for _ in range(iterations):
            model(example_input)
    finally:
        for handle in handles:
            handle.remove()

    aggregated = _aggregate(recorder.samples)
    total = sum(s.latency_seconds for s in aggregated.values()) or 1e-12
    layers = [
        LayerProfileEntry(
            name=s.name,
            module_type=s.module_type,
            latency_seconds=s.latency_seconds,
            latency_percent=(s.latency_seconds / total) * 100.0,
            parameter_count=s.parameter_count,
            input_shape=s.input_shape,
            output_shape=s.output_shape,
            dtype=s.dtype,
        )
        for s in sorted(aggregated.values(), key=lambda x: x.latency_seconds, reverse=True)
    ]
    return LayerProfileResult(
        layers=layers,
        total_latency_seconds=total,
        iterations=iterations,
        notes=[
            "Latencies are measured per leaf module via forward hooks.",
            "Parent modules are excluded to reduce double-counting.",
            "Layer energy is not claimed as measured.",
        ],
    )
