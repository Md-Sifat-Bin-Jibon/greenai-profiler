"""Compare two versioned profiling result documents."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from greenai.utils.formatting import pct_change


class MetricDelta(BaseModel):
    name: str
    baseline: float | None
    optimized: float | None
    change_percent: float | None
    unit: str = ""


class ComparisonResult(BaseModel):
    metrics: list[MetricDelta] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def _get(doc: dict[str, Any], *path: str) -> Any:
    cur: Any = doc
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def compare_results(baseline: dict[str, Any], optimized: dict[str, Any]) -> ComparisonResult:
    """Compare selected numeric fields from two profile result dicts."""
    pairs: list[tuple[str, tuple[str, ...], tuple[str, ...], str]] = [
        (
            "Latency (mean)",
            ("benchmark", "latency", "mean_seconds"),
            ("benchmark", "latency", "mean_seconds"),
            "s",
        ),
        (
            "Throughput",
            ("benchmark", "latency", "throughput_samples_per_sec"),
            ("benchmark", "latency", "throughput_samples_per_sec"),
            "samples/s",
        ),
        (
            "GPU Peak Allocated",
            ("benchmark", "memory", "gpu_peak_allocated_bytes"),
            ("benchmark", "memory", "gpu_peak_allocated_bytes"),
            "bytes",
        ),
        (
            "CPU RSS",
            ("benchmark", "memory", "cpu_rss_bytes"),
            ("benchmark", "memory", "cpu_rss_bytes"),
            "bytes",
        ),
        (
            "Energy / Inference",
            ("benchmark", "energy", "energy_per_inference_joules"),
            ("benchmark", "energy", "energy_per_inference_joules"),
            "J",
        ),
        ("Model Size", ("model", "size_bytes"), ("model", "size_bytes"), "bytes"),
        ("Parameters", ("model", "parameter_count"), ("model", "parameter_count"), "count"),
        ("Accuracy", ("accuracy", "value"), ("accuracy", "value"), "ratio"),
    ]

    metrics: list[MetricDelta] = []
    notes: list[str] = []
    for name, b_path, o_path, unit in pairs:
        b_val = _get(baseline, *b_path)
        o_val = _get(optimized, *o_path)
        if b_val is None and o_val is None:
            continue
        try:
            b_f = float(b_val) if b_val is not None else None
            o_f = float(o_val) if o_val is not None else None
        except (TypeError, ValueError):
            continue
        metrics.append(
            MetricDelta(
                name=name,
                baseline=b_f,
                optimized=o_f,
                change_percent=pct_change(b_f, o_f),
                unit=unit,
            )
        )

    if _get(baseline, "accuracy", "value") is None and _get(optimized, "accuracy", "value") is None:
        notes.append("Accuracy omitted because no evaluator results were supplied.")

    return ComparisonResult(metrics=metrics, notes=notes)
