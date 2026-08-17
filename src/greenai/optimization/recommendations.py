"""Evidence-based optimization recommendations."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Recommendation(BaseModel):
    title: str
    detail: str
    evidence: str


class RecommendationReport(BaseModel):
    recommendations: list[Recommendation] = Field(default_factory=list)


def generate_recommendations(profile: dict[str, Any]) -> RecommendationReport:
    """Produce recommendations from measured profile fields only."""
    recs: list[Recommendation] = []

    model = profile.get("model") or {}
    hardware = profile.get("hardware") or {}
    benchmark = profile.get("benchmark") or {}
    memory = benchmark.get("memory") or {}
    energy = benchmark.get("energy") or {}
    layers = (profile.get("layers") or {}).get("layers") or []

    size = model.get("size_bytes")
    params = model.get("parameter_count")
    if size and params and params > 0:
        bytes_per_param = size / params
        if bytes_per_param > 6:
            recs.append(
                Recommendation(
                    title="Model size is large relative to parameter count",
                    detail="Inspect serialization format and parameter dtypes.",
                    evidence=f"~{bytes_per_param:.1f} bytes/parameter",
                )
            )

    peak = memory.get("gpu_peak_allocated_bytes") or memory.get("cpu_rss_bytes")
    if peak and peak > 500 * 1024 * 1024:
        recs.append(
            Recommendation(
                title="High memory usage observed",
                detail="Consider FP16 or INT8 if accuracy budgets allow.",
                evidence=f"Peak/RSS reported as {peak} bytes",
            )
        )

    if layers:
        top = layers[0]
        pct = float(top.get("latency_percent") or 0)
        if pct >= 10:
            recs.append(
                Recommendation(
                    title="Latency concentrated in a few layers",
                    detail="Investigate layer-specific kernels, fusion, or replacement.",
                    evidence=f"{top.get('name')} ~ {pct:.1f}% of leaf latency",
                )
            )

    if hardware.get("cuda_available") and str(
        benchmark.get("config", {}).get("device", "")
    ).startswith("cpu"):
        recs.append(
            Recommendation(
                title="CUDA is available but benchmark ran on CPU",
                detail="Consider GPU inference benchmarking for deployment-relevant numbers.",
                evidence="hardware.cuda_available=true with device=cpu",
            )
        )

    if (energy.get("status") or "") == "unavailable":
        recs.append(
            Recommendation(
                title="Energy measurement unavailable",
                detail="Use NVIDIA NVML or Intel RAPL hosts, or import external measurements.",
                evidence=energy.get("reason") or "No energy backend",
            )
        )

    dtypes = model.get("dtypes") or []
    if dtypes and all("float32" in d or d == "float32" for d in dtypes):
        recs.append(
            Recommendation(
                title="Model appears to use FP32 weights",
                detail="Try FP16 on supported GPUs and compare accuracy/latency/energy.",
                evidence=f"dtypes={dtypes}",
            )
        )

    if not recs:
        recs.append(
            Recommendation(
                title="No strong optimization signals from current measurements",
                detail="Collect layer profiles, energy on supported hardware, or accuracy deltas.",
                evidence="Recommendation engine found no threshold breaches",
            )
        )

    return RecommendationReport(recommendations=recs)
