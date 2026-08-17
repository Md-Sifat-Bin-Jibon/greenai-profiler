"""Bottleneck analysis from layer profiles."""

from __future__ import annotations

from pydantic import BaseModel, Field

from greenai.profiling.layer_profiler import LayerProfileResult


class Bottleneck(BaseModel):
    name: str
    category: str
    latency_percent: float
    detail: str


class BottleneckReport(BaseModel):
    bottlenecks: list[Bottleneck] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def analyze_bottlenecks(
    layer_profile: LayerProfileResult,
    *,
    top_k: int = 5,
    high_latency_pct: float = 8.0,
) -> BottleneckReport:
    """Identify high-latency layers from a layer profile."""
    items: list[Bottleneck] = []
    for entry in layer_profile.layers[:top_k]:
        category = "HIGH LATENCY" if entry.latency_percent >= high_latency_pct else "LATENCY"
        items.append(
            Bottleneck(
                name=entry.name,
                category=category,
                latency_percent=entry.latency_percent,
                detail=f"{entry.latency_percent:.1f}% of measured leaf-module latency",
            )
        )
    return BottleneckReport(
        bottlenecks=items,
        notes=[
            "Categories currently focus on latency. Memory/energy categories require "
            "per-layer measurements that are not always available.",
        ],
    )
