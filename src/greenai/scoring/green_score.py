"""Project-defined efficiency score (not a scientific authority).

The Green Score is a transparent composite heuristic for relative comparison
within this tool. Always inspect underlying measurements.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GreenScore(BaseModel):
    score: float
    max_score: float = 100.0
    components: dict[str, float] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)


def compute_green_score(profile: dict[str, Any]) -> GreenScore:
    """Compute a simple 0-100 composite from available metrics.

    Weighting (documented):
    - latency: 30
    - model size: 25
    - memory peak/RSS: 25
    - energy per inference: 20 (neutral mid-score if unavailable)
    """
    bench = profile.get("benchmark") or {}
    model = profile.get("model") or {}
    latency = (bench.get("latency") or {}).get("mean_seconds")
    size = model.get("size_bytes")
    memory = (bench.get("memory") or {}).get("gpu_peak_allocated_bytes")
    if memory is None:
        memory = (bench.get("memory") or {}).get("cpu_rss_bytes")
    energy = (bench.get("energy") or {}).get("energy_per_inference_joules")
    energy_status = (bench.get("energy") or {}).get("status")

    components: dict[str, float] = {}

    # Lower is better for all of these; map with soft caps.
    components["latency"] = _invert_score(latency, good=0.002, bad=0.1) * 30
    components["size"] = _invert_score(size, good=5_000_000, bad=500_000_000) * 25
    components["memory"] = _invert_score(memory, good=50_000_000, bad=2_000_000_000) * 25
    if energy_status == "measured" and energy is not None:
        components["energy"] = _invert_score(energy, good=0.0005, bad=0.05) * 20
    else:
        components["energy"] = 10.0  # neutral when unavailable

    total = sum(components.values())
    return GreenScore(
        score=round(total, 1),
        components={k: round(v, 2) for k, v in components.items()},
        limitations=[
            "This is a project-defined heuristic, not a standardized scientific index.",
            "Unavailable energy contributes a neutral mid score rather than a penalty.",
            "Always report and review the underlying measured metrics.",
        ],
    )


def _invert_score(value: float | None, *, good: float, bad: float) -> float:
    if value is None:
        return 0.5
    if value <= good:
        return 1.0
    if value >= bad:
        return 0.0
    return 1.0 - ((value - good) / (bad - good))
