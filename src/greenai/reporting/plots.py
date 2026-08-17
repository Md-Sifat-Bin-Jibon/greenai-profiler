"""Optional matplotlib visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_layer_latency(layers: list[dict[str, Any]], path: str | Path, *, top_k: int = 15) -> None:
    """Write a horizontal bar chart of the slowest layers."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plots. Install with: pip install 'green-ai-profiler[viz]'"
        ) from exc

    subset = layers[:top_k]
    names = [str(item.get("name")) for item in reversed(subset)]
    values = [float(item.get("latency_seconds") or 0) * 1000.0 for item in reversed(subset)]

    fig, ax = plt.subplots(figsize=(8, max(3, len(subset) * 0.35)))
    ax.barh(names, values, color="#2f6f4e")
    ax.set_xlabel("Latency (ms)")
    ax.set_title("Layer latency ranking")
    fig.tight_layout()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target)
    plt.close(fig)
