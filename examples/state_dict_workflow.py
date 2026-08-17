"""Safer checkpoint workflow: save and load a ``state_dict``.

Full ``torch.save(model, ...)`` files need ``--allow-pickle``. Weights-only
files do not, but you must reconstruct the architecture in Python.

Usage::

    python examples/state_dict_workflow.py
"""

from __future__ import annotations

from pathlib import Path

import torch

from greenai.benchmark import BenchmarkConfig, run_benchmark
from greenai.models.pytorch import PyTorchModelAdapter


def build_tiny_cnn() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(8, 16, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(16, 10),
    )


def main() -> None:
    artifacts = Path(__file__).resolve().parent / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    path = artifacts / "tiny_cnn_state_dict.pt"

    model = build_tiny_cnn()
    model.eval()
    torch.save(model.state_dict(), path)
    print(f"Wrote {path}")

    architecture = build_tiny_cnn()
    adapter = PyTorchModelAdapter(
        path,
        device="cpu",
        architecture=architecture,
        example_input_shape=[1, 28, 28],
    )
    result = run_benchmark(
        adapter,
        BenchmarkConfig(
            device="cpu",
            input_shape=[1, 28, 28],
            warmup=2,
            iterations=10,
        ),
    )
    print(f"loaded: {path.name}")
    print(f"mean latency: {result.latency.mean_seconds * 1000:.3f} ms")
    print(f"energy status: {result.energy.status if result.energy else 'n/a'}")


if __name__ == "__main__":
    main()
