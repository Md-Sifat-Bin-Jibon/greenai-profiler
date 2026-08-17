"""Minimal programmatic API example."""

from __future__ import annotations


def main() -> None:
    import torch

    from greenai.benchmark import BenchmarkConfig, run_benchmark
    from greenai.models.pytorch import PyTorchModelAdapter

    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 4),
    )
    adapter = PyTorchModelAdapter(model, device="cpu", example_input_shape=[64])
    result = run_benchmark(
        adapter,
        BenchmarkConfig(device="cpu", input_shape=[64], warmup=2, iterations=10),
    )
    print(f"mean latency: {result.latency.mean_seconds * 1000:.3f} ms")
    print(f"energy status: {result.energy.status if result.energy else 'n/a'}")


if __name__ == "__main__":
    main()
