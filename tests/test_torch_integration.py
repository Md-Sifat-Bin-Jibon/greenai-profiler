"""Model inspection and benchmark integration tests (requires torch)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def tiny_model() -> torch.nn.Module:
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(28 * 28, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 10),
    )
    model.eval()
    return model


def test_inspect_in_memory_model(tiny_model: torch.nn.Module) -> None:
    from greenai.models.pytorch import PyTorchModelAdapter

    adapter = PyTorchModelAdapter(tiny_model, device="cpu", example_input_shape=[1, 28, 28])
    info = adapter.inspect()
    assert info.framework == "pytorch"
    assert info.parameter_count is not None and info.parameter_count > 0
    assert info.size_bytes is not None and info.size_bytes > 0
    assert "float32" in info.dtypes


def test_benchmark_and_profile_layers(tiny_model: torch.nn.Module) -> None:
    from greenai.benchmark import BenchmarkConfig, run_benchmark
    from greenai.models.pytorch import PyTorchModelAdapter
    from greenai.profiling.layer_profiler import profile_layers

    adapter = PyTorchModelAdapter(tiny_model, device="cpu", example_input_shape=[1, 28, 28])
    result = run_benchmark(
        adapter,
        BenchmarkConfig(
            device="cpu",
            batch_size=1,
            input_shape=[1, 28, 28],
            warmup=2,
            iterations=5,
            measure_energy=True,
        ),
    )
    assert result.latency.iterations == 5
    assert result.latency.mean_seconds > 0
    assert result.memory.cpu_rss_bytes is not None
    assert result.energy is not None

    example = adapter.create_example_input(1, [1, 28, 28])
    layers = profile_layers(tiny_model, example, device="cpu", warmup=1, iterations=3)
    assert layers.layers
    assert abs(sum(layer.latency_percent for layer in layers.layers) - 100.0) < 1.0


def test_recommendations_and_compare(tiny_model: torch.nn.Module) -> None:
    from greenai import __version__
    from greenai.benchmark import BenchmarkConfig, run_benchmark
    from greenai.comparison import compare_results
    from greenai.hardware import detect_hardware
    from greenai.models.pytorch import PyTorchModelAdapter
    from greenai.optimization.recommendations import generate_recommendations
    from greenai.reporting.schema import ProfileResult, build_metadata

    adapter = PyTorchModelAdapter(tiny_model, device="cpu", example_input_shape=[1, 28, 28])
    bench = run_benchmark(
        adapter,
        BenchmarkConfig(device="cpu", warmup=1, iterations=3, input_shape=[1, 28, 28]),
    )
    doc = ProfileResult(
        metadata=build_metadata(__version__),
        model=bench.model.model_dump(),
        hardware=detect_hardware().model_dump(),
        benchmark={
            "latency": bench.latency.model_dump(),
            "memory": bench.memory.model_dump(),
            "energy": None if bench.energy is None else bench.energy.model_dump(),
            "config": bench.config.model_dump(),
        },
    ).to_dict()
    recs = generate_recommendations(doc)
    assert recs.recommendations

    optimized = {
        **doc,
        "benchmark": {
            **doc["benchmark"],
            "latency": {
                **doc["benchmark"]["latency"],
                "mean_seconds": bench.latency.mean_seconds * 0.5,
            },
        },
    }
    comparison = compare_results(doc, optimized)
    assert any(m.name.startswith("Latency") for m in comparison.metrics)
