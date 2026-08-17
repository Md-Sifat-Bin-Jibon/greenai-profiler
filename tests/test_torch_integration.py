"""Model inspection and benchmark integration tests (requires torch)."""

from __future__ import annotations

from pathlib import Path

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


def test_state_dict_requires_architecture(tiny_model: torch.nn.Module, tmp_path: Path) -> None:
    from greenai.exceptions import ModelLoadError
    from greenai.models.pytorch import PyTorchModelAdapter

    path = tmp_path / "weights.pt"
    torch.save(tiny_model.state_dict(), path)

    with pytest.raises(ModelLoadError, match="state_dict"):
        PyTorchModelAdapter(path, device="cpu", example_input_shape=[1, 28, 28]).load()

    architecture = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(28 * 28, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 10),
    )
    adapter = PyTorchModelAdapter(
        path,
        device="cpu",
        architecture=architecture,
        example_input_shape=[1, 28, 28],
    )
    loaded = adapter.load()
    info = adapter.inspect()
    assert loaded is architecture
    assert info.parameter_count is not None and info.parameter_count > 0
    assert any("state_dict" in note for note in info.notes)


def test_pickled_module_requires_allow_pickle(tiny_model: torch.nn.Module, tmp_path: Path) -> None:
    from greenai.exceptions import ModelLoadError
    from greenai.models.pytorch import PyTorchModelAdapter

    path = tmp_path / "module.pt"
    torch.save(tiny_model, path)
    with pytest.raises(ModelLoadError, match="allow-pickle"):
        PyTorchModelAdapter(path, device="cpu").load()

    loaded = PyTorchModelAdapter(path, device="cpu", allow_pickle=True).load()
    assert isinstance(loaded, torch.nn.Module)


def test_benchmark_polls_energy_monitor(tiny_model: torch.nn.Module) -> None:
    from greenai.benchmark import BenchmarkConfig, run_benchmark
    from greenai.hardware.energy import EnergyMonitor, EnergyResult, EnergyStatus
    from greenai.models.pytorch import PyTorchModelAdapter

    class CountingMonitor(EnergyMonitor):
        name = "counting"

        def __init__(self) -> None:
            self.polls = 0

        def available(self) -> bool:
            return True

        def start(self) -> None:
            return None

        def poll(self) -> None:
            self.polls += 1

        def stop(self, inferences: int) -> EnergyResult:
            return EnergyResult(
                status=EnergyStatus.UNAVAILABLE,
                method=self.name,
                inferences=inferences,
                reason="test monitor",
            )

    monitor = CountingMonitor()
    adapter = PyTorchModelAdapter(tiny_model, device="cpu", example_input_shape=[1, 28, 28])
    run_benchmark(
        adapter,
        BenchmarkConfig(
            device="cpu",
            input_shape=[1, 28, 28],
            warmup=1,
            iterations=5,
            measure_energy=True,
        ),
        energy_monitor=monitor,
    )
    assert monitor.polls == 5
