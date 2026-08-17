"""Command-line interface for Green AI Profiler."""

from __future__ import annotations

from pathlib import Path

import typer

from greenai import __version__

app = typer.Typer(
    name="greenai",
    help="Profile AI model efficiency: latency, memory, energy, and layers.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool | None = typer.Option(
        None,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the package version and exit.",
    ),
) -> None:
    """Green AI Profiler CLI."""


@app.command("system-info")
def system_info() -> None:
    """Detect and display host hardware / runtime information."""
    from greenai.hardware import detect_hardware
    from greenai.reporting.terminal import print_system_info

    info = detect_hardware()
    print_system_info(info.model_dump())


@app.command("inspect")
def inspect_model(
    model_path: Path = typer.Argument(
        ..., exists=True, readable=True, help="Path to a .pt/.pth model."
    ),
    device: str = typer.Option("cpu", help="Device string, e.g. cpu or cuda"),
    input_shape: str | None = typer.Option(
        None,
        help="Comma-separated input shape without batch, e.g. 3,224,224",
    ),
    allow_pickle: bool = typer.Option(
        False,
        help="Allow unsafe pickle deserialization of full nn.Module checkpoints.",
    ),
) -> None:
    """Inspect a PyTorch model checkpoint or module file."""
    from greenai.models.pytorch import PyTorchModelAdapter
    from greenai.reporting.terminal import print_model_info

    shape = _parse_shape(input_shape)
    adapter = PyTorchModelAdapter(
        model_path,
        device=device,
        allow_pickle=allow_pickle,
        example_input_shape=shape,
    )
    info = adapter.inspect()
    print_model_info(info.model_dump())


@app.command("benchmark")
def benchmark(
    model_path: Path = typer.Argument(..., exists=True, readable=True),
    device: str = typer.Option("cpu"),
    batch_size: int = typer.Option(1, min=1),
    input_shape: str | None = typer.Option(None, help="e.g. 3,224,224"),
    warmup: int = typer.Option(10, min=0),
    iterations: int = typer.Option(50, min=1),
    allow_pickle: bool = typer.Option(False),
    no_energy: bool = typer.Option(False, help="Skip energy measurement."),
) -> None:
    """Benchmark latency, throughput, memory, and energy where available."""
    from greenai.benchmark import BenchmarkConfig, run_benchmark
    from greenai.models.pytorch import PyTorchModelAdapter
    from greenai.reporting.terminal import print_benchmark, print_model_info

    adapter = PyTorchModelAdapter(
        model_path,
        device=device,
        allow_pickle=allow_pickle,
        example_input_shape=_parse_shape(input_shape),
    )
    result = run_benchmark(
        adapter,
        BenchmarkConfig(
            device=device,
            batch_size=batch_size,
            input_shape=_parse_shape(input_shape),
            warmup=warmup,
            iterations=iterations,
            measure_energy=not no_energy,
        ),
    )
    print_model_info(result.model.model_dump())
    print_benchmark(
        result.latency.model_dump(),
        result.memory.model_dump(),
        None if result.energy is None else result.energy.model_dump(),
    )


@app.command("profile")
def profile(
    model_path: Path = typer.Argument(..., exists=True, readable=True),
    device: str = typer.Option("cpu"),
    batch_size: int = typer.Option(1, min=1),
    input_shape: str | None = typer.Option(None),
    warmup: int = typer.Option(10, min=0),
    iterations: int = typer.Option(50, min=1),
    allow_pickle: bool = typer.Option(False),
    layers: bool = typer.Option(True, help="Include layer-wise profiling."),
    output: Path | None = typer.Option(None, help="Write JSON results to this path."),
    output_format: str = typer.Option("json", "--format", help="json or csv (with --output)"),
    no_energy: bool = typer.Option(False),
) -> None:
    """Run a unified efficiency profile."""
    from greenai.benchmark import BenchmarkConfig, run_benchmark
    from greenai.hardware import detect_hardware
    from greenai.models.pytorch import PyTorchModelAdapter
    from greenai.optimization.recommendations import generate_recommendations
    from greenai.profiling.bottlenecks import analyze_bottlenecks
    from greenai.profiling.layer_profiler import profile_layers
    from greenai.reporting.csv import write_csv_summary
    from greenai.reporting.json import write_json
    from greenai.reporting.schema import ProfileResult, build_metadata
    from greenai.reporting.terminal import (
        print_layers,
        print_profile_summary,
        print_recommendations,
    )
    from greenai.scoring.green_score import compute_green_score

    shape = _parse_shape(input_shape)
    adapter = PyTorchModelAdapter(
        model_path,
        device=device,
        allow_pickle=allow_pickle,
        example_input_shape=shape,
    )
    bench = run_benchmark(
        adapter,
        BenchmarkConfig(
            device=device,
            batch_size=batch_size,
            input_shape=shape,
            warmup=warmup,
            iterations=iterations,
            measure_energy=not no_energy,
        ),
    )

    layer_result = None
    bottleneck_result = None
    if layers:
        model = adapter.load()
        example = adapter.create_example_input(batch_size, shape)
        layer_result = profile_layers(
            model,
            example,
            device=device,
            warmup=max(1, warmup // 2),
            iterations=max(5, iterations // 2),
        )
        bottleneck_result = analyze_bottlenecks(layer_result)

    document = ProfileResult(
        metadata=build_metadata(
            __version__,
            notes=[
                "Latency uses perf_counter with CUDA synchronize when device is cuda.",
                "Energy is measured only via supported backends; never fabricated.",
            ],
        ),
        model=bench.model.model_dump(),
        hardware=detect_hardware().model_dump(),
        benchmark={
            "latency": bench.latency.model_dump(),
            "memory": bench.memory.model_dump(),
            "energy": None if bench.energy is None else bench.energy.model_dump(),
            "config": bench.config.model_dump(),
        },
        layers=None if layer_result is None else layer_result.model_dump(),
        bottlenecks=None if bottleneck_result is None else bottleneck_result.model_dump(),
    )
    payload = document.to_dict()
    recs = generate_recommendations(payload)
    payload["recommendations"] = recs.model_dump()
    payload["green_score"] = compute_green_score(payload).model_dump()

    if output is not None:
        if output_format == "csv":
            write_csv_summary(payload, output)
        else:
            write_json(payload, output)
        typer.echo(f"Wrote {output}")

    print_profile_summary(payload)
    if layer_result is not None:
        print_layers([layer.model_dump() for layer in layer_result.layers])
    print_recommendations([rec.model_dump() for rec in recs.recommendations])


@app.command("layer-profile")
def layer_profile(
    model_path: Path = typer.Argument(..., exists=True, readable=True),
    device: str = typer.Option("cpu"),
    batch_size: int = typer.Option(1, min=1),
    input_shape: str | None = typer.Option(None),
    warmup: int = typer.Option(5, min=0),
    iterations: int = typer.Option(20, min=1),
    allow_pickle: bool = typer.Option(False),
) -> None:
    """Profile leaf-module latency for a PyTorch model."""
    from greenai.models.pytorch import PyTorchModelAdapter
    from greenai.profiling.bottlenecks import analyze_bottlenecks
    from greenai.profiling.layer_profiler import profile_layers
    from greenai.reporting.terminal import console, print_layers

    adapter = PyTorchModelAdapter(
        model_path,
        device=device,
        allow_pickle=allow_pickle,
        example_input_shape=_parse_shape(input_shape),
    )
    model = adapter.load()
    example = adapter.create_example_input(batch_size, _parse_shape(input_shape))
    result = profile_layers(
        model,
        example,
        device=device,
        warmup=warmup,
        iterations=iterations,
    )
    print_layers([layer.model_dump() for layer in result.layers])
    report = analyze_bottlenecks(result)
    console.print("\n[bold]Performance Bottlenecks[/bold]")
    for idx, item in enumerate(report.bottlenecks, start=1):
        console.print(f"{idx}. {item.name}")
        console.print(f"   {item.detail} [{item.category}]")


@app.command("compare")
def compare(
    baseline: Path = typer.Argument(..., exists=True, readable=True),
    optimized: Path = typer.Argument(..., exists=True, readable=True),
) -> None:
    """Compare two JSON profile results."""
    from greenai.comparison import compare_results
    from greenai.reporting.json import read_json
    from greenai.reporting.terminal import print_comparison

    result = compare_results(read_json(baseline), read_json(optimized))
    print_comparison([m.model_dump() for m in result.metrics])
    for note in result.notes:
        typer.echo(f"Note: {note}")


@app.command("report")
def report(
    results: Path = typer.Argument(..., exists=True, readable=True, help="Profile JSON file."),
    html_output: Path | None = typer.Option(None, "--html", help="Optional HTML output path."),
) -> None:
    """Render a terminal report (and optional HTML) from saved JSON results."""
    from greenai.reporting.html import write_html
    from greenai.reporting.json import read_json
    from greenai.reporting.terminal import (
        print_layers,
        print_profile_summary,
        print_recommendations,
    )

    data = read_json(results)
    print_profile_summary(data)
    layers = ((data.get("layers") or {}).get("layers")) or []
    if layers:
        print_layers(layers)
    recs = ((data.get("recommendations") or {}).get("recommendations")) or []
    if recs:
        print_recommendations(recs)
    if html_output is not None:
        write_html(data, html_output)
        typer.echo(f"Wrote {html_output}")


def _parse_shape(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    parts = [p.strip() for p in value.split(",")]
    try:
        return [int(p) for p in parts]
    except ValueError as exc:
        raise typer.BadParameter("input shape must be comma-separated integers") from exc


if __name__ == "__main__":
    app()
