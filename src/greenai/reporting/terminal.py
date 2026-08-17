"""Rich terminal rendering."""

from __future__ import annotations

import sys
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from greenai.utils.formatting import format_bytes, format_energy_j, format_ms, format_params

# Prefer UTF-8 on Windows consoles that support it; fall back safely.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass

console = Console(legacy_windows=False)


def print_system_info(info: dict[str, Any]) -> None:
    lines = [
        f"CPU: {info.get('cpu_name') or 'unknown'}",
        f"CPU Cores: {info.get('cpu_cores_physical') or '?'} physical / "
        f"{info.get('cpu_cores_logical') or '?'} logical",
        f"RAM: {format_bytes(info.get('ram_total_bytes'))}",
        "",
        f"GPU: {info.get('gpu_name') or 'none detected'}",
        f"GPU Memory: {format_bytes(info.get('gpu_memory_bytes'))}",
        f"CUDA: {'Available' if info.get('cuda_available') else 'Unavailable'}",
        "",
        f"OS: {info.get('os_name')} {info.get('os_release')}",
        f"Python: {info.get('python_version')}",
        f"PyTorch: {info.get('pytorch_version') or 'not installed'}",
    ]
    console.print(Panel("\n".join(lines), title="Green AI Profiler", border_style="green"))


def print_model_info(info: dict[str, Any]) -> None:
    table = Table(title="Model Inspection", show_header=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    rows = [
        ("Name", info.get("name") or "unknown"),
        ("Framework", info.get("framework")),
        ("Parameters", format_params(info.get("parameter_count"))),
        ("Trainable", format_params(info.get("trainable_parameter_count"))),
        ("Size", format_bytes(info.get("size_bytes"))),
        ("Dtypes", ", ".join(info.get("dtypes") or []) or "n/a"),
        ("Device", info.get("device") or "n/a"),
        ("Input shape", str(info.get("input_shape") or "unknown")),
        ("Output shape", str(info.get("output_shape") or "unknown")),
    ]
    for key, value in rows:
        table.add_row(key, str(value))
    console.print(table)
    notes = info.get("notes") or []
    for note in notes:
        console.print(f"[yellow]Note:[/yellow] {note}")


def print_benchmark(
    latency: dict[str, Any], memory: dict[str, Any], energy: dict[str, Any] | None
) -> None:
    lat = Table(title="Latency", show_header=False)
    lat.add_column("Metric", style="bold")
    lat.add_column("Value")
    lat.add_row("Mean", format_ms(latency.get("mean_seconds")))
    lat.add_row("Median", format_ms(latency.get("median_seconds")))
    lat.add_row("P95", format_ms(latency.get("p95_seconds")))
    lat.add_row("P99", format_ms(latency.get("p99_seconds")))
    lat.add_row("Min", format_ms(latency.get("min_seconds")))
    lat.add_row("Max", format_ms(latency.get("max_seconds")))
    lat.add_row("StdDev", format_ms(latency.get("stddev_seconds")))
    throughput = latency.get("throughput_samples_per_sec")
    lat.add_row("Throughput", f"{throughput:.2f} samples/sec" if throughput is not None else "n/a")
    console.print(lat)

    mem = Table(title="Memory", show_header=False)
    mem.add_column("Metric", style="bold")
    mem.add_column("Value")
    mem.add_row("CPU RSS", format_bytes(memory.get("cpu_rss_bytes")))
    mem.add_row("GPU Allocated", format_bytes(memory.get("gpu_allocated_bytes")))
    mem.add_row("GPU Peak", format_bytes(memory.get("gpu_peak_allocated_bytes")))
    console.print(mem)
    for note in memory.get("notes") or []:
        console.print(f"[yellow]Note:[/yellow] {note}")

    if energy is None:
        return
    en = Table(title="Energy", show_header=False)
    en.add_column("Metric", style="bold")
    en.add_column("Value")
    status = energy.get("status")
    if status == "unavailable":
        en.add_row("Status", "unavailable")
        en.add_row("Reason", str(energy.get("reason") or "n/a"))
    else:
        en.add_row("Measurement", str(energy.get("method")))
        en.add_row("Status", str(status))
        en.add_row("Total Energy", format_energy_j(energy.get("total_energy_joules")))
        en.add_row("Inferences", str(energy.get("inferences")))
        en.add_row("Energy / Inference", format_energy_j(energy.get("energy_per_inference_joules")))
    console.print(en)


def print_profile_summary(result: dict[str, Any]) -> None:
    model = result.get("model") or {}
    bench = result.get("benchmark") or {}
    latency = bench.get("latency") or {}
    memory = bench.get("memory") or {}
    energy = bench.get("energy") or {}

    text = Text()
    text.append("Model\n", style="bold")
    text.append(f"  {model.get('name') or 'unknown'}\n\n")
    text.append("Performance\n", style="bold")
    text.append(f"  Latency:       {format_ms(latency.get('mean_seconds'))}\n")
    text.append(f"  P95:           {format_ms(latency.get('p95_seconds'))}\n")
    thr = latency.get("throughput_samples_per_sec")
    text.append(
        f"  Throughput:    {thr:.2f} samples/sec\n\n"
        if thr is not None
        else "  Throughput:    n/a\n\n"
    )
    text.append("Memory\n", style="bold")
    peak = memory.get("gpu_peak_allocated_bytes") or memory.get("cpu_rss_bytes")
    text.append(f"  Peak/RSS:      {format_bytes(peak)}\n\n")
    text.append("Energy\n", style="bold")
    if (energy.get("status") or "") == "unavailable":
        text.append("  Status:        unavailable\n")
        text.append(f"  Reason:        {energy.get('reason')}\n\n")
    else:
        text.append(
            f"  Per inference: {format_energy_j(energy.get('energy_per_inference_joules'))}\n\n"
        )
    text.append("Model Stats\n", style="bold")
    text.append(f"  Parameters:    {format_params(model.get('parameter_count'))}\n")
    text.append(f"  Size:          {format_bytes(model.get('size_bytes'))}\n")
    console.print(Panel(text, title="Green AI Profiler", border_style="green"))


def print_layers(layers: list[dict[str, Any]], limit: int = 20) -> None:
    table = Table(title="Layer Latency")
    table.add_column("Layer")
    table.add_column("Type")
    table.add_column("Latency", justify="right")
    table.add_column("%", justify="right")
    for entry in layers[:limit]:
        table.add_row(
            str(entry.get("name")),
            str(entry.get("module_type")),
            format_ms(entry.get("latency_seconds")),
            f"{float(entry.get('latency_percent') or 0):.1f}%",
        )
    console.print(table)


def print_comparison(metrics: list[dict[str, Any]]) -> None:
    table = Table(title="Comparison")
    table.add_column("Metric")
    table.add_column("Baseline", justify="right")
    table.add_column("Optimized", justify="right")
    table.add_column("Change", justify="right")
    for item in metrics:
        change = item.get("change_percent")
        change_s = "n/a" if change is None else f"{change:+.1f}%"
        table.add_row(
            str(item.get("name")),
            _fmt(item.get("baseline"), item.get("unit")),
            _fmt(item.get("optimized"), item.get("unit")),
            change_s,
        )
    console.print(table)


def _fmt(value: Any, unit: str | None) -> str:
    if value is None:
        return "n/a"
    unit = unit or ""
    if unit == "bytes":
        return format_bytes(float(value))
    if unit == "s":
        return format_ms(float(value))
    if unit == "J":
        return format_energy_j(float(value))
    if unit == "samples/s":
        return f"{float(value):.2f}"
    if unit == "ratio":
        return f"{float(value) * 100:.2f}%"
    if unit == "count":
        return format_params(int(value))
    return str(value)


def print_recommendations(recs: list[dict[str, Any]]) -> None:
    console.print("[bold]Recommendations[/bold]")
    for rec in recs:
        console.print(f"* {rec.get('title')}")
        console.print(f"  {rec.get('detail')}")
        console.print(f"  [dim]Evidence: {rec.get('evidence')}[/dim]")
