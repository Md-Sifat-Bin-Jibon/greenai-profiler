"""Self-contained HTML report generation."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from greenai.utils.formatting import format_bytes, format_energy_j, format_ms, format_params


def _e(value: Any) -> str:
    return html.escape(str(value))


def render_html(result: dict[str, Any]) -> str:
    model = result.get("model") or {}
    hardware = result.get("hardware") or {}
    bench = result.get("benchmark") or {}
    latency = bench.get("latency") or {}
    memory = bench.get("memory") or {}
    energy = bench.get("energy") or {}
    layers = ((result.get("layers") or {}).get("layers")) or []
    recs = ((result.get("recommendations") or {}).get("recommendations")) or []

    layer_rows_parts: list[str] = []
    for layer in layers[:30]:
        pct = f"{float(layer.get('latency_percent') or 0):.1f}%"
        layer_rows_parts.append(
            "<tr>"
            f"<td>{_e(layer.get('name'))}</td>"
            f"<td>{_e(format_ms(layer.get('latency_seconds')))}</td>"
            f"<td>{_e(pct)}</td>"
            "</tr>"
        )
    layer_rows = "\n".join(layer_rows_parts)
    rec_items = "\n".join(
        f"<li><strong>{_e(r.get('title'))}</strong> — {_e(r.get('detail'))} "
        f"<em>({_e(r.get('evidence'))})</em></li>"
        for r in recs
    )

    energy_block = (
        f"<p>Status: unavailable — {_e(energy.get('reason'))}</p>"
        if (energy.get("status") or "") == "unavailable"
        else (
            f"<p>Method: {_e(energy.get('method'))}<br/>"
            f"Per inference: {_e(format_energy_j(energy.get('energy_per_inference_joules')))}</p>"
        )
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Green AI Profiler Report</title>
<style>
body {{ font-family: Georgia, "Times New Roman", serif; margin: 2rem; color: #122; background: #f7faf7; }}
h1,h2 {{ font-family: "Segoe UI", sans-serif; color: #0b3d2e; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; background: white; }}
th,td {{ border: 1px solid #cfe0d6; padding: 0.5rem 0.75rem; text-align: left; }}
th {{ background: #e5f2ea; }}
.card {{ background: white; border: 1px solid #cfe0d6; padding: 1rem 1.25rem; margin-bottom: 1rem; }}
</style>
</head>
<body>
<h1>Green AI Profiler Report</h1>
<div class="card">
<h2>Model</h2>
<p><strong>{_e(model.get("name") or "unknown")}</strong><br/>
Parameters: {_e(format_params(model.get("parameter_count")))}<br/>
Size: {_e(format_bytes(model.get("size_bytes")))}</p>
</div>
<div class="card">
<h2>Hardware</h2>
<p>CPU: {_e(hardware.get("cpu_name"))}<br/>
GPU: {_e(hardware.get("gpu_name") or "none")}<br/>
CUDA: {_e(hardware.get("cuda_available"))}<br/>
Python: {_e(hardware.get("python_version"))}</p>
</div>
<div class="card">
<h2>Latency</h2>
<p>Mean: {_e(format_ms(latency.get("mean_seconds")))}<br/>
P95: {_e(format_ms(latency.get("p95_seconds")))}<br/>
Throughput: {_e(latency.get("throughput_samples_per_sec"))} samples/sec</p>
</div>
<div class="card">
<h2>Memory</h2>
<p>CPU RSS: {_e(format_bytes(memory.get("cpu_rss_bytes")))}<br/>
GPU Peak: {_e(format_bytes(memory.get("gpu_peak_allocated_bytes")))}</p>
</div>
<div class="card">
<h2>Energy</h2>
{energy_block}
</div>
<div class="card">
<h2>Layer Bottlenecks</h2>
<table><thead><tr><th>Layer</th><th>Latency</th><th>%</th></tr></thead>
<tbody>
{layer_rows or '<tr><td colspan="3">No layer data</td></tr>'}
</tbody></table>
</div>
<div class="card">
<h2>Recommendations</h2>
<ul>
{rec_items or "<li>None</li>"}
</ul>
</div>
</body>
</html>
"""


def write_html(result: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_html(result), encoding="utf-8")
