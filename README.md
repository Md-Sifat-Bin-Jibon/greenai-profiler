# Green AI Profiler

[![CI](https://github.com/Md-Sifat-Bin-Jibon/greenai-profiler/actions/workflows/ci.yml/badge.svg)](https://github.com/Md-Sifat-Bin-Jibon/greenai-profiler/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/green-ai-profiler.svg)](https://pypi.org/project/green-ai-profiler/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)

**Open-source AI model profiler** for **PyTorch** (and ONNX) that measures **latency**, **throughput**, **memory**, **energy consumption**, and **layer-wise bottlenecks**—so you can build faster, cheaper, and greener models without guessing.

Search terms this project covers: *AI model profiler*, *PyTorch latency benchmark*, *ML energy measurement*, *green AI*, *model efficiency toolkit*, *inference performance profiler*, *NVML / RAPL energy*, *layer-wise profiling*, *sustainable machine learning*.

```text
pip install "green-ai-profiler[torch]"
greenai profile model.pt --allow-pickle --input-shape 1,28,28
```

## What you can measure

| Goal | What Green AI Profiler does |
|---|---|
| **Inference latency & throughput** | Warmup + timed runs on CPU or CUDA |
| **Memory footprint** | Process RSS and CUDA allocator stats |
| **Energy / power** | Real hardware telemetry (NVIDIA NVML, Intel RAPL)—never fabricated joules |
| **Layer bottlenecks** | Per-module latency to find expensive ops |
| **Optimization hints** | Actionable recommendations from profile data |

## Why this exists

Accuracy alone is not enough. ML engineers, researchers, and MLOps teams also need:

- model latency and samples/sec
- memory and peak RSS behavior
- parameter count and model size
- energy per inference and carbon-aware efficiency (when measurable)

Energy is hardware-dependent. This project **never fabricates joules**. If a backend is unavailable, you get an explicit status and reason.

## How people run this

### 1. Install from PyPI (recommended)

**Requirements:** Python 3.11+

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -U pip
pip install "green-ai-profiler[torch]"
```

Useful extras:

| Extra | Install | Purpose |
|---|---|---|
| `torch` | `pip install "green-ai-profiler[torch]"` | Profile PyTorch models (recommended) |
| `onnx` | `pip install "green-ai-profiler[onnx]"` | ONNX inspection / runtime support |
| `nvidia` | `pip install "green-ai-profiler[nvidia]"` | NVIDIA NVML energy via `nvidia-ml-py` |
| `viz` | `pip install "green-ai-profiler[viz]"` | Plotting helpers |
| `dev` | `pip install "green-ai-profiler[dev]"` | Tests and lint tools |
| `all` | `pip install "green-ai-profiler[all]"` | Everything above |

Verify the install:

```bash
greenai --version
greenai --help
greenai system-info
```

### From source (contributors)

```bash
git clone https://github.com/Md-Sifat-Bin-Jibon/greenai-profiler.git
cd greenai-profiler
python -m venv .venv
# activate the venv, then:
pip install -e ".[torch,dev]"
```

### 2. Profile a model (CLI)

Try the built-in example first:

```bash
python examples/save_example_model.py

greenai inspect examples/artifacts/tiny_cnn.pt --allow-pickle --input-shape 1,28,28
greenai benchmark examples/artifacts/tiny_cnn.pt --allow-pickle --input-shape 1,28,28
greenai profile examples/artifacts/tiny_cnn.pt --allow-pickle --input-shape 1,28,28 --output results.json
greenai report results.json --html report.html
```

Then point `greenai` at your own checkpoint:

```bash
# CPU
greenai profile path/to/model.pt --allow-pickle --input-shape 3,224,224

# CUDA (if PyTorch CUDA is available)
greenai profile path/to/model.pt --allow-pickle --device cuda --input-shape 3,224,224

# Save JSON for later comparison / HTML report
greenai profile path/to/model.pt --allow-pickle --input-shape 3,224,224 --output results.json
greenai compare baseline.json optimized.json
greenai report results.json --html report.html
```

`--allow-pickle` is required for full `torch.save(model, ...)` module checkpoints. Only use it for models you trust. Prefer a `state_dict` plus reconstructed architecture — see [`examples/state_dict_workflow.py`](examples/state_dict_workflow.py) and [docs/security.md](docs/security.md).

### 3. Use it from Python

```python
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
```

Runnable copy: [`examples/basic_benchmark.py`](examples/basic_benchmark.py).

### 4. CLI reference

| Command | What it does |
|---|---|
| `greenai system-info` | Detect host CPU/GPU/energy capabilities |
| `greenai inspect <model>` | Model size, parameters, basic structure |
| `greenai benchmark <model>` | Latency, throughput, memory, energy |
| `greenai profile <model>` | Full efficiency profile + layers + recommendations |
| `greenai layer-profile <model>` | Leaf-module latency breakdown |
| `greenai compare <a.json> <b.json>` | Diff two saved profiles |
| `greenai report <results.json>` | Re-print results; optional `--html` |

Common options on profile/benchmark commands:

- `--device cpu|cuda`
- `--batch-size N`
- `--input-shape C,H,W` (no batch dim)
- `--warmup N` / `--iterations N`
- `--allow-pickle`
- `--output path` / `--format json|csv`
- `--no-energy`

## Example output

```text
╭──────── Green AI Profiler ────────╮
Model
  TinyCNN

Performance
  Latency:       0.42 ms
  P95:           0.55 ms
  Throughput:    2380.1 samples/sec

Memory
  Peak/RSS:      184.12 MB

Energy
  Status:        unavailable
  Reason:        No supported hardware energy interface detected.

Model Stats
  Parameters:    11.98K
  Size:          47.92 KB
╰───────────────────────────────────╯
```

## Supported hardware

| Capability | Status |
|---|---|
| CPU latency / RSS | Supported |
| CUDA latency / allocator memory | Supported (with PyTorch CUDA) |
| NVIDIA power → energy (NVML) | Supported when `nvidia-ml-py` + driver allow it |
| Intel RAPL energy | Supported on Linux hosts with powercap |
| Android / edge direct profiling | Planned (import protocol only for now) |
| Exact per-layer energy | Unavailable on typical desktop APIs |

Details: [docs/hardware-support.md](docs/hardware-support.md), [docs/energy-measurement.md](docs/energy-measurement.md).

## Architecture

```text
CLI (Typer)
   │
   ▼
Models ──► Benchmark runner ──► Reporting (terminal / JSON / CSV / HTML)
   │              │
   │              ├── Latency / throughput
   │              ├── Memory (CPU RSS, CUDA)
   │              └── Energy monitors (NVML / RAPL / unavailable)
   │
   └── Layer profiler ──► Bottlenecks ──► Recommendations
```

See [docs/architecture.md](docs/architecture.md).

## Documentation

| Guide | Link |
|---|---|
| Installation | [docs/installation.md](docs/installation.md) |
| Quickstart | [docs/quickstart.md](docs/quickstart.md) |
| Benchmarking | [docs/benchmarking.md](docs/benchmarking.md) |
| Layer profiling | [docs/layer-profiling.md](docs/layer-profiling.md) |
| Energy measurement | [docs/energy-measurement.md](docs/energy-measurement.md) |
| Limitations | [docs/limitations.md](docs/limitations.md) |
| Security | [SECURITY.md](SECURITY.md), [docs/security.md](docs/security.md) |
| Contributing | [CONTRIBUTING.md](CONTRIBUTING.md) |

## Limitations

- Pickle-based `.pt` files can execute code; safe loading is default, `--allow-pickle` is explicit.
- Process RSS is **not** model-only memory.
- Energy requires real hardware telemetry; estimates are labeled if ever used.
- ONNX support is inspection-oriented in this release.
- Green Score is a **project-defined heuristic**, not a scientific standard.

## Development

```bash
pip install -e ".[dev,torch]"
ruff check src tests
mypy src
pytest
```

## Roadmap

- Broader ONNX Runtime benchmarking
- Static INT8 calibration workflows
- Edge/Android measurement import protocol
- Richer visualization pack
- Expanded hardware energy backends

## License

Apache License 2.0 — see [LICENSE](LICENSE).
