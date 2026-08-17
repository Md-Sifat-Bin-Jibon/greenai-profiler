# Architecture

## Goals

Green AI Profiler is a modular toolkit for measuring model efficiency metrics that
matter in practice: latency, throughput, memory, energy (when measurable), and
layer-wise latency. The design prioritizes honesty over completeness—unavailable
measurements are reported as unavailable.

## Package layout

```text
src/greenai/
  cli.py                 # Typer CLI
  models/                # Framework adapters (PyTorch first, ONNX inspect)
  benchmark/             # Latency, memory, runner, statistics
  hardware/              # Detection + energy backends
  profiling/             # Layer hooks + bottleneck analysis
  optimization/          # FP16/INT8/pruning helpers + recommendations
  comparison/            # Result diffing
  reporting/             # Terminal/JSON/CSV/HTML + schema
  evaluation/            # User-supplied accuracy interface
  scoring/               # Transparent composite Green Score
  utils/                 # Formatting and timing
```

## Design choices

1. **Adapters over framework lock-in** — `BaseModelAdapter` isolates load/inspect/input creation.
2. **Dependency injection for energy** — `EnergyMonitor` implementations are selectable and mockable.
3. **Versioned results** — `schema_version: "1.0"` keeps historical JSON useful.
4. **Hardware isolation** — NVML/RAPL live under `hardware/`; CI does not require GPUs.
5. **No fabricated energy** — `UnsupportedEnergyMonitor` is a first-class outcome.

## Data flow for `greenai profile`

1. Load model via adapter (safe-by-default deserialization).
2. Detect hardware snapshot.
3. Run warmup + timed iterations (CUDA sync when needed).
4. Capture memory stats.
5. Stop energy monitor (or record unavailable).
6. Optionally profile leaf modules.
7. Emit recommendations + optional Green Score.
8. Render terminal output and optional JSON/CSV.

## Non-goals (current release)

- Claiming scientifically authoritative “green” rankings
- Silent pickle execution
- Pretending desktop hosts can measure Android energy
