# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-13

### Added

- Project scaffolding with `src` layout and `pyproject.toml`.
- CLI entry point `greenai` with `--help` and `--version`.
- Hardware detection and `greenai system-info`.
- PyTorch model inspection via `greenai inspect`.
- Latency and throughput benchmarking via `greenai benchmark`.
- CPU/GPU memory profiling.
- Extensible energy monitors (NVIDIA telemetry, Intel RAPL, unavailable fallback).
- Unified `greenai profile` with JSON/CSV export.
- Versioned result schema (`schema_version: "1.0"`).
- Layer-wise latency profiling and bottleneck analysis.
- Result comparison via `greenai compare`.
- Evidence-based optimization recommendations.
- Optional FP16 conversion helper.
- Accuracy evaluation interface (user-supplied evaluator).
- Terminal, JSON, CSV, and lightweight HTML reports.
- Documentation, examples, tests, and GitHub Actions CI.

### Notes

- Energy values are never fabricated. Unavailable hardware reports an explicit status.
- Pickle-based checkpoint loading is opt-in and documented as a security risk.
