# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- GitHub Actions release workflow for PyPI Trusted Publisher (`release.yml`).
- `python -m greenai` entry point for hosts where the console script is not on PATH.

### Fixed

- CI type-check (`mypy`) on Linux: Windows-only `winreg` access in CPU detection
  is now resolved dynamically instead of failing `attr-defined` checks.

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
- GitHub CI / license / Python badges and canonical repo URLs.
- `SECURITY.md` and a pull request template.
- `state_dict` loading via `PyTorchModelAdapter(..., architecture=...)` with
  clearer errors for weights-only vs pickled modules.
- Example workflow: `examples/state_dict_workflow.py`.
- CLI tests for `profile`, `benchmark`, `compare`, `report`, and pickle refusal.
- Mid-run energy `poll()` during timed benchmark iterations (NVML sampling).

### Changed

- Package metadata (`[project.urls]`) now points at
  `Md-Sifat-Bin-Jibon/greenai-profiler`.

### Notes

- Energy values are never fabricated. Unavailable hardware reports an explicit status.
- Pickle-based checkpoint loading is opt-in and documented as a security risk.
