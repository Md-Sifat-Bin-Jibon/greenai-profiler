# Contributing to Green AI Profiler

Thanks for contributing. This project aims to be a reliable, scientifically careful
toolkit for measuring model efficiency—not a collection of optimistic estimates.

Repository: [github.com/Md-Sifat-Bin-Jibon/greenai-profiler](https://github.com/Md-Sifat-Bin-Jibon/greenai-profiler)

## Development setup

```bash
git clone https://github.com/Md-Sifat-Bin-Jibon/greenai-profiler.git
cd greenai-profiler

python -m venv .venv
# Windows
.venv\Scripts\activate
# Unix
source .venv/bin/activate

pip install -e ".[dev,torch]"
```

## Checks before opening a PR

```bash
ruff check src tests
ruff format --check src tests
mypy src
pytest
```

## Guidelines

- Prefer clear type hints and small modules.
- Never invent energy or power numbers. Label estimates explicitly.
- Keep hardware-specific code under `greenai.hardware`.
- Add or update tests for behavior changes.
- Update `CHANGELOG.md` for user-visible changes.
- Document security-sensitive paths (especially model loading).

## Pull requests

1. Open a focused PR with a clear description of *why*.
2. Link related issues when applicable.
3. Ensure CI is green.

## Security reports

Do not file public issues for vulnerabilities. See [SECURITY.md](SECURITY.md).

## Repository listing (maintainers)

GitHub **About**:

> Open-source PyTorch AI model profiler for latency, memory, energy (NVML/RAPL), and layer-wise bottlenecks. Measure green AI / ML efficiency—no fabricated joules.

GitHub **Topics**: `pytorch` `green-ai` `sustainable-ai` `model-profiler` `energy-efficiency` `inference-latency` `ml-benchmark` `layer-profiling` `mlops` `deep-learning` `onnx` `nvml` `machine-learning` `performance` `carbon-aware`

## Code of conduct

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
