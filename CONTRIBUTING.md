# Contributing to Green AI Profiler

Thanks for contributing. This project aims to be a reliable, scientifically careful
toolkit for measuring model efficiency—not a collection of optimistic estimates.

## Development setup

```bash
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

## Code of conduct

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
