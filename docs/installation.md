# Installation

## Requirements

- Python 3.11+
- Optional: PyTorch, ONNX, NVIDIA driver + `nvidia-ml-py`, matplotlib

## Install from PyPI (recommended)

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Unix

pip install "green-ai-profiler[torch]"
```

## Editable install (development)

```bash
git clone https://github.com/Md-Sifat-Bin-Jibon/greenai-profiler.git
cd greenai-profiler

python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Unix

pip install -e ".[dev,torch]"
```

## Optional extras

```bash
pip install "green-ai-profiler[onnx]"
pip install "green-ai-profiler[nvidia]"
pip install "green-ai-profiler[viz]"
pip install "green-ai-profiler[all]"
```

## Verify

```bash
greenai --version
greenai --help
greenai system-info
```

## Maintainers: publishing a release

1. Ensure version in `pyproject.toml` is bumped (PyPI rejects re-uploads).
2. Create a GitHub Environment named exactly `pypi`
   (Settings → Environments → New environment).
3. Push a tag or GitHub Release, e.g. `v0.1.0`.
4. `.github/workflows/release.yml` publishes via Trusted Publisher (OIDC).
   No API token is required when the pending publisher matches this workflow.
