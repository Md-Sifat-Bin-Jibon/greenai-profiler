# Installation

## Requirements

- Python 3.11+
- Optional: PyTorch, ONNX, NVIDIA driver + `nvidia-ml-py`, matplotlib

## Editable install (development)

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Unix

pip install -e ".[dev,torch]"
```

## Optional extras

```bash
pip install -e ".[onnx]"
pip install -e ".[nvidia]"
pip install -e ".[viz]"
pip install -e ".[all]"
```

## Verify

```bash
greenai --version
greenai --help
greenai system-info
```
