# Quickstart

```bash
pip install -e ".[torch,dev]"
python examples/save_example_model.py
greenai system-info
greenai inspect examples/artifacts/tiny_cnn.pt --allow-pickle --input-shape 1,28,28
greenai benchmark examples/artifacts/tiny_cnn.pt --allow-pickle --input-shape 1,28,28 --iterations 30
greenai profile examples/artifacts/tiny_cnn.pt --allow-pickle --input-shape 1,28,28 --output results.json
greenai report results.json --html report.html
```

Security: only pass `--allow-pickle` for checkpoints you trust.
