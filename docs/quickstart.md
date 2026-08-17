# Quickstart

```bash
git clone https://github.com/Md-Sifat-Bin-Jibon/greenai-profiler.git
cd greenai-profiler
pip install -e ".[torch,dev]"
python examples/save_example_model.py
greenai system-info
greenai inspect examples/artifacts/tiny_cnn.pt --allow-pickle --input-shape 1,28,28
greenai benchmark examples/artifacts/tiny_cnn.pt --allow-pickle --input-shape 1,28,28 --iterations 30
greenai profile examples/artifacts/tiny_cnn.pt --allow-pickle --input-shape 1,28,28 --output results.json
greenai report results.json --html report.html
```

Prefer a `state_dict` plus reconstructed architecture when you can:

```bash
python examples/state_dict_workflow.py
```

Security: only pass `--allow-pickle` for checkpoints you trust.
