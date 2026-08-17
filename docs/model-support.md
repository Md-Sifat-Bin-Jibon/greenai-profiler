# Model support

## PyTorch

- In-memory `nn.Module`
- `state_dict` checkpoints via `architecture=` (see `examples/state_dict_workflow.py`)
- Checkpoints loadable with `weights_only=True`
- Full pickled modules only with `--allow-pickle`

## ONNX

- Graph inspection (node count, input dims, file size)
- Runtime benchmarking is limited in this release

## Security

`torch.load` without `weights_only` can execute arbitrary code. Prefer saving
`state_dict` plus reconstruction code, or trusted full-module checkpoints with
an explicit opt-in flag.
