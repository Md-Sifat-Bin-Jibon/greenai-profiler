# Security considerations

## Model loading

- Default path uses `torch.load(..., weights_only=True)` when available.
- `--allow-pickle` enables full unpickling and prints a warning.
- Do not load untrusted checkpoints.
- Prefer `torch.save(model.state_dict(), path)` plus a reconstructed `nn.Module`.
  Load weights with `PyTorchModelAdapter(..., architecture=model)`.
  See [examples/state_dict_workflow.py](../examples/state_dict_workflow.py).
- A weights-only checkpoint cannot be profiled from the CLI until you reconstruct
  the architecture in Python; the error message explains this instead of implying
  `--allow-pickle` will help.
- Public reporting process: [SECURITY.md](../SECURITY.md).

## Paths

- CLI arguments use Typer path checks (`exists`, `readable`).
- Report writers create parent directories but do not execute user strings.

## HTML reports

- Values are HTML-escaped before interpolation.

## Subprocess usage

- Hardware detection may call `sysctl` on macOS only with a fixed argument list.
- No shell=`True` invocation of user input.

## Generative metadata

- Model metadata is never executed as code or shell commands.
