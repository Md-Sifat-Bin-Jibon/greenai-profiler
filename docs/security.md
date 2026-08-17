# Security considerations

## Model loading

- Default path uses `torch.load(..., weights_only=True)` when available.
- `--allow-pickle` enables full unpickling and prints a warning.
- Do not load untrusted checkpoints.

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
