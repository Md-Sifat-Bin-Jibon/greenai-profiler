# Security policy

Please report vulnerabilities privately. Do not open a public issue for
security-sensitive reports (especially anything involving model loading,
pickle deserialization, or path handling).

## Report a vulnerability

Use [GitHub private vulnerability reporting](https://github.com/Md-Sifat-Bin-Jibon/greenai-profiler/security/advisories/new)
for this repository.

Include:

- A description of the issue and impact
- Steps to reproduce, or a proof of concept
- Affected version / commit if known

We will acknowledge reports and work on a fix before any public disclosure.

## Model loading

Green AI Profiler loads PyTorch checkpoints. `torch.load` can execute arbitrary
code unless `weights_only=True` is used.

- Safe loading is the default.
- `--allow-pickle` / `allow_pickle=True` is an explicit opt-in for trusted files only.
- Prefer `state_dict` plus a reconstructed architecture. See
  [examples/state_dict_workflow.py](examples/state_dict_workflow.py).

Details: [docs/security.md](docs/security.md).
