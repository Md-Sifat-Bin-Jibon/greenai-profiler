"""Save a tiny CNN for CLI examples.

Uses only ``torch.nn`` built-ins so the checkpoint unpickles without
requiring a custom class on ``sys.path``.
"""

from __future__ import annotations

from pathlib import Path

import torch


def build_tiny_cnn() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(8, 16, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(16, 10),
    )


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "tiny_cnn.pt"
    model = build_tiny_cnn()
    model.eval()
    # Full module checkpoint for the example workflow; requires --allow-pickle.
    torch.save(model, path)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
