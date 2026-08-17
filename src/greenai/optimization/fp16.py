"""Optional FP16 conversion helpers for PyTorch modules."""

from __future__ import annotations

from typing import Any

from greenai.exceptions import ModelLoadError


def to_fp16(model: Any, device: str | None = None) -> Any:
    """Return a half-precision copy of a PyTorch module."""
    try:
        import torch
    except ImportError as exc:
        raise ModelLoadError("PyTorch is required for FP16 conversion.") from exc

    if not isinstance(model, torch.nn.Module):
        raise ModelLoadError("FP16 conversion expects a torch.nn.Module.")

    converted = model.half()
    if device is not None:
        converted = converted.to(device)
    converted.eval()
    return converted
