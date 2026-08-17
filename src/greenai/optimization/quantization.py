"""Quantization helpers (dynamic INT8 where supported)."""

from __future__ import annotations

from typing import Any

from greenai.exceptions import ModelLoadError


def dynamic_quantize_linear(model: Any) -> Any:
    """Apply PyTorch dynamic quantization to Linear layers (CPU-oriented)."""
    try:
        import torch
    except ImportError as exc:
        raise ModelLoadError("PyTorch is required for quantization.") from exc

    if not isinstance(model, torch.nn.Module):
        raise ModelLoadError("Quantization expects a torch.nn.Module.")

    quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    quantized.eval()
    return quantized
