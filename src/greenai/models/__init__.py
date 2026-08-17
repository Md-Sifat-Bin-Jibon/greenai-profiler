"""Model adapters."""

from __future__ import annotations

from greenai.models.base import BaseModelAdapter, ModelInfo
from greenai.models.pytorch import PyTorchModelAdapter

__all__ = [
    "BaseModelAdapter",
    "ModelInfo",
    "PyTorchModelAdapter",
]
