"""Pruning placeholders — structural pruning is opt-in and model-specific."""

from __future__ import annotations

from typing import Any

from greenai.exceptions import GreenAIError


def magnitude_prune_linear(model: Any, amount: float = 0.2) -> Any:
    """Apply unstructured magnitude pruning to Linear layers.

    This modifies the model in-place via PyTorch pruning utilities.
    """
    if amount <= 0 or amount >= 1:
        raise GreenAIError("Pruning amount must be in (0, 1).")
    try:
        import torch
        import torch.nn.utils.prune as prune
    except ImportError as exc:
        raise GreenAIError("PyTorch is required for pruning.") from exc

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
    model.eval()
    return model
