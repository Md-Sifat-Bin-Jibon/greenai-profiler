"""User-supplied accuracy evaluation interface.

The profiler does not ship task-specific datasets. Callers provide a callable
that scores a model and returns a float in a documented metric space.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel, Field


class Evaluator(Protocol):
    def __call__(self, model: Any) -> float:
        """Return an accuracy-like score for the model."""


class AccuracyEvaluation(BaseModel):
    metric_name: str
    value: float
    notes: list[str] = Field(default_factory=list)


def evaluate_accuracy(
    model: Any,
    evaluator: Callable[[Any], float],
    *,
    metric_name: str = "accuracy",
    notes: list[str] | None = None,
) -> AccuracyEvaluation:
    """Run a user-provided evaluator and wrap the result."""
    value = float(evaluator(model))
    return AccuracyEvaluation(
        metric_name=metric_name,
        value=value,
        notes=notes
        or [
            "Accuracy is task-specific and only as meaningful as the supplied evaluator.",
        ],
    )


def accuracy_delta(before: AccuracyEvaluation, after: AccuracyEvaluation) -> dict[str, float]:
    return {
        "accuracy_before": before.value,
        "accuracy_after": after.value,
        "accuracy_delta": after.value - before.value,
    }
