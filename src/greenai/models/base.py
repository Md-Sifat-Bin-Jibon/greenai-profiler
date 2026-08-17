"""Model abstraction layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Inspection result for a loaded model."""

    name: str | None = None
    framework: str
    path: str | None = None
    parameter_count: int | None = None
    trainable_parameter_count: int | None = None
    size_bytes: int | None = None
    dtypes: list[str] = Field(default_factory=list)
    input_shape: list[int] | None = None
    output_shape: list[int] | None = None
    device: str | None = None
    architecture_summary: str | None = None
    notes: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class BaseModelAdapter(ABC):
    """Common interface for framework-specific model adapters."""

    framework: str

    @abstractmethod
    def load(self) -> Any:
        """Load and return the underlying model object."""

    @abstractmethod
    def inspect(self) -> ModelInfo:
        """Return structured model metadata."""

    @abstractmethod
    def create_example_input(self, batch_size: int, input_shape: list[int] | None) -> Any:
        """Create a synthetic input tensor/batch for benchmarking."""
