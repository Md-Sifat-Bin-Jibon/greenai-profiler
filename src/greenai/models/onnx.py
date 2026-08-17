"""ONNX model adapter (inspection-focused in this release)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from greenai.exceptions import ModelLoadError
from greenai.models.base import BaseModelAdapter, ModelInfo


class OnnxModelAdapter(BaseModelAdapter):
    """Minimal ONNX inspection adapter."""

    framework = "onnx"

    def __init__(self, source: str | Path, *, name: str | None = None) -> None:
        self.source = Path(source)
        self._name = name or self.source.stem
        self._model: Any | None = None

    def load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import onnx
        except ImportError as exc:
            raise ModelLoadError(
                "ONNX is required. Install with: pip install 'green-ai-profiler[onnx]'"
            ) from exc
        if not self.source.exists():
            raise ModelLoadError(f"ONNX file not found: {self.source}")
        self._model = onnx.load(str(self.source))
        return self._model

    def inspect(self) -> ModelInfo:
        model = self.load()
        size = self.source.stat().st_size
        input_shape = None
        try:
            dims = model.graph.input[0].type.tensor_type.shape.dim
            input_shape = [int(d.dim_value) if d.dim_value else -1 for d in dims]
        except Exception:
            input_shape = None
        return ModelInfo(
            name=self._name,
            framework=self.framework,
            path=str(self.source),
            parameter_count=None,
            trainable_parameter_count=None,
            size_bytes=size,
            dtypes=[],
            input_shape=input_shape,
            output_shape=None,
            device=None,
            architecture_summary=f"ONNX graph with {len(model.graph.node)} nodes",
            notes=["ONNX runtime benchmarking is limited in this release."],
        )

    def create_example_input(self, batch_size: int, input_shape: list[int] | None) -> Any:
        raise ModelLoadError("ONNX synthetic input creation is not implemented in this release.")
