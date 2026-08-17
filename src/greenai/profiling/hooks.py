"""Forward-hook utilities for layer profiling."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from greenai.utils.timing import synchronize_if_cuda


@dataclass
class LayerSample:
    name: str
    module_type: str
    latency_seconds: float
    input_shape: list[int] | None = None
    output_shape: list[int] | None = None
    parameter_count: int | None = None
    dtype: str | None = None


@dataclass
class HookRecorder:
    device: str = "cpu"
    samples: list[LayerSample] = field(default_factory=list)
    _starts: dict[int, float] = field(default_factory=dict)

    def pre_hook(self, name: str) -> Callable[..., None]:
        def _pre(module: Any, inputs: Any) -> None:
            synchronize_if_cuda(self.device)
            self._starts[id(module)] = time.perf_counter()

        return _pre

    def forward_hook(self, name: str) -> Callable[..., None]:
        def _fwd(module: Any, inputs: Any, output: Any) -> None:
            synchronize_if_cuda(self.device)
            start = self._starts.pop(id(module), None)
            if start is None:
                return
            elapsed = time.perf_counter() - start
            in_shape = _first_shape(inputs)
            out_shape = _first_shape(output)
            params = sum(int(p.numel()) for p in module.parameters(recurse=False))
            dtype = None
            try:
                first = next(module.parameters(recurse=False))
                dtype = str(first.dtype).replace("torch.", "")
            except StopIteration:
                dtype = None
            self.samples.append(
                LayerSample(
                    name=name,
                    module_type=module.__class__.__name__,
                    latency_seconds=elapsed,
                    input_shape=in_shape,
                    output_shape=out_shape,
                    parameter_count=params,
                    dtype=dtype,
                )
            )

        return _fwd


def _first_shape(obj: Any) -> list[int] | None:
    if obj is None:
        return None
    if hasattr(obj, "shape"):
        try:
            return [int(x) for x in obj.shape]
        except Exception:
            return None
    if isinstance(obj, (tuple, list)) and obj:
        return _first_shape(obj[0])
    return None
