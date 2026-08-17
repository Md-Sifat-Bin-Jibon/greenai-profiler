"""PyTorch model loading and inspection.

Security note: ``torch.load`` can execute arbitrary code via pickle.
By default this adapter only accepts ``weights_only=True`` loads when supported,
or state-dict style checkpoints. Full pickle loading requires
``allow_pickle=True`` and emits an explicit warning.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from greenai.exceptions import ModelLoadError
from greenai.models.base import BaseModelAdapter, ModelInfo


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ModelLoadError(
            "PyTorch is required for this operation. "
            "Install with: pip install 'green-ai-profiler[torch]'"
        ) from exc
    return torch


def count_parameters(model: Any) -> tuple[int, int]:
    total = 0
    trainable = 0
    for param in model.parameters():
        n = int(param.numel())
        total += n
        if param.requires_grad:
            trainable += n
    return total, trainable


def model_size_bytes(model: Any) -> int:
    total = 0
    for param in model.parameters():
        total += int(param.nelement()) * int(param.element_size())
    for buf in model.buffers():
        total += int(buf.nelement()) * int(buf.element_size())
    return total


def collect_dtypes(model: Any) -> list[str]:
    dtypes: set[str] = set()
    for param in model.parameters():
        dtypes.add(str(param.dtype).replace("torch.", ""))
    for buf in model.buffers():
        dtypes.add(str(buf.dtype).replace("torch.", ""))
    return sorted(dtypes)


def architecture_summary(model: Any, max_lines: int = 40) -> str:
    lines = str(model).splitlines()
    if len(lines) > max_lines:
        omitted = len(lines) - max_lines
        lines = [*lines[:max_lines], f"... ({omitted} more lines omitted)"]
    return "\n".join(lines)


_STATE_DICT_KEYS = ("state_dict", "model_state_dict", "model", "module")


def _is_tensor_map(obj: Any, torch: Any) -> bool:
    if not isinstance(obj, dict) or not obj:
        return False
    return all(isinstance(value, torch.Tensor) for value in obj.values())


def _extract_state_dict(obj: Any, torch: Any) -> Any | None:
    """Return a tensor mapping if ``obj`` looks like a weights-only checkpoint."""
    if _is_tensor_map(obj, torch):
        return obj
    if not isinstance(obj, dict):
        return None
    for key in _STATE_DICT_KEYS:
        inner = obj.get(key)
        if isinstance(inner, torch.nn.Module):
            continue
        if _is_tensor_map(inner, torch):
            return inner
    return None


class PyTorchModelAdapter(BaseModelAdapter):
    """Adapter for ``nn.Module`` instances and ``.pt`` / ``.pth`` checkpoints."""

    framework = "pytorch"

    def __init__(
        self,
        source: str | Path | Any,
        *,
        device: str = "cpu",
        allow_pickle: bool = False,
        example_input_shape: list[int] | None = None,
        name: str | None = None,
        architecture: Any | None = None,
    ) -> None:
        self.source = source
        self.device = device
        self.allow_pickle = allow_pickle
        self.example_input_shape = example_input_shape
        self.architecture = architecture
        self._name = name
        self._model: Any | None = None
        self._path: str | None = None
        self._notes: list[str] = []

    def load(self) -> Any:
        if self._model is not None:
            return self._model

        torch = _require_torch()

        if not isinstance(self.source, (str, Path)):
            model = self.source
            if not isinstance(model, torch.nn.Module):
                raise ModelLoadError("In-memory source must be a torch.nn.Module.")
            self._model = model.to(self.device)
            self._model.eval()
            if self._name is None:
                self._name = model.__class__.__name__
            return self._model

        path = Path(self.source)
        self._path = str(path)
        if not path.exists():
            raise ModelLoadError(f"Model file not found: {path}")

        obj = self._torch_load(path, torch)
        model = self._coerce_module(obj, torch)
        self._model = model.to(self.device)
        self._model.eval()
        if self._name is None:
            self._name = model.__class__.__name__
        return self._model

    def _torch_load(self, path: Path, torch: Any) -> Any:
        if self.allow_pickle:
            warnings.warn(
                "Loading with allow_pickle=True deserializes a pickle stream and may "
                "execute arbitrary code. Only use this for files you fully trust.",
                UserWarning,
                stacklevel=2,
            )
            self._notes.append("Loaded with allow_pickle=True (unsafe pickle deserialization).")
            return torch.load(path, map_location="cpu", weights_only=False)

        # Prefer weights_only when available (PyTorch >= 2.0).
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            # Older torch without weights_only — refuse by default.
            raise ModelLoadError(
                "This PyTorch build cannot load with weights_only=True. "
                "Upgrade PyTorch or pass allow_pickle=True only for trusted files."
            ) from None
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load checkpoint safely from {path}. "
                "This usually means the file is a pickled nn.Module (not a state_dict). "
                "If you fully trust the file, re-run with --allow-pickle. "
                "Safer path: save model.state_dict() and load it into a reconstructed "
                "architecture (see examples/state_dict_workflow.py). "
                f"Underlying error: {exc}"
            ) from exc

    def _coerce_module(self, obj: Any, torch: Any) -> Any:
        if isinstance(obj, torch.nn.Module):
            return obj
        if isinstance(obj, dict):
            for key in ("model", "module"):
                if key in obj and isinstance(obj[key], torch.nn.Module):
                    return obj[key]
            state = _extract_state_dict(obj, torch)
            if state is not None:
                return self._load_state_dict(state, torch)
            raise ModelLoadError(
                "Checkpoint is a dict without an nn.Module or a recognizable state_dict. "
                "Reconstruct the model in Python and pass architecture=..., "
                "or see examples/state_dict_workflow.py."
            )
        raise ModelLoadError(f"Unsupported checkpoint type: {type(obj)!r}")

    def _load_state_dict(self, state: Any, torch: Any) -> Any:
        if self.architecture is None:
            raise ModelLoadError(
                "Checkpoint is a state_dict (weights only), not a full nn.Module. "
                "Reconstruct the model in Python and pass it as architecture=..., "
                "or see examples/state_dict_workflow.py. "
                "--allow-pickle will not help: there is no serialized module to unpickle."
            )
        if not isinstance(self.architecture, torch.nn.Module):
            raise ModelLoadError("architecture must be a torch.nn.Module.")
        try:
            incompatible = self.architecture.load_state_dict(state)
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load state_dict into the provided architecture: {exc}"
            ) from exc
        missing = list(getattr(incompatible, "missing_keys", []) or [])
        unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
        if missing or unexpected:
            raise ModelLoadError(
                "state_dict does not match the provided architecture. "
                f"Missing keys: {missing[:8]}; unexpected keys: {unexpected[:8]}."
            )
        self._notes.append("Loaded weights into user-provided architecture from state_dict.")
        if self._name is None:
            self._name = self.architecture.__class__.__name__
        return self.architecture

    def inspect(self) -> ModelInfo:
        model = self.load()
        total, trainable = count_parameters(model)
        size = model_size_bytes(model)
        dtypes = collect_dtypes(model)
        input_shape = self.example_input_shape
        output_shape = None
        notes = list(self._notes)

        if input_shape is not None:
            try:
                example = self.create_example_input(1, input_shape)
                with _require_torch().no_grad():
                    out = model(example)
                if hasattr(out, "shape"):
                    output_shape = [int(x) for x in out.shape]
                elif isinstance(out, (tuple, list)) and out and hasattr(out[0], "shape"):
                    output_shape = [int(x) for x in out[0].shape]
            except Exception as exc:
                notes.append(f"Could not infer output shape: {exc}")

        return ModelInfo(
            name=self._name,
            framework=self.framework,
            path=self._path,
            parameter_count=total,
            trainable_parameter_count=trainable,
            size_bytes=size,
            dtypes=dtypes,
            input_shape=input_shape,
            output_shape=output_shape,
            device=self.device,
            architecture_summary=architecture_summary(model),
            notes=notes,
        )

    def create_example_input(self, batch_size: int, input_shape: list[int] | None) -> Any:
        torch = _require_torch()
        shape = input_shape or self.example_input_shape
        if shape is None:
            # Sensible default for vision-style models; callers should override.
            shape = [3, 224, 224]
            self._notes.append(
                "No input shape provided; defaulting to [3, 224, 224] for synthetic inputs."
            )
        full = [batch_size, *shape]
        dtype = torch.float32
        model = self.load()
        # Prefer first parameter dtype when available.
        try:
            first = next(model.parameters())
            dtype = first.dtype
        except StopIteration:
            pass
        return torch.randn(*full, device=self.device, dtype=dtype)
