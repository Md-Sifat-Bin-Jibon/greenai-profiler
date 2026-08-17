"""Shared exceptions for Green AI Profiler."""

from __future__ import annotations


class GreenAIError(Exception):
    """Base error for the greenai package."""


class ModelLoadError(GreenAIError):
    """Raised when a model cannot be loaded safely or successfully."""


class UnsupportedHardwareError(GreenAIError):
    """Raised when a requested hardware capability is unavailable."""


class BenchmarkError(GreenAIError):
    """Raised when a benchmark cannot be completed."""


class SchemaError(GreenAIError):
    """Raised when a result document fails schema validation."""
