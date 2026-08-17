"""Versioned profile result schema."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"


class AccuracyResult(BaseModel):
    value: float | None = None
    metric_name: str | None = None
    notes: list[str] = Field(default_factory=list)


class ProfileMetadata(BaseModel):
    schema_version: str = SCHEMA_VERSION
    tool_version: str
    timestamp: str
    measurement_method_notes: list[str] = Field(default_factory=list)


class ProfileResult(BaseModel):
    """Stable machine-readable profiling document."""

    schema_version: str = SCHEMA_VERSION
    metadata: ProfileMetadata
    model: dict[str, Any] = Field(default_factory=dict)
    hardware: dict[str, Any] = Field(default_factory=dict)
    benchmark: dict[str, Any] = Field(default_factory=dict)
    layers: dict[str, Any] | None = None
    bottlenecks: dict[str, Any] | None = None
    recommendations: dict[str, Any] | None = None
    accuracy: AccuracyResult | None = None
    green_score: dict[str, Any] | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def build_metadata(tool_version: str, notes: list[str] | None = None) -> ProfileMetadata:
    return ProfileMetadata(
        schema_version=SCHEMA_VERSION,
        tool_version=tool_version,
        timestamp=datetime.now(UTC).isoformat(),
        measurement_method_notes=notes or [],
    )


def validate_result_dict(data: dict[str, Any]) -> ProfileResult:
    return ProfileResult.model_validate(data)
