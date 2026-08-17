"""Formatting and schema helpers."""

from __future__ import annotations

from greenai.reporting.schema import SCHEMA_VERSION, build_metadata, validate_result_dict
from greenai.scoring.green_score import compute_green_score
from greenai.utils.formatting import format_bytes, format_params, pct_change


def test_formatters() -> None:
    assert format_bytes(1024) == "1.00 KB"
    assert format_params(3_500_000) == "3.50M"
    assert pct_change(10, 5) == -50.0


def test_schema_roundtrip() -> None:
    meta = build_metadata("0.1.0")
    assert meta.schema_version == SCHEMA_VERSION
    doc = {
        "schema_version": SCHEMA_VERSION,
        "metadata": meta.model_dump(),
        "model": {"name": "x"},
        "hardware": {},
        "benchmark": {},
    }
    parsed = validate_result_dict(doc)
    assert parsed.model["name"] == "x"


def test_green_score_transparent() -> None:
    score = compute_green_score(
        {
            "model": {"size_bytes": 10_000_000},
            "benchmark": {
                "latency": {"mean_seconds": 0.01},
                "memory": {"cpu_rss_bytes": 100_000_000},
                "energy": {"status": "unavailable"},
            },
        }
    )
    assert 0 <= score.score <= 100
    assert "energy" in score.components
    assert score.limitations
