"""CSV export for flat summary metrics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), nested, out)
    elif isinstance(value, list):
        out[prefix] = json_safe_list(value)
    else:
        out[prefix] = value


def json_safe_list(values: list[Any]) -> str:
    return ";".join(str(v) for v in values)


def write_csv_summary(data: dict[str, Any], path: str | Path) -> None:
    flat: dict[str, Any] = {}
    for key in ("schema_version", "metadata", "model", "hardware", "benchmark"):
        if key in data:
            _flatten(key, data[key], flat)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)
