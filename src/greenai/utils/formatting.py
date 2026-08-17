"""Common formatting helpers."""

from __future__ import annotations


def format_bytes(num_bytes: float | None) -> str:
    """Format a byte count into a human-readable string."""
    if num_bytes is None:
        return "unavailable"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def format_params(count: int | None) -> str:
    """Format a parameter count (e.g. 3.50M)."""
    if count is None:
        return "unavailable"
    abs_count = abs(count)
    if abs_count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.2f}B"
    if abs_count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M"
    if abs_count >= 1_000:
        return f"{count / 1_000:.2f}K"
    return str(count)


def format_ms(seconds: float | None) -> str:
    """Format seconds as milliseconds."""
    if seconds is None:
        return "unavailable"
    return f"{seconds * 1000.0:.2f} ms"


def format_energy_j(joules: float | None) -> str:
    """Format energy in joules with an appropriate unit."""
    if joules is None:
        return "unavailable"
    if abs(joules) < 1.0:
        return f"{joules * 1000.0:.2f} mJ"
    return f"{joules:.2f} J"


def pct_change(baseline: float | None, current: float | None) -> float | None:
    """Return percentage change from baseline to current, or None if undefined."""
    if baseline is None or current is None or baseline == 0:
        return None
    return ((current - baseline) / baseline) * 100.0
