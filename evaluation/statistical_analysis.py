
"""Descriptive statistics for experiment measurements."""

import math
import statistics


def summarize(values) -> dict:
    """Calculate descriptive statistics for numeric values."""

    if values is None:
        values = []

    numeric = []

    for value in values:
        if value is None:
            continue

        if isinstance(value, bool):
            raise ValueError("Boolean values are not valid measurements.")

        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid numeric measurement: {value!r}"
            ) from None

        if not math.isfinite(number):
            raise ValueError(
                "Measurements must be finite numbers."
            )

        numeric.append(number)

    if not numeric:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "stdev": None,
            "min": None,
            "max": None,
        }

    return {
        "count": len(numeric),
        "mean": round(statistics.mean(numeric), 4),
        "median": round(statistics.median(numeric), 4),
        "stdev": (
            round(statistics.stdev(numeric), 4)
            if len(numeric) > 1 else 0.0
        ),
        "min": round(min(numeric), 4),
        "max": round(max(numeric), 4),
    }
