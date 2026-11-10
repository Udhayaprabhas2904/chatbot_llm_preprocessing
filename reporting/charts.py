
"""
Professional chart generation for LLM preprocessing experiments.

Charts:
1. Average latency: baseline vs preprocessed.
2. Average response length: baseline vs preprocessed.
3. Successful vs failed requests by model.

Supports both SQLite database field names and API field names.
"""

from pathlib import Path
from collections import defaultdict
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# DATA NORMALIZATION
# ============================================================

def _normalize_record(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize database/API result fields."""

    model = (
        row.get("model")
        or row.get("model_name")
        or "unknown"
    )

    condition = (
        row.get("condition")
        or row.get("condition_name")
        or "unknown"
    )

    condition = str(condition).strip().lower()

    response = (
        row.get("response")
        if row.get("response") is not None
        else row.get("response_text", "")
    )

    response = str(response or "")

    latency = (
        row.get("processing_time_seconds")
        if row.get("processing_time_seconds") is not None
        else row.get("latency_seconds")
    )

    try:
        latency = (
            float(latency)
            if latency is not None
            else None
        )
    except (TypeError, ValueError):
        latency = None

    response_length = row.get("response_length")

    if response_length is None:
        response_length = len(response)

    try:
        response_length = int(response_length)
    except (TypeError, ValueError):
        response_length = len(response)

    error_message = row.get("error_message") or ""

    success_value = row.get("success")

    if success_value is None:
        successful = (
            bool(response.strip())
            and not bool(error_message)
        )
    else:
        if isinstance(success_value, str):
            successful = success_value.strip().lower() in {
                "1", "true", "yes", "success"
            }
        else:
            successful = bool(success_value)

        successful = (
            successful
            and bool(response.strip())
            and not bool(error_message)
        )

    return {
        "model": str(model),
        "condition": condition,
        "response": response,
        "latency": latency,
        "response_length": response_length,
        "successful": successful,
        "error_message": str(error_message),
    }


def _normalize_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize all experiment records."""

    if not rows:
        return []

    return [
        _normalize_record(row)
        for row in rows
    ]


def _condition_label(condition: str) -> str:
    """Convert condition names into readable labels."""

    aliases = {
        "baseline": "Baseline",
        "without_preprocessing": "Baseline",
        "without preprocessing": "Baseline",
        "preprocessed": "Preprocessed",
        "with_preprocessing": "Preprocessed",
        "with preprocessing": "Preprocessed",
    }

    return aliases.get(
        condition,
        condition.replace("_", " ").title(),
    )


def _get_condition_order(conditions: set[str]) -> list[str]:
    """Keep baseline first and preprocessed second."""

    preferred = [
        "baseline",
        "without_preprocessing",
        "without preprocessing",
        "preprocessed",
        "with_preprocessing",
        "with preprocessing",
    ]

    ordered = [
        condition
        for condition in preferred
        if condition in conditions
    ]

    ordered.extend(
        sorted(conditions - set(ordered))
    )

    return ordered


def _prepare_output_dir(output_dir: Path) -> Path:
    """Create output directory if required."""

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir


def _finish_chart(
    fig: Any,
    path: Path,
) -> Path:
    """Save and close a Matplotlib figure."""

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    return path


# ============================================================
# CHART 1: AVERAGE LATENCY
# ============================================================

def create_latency_chart(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> Optional[Path]:
    """
    Compare average successful response latency by model
    and experiment condition.
    """

    records = _normalize_rows(rows)

    if not records:
        return None

    grouped = defaultdict(list)

    for record in records:

        if not record["successful"]:
            continue

        if record["latency"] is None:
            continue

        key = (
            record["model"],
            record["condition"],
        )

        grouped[key].append(record["latency"])

    if not grouped:
        return None

    models = sorted({
        model
        for model, condition in grouped
    })

    conditions = _get_condition_order({
        condition
        for model, condition in grouped
    })

    x_positions = list(range(len(models)))

    width = 0.8 / max(len(conditions), 1)

    fig, ax = plt.subplots(figsize=(11, 6))

    for index, condition in enumerate(conditions):

        values = []

        for model in models:

            samples = grouped.get(
                (model, condition),
                [],
            )

            values.append(
                sum(samples) / len(samples)
                if samples
                else 0
            )

        positions = [
            x + (index - (len(conditions) - 1) / 2) * width
            for x in x_positions
        ]

        # Do not plot missing conditions as zero.
        valid_positions = [
            position
            for position, model, value in zip(
                positions,
                models,
                values,
            )
            if (model, condition) in grouped
        ]

        valid_values = [
            value
            for model, value in zip(models, values)
            if (model, condition) in grouped
        ]

        if valid_values:
            ax.bar(
                valid_positions,
                valid_values,
                width,
                label=_condition_label(condition),
            )

    ax.set_title(
        "Average Response Latency: Baseline vs Preprocessed"
    )
    ax.set_xlabel("Language Model")
    ax.set_ylabel("Average latency (seconds)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.legend()
    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    output_dir = _prepare_output_dir(output_dir)

    return _finish_chart(
        fig,
        output_dir / "average_latency.png",
    )


# ============================================================
# CHART 2: RESPONSE LENGTH
# ============================================================

def create_response_length_chart(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> Optional[Path]:
    """
    Compare average successful response character counts
    by model and experiment condition.
    """

    records = _normalize_rows(rows)

    if not records:
        return None

    grouped = defaultdict(list)

    for record in records:

        if not record["successful"]:
            continue

        key = (
            record["model"],
            record["condition"],
        )

        grouped[key].append(
            record["response_length"]
        )

    if not grouped:
        return None

    models = sorted({
        model
        for model, condition in grouped
    })

    conditions = _get_condition_order({
        condition
        for model, condition in grouped
    })

    x_positions = list(range(len(models)))

    width = 0.8 / max(len(conditions), 1)

    fig, ax = plt.subplots(figsize=(11, 6))

    for index, condition in enumerate(conditions):

        values = [
            (
                sum(grouped[(model, condition)])
                / len(grouped[(model, condition)])
            )
            if (model, condition) in grouped
            else 0
            for model in models
        ]

        positions = [
            x + (index - (len(conditions) - 1) / 2) * width
            for x in x_positions
        ]

        valid_positions = [
            position
            for position, model in zip(positions, models)
            if (model, condition) in grouped
        ]

        valid_values = [
            value
            for model, value in zip(models, values)
            if (model, condition) in grouped
        ]

        if valid_values:
            ax.bar(
                valid_positions,
                valid_values,
                width,
                label=_condition_label(condition),
            )

    ax.set_title(
        "Average Response Length: Baseline vs Preprocessed"
    )
    ax.set_xlabel("Language Model")
    ax.set_ylabel("Average response length (characters)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.legend()
    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    output_dir = _prepare_output_dir(output_dir)

    return _finish_chart(
        fig,
        output_dir / "response_length_comparison.png",
    )


# ============================================================
# CHART 3: SUCCESS VS FAILURE
# ============================================================

def create_success_failure_chart(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> Optional[Path]:
    """
    Compare successful and failed request counts by model.
    """

    records = _normalize_rows(rows)

    if not records:
        return None

    grouped = defaultdict(
        lambda: {
            "successful": 0,
            "failed": 0,
        }
    )

    for record in records:

        model = record["model"]

        if record["successful"]:
            grouped[model]["successful"] += 1
        else:
            grouped[model]["failed"] += 1

    if not grouped:
        return None

    models = sorted(grouped)

    success_counts = [
        grouped[model]["successful"]
        for model in models
    ]

    failure_counts = [
        grouped[model]["failed"]
        for model in models
    ]

    x_positions = list(range(len(models)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar(
        [x - width / 2 for x in x_positions],
        success_counts,
        width,
        label="Successful",
    )

    ax.bar(
        [x + width / 2 for x in x_positions],
        failure_counts,
        width,
        label="Failed",
    )

    ax.set_title(
        "Successful vs Failed Model Requests"
    )
    ax.set_xlabel("Language Model")
    ax.set_ylabel("Number of requests")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.legend()
    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    output_dir = _prepare_output_dir(output_dir)

    return _finish_chart(
        fig,
        output_dir / "success_failure_comparison.png",
    )


# ============================================================
# CHART 4: ALL COMPARISON CHARTS
# ============================================================

def create_comparison_charts(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Optional[Path]]:
    """
    Generate all comparison charts.

    Returns a dictionary containing chart names and paths.
    """

    return {
        "latency": create_latency_chart(
            rows,
            output_dir,
        ),

        "response_length": create_response_length_chart(
            rows,
            output_dir,
        ),

        "success_failure": create_success_failure_chart(
            rows,
            output_dir,
        ),
    }


# ============================================================
# PUBLIC EXPORTS
# ============================================================

__all__ = [
    "create_latency_chart",
    "create_response_length_chart",
    "create_success_failure_chart",
    "create_comparison_charts",
]
