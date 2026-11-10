
"""Comparison helpers for paired baseline and preprocessed results."""

from evaluation.metrics import compare_texts


def compare_response_pair(
    baseline_row: dict,
    preprocessed_row: dict,
) -> dict:
    """Compare two responses for the same query and model."""

    if baseline_row.get("model") != preprocessed_row.get("model"):
        raise ValueError("Both responses must use the same model.")

    if baseline_row.get("query") != preprocessed_row.get("query"):
        raise ValueError("Both responses must use the same query.")

    if baseline_row.get("condition") not in (None, "baseline"):
        raise ValueError("First response must be a baseline result.")

    if preprocessed_row.get("condition") not in (None, "preprocessed"):
        raise ValueError("Second response must be a preprocessed result.")

    baseline_time = float(
        baseline_row.get("processing_time_seconds") or 0
    )
    preprocessed_time = float(
        preprocessed_row.get("processing_time_seconds") or 0
    )

    baseline_quality = baseline_row.get("quality_score")
    preprocessed_quality = preprocessed_row.get("quality_score")

    quality_delta = None

    if baseline_quality is not None and preprocessed_quality is not None:
        quality_delta = round(
            float(preprocessed_quality) - float(baseline_quality),
            4,
        )

    return {
        "model": baseline_row.get("model"),
        "query": baseline_row.get("query"),

        **compare_texts(
            baseline_row.get("response", ""),
            preprocessed_row.get("response", ""),
        ),

        "baseline_time": baseline_time,
        "preprocessed_time": preprocessed_time,

        "latency_delta_seconds": round(
            preprocessed_time - baseline_time, 4
        ),

        "baseline_quality_score": baseline_quality,
        "preprocessed_quality_score": preprocessed_quality,
        "quality_score_delta": quality_delta,
    }
