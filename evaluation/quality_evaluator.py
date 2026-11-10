
"""Transparent rubric-based evaluation of response quality."""


def score_with_rubric(
    relevance: int,
    accuracy: int,
    clarity: int,
    completeness: int,
) -> float:
    """Calculate the average of four human-assigned scores."""

    scores = [relevance, accuracy, clarity, completeness]

    if any(
        isinstance(score, bool)
        or not isinstance(score, int)
        or not 1 <= score <= 5
        for score in scores
    ):
        raise ValueError(
            "Every rubric score must be an integer from 1 to 5."
        )

    return round(sum(scores) / len(scores), 2)


def normalize_score(score: float) -> float:
    """Convert a 1–5 score to a 0–1 scale."""

    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not 1 <= score <= 5
    ):
        raise ValueError("Score must be between 1 and 5.")

    return round((score - 1) / 4, 4)


def evaluate_response(
    relevance: int,
    accuracy: int,
    clarity: int,
    completeness: int,
) -> dict:
    """Return a structured human-based quality evaluation."""

    average_score = score_with_rubric(
        relevance,
        accuracy,
        clarity,
        completeness,
    )

    return {
        "relevance": relevance,
        "accuracy": accuracy,
        "clarity": clarity,
        "completeness": completeness,
        "average_score": average_score,
        "normalized_score": normalize_score(average_score),
        "evaluation_type": "human_rubric",
    }
