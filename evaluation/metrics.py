
"""Descriptive metrics for LLM response evaluation."""

import re
from typing import Optional


def response_metrics(response: str) -> dict:
    """Calculate descriptive metrics for an LLM response."""

    text = (response or "").strip()

    words = re.findall(r"\b[\w'-]+\b", text.lower())

    sentences = [
        sentence.strip()
        for sentence in re.split(r"[.!?]+", text)
        if sentence.strip()
    ]

    word_count = len(words)
    unique_words = len(set(words))

    return {
        "quality_score": None,
        "character_count": len(text),
        "word_count": word_count,
        "sentence_count": len(sentences),
        "lexical_diversity": (
            round(unique_words / word_count, 4)
            if word_count else 0.0
        ),
        "average_word_length": (
            round(
                sum(len(word) for word in words) / word_count,
                2,
            )
            if word_count else 0.0
        ),
    }


def compare_texts(
    baseline: str,
    candidate: str,
) -> dict:
    """Compare baseline and preprocessing chatbot responses."""

    baseline_metrics = response_metrics(baseline)
    candidate_metrics = response_metrics(candidate)

    baseline_length = baseline_metrics["character_count"]
    candidate_length = candidate_metrics["character_count"]

    return {
        "baseline_length": baseline_length,
        "candidate_length": candidate_length,
        "length_delta": candidate_length - baseline_length,
        "baseline_word_count": baseline_metrics["word_count"],
        "candidate_word_count": candidate_metrics["word_count"],
        "baseline_lexical_diversity": (
            baseline_metrics["lexical_diversity"]
        ),
        "candidate_lexical_diversity": (
            candidate_metrics["lexical_diversity"]
        ),
    }


def calculate_improvement(
    baseline: Optional[float],
    candidate: Optional[float],
) -> Optional[float]:
    """Calculate percentage improvement when baseline is nonzero."""

    if baseline is None or candidate is None:
        return None

    if baseline == 0:
        return None

    return round(
        ((candidate - baseline) / abs(baseline)) * 100,
        2,
    )
