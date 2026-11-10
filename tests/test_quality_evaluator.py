
import pytest

from evaluation.quality_evaluator import (
    score_with_rubric,
    normalize_score,
    evaluate_response,
)


def test_score_with_rubric():
    result = score_with_rubric(4, 5, 4, 5)
    assert result == 4.5


def test_score_with_rubric_rejects_invalid_score():
    with pytest.raises(ValueError):
        score_with_rubric(6, 4, 5, 3)


def test_normalize_score():
    assert normalize_score(5) == 1.0
    assert normalize_score(1) == 0.0


def test_normalize_score_rejects_invalid_score():
    with pytest.raises(ValueError):
        normalize_score(6)


def test_evaluate_response():
    result = evaluate_response(4, 5, 4, 5)

    assert result["average_score"] == 4.5
    assert result["normalized_score"] == 0.875
    assert result["evaluation_type"] == "human_rubric"
