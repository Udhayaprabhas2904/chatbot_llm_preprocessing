
import pytest

from evaluation.compare_results import compare_response_pair


def test_compare_response_pair():
    baseline = {
        "model": "llama3.2:3b",
        "query": "Explain Python",
        "condition": "baseline",
        "response": "Python is a programming language.",
        "processing_time_seconds": 2.0,
        "quality_score": 3.5,
    }

    preprocessed = {
        "model": "llama3.2:3b",
        "query": "Explain Python",
        "condition": "preprocessed",
        "response": "Python is a popular programming language.",
        "processing_time_seconds": 1.5,
        "quality_score": 4.5,
    }

    result = compare_response_pair(baseline, preprocessed)

    assert result["quality_score_delta"] == 1.0
    assert result["latency_delta_seconds"] == -0.5
    assert result["baseline_quality_score"] == 3.5


def test_compare_rejects_different_models():
    baseline = {"model": "model-a", "query": "Hello"}
    preprocessed = {"model": "model-b", "query": "Hello"}

    with pytest.raises(ValueError):
        compare_response_pair(baseline, preprocessed)


def test_quality_delta_is_none_when_scores_missing():
    baseline = {"model": "model-a", "query": "Hello"}
    preprocessed = {"model": "model-a", "query": "Hello"}

    result = compare_response_pair(baseline, preprocessed)

    assert result["quality_score_delta"] is None
