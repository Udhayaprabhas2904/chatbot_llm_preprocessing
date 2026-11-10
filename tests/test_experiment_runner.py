
import pytest

from evaluation.experiment_runner import build_experiment_matrix


def test_build_experiment_matrix_returns_all_combinations():
    queries = ["Hello", "Explain Python"]
    models = ["llama3.2:3b"]

    result = build_experiment_matrix(queries, models)

    assert len(result) == 4
    assert result[0]["condition"] == "baseline"
    assert result[1]["condition"] == "preprocessed"


def test_build_experiment_matrix_rejects_empty_queries():
    with pytest.raises(ValueError):
        build_experiment_matrix([], ["llama3.2:3b"])


def test_build_experiment_matrix_rejects_empty_models():
    with pytest.raises(ValueError):
        build_experiment_matrix(["Hello"], [])


def test_build_experiment_matrix_rejects_invalid_conditions():
    with pytest.raises(ValueError):
        build_experiment_matrix(
            ["Hello"],
            ["llama3.2:3b"],
            conditions=("baseline", "invalid"),
        )


def test_build_experiment_matrix_rejects_duplicate_conditions():
    with pytest.raises(ValueError):
        build_experiment_matrix(
            ["Hello"],
            ["llama3.2:3b"],
            conditions=("baseline", "baseline"),
        )
