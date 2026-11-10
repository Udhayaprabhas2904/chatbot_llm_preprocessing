
import pytest

from evaluation.statistical_analysis import summarize


def test_summarize_numeric_values():
    result = summarize([10, 20, 30])

    assert result["count"] == 3
    assert result["mean"] == 20
    assert result["median"] == 20
    assert result["min"] == 10
    assert result["max"] == 30


def test_summarize_empty_values():
    result = summarize([])

    assert result["count"] == 0
    assert result["mean"] is None


def test_summarize_single_value():
    result = summarize([15])

    assert result["mean"] == 15
    assert result["stdev"] == 0.0


def test_summarize_ignores_none():
    result = summarize([10, None, 20])

    assert result["count"] == 2
    assert result["mean"] == 15


def test_summarize_rejects_invalid_values():
    with pytest.raises(ValueError):
        summarize([10, "invalid", 20])


def test_summarize_rejects_infinite_values():
    with pytest.raises(ValueError):
        summarize([10, float("inf")])


def test_summarize_rejects_boolean_values():
    with pytest.raises(ValueError):
        summarize([10, True])
