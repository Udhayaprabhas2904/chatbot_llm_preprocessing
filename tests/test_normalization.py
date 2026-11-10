
from preprocessing.normalization import normalize_text


def test_normalize_text_returns_string():
    result = normalize_text("Hello WORLD")
    assert isinstance(result, str)


def test_normalize_text_is_repeatable():
    text = "Sample TEXT 123"
    assert normalize_text(text) == normalize_text(text)
