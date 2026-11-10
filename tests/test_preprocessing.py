
from preprocessing.pipeline import preprocess_text


def test_preprocessing_returns_nonempty_text():
    result = preprocess_text("  Explain machine learning.  ")
    assert isinstance(result, str)
    assert result.strip()


def test_preprocessing_is_repeatable():
    text = "Explain   machine learning."
    assert preprocess_text(text) == preprocess_text(text)
