
import pytest

from preprocessing.advanced_techniques import (
    ADVANCED_TECHNIQUES,
    apply_advanced_technique,
    detect_language,
)


SAMPLE_TEXT = (
    "MACHINE learning is a powerfull technology. "
    "Contact test@example.com. "
    "Machine learning helps computers learn."
)


def test_all_20_techniques_are_registered():
    assert len(ADVANCED_TECHNIQUES) == 20


@pytest.mark.parametrize(
    "technique",
    list(ADVANCED_TECHNIQUES.keys()),
)
def test_each_advanced_technique_executes(technique):
    result = apply_advanced_technique(
        technique,
        SAMPLE_TEXT,
    )

    assert isinstance(result, str)


def test_spelling_correction():
    result = apply_advanced_technique(
        "spelling_correction",
        "This is a powerfull technology.",
    )
    assert "powerful" in result


def test_pii_removal():
    result = apply_advanced_technique(
        "pii_removal",
        "Contact test@example.com",
    )
    assert "test@example.com" not in result
    assert "[REDACTED]" in result


def test_language_detection_tamil():
    assert detect_language("வணக்கம்") == "Tamil"


def test_text_truncation():
    result = apply_advanced_technique(
        "text_truncation",
        "one two three four five",
        max_length=3,
        mode="words",
    )
    assert result == "one two three"


def test_unknown_technique():
    with pytest.raises(ValueError):
        apply_advanced_technique(
            "unknown_technique",
            SAMPLE_TEXT,
        )
