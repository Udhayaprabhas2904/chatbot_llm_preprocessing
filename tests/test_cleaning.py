
import pytest

from preprocessing.cleaning import (
    normalize_encoding,
    normalize_unicode,
    normalize_whitespace,
    clean_html,
    decode_html_entities,
    handle_special_characters,
    clean_text,
    clean_query,
    clean_response,
    remove_extra_punctuation,
)

from preprocessing.pipeline import PreprocessingPipeline


# ---------------------------------------------------------
# Encoding normalization
# ---------------------------------------------------------

def test_normalize_encoding_from_bytes():
    result = normalize_encoding(b"Hello", encoding="utf-8")
    assert result == "Hello"


def test_normalize_encoding_from_string():
    assert normalize_encoding("Hello") == "Hello"


def test_normalize_encoding_invalid_input():
    with pytest.raises(TypeError):
        normalize_encoding(123)


def test_normalize_encoding_invalid_encoding():
    with pytest.raises(ValueError):
        normalize_encoding(b"Hello", encoding="invalid-encoding")


# ---------------------------------------------------------
# Unicode normalization
# ---------------------------------------------------------

def test_unicode_nfc_normalization():
    result = normalize_unicode("cafe\u0301", "NFC")
    assert result == "café"


def test_unicode_nfkc_normalization():
    result = normalize_unicode("ＨＥＬＬＯ", "NFKC")
    assert result == "HELLO"


def test_unicode_invalid_form():
    with pytest.raises(ValueError):
        normalize_unicode("Hello", "INVALID")


# ---------------------------------------------------------
# Whitespace normalization
# ---------------------------------------------------------

def test_whitespace_normalization():
    result = normalize_whitespace("Hello    World")
    assert result == "Hello World"


def test_whitespace_preserve_line_breaks():
    result = normalize_whitespace(
        "Hello   World\n\n\nPython",
        preserve_line_breaks=True,
    )
    assert result == "Hello World\n\nPython"


# ---------------------------------------------------------
# HTML cleaning
# ---------------------------------------------------------

def test_html_cleaning():
    result = clean_html("<p>Hello <b>World</b></p>")
    assert result == "Hello World"


def test_html_removes_script_and_style():
    text = (
        "<p>Hello</p>"
        "<script>alert('test')</script>"
        "<style>.abc { color: red; }</style>"
    )

    result = clean_html(text)

    assert "Hello" in result
    assert "alert" not in result
    assert "color" not in result


def test_html_removes_comments():
    result = clean_html("Hello<!-- hidden -->World")
    assert "hidden" not in result
    assert "Hello" in result
    assert "World" in result


# ---------------------------------------------------------
# HTML entity decoding
# ---------------------------------------------------------

def test_html_entity_decoding():
    result = decode_html_entities(
        "Tom &amp; Jerry &lt;Friends&gt;"
    )
    assert result == "Tom & Jerry <Friends>"


# ---------------------------------------------------------
# Special character handling
# ---------------------------------------------------------

def test_special_character_replacement():
    result = handle_special_characters(
        "Hello @World!",
        mode="replace",
    )
    assert result == "Hello  World!"


def test_special_character_removal():
    result = handle_special_characters(
        "Hello @World!",
        mode="remove",
    )
    assert result == "Hello World!"


def test_special_character_preservation():
    text = "Hello @World!"
    assert handle_special_characters(
        text,
        mode="preserve",
    ) == text


def test_special_character_invalid_mode():
    with pytest.raises(ValueError):
        handle_special_characters("Hello", mode="invalid")


# ---------------------------------------------------------
# General cleaning
# ---------------------------------------------------------

def test_clean_text():
    result = clean_text(
        "  Hello    World!!!  "
    )
    assert result == "Hello World!!!"


def test_clean_query():
    result = clean_query("  Hello     World!!!  ")
    assert result == "Hello World!"


def test_remove_extra_punctuation():
    result = remove_extra_punctuation("Hello!!!! Really???")
    assert result == "Hello! Really?"


def test_clean_response():
    result = clean_response("Hello\r\n   World")
    assert result == "Hello World"


def test_clean_response_preserves_paragraphs():
    result = clean_response(
        "First paragraph.\n\nSecond paragraph.",
        preserve_line_breaks=True,
    )
    assert result == "First paragraph.\n\nSecond paragraph."


# ---------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "technique",
    [
        "cleaning",
        "whitespace_normalization",
        "unicode_normalization",
        "html_cleaning",
        "html_entity_decoding",
        "special_character_handling",
        "encoding_normalization",
    ],
)
def test_stage_one_techniques_are_available(technique):
    assert technique in PreprocessingPipeline.AVAILABLE_TECHNIQUES


def test_pipeline_html_cleaning():
    pipeline = PreprocessingPipeline(
        techniques=["html_cleaning"]
    )

    result = pipeline.process("<p>Hello World</p>")

    assert result.processed_query == "Hello World"
    assert "html_cleaning" in result.applied_techniques


def test_pipeline_whitespace_normalization():
    pipeline = PreprocessingPipeline(
        techniques=["whitespace_normalization"]
    )

    result = pipeline.process("Hello     World")

    assert result.processed_query == "Hello World"


def test_pipeline_records_transformations():
    pipeline = PreprocessingPipeline(
        techniques=[
            "html_cleaning",
            "whitespace_normalization",
        ]
    )

    result = pipeline.process("<p>Hello     World</p>")

    assert result.changed is True
    assert len(result.metadata["transformation_log"]) == 2
    assert result.metadata["transformation_log"][0]["changed"] is True
