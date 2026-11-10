
"""Deduplication utilities for LLM input preprocessing.

Provides offline duplicate detection for lines, sentences, and items.
Preserves the first occurrence and original text formatting.
"""

import re
import unicodedata
from typing import Iterable


def _validate_text(text: str) -> None:
    """Validate text input."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")


def _validate_boolean(name: str, value: bool) -> None:
    """Validate boolean options."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")


def _normalize_for_comparison(
    text: str,
    *,
    case_sensitive: bool = False,
    normalize_whitespace: bool = True,
) -> str:
    """Normalize text for comparison without changing output text."""
    _validate_text(text)
    _validate_boolean("case_sensitive", case_sensitive)
    _validate_boolean("normalize_whitespace", normalize_whitespace)

    text = unicodedata.normalize("NFC", text)

    if normalize_whitespace:
        text = re.sub(r"\s+", " ", text).strip()

    return text if case_sensitive else text.casefold()


def remove_duplicate_lines(
    text: str,
    *,
    case_sensitive: bool = False,
    preserve_empty_lines: bool = False,
) -> str:
    """Remove repeated lines while retaining their first occurrence.

    Args:
        text: Input text.
        case_sensitive: Treat differently cased lines as distinct.
        preserve_empty_lines: Preserve blank lines in the output.
    """
    _validate_text(text)
    _validate_boolean("case_sensitive", case_sensitive)
    _validate_boolean("preserve_empty_lines", preserve_empty_lines)

    seen: set[str] = set()
    result: list[str] = []

    for line in text.splitlines():
        key = _normalize_for_comparison(
            line,
            case_sensitive=case_sensitive,
        )

        if not key:
            if preserve_empty_lines:
                result.append(line)
            continue

        if key not in seen:
            seen.add(key)
            result.append(line)

    return "\n".join(result)


def remove_duplicate_sentences(
    text: str,
    *,
    case_sensitive: bool = False,
) -> str:
    """Remove repeated sentences while retaining their first occurrence.

    Sentence splitting uses punctuation followed by whitespace.
    It is intentionally lightweight and may not handle every
    abbreviation or language correctly.
    """
    _validate_text(text)
    _validate_boolean("case_sensitive", case_sensitive)

    if not text.strip():
        return ""

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip(),
    )

    seen: set[str] = set()
    result: list[str] = []

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        key = _normalize_for_comparison(
            sentence,
            case_sensitive=case_sensitive,
        )

        if key not in seen:
            seen.add(key)
            result.append(sentence)

    return " ".join(result)


def remove_duplicate_items(
    items: Iterable[str],
    *,
    case_sensitive: bool = False,
) -> list[str]:
    """Remove duplicate strings while preserving original order.

    Empty strings are retained only once.
    """
    _validate_boolean("case_sensitive", case_sensitive)

    if isinstance(items, (str, bytes)):
        raise TypeError(
            "items must be an iterable of strings, not a string"
        )

    try:
        iterator = iter(items)
    except TypeError:
        raise TypeError(
            "items must be an iterable of strings"
        ) from None

    seen: set[str] = set()
    result: list[str] = []

    for item in iterator:
        if not isinstance(item, str):
            raise TypeError("all items must be strings")

        key = _normalize_for_comparison(
            item,
            case_sensitive=case_sensitive,
        )

        if key not in seen:
            seen.add(key)
            result.append(item)

    return result


def deduplicate_query(text: str) -> str:
    """Conservatively remove repeated query lines.

    Semantic similarity is not evaluated.
    """
    return remove_duplicate_lines(
        text,
        case_sensitive=False,
        preserve_empty_lines=False,
    )


def deduplicate_text(
    text: str,
    *,
    remove_lines: bool = True,
    remove_sentences: bool = False,
) -> str:
    """Apply configurable deduplication to LLM input text.

    Line deduplication is enabled by default.
    Sentence deduplication is optional because simple sentence
    splitting can affect abbreviations and language-specific text.
    """
    _validate_text(text)
    _validate_boolean("remove_lines", remove_lines)
    _validate_boolean("remove_sentences", remove_sentences)

    if remove_lines:
        text = remove_duplicate_lines(text)

    if remove_sentences:
        text = remove_duplicate_sentences(text)

    return text.strip()


__all__ = [
    "_normalize_for_comparison",
    "remove_duplicate_lines",
    "remove_duplicate_sentences",
    "remove_duplicate_items",
    "deduplicate_query",
    "deduplicate_text",
]
