
"""Noise filtering utilities for LLM text preprocessing.

Provides offline utilities to remove unwanted control characters,
repeated characters, excessive punctuation, and placeholder noise.

All operations use Python's standard library.
"""

import re
import unicodedata


def _validate_text(text: str) -> None:
    """Validate that the input is a string."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")


def remove_control_noise(text: str) -> str:
    """Remove unwanted control characters while preserving normal whitespace."""
    _validate_text(text)

    return "".join(
        char
        for char in text
        if char in "\n\r\t"
        or unicodedata.category(char) != "Cc"
    )


def remove_repeated_characters(
    text: str,
    max_repeats: int = 3,
) -> str:
    """Limit excessively repeated consecutive characters.

    Example:
        Hellooooooo!!! -> Hellooo!!!
    """
    _validate_text(text)

    if isinstance(max_repeats, bool) or not isinstance(max_repeats, int):
        raise TypeError("max_repeats must be an integer")

    if max_repeats < 1:
        raise ValueError("max_repeats must be at least 1")

    pattern = re.compile(
        r"(.)\1{" + str(max_repeats) + r",}",
        flags=re.DOTALL,
    )

    return pattern.sub(
        lambda match: match.group(1) * max_repeats,
        text,
    )


def remove_excessive_punctuation(text: str) -> str:
    """Reduce repeated punctuation while preserving sentence boundaries."""
    _validate_text(text)

    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r",{2,}", ",", text)
    text = re.sub(r";{2,}", ";", text)
    text = re.sub(r"\.{4,}", "...", text)

    return text


def remove_placeholder_noise(text: str) -> str:
    """Remove recognized standalone noise placeholders.

    Examples:
        [noise]
        [unintelligible]
        <noise>
    """
    _validate_text(text)

    patterns = [
        r"(?<!\w)\[noise\](?!\w)",
        r"(?<!\w)\[unintelligible\](?!\w)",
        r"(?<!\w)<noise>(?!\w)",
    ]

    for pattern in patterns:
        text = re.sub(
            pattern,
            " ",
            text,
            flags=re.IGNORECASE,
        )

    return text


def filter_noise(
    text: str,
    *,
    remove_control_characters: bool = True,
    limit_repeated_characters: bool = False,
    max_repeats: int = 3,
    reduce_punctuation: bool = True,
    remove_placeholders: bool = False,
) -> str:
    """Apply configurable offline noise filtering to input text.

    Conservative defaults help preserve the user's original meaning.

    Args:
        text: Input text.
        remove_control_characters: Remove unwanted control characters.
        limit_repeated_characters: Limit consecutive repeated characters.
        max_repeats: Maximum allowed consecutive repetitions.
        reduce_punctuation: Reduce repeated punctuation.
        remove_placeholders: Remove recognized noise placeholders.

    Returns:
        Filtered text.
    """
    _validate_text(text)

    # Validate options before processing.
    for option_name, option_value in (
        ("remove_control_characters", remove_control_characters),
        ("limit_repeated_characters", limit_repeated_characters),
        ("reduce_punctuation", reduce_punctuation),
        ("remove_placeholders", remove_placeholders),
    ):
        if not isinstance(option_value, bool):
            raise TypeError(f"{option_name} must be a boolean")

    if isinstance(max_repeats, bool) or not isinstance(max_repeats, int):
        raise TypeError("max_repeats must be an integer")

    if max_repeats < 1:
        raise ValueError("max_repeats must be at least 1")

    # Normalize Unicode representations.
    text = unicodedata.normalize("NFC", text)

    # Remove unwanted control characters.
    if remove_control_characters:
        text = remove_control_noise(text)

    # Limit repeated characters when enabled.
    if limit_repeated_characters:
        text = remove_repeated_characters(
            text,
            max_repeats=max_repeats,
        )

    # Reduce excessive punctuation.
    if reduce_punctuation:
        text = remove_excessive_punctuation(text)

    # Remove recognized placeholder noise.
    if remove_placeholders:
        text = remove_placeholder_noise(text)

    # Normalize horizontal whitespace while preserving line breaks.
    text = re.sub(r"[^\S\n]+", " ", text)

    # Remove spaces around line breaks.
    text = re.sub(r" *\n *", "\n", text)

    # Limit excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


__all__ = [
    "remove_control_noise",
    "remove_repeated_characters",
    "remove_excessive_punctuation",
    "remove_placeholder_noise",
    "filter_noise",
]
