
"""Text normalization utilities for LLM preprocessing."""

import re
import unicodedata


# ---------------------------------------------------------
# 1. Unicode Normalization
# ---------------------------------------------------------

def normalize_unicode(
    text: str,
    form: str = "NFKC",
) -> str:
    """
    Normalize Unicode characters into a consistent representation.

    Supported forms: NFC, NFD, NFKC, NFKD.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if form not in {"NFC", "NFD", "NFKC", "NFKD"}:
        raise ValueError(
            "form must be NFC, NFD, NFKC, or NFKD"
        )

    return unicodedata.normalize(form, text)


# ---------------------------------------------------------
# 2. Whitespace Normalization
# ---------------------------------------------------------

def normalize_whitespace(
    text: str,
    *,
    preserve_line_breaks: bool = False,
) -> str:
    """
    Normalize unnecessary whitespace.

    Args:
        text: Input text.
        preserve_line_breaks: Preserve meaningful line breaks.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if preserve_line_breaks:
        # Standardize Windows and Mac line endings.
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Normalize spaces and tabs without removing newlines.
        text = re.sub(r"[^\S\n]+", " ", text)

        # Remove unnecessary spaces around line breaks.
        text = re.sub(r" *\n *", "\n", text)

        # Limit consecutive blank lines.
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    # Convert all whitespace sequences into one space.
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------
# 3. Case Normalization
# ---------------------------------------------------------

def normalize_case(
    text: str,
    mode: str = "preserve",
) -> str:
    """
    Normalize letter case.

    Supported modes:
        preserve : Keep original capitalization.
        lower    : Convert to lowercase.
        upper    : Convert to uppercase.
        casefold : Apply Unicode-aware case folding.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if mode == "preserve":
        return text

    if mode == "lower":
        return text.lower()

    if mode == "upper":
        return text.upper()

    if mode == "casefold":
        return text.casefold()

    raise ValueError(
        "mode must be preserve, lower, upper, or casefold"
    )


# ---------------------------------------------------------
# 4. Quote Normalization
# ---------------------------------------------------------

def normalize_quotes(text: str) -> str:
    """
    Convert common Unicode quotation marks to ASCII quotes.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2032": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2033": '"',
    }

    return text.translate(str.maketrans(replacements))


# ---------------------------------------------------------
# 5. Accent Normalization
# ---------------------------------------------------------

def normalize_accents(
    text: str,
    *,
    remove_accents: bool = False,
) -> str:
    """
    Optionally remove combining accent marks.

    Disabled by default to preserve original meaning.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not remove_accents:
        return text

    decomposed = unicodedata.normalize("NFD", text)

    result = "".join(
        char
        for char in decomposed
        if unicodedata.category(char) != "Mn"
    )

    return unicodedata.normalize("NFC", result)


# ---------------------------------------------------------
# 6. Combined Query Normalization
# ---------------------------------------------------------

def normalize_query(
    text: str,
    *,
    unicode_form: str = "NFC",
    case_mode: str = "preserve",
    normalize_quotes_enabled: bool = True,
    preserve_line_breaks: bool = False,
    remove_accents: bool = False,
) -> str:
    """
    Apply configurable normalization to an LLM query.

    Default settings preserve capitalization and accents.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    # 1. Unicode normalization.
    text = normalize_unicode(
        text,
        form=unicode_form,
    )

    # 2. Quote normalization.
    if normalize_quotes_enabled:
        text = normalize_quotes(text)

    # 3. Optional accent removal.
    text = normalize_accents(
        text,
        remove_accents=remove_accents,
    )

    # 4. Whitespace normalization.
    text = normalize_whitespace(
        text,
        preserve_line_breaks=preserve_line_breaks,
    )

    # 5. Case normalization.
    text = normalize_case(
        text,
        mode=case_mode,
    )

    return text


# ---------------------------------------------------------
# 7. Backward-Compatible Function
# ---------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normalize text using default settings.

    Retained for existing modules and tests.
    """
    return normalize_query(text)


# ---------------------------------------------------------
# 8. Public Module Interface
# ---------------------------------------------------------

__all__ = [
    "normalize_unicode",
    "normalize_whitespace",
    "normalize_case",
    "normalize_quotes",
    "normalize_accents",
    "normalize_query",
    "normalize_text",
]
