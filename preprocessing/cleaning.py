
"""
Text Cleaning Module
--------------------

Offline text-cleaning utilities for LLM preprocessing.

Features:
1. Encoding normalization
2. Unicode normalization
3. Whitespace normalization
4. HTML cleaning and entity decoding
5. Special-character handling
6. URL removal
7. Punctuation cleanup
8. Optional filler-word removal
9. Query and response cleaning
10. Control-character removal

Uses Python standard-library modules only.
"""

import html
import re
import unicodedata

from html.parser import HTMLParser
from typing import Optional, Union


# ============================================================
# 1. ENCODING NORMALIZATION
# ============================================================

def normalize_encoding(
    text: Union[str, bytes],
    *,
    encoding: str = "utf-8",
    errors: str = "replace",
) -> str:
    """Decode bytes and normalize Unicode text."""

    if not isinstance(text, (str, bytes)):
        raise TypeError("text must be a string or bytes.")

    if errors not in {"strict", "replace", "ignore"}:
        raise ValueError(
            "errors must be 'strict', 'replace', or 'ignore'."
        )

    if isinstance(text, bytes):
        try:
            text = text.decode(encoding, errors=errors)
        except LookupError as exc:
            raise ValueError(
                f"Unknown text encoding: {encoding}"
            ) from exc

    return unicodedata.normalize("NFC", text)


# ============================================================
# 2. UNICODE NORMALIZATION
# ============================================================

def normalize_unicode(
    text: str,
    form: str = "NFC",
) -> str:
    """Normalize Unicode using NFC, NFD, NFKC, or NFKD."""

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    form = form.upper()

    if form not in {"NFC", "NFD", "NFKC", "NFKD"}:
        raise ValueError(
            "Unicode form must be NFC, NFD, NFKC, or NFKD."
        )

    return unicodedata.normalize(form, text)


# ============================================================
# 3. WHITESPACE NORMALIZATION
# ============================================================

def normalize_whitespace(
    text: str,
    *,
    preserve_line_breaks: bool = False,
) -> str:
    """
    Normalize repeated whitespace.

    Preserves meaningful paragraphs when requested.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if preserve_line_breaks:

        # Normalize spaces and tabs without removing newlines.
        text = re.sub(r"[^\S\n]+", " ", text)

        # Remove spaces around line breaks.
        text = re.sub(r" *\n *", "\n", text)

        # Limit consecutive blank lines.
        text = re.sub(r"\n{3,}", "\n\n", text)

    else:
        text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# 4. HTML CLEANING
# ============================================================

class _HTMLTextExtractor(HTMLParser):
    """Extract readable text while ignoring scripts and styles."""

    BLOCK_TAGS = {
        "address", "article", "aside", "blockquote",
        "br", "div", "footer", "h1", "h2", "h3",
        "h4", "h5", "h6", "header", "hr", "li",
        "main", "ol", "p", "section", "table",
        "td", "th", "tr", "ul",
    }

    HIDDEN_TAGS = {"script", "style"}

    def __init__(self):
        super().__init__(convert_charrefs=True)

        self.parts = []
        self.hidden_tags = []

    def handle_starttag(self, tag, attrs):

        tag = tag.lower()

        if tag in self.HIDDEN_TAGS:
            self.hidden_tags.append(tag)
            return

        if not self.hidden_tags and tag in self.BLOCK_TAGS:
            self.parts.append(" ")

    def handle_endtag(self, tag):

        tag = tag.lower()

        if tag in self.HIDDEN_TAGS:

            for index in range(
                len(self.hidden_tags) - 1,
                -1,
                -1,
            ):
                if self.hidden_tags[index] == tag:
                    del self.hidden_tags[index]
                    break

            return

        if not self.hidden_tags and tag in self.BLOCK_TAGS:
            self.parts.append(" ")

    def handle_data(self, data):

        if not self.hidden_tags:
            self.parts.append(data)

    def get_text(self):
        return "".join(self.parts)


def clean_html(text: str) -> str:
    """Remove HTML tags, comments, scripts, and styles."""

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    parser = _HTMLTextExtractor()

    try:
        parser.feed(text)
        parser.close()

    except Exception as exc:
        raise ValueError(
            "Unable to parse HTML input."
        ) from exc

    return normalize_whitespace(parser.get_text())


# ============================================================
# 5. HTML ENTITY DECODING
# ============================================================

def decode_html_entities(text: str) -> str:
    """
    Decode HTML entities.

    Examples:
        &amp;  -> &
        &lt;   -> <
        &gt;   -> >
        &nbsp; -> space
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    return html.unescape(text)


# ============================================================
# 6. SPECIAL CHARACTER HANDLING
# ============================================================

def handle_special_characters(
    text: str,
    *,
    mode: str = "replace",
    replacement: str = " ",
    keep_unicode: bool = True,
    keep_punctuation: bool = True,
) -> str:
    """
    Handle special characters safely.

    Modes:
        preserve: Keep original text unchanged.
        remove: Remove disallowed characters.
        replace: Replace disallowed characters.

    Important:
        Preserves all letters, including y and Y.
        Preserves digits and Unicode letters by default.
        Preserves punctuation when enabled.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    if mode not in {"preserve", "remove", "replace"}:
        raise ValueError(
            "mode must be 'preserve', 'remove', or 'replace'."
        )

    if not isinstance(replacement, str):
        raise TypeError("replacement must be a string.")

    if not isinstance(keep_unicode, bool):
        raise TypeError("keep_unicode must be a boolean.")

    if not isinstance(keep_punctuation, bool):
        raise TypeError("keep_punctuation must be a boolean.")

    if mode == "preserve":
        return text

    result = []

    for char in text:

        category = unicodedata.category(char)

        # Preserve whitespace.
        if char.isspace():
            result.append(char)
            continue

        # Preserve letters and numbers.
        if category[0] in {"L", "N"}:
            result.append(char)
            continue

        # Preserve combining marks for Unicode text.
        if keep_unicode and category[0] == "M":
            result.append(char)
            continue

        # Preserve punctuation except the @ symbol.
        if keep_punctuation and category[0] == "P" and char != "@":
            result.append(char)
            continue

        # Preserve Unicode symbols when requested.
        if keep_unicode and ord(char) > 127 and category[0] == "S":
            result.append(char)
            continue

        # Remove or replace disallowed characters.
        if mode == "replace":
            result.append(replacement)

    return "".join(result)


# ============================================================
# 7. PUNCTUATION CLEANUP
# ============================================================

def remove_extra_punctuation(text: str) -> str:
    """
    Reduce excessive repeated punctuation.

    Preserves a maximum of three periods for ellipses.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r",{2,}", ",", text)
    text = re.sub(r";{2,}", ";", text)

    return text


# ============================================================
# 8. FILLER WORD REMOVAL
# ============================================================

DEFAULT_FILLER_PHRASES = (
    "you know",
    "i mean",
    "kind of",
    "sort of",
)

DEFAULT_FILLER_WORDS = (
    "um",
    "uh",
    "erm",
    "ah",
    "basically",
    "literally",
    "actually",
)


def remove_filler_words(
    text: str,
    *,
    filler_words: Optional[tuple[str, ...]] = None,
    filler_phrases: Optional[tuple[str, ...]] = None,
    case_insensitive: bool = True,
) -> str:
    """
    Remove optional spoken-language filler words.

    Uses word boundaries to avoid changing parts of other words.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    if not isinstance(case_insensitive, bool):
        raise TypeError(
            "case_insensitive must be a boolean."
        )

    words = (
        DEFAULT_FILLER_WORDS
        if filler_words is None
        else filler_words
    )

    phrases = (
        DEFAULT_FILLER_PHRASES
        if filler_phrases is None
        else filler_phrases
    )

    if isinstance(words, str) or isinstance(phrases, str):
        raise TypeError(
            "filler_words and filler_phrases must be iterables "
            "of strings, not individual strings."
        )

    try:
        terms = list(words) + list(phrases)

    except TypeError as exc:
        raise TypeError(
            "Filler words and phrases must be iterable."
        ) from exc

    if not all(isinstance(term, str) for term in terms):
        raise TypeError(
            "Every filler word and phrase must be a string."
        )

    terms = sorted(
        {
            term.strip()
            for term in terms
            if term.strip()
        },
        key=len,
        reverse=True,
    )

    if not terms:
        return normalize_whitespace(text)

    flags = re.IGNORECASE if case_insensitive else 0

    alternatives = "|".join(
        re.escape(term)
        for term in terms
    )

    pattern = re.compile(
        r"(?<!\w)(?:" + alternatives + r")(?!\w)",
        flags,
    )

    text = pattern.sub(" ", text)

    # Remove spaces before punctuation.
    text = re.sub(r"\s+([,!?;:.])", r"\1", text)

    return normalize_whitespace(text)


# ============================================================
# 9. URL REMOVAL
# ============================================================

def remove_urls(text: str) -> str:
    """
    Remove HTTP, HTTPS, and www URLs.

    Examples:
        https://example.com
        http://example.com/page
        www.example.com
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    pattern = re.compile(
        r"(?:https?://|www\.)"
        r"[^\s<>\"']*"
        r"[^\s<>\"'.,!?;:)]",
        re.IGNORECASE,
    )

    text = pattern.sub(" ", text)

    return normalize_whitespace(text)


# ============================================================
# 10. CONTROL CHARACTER REMOVAL
# ============================================================

def remove_control_characters(
    text: str,
    *,
    preserve_line_breaks: bool = True,
) -> str:
    """Remove unwanted Unicode control characters."""

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    allowed_controls = (
        "\n\r\t"
        if preserve_line_breaks
        else ""
    )

    return "".join(
        char
        for char in text
        if (
            unicodedata.category(char) != "Cc"
            or char in allowed_controls
        )
    )


# ============================================================
# 11. GENERAL TEXT CLEANING
# ============================================================

def clean_text(
    text: str,
    *,
    remove_html: bool = True,
    remove_urls: bool = False,
    normalize_whitespace_enabled: bool = True,
    remove_control_characters: bool = True,
    decode_entities: bool = False,
    unicode_form: str = "NFC",
    preserve_line_breaks: bool = False,
) -> str:
    """Apply configurable general text-cleaning operations."""

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    text = normalize_unicode(text, unicode_form)

    if decode_entities:
        text = decode_html_entities(text)

    if remove_html:
        text = clean_html(text)

    if remove_urls:
        text = globals()["remove_urls"](text)

    if remove_control_characters:
        text = globals()["remove_control_characters"](
            text,
            preserve_line_breaks=preserve_line_breaks,
        )

    if normalize_whitespace_enabled:
        text = normalize_whitespace(
            text,
            preserve_line_breaks=preserve_line_breaks,
        )

    return text.strip()


# ============================================================
# 12. QUERY CLEANING
# ============================================================

def clean_query(
    text: str,
    *,
    remove_fillers: bool = False,
    filler_words: Optional[tuple[str, ...]] = None,
    filler_phrases: Optional[tuple[str, ...]] = None,
    remove_urls: bool = False,
) -> str:
    """Clean a user query before sending it to an LLM."""

    text = clean_text(
        text,
        remove_html=True,
        remove_urls=remove_urls,
        normalize_whitespace_enabled=True,
        remove_control_characters=True,
        decode_entities=True,
    )

    if remove_fillers:
        text = remove_filler_words(
            text,
            filler_words=filler_words,
            filler_phrases=filler_phrases,
        )

    return remove_extra_punctuation(text)


# ============================================================
# 13. RESPONSE CLEANING
# ============================================================

def clean_response(
    response: str,
    *,
    preserve_line_breaks: bool = False,
) -> str:
    """Clean an LLM response while optionally preserving paragraphs."""

    if not isinstance(response, str):
        raise TypeError("response must be a string.")

    response = normalize_unicode(response, "NFC")

    response = remove_control_characters(
        response,
        preserve_line_breaks=True,
    )

    return normalize_whitespace(
        response,
        preserve_line_breaks=preserve_line_breaks,
    )


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "normalize_encoding",
    "normalize_unicode",
    "normalize_whitespace",
    "clean_html",
    "decode_html_entities",
    "handle_special_characters",
    "remove_extra_punctuation",
    "remove_filler_words",
    "remove_urls",
    "remove_control_characters",
    "clean_text",
    "clean_query",
    "clean_response",
    "DEFAULT_FILLER_PHRASES",
    "DEFAULT_FILLER_WORDS",
]
