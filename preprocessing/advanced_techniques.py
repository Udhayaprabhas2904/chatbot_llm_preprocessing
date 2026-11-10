
"""
Advanced Preprocessing Techniques
---------------------------------

Offline implementations for 20 advanced LLM preprocessing techniques.

No external API calls are made.

Notes:
- Spelling and grammar correction use conservative rule-based corrections.
- Translation uses a user-provided offline dictionary.
- Language detection is heuristic.
- Stemming and lemmatization are lightweight English implementations.
"""

import json
import re
from typing import Any, Optional


# ============================================================
# COMMON VALIDATION
# ============================================================

def _validate_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    return text


# ============================================================
# 1. CASE NORMALIZATION
# ============================================================

def case_normalization(text: str, mode: str = "lower") -> str:
    _validate_text(text)

    if not isinstance(mode, str):
        raise TypeError("mode must be a string.")

    mode = mode.lower().strip()

    if mode == "lower":
        return text.lower()
    if mode == "upper":
        return text.upper()
    if mode == "title":
        return text.title()
    if mode == "sentence":
        return text[:1].upper() + text[1:].lower()

    raise ValueError("mode must be lower, upper, title, or sentence.")


# ============================================================
# 2. SPELLING CORRECTION
# ============================================================

SPELLING_DICTIONARY = {
    "teh": "the",
    "recieve": "receive",
    "recieved": "received",
    "seperate": "separate",
    "definately": "definitely",
    "occured": "occurred",
    "untill": "until",
    "becuase": "because",
    "wierd": "weird",
    "thier": "their",
    "adress": "address",
    "langauge": "language",
    "programing": "programming",
    "powerfull": "powerful",
    "machne": "machine",
    "computr": "computer",
    "artifical": "artificial",
    "inteligence": "intelligence",
    "explaination": "explanation",
    "enviroment": "environment",
}


def spelling_correction(
    text: str,
    custom_dictionary: Optional[dict[str, str]] = None,
) -> str:
    _validate_text(text)

    dictionary = dict(SPELLING_DICTIONARY)

    if custom_dictionary is not None:
        if not isinstance(custom_dictionary, dict):
            raise TypeError("custom_dictionary must be a dictionary.")

        for wrong, correct in custom_dictionary.items():
            if not isinstance(wrong, str) or not isinstance(correct, str):
                raise TypeError("Dictionary keys and values must be strings.")
            dictionary[wrong.lower()] = correct

    pattern = re.compile(r"\b[\w']+\b", re.UNICODE)

    def replace(match: re.Match) -> str:
        word = match.group(0)
        corrected = dictionary.get(word.lower())

        if corrected is None:
            return word

        if word.isupper():
            return corrected.upper()
        if word.istitle():
            return corrected.capitalize()

        return corrected

    return pattern.sub(replace, text)


# ============================================================
# 3. GRAMMAR CORRECTION
# ============================================================

GRAMMAR_REPLACEMENTS = [
    (r"\bi is\b", "I am"),
    (r"\bhe are\b", "he is"),
    (r"\bshe are\b", "she is"),
    (r"\bthey is\b", "they are"),
    (r"\bwe is\b", "we are"),
    (r"\byou is\b", "you are"),
    (r"\bi am agree\b", "I agree"),
    (r"\bdoes not has\b", "does not have"),
    (r"\bdid not went\b", "did not go"),
    (r"\bmore better\b", "better"),
    (r"\bvery unique\b", "unique"),
]


def grammar_correction(text: str) -> str:
    _validate_text(text)

    result = text

    for pattern, replacement in GRAMMAR_REPLACEMENTS:
        result = re.sub(
            pattern,
            replacement,
            result,
            flags=re.IGNORECASE,
        )

    result = re.sub(r"\s+([,.!?;:])", r"\1", result)
    result = re.sub(r"([,.!?])(?=[A-Za-z])", r"\1 ", result)

    return result


# ============================================================
# 4. ENTITY NORMALIZATION
# ============================================================

DEFAULT_ENTITY_ALIASES = {
    "artificial intelligence": "AI",
    "machine learning": "ML",
    "natural language processing": "NLP",
    "large language model": "LLM",
    "large language models": "LLMs",
    "united states of america": "USA",
    "united states": "USA",
    "application programming interface": "API",
}


def entity_normalization(
    text: str,
    aliases: Optional[dict[str, str]] = None,
) -> str:
    _validate_text(text)

    mapping = dict(DEFAULT_ENTITY_ALIASES)

    if aliases is not None:
        if not isinstance(aliases, dict):
            raise TypeError("aliases must be a dictionary.")

        for original, normalized in aliases.items():
            if not isinstance(original, str) or not isinstance(normalized, str):
                raise TypeError("Alias keys and values must be strings.")
            mapping[original] = normalized

    for original, normalized in sorted(
        mapping.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if not original:
            continue

        pattern = re.compile(
            r"(?<!\w)" + re.escape(original) + r"(?!\w)",
            re.IGNORECASE,
        )

        text = pattern.sub(lambda _: normalized, text)

    return text


# ============================================================
# 5. STEMMING
# ============================================================

def stemming(text: str) -> str:
    """
    Lightweight English suffix-based stemmer.

    This is not a full Porter stemmer.
    """

    _validate_text(text)

    def stem_word(word: str) -> str:
        lower = word.lower()

        if len(lower) <= 3:
            return lower

        # Handle common irregular-looking suffix patterns first.
        if lower.endswith("ies") and len(lower) > 4:
            return lower[:-3] + "y"

        if lower.endswith("ied") and len(lower) > 4:
            return lower[:-3] + "y"

        # Remove -ing.
        if lower.endswith("ing") and len(lower) > 5:
            base = lower[:-3]

            # running -> run, swimming -> swim
            if len(base) >= 2 and base[-1] == base[-2]:
                base = base[:-1]

            # playing -> play
            return base

        # Remove -ed.
        if lower.endswith("ed") and len(lower) > 4:
            base = lower[:-2]

            if len(base) >= 2 and base[-1] == base[-2]:
                base = base[:-1]

            return base

        # Remove selected suffixes conservatively.
        for suffix in ("ments", "ment", "ness", "ful", "less", "ly"):
            if lower.endswith(suffix) and len(lower) - len(suffix) >= 3:
                return lower[:-len(suffix)]

        # Remove plural s only when the word is long enough.
        if lower.endswith("s") and not lower.endswith(("ss", "us", "is")):
            if len(lower) > 4:
                return lower[:-1]

        return lower

    return re.sub(
        r"\b[A-Za-z]+\b",
        lambda match: stem_word(match.group(0)),
        text,
    )


# ============================================================
# 6. LEMMATIZATION
# ============================================================

IRREGULAR_LEMMAS = {
    "am": "be",
    "is": "be",
    "are": "be",
    "was": "be",
    "were": "be",
    "been": "be",
    "being": "be",
    "has": "have",
    "had": "have",
    "does": "do",
    "did": "do",
    "done": "do",
    "went": "go",
    "gone": "go",
    "ran": "run",
    "running": "run",
    "better": "good",
    "best": "good",
    "worse": "bad",
    "worst": "bad",
    "children": "child",
    "men": "man",
    "women": "woman",
    "mice": "mouse",
    "feet": "foot",
    "teeth": "tooth",
}


def lemmatization(text: str) -> str:
    _validate_text(text)

    def lemma(word: str) -> str:
        lower = word.lower()

        if lower in IRREGULAR_LEMMAS:
            return IRREGULAR_LEMMAS[lower]

        if len(lower) > 4 and lower.endswith("ies"):
            return lower[:-3] + "y"

        if len(lower) > 5 and lower.endswith("ing"):
            base = lower[:-3]

            if len(base) > 2 and base[-1] == base[-2]:
                base = base[:-1]

            return base

        if len(lower) > 4 and lower.endswith("ed"):
            base = lower[:-2]

            if len(base) > 2 and base[-1] == base[-2]:
                base = base[:-1]

            return base

        if len(lower) > 4 and lower.endswith("es"):
            return lower[:-2]

        if len(lower) > 3 and lower.endswith("s"):
            return lower[:-1]

        return lower

    return re.sub(
        r"\b[A-Za-z]+\b",
        lambda match: lemma(match.group(0)),
        text,
    )


# ============================================================
# 7. STOPWORD REMOVAL
# ============================================================

ENGLISH_STOPWORDS = {
    "a", "an", "the", "is", "am", "are", "was", "were",
    "be", "been", "being", "to", "of", "in", "on", "at",
    "for", "from", "by", "with", "and", "or", "but",
    "if", "then", "than", "as", "it", "its", "this",
    "that", "these", "those", "i", "you", "he", "she",
    "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "our", "their", "what", "which",
    "who", "whom", "when", "where", "why", "how",
    "do", "does", "did", "have", "has", "had",
    "can", "could", "will", "would", "should",
}


def stopword_removal(
    text: str,
    custom_stopwords: Optional[set[str]] = None,
) -> str:
    _validate_text(text)

    stopwords = set(ENGLISH_STOPWORDS)

    if custom_stopwords is not None:
        stopwords.update(word.lower() for word in custom_stopwords)

    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

    filtered = [
        token for token in tokens
        if not (token.isalpha() and token.lower() in stopwords)
    ]

    result = " ".join(filtered)

    # Restore normal punctuation spacing.
    result = re.sub(r"\s+([,.!?;:%])", r"\1", result)
    result = re.sub(r"([(])\s+", r"\1", result)
    result = re.sub(r"\s+([)])", r"\1", result)

    return result.strip()


# ============================================================
# 8. PII REMOVAL
# ============================================================

PII_PATTERNS = [
    (
        "email",
        re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        ),
    ),
    (
        "phone",
        re.compile(
            r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)"
        ),
    ),
]


def pii_removal(text: str) -> str:
    _validate_text(text)

    for _, pattern in PII_PATTERNS:
        text = pattern.sub("[REDACTED]", text)

    return text


# ============================================================
# 9. SENSITIVE DATA MASKING
# ============================================================

SENSITIVE_PATTERNS = [
    re.compile(r"\b\d{12}\b"),
    re.compile(
        r"\b(?:password|passwd|secret|api[\s_-]?key)"
        r"\s*[:=]\s*[^\s,;]+",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:ssn|social security number)"
        r"\s*[:=]?\s*\d{3}-?\d{2}-?\d{4}\b",
        re.IGNORECASE,
    ),
]


def sensitive_data_masking(text: str) -> str:
    _validate_text(text)

    for pattern in SENSITIVE_PATTERNS:
        text = pattern.sub("[SENSITIVE_DATA]", text)

    return text


# ============================================================
# 10. TOKENIZATION
# ============================================================

def tokenization(text: str) -> str:
    _validate_text(text)

    tokens = re.findall(
        r"\w+|[^\w\s]",
        text,
        flags=re.UNICODE,
    )

    return " ".join(tokens)


# ============================================================
# 11. SENTENCE SEGMENTATION
# ============================================================

def sentence_segmentation(text: str) -> str:
    _validate_text(text)

    if not text.strip():
        return ""

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip(),
    )

    return "\n".join(
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    )


# ============================================================
# 12. LANGUAGE DETECTION
# ============================================================

LANGUAGE_MARKERS = {
    "english": {"the", "is", "are", "and", "what", "how"},
    "spanish": {"el", "la", "los", "las", "que", "como"},
    "french": {"le", "la", "les", "une", "des", "est"},
    "german": {"der", "die", "das", "und", "ist", "ein"},
}


def detect_language(text: str) -> str:
    _validate_text(text)

    if re.search(r"[\u0B80-\u0BFF]", text):
        return "Tamil"

    if re.search(r"[\u0900-\u097F]", text):
        return "Hindi"

    if re.search(r"[\u0600-\u06FF]", text):
        return "Arabic"

    if re.search(r"[\u4E00-\u9FFF]", text):
        return "Chinese"

    if re.search(r"[\u3040-\u30FF]", text):
        return "Japanese"

    words = set(re.findall(r"[a-z]+", text.lower()))

    scores = {
        language: len(words.intersection(markers))
        for language, markers in LANGUAGE_MARKERS.items()
    }

    if not scores:
        return "unknown"

    language, score = max(
        scores.items(),
        key=lambda item: item[1],
    )

    if score == 0:
        return "unknown"

    return language.capitalize()


def language_detection(text: str) -> str:
    _validate_text(text)

    language = detect_language(text)

    return (
        f"[Detected language: {language}]\n"
        f"{text}"
    )


# ============================================================
# 13. TEXT CHUNKING
# ============================================================

def text_chunking(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> str:
    _validate_text(text)

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be >= 0 and smaller than chunk_size."
        )

    words = text.split()

    if not words:
        return ""

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk = words[start:start + chunk_size]

        if chunk:
            chunks.append(" ".join(chunk))

        if start + chunk_size >= len(words):
            break

    return "\n\n[CHUNK SEPARATOR]\n\n".join(chunks)


# ============================================================
# 14. TEXT TRUNCATION
# ============================================================

def text_truncation(
    text: str,
    max_length: int = 4000,
    *,
    mode: str = "characters",
) -> str:
    _validate_text(text)

    if max_length <= 0:
        raise ValueError("max_length must be greater than zero.")

    if mode == "characters":
        return text[:max_length]

    if mode == "words":
        return " ".join(text.split()[:max_length])

    raise ValueError("mode must be 'characters' or 'words'.")


# ============================================================
# 15. QUERY REWRITING
# ============================================================

def query_rewriting(text: str) -> str:
    _validate_text(text)

    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"!{2,}", "!", text)

    return text


# ============================================================
# 16. INSTRUCTION FORMATTING
# ============================================================

def instruction_formatting(
    text: str,
    instruction: str = "Answer the following user request clearly and accurately.",
) -> str:
    _validate_text(text)
    _validate_text(instruction)

    return (
        "INSTRUCTION:\n"
        f"{instruction.strip()}\n\n"
        "USER INPUT:\n"
        f"{text.strip()}\n\n"
        "RESPONSE:\n"
    )


# ============================================================
# 17. CONTEXT PREPARATION
# ============================================================

def context_preparation(
    text: str,
    context: Optional[str] = None,
) -> str:
    _validate_text(text)

    if context is not None:
        _validate_text(context)

    if not context or not context.strip():
        return text

    return (
        "CONTEXT:\n"
        f"{context.strip()}\n\n"
        "USER QUERY:\n"
        f"{text.strip()}"
    )


# ============================================================
# 18. METADATA ENRICHMENT
# ============================================================

def metadata_enrichment(text: str) -> str:
    _validate_text(text)

    words = re.findall(r"\b\w+\b", text, re.UNICODE)

    sentences = [
        part for part in re.split(r"[.!?]+", text)
        if part.strip()
    ]

    metadata = {
        "character_count": len(text),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "detected_language": detect_language(text),
    }

    return (
        "TEXT METADATA:\n"
        f"{json.dumps(metadata, ensure_ascii=False)}\n\n"
        "TEXT:\n"
        f"{text}"
    )


# ============================================================
# 19. LANGUAGE TRANSLATION
# ============================================================

def language_translation(
    text: str,
    translation_dictionary: Optional[dict[str, str]] = None,
) -> str:
    _validate_text(text)

    if translation_dictionary is None:
        return text

    if not isinstance(translation_dictionary, dict):
        raise TypeError("translation_dictionary must be a dictionary.")

    for source, target in translation_dictionary.items():
        if not isinstance(source, str) or not isinstance(target, str):
            raise TypeError(
                "Translation dictionary keys and values must be strings."
            )

    result = text

    for source, target in sorted(
        translation_dictionary.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if source:
            result = result.replace(source, target)

    return result


# ============================================================
# 20. STRUCTURED DATA FORMATTING
# ============================================================

def structured_data_formatting(text: str) -> str:
    _validate_text(text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text

    return json.dumps(
        data,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    )


# ============================================================
# TECHNIQUE REGISTRY
# ============================================================

ADVANCED_TECHNIQUES = {
    "case_normalization": case_normalization,
    "spelling_correction": spelling_correction,
    "grammar_correction": grammar_correction,
    "entity_normalization": entity_normalization,
    "stemming": stemming,
    "lemmatization": lemmatization,
    "stopword_removal": stopword_removal,
    "pii_removal": pii_removal,
    "sensitive_data_masking": sensitive_data_masking,
    "tokenization": tokenization,
    "sentence_segmentation": sentence_segmentation,
    "language_detection": language_detection,
    "text_chunking": text_chunking,
    "text_truncation": text_truncation,
    "query_rewriting": query_rewriting,
    "instruction_formatting": instruction_formatting,
    "context_preparation": context_preparation,
    "metadata_enrichment": metadata_enrichment,
    "language_translation": language_translation,
    "structured_data_formatting": structured_data_formatting,
}


def get_advanced_techniques() -> list[str]:
    return list(ADVANCED_TECHNIQUES.keys())


def apply_advanced_technique(
    name: str,
    text: str,
    **options: Any,
) -> str:
    _validate_text(text)

    technique = ADVANCED_TECHNIQUES.get(name)

    if technique is None:
        raise ValueError(f"Unknown advanced technique: {name}")

    return technique(text, **options)


__all__ = [
    "ADVANCED_TECHNIQUES",
    "get_advanced_techniques",
    "apply_advanced_technique",
    "case_normalization",
    "spelling_correction",
    "grammar_correction",
    "entity_normalization",
    "stemming",
    "lemmatization",
    "stopword_removal",
    "pii_removal",
    "sensitive_data_masking",
    "tokenization",
    "sentence_segmentation",
    "detect_language",
    "language_detection",
    "text_chunking",
    "text_truncation",
    "query_rewriting",
    "instruction_formatting",
    "context_preparation",
    "metadata_enrichment",
    "language_translation",
    "structured_data_formatting",
]
