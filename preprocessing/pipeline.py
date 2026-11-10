
"""
Preprocessing Pipeline
----------------------

Coordinates all 31 offline preprocessing techniques before
sending queries to a Large Language Model.

Features:
- 31 technique registration and execution
- Automatic implementation tracking
- Technique validation
- Fixed execution order
- Transformation metadata
- Configurable processing options
- Context and instruction support
- Offline processing
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import inspect


# ============================================================
# IMPORT EXISTING IMPLEMENTATIONS
# ============================================================

from preprocessing.cleaning import (
    clean_query,
    normalize_encoding,
    normalize_unicode,
    normalize_whitespace,
    clean_html,
    decode_html_entities,
    handle_special_characters,
)

from preprocessing.normalization import normalize_query
from preprocessing.noise_filtering import filter_noise
from preprocessing.deduplication import deduplicate_query
from preprocessing.prompt_formatter import format_prompt

from preprocessing.advanced_techniques import (
    ADVANCED_TECHNIQUES,
    apply_advanced_technique,
)


# ============================================================
# TECHNIQUE DEFINITIONS
# ============================================================

AVAILABLE_TECHNIQUES = (
    # Existing techniques
    "cleaning",
    "normalization",
    "noise_filtering",
    "deduplication",
    "prompt_formatting",

    # Text cleaning
    "whitespace_normalization",
    "unicode_normalization",
    "html_cleaning",
    "html_entity_decoding",
    "special_character_handling",
    "encoding_normalization",

    # Text normalization
    "case_normalization",
    "spelling_correction",
    "grammar_correction",
    "entity_normalization",
    "stemming",
    "lemmatization",
    "stopword_removal",

    # Noise and privacy
    "pii_removal",
    "sensitive_data_masking",

    # Text processing
    "tokenization",
    "sentence_segmentation",
    "language_detection",
    "text_chunking",
    "text_truncation",

    # Prompt preparation
    "query_rewriting",
    "instruction_formatting",
    "context_preparation",
    "metadata_enrichment",
    "language_translation",
    "structured_data_formatting",
)


# Techniques implemented in existing modules.
EXISTING_TECHNIQUES = (
    "cleaning",
    "normalization",
    "noise_filtering",
    "deduplication",
    "prompt_formatting",
    "whitespace_normalization",
    "unicode_normalization",
    "html_cleaning",
    "html_entity_decoding",
    "special_character_handling",
    "encoding_normalization",
)


# ============================================================
# IMPLEMENTATION TRACKING
# ============================================================

def _get_implemented_techniques() -> tuple[str, ...]:
    """Return techniques that have registered implementations."""

    implemented = set(EXISTING_TECHNIQUES)
    implemented.update(ADVANCED_TECHNIQUES.keys())

    return tuple(
        name
        for name in AVAILABLE_TECHNIQUES
        if name in implemented
    )


def _get_pending_techniques() -> tuple[str, ...]:
    """Return registered techniques without implementations."""

    implemented = set(_get_implemented_techniques())

    return tuple(
        name
        for name in AVAILABLE_TECHNIQUES
        if name not in implemented
    )


# ============================================================
# RESULT MODEL
# ============================================================

@dataclass
class PreprocessingResult:
    """Store preprocessing output and experiment metadata."""

    original_query: str
    processed_query: str

    applied_techniques: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        """Check whether preprocessing changed the input."""

        return self.original_query != self.processed_query

    def to_dict(self) -> dict[str, Any]:
        """Convert the result into a dictionary."""

        return {
            "original_query": self.original_query,
            "processed_query": self.processed_query,
            "applied_techniques": list(self.applied_techniques),
            "changed": self.changed,
            "metadata": dict(self.metadata),
        }


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _validate_options(
    name: str,
    options: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Validate an optional dictionary."""

    if options is None:
        return {}

    if not isinstance(options, dict):
        raise TypeError(f"{name} must be a dictionary.")

    return dict(options)


def _call_compatible(
    function: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Call a function using only keyword arguments supported
    by its signature.

    This helps the pipeline work with preprocessing modules
    that expose different optional parameters.
    """

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(*args, **kwargs)

    parameters = signature.parameters

    accepts_extra_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    if accepts_extra_kwargs:
        supported_kwargs = kwargs
    else:
        supported_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in parameters
        }

    return function(*args, **supported_kwargs)


def _validate_string_result(
    technique: str,
    result: Any,
) -> str:
    """Ensure every preprocessing function returns text."""

    if not isinstance(result, str):
        raise TypeError(
            f"Technique '{technique}' returned "
            f"{type(result).__name__}, expected str."
        )

    return result


# ============================================================
# PREPROCESSING PIPELINE
# ============================================================

class PreprocessingPipeline:
    """Execute selected preprocessing techniques."""

    AVAILABLE_TECHNIQUES = AVAILABLE_TECHNIQUES

    IMPLEMENTED_TECHNIQUES = _get_implemented_techniques()

    PENDING_TECHNIQUES = _get_pending_techniques()

    # Default techniques preserve the original pipeline behavior.
    DEFAULT_TECHNIQUES = (
        "encoding_normalization",
        "html_cleaning",
        "html_entity_decoding",
        "cleaning",
        "whitespace_normalization",
        "unicode_normalization",
        "special_character_handling",
        "normalization",
        "noise_filtering",
        "deduplication",
        "prompt_formatting",
    )

    # Fixed execution order for consistent experiments.
    EXECUTION_ORDER = (
        # Stage 1: Text cleaning
        "encoding_normalization",
        "html_cleaning",
        "html_entity_decoding",
        "cleaning",
        "whitespace_normalization",
        "unicode_normalization",
        "special_character_handling",

        # Stage 2: Text normalization
        "normalization",
        "case_normalization",
        "spelling_correction",
        "grammar_correction",
        "entity_normalization",
        "stemming",
        "lemmatization",
        "stopword_removal",

        # Stage 3: Noise and privacy
        "noise_filtering",
        "pii_removal",
        "sensitive_data_masking",
        "deduplication",

        # Stage 4: Text processing
        "query_rewriting",
        "language_detection",
        "tokenization",
        "sentence_segmentation",
        "text_chunking",
        "text_truncation",

        # Stage 5: Structured data and translation
        "structured_data_formatting",
        "language_translation",

        # Stage 6: Prompt preparation
        "metadata_enrichment",
        "context_preparation",
        "instruction_formatting",
        "prompt_formatting",
    )

    # --------------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------------

    def __init__(
        self,
        techniques: Optional[list[str]] = None,
    ) -> None:

        selected = (
            list(self.DEFAULT_TECHNIQUES)
            if techniques is None
            else techniques
        )

        if not isinstance(selected, list):
            raise TypeError(
                "Techniques must be provided as a list."
            )

        normalized = []

        for item in selected:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    "Technique names must be non-empty strings."
                )

            normalized.append(item.strip().lower())

        invalid = (
            set(normalized)
            - set(self.AVAILABLE_TECHNIQUES)
        )

        if invalid:
            raise ValueError(
                f"Unknown techniques: {sorted(invalid)}. "
                f"Available techniques: "
                f"{list(self.AVAILABLE_TECHNIQUES)}"
            )

        pending = (
            set(normalized)
            - set(self.IMPLEMENTED_TECHNIQUES)
        )

        if pending:
            raise NotImplementedError(
                "The following techniques are not implemented: "
                f"{sorted(pending)}"
            )

        # Remove duplicates while preserving selection order.
        self.techniques = list(dict.fromkeys(normalized))

        selected_set = set(self.techniques)

        self.execution_plan = [
            name
            for name in self.EXECUTION_ORDER
            if name in selected_set
        ]

    # --------------------------------------------------------
    # MAIN PROCESSING METHOD
    # --------------------------------------------------------

    def process(
        self,
        query: str,
        *,
        context: Optional[str] = None,
        system_instruction: Optional[str] = None,
        response_format: str = "standard",
        noise_options: Optional[dict[str, Any]] = None,
        normalization_options: Optional[dict[str, Any]] = None,
        special_character_options: Optional[dict[str, Any]] = None,
        advanced_options: Optional[dict[str, dict[str, Any]]] = None,
    ) -> PreprocessingResult:
        """Execute all selected preprocessing techniques."""

        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not isinstance(response_format, str):
            raise TypeError(
                "response_format must be a string."
            )

        for name, value in (
            ("context", context),
            ("system_instruction", system_instruction),
        ):
            if value is not None and not isinstance(value, str):
                raise TypeError(
                    f"{name} must be a string or None."
                )

        noise_options = _validate_options(
            "noise_options",
            noise_options,
        )

        normalization_options = _validate_options(
            "normalization_options",
            normalization_options,
        )

        special_character_options = _validate_options(
            "special_character_options",
            special_character_options,
        )

        advanced_options = _validate_options(
            "advanced_options",
            advanced_options,
        )

        # Validate every advanced technique's options.
        for technique, options in advanced_options.items():

            if technique not in self.AVAILABLE_TECHNIQUES:
                raise ValueError(
                    f"Unknown technique in advanced_options: "
                    f"{technique}"
                )

            if not isinstance(options, dict):
                raise TypeError(
                    f"Options for '{technique}' must be a dictionary."
                )

        original_query = query
        processed_query = query

        applied = []
        transformation_log = []

        # ----------------------------------------------------
        # TRANSFORMATION LOGGER
        # ----------------------------------------------------

        def record_step(
            name: str,
            before: str,
            after: Any,
        ) -> str:

            after = _validate_string_result(name, after)

            transformation_log.append({
                "technique": name,
                "changed": before != after,
                "before_length": len(before),
                "after_length": len(after),
                "length_difference": len(after) - len(before),
            })

            applied.append(name)

            return after

        # ----------------------------------------------------
        # EXECUTE SELECTED TECHNIQUES
        # ----------------------------------------------------

        for technique in self.execution_plan:

            before = processed_query

            # Start with technique-specific options.
            options = dict(
                advanced_options.get(technique, {})
            )

            try:

                # ============================================
                # EXISTING TECHNIQUES
                # ============================================

                if technique == "encoding_normalization":

                    processed_query = normalize_encoding(
                        processed_query
                    )

                elif technique == "html_cleaning":

                    processed_query = clean_html(
                        processed_query
                    )

                elif technique == "html_entity_decoding":

                    processed_query = decode_html_entities(
                        processed_query
                    )

                elif technique == "cleaning":

                    processed_query = clean_query(
                        processed_query
                    )

                elif technique == "whitespace_normalization":

                    processed_query = normalize_whitespace(
                        processed_query
                    )

                elif technique == "unicode_normalization":

                    options.setdefault("form", "NFC")

                    processed_query = _call_compatible(
                        normalize_unicode,
                        processed_query,
                        **options,
                    )

                elif technique == "special_character_handling":

                    options = {
                        **special_character_options,
                        **options,
                    }

                    processed_query = _call_compatible(
                        handle_special_characters,
                        processed_query,
                        **options,
                    )

                elif technique == "normalization":

                    options = {
                        **normalization_options,
                        **options,
                    }

                    processed_query = _call_compatible(
                        normalize_query,
                        processed_query,
                        **options,
                    )

                elif technique == "noise_filtering":

                    options = {
                        **noise_options,
                        **options,
                    }

                    processed_query = _call_compatible(
                        filter_noise,
                        processed_query,
                        **options,
                    )

                elif technique == "deduplication":

                    processed_query = deduplicate_query(
                        processed_query
                    )

                # ============================================
                # ADVANCED TECHNIQUES
                # ============================================

                elif technique in ADVANCED_TECHNIQUES:

                    if technique == "context_preparation":
                        if context is not None:
                            options.setdefault("context", context)

                    elif technique == "instruction_formatting":

                        instruction = (
                            system_instruction
                            or (
                                "Answer the following user request "
                                "clearly and accurately."
                            )
                        )

                        options.setdefault(
                            "instruction",
                            instruction,
                        )

                    processed_query = apply_advanced_technique(
                        technique,
                        processed_query,
                        **options,
                    )

                # ============================================
                # PROMPT FORMATTING
                # ============================================

                elif technique == "prompt_formatting":

                    prompt_context = (
                        None
                        if "context_preparation" in self.techniques
                        else context
                    )

                    prompt_instruction = (
                        None
                        if "instruction_formatting" in self.techniques
                        else system_instruction
                    )

                    prompt_options = {
                        "system_instruction": prompt_instruction,
                        "context": prompt_context,
                        "response_format": response_format,
                    }

                    prompt_options.update(options)

                    processed_query = _call_compatible(
                        format_prompt,
                        processed_query,
                        **prompt_options,
                    )

                else:
                    raise RuntimeError(
                        f"No executor found for technique: "
                        f"{technique}"
                    )

                processed_query = record_step(
                    technique,
                    before,
                    processed_query,
                )

            except Exception as exc:

                raise RuntimeError(
                    f"Preprocessing failed at technique "
                    f"'{technique}': {exc}"
                ) from exc

        # ----------------------------------------------------
        # CALCULATE METADATA
        # ----------------------------------------------------

        original_length = len(original_query)
        processed_length = len(processed_query)

        length_difference = (
            processed_length - original_length
        )

        reduction_percentage = (
            (original_length - processed_length)
            / original_length * 100
            if original_length > 0
            else 0.0
        )

        metadata = {
            "original_length": original_length,
            "processed_length": processed_length,
            "length_difference": length_difference,

            "length_reduction_percentage": round(
                reduction_percentage,
                2,
            ),

            "query_changed": (
                original_query != processed_query
            ),

            "total_available_techniques": len(
                self.AVAILABLE_TECHNIQUES
            ),

            "total_implemented_techniques": len(
                self.IMPLEMENTED_TECHNIQUES
            ),

            "techniques_requested": list(
                self.techniques
            ),

            "techniques_execution_order": list(
                self.execution_plan
            ),

            "techniques_applied": list(applied),

            "techniques_not_selected": [
                name
                for name in self.AVAILABLE_TECHNIQUES
                if name not in self.techniques
            ],

            "techniques_pending": list(
                self.PENDING_TECHNIQUES
            ),

            "transformation_count": len(applied),

            "changed_transformation_count": sum(
                1
                for step in transformation_log
                if step["changed"]
            ),

            "transformation_log": transformation_log,
        }

        return PreprocessingResult(
            original_query=original_query,
            processed_query=processed_query,
            applied_techniques=applied,
            metadata=metadata,
        )

    # --------------------------------------------------------
    # SIMPLE QUERY PROCESSING
    # --------------------------------------------------------

    def process_query(
        self,
        query: str,
        **options: Any,
    ) -> str:
        """Return only the processed query."""

        return self.process(
            query,
            **options,
        ).processed_query


# ============================================================
# DEFAULT PIPELINE
# ============================================================

preprocessing_pipeline = PreprocessingPipeline()


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def preprocess_text(text: str) -> str:
    """Process text using the default preprocessing techniques."""

    return preprocessing_pipeline.process_query(text)


# ============================================================
# PUBLIC EXPORTS
# ============================================================

__all__ = [
    "AVAILABLE_TECHNIQUES",
    "EXISTING_TECHNIQUES",
    "PreprocessingResult",
    "PreprocessingPipeline",
    "preprocessing_pipeline",
    "preprocess_text",
]
