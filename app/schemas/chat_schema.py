from typing import Literal

from pydantic import BaseModel, Field, ConfigDict, field_validator


# All supported preprocessing techniques
Technique = Literal[
    # Existing category aliases
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

    # Noise filtering
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
]


# Five configured Ollama model families
ModelFamily = Literal[
    "smollm2",
    "qwen",
    "tinyllama",
    "llama",
    "gemma",
]


class ChatRequest(BaseModel):
    """Request schema for baseline LLM chat."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    query: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The user's original query.",
    )

    model_family: ModelFamily = Field(
        default="llama",
        description="The Ollama model family to use.",
    )


class PreprocessedChatRequest(ChatRequest):
    """Request schema for preprocessing LLM chat."""

    techniques: list[Technique] = Field(
        default_factory=lambda: ["cleaning"],
        description="Preprocessing techniques to apply.",
    )

    @field_validator("techniques")
    @classmethod
    def remove_duplicate_techniques(cls, value):
        """Remove duplicate techniques while preserving order."""
        return list(dict.fromkeys(value))


class ChatResponse(BaseModel):
    """Response schema for chatbot results."""

    success: bool
    query: str
    model_name: str

    condition: Literal["baseline", "preprocessed"]

    preprocessing_techniques: list[str] = Field(
        default_factory=list
    )

    response: str

    latency_seconds: float | None = Field(
        default=None,
        ge=0,
    )

    input_tokens: int | None = Field(
        default=None,
        ge=0,
    )

    output_tokens: int | None = Field(
        default=None,
        ge=0,
    )

    error_message: str | None = None


class ModelInfoResponse(BaseModel):
    """Information about an available LLM."""

    family: str
    model_name: str
    provider: str = "ollama"
    description: str = ""
    enabled: bool = True
    installed: bool = False