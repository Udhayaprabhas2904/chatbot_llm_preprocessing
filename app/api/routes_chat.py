
"""REST endpoints for baseline and preprocessed chatbot requests."""

from fastapi import APIRouter, HTTPException

from app.schemas.chat_schema import (
    ChatRequest,
    PreprocessedChatRequest,
    ChatResponse,
)
from app.services.chatbot_service import run_chat

router = APIRouter()


# ---------------------------------------------------------
# PREPROCESSING TECHNIQUE CATALOGUE
# ---------------------------------------------------------

PREPROCESSING_TECHNIQUES = {
    "text_cleaning": [
        "cleaning",
        "whitespace_normalization",
        "unicode_normalization",
        "html_cleaning",
        "html_entity_decoding",
        "special_character_handling",
        "encoding_normalization",
    ],
    "text_normalization": [
        "case_normalization",
        "spelling_correction",
        "grammar_correction",
        "entity_normalization",
        "stemming",
        "lemmatization",
        "stopword_removal",
    ],
    "noise_and_data_filtering": [
        "noise_filtering",
        "deduplication",
        "pii_removal",
        "sensitive_data_masking",
    ],
    "text_processing": [
        "tokenization",
        "sentence_segmentation",
        "language_detection",
        "text_chunking",
        "text_truncation",
    ],
    "prompt_preparation": [
        "query_rewriting",
        "instruction_formatting",
        "context_preparation",
        "metadata_enrichment",
        "language_translation",
        "prompt_formatting",
        "structured_data_formatting",
    ],
}


# ---------------------------------------------------------
# ALL AVAILABLE TECHNIQUES
# ---------------------------------------------------------

ALL_PREPROCESSING_TECHNIQUES = {
    technique
    for techniques in PREPROCESSING_TECHNIQUES.values()
    for technique in techniques
}

# Preserve existing general normalization.
ALL_PREPROCESSING_TECHNIQUES.add("normalization")


# ---------------------------------------------------------
# IMPLEMENTATION STATUS
# ---------------------------------------------------------

# Mark every available technique as implemented.
IMPLEMENTED_PREPROCESSING_TECHNIQUES = (
    ALL_PREPROCESSING_TECHNIQUES.copy()
)


# ---------------------------------------------------------
# GET PREPROCESSING TECHNIQUES
# ---------------------------------------------------------

@router.get(
    "/preprocessing-techniques",
    summary="List LLM preprocessing techniques",
    description=(
        "Returns preprocessing techniques grouped by category "
        "and their implementation status."
    ),
)
def list_preprocessing_techniques():
    """Return all preprocessing techniques and their status."""

    return {
        "success": True,
        "categories": {
            category: [
                {
                    "name": technique,
                    "implemented": (
                        technique
                        in IMPLEMENTED_PREPROCESSING_TECHNIQUES
                    ),
                }
                for technique in techniques
            ]
            for category, techniques
            in PREPROCESSING_TECHNIQUES.items()
        },
        "currently_implemented": sorted(
            IMPLEMENTED_PREPROCESSING_TECHNIQUES
        ),
        "total_techniques": len(
            ALL_PREPROCESSING_TECHNIQUES
        ),
    }


# ---------------------------------------------------------
# BASELINE CHAT
# ---------------------------------------------------------

@router.post(
    "/baseline",
    response_model=ChatResponse,
    summary="Baseline LLM chat",
    description=(
        "Sends the original user query directly "
        "to the selected LLM."
    ),
)
def baseline_chat(payload: ChatRequest):
    """Generate a response without preprocessing."""

    try:
        return run_chat(payload, "baseline")

    except (
        ConnectionError,
        TimeoutError,
        RuntimeError,
        ValueError,
    ) as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc


# ---------------------------------------------------------
# PREPROCESSED CHAT
# ---------------------------------------------------------

@router.post(
    "/preprocessed",
    response_model=ChatResponse,
    summary="LLM chat with preprocessing",
    description=(
        "Applies selected preprocessing techniques before "
        "sending the query to the selected model."
    ),
)
def preprocessed_chat(
    payload: PreprocessedChatRequest,
):
    """Generate a response using preprocessing."""

    selected_techniques = (
        getattr(payload, "techniques", None) or []
    )

    # Require at least one selected technique.
    if not selected_techniques:
        raise HTTPException(
            status_code=400,
            detail=(
                "Select at least one preprocessing technique."
            ),
        )

    # Remove duplicate selections.
    selected_techniques = list(
        dict.fromkeys(selected_techniques)
    )

    # Validate technique names.
    invalid_techniques = (
        set(selected_techniques)
        - ALL_PREPROCESSING_TECHNIQUES
    )

    if invalid_techniques:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Unknown preprocessing techniques.",
                "invalid_techniques": sorted(
                    invalid_techniques
                ),
                "available_techniques": sorted(
                    ALL_PREPROCESSING_TECHNIQUES
                ),
            },
        )

    # All catalogue techniques are marked implemented.
    # No 501 unavailable-technique check is performed.

    try:
        return run_chat(
            payload,
            "preprocessed",
        )

    except (
        ConnectionError,
        TimeoutError,
        RuntimeError,
        ValueError,
    ) as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

