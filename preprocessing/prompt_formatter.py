
"""
Prompt Formatter
----------------
Creates structured and consistent prompts for Large Language
Models (LLMs).

Features:
1. Standard prompt formatting
2. Concise and detailed response instructions
3. Optional supporting context
4. Controlled comparison prompts
5. JSON response formatting
6. Optional JSON schema guidance
"""

import json
from typing import Any, Optional


DEFAULT_SYSTEM_INSTRUCTION = (
    "You are a helpful, accurate, and clear AI assistant. "
    "Answer the user's question directly. If you are uncertain, "
    "state the uncertainty rather than inventing facts."
)


SUPPORTED_RESPONSE_FORMATS = {
    "standard": "Provide a clear, relevant answer.",
    "concise": (
        "Answer concisely and focus on the key points. "
        "Avoid unnecessary explanations."
    ),
    "detailed": (
        "Provide a detailed, well-organized answer with "
        "explanations and examples when useful."
    ),
    "json": (
        "Return only valid JSON. Do not use Markdown code fences "
        "or include explanatory text outside the JSON."
    ),
}


def _validate_optional_text(
    value: Optional[str],
    field_name: str,
) -> Optional[str]:
    """Validate optional string parameters."""

    if value is not None and not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or None.")

    if value is None:
        return None

    value = value.strip()
    return value if value else None


def _validate_query(query: str) -> str:
    """Validate and normalize the user query."""

    if not isinstance(query, str):
        raise TypeError("query must be a string.")

    query = query.strip()

    if not query:
        raise ValueError("query cannot be empty.")

    return query


def _validate_response_format(response_format: str) -> str:
    """Validate the requested response format."""

    if not isinstance(response_format, str):
        raise TypeError("response_format must be a string.")

    response_format = response_format.strip().lower()

    if response_format not in SUPPORTED_RESPONSE_FORMATS:
        supported = ", ".join(SUPPORTED_RESPONSE_FORMATS)

        raise ValueError(
            f"Unsupported response_format: {response_format}. "
            f"Choose from: {supported}."
        )

    return response_format


def format_prompt(
    query: str,
    *,
    system_instruction: Optional[str] = None,
    context: Optional[str] = None,
    response_format: str = "standard",
) -> str:
    """
    Format a user query into a structured LLM prompt.

    Args:
        query: User question or instruction.
        system_instruction: Optional system-level instruction.
        context: Optional supporting information.
        response_format: standard, concise, detailed, or json.

    Returns:
        A structured prompt string.
    """

    query = _validate_query(query)

    response_format = _validate_response_format(
        response_format
    )

    instruction = _validate_optional_text(
        system_instruction,
        "system_instruction",
    )

    supporting_context = _validate_optional_text(
        context,
        "context",
    )

    if instruction is None:
        instruction = DEFAULT_SYSTEM_INSTRUCTION

    format_instruction = SUPPORTED_RESPONSE_FORMATS[
        response_format
    ]

    sections = [
        "SYSTEM INSTRUCTIONS:",
        instruction,
        "",
        "RESPONSE REQUIREMENTS:",
        format_instruction,
    ]

    if supporting_context:
        sections.extend([
            "",
            "SUPPORTING CONTEXT:",
            supporting_context,
        ])

    sections.extend([
        "",
        "USER QUERY:",
        query,
        "",
        "ANSWER:",
    ])

    return "\n".join(sections)


def format_comparison_prompt(
    query: str,
    *,
    context: Optional[str] = None,
) -> str:
    """
    Create a consistent prompt for baseline and preprocessed
    experiments.

    Both conditions use the same system instruction and response
    format to support fair comparisons.
    """

    return format_prompt(
        query,
        system_instruction=DEFAULT_SYSTEM_INSTRUCTION,
        context=context,
        response_format="standard",
    )


def format_json_prompt(
    query: str,
    *,
    system_instruction: Optional[str] = None,
) -> str:
    """Format a query that requires a valid JSON response."""

    return format_prompt(
        query,
        system_instruction=system_instruction,
        response_format="json",
    )


def format_json_schema_prompt(
    query: str,
    schema: dict[str, Any],
    *,
    system_instruction: Optional[str] = None,
) -> str:
    """
    Format a query with an expected JSON structure.

    The schema is included as guidance. It does not guarantee
    that the model will produce schema-valid JSON.
    """

    query = _validate_query(query)

    if not isinstance(schema, dict):
        raise TypeError("schema must be a dictionary.")

    schema_text = json.dumps(
        schema,
        indent=2,
        ensure_ascii=False,
    )

    instruction = _validate_optional_text(
        system_instruction,
        "system_instruction",
    )

    if instruction is None:
        instruction = DEFAULT_SYSTEM_INSTRUCTION

    schema_context = (
        "Follow this JSON structure when generating your answer:\n"
        f"{schema_text}"
    )

    return format_prompt(
        query,
        system_instruction=instruction,
        context=schema_context,
        response_format="json",
    )


__all__ = [
    "DEFAULT_SYSTEM_INSTRUCTION",
    "SUPPORTED_RESPONSE_FORMATS",
    "format_prompt",
    "format_comparison_prompt",
    "format_json_prompt",
    "format_json_schema_prompt",
]
