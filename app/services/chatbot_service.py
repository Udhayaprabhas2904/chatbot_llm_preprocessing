
"""Application service for baseline and preprocessed chat."""

from app.schemas.chat_schema import (
    ChatRequest,
    PreprocessedChatRequest,
)
from chatbots.baseline_chatbot import BaselineChatbot
from chatbots.preprocessing_chatbot import PreprocessingChatbot


def clean_response_text(text: str) -> str:
    """Remove escaped and actual newline characters from model output."""
    if not text:
        return ""

    # Remove literal escaped newline sequences.
    text = text.replace("\\n", " ")

    # Remove actual newline and carriage-return characters.
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")

    # Normalize repeated whitespace.
    return " ".join(text.split())


def run_chat(
    payload: ChatRequest | PreprocessedChatRequest,
    condition: str,
) -> dict:
    """Run a baseline or preprocessed chat request."""

    if condition not in {"baseline", "preprocessed"}:
        raise ValueError("Condition must be baseline or preprocessed.")

    model_family = payload.model_family

    if condition == "baseline":
        chatbot = BaselineChatbot()

        result = chatbot.respond(
            payload.query,
            model_family,
        )

        techniques = []

    else:
        if not isinstance(payload, PreprocessedChatRequest):
            raise ValueError(
                "Preprocessed requests must include the "
                "PreprocessedChatRequest schema."
            )

        chatbot = PreprocessingChatbot()

        result = chatbot.respond(
            payload.query,
            model_family,
            techniques=payload.techniques,
        )

        techniques = result.get(
            "preprocessing_techniques",
            [],
        )

    # Get the model response and remove newline characters.
    response_text = clean_response_text(
        result.get("response", "")
    )

    return {
        "success": bool(response_text),
        "query": payload.query,
        "model_name": model_family,
        "condition": condition,
        "preprocessing_techniques": techniques,
        "response": response_text,
        "latency_seconds": result.get(
            "processing_time_seconds"
        ),
        "input_tokens": result.get("input_tokens"),
        "output_tokens": result.get("output_tokens"),
        "error_message": (
            result.get("error_message")
            if not response_text
            else None
        ),
    }
