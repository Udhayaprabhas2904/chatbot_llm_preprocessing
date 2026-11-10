
"""
REST endpoints for configured local Ollama models and server health.
Supports smollm2, qwen, tinyllama, llama, and gemma.
"""

import logging
from typing import Any

from fastapi import APIRouter

from config.settings import settings
from llm.model_registry import model_registry
from llm.ollama_client import OllamaClient

router = APIRouter()
logger = logging.getLogger(__name__)


SUPPORTED_MODEL_FAMILIES = {
    "smollm2",
    "qwen",
    "tinyllama",
    "llama",
    "gemma",
}


def _provider_configured(provider: str) -> bool:
    """Check whether the provider is supported and configured."""

    return provider.lower() == "ollama"


def _get_installed_model_names(
    client: OllamaClient,
) -> tuple[list[str], bool]:
    """Retrieve installed Ollama model names safely."""

    try:
        response = client.list_installed_models()

        if isinstance(response, dict):
            response = response.get("models", [])

        if not isinstance(response, list):
            raise ValueError("Unexpected Ollama model-list response.")

        names = []

        for item in response:
            if isinstance(item, str):
                name = item
            elif isinstance(item, dict):
                name = item.get("name", "")
            else:
                continue

            if isinstance(name, str) and name.strip():
                names.append(name.strip())

        return sorted(set(names)), True

    except Exception as exc:
        logger.warning("Unable to retrieve Ollama models: %s", exc)
        return [], False


@router.get("/", summary="List configured and installed Ollama models")
def list_models() -> dict[str, Any]:
    """List the five supported local model families."""

    client = OllamaClient()

    installed_names, ollama_available = (
        _get_installed_model_names(client)
    )

    installed_names_lower = {
        name.lower() for name in installed_names
    }

    installed_families = {
        name.split(":")[0].lower()
        for name in installed_names
    }

    configured_models = []

    for model in model_registry.list_models():
        family = model["family"].lower()

        # Ignore models outside the project's five supported families.
        if family not in SUPPORTED_MODEL_FAMILIES:
            continue

        tag = model["model_tag"]
        provider = model["provider"]

        configured = _provider_configured(provider)

        if ollama_available and configured:
            installed = (
                tag.lower() in installed_names_lower
                or tag.split(":")[0].lower()
                in installed_families
            )
        else:
            installed = False

        configured_models.append({
            "family": family,
            "provider": provider,
            "tag": tag,
            "configured": configured,
            "installed": installed,
            "enabled": model["enabled"],
        })

    return {
        "ollama_available": ollama_available,
        "total_providers": len(configured_models),
        "configured_models": configured_models,
        "installed_models": installed_names,
    }


@router.get("/health", summary="Check Ollama server availability")
def model_server_health() -> dict[str, Any]:
    """Check whether the local Ollama server is available."""

    try:
        available = OllamaClient().is_available()
    except Exception as exc:
        logger.warning("Ollama health check failed: %s", exc)
        available = False

    return {
        "ollama_available": available,
        "base_url": settings.OLLAMA_BASE_URL,
    }
