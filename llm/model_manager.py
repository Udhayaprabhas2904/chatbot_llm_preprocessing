
import logging
import time
from typing import Any, Optional

import httpx

from config.settings import settings
from llm.model_registry import model_registry
from llm.ollama_client import ollama_client
from storage.models import ModelResponse


logger = logging.getLogger(__name__)


class ModelManager:
    """Manage local and cloud LLM providers."""

    def __init__(self) -> None:
        self.timeout = settings.OLLAMA_TIMEOUT
        self.gemini_timeout = 180.0

    def _provider_configured(self, provider: str) -> bool:
        """Check whether the required provider credentials exist."""

        credentials = {
            "ollama": True,
            "gemini": bool(settings.GEMINI_API_KEY),
            "groq": bool(settings.GROQ_API_KEY),
            "openrouter": bool(settings.OPENROUTER_API_KEY),
            "cloudflare": bool(
                settings.CLOUDFLARE_API_TOKEN
                and settings.CLOUDFLARE_ACCOUNT_ID
            ),
        }

        return credentials.get(provider, False)

    def list_models(self) -> list[dict[str, Any]]:
        """List registered models and their availability."""

        registered_models = model_registry.list_models()

        try:
            installed = ollama_client.list_installed_models()

            installed_tags = {
                str(item.get("name", "")).lower()
                for item in installed
            }

        except Exception as exc:
            logger.warning(
                "Ollama model listing failed: %s",
                exc,
            )
            installed_tags = set()

        for model in registered_models:
            provider = model["provider"]

            if provider == "ollama":
                model["installed"] = (
                    model["model_tag"].lower() in installed_tags
                )
            else:
                model["installed"] = self._provider_configured(
                    provider
                )

        return registered_models

    def get_model_tag(self, family: str) -> str:
        return model_registry.get_model_tag(family)

    def is_available(self) -> bool:
        """Check local Ollama availability."""

        return ollama_client.is_available()

    def _failure(
        self,
        model_name: str,
        prompt: str,
        started: float,
        error: str,
    ) -> ModelResponse:

        return ModelResponse(
            model_name=model_name,
            prompt=prompt,
            response_text="",
            latency_seconds=round(
                time.perf_counter() - started,
                4,
            ),
            success=False,
            error_message=error,
        )

    def _post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        """Send an HTTP POST request and return its JSON response."""

        response = httpx.post(
            url,
            headers=headers,
            json=payload,
            timeout=(
                timeout
                if timeout is not None
                else self.timeout
            ),
        )

        response.raise_for_status()

        return response.json()

    def _generate_openai_compatible(
        self,
        provider: str,
        model_name: str,
        prompt: str,
        api_key: str,
        base_url: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        system_prompt: Optional[str],
    ) -> dict[str, Any]:
        """Generate responses from OpenAI-compatible providers."""

        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        messages.append({
            "role": "user",
            "content": prompt,
        })

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        if provider == "openrouter":
            headers["X-Title"] = settings.APP_NAME
            headers["HTTP-Referer"] = "http://localhost"

        return self._post_json(
            url=base_url,
            headers=headers,
            payload=payload,
        )

    def _generate_gemini(
        self,
        model_name: str,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        system_prompt: Optional[str],
    ) -> dict[str, Any]:

        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/{model_name}:generateContent"
        )

        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": temperature,
                "topP": top_p,
                "maxOutputTokens": max_tokens,
            },
        }

        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}]
            }

        return self._post_json(
            url=url,
            headers={
                "x-goog-api-key": settings.GEMINI_API_KEY,
                "Content-Type": "application/json",
            },
            payload=payload,
            timeout=self.gemini_timeout,
        )

    def _generate_cloudflare(
        self,
        model_name: str,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        system_prompt: Optional[str],
    ) -> dict[str, Any]:
        """Generate responses using Cloudflare Workers AI."""

        account_id = settings.CLOUDFLARE_ACCOUNT_ID

        url = (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{account_id}/ai/run/{model_name}"
        )

        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        messages.append({
            "role": "user",
            "content": prompt,
        })

        payload = {
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        return self._post_json(
            url=url,
            headers={
                "Authorization": (
                    f"Bearer {settings.CLOUDFLARE_API_TOKEN}"
                ),
                "Content-Type": "application/json",
            },
            payload=payload,
        )

    def _extract_response(
        self,
        provider: str,
        data: dict[str, Any],
    ) -> tuple[str, Optional[int], Optional[int]]:

        # OpenAI-compatible providers
        if provider in {"groq", "openrouter"}:

            if data.get("error"):
                raise ValueError(
                    f"{provider} API error: {data['error']}"
                )

            choices = data.get("choices", [])

            if not choices:
                raise ValueError(
                    f"{provider} returned no response choices. "
                    f"Response fields: {list(data.keys())}"
                )

            choice = choices[0]
            message = choice.get("message") or {}

            content = message.get("content")

            if isinstance(content, str):
                response_text = content

            elif isinstance(content, list):
                response_text = "\n".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict)
                    and item.get("text")
                )

            else:
                response_text = ""

            usage = data.get("usage", {})

            if not response_text.strip():
                raise ValueError(
                    f"{provider} returned empty content. "
                    f"Model: {data.get('model')}; "
                    f"Finish reason: {choice.get('finish_reason')}; "
                    f"Message fields: {list(message.keys())}; "
                    f"Choice fields: {list(choice.keys())}; "
                    f"Usage: {usage}"
                )

            return (
                response_text,
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
            )

        # Google Gemini
        if provider == "gemini":

            candidates = data.get("candidates", [])

            if not candidates:
                raise ValueError(
                    "Gemini returned no candidates. "
                    "Check safety filters or API response."
                )

            parts = (
                candidates[0]
                .get("content", {})
                .get("parts", [])
            )

            response_text = "\n".join(
                part.get("text", "")
                for part in parts
                if part.get("text")
            )

            usage = data.get("usageMetadata", {})

            return (
                response_text,
                usage.get("promptTokenCount"),
                usage.get("candidatesTokenCount"),
            )

        # Cloudflare Workers AI
        if provider == "cloudflare":

            if data.get("success") is False:
                raise ValueError(
                    str(data.get("errors", data))
                )

            result = data.get("result", {})

            response_text = (
                result.get("response", "")
                or result.get("output_text", "")
                if isinstance(result, dict)
                else str(result)
            )

            return response_text, None, None

        raise ValueError(
            f"Unsupported provider: {provider}"
        )

    def generate(
        self,
        family: str,
        prompt: str,
        *,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 512,
        system_prompt: Optional[str] = None,
    ) -> ModelResponse:
        """Generate a response through the selected provider."""

        model = model_registry.get_model(family)

        if not model.enabled:
            raise ValueError(
                f"The model family '{family}' is disabled."
            )

        provider = model.provider
        model_name = model.model_tag
        started = time.perf_counter()

        try:

            # Provider 1: Ollama
            if provider == "ollama":

                return ollama_client.generate(
                    model_name=model_name,
                    prompt=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                )

            if not self._provider_configured(provider):
                raise ValueError(
                    f"API credentials for provider '{provider}' "
                    "are missing. Check your .env file."
                )

            # Provider 2: Gemini
            if provider == "gemini":

                data = self._generate_gemini(
                    model_name,
                    prompt,
                    temperature,
                    top_p,
                    max_tokens,
                    system_prompt,
                )

            # Provider 3: Groq
            elif provider == "groq":

                data = self._generate_openai_compatible(
                    provider,
                    model_name,
                    prompt,
                    settings.GROQ_API_KEY,
                    "https://api.groq.com/openai/v1/chat/completions",
                    temperature,
                    top_p,
                    max_tokens,
                    system_prompt,
                )

            # Provider 4: OpenRouter
            elif provider == "openrouter":

                data = self._generate_openai_compatible(
                    provider,
                    model_name,
                    prompt,
                    settings.OPENROUTER_API_KEY,
                    "https://openrouter.ai/api/v1/chat/completions",
                    temperature,
                    top_p,
                    max_tokens,
                    system_prompt,
                )

            # Provider 5: Cloudflare
            elif provider == "cloudflare":

                data = self._generate_cloudflare(
                    model_name,
                    prompt,
                    temperature,
                    top_p,
                    max_tokens,
                    system_prompt,
                )

            else:
                raise ValueError(
                    f"Unsupported provider: {provider}"
                )

            response_text, input_tokens, output_tokens = (
                self._extract_response(provider, data)
            )

            if not response_text.strip():
                raise ValueError(
                    "The provider returned an empty response."
                )

            return ModelResponse(
                model_name=model_name,
                prompt=prompt,
                response_text=response_text,
                latency_seconds=round(
                    time.perf_counter() - started,
                    4,
                ),
                success=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except httpx.HTTPStatusError as exc:

            error = (
                f"HTTP {exc.response.status_code}: "
                f"{exc.response.text[:1000]}"
            )

            logger.error(
                "%s API error: %s",
                provider,
                error,
            )

            return self._failure(
                model_name,
                prompt,
                started,
                error,
            )

        except (
            httpx.RequestError,
            ValueError,
            KeyError,
        ) as exc:

            logger.error(
                "%s generation failed: %s",
                provider,
                exc,
            )

            return self._failure(
                model_name,
                prompt,
                started,
                str(exc),
            )

        except Exception as exc:

            logger.exception(
                "Unexpected %s error",
                provider,
            )

            return self._failure(
                model_name,
                prompt,
                started,
                str(exc),
            )

    def generate_by_tag(
        self,
        model_tag: str,
        prompt: str,
        *,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 512,
        system_prompt: Optional[str] = None,
    ) -> ModelResponse:
        """Generate using an explicit Ollama model tag."""

        if not isinstance(model_tag, str) or not model_tag.strip():
            raise ValueError(
                "Model tag cannot be empty."
            )

        return ollama_client.generate(
            model_name=model_tag.strip(),
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )

    def enable_model(self, family: str) -> None:
        model_registry.enable_model(family)

    def disable_model(self, family: str) -> None:
        model_registry.disable_model(family)


model_manager = ModelManager()
