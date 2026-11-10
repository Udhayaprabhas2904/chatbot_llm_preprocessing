
"""HTTP client for interacting with a local Ollama server."""

import time
from typing import Any, Optional

import httpx

from config.settings import settings
from storage.models import ModelResponse


class OllamaClient:
    """HTTP client for interacting with a local Ollama server."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.base_url = (
            base_url or settings.OLLAMA_BASE_URL
        ).rstrip("/")

        self.timeout = (
            timeout
            if timeout is not None
            else settings.OLLAMA_TIMEOUT
        )

    def _post(
        self,
        endpoint: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Send a POST request and validate the JSON response."""

        url = f"{self.base_url}{endpoint}"

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()

        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.base_url}. "
                "Check that Ollama is installed and running."
            ) from exc

        except httpx.TimeoutException as exc:
            raise TimeoutError(
                f"Ollama request timed out after {self.timeout} seconds."
            ) from exc

        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500]
            raise RuntimeError(
                f"Ollama returned HTTP {exc.response.status_code}: "
                f"{detail}"
            ) from exc

        except ValueError as exc:
            raise RuntimeError(
                "Ollama returned an invalid JSON response."
            ) from exc

        if not isinstance(data, dict):
            raise RuntimeError(
                "Ollama returned an unexpected response format."
            )

        return data

    def _get(self, endpoint: str) -> dict[str, Any]:
        """Send a GET request and validate the JSON response."""

        url = f"{self.base_url}{endpoint}"

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url)
                response.raise_for_status()
                data = response.json()

        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.base_url}."
            ) from exc

        except httpx.TimeoutException as exc:
            raise TimeoutError(
                f"Ollama request timed out after {self.timeout} seconds."
            ) from exc

        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500]
            raise RuntimeError(
                f"Ollama returned HTTP {exc.response.status_code}: "
                f"{detail}"
            ) from exc

        except ValueError as exc:
            raise RuntimeError(
                "Ollama returned invalid JSON."
            ) from exc

        if not isinstance(data, dict):
            raise RuntimeError(
                "Ollama returned an unexpected response format."
            )

        return data

    def list_installed_models(self) -> list[dict[str, Any]]:
        """Return the models currently installed in Ollama."""

        data = self._get("/api/tags")
        models = data.get("models", [])

        if not isinstance(models, list):
            raise RuntimeError(
                "Unexpected model list returned by Ollama."
            )

        return models

    def is_available(self) -> bool:
        """Check whether the Ollama server is reachable."""

        try:
            self._get("/api/tags")
            return True
        except (ConnectionError, TimeoutError, RuntimeError):
            return False

    def is_model_installed(self, model_name: str) -> bool:
        """Check whether a specific model tag is installed."""

        target = model_name.strip()

        if not target:
            return False

        try:
            models = self.list_installed_models()
        except (ConnectionError, TimeoutError, RuntimeError):
            return False

        installed_names = {
            str(model.get("name", "")).lower()
            for model in models
        }

        return target.lower() in installed_names

    def generate(
        self,
        model_name: str,
        prompt: str,
        *,
        temperature: float = settings.DEFAULT_TEMPERATURE,
        top_p: float = settings.DEFAULT_TOP_P,
        max_tokens: int = settings.DEFAULT_MAX_TOKENS,
        system_prompt: Optional[str] = None,
    ) -> ModelResponse:
        """Generate a response from an Ollama model."""

        model_name = model_name.strip()
        prompt = prompt.strip()

        if not model_name:
            raise ValueError("Model name cannot be empty.")

        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        if not 0.0 <= temperature <= 2.0:
            raise ValueError(
                "temperature must be between 0 and 2."
            )

        if not 0.0 < top_p <= 1.0:
            raise ValueError(
                "top_p must be greater than 0 and at most 1."
            )

        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1.")

        payload: dict[str, Any] = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        start_time = time.perf_counter()

        try:
            data = self._post("/api/generate", payload)

            elapsed = time.perf_counter() - start_time

            # Check whether Ollama returned an explicit error.
            if data.get("error"):
                return ModelResponse(
                    model_name=model_name,
                    prompt=prompt,
                    response_text="",
                    latency_seconds=round(elapsed, 6),
                    success=False,
                    error_message=str(data["error"]),
                )

            # Extract generated text.
            response_text = data.get("response", "")

            if not isinstance(response_text, str):
                response_text = ""

            response_text = response_text.strip()

            # Do not mark an empty response as successful.
            if not response_text:
                return ModelResponse(
                    model_name=model_name,
                    prompt=prompt,
                    response_text="",
                    latency_seconds=round(elapsed, 6),
                    success=False,
                    input_tokens=data.get("prompt_eval_count"),
                    output_tokens=data.get("eval_count"),
                    error_message=(
                        "Ollama returned an empty response. "
                        "Check the model tag and Ollama generation output."
                    ),
                )

            return ModelResponse(
                model_name=model_name,
                prompt=prompt,
                response_text=response_text,
                latency_seconds=round(elapsed, 6),
                success=True,
                input_tokens=data.get("prompt_eval_count"),
                output_tokens=data.get("eval_count"),
                error_message=None,
            )

        except (ConnectionError, TimeoutError, RuntimeError) as exc:
            elapsed = time.perf_counter() - start_time

            return ModelResponse(
                model_name=model_name,
                prompt=prompt,
                response_text="",
                latency_seconds=round(elapsed, 6),
                success=False,
                error_message=str(exc),
            )


# Shared client instance for application services.
ollama_client = OllamaClient()
