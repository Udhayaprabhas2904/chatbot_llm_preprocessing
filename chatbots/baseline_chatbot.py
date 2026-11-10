
"""Baseline chatbot implementation."""

from llm.model_manager import model_manager
from storage.models import ChatResult
from config.settings import settings


class BaselineChatbot:
    """Send the original query directly to the language model."""

    def __init__(self, client=None):
        self.client = client
        self.temperature = settings.DEFAULT_TEMPERATURE
        self.top_p = settings.DEFAULT_TOP_P
        self.max_tokens = settings.DEFAULT_MAX_TOKENS

    def _generate(self, prompt: str, model_family: str):
        """Generate a response using an injected client or model manager."""

        if self.client is not None:
            return self.client.generate(
                model=model_family,
                prompt=prompt,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )

        return model_manager.generate(
            family=model_family,
            prompt=prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

    @staticmethod
    def _extract_response(response):
        """Extract generated text and metadata from the model response."""

        if isinstance(response, dict):
            response_text = (
                response.get("response_text")
                or response.get("response")
                or response.get("text")
                or ""
            )

            latency = (
                response.get("latency_seconds")
                or response.get("processing_time_seconds")
                or 0.0
            )

            return {
                "response": response_text,
                "processing_time_seconds": latency,
                "model_name": response.get("model_name"),
                "input_tokens": response.get("input_tokens"),
                "output_tokens": response.get("output_tokens"),
                "error_message": response.get("error_message"),
            }

        return {
            "response": getattr(response, "response_text", ""),
            "processing_time_seconds": getattr(
                response, "latency_seconds", 0.0
            ),
            "model_name": getattr(response, "model_name", None),
            "input_tokens": getattr(response, "input_tokens", None),
            "output_tokens": getattr(response, "output_tokens", None),
            "error_message": getattr(response, "error_message", None),
        }

    def respond(self, query: str, model: str) -> dict:
        """Generate a response for a single query."""

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        result = self._generate(query, model)
        extracted = self._extract_response(result)

        return {
            **extracted,
            "submitted_prompt": query,
        }

    def chat(
        self,
        query: str,
        model_family: str,
        experiment_id: int,
    ) -> ChatResult:
        """Generate a baseline response and return experiment metadata."""

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if experiment_id < 1:
            raise ValueError("A valid experiment_id is required.")

        result = self._generate(query, model_family)
        extracted = self._extract_response(result)

        return ChatResult(
            experiment_id=experiment_id,
            query_text=query,
            model_name=extracted["model_name"] or model_family,
            condition_name="baseline",
            preprocessing_technique="none",
            response_text=extracted["response"],
            latency_seconds=extracted["processing_time_seconds"],
            input_tokens=extracted["input_tokens"] or 0,
            output_tokens=extracted["output_tokens"] or 0,
            error_message=extracted["error_message"],
        )


baseline_chatbot = BaselineChatbot()
