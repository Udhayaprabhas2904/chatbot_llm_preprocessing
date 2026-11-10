
"""
Preprocessing Chatbot
---------------------
Preprocesses user queries before sending them to an LLM
and records response metadata for experiments.
"""

from typing import Any

from llm.model_manager import model_manager
from preprocessing.pipeline import (
    PreprocessingPipeline,
    preprocessing_pipeline,
)
from storage.models import ChatResult
from config.settings import settings


class PreprocessingChatbot:
    """Preprocess user queries before generating LLM responses."""

    def __init__(self, client: Any = None) -> None:
        self.client = client
        self.temperature = settings.DEFAULT_TEMPERATURE
        self.top_p = settings.DEFAULT_TOP_P
        self.max_tokens = settings.DEFAULT_MAX_TOKENS

    def _generate(self, prompt: str, model_family: str) -> Any:
        """Generate a response using the configured model."""

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
    def _validate_query(query: str) -> None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty.")

    @staticmethod
    def _process_query(
        query: str,
        techniques: list[str] | None = None,
    ):
        """Apply the selected preprocessing techniques."""

        if techniques is None:
            return preprocessing_pipeline.process(query=query)

        pipeline = PreprocessingPipeline(techniques=techniques)
        return pipeline.process(query=query)

    @staticmethod
    def _extract_response(response: Any) -> dict[str, Any]:
        """Normalize model output into a consistent dictionary."""

        if isinstance(response, dict):
            return {
                "response": response.get(
                    "response",
                    response.get("response_text", ""),
                ),
                "processing_time_seconds": response.get(
                    "processing_time_seconds",
                    response.get("latency_seconds"),
                ),
                "input_tokens": response.get("input_tokens"),
                "output_tokens": response.get("output_tokens"),
                "model_name": response.get("model_name"),
                "error_message": response.get("error_message"),
            }

        return {
            "response": getattr(response, "response_text", ""),
            "processing_time_seconds": getattr(
                response, "latency_seconds", None
            ),
            "input_tokens": getattr(response, "input_tokens", None),
            "output_tokens": getattr(response, "output_tokens", None),
            "model_name": getattr(response, "model_name", None),
            "error_message": getattr(response, "error_message", None),
        }

    def respond(
        self,
        query: str,
        model: str,
        techniques: list[str] | None = None,
    ) -> dict[str, Any]:
        """Preprocess a query and return response metadata."""

        self._validate_query(query)

        preprocessing_result = self._process_query(
            query=query,
            techniques=techniques,
        )

        processed_query = preprocessing_result.processed_query

        raw_response = self._generate(
            prompt=processed_query,
            model_family=model,
        )

        result = self._extract_response(raw_response)

        return {
            **result,
            "submitted_prompt": processed_query,
            "original_query": query,
            "preprocessing_techniques": list(
                preprocessing_result.applied_techniques
            ),
        }

    def chat(
        self,
        query: str,
        model_family: str,
        experiment_id: int,
        techniques: list[str] | None = None,
    ) -> ChatResult:
        """Return an experiment-ready ChatResult."""

        self._validate_query(query)

        if experiment_id < 1:
            raise ValueError("A valid experiment_id is required.")

        preprocessing_result = self._process_query(
            query=query,
            techniques=techniques,
        )

        raw_response = self._generate(
            prompt=preprocessing_result.processed_query,
            model_family=model_family,
        )

        result = self._extract_response(raw_response)

        return ChatResult(
            experiment_id=experiment_id,
            query_text=query,
            model_name=result["model_name"] or model_family,
            condition_name="preprocessed",
            preprocessing_technique=(
                ",".join(preprocessing_result.applied_techniques) or "none"
            ),
            response_text=result["response"],
            latency_seconds=result["processing_time_seconds"] or 0.0,
            input_tokens=result["input_tokens"] or 0,
            output_tokens=result["output_tokens"] or 0,
            error_message=result["error_message"],
        )


preprocessing_chatbot = PreprocessingChatbot()
