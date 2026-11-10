
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


def utc_now() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Experiment:
    """Represents one experiment comparing LLM responses."""

    name: str
    description: str = ""
    status: str = "pending"
    id: Optional[int] = None
    created_at: str = field(default_factory=utc_now)
    completed_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the experiment to a dictionary."""
        return asdict(self)


@dataclass
class ChatResult:
    """Represents one chatbot response within an experiment."""

    experiment_id: int
    query_text: str
    model_name: str
    condition_name: str
    preprocessing_technique: str = "none"
    response_text: str = ""
    latency_seconds: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    quality_score: Optional[float] = None
    error_message: Optional[str] = None
    id: Optional[int] = None
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        """Convert the chatbot result to a dictionary."""
        return asdict(self)


@dataclass
class EvaluationMetric:
    """Represents an evaluation metric for a chatbot result."""

    result_id: int
    metric_name: str
    metric_value: Optional[float] = None
    id: Optional[int] = None
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        """Convert the evaluation metric to a dictionary."""
        return asdict(self)


@dataclass
class ModelResponse:
    """Represents a response returned by a local language model."""

    model_name: str
    prompt: str
    response_text: str
    latency_seconds: float
    success: bool = True
    error_message: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the model response to a dictionary."""
        return asdict(self)


@dataclass
class ComparisonResult:
    """Groups baseline and preprocessed results for comparison."""

    query_text: str
    model_name: str
    baseline: Optional[ChatResult] = None
    preprocessed: Optional[ChatResult] = None
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the comparison into a serializable dictionary."""
        return {
            "query_text": self.query_text,
            "model_name": self.model_name,
            "baseline": (
                self.baseline.to_dict() if self.baseline else None
            ),
            "preprocessed": (
                self.preprocessed.to_dict()
                if self.preprocessed
                else None
            ),
            "metrics": self.metrics.copy(),
        }
