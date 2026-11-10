
from pydantic import BaseModel, ConfigDict, Field


class ResultRecord(BaseModel):
    """One model response recorded during an experiment."""

    model_config = ConfigDict(from_attributes=True)

    experiment_id: str
    query: str
    model: str
    condition: str
    response: str
    processing_time_seconds: float = Field(ge=0)
    response_length: int = Field(ge=0)
    quality_score: float | None = Field(default=None, ge=0, le=1)


class ComparisonSummary(BaseModel):
    """Aggregated performance statistics for a model and condition."""

    model: str
    condition: str
    total_responses: int
    average_latency_seconds: float
    average_response_length: float
    average_quality_score: float | None = None
