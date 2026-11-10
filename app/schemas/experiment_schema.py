"""Pydantic schemas for multi-provider comparison experiments."""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from llm.model_registry import model_registry
from preprocessing.pipeline import PreprocessingPipeline


class ExperimentRequest(BaseModel):
    """Request model for baseline/preprocessed LLM experiments.

    ``models`` contains model-family names from the registry (for example,
    ``llama`` or ``gemini``), not provider model tags.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    queries: list[str] = Field(
        min_length=1,
        max_length=100,
        description="Queries to evaluate (1–100)",
    )
    models: list[str] | None = Field(
        default=None,
        description=(
            "Configured model-family names. Defaults to all enabled models."
        ),
    )
    conditions: list[str] = Field(
        default_factory=lambda: ["baseline", "preprocessed"],
        description="Experiment conditions to execute",
    )
    preprocessing_techniques: list[str] | None = Field(
        default=None,
        description=(
            "Techniques to apply to preprocessed queries. Defaults to the "
            "pipeline's existing default techniques."
        ),
    )
    temperature: float = Field(default=0.2, ge=0, le=2)
    top_p: float = Field(default=0.9, gt=0, le=1)
    max_tokens: int = Field(default=512, ge=1, le=8192)

    @field_validator("queries")
    @classmethod
    def validate_queries(cls, queries: list[str]) -> list[str]:
        if any(not query.strip() for query in queries):
            raise ValueError("Queries must not be empty.")
        return queries

    @field_validator("models")
    @classmethod
    def normalize_models(
        cls,
        models: list[str] | None,
    ) -> list[str] | None:
        if models is None:
            return None
        if not models:
            raise ValueError("Models must not be an empty list.")

        normalized = [model.strip().lower() for model in models]
        if any(not model for model in normalized):
            raise ValueError("Model-family names must not be empty.")
        if len(normalized) != len(set(normalized)):
            raise ValueError("Duplicate model families are not allowed.")
        return normalized

    @field_validator("conditions")
    @classmethod
    def validate_conditions(cls, conditions: list[str]) -> list[str]:
        normalized = [condition.strip().lower() for condition in conditions]
        if not normalized:
            raise ValueError("At least one condition is required.")

        allowed = {"baseline", "preprocessed"}
        invalid = set(normalized) - allowed
        if invalid:
            raise ValueError(
                "Conditions must be baseline or preprocessed. "
                f"Invalid values: {sorted(invalid)}"
            )
        if len(normalized) != len(set(normalized)):
            raise ValueError("Duplicate conditions are not allowed.")
        return normalized

    @field_validator("preprocessing_techniques")
    @classmethod
    def validate_techniques(
        cls,
        techniques: list[str] | None,
    ) -> list[str] | None:
        if techniques is None:
            return None
        if not techniques:
            raise ValueError(
                "preprocessing_techniques must not be empty when supplied."
            )

        normalized = [technique.strip().lower() for technique in techniques]
        if any(not technique for technique in normalized):
            raise ValueError("Technique names must not be empty.")
        if len(normalized) != len(set(normalized)):
            raise ValueError("Duplicate preprocessing techniques are not allowed.")

        available = set(PreprocessingPipeline.AVAILABLE_TECHNIQUES)
        invalid = set(normalized) - available
        if invalid:
            raise ValueError(
                f"Unknown preprocessing techniques: {sorted(invalid)}"
            )

        implemented = set(PreprocessingPipeline.IMPLEMENTED_TECHNIQUES)
        pending = set(normalized) - implemented
        if pending:
            raise ValueError(
                f"Techniques are registered but not implemented: {sorted(pending)}"
            )
        return normalized

    @model_validator(mode="after")
    def validate_model_families(self):
        if self.models is not None:
            known = set(model_registry.list_families())
            unknown = set(self.models) - known
            if unknown:
                raise ValueError(
                    f"Unconfigured model families: {sorted(unknown)}. "
                    f"Available: {sorted(known)}"
                )

            disabled = {
                family
                for family in self.models
                if not model_registry.get_model(family).enabled
            }
            if disabled:
                raise ValueError(
                    f"Disabled model families cannot be selected: {sorted(disabled)}"
                )
        return self


class ExperimentSummary(BaseModel):
    """Summary returned when an experiment is created or queried."""

    experiment_id: str
    status: str
    total_cases: int
    completed_cases: int
    message: str
