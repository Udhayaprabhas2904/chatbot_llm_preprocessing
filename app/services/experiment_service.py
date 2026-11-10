
"""Create and execute multi-provider LLM comparison experiments."""

import logging
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from config.settings import settings
from evaluation.metrics import response_metrics
from llm.model_manager import model_manager
from preprocessing.pipeline import PreprocessingPipeline
from storage.database import database_connection
from storage.models import EvaluationMetric
from storage.repository import (
    create_experiment_record,
    get_experiment_record,
    repository,
    save_result,
    update_experiment_record,
)

logger = logging.getLogger(__name__)


def _default_models() -> list[str]:
    """Return enabled model-family identifiers."""
    from llm.model_registry import model_registry

    return model_registry.list_families(enabled_only=True)


def _default_techniques() -> list[str]:
    """Return default preprocessing techniques."""
    return list(PreprocessingPipeline.DEFAULT_TECHNIQUES)


def create_experiment(payload, models: list[str] | None = None) -> dict:
    """Create and persist an experiment configuration."""

    experiment_id = uuid.uuid4().hex[:12]

    selected_models = list(
        models if models is not None
        else (payload.models or _default_models())
    )

    selected_techniques = (
        list(payload.preprocessing_techniques)
        if payload.preprocessing_techniques is not None
        else _default_techniques()
    )

    # Validate preprocessing configuration before saving.
    PreprocessingPipeline(techniques=selected_techniques)

    total_cases = (
        len(payload.queries)
        * len(selected_models)
        * len(payload.conditions)
    )

    created_at = datetime.now(timezone.utc).isoformat()

    record = {
        "experiment_id": experiment_id,
        "status": "queued",
        "total_cases": total_cases,
        "completed_cases": 0,
        "created_at": created_at,
        "models": selected_models,
        "conditions": list(payload.conditions),
        "preprocessing_techniques": selected_techniques,
        "query_count": len(payload.queries),
        "message": "Experiment queued.",
    }

    create_experiment_record(record)

    return record


def _set_terminal_timestamp(experiment_id: str) -> None:
    """Persist an experiment completion timestamp."""

    completed_at = datetime.now(timezone.utc).isoformat()

    with database_connection() as connection:
        connection.execute(
            """
            UPDATE experiments
            SET completed_at = ?
            WHERE name = ?
            """,
            (completed_at, experiment_id),
        )


def _save_experiment_result(
    *,
    experiment_id: str,
    query: str,
    model_family: str,
    condition: str,
    response: str,
    generation_seconds: float,
    preprocessing_seconds: float,
    end_to_end_seconds: float,
    prompt_length: int,
    techniques: list[str],
    error_message: str | None,
) -> None:
    """Save the response and associated evaluation metrics."""

    metrics = response_metrics(response)

    result_id = save_result({
        "experiment_id": experiment_id,
        "query": query,
        "model": model_family,
        "condition": condition,
        "response": response,
        "processing_time_seconds": generation_seconds,
        "response_length": len(response),
        "prompt_length": prompt_length,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **metrics,
    })

    technique_label = (
        ",".join(techniques)
        if condition == "preprocessed"
        else "none"
    )

    with database_connection() as connection:
        connection.execute(
            """
            UPDATE results
            SET preprocessing_technique = ?,
                error_message = ?
            WHERE id = ?
            """,
            (technique_label, error_message, result_id),
        )

    numeric_metrics = {
        "preprocessing_seconds": preprocessing_seconds,
        "generation_seconds": generation_seconds,
        "end_to_end_seconds": end_to_end_seconds,
        "prompt_characters": float(prompt_length),
        "response_characters": float(len(response)),
        "word_count": float(metrics.get("word_count", 0)),
        "sentence_count": float(metrics.get("sentence_count", 0)),
        "lexical_diversity": float(
            metrics.get("lexical_diversity", 0.0)
        ),
    }

    for metric_name, metric_value in numeric_metrics.items():
        repository.save_metric(
            EvaluationMetric(
                result_id=result_id,
                metric_name=metric_name,
                metric_value=round(metric_value, 6),
            )
        )


def execute_experiment(
    experiment_id: str,
    data: dict[str, Any],
) -> None:
    """Execute and persist all query/model/condition combinations."""

    queries = data["queries"]
    models = data.get("models") or _default_models()
    conditions = data.get(
        "conditions",
        ["baseline", "preprocessed"],
    )

    techniques = (
        list(data["preprocessing_techniques"])
        if data.get("preprocessing_techniques") is not None
        else _default_techniques()
    )

    total_cases = len(queries) * len(models) * len(conditions)

    completed = 0
    failed = 0

    # Optional random seed allows reproducible condition ordering.
    randomizer = random.Random(data.get("random_seed"))

    pipeline = PreprocessingPipeline(
        techniques=techniques,
    )

    update_experiment_record(
        experiment_id,
        status="running",
        completed_cases=0,
        message=f"Experiment running: 0 of {total_cases} cases.",
    )

    try:
        for query_index, query in enumerate(queries):

            for model_family in models:

                # Randomize baseline/preprocessed execution order
                # to reduce systematic order bias.
                case_conditions = list(conditions)
                randomizer.shuffle(case_conditions)

                for condition in case_conditions:

                    case_started = time.perf_counter()

                    preprocessing_seconds = 0.0
                    generation_seconds = 0.0

                    prompt = query
                    response_text = ""
                    error_message = None
                    applied_techniques = []

                    try:
                        # Preprocessing timing is measured separately.
                        if condition == "preprocessed":

                            preprocessing_started = (
                                time.perf_counter()
                            )

                            processed = pipeline.process(query)

                            preprocessing_seconds = (
                                time.perf_counter()
                                - preprocessing_started
                            )

                            prompt = processed.processed_query

                            applied_techniques = list(
                                processed.applied_techniques
                            )

                        # Measure generation duration.
                        generation_started = time.perf_counter()

                        result = model_manager.generate(
                            model_family,
                            prompt,
                            temperature=data.get(
                                "temperature",
                                settings.DEFAULT_TEMPERATURE,
                            ),
                            top_p=data.get(
                                "top_p",
                                settings.DEFAULT_TOP_P,
                            ),
                            max_tokens=data.get(
                                "max_tokens",
                                settings.DEFAULT_MAX_TOKENS,
                            ),
                        )

                        measured_generation = (
                            time.perf_counter()
                            - generation_started
                        )

                        provider_latency = getattr(
                            result,
                            "latency_seconds",
                            None,
                        )

                        generation_seconds = (
                            float(provider_latency)
                            if provider_latency is not None
                            else measured_generation
                        )

                        if not getattr(result, "success", False):

                            error_message = (
                                getattr(
                                    result,
                                    "error_message",
                                    None,
                                )
                                or "Model generation failed."
                            )

                        else:

                            response_text = str(
                                getattr(
                                    result,
                                    "response_text",
                                    "",
                                )
                                or ""
                            )

                            if not response_text.strip():
                                error_message = (
                                    "The model returned an empty response."
                                )

                    except Exception as exc:

                        error_message = (
                            f"{type(exc).__name__}: {str(exc)[:800]}"
                        )

                        logger.exception(
                            "Experiment %s failed: query=%s model=%s condition=%s",
                            experiment_id,
                            query_index,
                            model_family,
                            condition,
                        )

                    # Total elapsed time includes preprocessing,
                    # generation and local processing overhead.
                    end_to_end_seconds = (
                        time.perf_counter()
                        - case_started
                    )

                    if error_message:
                        failed += 1

                    _save_experiment_result(
                        experiment_id=experiment_id,
                        query=query,
                        model_family=model_family,
                        condition=condition,
                        response=response_text,
                        generation_seconds=generation_seconds,
                        preprocessing_seconds=preprocessing_seconds,
                        end_to_end_seconds=end_to_end_seconds,
                        prompt_length=len(prompt),
                        techniques=applied_techniques,
                        error_message=error_message,
                    )

                    completed += 1

                    update_experiment_record(
                        experiment_id,
                        status="running",
                        completed_cases=completed,
                        message=(
                            f"Completed {completed}/{total_cases} cases. "
                            f"Failures: {failed}."
                        ),
                    )

        if failed == 0:
            final_message = (
                "Experiment completed successfully."
            )
        else:
            final_message = (
                f"Experiment completed with {failed} failed case(s). "
                "Check result error_message fields."
            )

        update_experiment_record(
            experiment_id,
            status="completed",
            completed_cases=completed,
            message=final_message,
        )

        _set_terminal_timestamp(experiment_id)

    except Exception as exc:

        logger.exception(
            "Experiment %s failed unexpectedly.",
            experiment_id,
        )

        update_experiment_record(
            experiment_id,
            status="failed",
            completed_cases=completed,
            message=(
                f"{type(exc).__name__}: {str(exc)[:800]}"
            ),
        )

        _set_terminal_timestamp(experiment_id)


def get_experiment(experiment_id: str):
    """Retrieve experiment details and progress."""
    return get_experiment_record(experiment_id)
