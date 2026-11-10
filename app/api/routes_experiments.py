"""REST endpoints for creating and tracking comparison experiments."""

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.schemas.experiment_schema import ExperimentRequest
from app.services.experiment_service import (
    create_experiment,
    execute_experiment,
    get_experiment,
)
from llm.model_registry import model_registry

router = APIRouter()


@router.post(
    "/run",
    status_code=202,
    summary="Start a multi-LLM comparison experiment",
    description=(
        "Runs selected model families under baseline and/or preprocessed "
        "conditions. If models are omitted, all enabled configured models "
        "are selected."
    ),
)
def run_experiment(
    payload: ExperimentRequest,
    background_tasks: BackgroundTasks,
):
    """Validate and queue an experiment as a background task."""

    configured_families = set(model_registry.list_families())
    selected_models = (
        payload.models
        if payload.models is not None
        else model_registry.list_families(enabled_only=True)
    )

    invalid_models = set(selected_models) - configured_families
    if invalid_models:
        raise HTTPException(
            status_code=422,
            detail=f"Unconfigured model families: {sorted(invalid_models)}",
        )

    disabled_models = {
        family
        for family in selected_models
        if not model_registry.get_model(family).enabled
    }
    if disabled_models:
        raise HTTPException(
            status_code=422,
            detail=f"Disabled model families: {sorted(disabled_models)}",
        )

    # Persist the resolved model list so the background task executes
    # exactly the same set of models that was validated here.
    experiment_data = payload.model_dump()
    experiment_data["models"] = selected_models

    record = create_experiment(payload, models=selected_models)

    background_tasks.add_task(
        execute_experiment,
        record["experiment_id"],
        experiment_data,
    )

    return record


@router.get("/{experiment_id}", summary="Get experiment status and progress")
def experiment_status(experiment_id: str):
    """Return the saved status and progress of an experiment."""

    record = get_experiment(experiment_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Experiment not found.",
        )
    return record
