
"""REST endpoints for retrieving and comparing experiment results."""

from fastapi import APIRouter, Query

from storage.repository import list_results, compare_results

router = APIRouter()


@router.get("/", summary="List saved response records")
def get_results(
    experiment_id: str | None = None,
    limit: int = Query(default=500, ge=1, le=5000),
):
    records = list_results(
        experiment_id=experiment_id,
        limit=limit,
    )

    return {
        "count": len(records),
        "results": records,
    }


@router.get("/compare", summary="Compare experimental conditions")
def compare_experiment_results(experiment_id: str | None = None):
    comparisons = compare_results(experiment_id=experiment_id)

    return {
        "experiment_id": experiment_id,
        "comparisons": comparisons,
    }
