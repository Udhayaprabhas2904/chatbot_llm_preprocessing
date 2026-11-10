
"""REST endpoint for generating downloadable PDF reports."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.services.report_service import generate_report

router = APIRouter()


@router.post("/generate", summary="Generate a PDF report")
def create_report(experiment_id: str | None = None):
    try:
        report_path = Path(generate_report(experiment_id))

        if not report_path.is_file():
            raise HTTPException(
                status_code=500,
                detail="The report file was not created.",
            )

        return FileResponse(
            path=report_path,
            media_type="application/pdf",
            filename=report_path.name,
        )

    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
