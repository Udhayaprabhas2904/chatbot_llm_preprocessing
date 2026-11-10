
"""Service layer for report generation; includes timing metrics from SQLite."""

from config.settings import settings
from reporting.report_generator import build_pdf_report
from storage.repository import list_results
from storage.database import database_connection


def _attach_metrics(rows: list[dict]) -> list[dict]:
    """Attach saved evaluation_metrics to each result row."""
    if not rows:
        return rows

    ids = [
        row.get("id")
        for row in rows
        if row.get("id") is not None
    ]

    if not ids:
        return rows

    placeholders = ",".join("?" for _ in ids)

    with database_connection() as connection:
        metric_rows = connection.execute(
            f"""
            SELECT result_id, metric_name, metric_value
            FROM evaluation_metrics
            WHERE result_id IN ({placeholders})
            """,
            ids,
        ).fetchall()

    metrics_by_result = {}

    for metric in metric_rows:
        result_id = int(metric["result_id"])

        metrics_by_result.setdefault(result_id, {})[
            metric["metric_name"]
        ] = metric["metric_value"]

    for row in rows:
        row_id = row.get("id")

        if row_id is not None:
            row.update(
                metrics_by_result.get(int(row_id), {})
            )

    return rows


def generate_report(
    experiment_id: str | None = None,
) -> str:
    """Generate PDF report with result rows and timing metrics."""

    rows = list_results(
        experiment_id=experiment_id,
        limit=5000,
    )

    if not rows:
        raise ValueError(
            "No results are available for the requested experiment."
        )

    rows = _attach_metrics(rows)

    return str(
        build_pdf_report(
            rows,
            experiment_id,
            settings.REPORTS_DIR,
        )
    )
