
import json
import sqlite3
from typing import Any, Optional

from storage.database import (
    database_connection,
    initialize_database,
)
from storage.models import (
    ChatResult,
    EvaluationMetric,
    Experiment,
)


class Repository:
    """Central repository for experiment and evaluation data."""

    def __init__(self) -> None:
        # Ensure tables exist before repository operations.
        initialize_database()

    # ---------------------------------------------------------
    # Experiment operations
    # ---------------------------------------------------------

    def create_experiment(self, experiment: Experiment) -> int:
        """Insert an experiment and return its generated ID."""
        with database_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO experiments (
                    name,
                    description,
                    status,
                    created_at,
                    completed_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    experiment.name,
                    experiment.description,
                    experiment.status,
                    experiment.created_at,
                    experiment.completed_at,
                ),
            )
            return int(cursor.lastrowid)

    def get_experiment(self, experiment_id: int) -> Optional[dict[str, Any]]:
        """Retrieve one experiment by ID."""
        with database_connection() as connection:
            row = connection.execute(
                "SELECT * FROM experiments WHERE id = ?",
                (experiment_id,),
            ).fetchone()

        return dict(row) if row else None

    def list_experiments(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return experiments ordered from newest to oldest."""
        limit = max(1, min(limit, 1000))
        offset = max(0, offset)

        with database_connection() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM experiments
                ORDER BY id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()

        return [dict(row) for row in rows]

    def update_experiment_status(
        self,
        experiment_id: int,
        status: str,
    ) -> bool:
        """Update an experiment status and completion timestamp."""
        allowed_statuses = {
            "pending",
            "running",
            "completed",
            "failed",
            "cancelled",
        }

        if status not in allowed_statuses:
            raise ValueError(
                f"Invalid experiment status: {status}"
            )

        completed_at = (
            "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"
            if status in {"completed", "failed", "cancelled"}
            else None
        )

        with database_connection() as connection:
            if completed_at:
                cursor = connection.execute(
                    f"""
                    UPDATE experiments
                    SET status = ?,
                        completed_at = {completed_at}
                    WHERE id = ?
                    """,
                    (status, experiment_id),
                )
            else:
                cursor = connection.execute(
                    """
                    UPDATE experiments
                    SET status = ?, completed_at = NULL
                    WHERE id = ?
                    """,
                    (status, experiment_id),
                )

            return cursor.rowcount > 0

    # ---------------------------------------------------------
    # Chat result operations
    # ---------------------------------------------------------

    def save_result(self, result: ChatResult) -> int:
        """Save a chatbot result and return its generated ID."""
        with database_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO results (
                    experiment_id,
                    query_text,
                    model_name,
                    condition_name,
                    preprocessing_technique,
                    response_text,
                    latency_seconds,
                    input_tokens,
                    output_tokens,
                    quality_score,
                    error_message,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.experiment_id,
                    result.query_text,
                    result.model_name,
                    result.condition_name,
                    result.preprocessing_technique,
                    result.response_text,
                    result.latency_seconds,
                    result.input_tokens,
                    result.output_tokens,
                    result.quality_score,
                    result.error_message,
                    result.created_at,
                ),
            )
            return int(cursor.lastrowid)

    def get_result(self, result_id: int) -> Optional[dict[str, Any]]:
        """Retrieve one chatbot result by ID."""
        with database_connection() as connection:
            row = connection.execute(
                "SELECT * FROM results WHERE id = ?",
                (result_id,),
            ).fetchone()

        return dict(row) if row else None

    def list_results(
        self,
        experiment_id: Optional[int] = None,
        model_name: Optional[str] = None,
        condition_name: Optional[str] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List results with optional filters and pagination."""
        limit = max(1, min(limit, 5000))
        offset = max(0, offset)

        conditions: list[str] = []
        parameters: list[Any] = []

        if experiment_id is not None:
            conditions.append("experiment_id = ?")
            parameters.append(experiment_id)

        if model_name is not None:
            conditions.append("model_name = ?")
            parameters.append(model_name)

        if condition_name is not None:
            conditions.append("condition_name = ?")
            parameters.append(condition_name)

        where_clause = (
            "WHERE " + " AND ".join(conditions)
            if conditions
            else ""
        )

        parameters.extend([limit, offset])

        query = f"""
            SELECT *
            FROM results
            {where_clause}
            ORDER BY id DESC
            LIMIT ? OFFSET ?
        """

        with database_connection() as connection:
            rows = connection.execute(
                query,
                parameters,
            ).fetchall()

        return [dict(row) for row in rows]

    # ---------------------------------------------------------
    # Evaluation metric operations
    # ---------------------------------------------------------

    def save_metric(self, metric: EvaluationMetric) -> int:
        """Save an evaluation metric and return its ID."""
        with database_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO evaluation_metrics (
                    result_id,
                    metric_name,
                    metric_value,
                    created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    metric.result_id,
                    metric.metric_name,
                    metric.metric_value,
                    metric.created_at,
                ),
            )
            return int(cursor.lastrowid)

    def list_metrics(
        self,
        result_id: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve evaluation metrics, optionally by result ID."""
        with database_connection() as connection:
            if result_id is None:
                rows = connection.execute(
                    """
                    SELECT *
                    FROM evaluation_metrics
                    ORDER BY id DESC
                    """
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT *
                    FROM evaluation_metrics
                    WHERE result_id = ?
                    ORDER BY id
                    """,
                    (result_id,),
                ).fetchall()

        return [dict(row) for row in rows]

    # ---------------------------------------------------------
    # Comparison and summary operations
    # ---------------------------------------------------------

    def get_comparison_results(
        self,
        experiment_id: int,
    ) -> list[dict[str, Any]]:
        """Return baseline and preprocessed records for an experiment."""
        with database_connection() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM results
                WHERE experiment_id = ?
                ORDER BY query_text, model_name, condition_name
                """,
                (experiment_id,),
            ).fetchall()

        return [dict(row) for row in rows]

    def get_experiment_summary(
        self,
        experiment_id: int,
    ) -> dict[str, Any]:
        """Return result counts and average response latency."""
        with database_connection() as connection:
            experiment = connection.execute(
                """
                SELECT id, name, status
                FROM experiments
                WHERE id = ?
                """,
                (experiment_id,),
            ).fetchone()

            if experiment is None:
                raise ValueError(
                    f"Experiment {experiment_id} was not found."
                )

            summary = connection.execute(
                """
                SELECT
                    COUNT(*) AS total_results,
                    COUNT(DISTINCT model_name) AS models_tested,
                    COUNT(DISTINCT query_text) AS queries_tested,
                    AVG(latency_seconds) AS average_latency_seconds,
                    SUM(
                        CASE
                            WHEN error_message IS NOT NULL
                                 AND error_message != ''
                            THEN 1
                            ELSE 0
                        END
                    ) AS failed_results
                FROM results
                WHERE experiment_id = ?
                """,
                (experiment_id,),
            ).fetchone()

        return {
            "experiment": dict(experiment),
            "summary": dict(summary),
        }


# Shared repository instance for application services.
repository = Repository()

def create_experiment_record(record: dict[str, Any]) -> int:
    """Create an experiment while preserving its public string ID."""
    experiment_id = record["experiment_id"]

    experiment = Experiment(
        name=experiment_id,
        description=json.dumps(record),
        status=record.get("status", "queued"),
        created_at=record["created_at"],
    )
    return repository.create_experiment(experiment)


def update_experiment_record(
    experiment_id: str,
    *,
    status: str,
    completed_cases: Optional[int] = None,
    message: Optional[str] = None,
) -> bool:
    """Update experiment status and progress metadata."""
    with database_connection() as connection:
        row = connection.execute(
            "SELECT id, description FROM experiments WHERE name = ?",
            (experiment_id,),
        ).fetchone()

        if row is None:
            return False

        try:
            metadata = json.loads(row["description"] or "{}")
        except (TypeError, json.JSONDecodeError):
            metadata = {}

        metadata["experiment_id"] = experiment_id

        if completed_cases is not None:
            metadata["completed_cases"] = completed_cases

        if message is not None:
            metadata["message"] = message

        terminal = status in {"completed", "failed", "cancelled"}
        completed_at = (
            "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"
            if terminal else None
        )

        if terminal:
            connection.execute(
                """
                UPDATE experiments
                SET status = ?, completed_at = ?, description = ?
                WHERE id = ?
                """,
                (status, completed_at, json.dumps(metadata), row["id"]),
            )
        else:
            connection.execute(
                """
                UPDATE experiments
                SET status = ?, completed_at = NULL, description = ?
                WHERE id = ?
                """,
                (status, json.dumps(metadata), row["id"]),
            )

        return True


def save_result(result: dict[str, Any]) -> int:
    """Save a result using the existing ChatResult database model."""
    with database_connection() as connection:
        row = connection.execute(
            "SELECT id FROM experiments WHERE name = ?",
            (str(result["experiment_id"]),),
        ).fetchone()

    if row is None:
        raise ValueError(
            f"Experiment {result['experiment_id']} was not found."
        )

    metrics = result.get("quality_score")

    chat_result = ChatResult(
        experiment_id=int(row["id"]),
        query_text=result["query"],
        model_name=result["model"],
        condition_name=result["condition"],
        preprocessing_technique=(
            "preprocessed"
            if result["condition"] == "preprocessed"
            else "none"
        ),
        response_text=result.get("response", ""),
        latency_seconds=result.get("processing_time_seconds", 0.0),
        quality_score=metrics,
        created_at=result["created_at"],
    )

    result_id = repository.save_result(chat_result)

    # Store lexical diversity in the existing metrics table.
    lexical_diversity = result.get("lexical_diversity")
    if lexical_diversity is not None:
        repository.save_metric(
            EvaluationMetric(
                result_id=result_id,
                metric_name="lexical_diversity",
                metric_value=float(lexical_diversity),
            )
        )

    return result_id


def get_experiment_record(
    experiment_id: str,
) -> Optional[dict[str, Any]]:
    """Retrieve an experiment by its public string ID."""
    with database_connection() as connection:
        row = connection.execute(
            "SELECT * FROM experiments WHERE name = ?",
            (experiment_id,),
        ).fetchone()

        if row is None:
            return None

        record = dict(row)

        count_row = connection.execute(
            "SELECT COUNT(*) AS total FROM results WHERE experiment_id = ?",
            (row["id"],),
        ).fetchone()

    try:
        metadata = json.loads(record["description"] or "{}")
    except (TypeError, json.JSONDecodeError):
        metadata = {}

    metadata.update(
        {
            "experiment_id": experiment_id,
            "status": record["status"],
            "created_at": record["created_at"],
            "completed_at": record["completed_at"],
            "completed_cases": count_row["total"],
        }
    )

    return metadata

def list_results(
    experiment_id: Optional[str] = None,
    model_name: Optional[str] = None,
    condition_name: Optional[str] = None,
    limit: int = 500,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """List saved results, accepting a public experiment ID."""
    database_experiment_id = None

    if experiment_id is not None:
        with database_connection() as connection:
            row = connection.execute(
                "SELECT id FROM experiments WHERE name = ?",
                (str(experiment_id),),
            ).fetchone()

        if row is None:
            return []

        database_experiment_id = int(row["id"])

    return repository.list_results(
        experiment_id=database_experiment_id,
        model_name=model_name,
        condition_name=condition_name,
        limit=limit,
        offset=offset,
    )


def compare_results(
    experiment_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Group baseline and preprocessed responses for comparison."""
    records = list_results(experiment_id=experiment_id, limit=5000)

    grouped: dict[tuple[str, str], dict[str, Any]] = {}

    for record in records:
        key = (record["query_text"], record["model_name"])

        if key not in grouped:
            grouped[key] = {
                "query": record["query_text"],
                "model": record["model_name"],
                "baseline": None,
                "preprocessed": None,
            }

        condition = record["condition_name"]
        if condition in {"baseline", "preprocessed"}:
            grouped[key][condition] = record

    return list(grouped.values())

