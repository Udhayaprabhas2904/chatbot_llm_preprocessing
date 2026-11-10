
from contextlib import contextmanager
import sqlite3
from typing import Iterator

from config.settings import settings, create_project_directories


def get_connection() -> sqlite3.Connection:
    """Create and return a SQLite database connection."""
    create_project_directories()

    connection = sqlite3.connect(
        str(settings.DATABASE_PATH),
        timeout=30,
    )

    # Support both index-based and column-name-based row access.
    connection.row_factory = sqlite3.Row

    # Enable foreign-key constraints.
    connection.execute("PRAGMA foreign_keys = ON")

    return connection


@contextmanager
def database_connection() -> Iterator[sqlite3.Connection]:
    """
    Provide a database connection.

    Commits when successful, rolls back on exceptions,
    and always closes the connection.
    """
    connection = get_connection()

    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database() -> None:
    """Create the database and required tables if they do not exist."""
    create_project_directories()

    with database_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                selected_models TEXT,
                selected_techniques TEXT,
                conditions TEXT,
                total_cases INTEGER DEFAULT 0,
                completed_cases INTEGER DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT
                    (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                started_at TEXT,
                completed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                query_text TEXT NOT NULL,
                model_name TEXT NOT NULL,
                condition_name TEXT NOT NULL,
                preprocessing_technique TEXT NOT NULL DEFAULT 'none',
                preprocessed_query TEXT,
                response_text TEXT DEFAULT '',
                success INTEGER DEFAULT 1,
                latency_seconds REAL,
                preprocessing_seconds REAL DEFAULT 0,
                generation_seconds REAL DEFAULT 0,
                end_to_end_seconds REAL DEFAULT 0,
                input_tokens INTEGER,
                output_tokens INTEGER,
                quality_score REAL,
                error_message TEXT,
                created_at TEXT NOT NULL DEFAULT
                    (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),

                FOREIGN KEY (experiment_id)
                    REFERENCES experiments(id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS evaluation_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                result_id INTEGER NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL,
                created_at TEXT NOT NULL DEFAULT
                    (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),

                FOREIGN KEY (result_id)
                    REFERENCES results(id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_results_experiment
                ON results(experiment_id);

            CREATE INDEX IF NOT EXISTS idx_results_model
                ON results(model_name);

            CREATE INDEX IF NOT EXISTS idx_results_condition
                ON results(condition_name);

            CREATE INDEX IF NOT EXISTS idx_metrics_result
                ON evaluation_metrics(result_id);
            """
        )


def check_database_connection() -> bool:
    """Return True if the database can execute a simple query."""
    try:
        with database_connection() as connection:
            connection.execute("SELECT 1")
        return True
    except sqlite3.Error:
        return False
