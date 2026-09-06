"""
Manifest — idempotency checks and audit-trail writes against
ingestion_manifest (in the airflow database, separate from rag_data).
"""

import json

from airflow.providers.postgres.hooks.postgres import PostgresHook

MANIFEST_DB_CONN_ID = "manifest_db"


def _hook() -> PostgresHook:
    return PostgresHook(postgres_conn_id=MANIFEST_DB_CONN_ID)


def already_processed(object_key: str) -> bool:
    """True if this object_key has a prior successful manifest row.
    Guards against duplicate processing from S3's at-least-once event
    delivery, Lambda retries, or manual re-triggers."""
    existing = _hook().get_first(
        """
        SELECT id FROM ingestion_manifest
        WHERE object_key = %(object_key)s AND status = 'success'
        LIMIT 1
        """,
        parameters={"object_key": object_key},
    )
    return existing is not None


def record(
    object_key: str,
    dag_run_id: str,
    trigger_source: str,
    event_time: str | None,
    status: str,
    raw_conf: dict,
    error_message: str | None = None,
) -> None:
    """Insert one manifest row. Every call inserts a new row (append-only
    audit trail) rather than updating in place — the full history of
    attempts for an object_key stays queryable."""
    _hook().run(
        """
        INSERT INTO ingestion_manifest
            (object_key, dag_run_id, trigger_source, event_time, status, error_message, raw_conf)
        VALUES
            (%(object_key)s, %(dag_run_id)s, %(trigger_source)s, %(event_time)s, %(status)s, %(error_message)s, %(raw_conf)s::jsonb)
        """,
        parameters={
            "object_key": object_key,
            "dag_run_id": dag_run_id,
            "trigger_source": trigger_source,
            "event_time": event_time,
            "status": status,
            "error_message": error_message,
            "raw_conf": json.dumps(raw_conf),
        },
    )