"""
RAG ingestion DAG for Amazon MWAA.

Flow:

S3 upload
    ↓
Lambda
    ↓
MWAA REST API
    ↓
rag-ingest DAG
    ↓
idempotency check
    ↓
extract
    ↓
chunk
    ↓
embed
    ↓
vector upsert
    ↓
manifest

This DAG is externally triggered only.
There is no cron schedule.
"""

from __future__ import annotations

import logging
from datetime import datetime

from airflow.decorators import dag, task
from airflow.operators.python import get_current_context

from rag import (
    chunking,
    embeddings,
    extraction,
    manifest,
    s3_client,
    vector_store,
)

logger = logging.getLogger(__name__)


@dag(
    dag_id="rag-ingest",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["event-driven", "rag"],
)
def rag_ingest():

    @task
    def process():

        context = get_current_context()

        dag_run = context["dag_run"]

        # Configuration sent by Lambda
        conf = dag_run.conf or {}

        bucket = conf.get("bucket")
        object_key = conf.get("object_key")
        trigger_source = conf.get(
            "trigger_source",
            "s3-lambda",
        )
        event_time = conf.get("event_time")

        dag_run_id = dag_run.run_id

        logger.info(
            "Starting RAG ingestion: bucket=%s object_key=%s "
            "trigger_source=%s dag_run_id=%s",
            bucket,
            object_key,
            trigger_source,
            dag_run_id,
        )

        # --------------------------------------------------
        # Validate input
        # --------------------------------------------------

        if not bucket:
            raise ValueError("Missing 'bucket' in dag_run.conf")

        if not object_key:
            raise ValueError(
                "Missing 'object_key' in dag_run.conf"
            )

        # --------------------------------------------------
        # Idempotency check
        # --------------------------------------------------

        if manifest.already_processed(
            bucket=bucket,
            object_key=object_key,
        ):

            logger.info(
                "Object already processed successfully: %s",
                object_key,
            )

            manifest.record(
                bucket=bucket,
                object_key=object_key,
                dag_run_id=dag_run_id,
                trigger_source=trigger_source,
                event_time=event_time,
                status="skipped",
                conf=conf,
            )

            return

        # --------------------------------------------------
        # Main RAG pipeline
        # --------------------------------------------------

        try:

            # 1. Read object from S3
            content_bytes = s3_client.read_object(
                bucket,
                object_key,
            )

            logger.info(
                "Downloaded %s from bucket %s",
                object_key,
                bucket,
            )

            # 2. Extract text
            text = extraction.extract_text(
                content_bytes,
                object_key,
            )

            logger.info(
                "Extracted text from %s",
                object_key,
            )

            # 3. Chunk text
            chunks = chunking.chunk_text(text)

            logger.info(
                "Created %d chunks",
                len(chunks),
            )

            # 4. Generate embeddings
            vectors = embeddings.embed_chunks(
                chunks
            )

            logger.info(
                "Generated embeddings for %d chunks",
                len(vectors),
            )

            # 5. Store vectors
            vector_store.upsert_chunks(
                object_key,
                chunks,
                vectors,
            )

            logger.info(
                "Successfully upserted vectors for %s",
                object_key,
            )

            # 6. Record success
            manifest.record(
                bucket=bucket,
                object_key=object_key,
                dag_run_id=dag_run_id,
                trigger_source=trigger_source,
                event_time=event_time,
                status="success",
                conf=conf,
            )

        except Exception as exc:

            logger.exception(
                "RAG processing failed for %s",
                object_key,
            )

            # Record failure
            manifest.record(
                bucket=bucket,
                object_key=object_key,
                dag_run_id=dag_run_id,
                trigger_source=trigger_source,
                event_time=event_time,
                status="failed",
                conf=conf,
                error_message=str(exc),
            )

            # Important:
            # Airflow task must also become FAILED.
            raise

    process()


rag_ingest()