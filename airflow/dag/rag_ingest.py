"""
rag-ingest DAG — orchestration only.

Pipeline:
idempotency check -> extract -> chunk -> embed -> upsert

Every attempt is recorded in ingestion_manifest as:
- success
- failed
- skipped

The actual RAG logic lives in include/rag/*.

This DAG is externally triggered only:
S3 -> Lambda -> Airflow

No polling and no cron schedule.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

from airflow.decorators import dag, task

from rag import (
    chunking,
    embeddings,
    extraction,
    manifest,
    s3_client,
    vector_store,
)

logger = logging.getLogger(__name__)

S3_BUCKET = os.environ["S3_BUCKET"]


@dag(
    dag_id="rag-ingest",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["event-driven", "rag"],
)
def rag_ingest():

    @task
    def process(**context):

        dag_run = context["dag_run"]
        conf = dag_run.conf or {}

        dag_run_id = dag_run.run_id

        object_key = conf.get("object_key")
        trigger_source = conf.get("trigger_source", "manual")
        event_time = conf.get("event_time")

        if not object_key:
            raise ValueError(
                "Missing required DAG configuration: object_key"
            )

        logger.info(
            "Processing object_key=%s trigger_source=%s",
            object_key,
            trigger_source,
        )

        

        if manifest.already_processed(object_key):

            logger.info(
                "object_key=%s already processed successfully - skipping",
                object_key,
            )

            manifest.record(
                object_key,
                dag_run_id,
                trigger_source,
                event_time,
                "skipped",
                conf,
            )

            return



        try:

            # S3 -> bytes
            content_bytes = s3_client.read_object(
                S3_BUCKET,
                object_key,
            )

            # bytes -> text
            text = extraction.extract_text(
                content_bytes,
                object_key,
            )

            # text -> chunks
            chunks = chunking.chunk_text(text)

            # chunks -> embeddings
            vectors = embeddings.embed_chunks(chunks)

            # embeddings -> vector store
            vector_store.upsert_chunks(
                object_key,
                chunks,
                vectors,
            )

            logger.info(
                "Upserted %d chunks for object_key=%s",
                len(chunks),
                object_key,
            )

  

            manifest.record(
                object_key,
                dag_run_id,
                trigger_source,
                event_time,
                "success",
                conf,
            )

        except Exception as exc:

            logger.exception(
                "Processing failed for object_key=%s",
                object_key,
            )



            manifest.record(
                object_key,
                dag_run_id,
                trigger_source,
                event_time,
                "failed",
                conf,
                error_message=str(exc),
            )

            
            raise

    process()


rag_ingest()