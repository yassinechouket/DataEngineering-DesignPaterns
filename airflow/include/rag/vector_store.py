"""
Vector store — upserts chunks + embeddings into document_chunks.

Lives in the separate rag_data database (see AIRFLOW_CONN_RAG_DATA),
kept apart from Airflow's own metadata and from ingestion_manifest.
"""

from airflow.providers.postgres.hooks.postgres import PostgresHook

RAG_DATA_CONN_ID = "rag_data"


def _embedding_to_pg_literal(embedding: list[float]) -> str:
    """pgvector accepts a bracketed literal like '[0.1,0.2,...]'::vector.
    Building this string ourselves avoids needing psycopg's pgvector
    type-adapter registered just for this one write path."""
    return "[" + ",".join(repr(x) for x in embedding) + "]"


def upsert_chunks(object_key: str, chunks: list[str], embeddings: list[list[float]]) -> None:
    """Write chunks + embeddings for one object. Re-running for the same
    object_key overwrites existing rows for matching chunk_index (see the
    ON CONFLICT clause) rather than duplicating them."""
    hook = PostgresHook(postgres_conn_id=RAG_DATA_CONN_ID)
    conn = hook.get_conn()
    try:
        with conn.cursor() as cur:
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                cur.execute(
                    """
                    INSERT INTO document_chunks (object_key, chunk_index, content, embedding)
                    VALUES (%(object_key)s, %(chunk_index)s, %(content)s, %(embedding)s::vector)
                    ON CONFLICT (object_key, chunk_index)
                    DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding
                    """,
                    {
                        "object_key": object_key,
                        "chunk_index": i,
                        "content": chunk,
                        "embedding": _embedding_to_pg_literal(embedding),
                    },
                )
        conn.commit()
    finally:
        conn.close()