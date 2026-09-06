"""
Embeddings — one embedding call, two interchangeable providers.

  local  (default) : fastembed / ONNX, runs in-container. No API key, no
                     network at task time (the model is baked into the
                     image), no rate limits. Good enough for a demo corpus.
  voyage           : hosted Voyage AI API. Better quality, needs a real
                     VOYAGE_API_KEY (they start with "pa-"; a MongoDB Atlas
                     key looks similar but is rejected with a 403).

Switch with EMBEDDING_PROVIDER. The dimension differs per provider, so
document_chunks.embedding must be sized to match EMBEDDING_DIM — see
sql/create_document_chunks_table.sql.
"""

import os
from functools import lru_cache

EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "local")

# provider -> (model name, output dimension)
_PROVIDERS = {
    "local": ("BAAI/bge-small-en-v1.5", 384),
    "voyage": ("voyage-3.5", 1024),
}

if EMBEDDING_PROVIDER not in _PROVIDERS:
    raise RuntimeError(
        f"EMBEDDING_PROVIDER={EMBEDDING_PROVIDER!r} is not one of {sorted(_PROVIDERS)}"
    )

EMBEDDING_MODEL, EMBEDDING_DIM = _PROVIDERS[EMBEDDING_PROVIDER]

# Where fastembed keeps the downloaded ONNX weights. Pre-populated in the
# Dockerfile so a task never pays the download cost (or fails offline).
FASTEMBED_CACHE_DIR = os.environ.get("FASTEMBED_CACHE_DIR", "/home/airflow/.cache/fastembed")


@lru_cache(maxsize=1)
def _local_model():
    """Load the ONNX model once per process — construction is the expensive
    part, so it's cached across tasks running in the same worker."""
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=EMBEDDING_MODEL, cache_dir=FASTEMBED_CACHE_DIR)


def _embed_local(chunks: list[str]) -> list[list[float]]:
    # bge-* models want a prefix only on *queries*; documents go in bare.
    return [vector.tolist() for vector in _local_model().embed(chunks)]


def _embed_voyage(chunks: list[str]) -> list[list[float]]:
    import voyageai

    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise RuntimeError("VOYAGE_API_KEY is not set")

    client = voyageai.Client(api_key=api_key)
    result = client.embed(
        chunks, model=EMBEDDING_MODEL, input_type="document", output_dtype="float"
    )
    return result.embeddings


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Embed a list of text chunks for storage/retrieval. These are being
    indexed as documents, not used as search queries — the two are embedded
    slightly differently for better retrieval quality."""
    if not chunks:
        return []

    embed = _embed_local if EMBEDDING_PROVIDER == "local" else _embed_voyage
    vectors = embed(chunks)

    if vectors and len(vectors[0]) != EMBEDDING_DIM:
        raise RuntimeError(
            f"{EMBEDDING_MODEL} returned {len(vectors[0])}-dim vectors, "
            f"expected {EMBEDDING_DIM} — document_chunks.embedding will reject these"
        )
    return vectors