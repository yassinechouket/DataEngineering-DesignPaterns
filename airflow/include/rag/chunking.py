"""
Chunking — splits extracted text into overlapping, boundary-aware pieces.

Uses recursive splitting (paragraph -> line -> sentence -> word) rather
than blind fixed-size cuts, so chunks stay semantically coherent. Overlap
keeps context continuous across chunk boundaries.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ~1000 chars ~= 200-250 tokens for English text — a common starting
# point for RAG chunking: large enough to retain context, small enough
# to keep retrieval precise and embedding cost reasonable.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks on the cleanest available
    boundary (paragraph, then line, then sentence, then word)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)