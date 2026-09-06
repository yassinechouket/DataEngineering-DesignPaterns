"""
Text extraction — turns raw file bytes into plain text, dispatched by
file extension. Add new formats here as separate branches; keep the
public function signature stable so callers never need to change.
"""

import io
from pathlib import PurePosixPath

import fitz  


class UnsupportedFileType(ValueError):
    """Raised when an object's extension has no extraction handler."""


def extract_text(content_bytes: bytes, object_key: str) -> str:
    """Extract plain text from raw file bytes based on the object key's
    extension. object_key is only used to determine file type."""
    suffix = PurePosixPath(object_key).suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(content_bytes)
    elif suffix in (".txt", ".md"):
        return content_bytes.decode("utf-8")
    else:
        raise UnsupportedFileType(f"No extraction handler for suffix: {suffix}")


def _extract_pdf(content_bytes: bytes) -> str:
    doc = fitz.open(stream=io.BytesIO(content_bytes), filetype="pdf")
    try:
        pages = [page.get_text() for page in doc]
    finally:
        doc.close()
    return "\n\n".join(pages)