"""Chunk metadata stored with each vector in Pinecone, and how it is read back."""
from typing import Any
from uuid import UUID

# Pinecone metadata values: str, number, bool or list[str]. null is rejected; 40KB per vector max.
_ALLOWED = (str, int, float, bool)
TEXT_KEY = "text"  # chunk text travels in metadata so search results need no second lookup


def vector_prefix(document_id: UUID) -> str:
    """Every chunk id of a document starts with this; deletion lists by prefix (serverless has no delete-by-filter)."""
    return f"{document_id}#"


def vector_id(document_id: UUID, chunk_index: int) -> str:
    return f"{vector_prefix(document_id)}{chunk_index}"


def sanitize(md: dict[str, Any]) -> dict[str, Any]:
    """Drops None, stringifies UUIDs and other types, keeps list[str]."""
    out: dict[str, Any] = {}
    for k, v in md.items():
        if v is None:
            continue
        if isinstance(v, list):
            out[k] = [str(x) for x in v]
        else:
            out[k] = v if isinstance(v, _ALLOWED) else str(v)
    return out


def chunk_metadata(*, document_id: UUID, filename: str, source: str, chunk_index: int, text: str,
                   embedding_model: str, page: int | None = None, section: str | None = None) -> dict[str, Any]:
    return sanitize({"document_id": document_id, "filename": filename, "source": source, "page": page,
                     "section": section, "chunk_index": chunk_index, "embedding_model": embedding_model,
                     TEXT_KEY: text})


def search_filter(embedding_model: str, document_ids: list[UUID] | None = None) -> dict[str, Any]:
    """Only match vectors made by the current embedding model (mixing models returns garbage)."""
    flt: dict[str, Any] = {"embedding_model": {"$eq": embedding_model}}
    if document_ids:
        flt = {"$and": [flt, {"document_id": {"$in": [str(d) for d in document_ids]}}]}
    return flt


def split_hit(metadata: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    """Pinecone match metadata -> (chunk text, metadata without the text)."""
    md = dict(metadata or {})
    return md.pop(TEXT_KEY, ""), md
