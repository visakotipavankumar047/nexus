from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from app.rag.retriever import search
from app.schemas.common import Source
from app.utils.helpers import UNTRUSTED_BANNER, source_id


class VectorSearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=2000, description="What to look for in the user's private documents")
    top_k: int = Field(default=5, ge=1, le=20)


def to_evidence(hits: list[dict[str, Any]]) -> tuple[str, list[Source]]:
    """(text for the LLM, sources for the API response)."""
    if not hits:
        return "No matching private documents found.", []
    sources, blocks = [], []
    for h in hits:
        md = h["metadata"]
        src = Source(id=source_id(f"{md.get('document_id')}:{md.get('chunk_index')}:{h['content'][:50]}"),
                     title=md.get("filename", "Private document"), type="private",
                     document_id=md.get("document_id"), page=md.get("page"), score=round(h["score"], 4))
        sources.append(src)
        blocks.append(f"[{src.id}] {src.title}" + (f" (page {src.page})" if src.page else "") + f"\n{h['content']}")
    return f"{UNTRUSTED_BANNER}\n\n" + "\n\n".join(blocks), sources


async def vector_search(query: str, top_k: int = 5, *, user_id: UUID | None) -> tuple[str, list[Source]]:
    return to_evidence(await search(query, top_k, user_id))
