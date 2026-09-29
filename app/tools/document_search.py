from uuid import UUID

from pydantic import BaseModel, Field

from app.rag.retriever import search
from app.schemas.common import Source
from app.tools.vector_search import to_evidence


class DocumentSearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    document_ids: list[UUID] = Field(min_length=1, max_length=20,
                                     description="Only search these documents (ids from earlier vector_search sources)")
    top_k: int = Field(default=5, ge=1, le=20)


async def document_search(query: str, document_ids: list[UUID], top_k: int = 5, *,
                          user_id: UUID | None) -> tuple[str, list[Source]]:
    # Still scoped to the caller's namespace, so foreign document ids simply match nothing.
    return to_evidence(await search(query, top_k, user_id, document_ids))
