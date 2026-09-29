from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    filename: str
    file_type: str
    source: str
    status: str
    metadata: dict[str, Any] = Field(default={}, validation_alias="meta")  # ORM attr is `meta`
    created_at: datetime
    updated_at: datetime


class DocumentList(BaseModel):
    documents: list[Document]


class VectorSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    rewrite: bool = Field(default=True, description="Let the RAG LLM rewrite the question into a search query first")


class SearchResult(BaseModel):
    content: str
    score: float
    metadata: dict[str, Any]


class VectorSearchResponse(BaseModel):
    results: list[SearchResult]
