from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_document_service, get_meta
from app.schemas.common import ERROR_RESPONSES, Envelope, Meta
from app.schemas.document import VectorSearchRequest, VectorSearchResponse
from app.services.document_service import DocumentService

router = APIRouter(prefix="/search", tags=["search"], responses=ERROR_RESPONSES)


@router.post("/vector", response_model=Envelope[VectorSearchResponse], summary="Vector search",
             description="Similarity search over the caller's private knowledge in Pinecone.")
async def vector_search(
    body: VectorSearchRequest,
    service: Annotated[DocumentService, Depends(get_document_service)],
    meta: Annotated[Meta, Depends(get_meta)],
):
    return Envelope(data=VectorSearchResponse(results=await service.vector_search(body)), meta=meta)
