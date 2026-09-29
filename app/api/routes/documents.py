from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from app.api.dependencies import get_document_service, get_meta
from app.schemas.common import ERROR_RESPONSES, Envelope, Meta
from app.schemas.document import Document, DocumentList
from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["documents"], responses=ERROR_RESPONSES)
Service = Annotated[DocumentService, Depends(get_document_service)]
MetaDep = Annotated[Meta, Depends(get_meta)]


@router.post("/upload", response_model=Envelope[Document], status_code=status.HTTP_201_CREATED,
             summary="Upload document",
             description="Upload a PDF/TXT/DOCX/Markdown file or submit a URL for ingestion.")
async def upload_document(
    service: Service,
    meta: MetaDep,
    file: Annotated[UploadFile | None, File()] = None,
    url: Annotated[str | None, Form(max_length=2048)] = None,
):
    return Envelope(data=await service.upload(file, url), meta=meta)


@router.get("", response_model=Envelope[DocumentList], summary="List documents",
            description="Returns the caller's uploaded documents.")
async def list_documents(service: Service, meta: MetaDep):
    return Envelope(data=DocumentList(documents=await service.list_all()), meta=meta)


@router.get("/{document_id}", response_model=Envelope[Document], summary="Get document",
            description="Returns one document's metadata.")
async def get_document(document_id: UUID, service: Service, meta: MetaDep):
    return Envelope(data=await service.get(document_id), meta=meta)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete document",
               description="Deletes document metadata and its Pinecone vectors.")
async def delete_document(document_id: UUID, service: Service):
    await service.delete(document_id)
