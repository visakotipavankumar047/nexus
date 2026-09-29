from pathlib import Path
from uuid import UUID

from fastapi import UploadFile

from app.config import get_settings
from app.database.repositories import DocumentRepository
from app.database.supabase import run_in_session
from app.rag import delete_vectors, ingest, namespace, search
from app.schemas.document import Document, SearchResult, VectorSearchRequest
from app.services.conversation_service import USER_ID
from app.utils.errors import NotFoundException, ValidationException


class DocumentService:
    async def upload(self, file: UploadFile | None, url: str | None) -> Document:
        if (file is None) == (url is None):
            raise ValidationException("Provide exactly one of 'file' or 'url'")
        if url is not None:
            return await ingest(user_id=USER_ID, filename=url[:200], url=url)
        filename = self._validate(file)
        limit = get_settings().max_upload_mb * 1024 * 1024
        data = await file.read(limit + 1)  # file.size can be missing; never trust it alone
        if len(data) > limit:
            raise ValidationException(f"File exceeds {get_settings().max_upload_mb} MB")
        if not data:
            raise ValidationException("File is empty")
        return await ingest(user_id=USER_ID, filename=filename, data=data)

    async def list_all(self) -> list[Document]:
        return await run_in_session(lambda s: [
            Document.model_validate(d) for d in DocumentRepository(s).list_all(USER_ID)])

    async def get(self, document_id: UUID) -> Document:
        doc = await run_in_session(lambda s: DocumentRepository(s).get(document_id, USER_ID))
        if doc is None:
            raise NotFoundException("Document not found")
        return Document.model_validate(doc)

    async def delete(self, document_id: UUID) -> None:
        """Deletes Pinecone vectors first, then Supabase metadata, so a failure leaves a retryable row."""
        await self.get(document_id)  # 404 + ownership check
        await delete_vectors(document_id, namespace(USER_ID))
        await run_in_session(lambda s: DocumentRepository(s).delete(document_id, USER_ID))

    async def vector_search(self, request: VectorSearchRequest) -> list[SearchResult]:
        hits = await search(request.query, request.top_k, USER_ID, rewrite=request.rewrite)
        return [SearchResult(**h) for h in hits]

    @staticmethod
    def _validate(file: UploadFile) -> str:
        name = Path(file.filename or "").name  # strips any client-supplied path
        ext = Path(name).suffix.lower().lstrip(".")
        if (ext if ext != "markdown" else "md") not in get_settings().allowed_upload_types:
            raise ValidationException(f"Unsupported file type '{ext}'")
        return name
