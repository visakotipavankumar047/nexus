import asyncio
from hashlib import sha256
from uuid import UUID

from app.database.repositories import DocumentRepository
from app.database.supabase import run_in_session
from app.llm import embedding_model_id
from app.rag.chunking import chunk
from app.rag.embeddings import embed_documents
from app.rag.loaders import file_type, load_bytes, load_url
from app.rag.metadata import chunk_metadata, vector_id, vector_prefix
from app.rag.retriever import get_index, namespace, pinecone_call
from app.schemas.document import Document
from app.utils.errors import DocumentProcessingException, NexusException
from app.utils.logging import get_logger

logger = get_logger("ingestion")
UPSERT_BATCH = 100


async def ingest(*, user_id: UUID | None, filename: str, data: bytes | None = None, url: str | None = None) -> Document:
    """Upload -> extract -> clean -> chunk -> embed -> Pinecone -> Supabase metadata.
    Idempotent per (content, embedding model): re-uploading the same text returns the existing document;
    after switching embedding provider the same text is indexed again."""
    pages = await load_url(url) if url else await asyncio.to_thread(load_bytes, data, filename)
    model = embedding_model_id()
    digest = sha256((model + "\n" + "\n".join(t for _, t in pages)).encode()).hexdigest()
    if existing := await run_in_session(lambda s: DocumentRepository(s).find_by_hash(digest, user_id)):
        return Document.model_validate(existing)

    ns = namespace(user_id)
    doc = await run_in_session(lambda s: Document.model_validate(DocumentRepository(s).create(
        user_id, filename=filename, file_type="url" if url else file_type(filename), source=url or "upload",
        status="processing", pinecone_namespace=ns, meta={"sha256": digest, "embedding_model": model})))
    try:
        chunks = await asyncio.to_thread(chunk, pages)
        n = await _index_chunks(doc.id, filename, url, ns, chunks, model)
    except Exception as e:
        await run_in_session(lambda s: DocumentRepository(s).update(doc.id, status="failed"))
        logger.error("ingestion_failed", exc_info=e, extra={"fields": {"document_id": str(doc.id)}})
        await _safe_delete_vectors(doc.id, ns)
        if isinstance(e, NexusException):
            raise
        raise DocumentProcessingException() from e

    meta = {"sha256": digest, "embedding_model": model, "chunks": n, "pages": len(pages)}
    await run_in_session(lambda s: DocumentRepository(s).update(doc.id, status="indexed", meta=meta))
    logger.info("document_indexed", extra={"fields": {"document_id": str(doc.id), "chunks": n}})
    return doc.model_copy(update={"status": "indexed", "metadata": meta})


async def _index_chunks(doc_id: UUID, filename: str, url: str | None, ns: str,
                        chunks: list[tuple[int | None, str]], model: str) -> int:
    vectors = await embed_documents([t for _, t in chunks])
    records = [
        {"id": vector_id(doc_id, i), "values": vec,
         "metadata": chunk_metadata(document_id=doc_id, filename=filename, source=url or "upload",
                                    chunk_index=i, text=text, page=page, embedding_model=model)}
        for i, ((page, text), vec) in enumerate(zip(chunks, vectors, strict=True))
    ]
    for i in range(0, len(records), UPSERT_BATCH):
        await pinecone_call(get_index().upsert, vectors=records[i:i + UPSERT_BATCH], namespace=ns)
    return len(records)


async def delete_vectors(doc_id: UUID, ns: str) -> None:
    index = get_index()
    pages = await pinecone_call(lambda: list(index.list(prefix=vector_prefix(doc_id), namespace=ns)))
    for ids in pages:
        if ids:
            await pinecone_call(index.delete, ids=ids, namespace=ns)


async def _safe_delete_vectors(doc_id: UUID, ns: str) -> None:
    try:
        await delete_vectors(doc_id, ns)
    except NexusException:
        logger.warning("vector_cleanup_failed", extra={"fields": {"document_id": str(doc_id)}})
