"""RAG-side embedding calls: batching + error mapping. The model itself lives in app/llm/embeddings.py,
so indexing and querying are guaranteed to use the same one."""
from app.config import get_settings
from app.llm import get_embeddings
from app.utils.errors import DocumentProcessingException, RetrievalException


async def embed_query(text: str) -> list[float]:
    try:
        return await get_embeddings().aembed_query(text)
    except Exception as e:
        raise RetrievalException("Embedding the query failed") from e


async def embed_documents(texts: list[str]) -> list[list[float]]:
    size = get_settings().embed_batch_size
    vectors: list[list[float]] = []
    try:
        for i in range(0, len(texts), size):
            vectors += await get_embeddings().aembed_documents(texts[i:i + size])
    except Exception as e:
        raise DocumentProcessingException("Embedding the document failed") from e
    if len(vectors) != len(texts):
        raise DocumentProcessingException("Embedding count mismatch")
    return vectors
