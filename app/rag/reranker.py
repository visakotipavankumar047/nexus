"""Second-stage reranking with Pinecone's hosted cross-encoder (same PINECONE_API_KEY, no extra dependency)."""
from typing import Any

from app.config import get_settings
from app.utils.errors import NexusException
from app.utils.logging import get_logger

logger = get_logger("reranker")


async def rerank(query: str, hits: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Reorders hits by cross-encoder relevance, keeps top_k and drops those under rerank_min_score.
    Reranking is optional: on failure, falls back to vector order instead of failing the search."""
    from app.rag.retriever import get_pinecone, pinecone_call  # retriever imports this module

    s = get_settings()
    if not s.rerank_enabled or len(hits) <= 1:
        return hits[:top_k]
    try:
        res = await pinecone_call(get_pinecone().inference.rerank, model=s.rerank_model, query=query,
                                  documents=[h["content"] for h in hits], top_n=top_k, return_documents=False,
                                  parameters={"truncate": "END"})  # ~1000-token chunks can exceed its window
    except NexusException as e:
        logger.warning("rerank_failed_fallback_to_vector_order", exc_info=e)
        return hits[:top_k]
    return [{**hits[r.index], "score": r.score,
             "metadata": {**hits[r.index]["metadata"], "vector_score": hits[r.index]["score"]}}
            for r in res.data if r.score >= s.rerank_min_score]
