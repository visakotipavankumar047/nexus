import asyncio
from functools import lru_cache
from typing import Any
from uuid import UUID

from pinecone import Pinecone

from app.config import get_settings
from app.llm import embedding_model_id, get_llm
from app.rag.embeddings import embed_query
from app.rag.metadata import search_filter, split_hit
from app.rag.reranker import rerank
from app.utils.errors import VectorDatabaseException
from app.utils.logging import get_logger

logger = get_logger("retriever")


@lru_cache
def get_pinecone() -> Pinecone:
    return Pinecone(api_key=get_settings().pinecone_api_key.get_secret_value())


@lru_cache
def get_index():
    return get_pinecone().Index(get_settings().pinecone_index)


def namespace(user_id: UUID | None) -> str:
    """One namespace per user: a query can never reach another user's vectors."""
    return f"user:{user_id or 'anonymous'}"


async def pinecone_call(fn, *args, **kwargs):
    """Pinecone SDK is sync: run it off the event loop, with the configured timeout."""
    try:
        return await asyncio.wait_for(asyncio.to_thread(fn, *args, **kwargs), get_settings().pinecone_timeout_s)
    except Exception as e:
        raise VectorDatabaseException() from e


REWRITE_PROMPT = """Rewrite the user's question into one concise search query for a semantic document index.
Keep names, numbers and technical terms. Drop filler and conversational words. Output only the query.

Question: {question}"""


async def rewrite_query(question: str) -> str:
    """LLM query rewriting (SKILL.md query flow). Best effort: on any failure the original question is used."""
    try:
        out = (await get_llm(get_settings().rag_llm_model).ainvoke(REWRITE_PROMPT.format(question=question))).text
    except Exception as e:
        logger.warning("query_rewrite_failed", exc_info=e)
        return question
    out = out.strip().splitlines()[0].strip().strip("\"'") if out.strip() else ""
    return out[:2000] or question


async def search(query: str, top_k: int, user_id: UUID | None, document_ids: list[UUID] | None = None,
                 *, rewrite: bool = False) -> list[dict[str, Any]]:
    """Query -> (LLM rewrite) -> embedding -> Pinecone similarity (over-fetch) -> rerank -> top_k.
    Returns [{content, score, metadata}]. Tools pass rewrite=False: the agent LLM already wrote the query."""
    s = get_settings()
    search_query = await rewrite_query(query) if rewrite and s.rag_query_rewrite else query
    candidates = min(top_k * s.rerank_candidates_factor, 50) if s.rerank_enabled else top_k
    res = await pinecone_call(get_index().query, vector=await embed_query(search_query), top_k=candidates,
                              namespace=namespace(user_id), include_metadata=True,
                              filter=search_filter(embedding_model_id(), document_ids))
    hits = []
    for m in res.matches:
        text, md = split_hit(m.metadata)
        hits.append({"content": text, "score": m.score, "metadata": md})
    return await rerank(search_query, hits, top_k)
