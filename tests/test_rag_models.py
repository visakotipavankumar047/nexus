import asyncio
from uuid import uuid4

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_openai import OpenAIEmbeddings

from app.llm import embeddings as llm_embeddings
from app.rag import metadata, retriever
from app.utils.errors import LLMException


@pytest.fixture
def settings(monkeypatch):
    s = llm_embeddings.get_settings()
    llm_embeddings.get_embeddings.cache_clear()
    yield lambda **kw: [monkeypatch.setattr(s, k, v) for k, v in kw.items()]
    llm_embeddings.get_embeddings.cache_clear()


def test_embedding_provider_switch(settings):
    settings(embedding_provider="huggingface")
    assert isinstance(llm_embeddings.get_embeddings(), HuggingFaceEndpointEmbeddings)
    assert llm_embeddings.embedding_model_id() == "huggingface:BAAI/bge-large-en-v1.5"

    llm_embeddings.get_embeddings.cache_clear()
    settings(embedding_provider="openai")
    emb = llm_embeddings.get_embeddings()
    assert isinstance(emb, OpenAIEmbeddings) and emb.dimensions == 1024  # fits the 1024-d index
    assert llm_embeddings.embedding_model_id() == "openai:text-embedding-3-large:1024"

    llm_embeddings.get_embeddings.cache_clear()
    settings(embedding_provider="cohere")
    with pytest.raises(LLMException):
        llm_embeddings.get_embeddings()


def test_search_filter_isolates_embedding_models():
    assert metadata.search_filter("m1") == {"embedding_model": {"$eq": "m1"}}
    d = uuid4()
    assert metadata.search_filter("m1", [d]) == {"$and": [{"embedding_model": {"$eq": "m1"}},
                                                          {"document_id": {"$in": [str(d)]}}]}


def _llm(text):
    return GenericFakeChatModel(messages=iter([AIMessage(text)]))


def test_rewrite_query(monkeypatch):
    monkeypatch.setattr(retriever, "get_llm", lambda name=None: _llm('"agentic RAG authentication design"\nextra'))
    assert asyncio.run(retriever.rewrite_query("hey so what does our doc say about auth?")) == \
        "agentic RAG authentication design"


def test_rewrite_falls_back_on_failure(monkeypatch):
    def boom(name=None):
        raise LLMException("no key")
    monkeypatch.setattr(retriever, "get_llm", boom)
    assert asyncio.run(retriever.rewrite_query("original question")) == "original question"
    monkeypatch.setattr(retriever, "get_llm", lambda name=None: _llm("   "))
    assert asyncio.run(retriever.rewrite_query("original question")) == "original question"


def test_search_uses_rewrite_and_model_filter(monkeypatch):
    seen = {}

    class Index:
        def query(self, **kw):
            seen.update(kw)
            return type("R", (), {"matches": []})()

    async def fake_embed(q):
        seen["embedded"] = q
        return [0.0]

    async def fake_rewrite(q):
        return "rewritten"

    monkeypatch.setattr(retriever, "get_index", lambda: Index())
    monkeypatch.setattr(retriever, "embed_query", fake_embed)
    monkeypatch.setattr(retriever, "rewrite_query", fake_rewrite)

    asyncio.run(retriever.search("raw question", 3, None, rewrite=True))
    assert seen["embedded"] == "rewritten"
    assert seen["filter"] == {"embedding_model": {"$eq": retriever.embedding_model_id()}}

    asyncio.run(retriever.search("raw question", 3, None))  # tools: no rewrite
    assert seen["embedded"] == "raw question"
