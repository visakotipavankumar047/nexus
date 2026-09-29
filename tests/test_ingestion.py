import asyncio
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import supabase
from app.database.models import Base
from app.rag import embeddings as rag_embeddings
from app.rag import ingestion, loaders, metadata, reranker, retriever
from app.utils.errors import DocumentProcessingException, VectorDatabaseException


class FakeEmbeddings:
    async def aembed_documents(self, texts):
        return [[0.1] * 4 for _ in texts]


class FakeIndex:
    def __init__(self):
        self.vectors: dict[str, dict] = {}

    def upsert(self, vectors, namespace):
        self.vectors |= {v["id"]: {**v, "ns": namespace} for v in vectors}

    def list(self, prefix, namespace):
        yield [i for i, v in self.vectors.items() if i.startswith(prefix) and v["ns"] == namespace]

    def delete(self, ids, namespace):
        for i in ids:
            self.vectors.pop(i)


@pytest.fixture
def index(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    monkeypatch.setattr(supabase, "_session_factory", lambda: sessionmaker(engine, expire_on_commit=False))
    idx = FakeIndex()
    monkeypatch.setattr(ingestion, "get_index", lambda: idx)
    monkeypatch.setattr(rag_embeddings, "get_embeddings", lambda: FakeEmbeddings())
    return idx


def test_ingest_chunks_upserts_dedupes_and_deletes(index):
    text = ("NEXUS architecture. " * 800).encode()  # ~3k tokens -> several chunks
    doc = asyncio.run(ingestion.ingest(user_id=None, filename="arch.md", data=text))
    assert doc.status == "indexed" and doc.metadata["chunks"] > 1
    ids = sorted(index.vectors)
    assert all(i.startswith(f"{doc.id}#") for i in ids) and len(ids) == doc.metadata["chunks"]
    md = index.vectors[ids[0]]["metadata"]
    assert md["filename"] == "arch.md" and "page" not in md and md["text"]  # no null metadata
    assert md["embedding_model"] == doc.metadata["embedding_model"] == "huggingface:BAAI/bge-large-en-v1.5"
    assert index.vectors[ids[0]]["ns"] == "user:anonymous"

    again = asyncio.run(ingestion.ingest(user_id=None, filename="copy.md", data=text))
    assert again.id == doc.id  # duplicate content -> same document, no re-index

    asyncio.run(ingestion.delete_vectors(doc.id, "user:anonymous"))
    assert index.vectors == {}


def test_failed_ingest_marks_document_failed_and_cleans_up(index, monkeypatch):
    class Boom(FakeEmbeddings):
        async def aembed_documents(self, texts):
            raise RuntimeError("hf down")
    monkeypatch.setattr(rag_embeddings, "get_embeddings", lambda: Boom())
    with pytest.raises(DocumentProcessingException):
        asyncio.run(ingestion.ingest(user_id=None, filename="a.txt", data=b"hello world"))
    assert index.vectors == {}


@pytest.mark.parametrize("data, name", [(b"", "a.txt"), (b"not a pdf", "a.pdf"), (b"x", "a.exe")])
def test_loader_rejects_bad_input(data, name):
    with pytest.raises(DocumentProcessingException):
        loaders.load_bytes(data, name)


@pytest.mark.parametrize("url", ["http://127.0.0.1/admin", "http://169.254.169.254/latest", "file:///etc/passwd",
                                 "http://localhost:8000/"])
def test_url_loader_blocks_ssrf(url):
    with pytest.raises(DocumentProcessingException):
        asyncio.run(loaders.load_url(url))


def test_metadata_is_pinecone_safe():
    md = metadata.chunk_metadata(document_id=uuid4(), filename="a.pdf", source="upload", chunk_index=0,
                                 text="t", embedding_model="m", page=None, section=None)
    assert None not in md.values() and "page" not in md and isinstance(md["document_id"], str)
    assert metadata.split_hit({"text": "body", "page": 2}) == ("body", {"page": 2})


class Match:
    def __init__(self, i):
        self.score, self.metadata = 0.9 - i / 100, {"text": f"chunk {i}", "document_id": "d", "chunk_index": i}


class FakePinecone:
    def __init__(self, fail=False):
        self.fail, self.seen = fail, {}
        self.inference = self

    def query(self, **kw):
        self.seen["query"] = kw
        return type("R", (), {"matches": [Match(i) for i in range(kw["top_k"])]})()

    def rerank(self, **kw):
        if self.fail:
            raise RuntimeError("rerank down")
        self.seen["rerank"] = kw
        # reverse relevance: last candidate is best
        n = len(kw["documents"])
        return type("R", (), {"data": [type("D", (), {"index": n - 1 - j, "score": 0.99 - j / 10})()
                                       for j in range(kw["top_n"])]})()


@pytest.mark.parametrize("fail", [False, True])
def test_search_overfetches_then_reranks(monkeypatch, fail):
    pc = FakePinecone(fail)
    monkeypatch.setattr(retriever, "get_index", lambda: pc)
    monkeypatch.setattr(retriever, "get_pinecone", lambda: pc)

    async def fake_embed(q):
        return [0.1] * 4
    monkeypatch.setattr(retriever, "embed_query", fake_embed)

    hits = asyncio.run(retriever.search("q", 2, None))
    assert pc.seen["query"]["top_k"] == 6 and pc.seen["query"]["namespace"] == "user:anonymous"
    assert len(hits) == 2 and "text" not in hits[0]["metadata"]
    if fail:  # reranker down -> vector order, search still works
        assert [h["content"] for h in hits] == ["chunk 0", "chunk 1"]
    else:
        assert [h["content"] for h in hits] == ["chunk 5", "chunk 4"]
        assert hits[0]["score"] == 0.99 and hits[0]["metadata"]["vector_score"] == pytest.approx(0.85)


def test_pinecone_failure_is_typed(monkeypatch):
    def boom(**kw):
        raise RuntimeError("down")
    with pytest.raises(VectorDatabaseException):
        asyncio.run(retriever.pinecone_call(boom))


def test_rerank_disabled_keeps_order(monkeypatch):
    monkeypatch.setattr(reranker.get_settings(), "rerank_enabled", False)
    hits = [{"content": str(i), "score": 1, "metadata": {}} for i in range(5)]
    assert asyncio.run(reranker.rerank("q", hits, 2)) == hits[:2]


def test_rerank_drops_irrelevant(monkeypatch):
    pc = FakePinecone()
    monkeypatch.setattr(retriever, "get_pinecone", lambda: pc)
    monkeypatch.setattr(reranker.get_settings(), "rerank_min_score", 0.9)
    hits = [{"content": str(i), "score": 0.5, "metadata": {}} for i in range(5)]
    assert [h["score"] for h in asyncio.run(reranker.rerank("q", hits, 3))] == [0.99]  # 0.89, 0.79 dropped
