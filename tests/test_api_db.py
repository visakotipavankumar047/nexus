from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import supabase
from app.database.models import Base
from app.main import app
from app.schemas.common import Source
from app.services import chat_service, document_service
from app.tools import tool_registry
from app.utils.errors import VectorDatabaseException

API = "/api/v1"


class FakeLLM(GenericFakeChatModel):
    bound: list = []

    def bind_tools(self, tools, **kwargs):
        self.bound = [t.name for t in tools]
        return self


class BrokenLLM(FakeLLM):
    async def _astream(self, *a, **k):
        raise RuntimeError("provider down")
        yield

    async def _agenerate(self, *a, **k):
        raise RuntimeError("provider down")


class ScriptedLLM(BaseChatModel):
    """Replays AIMessages in order (tool calls included); records what it was sent."""
    script: list
    seen: list = []

    @property
    def _llm_type(self):
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])


@pytest.fixture
def client(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    monkeypatch.setattr(supabase, "_session_factory", lambda: sessionmaker(engine, expire_on_commit=False))
    seen = []

    def fake_llm(name=None):
        seen.append(name)
        return FakeLLM(messages=iter([AIMessage("Hello there friend")] * 10))
    monkeypatch.setattr(chat_service, "get_llm", fake_llm)
    c = TestClient(app, raise_server_exceptions=False)
    c.models_seen = seen
    return c


def test_chat_creates_conversation_and_keeps_memory(client):
    first = client.post(f"{API}/chat", json={"message": "Explain RAG", "model": "kimi"}).json()["data"]
    assert first["answer"] == "Hello there friend" and first["grounded"] is False
    cid = first["conversation_id"]

    client.post(f"{API}/chat", json={"message": "More", "conversation_id": cid})
    msgs = client.get(f"{API}/conversations/{cid}/messages").json()["data"]["messages"]
    assert [(m["role"], m["content"]) for m in msgs] == [
        ("user", "Explain RAG"), ("assistant", "Hello there friend"), ("user", "More"), ("assistant", "Hello there friend")]
    assert client.models_seen == ["kimi", None]
    assert client.post(f"{API}/chat", json={"message": "x", "conversation_id": str(uuid4())}).status_code == 404


def test_chat_stream_tokens_and_persists(client):
    r = client.post(f"{API}/chat/stream", json={"message": "hi"})
    assert r.text.startswith("event: agent_step")
    assert r.text.count("event: token") >= 2 and "event: done" in r.text
    cid = r.text.split('"conversation_id": "')[1].split('"')[0]
    msgs = client.get(f"{API}/conversations/{cid}/messages").json()["data"]["messages"]
    assert msgs[1]["content"] == "Hello there friend"


def test_llm_failure_maps_to_502_and_stream_error(client, monkeypatch):
    monkeypatch.setattr(chat_service, "get_llm", lambda name=None: BrokenLLM(messages=iter([])))
    r = client.post(f"{API}/chat", json={"message": "hi"})
    assert r.status_code == 502 and r.json()["error"]["code"] == "LLM_FAILED"
    assert "provider down" not in r.text  # no internals leaked
    s = client.post(f"{API}/chat/stream", json={"message": "hi"})
    assert "event: error" in s.text and "LLM_FAILED" in s.text


def _call(name, args, i=0):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"call_{name}_{i}"}])


def test_chat_tool_loop_collects_tools_and_sources(client, monkeypatch):
    web = Source(id="src_web1", title="LangChain docs", type="web", url="https://docs.langchain.com")

    async def fake_web(query, **kwargs):
        return f"results for {query} [src_web1]", [web]

    async def failing_vector(query, top_k=5, *, user_id):
        raise VectorDatabaseException()

    monkeypatch.setattr(tool_registry, "web_search", fake_web)
    monkeypatch.setattr(tool_registry, "vector_search", failing_vector)
    llm = ScriptedLLM(script=[
        AIMessage("", tool_calls=[{"name": "calculator", "args": {"expression": "25 * 48"}, "id": "c1"},
                                  {"name": "web_search", "args": {"query": "latest langchain"}, "id": "c2"}]),
        _call("vector_search", {"query": "our architecture"}),
        AIMessage("25*48 is 1200 and LangChain changed [src_web1]"),
    ], seen=[])
    monkeypatch.setattr(chat_service, "get_llm", lambda name=None: llm)

    data = client.post(f"{API}/chat", json={"message": "q"}).json()["data"]
    assert data["answer"] == "25*48 is 1200 and LangChain changed [src_web1]"
    assert data["tools_used"] == ["calculator", "web_search", "vector_search"]
    assert [s["id"] for s in data["sources"]] == ["src_web1"] and data["grounded"] is True

    tool_msgs = [m for m in llm.seen[-1] if isinstance(m, ToolMessage)]
    assert tool_msgs[0].content == "25 * 48 = 1200"
    assert tool_msgs[2].status == "error" and "vector_search failed" in tool_msgs[2].content  # failure fed back

    from sqlalchemy import select
    from app.database.models import AgentRun, ToolCall
    with supabase.session() as db:  # monitoring trail persisted
        run = db.scalar(select(AgentRun))
        assert run.status == "completed" and run.tools_used == ["calculator", "web_search", "vector_search"]
        calls = {c.tool_name: c for c in db.scalars(select(ToolCall))}
        assert calls["calculator"].status == "success" and calls["vector_search"].status == "error"
        assert calls["web_search"].output["sources"] == ["src_web1"] and run.latency_ms is not None


def test_tool_rounds_are_capped(client, monkeypatch):
    monkeypatch.setattr(chat_service.get_settings(), "max_tool_rounds", 2)
    llm = ScriptedLLM(script=[_call("calculator", {"expression": "1+1"}, i) for i in range(2)]
                      + [AIMessage("final")], seen=[])
    monkeypatch.setattr(chat_service, "get_llm", lambda name=None: llm)
    data = client.post(f"{API}/chat", json={"message": "loop forever"}).json()["data"]
    assert data["answer"] == "final" and len(llm.seen) == 3


def test_tools_respect_request_flags(client, monkeypatch):
    llm = FakeLLM(messages=iter([AIMessage("ok")]))
    monkeypatch.setattr(chat_service, "get_llm", lambda name=None: llm)
    client.post(f"{API}/chat", json={"message": "q", "use_web": False, "use_private_knowledge": False})
    assert llm.bound == ["calculator"]


def test_vector_search_route(client, monkeypatch):
    async def fake_search(query, top_k, user_id, document_ids=None, *, rewrite=False):
        return [{"content": "chunk", "score": 0.9, "metadata": {"document_id": "d1", "page": 4}}][:top_k]
    monkeypatch.setattr(document_service, "search", fake_search)
    r = client.post(f"{API}/search/vector", json={"query": "agentic rag", "top_k": 1})
    assert r.json()["data"] == {"results": [{"content": "chunk", "score": 0.9,
                                             "metadata": {"document_id": "d1", "page": 4}}]}


def test_conversation_lifecycle(client):
    created = client.post(f"{API}/conversations", json={"title": "RAG research"})
    assert created.status_code == 201
    cid = created.json()["data"]["id"]

    assert [c["id"] for c in client.get(f"{API}/conversations").json()["data"]["conversations"]] == [cid]
    assert client.get(f"{API}/conversations/{cid}").json()["data"]["title"] == "RAG research"
    assert client.get(f"{API}/conversations/{cid}/messages").json()["data"] == {"messages": []}
    assert client.delete(f"{API}/conversations/{cid}").status_code == 204

    gone = client.get(f"{API}/conversations/{cid}")
    assert gone.status_code == 404 and gone.json()["error"]["code"] == "NOT_FOUND"
    assert client.delete(f"{API}/conversations/{cid}").status_code == 404
    assert client.get(f"{API}/conversations/{cid}/messages").status_code == 404


def test_documents_empty_and_404(client):
    assert client.get(f"{API}/documents").json()["data"] == {"documents": []}
    assert client.get(f"{API}/documents/{uuid4()}").status_code == 404
    assert client.delete(f"{API}/documents/{uuid4()}").status_code == 404


def test_metrics_use_route_templates(client):
    client.get(f"{API}/conversations/{uuid4()}")
    data = client.get(f"{API}/metrics").json()["data"]
    assert data["GET /conversations/{conversation_id}"]["count"] >= 1
    assert not any(str(k).count("-") >= 4 for k in data)  # no raw UUIDs as keys
