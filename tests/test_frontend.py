"""Frontend <-> backend contract: the Streamlit ApiClient driven against the real FastAPI app (fake LLM, SQLite)."""
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
# before any frontend import: frontend/config.py reads these once
os.environ["NEXUS_AUTO_START_BACKEND"] = "false"
os.environ["NEXUS_BACKEND_URL"] = "http://127.0.0.1:9"  # nothing listens here
sys.path.append(str(FRONTEND))  # after the root, so `app` stays the backend package

from services.api_client import ApiClient, ApiError  # noqa: E402

from app.database import supabase  # noqa: E402
from app.database.models import Base  # noqa: E402
from app.main import app  # noqa: E402
from app.services import chat_service  # noqa: E402
from tests.test_api_db import FakeLLM  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402


@pytest.fixture
def api(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    monkeypatch.setattr(supabase, "_session_factory", lambda: sessionmaker(engine, expire_on_commit=False))
    monkeypatch.setattr(chat_service, "get_llm", lambda name=None: FakeLLM(messages=iter([AIMessage("Hi from NEXUS")])))
    return ApiClient(http=TestClient(app, base_url="http://testserver/api/v1", raise_server_exceptions=False))


def test_chat_stream_round_trip(api):
    events = list(api.chat_stream({"message": "hello", "use_web": False, "use_private_knowledge": False}))
    kinds = [e for e, _ in events]
    assert kinds[0] == "agent_step" and "token" in kinds and kinds[-1] == "done"
    done = events[-1][1]
    assert done["answer"] == "Hi from NEXUS"

    convs = api.conversations()
    assert [c["id"] for c in convs] == [done["conversation_id"]]
    assert [m["role"] for m in api.messages(done["conversation_id"])] == ["user", "assistant"]
    api.delete_conversation(done["conversation_id"])
    assert api.conversations() == []


def test_errors_become_api_error(api):
    with pytest.raises(ApiError) as e:
        api.messages("00000000-0000-0000-0000-000000000000")
    assert e.value.code == "NOT_FOUND" and e.value.request_id.startswith("req_")

    with pytest.raises(ApiError) as e:
        api.upload_file("virus.exe", b"MZ", None)
    assert e.value.code == "VALIDATION_ERROR"


def test_health_documents_evaluations(api):
    assert api.health()["status"] == "ok"
    assert api.documents() == []
    assert api.evaluations() == {"evaluations": [], "averages": {
        "faithfulness": None, "relevance": None, "context_precision": None, "context_recall": None}}


def test_unreachable_backend():
    with pytest.raises(ApiError) as e:
        ApiClient("http://127.0.0.1:9/api/v1").health()
    assert e.value.code == "BACKEND_UNREACHABLE"


def test_app_renders_without_backend():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(FRONTEND / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert any("Backend offline" in c.value for c in at.sidebar.caption)
    assert at.title[0].value == "What should I research?"
