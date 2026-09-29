from uuid import uuid4

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database.models import Base, User
from app.database.repositories import ConversationRepository, DocumentRepository, MessageRepository


def _session() -> Session:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return Session(engine)


def test_conversations_are_isolated_per_user():
    s = _session()
    alice, bob = User(email="a@x"), User(email="b@x")
    s.add_all([alice, bob])
    s.flush()
    repo = ConversationRepository(s)
    c = repo.create("mine", alice.id)
    repo.create("anon", None)

    assert [x.title for x in repo.list_all(alice.id)] == ["mine"]
    assert [x.title for x in repo.list_all(None)] == ["anon"]
    assert repo.get(c.id, bob.id) is None
    assert repo.delete(c.id, bob.id) is False
    assert repo.delete(c.id, alice.id) is True
    assert repo.get(uuid4(), alice.id) is None


def test_messages_and_documents_roundtrip():
    s = _session()
    c = ConversationRepository(s).create("t", None)
    msgs = MessageRepository(s)
    msgs.add(c.id, "user", "hi")
    msgs.add(c.id, "assistant", "hello")
    assert [m.role for m in msgs.list_all(c.id)] == ["user", "assistant"]

    docs = DocumentRepository(s)
    d = docs.create(None, filename="a.pdf", file_type="pdf", source="upload", pinecone_namespace="user:anon")
    assert d.status == "pending" and d.meta == {}
    docs.update(d.id, status="indexed", meta={"sha256": "abc"})
    assert docs.get(d.id, None).status == "indexed"
    assert docs.find_by_hash("abc", None).id == d.id and docs.find_by_hash("abc", uuid4()) is None
    assert docs.get(d.id, uuid4()) is None
