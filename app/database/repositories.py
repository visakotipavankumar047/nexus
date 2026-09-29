from typing import Any
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.database.models import Conversation, Document, Evaluation, Message

# Every user-owned query filters on user_id. IS NOT DISTINCT FROM so NULL (pre-auth) matches only NULL.


class ConversationRepository:
    def __init__(self, s: Session):
        self.s = s

    def list_all(self, user_id: UUID | None) -> list[Conversation]:
        q = select(Conversation).where(Conversation.user_id.is_not_distinct_from(user_id))
        return list(self.s.scalars(q.order_by(Conversation.updated_at.desc())))

    def get(self, conversation_id: UUID, user_id: UUID | None) -> Conversation | None:
        q = select(Conversation).where(Conversation.id == conversation_id,
                                       Conversation.user_id.is_not_distinct_from(user_id))
        return self.s.scalar(q)

    def create(self, title: str, user_id: UUID | None) -> Conversation:
        c = Conversation(title=title, user_id=user_id)
        self.s.add(c)
        self.s.flush()
        return c

    def delete(self, conversation_id: UUID, user_id: UUID | None) -> bool:
        q = delete(Conversation).where(Conversation.id == conversation_id,
                                       Conversation.user_id.is_not_distinct_from(user_id))
        return self.s.execute(q).rowcount > 0


class MessageRepository:
    """No user_id here: callers must authorize the conversation via ConversationRepository.get first."""

    def __init__(self, s: Session):
        self.s = s

    def list_all(self, conversation_id: UUID) -> list[Message]:
        q = select(Message).where(Message.conversation_id == conversation_id)
        return list(self.s.scalars(q.order_by(Message.created_at, Message.id)))

    def add(self, conversation_id: UUID, role: str, content: str) -> Message:
        m = Message(conversation_id=conversation_id, role=role, content=content)
        self.s.add(m)
        self.s.flush()
        return m


class EvaluationRepository:
    def __init__(self, s: Session):
        self.s = s

    def add(self, **fields: Any) -> Evaluation:
        e = Evaluation(**fields)
        self.s.add(e)
        self.s.flush()
        return e

    def list_recent(self, limit: int) -> list[Evaluation]:
        return list(self.s.scalars(select(Evaluation).order_by(Evaluation.created_at.desc()).limit(limit)))


class DocumentRepository:
    def __init__(self, s: Session):
        self.s = s

    def list_all(self, user_id: UUID | None) -> list[Document]:
        q = select(Document).where(Document.user_id.is_not_distinct_from(user_id))
        return list(self.s.scalars(q.order_by(Document.created_at.desc())))

    def get(self, document_id: UUID, user_id: UUID | None) -> Document | None:
        q = select(Document).where(Document.id == document_id, Document.user_id.is_not_distinct_from(user_id))
        return self.s.scalar(q)

    def create(self, user_id: UUID | None, **fields: Any) -> Document:
        d = Document(user_id=user_id, **fields)
        self.s.add(d)
        self.s.flush()
        return d

    def find_by_hash(self, sha256: str, user_id: UUID | None) -> Document | None:
        q = select(Document).where(Document.user_id.is_not_distinct_from(user_id),
                                   Document.meta["sha256"].as_string() == sha256, Document.status != "failed")
        return self.s.scalar(q.limit(1))

    def update(self, document_id: UUID, **fields: Any) -> None:
        if d := self.s.get(Document, document_id):
            for k, v in fields.items():
                setattr(d, k, v)

    def delete(self, document_id: UUID, user_id: UUID | None) -> bool:
        q = delete(Document).where(Document.id == document_id, Document.user_id.is_not_distinct_from(user_id))
        return self.s.execute(q).rowcount > 0
