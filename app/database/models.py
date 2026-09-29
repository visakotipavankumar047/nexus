from datetime import UTC, datetime, timedelta
from threading import Lock
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, CheckConstraint, DateTime, ForeignKey, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

Json = JSON().with_variant(JSONB, "postgresql")


_last = datetime.min.replace(tzinfo=UTC)
_lock = Lock()


def _now() -> datetime:
    # Strictly increasing so messages saved in one step keep their order.
    # ponytail: per-process only; add an identity `seq` column if several workers write one conversation at once
    global _last
    with _lock:
        _last = max(datetime.now(UTC), _last + timedelta(microseconds=1))
        return _last


class Base(DeclarativeBase):
    type_annotation_map = {datetime: DateTime(timezone=True), dict[str, Any]: Json, list[str]: Json}


class _Created:
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    created_at: Mapped[datetime] = mapped_column(default=_now, server_default=func.now())


class _Timestamps(_Created):
    updated_at: Mapped[datetime] = mapped_column(default=_now, onupdate=_now, server_default=func.now())


class User(_Timestamps, Base):
    __tablename__ = "users"
    email: Mapped[str] = mapped_column(Text, unique=True)


class Conversation(_Timestamps, Base):
    __tablename__ = "conversations"
    # ponytail: nullable until auth lands (Phase 8), then NOT NULL
    user_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(Text)


class Message(_Created, Base):
    __tablename__ = "messages"
    __table_args__ = (CheckConstraint("role IN ('user', 'assistant')", name="messages_role_check"),)
    conversation_id: Mapped[UUID] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)


class Document(_Timestamps, Base):
    __tablename__ = "documents"
    user_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    filename: Mapped[str] = mapped_column(Text)
    file_type: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, default="pending", server_default="pending")
    pinecone_namespace: Mapped[str] = mapped_column(Text)
    # "metadata" is reserved on declarative classes
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", default=dict, server_default="{}")


class AgentRun(_Created, Base):
    __tablename__ = "agent_runs"
    conversation_id: Mapped[UUID | None] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    query: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)
    latency_ms: Mapped[int | None]
    tools_used: Mapped[list[str]] = mapped_column(default=list, server_default="[]")


class ToolCall(_Created, Base):
    __tablename__ = "tool_calls"
    agent_run_id: Mapped[UUID] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True)
    tool_name: Mapped[str] = mapped_column(Text)
    input: Mapped[dict[str, Any]]
    output: Mapped[dict[str, Any] | None]
    latency_ms: Mapped[int | None]
    status: Mapped[str] = mapped_column(Text)


class Evaluation(_Created, Base):
    __tablename__ = "evaluations"
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    faithfulness: Mapped[float | None]
    relevance: Mapped[float | None]
    context_precision: Mapped[float | None]
    context_recall: Mapped[float | None]
