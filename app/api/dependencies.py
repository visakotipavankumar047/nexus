from functools import lru_cache

from fastapi import Request

from app.schemas.common import Meta
from app.services.chat_service import ChatService
from app.services.conversation_service import ConversationService
from app.services.document_service import DocumentService
from app.services.evaluation_service import EvaluationService


def get_meta(request: Request) -> Meta:
    return Meta(request_id=request.state.request_id)


@lru_cache
def get_chat_service() -> ChatService:
    return ChatService()


@lru_cache
def get_conversation_service() -> ConversationService:
    return ConversationService()


@lru_cache
def get_document_service() -> DocumentService:
    return DocumentService()


@lru_cache
def get_evaluation_service() -> EvaluationService:
    return EvaluationService()
