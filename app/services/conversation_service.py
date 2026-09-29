from uuid import UUID

from sqlalchemy.orm import Session

from app.database.repositories import ConversationRepository, MessageRepository
from app.database.supabase import run_in_session
from app.schemas.conversation import Conversation, ConversationCreate, Message
from app.utils.errors import NotFoundException

# ponytail: no auth yet (Phase 8), every call runs as the anonymous user; take user_id from the auth dependency then
USER_ID: UUID | None = None


class ConversationService:
    async def list_all(self) -> list[Conversation]:
        return await run_in_session(lambda s: [
            Conversation.model_validate(c) for c in ConversationRepository(s).list_all(USER_ID)])

    async def create(self, payload: ConversationCreate) -> Conversation:
        return await run_in_session(lambda s: Conversation.model_validate(
            ConversationRepository(s).create(payload.title, USER_ID)))

    async def get(self, conversation_id: UUID) -> Conversation:
        return await run_in_session(lambda s: Conversation.model_validate(_owned(s, conversation_id)))

    async def delete(self, conversation_id: UUID) -> None:
        if not await run_in_session(lambda s: ConversationRepository(s).delete(conversation_id, USER_ID)):
            raise NotFoundException("Conversation not found")

    async def messages(self, conversation_id: UUID) -> list[Message]:
        def load(s: Session) -> list[Message]:
            _owned(s, conversation_id)
            return [Message.model_validate(m) for m in MessageRepository(s).list_all(conversation_id)]
        return await run_in_session(load)


def _owned(s: Session, conversation_id: UUID):
    if (c := ConversationRepository(s).get(conversation_id, USER_ID)) is None:
        raise NotFoundException("Conversation not found")
    return c
