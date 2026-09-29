from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status

from app.api.dependencies import get_conversation_service, get_meta
from app.schemas.common import ERROR_RESPONSES, Envelope, Meta
from app.schemas.conversation import Conversation, ConversationCreate, ConversationList, MessageList
from app.services.conversation_service import ConversationService

router = APIRouter(prefix="/conversations", tags=["conversations"], responses=ERROR_RESPONSES)
Service = Annotated[ConversationService, Depends(get_conversation_service)]
MetaDep = Annotated[Meta, Depends(get_meta)]


@router.get("", response_model=Envelope[ConversationList], summary="List conversations",
            description="Returns the caller's conversations.")
async def list_conversations(service: Service, meta: MetaDep):
    return Envelope(data=ConversationList(conversations=await service.list_all()), meta=meta)


@router.post("", response_model=Envelope[Conversation], status_code=status.HTTP_201_CREATED,
             summary="Create conversation", description="Creates an empty conversation.")
async def create_conversation(body: ConversationCreate, service: Service, meta: MetaDep):
    return Envelope(data=await service.create(body), meta=meta)


@router.get("/{conversation_id}", response_model=Envelope[Conversation], summary="Get conversation",
            description="Returns one conversation by id.")
async def get_conversation(conversation_id: UUID, service: Service, meta: MetaDep):
    return Envelope(data=await service.get(conversation_id), meta=meta)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT,
               summary="Delete conversation", description="Deletes a conversation and its messages.")
async def delete_conversation(conversation_id: UUID, service: Service):
    await service.delete(conversation_id)


@router.get("/{conversation_id}/messages", response_model=Envelope[MessageList], summary="List messages",
            description="Returns the messages of a conversation in order.")
async def list_messages(conversation_id: UUID, service: Service, meta: MetaDep):
    return Envelope(data=MessageList(messages=await service.messages(conversation_id)), meta=meta)
