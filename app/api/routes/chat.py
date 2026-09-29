import json
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_chat_service, get_meta
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.common import ERROR_RESPONSES, Envelope, Meta
from app.services.chat_service import ChatService
from app.utils.errors import NexusException

router = APIRouter(prefix="/chat", tags=["chat"], responses=ERROR_RESPONSES)
Service = Annotated[ChatService, Depends(get_chat_service)]


@router.post("", response_model=Envelope[ChatResponse], summary="Ask NEXUS",
             description="Runs the agent over private knowledge and/or live web and returns a grounded answer.")
async def chat(body: ChatRequest, service: Service, meta: Meta = Depends(get_meta)):
    return Envelope(data=await service.run(body), meta=meta)


@router.post("/stream", summary="Ask NEXUS (SSE)",
             description="Server-Sent Events: agent_step, source, token, done, error.",
             response_class=StreamingResponse)
async def chat_stream(body: ChatRequest, service: Service, meta: Meta = Depends(get_meta)):
    async def events():
        try:
            async for event, data in service.stream(body):
                yield f"event: {event}\ndata: {json.dumps(data)}\n\n"
        except NexusException as e:  # headers already sent: report in-stream
            err = {"code": e.code, "message": e.message, "request_id": meta.request_id}
            yield f"event: error\ndata: {json.dumps(err)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
