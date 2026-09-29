from uuid import UUID

from pydantic import BaseModel, Field

from app.llm.provider import ModelName
from app.schemas.common import Source


class ChatRequest(BaseModel):
    conversation_id: UUID | None = None
    message: str = Field(min_length=1, max_length=8000)
    model: ModelName | None = Field(default=None, description="openai | kimi | glm; server default if omitted")
    use_web: bool = True
    use_private_knowledge: bool = True


class ChatResponse(BaseModel):
    conversation_id: UUID
    answer: str
    sources: list[Source]
    tools_used: list[str]
    grounded: bool
    latency_ms: int
