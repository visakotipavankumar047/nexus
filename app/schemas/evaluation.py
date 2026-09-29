from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class Evaluation(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    question: str
    answer: str
    faithfulness: float | None
    relevance: float | None
    context_precision: float | None
    context_recall: float | None
    created_at: datetime


class EvaluationList(BaseModel):
    evaluations: list[Evaluation]
    averages: dict[str, float | None]
