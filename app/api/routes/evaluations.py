from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import get_evaluation_service, get_meta
from app.schemas.common import ERROR_RESPONSES, Envelope, Meta
from app.schemas.evaluation import EvaluationList
from app.services.evaluation_service import EvaluationService

router = APIRouter(prefix="/evaluations", tags=["evaluations"], responses=ERROR_RESPONSES)


@router.get("", response_model=Envelope[EvaluationList], summary="Recent evaluation results",
            description="Latest rows written by evaluation/run_evaluation.py, with metric averages.")
async def list_evaluations(
    service: Annotated[EvaluationService, Depends(get_evaluation_service)],
    meta: Annotated[Meta, Depends(get_meta)],
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
):
    return Envelope(data=await service.recent(limit), meta=meta)
