from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_meta
from app.observability.metrics import snapshot
from app.schemas.common import Envelope, Meta

router = APIRouter(tags=["observability"])


@router.get("/metrics", response_model=Envelope[dict[str, dict[str, float]]], summary="Request metrics",
            description="Per-route request count, 5xx errors, avg and max latency (ms) since process start.")
async def metrics(meta: Annotated[Meta, Depends(get_meta)]):
    return Envelope(data=snapshot(), meta=meta)
