from fastapi import APIRouter
from pydantic import BaseModel

from app.config import get_settings

router = APIRouter(tags=["health"])


class Health(BaseModel):
    status: str
    service: str
    version: str


@router.get("/health", response_model=Health, summary="Health check",
            description="Liveness probe. Does not touch external services.")
async def health() -> Health:
    s = get_settings()
    return Health(status="ok", service=s.app_name, version=s.app_version)
