from typing import Any, Literal

from pydantic import BaseModel


class Meta(BaseModel):
    request_id: str


class Envelope[T](BaseModel):
    data: T
    meta: Meta


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody
    meta: Meta


class Source(BaseModel):
    id: str
    title: str
    type: Literal["private", "web"]
    url: str | None = None
    document_id: str | None = None
    page: int | None = None
    score: float | None = None
    published_at: str | None = None


ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse} for code in (404, 422, 500, 502, 503)
}
