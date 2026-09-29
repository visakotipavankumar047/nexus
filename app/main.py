import time
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import api_router
from app.config import get_settings
from app.observability import configure_langsmith, record, request_trace
from app.utils.errors import NexusException
from app.utils.logging import get_logger, setup_logging

settings = get_settings()
setup_logging(settings.log_level)
configure_langsmith()
logger = get_logger("api")

app = FastAPI(title="NEXUS API", version=settings.app_version)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                   allow_methods=["*"], allow_headers=["*"])
app.include_router(api_router, prefix=settings.api_prefix)


@app.middleware("http")
async def observe(request: Request, call_next):
    rid = request.state.request_id = f"req_{uuid4().hex[:12]}"
    start, status = time.perf_counter(), 500
    with request_trace(request.method, request.url.path, rid):
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = rid
            return response
        except Exception:
            logger.error("unhandled_error", exc_info=True)  # here, while request_id is still in context
            raise
        finally:
            ms = (time.perf_counter() - start) * 1000
            route = getattr(request.scope.get("route"), "path", "unmatched")  # template, not raw ids
            record(f"{request.method} {route}", status, ms)
            logger.info("request_completed", extra={"fields": {
                "method": request.method, "route": route, "status": status, "latency_ms": round(ms, 1)}})


def _error(request: Request, status: int, code: str, message: str) -> JSONResponse:
    rid = getattr(request.state, "request_id", None)
    return JSONResponse(status_code=status, headers={"X-Request-ID": rid or ""},
                        content={"error": {"code": code, "message": message}, "meta": {"request_id": rid}})


@app.exception_handler(NexusException)
async def nexus_error(request: Request, exc: NexusException):
    if exc.status_code >= 500:
        logger.error("nexus_error", exc_info=exc, extra={"fields": {"code": exc.code}})
    return _error(request, exc.status_code, exc.code, exc.message)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    msg = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
    return _error(request, 422, "VALIDATION_ERROR", msg)


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    return _error(request, 500, "INTERNAL_ERROR", "Internal server error")
