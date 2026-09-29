from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar

import langsmith as ls

from app.config import get_settings
from app.observability.langsmith import langsmith_enabled

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


@contextmanager
def request_trace(method: str, path: str, request_id: str) -> Iterator[None]:
    """Root LangSmith run for one API request; agent/tool/LLM runs (@traceable) nest under it."""
    token = request_id_var.set(request_id)
    root = (ls.trace(f"{method} {path}", "chain", inputs={"path": path}, metadata={"request_id": request_id},
                     project_name=get_settings().langsmith_project)
            if langsmith_enabled() else nullcontext())
    try:
        with root:
            yield
    finally:
        request_id_var.reset(token)
