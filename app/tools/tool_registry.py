from collections.abc import Awaitable, Callable
from functools import partial
from uuid import UUID

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel

from app.tools.calculator import CalculatorInput, calculator
from app.tools.document_search import DocumentSearchInput, document_search
from app.tools.vector_search import VectorSearchInput, vector_search
from app.tools.web_search import WebSearchInput, web_search
from app.utils.errors import NexusException
from app.utils.logging import get_logger

logger = get_logger("tools")


def _tool(name: str, description: str, schema: type[BaseModel], fn: Callable[..., Awaitable]) -> BaseTool:
    """Strict-schema tool returning (text for LLM, list[Source] artifact). Failures go back to the LLM as a
    tool error message instead of killing the whole answer."""
    async def run(**kwargs):
        try:
            return await fn(**kwargs)
        except NexusException as e:
            logger.warning("tool_failed", exc_info=e, extra={"fields": {"tool": name, "code": e.code}})
            raise ToolException(f"{name} failed: {e.message}") from e

    return StructuredTool.from_function(coroutine=run, name=name, description=description, args_schema=schema,
                                        response_format="content_and_artifact", handle_tool_error=True)


def build_tools(user_id: UUID | None, *, use_web: bool, use_private: bool) -> list[BaseTool]:
    """Tools for one request. user_id is bound here, never chosen by the LLM."""
    tools = [_tool("calculator", "Exact arithmetic. Use for any non-trivial calculation instead of computing it yourself.",
                   CalculatorInput, calculator)]
    if use_private:
        tools += [
            _tool("vector_search", "Semantic search over the user's private uploaded documents.",
                  VectorSearchInput, partial(vector_search, user_id=user_id)),
            _tool("document_search", "Search only within specific documents, by document id.",
                  DocumentSearchInput, partial(document_search, user_id=user_id)),
        ]
    if use_web:
        tools.append(_tool("web_search", "Live search over GDELT global news, OpenAlex research papers and RSS "
                                         "feeds. Use for anything latest, current, recent, today, news, releases "
                                         "or recent research. Pick `sources` to match the question.",
                           WebSearchInput, web_search))
    return tools
