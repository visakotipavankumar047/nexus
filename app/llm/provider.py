from functools import lru_cache
from typing import Literal, get_args

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.llm.huggingface import hf_chat_model
from app.utils.errors import LLMException

ModelName = Literal["openai", "kimi", "glm"]
MODEL_NAMES: tuple[str, ...] = get_args(ModelName)


@lru_cache
def get_llm(name: ModelName | None = None) -> BaseChatModel:
    """Chat model by short name. Retries (429/5xx/timeouts, exponential backoff) and timeouts are set here."""
    s = get_settings()
    name = name or s.default_model
    match name:
        case "openai":
            key = s.openai_api_key.get_secret_value()
            if not key:
                raise LLMException("OPENAI_API_KEY is not set")
            return ChatOpenAI(model=s.openai_model, api_key=key,
                              timeout=s.llm_timeout_s, max_retries=s.llm_max_retries)
        case "kimi":
            return hf_chat_model(s.kimi_model)
        case "glm":
            return hf_chat_model(s.glm_model)
    raise LLMException(f"Unknown model '{name}', expected one of {MODEL_NAMES}")
