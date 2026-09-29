from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.utils.errors import LLMException

# OpenAI-compatible router in front of HF Inference Providers (Together, Fireworks, ...)
HF_ROUTER_URL = "https://router.huggingface.co/v1"


def hf_token() -> str:
    token = get_settings().huggingfacehub_access_token.get_secret_value()
    if not token:
        raise LLMException("HUGGINGFACEHUB_ACCESS_TOKEN is not set")
    return token


def hf_chat_model(model: str) -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(model=model, base_url=HF_ROUTER_URL, api_key=hf_token(),
                      timeout=s.llm_timeout_s, max_retries=s.llm_max_retries)
