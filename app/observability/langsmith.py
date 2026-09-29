import os

from langsmith import utils

from app.config import get_settings


def configure_langsmith() -> bool:
    """Exports LangSmith settings to the env vars the SDK reads. Tracing stays off without an API key."""
    s = get_settings()
    key = s.langsmith_api_key.get_secret_value()
    enabled = bool(key) and s.langsmith_tracing
    os.environ["LANGSMITH_TRACING"] = "true" if enabled else "false"
    if enabled:
        os.environ["LANGSMITH_API_KEY"] = key
        # both names: the SDK and LangChain callbacks each honour one of them
        os.environ["LANGSMITH_PROJECT"] = os.environ["LANGCHAIN_PROJECT"] = s.langsmith_project
    utils.get_env_var.cache_clear()  # SDK caches env lookups
    return enabled


def langsmith_enabled() -> bool:
    return utils.tracing_is_enabled()
