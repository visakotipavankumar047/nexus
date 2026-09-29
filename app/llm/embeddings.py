from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_openai import OpenAIEmbeddings

from app.config import get_settings
from app.llm.huggingface import hf_token
from app.utils.errors import LLMException


def embedding_model_id() -> str:
    """Tag stored on every vector: indexing and querying must use the same model."""
    s = get_settings()
    if s.embedding_provider == "openai":
        return f"openai:{s.openai_embedding_model}:{s.embedding_dimensions}"
    return f"huggingface:{s.huggingface_embedding_model}"


@lru_cache
def get_embeddings() -> Embeddings:
    """EMBEDDING_PROVIDER=huggingface (BAAI/bge-large-en-v1.5, 1024-d) or openai (text-embedding-3-large, cut to
    EMBEDDING_DIMENSIONS so it fits the same Pinecone index). Switching providers requires re-ingesting."""
    s = get_settings()
    match s.embedding_provider:
        case "huggingface":
            return HuggingFaceEndpointEmbeddings(model=s.huggingface_embedding_model,
                                                 huggingfacehub_api_token=hf_token())
        case "openai":
            key = s.openai_api_key.get_secret_value()
            if not key:
                raise LLMException("OPENAI_API_KEY is not set")
            return OpenAIEmbeddings(model=s.openai_embedding_model, dimensions=s.embedding_dimensions, api_key=key,
                                    request_timeout=s.llm_timeout_s, max_retries=s.llm_max_retries)
    raise LLMException(f"Unknown EMBEDDING_PROVIDER '{s.embedding_provider}' (huggingface | openai)")
