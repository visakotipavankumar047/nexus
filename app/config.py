from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "nexus-api"
    app_env: str = "development"
    app_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = ["http://localhost:8501"]
    log_level: str = "INFO"

    max_upload_mb: int = 20
    allowed_upload_types: set[str] = {"pdf", "txt", "docx", "md"}
    chunk_size_tokens: int = 1000  # SKILL.md: start at 800-1200 / 100-200, tune with evaluation
    chunk_overlap_tokens: int = 150
    embed_batch_size: int = 64
    rerank_enabled: bool = True
    rerank_model: str = "bge-reranker-v2-m3"  # Pinecone-hosted, uses PINECONE_API_KEY
    rerank_candidates_factor: int = 3  # fetch top_k * 3 (max 50) from Pinecone, rerank down to top_k
    # Corrective RAG: reranked chunks scoring below this are dropped, so the agent sees "no match" and goes to
    # web_search instead of answering from noise. Calibration knob: tune with evaluation/run_evaluation.py.
    rerank_min_score: float = 0.05

    # Secrets: SecretStr keeps them out of logs/reprs; read with .get_secret_value()
    huggingfacehub_access_token: SecretStr = SecretStr("")
    openai_api_key: SecretStr = SecretStr("")
    groq_api_key: SecretStr = SecretStr("")
    google_api_key: SecretStr = SecretStr("")
    mistral_api_key: SecretStr = SecretStr("")
    openalex_api_key: SecretStr = SecretStr("")

    # LLMs: "openai" direct; "kimi"/"glm" via Hugging Face Inference Providers (HF token)
    default_model: str = "glm"
    openai_model: str = "gpt-5.4-mini"
    kimi_model: str = "moonshotai/Kimi-K3"
    glm_model: str = "zai-org/GLM-5.3"
    llm_timeout_s: float = 60
    llm_max_retries: int = 2
    eval_judge_model: str = "openai"  # LLM-as-judge for faithfulness/relevance
    # Embeddings: "huggingface" | "openai". Output must match the Pinecone index dimension (agentic-rag = 1024).
    # Vectors are tagged with the model; switching provider hides old vectors until you re-ingest.
    embedding_provider: str = "huggingface"
    huggingface_embedding_model: str = "BAAI/bge-large-en-v1.5"  # 1024-d
    openai_embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 1024  # OpenAI text-embedding-3 models can be shortened to this

    # LLM help inside RAG: rewrites raw user questions into search queries before vector search
    rag_llm_model: str = "openai"
    rag_query_rewrite: bool = True

    pinecone_api_key: SecretStr = SecretStr("")
    pinecone_index: str = "nexus-knowledge"
    pinecone_timeout_s: float = 10

    # Tools
    max_tool_rounds: int = 3  # hard cap on LLM↔tool round trips per answer (no infinite loops)
    # Live data: GDELT (news, no key) + OpenAlex (research, OPENALEX_API_KEY) + RSS feeds
    web_search_timeout_s: float = 10
    gdelt_timespan: str = "7d"  # how far back news search looks: 24h, 7d, 1m, ...
    # "{query}" is replaced with the search terms; feeds without it are keyword-filtered.
    # Override in .env as JSON: RSS_FEEDS='["https://example.com/feed.xml"]'
    rss_feeds: list[str] = [
        "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en",
        "https://hnrss.org/newest?q={query}&count=20",
    ]

    database_url: SecretStr = SecretStr("")

    langsmith_api_key: SecretStr = SecretStr("")
    langsmith_project: str = "nexus-agentic-rag"
    langsmith_tracing: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
