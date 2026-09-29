from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.rag.loaders import Page


@lru_cache
def _splitter() -> RecursiveCharacterTextSplitter:
    s = get_settings()
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=s.chunk_size_tokens, chunk_overlap=s.chunk_overlap_tokens)


def chunk(pages: list[Page]) -> list[tuple[int | None, str]]:
    """Token-sized chunks that keep their page number."""
    return [(page, piece) for page, text in pages for piece in _splitter().split_text(text)]
