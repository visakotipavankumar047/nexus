import asyncio
import io
import ipaddress
import re
import socket
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from pypdf import PdfReader

from app.utils.errors import DocumentProcessingException

# (page number or None, text). Pages only exist for PDFs.
Page = tuple[int | None, str]


def file_type(filename: str) -> str:
    ext = Path(filename).suffix.lower().lstrip(".")
    return "md" if ext == "markdown" else ext


def clean(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_bytes(data: bytes, filename: str) -> list[Page]:
    """Extract text from an uploaded file. Content is data only; nothing in it is ever executed."""
    kind = file_type(filename)
    try:
        match kind:
            case "pdf":
                pages = [(i, p.extract_text() or "") for i, p in enumerate(PdfReader(io.BytesIO(data)).pages, 1)]
            case "docx":
                pages = [(None, "\n".join(p.text for p in DocxDocument(io.BytesIO(data)).paragraphs))]
            case "txt" | "md":
                pages = [(None, data.decode("utf-8", errors="replace"))]
            case _:
                raise DocumentProcessingException(f"Unsupported file type '{kind}'")
    except DocumentProcessingException:
        raise
    except Exception as e:
        raise DocumentProcessingException(f"Could not read {kind.upper()} file") from e
    pages = [(n, clean(t)) for n, t in pages if t.strip()]
    if not pages:
        raise DocumentProcessingException("No extractable text (scanned PDFs need OCR)")
    return pages


async def _assert_public(url: str) -> None:
    """SSRF guard: only fetch http(s) URLs whose host resolves to public addresses."""
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise DocumentProcessingException("URL must be http(s)")
    try:
        infos = await asyncio.to_thread(socket.getaddrinfo, parts.hostname, parts.port or 443)
    except OSError as e:
        raise DocumentProcessingException("URL host could not be resolved") from e
    # ponytail: resolve-then-fetch leaves a DNS-rebinding window; pin the resolved IP if this is ever exposed publicly
    if any(not ipaddress.ip_address(info[4][0]).is_global for info in infos):
        raise DocumentProcessingException("URL points to a private or local address")


async def load_url(url: str) -> list[Page]:
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=False) as client:
            for _ in range(5):  # follow redirects manually so every hop passes the SSRF guard
                await _assert_public(url)
                resp = await client.get(url, headers={"User-Agent": "NEXUS-ingest/1.0"})
                if not resp.is_redirect:
                    break
                url = str(resp.next_request.url)
            else:
                raise DocumentProcessingException("Too many redirects")
            resp.raise_for_status()
    except httpx.HTTPError as e:
        raise DocumentProcessingException("Could not fetch URL") from e
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    text = clean(soup.get_text("\n"))
    if not text:
        raise DocumentProcessingException("URL has no extractable text")
    return [(None, text)]
