"""Live data from open sources only: GDELT (news), OpenAlex (research), RSS feeds.
Providers run concurrently; one failing provider degrades the result instead of failing it."""
import asyncio
import re
from dataclasses import dataclass
from datetime import datetime
from time import mktime
from typing import Literal
from urllib.parse import quote_plus

import feedparser
import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from app.config import get_settings
from app.schemas.common import Source
from app.utils.errors import ToolExecutionException
from app.utils.helpers import UNTRUSTED_BANNER, source_id
from app.utils.logging import get_logger

logger = get_logger("web_search")
Provider = Literal["news", "research", "rss"]


class WebSearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=300, description="Search terms (keywords work best)")
    sources: list[Provider] = Field(
        default=["news", "research", "rss"], min_length=1,
        description="news = GDELT global news, research = OpenAlex scholarly works, rss = configured RSS feeds")
    max_results: int = Field(default=8, ge=1, le=20)


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str
    published_at: str | None
    provider: str


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=get_settings().web_search_timeout_s, follow_redirects=True,
                             headers={"User-Agent": "NEXUS/1.0 (research assistant)"})


def _terms(query: str) -> list[str]:
    return [t for t in re.findall(r"\w+", query.lower()) if len(t) >= 3]


def _text(html: str, limit: int) -> str:
    return " ".join(BeautifulSoup(html or "", "html.parser").get_text(" ").split())[:limit]


# ---------- OpenAlex (research) ----------

def _abstract(inverted: dict[str, list[int]] | None) -> str:
    if not inverted:
        return ""
    return " ".join(w for _, w in sorted((p, w) for w, ps in inverted.items() for p in ps))


async def _openalex(client: httpx.AsyncClient, query: str, n: int) -> list[WebResult]:
    s = get_settings()
    params = {"search": query, "per-page": n,
              "select": "id,display_name,doi,publication_date,abstract_inverted_index,primary_location,cited_by_count"}
    if key := s.openalex_api_key.get_secret_value():
        params["api_key"] = key
    resp = await client.get("https://api.openalex.org/works", params=params)
    resp.raise_for_status()
    out = []
    for w in resp.json().get("results", []):
        url = w.get("doi") or ((w.get("primary_location") or {}).get("landing_page_url")) or w["id"]
        snippet = _abstract(w.get("abstract_inverted_index"))[:600]
        out.append(WebResult(title=w.get("display_name") or url, url=url,
                             snippet=f"{snippet} (cited by {w.get('cited_by_count', 0)})".strip(),
                             published_at=w.get("publication_date"), provider="OpenAlex"))
    return out


# ---------- GDELT (news) ----------

async def _gdelt(client: httpx.AsyncClient, query: str, n: int) -> list[WebResult]:
    terms = _terms(query)  # GDELT rejects very short words
    if not terms:
        return []
    resp = await client.get("https://api.gdeltproject.org/api/v2/doc/doc", params={
        "query": " ".join(terms), "mode": "ArtList", "format": "json", "maxrecords": n,
        "sort": "HybridRel", "timespan": get_settings().gdelt_timespan})
    resp.raise_for_status()
    try:
        data = resp.json()
    except ValueError as e:  # GDELT answers errors / rate limits (1 req per 5s) in plain text
        raise ToolExecutionException(f"GDELT: {resp.text[:120]}") from e
    out = []
    for a in data.get("articles", []):
        seen = a.get("seendate", "")
        date = f"{seen[:4]}-{seen[4:6]}-{seen[6:8]}" if len(seen) >= 8 else None
        out.append(WebResult(title=a.get("title") or a["url"], url=a["url"], published_at=date, provider="GDELT",
                             snippet=f"{a.get('domain', '')} · {a.get('sourcecountry', '')}".strip(" ·")))
    return out


# ---------- RSS ----------

async def _one_feed(client: httpx.AsyncClient, url: str, query: str, n: int) -> list[WebResult]:
    templated = "{query}" in url
    resp = await client.get(url.replace("{query}", quote_plus(query)) if templated else url)
    resp.raise_for_status()
    feed = feedparser.parse(resp.content)
    terms = _terms(query)
    out = []
    for e in feed.entries:
        title, summary = e.get("title", ""), _text(e.get("summary", ""), 400)
        # search feeds are already filtered; static feeds must mention every query term
        if not templated and not all(t in f"{title} {summary}".lower() for t in terms):
            continue
        parsed = e.get("published_parsed") or e.get("updated_parsed")
        date = datetime.fromtimestamp(mktime(parsed)).date().isoformat() if parsed else None
        out.append(WebResult(title=title or e.get("link", ""), url=e.get("link", ""), snippet=summary,
                             published_at=date, provider=feed.feed.get("title", "RSS")))
        if len(out) >= n:
            break
    return out


async def _rss(client: httpx.AsyncClient, query: str, n: int) -> list[WebResult]:
    feeds = get_settings().rss_feeds
    results = await asyncio.gather(*(_one_feed(client, f, query, n) for f in feeds), return_exceptions=True)
    ok = [r for r in results if not isinstance(r, BaseException)]
    if feeds and not ok:
        raise ToolExecutionException("All RSS feeds failed")
    return [item for r in ok for item in r]


PROVIDERS = {"news": _gdelt, "research": _openalex, "rss": _rss}


async def search_web(query: str, sources: list[str], max_results: int) -> tuple[list[WebResult], list[str]]:
    """Returns (merged results, names of providers that failed). Raises only if every provider failed."""
    async with _client() as client:
        results = await asyncio.gather(*(PROVIDERS[s](client, query, max_results) for s in sources),
                                       return_exceptions=True)
    failed, per_provider = [], []
    for name, r in zip(sources, results, strict=True):
        if isinstance(r, BaseException):
            logger.warning("web_provider_failed", exc_info=r, extra={"fields": {"provider": name}})
            failed.append(name)
        else:
            per_provider.append(r)
    if len(failed) == len(sources):
        raise ToolExecutionException("All live sources failed (GDELT / OpenAlex / RSS)")

    merged, seen = [], set()
    for rank in range(max(map(len, per_provider), default=0)):  # interleave so no provider drowns the others
        for items in per_provider:
            if rank < len(items) and items[rank].url and items[rank].url not in seen:
                seen.add(items[rank].url)
                merged.append(items[rank])
    return merged[:max_results], failed


async def web_search(query: str, sources: list[str] | None = None, max_results: int = 8) -> tuple[str, list[Source]]:
    results, failed = await search_web(query, sources or ["news", "research", "rss"], max_results)
    note = f"\n(Unavailable right now: {', '.join(failed)})" if failed else ""
    if not results:
        return f"No live results found for '{query}'.{note}", []
    srcs, blocks = [], []
    for r in results:
        src = Source(id=source_id(r.url), title=r.title, type="web", url=r.url, published_at=r.published_at)
        srcs.append(src)
        blocks.append(f"[{src.id}] {r.title} ({r.provider}, {r.published_at or 'date unknown'})\n{r.url}\n{r.snippet}")
    return f"{UNTRUSTED_BANNER}\n\n" + "\n\n".join(blocks) + note, srcs
