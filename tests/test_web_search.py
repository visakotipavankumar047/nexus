import asyncio

import httpx
import pytest

from app.tools import web_search as ws
from app.utils.errors import ToolExecutionException

OPENALEX = {"results": [{
    "id": "https://openalex.org/W1", "display_name": "Agentic RAG survey", "doi": "https://doi.org/10.1/x",
    "publication_date": "2026-05-01", "cited_by_count": 12,
    "abstract_inverted_index": {"Agents": [0], "retrieve": [1], "evidence": [2]}}]}
GDELT = {"articles": [
    {"url": "https://news.example/a", "title": "RAG news", "seendate": "20260928T101500Z", "domain": "news.example"},
    {"url": "https://doi.org/10.1/x", "title": "dup of openalex", "seendate": "20260927T000000Z"}]}
RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>HN</title>
<item><title>Agentic RAG in production</title><link>https://hn.example/1</link>
<description>&lt;p&gt;How teams ship &lt;b&gt;RAG&lt;/b&gt;&lt;/p&gt;</description>
<pubDate>Mon, 28 Sep 2026 10:00:00 GMT</pubDate></item>
<item><title>Unrelated cooking post</title><link>https://hn.example/2</link></item></channel></rss>"""


def use(monkeypatch, handler, feeds=("https://hnrss.org/newest?q={query}",)):
    monkeypatch.setattr(ws, "_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(ws.get_settings(), "rss_feeds", list(feeds))


def ok_handler(request: httpx.Request) -> httpx.Response:
    host = request.url.host
    if host == "api.openalex.org":
        assert request.url.params["search"] == "agentic rag"
        return httpx.Response(200, json=OPENALEX)
    if host == "api.gdeltproject.org":
        assert request.url.params["query"] == "agentic rag"
        return httpx.Response(200, json=GDELT)
    return httpx.Response(200, content=RSS)


def test_merges_all_providers(monkeypatch):
    use(monkeypatch, ok_handler)
    results, failed = asyncio.run(ws.search_web("agentic rag", ["news", "research", "rss"], 10))
    assert failed == []
    by_url = {r.url: r for r in results}
    assert len(results) == len(by_url) == 4  # doi duplicate collapsed
    assert by_url["https://doi.org/10.1/x"].snippet.startswith("Agents retrieve evidence")
    assert by_url["https://news.example/a"].published_at == "2026-09-28"
    assert by_url["https://hn.example/1"].snippet == "How teams ship RAG" and by_url["https://hn.example/1"].published_at


def test_tool_output_has_sources_and_banner(monkeypatch):
    use(monkeypatch, ok_handler)
    text, sources = asyncio.run(ws.web_search("agentic rag", ["research"], 5))
    assert text.startswith(ws.UNTRUSTED_BANNER) and sources[0].id in text
    assert sources[0].type == "web" and sources[0].published_at == "2026-05-01"


def test_static_feed_is_keyword_filtered(monkeypatch):
    use(monkeypatch, ok_handler, feeds=["https://blog.example/feed.xml"])
    results, _ = asyncio.run(ws.search_web("agentic rag", ["rss"], 10))
    assert [r.title for r in results] == ["Agentic RAG in production"]


def test_partial_failure_is_reported(monkeypatch):
    def handler(request):
        if request.url.host == "api.gdeltproject.org":  # GDELT rate-limit reply is plain text, not JSON
            return httpx.Response(200, text="Please limit requests to one every 5 seconds")
        return ok_handler(request)
    use(monkeypatch, handler)
    text, sources = asyncio.run(ws.web_search("agentic rag", ["news", "research"], 5))
    assert sources and "Unavailable right now: news" in text


def test_all_providers_failing_raises(monkeypatch):
    use(monkeypatch, lambda request: httpx.Response(503))
    with pytest.raises(ToolExecutionException):
        asyncio.run(ws.search_web("agentic rag", ["news", "research", "rss"], 5))


def test_short_query_skips_gdelt(monkeypatch):
    use(monkeypatch, ok_handler)
    assert asyncio.run(ws._gdelt(None, "AI", 5)) == []
