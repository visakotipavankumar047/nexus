"""Deterministic retrieval / citation / tool metrics. None = not applicable (excluded from averages)."""
import re
from statistics import mean, quantiles

CITATION = re.compile(r"\[(src_[0-9a-f]{8})\]")


def citation_accuracy(answer: str, sources: list[dict]) -> float | None:
    """Share of cited ids that exist in the returned sources (fabricated citations score 0)."""
    cited = set(CITATION.findall(answer))
    if not cited:
        return None
    return len(cited & {s["id"] for s in sources}) / len(cited)


def _private_titles(sources: list[dict]) -> list[str]:
    return [s["title"] for s in sources if s["type"] == "private"]


def context_precision(sources: list[dict], expected: list[str]) -> float | None:
    """Share of retrieved private chunks that come from an expected document."""
    titles = _private_titles(sources)
    if not titles:
        return None if not expected else 0.0
    return sum(t in expected for t in titles) / len(titles)


def context_recall(sources: list[dict], expected: list[str]) -> float | None:
    """Share of expected documents that were retrieved at least once."""
    if not expected:
        return None
    titles = set(_private_titles(sources))
    return sum(e in titles for e in expected) / len(expected)


def tool_selection(used: list[str], expected: list[str]) -> float:
    """1.0 when exactly the expected tools were used; partial credit otherwise (Jaccard)."""
    u, e = set(used), set(expected)
    return 1.0 if u == e else len(u & e) / len(u | e)


def summarize(rows: list[dict]) -> dict[str, float | None]:
    keys = ["faithfulness", "relevance", "context_precision", "context_recall", "citation_accuracy", "tool_selection"]
    out: dict[str, float | None] = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = round(mean(vals), 3) if vals else None
    lat = sorted(r["latency_ms"] for r in rows if r.get("latency_ms") is not None)
    out["latency_p50_ms"] = lat[len(lat) // 2] if lat else None
    out["latency_p95_ms"] = round(quantiles(lat, n=20)[-1]) if len(lat) >= 2 else (lat[0] if lat else None)
    out["errors"] = sum(1 for r in rows if r.get("error"))
    return out
