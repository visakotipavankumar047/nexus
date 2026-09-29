import json
from pathlib import Path

from evaluation import metrics

SRC = [{"id": "src_aaaaaaaa", "title": "nexus_architecture.md", "type": "private"},
       {"id": "src_bbbbbbbb", "title": "other.pdf", "type": "private"},
       {"id": "src_cccccccc", "title": "Python.org", "type": "web"}]


def test_metrics():
    assert metrics.citation_accuracy("x [src_aaaaaaaa] y [src_deadbeef]", SRC) == 0.5  # one fabricated
    assert metrics.citation_accuracy("no citations", SRC) is None
    assert metrics.context_precision(SRC, ["nexus_architecture.md"]) == 0.5
    assert metrics.context_recall(SRC, ["nexus_architecture.md", "missing.pdf"]) == 0.5
    assert metrics.context_recall(SRC, []) is None
    assert metrics.tool_selection([], []) == 1.0
    assert metrics.tool_selection(["vector_search", "calculator"], ["vector_search"]) == 0.5

    s = metrics.summarize([{"faithfulness": 1.0, "latency_ms": 100, "tool_selection": 1.0},
                           {"faithfulness": None, "latency_ms": 300, "tool_selection": 0.0},
                           {"error": "boom"}])
    assert s["faithfulness"] == 1.0 and s["tool_selection"] == 0.5 and s["errors"] == 1
    assert s["latency_p50_ms"] == 300


def test_dataset_is_valid():
    items = json.loads(Path("evaluation/datasets/rag_questions.json").read_text(encoding="utf-8"))
    tools = {"vector_search", "document_search", "web_search", "calculator"}
    assert len({i["id"] for i in items}) == len(items)
    for i in items:
        assert set(i["expected_tools"]) <= tools
        assert i["question"] and isinstance(i["use_web"], bool)
