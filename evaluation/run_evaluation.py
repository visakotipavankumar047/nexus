"""Run the evaluation dataset through the real chat pipeline, score it, store it.

    python evaluation/run_evaluation.py [--model glm|openai|kimi] [--only id1,id2] [--no-judge]

Pipeline: dataset -> ChatService (agent + tools, traced in LangSmith) -> metrics + LLM judge
-> evaluations table + evaluation/results/<timestamp>.json
"""
import argparse
import asyncio
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.database.repositories import EvaluationRepository  # noqa: E402
from app.database.supabase import run_in_session  # noqa: E402
from app.observability import configure_langsmith  # noqa: E402
from app.schemas.chat import ChatRequest  # noqa: E402
from app.services.chat_service import ChatService  # noqa: E402
from app.services.conversation_service import ConversationService  # noqa: E402
from evaluation import metrics  # noqa: E402
from evaluation.evaluators import judge  # noqa: E402

DATASET = ROOT / "evaluation" / "datasets" / "rag_questions.json"
RESULTS = ROOT / "evaluation" / "results"


async def evaluate_one(item: dict, model: str | None, use_judge: bool) -> dict:
    request = ChatRequest(message=item["question"], model=model, use_web=item["use_web"],
                          use_private_knowledge=item["use_private_knowledge"])
    start, done, evidence = time.perf_counter(), None, []
    async for event, data in ChatService().stream(request):
        if event == "tool_result":
            evidence.append(f"[{data['tool']}] {data['content']}")
        elif event == "done":
            done = data
    latency = int((time.perf_counter() - start) * 1000)
    await ConversationService().delete(UUID(done["conversation_id"]))  # keep eval runs out of the user's history

    row = {
        "id": item["id"], "question": item["question"], "answer": done["answer"], "latency_ms": latency,
        "tools_used": done["tools_used"], "sources": [s["title"] for s in done["sources"]],
        "context_precision": metrics.context_precision(done["sources"], item["expected_sources"]),
        "context_recall": metrics.context_recall(done["sources"], item["expected_sources"]),
        "citation_accuracy": metrics.citation_accuracy(done["answer"], done["sources"]),
        "tool_selection": metrics.tool_selection(done["tools_used"], item["expected_tools"]),
    }
    if use_judge:
        verdict = await judge(item["question"], done["answer"], evidence)
        row |= {"faithfulness": verdict.faithfulness, "relevance": verdict.relevance, "judge_reason": verdict.reason}
    await run_in_session(lambda s: EvaluationRepository(s).add(
        question=row["question"], answer=row["answer"], faithfulness=row.get("faithfulness"),
        relevance=row.get("relevance"), context_precision=row["context_precision"],
        context_recall=row["context_recall"]))
    return row


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["glm", "openai", "kimi"], default=None)
    p.add_argument("--only", help="comma-separated dataset ids")
    p.add_argument("--no-judge", action="store_true", help="skip LLM-as-judge (deterministic metrics only)")
    args = p.parse_args()

    configure_langsmith()
    items = json.loads(DATASET.read_text(encoding="utf-8"))
    if args.only:
        wanted = set(args.only.split(","))
        items = [i for i in items if i["id"] in wanted]

    rows = []
    for item in items:  # sequential: keeps provider rate limits and latency numbers honest
        try:
            row = await evaluate_one(item, args.model, not args.no_judge)
        except Exception as e:  # one failing question must not sink the run; it is counted in the summary
            row = {"id": item["id"], "question": item["question"], "error": f"{type(e).__name__}: {e}"}
        rows.append(row)
        status = row.get("error") or (f"tools={row['tools_used']} recall={row['context_recall']} "
                                      f"faith={row.get('faithfulness')} {row['latency_ms']}ms")
        print(f"{item['id']:22} {status}")

    summary = metrics.summarize(rows)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    out.write_text(json.dumps({"model": args.model or "default", "summary": summary, "rows": rows},
                              indent=2, default=str), encoding="utf-8")
    print("\nSUMMARY")
    for k, v in summary.items():
        print(f"  {k:20} {v}")
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
