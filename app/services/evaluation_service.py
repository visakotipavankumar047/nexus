from statistics import mean

from app.database.repositories import EvaluationRepository
from app.database.supabase import run_in_session
from app.schemas.evaluation import Evaluation, EvaluationList

METRICS = ("faithfulness", "relevance", "context_precision", "context_recall")


class EvaluationService:
    async def recent(self, limit: int) -> EvaluationList:
        rows = await run_in_session(lambda s: [
            Evaluation.model_validate(e) for e in EvaluationRepository(s).list_recent(limit)])
        averages = {}
        for m in METRICS:
            vals = [getattr(r, m) for r in rows if getattr(r, m) is not None]
            averages[m] = round(mean(vals), 3) if vals else None
        return EvaluationList(evaluations=rows, averages=averages)
