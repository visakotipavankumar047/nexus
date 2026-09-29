STEP_LABELS = {
    "start": "Query received",
    "vector_search": "Searching private documents (Pinecone)",
    "document_search": "Searching selected documents",
    "web_search": "Searching live sources (GDELT / OpenAlex / RSS)",
    "calculator": "Calculating",
}


def step_label(step: str) -> str:
    return STEP_LABELS.get(step, step.replace("_", " ").capitalize())


def score(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def short(text: str, limit: int = 300) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"
