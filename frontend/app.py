"""NEXUS frontend.   streamlit run frontend/app.py   (starts the FastAPI backend too)"""
import streamlit as st

from components.chat import render_chat_page
from components.document_upload import render_documents_page
from components.sidebar import open_conversation, render_sidebar
from config import AUTO_START_BACKEND, BACKEND_LOG
from services.api_client import ApiError, ensure_backend, get_client
from utils.formatting import pct, short

st.set_page_config(page_title="NEXUS", page_icon="🧠", layout="wide")

if AUTO_START_BACKEND:
    try:
        with st.spinner("Starting NEXUS backend…"):
            ensure_backend()
    except RuntimeError as e:
        st.error(f"Backend failed to start: {e}")
        if BACKEND_LOG.exists():
            st.code(BACKEND_LOG.read_text(encoding="utf-8", errors="replace")[-3000:])
        st.stop()

for key, default in {"conversation_id": None, "messages": [], "model": "glm",
                     "use_web": True, "use_private": True}.items():
    st.session_state.setdefault(key, default)


def conversations_page() -> None:
    st.title("Conversations")
    client = get_client()
    try:
        convs = client.conversations()
    except ApiError as e:
        st.error(str(e))
        return
    if not convs:
        st.info("No conversations yet. Start one in Chat.")
    for c in convs:
        c1, c2, c3 = st.columns([6, 1, 1])
        c1.markdown(f"**{c['title']}**  \n{c['updated_at'][:16].replace('T', ' ')} UTC")
        if c2.button("Open", key=f"open-{c['id']}"):
            open_conversation(c["id"])
            st.switch_page(chat)
        with c3.popover("🗑"):
            if st.button("Delete", key=f"delc-{c['id']}", type="primary"):
                client.delete_conversation(c["id"])
                if st.session_state.conversation_id == c["id"]:
                    st.session_state.conversation_id, st.session_state.messages = None, []
                st.rerun()


def evaluation_page() -> None:
    st.title("Evaluation")
    st.caption("Run `python evaluation/run_evaluation.py` to add results.")
    try:
        data = get_client().evaluations()
    except ApiError as e:
        st.error(str(e))
        return
    avg = data["averages"]
    cols = st.columns(4)
    for col, (name, label) in zip(cols, [("faithfulness", "Faithfulness"), ("relevance", "Relevance"),
                                          ("context_precision", "Context precision"),
                                          ("context_recall", "Context recall")], strict=True):
        col.metric(label, pct(avg.get(name)))
    rows = data["evaluations"]
    if not rows:
        st.info("No evaluation results yet.")
        return
    st.dataframe([{"when": r["created_at"][:16].replace("T", " "), "question": r["question"],
                   "answer": short(r["answer"], 160), "faithfulness": r["faithfulness"], "relevance": r["relevance"],
                   "precision": r["context_precision"], "recall": r["context_recall"]} for r in rows],
                 width="stretch", hide_index=True)


render_sidebar()
chat = st.Page(render_chat_page, title="Chat", icon=":material/chat:", default=True)
pg = st.navigation([chat,
                    st.Page(render_documents_page, title="Documents", icon=":material/description:"),
                    st.Page(conversations_page, title="Conversations", icon=":material/forum:"),
                    st.Page(evaluation_page, title="Evaluation", icon=":material/monitoring:")])
pg.run()
